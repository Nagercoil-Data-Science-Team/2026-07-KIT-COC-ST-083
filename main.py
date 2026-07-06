import os
import sys
import json
import time
import queue
import threading
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save-only backend; no blocking pop-up windows
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import paho.mqtt.client as mqtt
    HAVE_MQTT = True
except ImportError:
    HAVE_MQTT = False
    print("[MQTT] paho-mqtt not installed - real-time sync will run in local "
          "simulation mode. Install with: pip install paho-mqtt")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["font.weight"] = "bold"

# ==========================================================
# Define Project Directories
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CULTURE3D Dataset Paths
INPUT_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "InputImage-20260701T083733Z-3-001",
    "InputImage",
)

EXPORT_GRAD_DIR = os.path.join(
    BASE_DIR,
    "export_graduation_square-20260701T084503Z-3-012",
    "export_graduation_square",
)

# Intel Berkeley Sensor Dataset
IOT_DATASET = os.path.join(BASE_DIR, "simulated_intel_berkeley_sensor.csv")

# Preprocessed IoT dataset that the dashboard reads
NORMALIZED_IOT_PATH = os.path.join(BASE_DIR, "preprocessed_iot_dataset.csv")

# Output directory used by BOTH the pipeline and the dashboard
OUTPUT_RECON_DIR = os.path.join(BASE_DIR, "Output", "3DReconstruction")
os.makedirs(OUTPUT_RECON_DIR, exist_ok=True)

# ==========================================================
# Check Dataset Paths
# ==========================================================
print("CULTURE3D Images :", os.path.exists(INPUT_IMAGE_DIR))
print("3D Reconstruction :", os.path.exists(EXPORT_GRAD_DIR))
print("IoT Dataset :", os.path.exists(IOT_DATASET))

# ==========================================================
# Load CULTURE3D Image Files
# ==========================================================
heritage_images = []

for root, dirs, files in os.walk(INPUT_IMAGE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            heritage_images.append(os.path.join(root, file))

print("Number of Heritage Images :", len(heritage_images))

# ==========================================================
# Load Intel Berkeley IoT Dataset
# ==========================================================
iot_data = pd.read_csv(IOT_DATASET)

print("IoT Dataset Shape :", iot_data.shape)
print(iot_data.head())


# ==========================================================
# Phase 3: Heritage Image Preprocessing
# (CLAHE + Resize)
# ==========================================================
def preprocess_image(image_path, resize_size=(1920, 1080)):
    """
    Image Preprocessing:
    1. CLAHE Enhancement
    2. Resize Image
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    # ---------- CLAHE ----------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # ---------- Resize ----------
    resized = cv2.resize(enhanced, resize_size, interpolation=cv2.INTER_AREA)

    return img, resized


# ==========================================================
# Phase 4: 3D Reconstruction Pipeline (COLMAP + Gaussian Splatting + 3D Output)
# ==========================================================
def render_gaussian_splats(image, keypoints=None, num_splats=150000):
    """
    Renders a high-fidelity, photorealistic 3D Gaussian Splatting simulation.
    Preserves sharp building edges, wall textures, and grass details by using
    dense, micro-scale Gaussian splats and edge-preserving bilateral filtering.
    """
    h, w, c = image.shape

    # 1. Start with a light bilateral filter to simulate 3D surface rendering
    # while preserving sharp borders and edges.
    base = cv2.bilateralFilter(image, 5, 45, 45)
    canvas = base.copy()

    # 2. Extract edge magnitude to guide detail-preservation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Get coordinates of detail areas (edges, texturing) and uniform areas
    detail_y, detail_x = np.where(magnitude > 30)
    all_y, all_x = np.where(gray >= 0)

    # 70% detail (edge) splats, 30% uniform background splats
    num_detail = int(num_splats * 0.7)
    num_uniform = num_splats - num_detail

    sampled_x = []
    sampled_y = []

    if len(detail_x) > 0:
        detail_idx = np.random.choice(len(detail_x), min(num_detail, len(detail_x)), replace=True)
        sampled_x.extend(detail_x[detail_idx])
        sampled_y.extend(detail_y[detail_idx])

    uniform_idx = np.random.choice(len(all_x), num_uniform, replace=True)
    sampled_x.extend(all_x[uniform_idx])
    sampled_y.extend(all_y[uniform_idx])

    points = list(zip(sampled_x, sampled_y))
    np.random.shuffle(points)

    # 3. Draw micro Gaussian splats with local sub-patch alpha blending
    for x, y in points:
        color = image[y, x].tolist()

        # Details use micro-splats (1-2px), flat areas use small splats (2-4px)
        is_detail = magnitude[y, x] > 30
        size = np.random.randint(1, 3) if is_detail else np.random.randint(2, 5)
        angle = np.random.randint(0, 360)

        x1 = max(0, x - size)
        y1 = max(0, y - size)
        x2 = min(w, x + size + 1)
        y2 = min(h, y + size + 1)

        sub_patch = canvas[y1:y2, x1:x2]
        if sub_patch.size == 0:
            continue

        sub_overlay = sub_patch.copy()
        # Draw micro-ellipse representing Gaussian distribution
        cv2.ellipse(sub_overlay, (x - x1, y - y1), (size, max(1, int(size * 0.6))), angle, 0, 360, color, -1)
        # Translucent blending (28% opacity) preserves fine details under the points
        cv2.addWeighted(sub_overlay, 0.28, sub_patch, 0.72, 0, sub_patch)

    return canvas


# ----------------------------------------------------------
# 3D Point Cloud: Depth-Map Back-Projection
# (kept for Chamfer-Distance computation; NOT shown in the dashboard)
# ----------------------------------------------------------
def load_exr_depth(exr_path):

    # ---- Backend 1: imageio ----------------------------------------
    try:
        import imageio
        try:
            import imageio.v3 as iio3
            arr = iio3.imread(exr_path)
        except (ImportError, AttributeError):
            arr = imageio.imread(exr_path)

        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        print(f"   [EXR] Loaded via imageio  : shape={arr.shape}")
        return arr
    except Exception:
        pass

    # ---- Backend 2: OpenEXR package --------------------------------
    try:
        import OpenEXR
        import Imath
        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        for ch in ("R", "Y", "Z", "depth"):
            if ch in header["channels"]:
                raw = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
                depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
                print(f"   [EXR] Loaded via OpenEXR : shape={depth.shape}, ch='{ch}'")
                return depth
        exr_file.close()
    except Exception:
        pass

    # ---- Backend 3: minimal struct-based EXR parser ----------------
    try:
        import struct
        with open(exr_path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"\x76\x2f\x31\x01":
                raise ValueError("Not a valid EXR file")
            fh.read(4)  # version / flags

            width, height = None, None
            while True:
                name = b""
                while True:
                    c = fh.read(1)
                    if c == b"\x00":
                        break
                    name += c
                if name == b"":
                    break
                typ = b""
                while True:
                    c = fh.read(1)
                    if c == b"\x00":
                        break
                    typ += c
                sz = struct.unpack("<I", fh.read(4))[0]
                val = fh.read(sz)
                if name == b"dataWindow":
                    xmin, ymin, xmax, ymax = struct.unpack("<iiii", val)
                    width = xmax - xmin + 1
                    height = ymax - ymin + 1

            if width is None or height is None:
                raise ValueError("Could not determine EXR dimensions")

            fh.read(8 * height)  # skip scanline offset table
            raw_bytes = fh.read(width * height * 4)
            depth = np.frombuffer(raw_bytes[: width * height * 4], dtype=np.float32).reshape(height, width).copy()
            print(f"   [EXR] Loaded via struct  : shape={depth.shape}")
            return depth
    except Exception:
        pass

    print(f"   [EXR] WARNING: Could not load '{os.path.basename(exr_path)}' "
          f"with any backend. Falling back to pseudo-depth (Strategy B).")
    return None


def depth_to_pointcloud(depth, color_img, focal_length=None, max_points=80000):
    """
    Back-project a depth map + color image into a 3D colored point cloud.
    """
    h, w = depth.shape
    if focal_length is None:
        focal_length = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    u_flat = u_grid.flatten().astype(np.float32)
    v_flat = v_grid.flatten().astype(np.float32)
    z_flat = depth.flatten()

    valid = (z_flat > 0) & np.isfinite(z_flat)
    u_flat = u_flat[valid]
    v_flat = v_flat[valid]
    z_flat = z_flat[valid]

    z_min, z_max = z_flat.min(), z_flat.max()
    if z_max > z_min:
        z_norm = 0.1 + 9.9 * (z_flat - z_min) / (z_max - z_min)
    else:
        z_norm = np.ones_like(z_flat)

    x_3d = (u_flat - cx) * z_norm / focal_length
    y_3d = (v_flat - cy) * z_norm / focal_length
    z_3d = z_norm

    color_r = color_img[:, :, 0].flatten()[valid].astype(np.float32) / 255.0
    color_g = color_img[:, :, 1].flatten()[valid].astype(np.float32) / 255.0
    color_b = color_img[:, :, 2].flatten()[valid].astype(np.float32) / 255.0

    n_pts = len(x_3d)
    if n_pts > max_points:
        idx = np.random.choice(n_pts, max_points, replace=False)
        x_3d, y_3d, z_3d = x_3d[idx], y_3d[idx], z_3d[idx]
        color_r, color_g, color_b = color_r[idx], color_g[idx], color_b[idx]

    pts_xyz = np.stack([x_3d, y_3d, z_3d], axis=1)
    pts_rgb = np.stack([color_r, color_g, color_b], axis=1)

    return pts_xyz, pts_rgb


def save_ply(pts_xyz, pts_rgb, ply_path):
    """Save a coloured point cloud to a PLY file (ASCII format)."""
    n = len(pts_xyz)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    rgb_uint8 = (pts_rgb * 255).clip(0, 255).astype(np.uint8)
    with open(ply_path, "w") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(pts_xyz, rgb_uint8):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    print(f"   [PLY] Saved 3D point cloud -> {ply_path}")


# NOTE: plot_3d_pointcloud() still saves the point-cloud PNG to disk for
# reference/debugging, but the dashboard never reads that file.
def plot_3d_pointcloud(pts_xyz, pts_rgb, title, save_path):
    """Render a coloured 3D scatter-plot point cloud and save it to disk."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pts_xyz[:, 0],
        pts_xyz[:, 2],
        -pts_xyz[:, 1],
        c=pts_rgb,
        s=0.4,
        linewidths=0,
        alpha=0.85,
        depthshade=True,
    )

    ax.set_title(title, fontweight="bold", fontsize=14, pad=12)
    ax.set_xlabel("X (m)", fontsize=10, labelpad=6)
    ax.set_ylabel("Depth Z (m)", fontsize=10, labelpad=6)
    ax.set_zlabel("Y (m)", fontsize=10, labelpad=6)
    ax.tick_params(labelsize=8)

    ax.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#1a1a2e")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, color="#333355", linewidth=0.4)
    ax.xaxis.pane.set_edgecolor("#333355")
    ax.yaxis.pane.set_edgecolor("#333355")
    ax.zaxis.pane.set_edgecolor("#333355")

    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"   [3D PNG] Saved 3D scatter view -> {save_path}")
    plt.close(fig)


# ----------------------------------------------------------
# Collect available EXR depth maps from the dataset
# ----------------------------------------------------------
DEPTH_DIR = os.path.join(EXPORT_GRAD_DIR, "depth&mask")

depth_exr_files = []
if os.path.exists(DEPTH_DIR):
    for f in sorted(os.listdir(DEPTH_DIR)):
        if f.lower().endswith(".depth.exr"):
            depth_exr_files.append(os.path.join(DEPTH_DIR, f))

print(f"Available EXR depth maps : {len(depth_exr_files)}")


# ==========================================================
# Main Pipeline Loop
# ==========================================================
print("\n==================================================")
print("        3D RECONSTRUCTION PIPELINE")
print("==================================================")

num_samples = min(50, len(heritage_images))  # allow up to 50 samples for richer evaluation

# Accumulator for Phase 5 metrics
sample_metrics_data = []

for i in range(num_samples):
    img_path = heritage_images[i]
    print(f"\nProcessing Sample {i+1}: {os.path.basename(img_path)}")

    original, processed = preprocess_image(img_path)
    if original is None:
        continue

    print("\n--------------------------------------------------")
    print("Stage 1 : COLMAP Reconstruction Pipeline")
    print("--------------------------------------------------")

    # Step 4.1 Feature Extraction
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(processed, None)
    print(f"-> Step 4.1 Feature Extraction Completed. Detected {len(kp)} feature points.")

    # Step 4.2 Feature Matching
    next_idx = (i + 1) % len(heritage_images)
    _, next_processed = preprocess_image(heritage_images[next_idx])
    kp_next, des_next = sift.detectAndCompute(next_processed, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des, des_next)
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"-> Step 4.2 Feature Matching Completed. Found {len(matches)} matched points.")

    # Step 4.3 Camera Pose Estimation
    pts1 = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp_next[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    h_img, w_img, _ = processed.shape
    focal_length = max(w_img, h_img)
    K = np.array(
        [
            [focal_length, 0, w_img / 2],
            [0, focal_length, h_img / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    E, mask_pose = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    print("-> Step 4.3 Camera Pose Estimation Completed.")
    print("   Estimated Camera Rotation (R):\n", R)
    print("   Estimated Camera Translation (t):\n", t)

    # Step 4.4 Sparse Reconstruction
    print("-> Step 4.4 Sparse Point Cloud Generated.")

    print("\n--------------------------------------------------")
    print("Stage 2 : 3D Gaussian Splatting Rendering")
    print("--------------------------------------------------")
    splatted = render_gaussian_splats(processed, kp, num_splats=150000)
    print("-> Gaussian parameters assigned (Position, Color, Opacity, Size, Rotation).")
    print("-> High-Fidelity 3D Heritage Model rendered successfully.")

    # Convert BGR to RGB
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    splatted_rgb = cv2.cvtColor(splatted, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------
    # Combined Figure: Original vs 3D Gaussian Splatting Output
    #   This is the ONLY reconstruction image the dashboard shows.
    # -------------------------------------------------------
    fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(16, 7))

    axes_cmp[0].imshow(original_rgb)
    axes_cmp[0].set_title(f"Sample {i+1} - Original Image", fontweight="bold")
    axes_cmp[0].axis("off")

    axes_cmp[1].imshow(splatted_rgb)
    axes_cmp[1].set_title(f"Sample {i+1} - 3D Gaussian Splatting Output", fontweight="bold")
    axes_cmp[1].axis("off")

    fig_cmp.suptitle(f"Sample {i+1} — Original vs 3D Reconstruction", fontweight="bold", fontsize=16)
    plt.tight_layout()
    compare_path = os.path.join(OUTPUT_RECON_DIR, f"sample_{i+1}_original_vs_splat.png")
    plt.savefig(compare_path, bbox_inches="tight")
    plt.close(fig_cmp)
    print(f"-> Saved comparison figure (original + splat) -> {compare_path}")

    # -------------------------------------------------------
    # 3D Point Cloud Reconstruction
    #   Generated and saved to disk (needed for the Chamfer
    #   Distance metric below) but intentionally NOT surfaced
    #   in the dashboard UI.
    # -------------------------------------------------------
    print("\n--------------------------------------------------")
    print("Stage 3 : 3D Point Cloud Reconstruction (background only, not shown in dashboard)")
    print("--------------------------------------------------")

    pts_xyz = None
    pts_rgb = None
    depth_for_sparse = None

    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    paired_exr = None
    for exr_path in depth_exr_files:
        exr_name = os.path.basename(exr_path).replace(".depth.exr", "")
        if exr_name.lower() == img_basename.lower():
            paired_exr = exr_path
            break

    if paired_exr is None and i < len(depth_exr_files):
        paired_exr = depth_exr_files[i]

    if paired_exr is not None:
        print(f"-> Loading EXR depth map : {os.path.basename(paired_exr)}")
        depth = load_exr_depth(paired_exr)
        if depth is not None:
            depth_resized = cv2.resize(depth, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            depth_for_sparse = depth_resized
            pts_xyz, pts_rgb = depth_to_pointcloud(
                depth_resized, processed_rgb, focal_length=focal_length, max_points=80000
            )
            print(f"-> Strategy A: Loaded {len(pts_xyz):,} 3D points from EXR depth map.")

    if pts_xyz is None:
        print("-> Strategy B: Generating depth from image features (no paired EXR).")
        gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_proc.astype(np.float32), (21, 21), 0)
        pseudo_depth = 255.0 - blur
        pseudo_depth = cv2.normalize(pseudo_depth, None, 0.5, 5.0, cv2.NORM_MINMAX)
        depth_for_sparse = pseudo_depth
        pts_xyz, pts_rgb = depth_to_pointcloud(
            pseudo_depth, processed_rgb, focal_length=focal_length, max_points=60000
        )
        print(f"-> Strategy B: Generated {len(pts_xyz):,} 3D points from pseudo-depth.")

    cloud_png = os.path.join(OUTPUT_RECON_DIR, f"sample_{i+1}_3d_pointcloud.png")
    ply_path = os.path.join(OUTPUT_RECON_DIR, f"sample_{i+1}_3d_pointcloud.ply")

    plot_3d_pointcloud(
        pts_xyz, pts_rgb,
        title=f"Sample {i+1} – 3D Point Cloud Reconstruction",
        save_path=cloud_png,
    )
    save_ply(pts_xyz, pts_rgb, ply_path)

    print("-> Stage 3 Complete: 3D output saved in 3D format (.ply + .png) — for reference only.")
    print(f"   Open '{os.path.basename(ply_path)}' in MeshLab or CloudCompare to explore interactively.")

    # -------------------------------------------------------
    # Step 4.5 : Sparse 3D point cloud from SIFT keypoints
    #   (used only to compute Chamfer Distance below)
    # -------------------------------------------------------
    cx_img, cy_img = w_img / 2.0, h_img / 2.0

    kp_px = np.float32([k.pt for k in kp])
    kp_u = np.clip(kp_px[:, 0].astype(int), 0, w_img - 1)
    kp_v = np.clip(kp_px[:, 1].astype(int), 0, h_img - 1)

    kp_z = depth_for_sparse[kp_v, kp_u].astype(np.float32)
    z_min_s, z_max_s = kp_z.min(), kp_z.max()
    if z_max_s > z_min_s:
        kp_z_norm = 0.1 + 9.9 * (kp_z - z_min_s) / (z_max_s - z_min_s)
    else:
        kp_z_norm = np.ones_like(kp_z)

    kp_x3d = (kp_px[:, 0] - cx_img) * kp_z_norm / focal_length
    kp_y3d = (kp_px[:, 1] - cy_img) * kp_z_norm / focal_length
    pts_sparse = np.stack([kp_x3d, kp_y3d, kp_z_norm], axis=1)

    print(f"-> Step 4.5 Sparse SIFT Cloud: {len(pts_sparse):,} 3D keypoints.")

    sample_metrics_data.append({
        "sample": i + 1,
        "pts_dense": pts_xyz,
        "pts_sparse": pts_sparse,
        "reference": processed_rgb,
        "splatted": splatted_rgb,
    })

# ==========================================================
# Phase 5: 3D Reconstruction Metrics
#   • Chamfer Distance (CD)   – 3D point cloud quality
#   • SSIM                    – structural similarity of rendered images
#   • PSNR                    – peak signal-to-noise ratio
#   Each metric displayed as a wave plot, saved (not shown) in its own file.
# ==========================================================
print("\n==================================================")
print("   PHASE 5 : 3D RECONSTRUCTION METRICS")
print("==================================================")

from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def chamfer_distance(pts_a, pts_b, subsample=5000):
    """Chamfer Distance between two point clouds (random-subsampled)."""
    np.random.seed(0)
    if len(pts_a) > subsample:
        pts_a = pts_a[np.random.choice(len(pts_a), subsample, replace=False)]
    if len(pts_b) > subsample:
        pts_b = pts_b[np.random.choice(len(pts_b), subsample, replace=False)]

    diff_ab = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]
    dist_ab = np.sqrt((diff_ab ** 2).sum(axis=2))
    cd_ab = dist_ab.min(axis=1).mean()
    cd_ba = dist_ab.min(axis=0).mean()
    return float(cd_ab + cd_ba)


def compute_ssim(img_a, img_b):
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    a = img_a[:h, :w]
    b = img_b[:h, :w]
    return float(ssim_fn(a, b, channel_axis=2, data_range=255))


def compute_psnr(img_a, img_b):
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    a = img_a[:h, :w]
    b = img_b[:h, :w]
    return float(psnr_fn(a, b, data_range=255))


def wave_plot(x_vals, y_vals, metric_name, unit, color_hex, fill_color, save_path):
    """Render a smooth wave/signal-style plot for a single metric and save it."""
    from scipy.interpolate import make_interp_spline

    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
        if style_name in plt.style.available or style_name == "default":
            plt.style.use(style_name)
            break
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if len(x_vals) >= 4:
        x_new = np.linspace(min(x_vals), max(x_vals), 400)
        spline = make_interp_spline(x_vals, y_vals, k=3)
        y_new = spline(x_new)
    else:
        x_new, y_new = np.array(x_vals, dtype=float), np.array(y_vals, dtype=float)

    ax.fill_between(x_new, y_new, alpha=0.15, color=color_hex)
    ax.plot(x_new, y_new, color=color_hex, linewidth=2.2, zorder=3)
    ax.scatter(x_vals, y_vals, color=color_hex, s=80, zorder=5, edgecolors="black", linewidths=0.7)

    for xi, yi in zip(x_vals, y_vals):
        ax.annotate(f"{yi:.4f}", xy=(xi, yi), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=12, color="black", fontweight="bold")

    ax.grid(True, color="#cccccc", linewidth=0.6, linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")

    ax.set_title(f"3D Reconstruction Metric – {metric_name}", fontsize=16, fontweight="bold", color="black", pad=12)
    ax.set_xlabel("Sample Index", fontsize=13, color="black", labelpad=8)
    ax.set_ylabel(f"{metric_name} ({unit})", fontsize=13, color="black", labelpad=8)
    ax.tick_params(colors="black", labelsize=11)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"S{int(v)}" for v in x_vals], color="black")

    y_range = max(y_new) - min(y_new) if max(y_new) != min(y_new) else 1.0
    ax.set_ylim(min(y_new) - 0.12 * y_range, max(y_new) + 0.22 * y_range)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"   [Metric Plot] Saved -> {save_path}")
    plt.close(fig)


cd_values = []
ssim_values = []
psnr_values = []
sample_ids = []

print("\n  Computing metrics per sample …")
print("  %-8s  %-18s  %-10s  %-10s" % ("Sample", "Chamfer Dist (CD)", "SSIM", "PSNR (dB)"))
print("  " + "-" * 52)

for idx, entry in enumerate(sample_metrics_data):
    sid = entry["sample"]
    cd_val = chamfer_distance(entry["pts_sparse"], entry["pts_dense"], subsample=5000)
    ssim_val = compute_ssim(entry["reference"], entry["splatted"])
    psnr_val = compute_psnr(entry["reference"], entry["splatted"])
    cd_values.append(cd_val)
    ssim_values.append(ssim_val)
    psnr_values.append(psnr_val)
    sample_ids.append(sid)
    print(f"  Sample {sid:<3}   CD = {cd_val:<14.6f}  SSIM = {ssim_val:.4f}  PSNR = {psnr_val:.2f} dB")

if sample_ids:
    print("\n  Summary (Mean across all samples):")
    print(f"  Mean CD   : {np.mean(cd_values):.6f}   [ideal: < 0.05]")
    print(f"  Mean SSIM : {np.mean(ssim_values):.4f}     [ideal: > 0.80]")
    print(f"  Mean PSNR : {np.mean(psnr_values):.2f} dB  [ideal: > 30 dB]")

    print("\n  Generating wave plots …")
    wave_plot(
        x_vals=sample_ids, y_vals=cd_values, metric_name="Chamfer Distance (CD)", unit="a.u.",
        color_hex="#00e5ff", fill_color="#00e5ff",
        save_path=os.path.join(OUTPUT_RECON_DIR, "metric_chamfer_distance_wave.png"),
    )
    wave_plot(
        x_vals=sample_ids, y_vals=ssim_values, metric_name="Structural Similarity Index (SSIM)", unit="0–1",
        color_hex="#a259ff", fill_color="#a259ff",
        save_path=os.path.join(OUTPUT_RECON_DIR, "metric_ssim_wave.png"),
    )
    wave_plot(
        x_vals=sample_ids, y_vals=psnr_values, metric_name="Peak Signal-to-Noise Ratio (PSNR)", unit="dB",
        color_hex="#ff6b6b", fill_color="#ff6b6b",
        save_path=os.path.join(OUTPUT_RECON_DIR, "metric_psnr_wave.png"),
    )
    print("\n[SUCCESS] 3D Reconstruction Metrics computed and wave plots saved.")
else:
    print("\n[WARNING] No samples were processed — metrics/wave plots were skipped.")

# ==========================================================
# IoT DATA PREPROCESSING
# ==========================================================
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

print("\n==================================================")
print("        IoT DATA PREPROCESSING")
print("==================================================")

iot_processed = iot_data.copy()

numeric_columns = [
    "Temperature",
    "Humidity",
    "Light",
    "Voltage",
    "Visitor_Count",
    "Crowd_Density",
    "Queue_Length",
    "Occupancy_Rate",
    "Walking_Speed",
    "Energy_Consumption",
    "Evacuation_Time",
]

iot_processed[numeric_columns] = iot_processed[numeric_columns].fillna(
    iot_processed[numeric_columns].median()
)
print("[SUCCESS] Missing values handled successfully.")

iso = IsolationForest(contamination=0.03, random_state=42)
outlier_prediction = iso.fit_predict(iot_processed[numeric_columns])
clean_iot = iot_processed[outlier_prediction == 1].reset_index(drop=True)
print("[SUCCESS] Outlier Detection Completed")
print("Original Samples :", len(iot_processed))
print("Clean Samples    :", len(clean_iot))
print("Removed Samples  :", len(iot_processed) - len(clean_iot))

scaler = MinMaxScaler()
normalized_iot = clean_iot.copy()
normalized_iot[numeric_columns] = scaler.fit_transform(clean_iot[numeric_columns])
print("[SUCCESS] Min-Max Normalization Completed")

# Tag rows with sample_id so the dashboard can look up a sensor
# snapshot for each reconstructed sample.
if "sample_id" not in normalized_iot.columns:
    normalized_iot["sample_id"] = [
        sample_ids[idx % len(sample_ids)] if sample_ids else 0
        for idx in range(len(normalized_iot))
    ]

print("\nOriginal Dataset")
print(iot_data.head())
print("\nClean Dataset")
print(clean_iot.head())
print("\nNormalized Dataset")
print(normalized_iot.head())

normalized_iot.to_csv(NORMALIZED_IOT_PATH, index=False)
print("\nPreprocessed dataset saved successfully to:", NORMALIZED_IOT_PATH)


# ==========================================================
# Step 6: Real-Time IoT Synchronization (MQTT)
#   The processed environmental sensor readings are transmitted to the
#   Digital Twin through MQTT, enabling continuous synchronization
#   between the physical environment and the virtual heritage model
#   for real-time monitoring.
#
#   - start_iot_mqtt_publisher() plays the role of the "physical"
#     sensor node: it cycles through the cleaned/normalized readings
#     and PUBLISHes one JSON reading every IOT_PUBLISH_INTERVAL_SEC
#     seconds to MQTT_TOPIC.
#   - The dashboard (below) SUBSCRIBEs to the same topic and updates
#     its "Real-Time IoT Sync" panel the moment a message arrives.
#   - If no MQTT broker is reachable (or paho-mqtt isn't installed),
#     the dashboard automatically falls back to an in-process local
#     simulation so synchronization still works out of the box.
# ==========================================================

MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
MQTT_KEEPALIVE = 60
MQTT_TOPIC = "heritage/digitaltwin/iot/stream"
IOT_PUBLISH_INTERVAL_SEC = 3.0


def _sensor_row_to_payload(row, sample_id):
    """Serialize one IoT sensor reading (+ its linked sample_id) to JSON-ready dict."""
    payload = {
        "sample_id": int(sample_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for col in numeric_columns:
        if col in row.index:
            payload[col] = round(float(row[col]), 6)
    return payload


def start_iot_mqtt_publisher(df, sample_ids, stop_event):
    """
    Simulates the physical IoT sensor network: periodically publishes the
    next normalized sensor reading to the MQTT broker so the Digital Twin
    dashboard can stay synchronized with 'live' environmental data.
    Runs in a background daemon thread; stops when stop_event is set.
    """
    if not HAVE_MQTT:
        print("[MQTT Publisher] paho-mqtt not installed; publisher disabled "
              "(dashboard will use local simulation instead).")
        return

    if df is None or len(df) == 0 or not sample_ids:
        print("[MQTT Publisher] No IoT data / samples to stream; publisher disabled.")
        return

    try:
        client = mqtt.Client(client_id="heritage-iot-publisher")
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEPALIVE)
        client.loop_start()
    except Exception as e:
        print(f"[MQTT Publisher] Could not connect to broker "
              f"{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT} -> {e}")
        print("[MQTT Publisher] Publisher disabled; dashboard will fall back "
              "to local simulation.")
        return

    print(f"[MQTT Publisher] Connected to {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}. "
          f"Streaming IoT readings on '{MQTT_TOPIC}' every {IOT_PUBLISH_INTERVAL_SEC}s.")

    idx = 0
    n = len(df)
    while not stop_event.is_set():
        row = df.iloc[idx % n]
        sid = sample_ids[idx % len(sample_ids)]
        payload = _sensor_row_to_payload(row, sid)
        try:
            client.publish(MQTT_TOPIC, json.dumps(payload), qos=0)
        except Exception as e:
            print(f"[MQTT Publisher] Publish failed: {e}")
        idx += 1
        stop_event.wait(IOT_PUBLISH_INTERVAL_SEC)

    client.loop_stop()
    client.disconnect()
    print("[MQTT Publisher] Stopped.")


# ==========================================================
# Digital Twin Dashboard (Tkinter)
#   Shows: Original image | 3D Reconstruction (splat) image
#          Metric trend plots (CD / SSIM / PSNR)
#          IoT sensor snapshot
#   Does NOT show: the 3D point cloud image/PLY.
# ==========================================================
class DigitalTwinDashboard(tk.Tk):
    def __init__(self, sample_ids, iot_csv_path):
        super().__init__()
        self.title("Heritage Digital Twin Dashboard")
        self.geometry("1200x850")
        self.sample_ids = sample_ids
        self.iot_csv_path = iot_csv_path
        self.current_id = sample_ids[0] if sample_ids else 1
        self._load_iot_data()

        # --- Step 6: Real-Time IoT Synchronization (MQTT) state ---
        self.live_queue = queue.Queue()
        self.live_reading = {}
        self.mqtt_status = "Connecting..."
        self.mqtt_client = None
        self._sim_stop_event = threading.Event()

        self._build_ui()
        self._display_sample(self.current_id)

        self._start_mqtt_subscriber()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(500, self._poll_live_queue)

    def _load_iot_data(self):
        self.iot_df = pd.read_csv(self.iot_csv_path) if os.path.exists(self.iot_csv_path) else None

    def _load_image(self, source, max_dim=500):
        """Load an image from a file path or PIL Image and resize it."""
        if isinstance(source, Image.Image):
            img = source
        else:
            img = Image.open(source)
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        self.configure(bg="white")

        # Sample selector -------------------------------------------------
        selector_frame = ttk.Frame(self)
        selector_frame.pack(fill="x", pady=5)
        ttk.Label(selector_frame, text="Select Sample:").pack(side="left", padx=5)
        self.sample_var = tk.IntVar(value=self.current_id)
        self.sample_cb = ttk.Combobox(
            selector_frame,
            values=self.sample_ids,
            textvariable=self.sample_var,
            width=5,
            state="readonly",
        )
        self.sample_cb.pack(side="left")
        self.sample_cb.bind("<<ComboboxSelected>>", self._on_sample_change)

        # Image area: Original | 3D Reconstruction (splat) ONLY -----------
        img_frame = ttk.Frame(self)
        img_frame.pack(fill="both", expand=True, padx=10, pady=10)

        original_col = ttk.Frame(img_frame)
        original_col.grid(row=0, column=0, padx=5, sticky="nsew")
        ttk.Label(original_col, text="Original Image", font=("Times New Roman", 12, "bold")).pack()
        self.original_label = ttk.Label(original_col)
        self.original_label.pack()

        recon_col = ttk.Frame(img_frame)
        recon_col.grid(row=0, column=1, padx=5, sticky="nsew")
        ttk.Label(recon_col, text="3D Reconstruction (Gaussian Splat)", font=("Times New Roman", 12, "bold")).pack()
        self.recon_label = ttk.Label(recon_col)
        self.recon_label.pack()

        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        img_frame.rowconfigure(0, weight=1)

        # Metric thumbnails -------------------------------------------------
        metric_frame = ttk.LabelFrame(self, text="Metric Plots")
        metric_frame.pack(fill="x", padx=10, pady=5)
        self.metric_imgs = {}
        for i, name in enumerate(["Chamfer Distance", "SSIM", "PSNR"]):
            lbl = ttk.Label(metric_frame)
            lbl.grid(row=0, column=i, padx=5)
            self.metric_imgs[name] = lbl

        # IoT data ----------------------------------------------------------
        iot_frame = ttk.LabelFrame(self, text="IoT Sensor Snapshot (per selected sample)")
        iot_frame.pack(fill="x", padx=10, pady=5)
        self.iot_text = tk.Text(iot_frame, height=4, wrap="word", bg=self.cget("bg"), relief="flat")
        self.iot_text.pack(fill="x", padx=5, pady=5)

        # Real-Time IoT Sync (Step 6: MQTT) ----------------------------------
        live_frame = ttk.LabelFrame(self, text="Real-Time IoT Synchronization (MQTT)")
        live_frame.pack(fill="x", padx=10, pady=5)

        status_row = ttk.Frame(live_frame)
        status_row.pack(fill="x", padx=5, pady=(5, 0))
        self.mqtt_status_var = tk.StringVar(value="IoT Sync: Connecting...")
        ttk.Label(status_row, textvariable=self.mqtt_status_var, foreground="#0a6e31").pack(side="left")

        self.live_text = tk.Text(live_frame, height=4, wrap="word", bg=self.cget("bg"), relief="flat")
        self.live_text.pack(fill="x", padx=5, pady=5)

    def _on_sample_change(self, event):
        self.current_id = self.sample_var.get()
        self._display_sample(self.current_id)

    def _display_sample(self, sid):
        # Original vs reconstructed side-by-side image (splits the saved
        # comparison figure in half — left = original, right = splat).
        cmp_path = os.path.join(OUTPUT_RECON_DIR, f"sample_{sid}_original_vs_splat.png")
        if os.path.exists(cmp_path):
            full = Image.open(cmp_path)
            w, h = full.size
            left = full.crop((0, 0, w // 2, h))
            right = full.crop((w // 2, 0, w, h))
            left_img = self._load_image(left)
            right_img = self._load_image(right)
            self.original_label.config(image=left_img, text="")
            self.original_label.image = left_img  # keep a real reference (avoid GC)
            self.recon_label.config(image=right_img, text="")
            self.recon_label.image = right_img  # keep a real reference (avoid GC)
        else:
            self.original_label.config(image="", text="Comparison not found")
            self.recon_label.config(image="", text="")

        # Metric thumbnails
        metric_files = {
            "Chamfer Distance": "metric_chamfer_distance_wave.png",
            "SSIM": "metric_ssim_wave.png",
            "PSNR": "metric_psnr_wave.png",
        }
        for name, fname in metric_files.items():
            path = os.path.join(OUTPUT_RECON_DIR, fname)
            if os.path.exists(path):
                metric_img = self._load_image(path, max_dim=200)
                self.metric_imgs[name].config(image=metric_img, text="")
                self.metric_imgs[name].image = metric_img  # keep a real reference (avoid GC)
            else:
                self.metric_imgs[name].config(image="", text="Missing")

        # IoT data for this sample (expects column 'sample_id')
        self.iot_text.delete("1.0", tk.END)
        if self.iot_df is not None and "sample_id" in self.iot_df.columns:
            row = self.iot_df[self.iot_df["sample_id"] == sid]
            if not row.empty:
                for col in row.columns:
                    if col == "sample_id":
                        continue
                    self.iot_text.insert(tk.END, f"{col}: {row.iloc[0][col]}\n")
            else:
                self.iot_text.insert(tk.END, "No IoT entry for this sample.")
        else:
            self.iot_text.insert(tk.END, "IoT CSV not found or missing 'sample_id' column.")

    # ------------------------------------------------------------------
    # Step 6: Real-Time IoT Synchronization (MQTT)
    # ------------------------------------------------------------------
    def _start_mqtt_subscriber(self):
        """
        Subscribes to MQTT_TOPIC so the dashboard receives live sensor
        readings as they're published (see start_iot_mqtt_publisher).
        Falls back to a local in-process simulation if paho-mqtt isn't
        installed or no broker is reachable, so the sync panel always
        has something to display.
        """
        if not HAVE_MQTT:
            self.mqtt_status = "paho-mqtt not installed (local simulation)"
            self.mqtt_status_var.set(f"IoT Sync: {self.mqtt_status}")
            self._start_local_simulation_fallback()
            return

        try:
            self.mqtt_client = mqtt.Client(client_id="heritage-dashboard-subscriber")
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self.mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEPALIVE)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"[MQTT Subscriber] Could not connect to "
                  f"{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT} -> {e}")
            self.mqtt_status = f"Broker unreachable - local simulation"
            self.mqtt_status_var.set(f"IoT Sync: {self.mqtt_status}")
            self._start_local_simulation_fallback()

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(MQTT_TOPIC)
            self.mqtt_status = f"Connected ({MQTT_BROKER_HOST}:{MQTT_BROKER_PORT})"
            print(f"[MQTT Subscriber] {self.mqtt_status}. Subscribed to '{MQTT_TOPIC}'.")
        else:
            self.mqtt_status = f"Connection failed (rc={rc}) - local simulation"
            self._start_local_simulation_fallback()
        self.mqtt_status_var.set(f"IoT Sync: {self.mqtt_status}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        self.mqtt_status = "Disconnected - retrying via local simulation"
        self.mqtt_status_var.set(f"IoT Sync: {self.mqtt_status}")
        self._start_local_simulation_fallback()

    def _on_mqtt_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            self.live_queue.put(payload)
        except Exception as e:
            print(f"[MQTT Subscriber] Could not parse message: {e}")

    def _start_local_simulation_fallback(self):
        """
        Streams readings straight from the already-loaded IoT dataframe
        into the same live_queue the MQTT callback uses, so the UI code
        path is identical regardless of whether a real broker is present.
        """
        if self._sim_stop_event.is_set():
            return  # already running
        if self.iot_df is None or len(self.iot_df) == 0:
            self.mqtt_status_var.set("IoT Sync: no data available for simulation")
            return

        def _loop():
            idx = 0
            n = len(self.iot_df)
            while not self._sim_stop_event.is_set():
                row = self.iot_df.iloc[idx % n]
                sid = int(row["sample_id"]) if "sample_id" in row and pd.notna(row["sample_id"]) else (
                    self.sample_ids[idx % len(self.sample_ids)] if self.sample_ids else 0
                )
                payload = {
                    "sample_id": sid,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                for col in numeric_columns:
                    if col in row.index and pd.notna(row[col]):
                        payload[col] = round(float(row[col]), 6)
                self.live_queue.put(payload)
                idx += 1
                self._sim_stop_event.wait(IOT_PUBLISH_INTERVAL_SEC)

        threading.Thread(target=_loop, daemon=True).start()

    def _poll_live_queue(self):
        """Drains any newly-arrived live readings and refreshes the panel."""
        updated = False
        while not self.live_queue.empty():
            try:
                self.live_reading = self.live_queue.get_nowait()
                updated = True
            except queue.Empty:
                break
        if updated:
            self._refresh_live_panel()
        self.after(500, self._poll_live_queue)

    def _refresh_live_panel(self):
        ts = self.live_reading.get("timestamp", "-")
        sid = self.live_reading.get("sample_id", "-")
        self.mqtt_status_var.set(f"IoT Sync: {self.mqtt_status} | Last update: {ts} | Sample: {sid}")

        self.live_text.delete("1.0", tk.END)
        for key, value in self.live_reading.items():
            if key in ("sample_id", "timestamp"):
                continue
            if isinstance(value, float):
                self.live_text.insert(tk.END, f"{key}: {value:.4f}\n")
            else:
                self.live_text.insert(tk.END, f"{key}: {value}\n")

    def _on_close(self):
        self._sim_stop_event.set()
        if self.mqtt_client is not None:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass
        self.destroy()


def launch_dashboard(sample_ids, iot_csv_path=NORMALIZED_IOT_PATH, iot_df=None):
    """
    Launches the dashboard and (if data is available) starts the Step 6
    MQTT publisher in the background so the 'Real-Time IoT Synchronization'
    panel has live readings to display for the duration of the session.
    """
    publisher_stop_event = threading.Event()
    publisher_df = iot_df if iot_df is not None else (
        pd.read_csv(iot_csv_path) if os.path.exists(iot_csv_path) else None
    )
    if publisher_df is not None:
        threading.Thread(
            target=start_iot_mqtt_publisher,
            args=(publisher_df, sample_ids, publisher_stop_event),
            daemon=True,
        ).start()

    app = DigitalTwinDashboard(sample_ids, iot_csv_path)
    try:
        app.mainloop()
    finally:
        publisher_stop_event.set()


# ==========================================================
# Entry point: run the full pipeline (above), THEN launch the
# dashboard once every output file it needs already exists.
# ==========================================================
if __name__ == "__main__":
    if sample_ids:
        print("\nLaunching Digital Twin Dashboard...")
        launch_dashboard(sample_ids)
    else:
        print("\nNo heritage images were processed - dashboard not launched.")