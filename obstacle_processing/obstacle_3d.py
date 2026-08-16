import cv2
import numpy as np
from pathlib import Path

print("=" * 60)
print("TARTANAIR 3D OBSTACLE PROJECTION")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"..\tartanair_data\AbandonedFactory\Data_omni\P0000")

DEPTH_FILE = BASE / "depth_lcam_front" / "000010_lcam_front_depth.png"
MASK_FILE = Path("filtered_obstacles.png")

# ============================================================
# TARTANAIR CAMERA INTRINSICS
# ============================================================

FX = 320.0
FY = 320.0
CX = 320.0
CY = 320.0

# ============================================================
# LOAD DEPTH
# ============================================================

print("[TEST] Loading depth...")

depth_rgba = cv2.imread(str(DEPTH_FILE), cv2.IMREAD_UNCHANGED)

if depth_rgba is None:
    print("[ERROR] Depth file could not be loaded.")
    raise SystemExit

# TartanAir encoded depth
depth = depth_rgba.view("<f4").squeeze()

print("[TEST] Depth shape:", depth.shape)

# ============================================================
# LOAD OBSTACLE MASK
# ============================================================

print("[TEST] Loading obstacle mask...")

mask = cv2.imread(str(MASK_FILE), cv2.IMREAD_GRAYSCALE)

if mask is None:
    print("[ERROR] Obstacle mask not found:", MASK_FILE)
    raise SystemExit

print("[TEST] Mask shape:", mask.shape)

# Resize if necessary
if mask.shape != depth.shape:
    mask = cv2.resize(
        mask,
        (depth.shape[1], depth.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

# ============================================================
# GET OBSTACLE PIXELS
# ============================================================

obstacle_pixels = mask > 0

count = np.sum(obstacle_pixels)

print("[TEST] Obstacle pixels:", count)

if count == 0:
    print("[ERROR] No obstacle pixels detected.")
    raise SystemExit

# ============================================================
# PIXEL COORDINATES
# ============================================================

v, u = np.where(obstacle_pixels)

z = depth[v, u]

# Remove invalid depth
valid = (
    np.isfinite(z)
    & (z > 0.1)
    & (z < 100.0)
)

u = u[valid]
v = v[valid]
z = z[valid]

# ============================================================
# 3D PROJECTION
# ============================================================

# TartanAir point-cloud convention:
#
# forward = depth
# horizontal = x
# vertical = y

x = (u - CX) * z / FX
y = (v - CY) * z / FY

points_3d = np.stack(
    [z, x, y],
    axis=1
)

# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("3D OBSTACLE RESULTS")
print("=" * 60)

print("Valid obstacle points:", len(points_3d))

print()
print("Forward distance:")
print("Minimum :", round(np.min(z), 3), "m")
print("Mean    :", round(np.mean(z), 3), "m")
print("Maximum :", round(np.max(z), 3), "m")

print()
print("Horizontal position:")
print("Minimum X:", round(np.min(x), 3), "m")
print("Maximum X:", round(np.max(x), 3), "m")

print()
print("Vertical position:")
print("Minimum Y:", round(np.min(y), 3), "m")
print("Maximum Y:", round(np.max(y), 3), "m")

# ============================================================
# NEAREST OBSTACLE
# ============================================================

nearest_idx = np.argmin(z)

nearest_distance = z[nearest_idx]
nearest_x = x[nearest_idx]
nearest_y = y[nearest_idx]

print()
print("NEAREST OBSTACLE")
print("----------------")

print("Distance :", round(nearest_distance, 3), "m")
print("X        :", round(nearest_x, 3), "m")
print("Y        :", round(nearest_y, 3), "m")

# ============================================================
# LEFT / CENTER / RIGHT
# ============================================================

left = x < -0.5
center = (x >= -0.5) & (x <= 0.5)
right = x > 0.5

print()
print("OBSTACLE DISTANCES")
print("------------------")

if np.any(left):
    print("Left   :", round(np.min(z[left]), 3), "m")
else:
    print("Left   : No obstacle")

if np.any(center):
    print("Center :", round(np.min(z[center]), 3), "m")
else:
    print("Center : No obstacle")

if np.any(right):
    print("Right  :", round(np.min(z[right]), 3), "m")
else:
    print("Right  : No obstacle")

# ============================================================
# SAVE 3D POINTS
# ============================================================

output = Path("obstacle_points_3d.npy")

np.save(output, points_3d)

print()
print("[SUCCESS] Saved:")
print(output)

print("=" * 60)
print("3D OBSTACLE PROJECTION COMPLETE")
print("=" * 60)