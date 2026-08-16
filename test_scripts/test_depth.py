import os
import cv2
import numpy as np


# ============================================================
# TARTANAIR DEPTH DECODING TEST
# ============================================================

DEPTH_FILE = (
    r"..\tartanair_data\AbandonedFactory\Data_omni\P0000"
    r"\depth_lcam_front\000010_lcam_front_depth.png"
)


print("=" * 60)
print("TARTANAIR DEPTH DECODING TEST")
print("=" * 60)

print("[TEST] File:", DEPTH_FILE)
print("[TEST] Exists:", os.path.exists(DEPTH_FILE))


if not os.path.exists(DEPTH_FILE):
    print("[ERROR] Depth file not found.")
    raise SystemExit


# ============================================================
# LOAD 4-CHANNEL PNG
# ============================================================

depth_rgba = cv2.imread(
    DEPTH_FILE,
    cv2.IMREAD_UNCHANGED
)

print()
print("[TEST] Encoded depth loaded")
print("Shape:", depth_rgba.shape)
print("Data type:", depth_rgba.dtype)


# ============================================================
# DECODE UINT8 x 4 -> FLOAT32
# ============================================================

depth = depth_rgba.view("<f4")

depth = np.squeeze(depth, axis=-1)


# ============================================================
# DEPTH INFORMATION
# ============================================================

print()
print("[TEST] Decoded depth")

print("Shape:", depth.shape)
print("Data type:", depth.dtype)

print("Minimum:", np.nanmin(depth))
print("Maximum:", np.nanmax(depth))
print("Mean:", np.nanmean(depth))


# ============================================================
# CENTER PIXEL
# ============================================================

h, w = depth.shape

center_y = h // 2
center_x = w // 2

center_depth = depth[center_y, center_x]

print()
print("[TEST] Center pixel")

print("X:", center_x)
print("Y:", center_y)
print("Depth:", center_depth, "meters")


# ============================================================
# SOME SAMPLE PIXELS
# ============================================================

print()
print("[TEST] Sample depths")

points = [
    (320, 320),
    (160, 320),
    (480, 320),
    (320, 160),
    (320, 480),
]

for x, y in points:
    print(
        f"Pixel ({x:3d}, {y:3d}) -> "
        f"{depth[y, x]:.4f} m"
    )


print("=" * 60)
print("DEPTH DECODING TEST COMPLETE")
print("=" * 60)