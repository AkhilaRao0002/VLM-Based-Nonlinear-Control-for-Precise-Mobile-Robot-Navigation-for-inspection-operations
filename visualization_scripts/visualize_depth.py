import cv2
import numpy as np


# ============================================================
# TARTANAIR DEPTH VISUALIZATION
# ============================================================

DEPTH_FILE = (
    r"..\tartanair_data\AbandonedFactory\Data_omni\P0000"
    r"\depth_lcam_front\000010_lcam_front_depth.png"
)

OUTPUT_FILE = "depth_visualization.png"


print("=" * 60)
print("TARTANAIR DEPTH VISUALIZATION")
print("=" * 60)


# ============================================================
# LOAD DEPTH
# ============================================================

depth_rgba = cv2.imread(
    DEPTH_FILE,
    cv2.IMREAD_UNCHANGED
)

if depth_rgba is None:
    print("[ERROR] Could not load depth.")
    raise SystemExit


# ============================================================
# DECODE
# ============================================================

depth = depth_rgba.view("<f4")
depth = np.squeeze(depth, axis=-1)


# ============================================================
# LIMIT DEPTH
# ============================================================

MAX_DEPTH = 20.0

depth_display = np.clip(
    depth,
    0,
    MAX_DEPTH
)


# ============================================================
# NORMALIZE
# ============================================================

depth_normalized = (
    depth_display / MAX_DEPTH * 255
).astype(np.uint8)


# ============================================================
# CREATE DEPTH IMAGE
# ============================================================

depth_colored = cv2.applyColorMap(
    depth_normalized,
    cv2.COLORMAP_JET
)


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    OUTPUT_FILE,
    depth_colored
)


print()
print("[SUCCESS] Depth visualization saved:")
print(OUTPUT_FILE)

print()
print("Open this file and check whether")
print("near and far surfaces are visually separated.")

print("=" * 60)