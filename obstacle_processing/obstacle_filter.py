import cv2
import numpy as np


# ============================================================
# TARTANAIR DEPTH OBSTACLE FILTER
# ============================================================

DEPTH_FILE = (
    r"..\tartanair_data\AbandonedFactory\Data_omni\P0000"
    r"\depth_lcam_front\000010_lcam_front_depth.png"
)

OUTPUT_FILE = "filtered_obstacles.png"


print("=" * 60)
print("TARTANAIR DEPTH OBSTACLE FILTER")
print("=" * 60)


# ============================================================
# LOAD DEPTH
# ============================================================

depth_rgba = cv2.imread(
    DEPTH_FILE,
    cv2.IMREAD_UNCHANGED
)

if depth_rgba is None:
    print("[ERROR] Could not load depth image.")
    raise SystemExit


# ============================================================
# DECODE TARTANAIR FLOAT32 DEPTH
# ============================================================

depth = depth_rgba.view("<f4")
depth = np.squeeze(depth, axis=-1)

height, width = depth.shape

print("[TEST] Depth shape:", depth.shape)


# ============================================================
# VALID DEPTH
# ============================================================

valid = (
    np.isfinite(depth)
    & (depth > 0.1)
    & (depth < 20.0)
)


# ============================================================
# NEAR-OBJECT MASK
# ============================================================

near_mask = (
    valid
    & (depth < 3.0)
)


# ============================================================
# REMOVE VERY LOW IMAGE REGION
#
# The extreme bottom of the camera image is dominated by
# nearby ground. We don't want that region to dominate the
# obstacle detector.
# ============================================================

y_limit = int(height * 0.85)

near_mask[y_limit:, :] = False


# ============================================================
# DEPTH GRADIENT
#
# Obstacles often create depth discontinuities.
# A smooth floor generally changes more gradually.
# ============================================================

depth_clean = depth.copy()

# Replace invalid values before gradient calculation.
depth_clean[~valid] = 20.0


gradient_x = cv2.Sobel(
    depth_clean,
    cv2.CV_32F,
    1,
    0,
    ksize=3
)

gradient_y = cv2.Sobel(
    depth_clean,
    cv2.CV_32F,
    0,
    1,
    ksize=3
)

gradient_magnitude = np.sqrt(
    gradient_x ** 2 +
    gradient_y ** 2
)


# ============================================================
# DEPTH DISCONTINUITY
# ============================================================

GRADIENT_THRESHOLD = 0.8

edge_mask = (
    gradient_magnitude > GRADIENT_THRESHOLD
)


# ============================================================
# COMBINE NEAR DEPTH + DEPTH DISCONTINUITY
# ============================================================

obstacle_mask = (
    near_mask &
    edge_mask
)


# ============================================================
# MORPHOLOGICAL FILTERING
# ============================================================

mask = (
    obstacle_mask.astype(np.uint8) *
    255
)

kernel = np.ones(
    (5, 5),
    np.uint8
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)


# ============================================================
# REMOVE VERY SMALL COMPONENTS
# ============================================================

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mask,
    connectivity=8
)

filtered = np.zeros_like(mask)

MIN_AREA = 100

for label in range(1, num_labels):

    area = stats[label, cv2.CC_STAT_AREA]

    if area >= MIN_AREA:

        filtered[labels == label] = 255


# ============================================================
# STATISTICS
# ============================================================

obstacle_pixels = np.count_nonzero(filtered)

total_pixels = filtered.size

coverage = (
    obstacle_pixels /
    total_pixels *
    100
)


print()
print("=" * 60)
print("FILTERED OBSTACLE RESULTS")
print("=" * 60)

print("Gradient threshold:",
      GRADIENT_THRESHOLD)

print("Minimum component area:",
      MIN_AREA)

print("Obstacle pixels:",
      obstacle_pixels)

print(
    f"Obstacle coverage: "
    f"{coverage:.2f}%"
)


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    OUTPUT_FILE,
    filtered
)

print()
print("[SUCCESS] Saved:")
print(OUTPUT_FILE)

print("=" * 60)
print("OBSTACLE FILTER COMPLETE")
print("=" * 60)