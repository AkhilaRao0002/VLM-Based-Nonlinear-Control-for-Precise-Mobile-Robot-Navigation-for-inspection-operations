import cv2
import numpy as np


# ============================================================
# TARTANAIR GROUND-REMOVED OBSTACLE TEST
# ============================================================

DEPTH_FILE = (
    r"..\tartanair_data\AbandonedFactory\Data_omni\P0000"
    r"\depth_lcam_front\000010_lcam_front_depth.png"
)

OUTPUT_FILE = "ground_removed_obstacles.png"


print("=" * 60)
print("TARTANAIR GROUND-REMOVED OBSTACLE TEST")
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
# DECODE TARTANAIR DEPTH
# ============================================================

depth = depth_rgba.view("<f4")
depth = np.squeeze(depth, axis=-1)

height, width = depth.shape


# ============================================================
# VALID DEPTH
# ============================================================

valid = (
    np.isfinite(depth)
    & (depth > 0.1)
    & (depth < 30.0)
)


# ============================================================
# ESTIMATE LOCAL GROUND DEPTH
#
# For each image column, use the lower part of the image
# to estimate the expected ground depth.
# ============================================================

ground_start = int(height * 0.65)
ground_end = int(height * 0.85)

ground_reference = np.full(
    width,
    np.nan,
    dtype=np.float32
)


for x in range(width):

    column = depth[
        ground_start:ground_end,
        x
    ]

    column_valid = column[
        np.isfinite(column)
        & (column > 0.1)
        & (column < 30.0)
    ]

    if len(column_valid) > 0:

        ground_reference[x] = np.percentile(
            column_valid,
            50
        )


# ============================================================
# INTERPOLATE MISSING GROUND VALUES
# ============================================================

valid_columns = np.isfinite(
    ground_reference
)

if np.count_nonzero(valid_columns) < 10:

    print("[ERROR] Could not estimate ground.")
    raise SystemExit


x_indices = np.arange(width)

ground_reference = np.interp(
    x_indices,
    x_indices[valid_columns],
    ground_reference[valid_columns]
)


# ============================================================
# BUILD EXPECTED GROUND DEPTH IMAGE
# ============================================================

expected_ground = np.tile(
    ground_reference,
    (height, 1)
)


# ============================================================
# GROUND DEVIATION
# ============================================================

depth_difference = (
    expected_ground - depth
)


# ============================================================
# POTENTIAL OBSTACLES
#
# A surface substantially closer than the estimated ground
# is treated as a potential obstacle.
# ============================================================

GROUND_THRESHOLD = 0.5

obstacle_mask = (
    valid
    & (depth_difference > GROUND_THRESHOLD)
)


# ============================================================
# LIMIT NAVIGATION REGION
# ============================================================

top_limit = int(height * 0.15)
bottom_limit = int(height * 0.90)

obstacle_mask[:top_limit, :] = False
obstacle_mask[bottom_limit:, :] = False


# ============================================================
# MORPHOLOGICAL CLEANING
# ============================================================

mask = (
    obstacle_mask.astype(np.uint8)
    * 255
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
# CONNECTED COMPONENTS
# ============================================================

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )
)


MIN_AREA = 100

filtered = np.zeros_like(mask)

regions = []


for label in range(
    1,
    num_labels
):

    area = stats[
        label,
        cv2.CC_STAT_AREA
    ]

    if area < MIN_AREA:
        continue


    x = stats[
        label,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        label,
        cv2.CC_STAT_TOP
    ]

    w = stats[
        label,
        cv2.CC_STAT_WIDTH
    ]

    h = stats[
        label,
        cv2.CC_STAT_HEIGHT
    ]


    region_depth = depth[
        labels == label
    ]

    region_depth = region_depth[
        np.isfinite(region_depth)
    ]


    if len(region_depth) == 0:
        continue


    nearest = float(
        np.percentile(
            region_depth,
            5
        )
    )


    regions.append({
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "area": int(area),
        "nearest_depth": nearest
    })


    filtered[
        labels == label
    ] = 255


# ============================================================
# SORT BY DISTANCE
# ============================================================

regions.sort(
    key=lambda r: r["nearest_depth"]
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("GROUND-REMOVED OBSTACLE REGIONS")
print("=" * 60)


if not regions:

    print("No obstacle regions detected.")

else:

    for i, region in enumerate(
        regions[:10],
        start=1
    ):

        print(
            f"Region {i}: "
            f"x={region['x']} "
            f"y={region['y']} "
            f"width={region['width']} "
            f"height={region['height']} "
            f"area={region['area']} "
            f"nearest={region['nearest_depth']:.2f} m"
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
print("GROUND REMOVAL TEST COMPLETE")
print("=" * 60)