import cv2
import numpy as np


# ============================================================
# TARTANAIR OBSTACLE REGION DETECTION
# ============================================================

DEPTH_FILE = (
    r"..\tartanair_data\AbandonedFactory\Data_omni\P0000"
    r"\depth_lcam_front\000010_lcam_front_depth.png"
)

OUTPUT_FILE = "obstacle_regions.png"


print("=" * 60)
print("TARTANAIR OBSTACLE REGION DETECTION")
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

height, width = depth.shape


# ============================================================
# VALID DEPTH
# ============================================================

valid = (
    np.isfinite(depth)
    & (depth > 0.1)
    & (depth < 10.0)
)


# ============================================================
# NEAR DEPTH
# ============================================================

OBSTACLE_DISTANCE = 3.0

mask = (
    valid &
    (depth < OBSTACLE_DISTANCE)
)


# ============================================================
# REMOVE EXTREME BOTTOM
# ============================================================

bottom_start = int(height * 0.85)

mask[bottom_start:, :] = False


# ============================================================
# MORPHOLOGICAL CLEANING
# ============================================================

mask_image = (
    mask.astype(np.uint8) *
    255
)

kernel = np.ones(
    (7, 7),
    np.uint8
)

mask_image = cv2.morphologyEx(
    mask_image,
    cv2.MORPH_OPEN,
    kernel
)

mask_image = cv2.morphologyEx(
    mask_image,
    cv2.MORPH_CLOSE,
    kernel
)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        mask_image,
        connectivity=8
    )
)


MIN_AREA = 300

output = np.zeros_like(mask_image)

regions = []


for label in range(1, num_labels):

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
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "area": area,
        "nearest_depth": nearest
    })


    output[
        labels == label
    ] = 255


# ============================================================
# SORT BY DISTANCE
# ============================================================

regions.sort(
    key=lambda r: r["nearest_depth"]
)


# ============================================================
# PRINT REGIONS
# ============================================================

print()
print("=" * 60)
print("DETECTED OBSTACLE REGIONS")
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
    output
)


print()
print("[SUCCESS] Saved:")
print(OUTPUT_FILE)

print("=" * 60)
print("OBSTACLE REGION TEST COMPLETE")
print("=" * 60)