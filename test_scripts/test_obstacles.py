import os
import cv2
import numpy as np


# ============================================================
# TARTANAIR OBSTACLE DISTANCE TEST
# ============================================================

DEPTH_FILE = (
    r"..\tartanair_data\AbandonedFactory\Data_omni\P0000"
    r"\depth_lcam_front\000010_lcam_front_depth.png"
)


print("=" * 60)
print("TARTANAIR OBSTACLE DISTANCE TEST")
print("=" * 60)


# ============================================================
# LOAD ENCODED DEPTH
# ============================================================

print("[TEST] Loading depth...")

depth_rgba = cv2.imread(
    DEPTH_FILE,
    cv2.IMREAD_UNCHANGED
)

if depth_rgba is None:
    print("[ERROR] Could not load depth image.")
    raise SystemExit


# ============================================================
# DECODE TARTANAIR DEPTH
# ============================================================

depth = depth_rgba.view("<f4")
depth = np.squeeze(depth, axis=-1)


print("[TEST] Depth shape:", depth.shape)


# ============================================================
# FILTER INVALID / EXTREME DEPTH
# ============================================================

MAX_VALID_DEPTH = 50.0

valid_depth = np.where(
    np.isfinite(depth) & (depth > 0) & (depth < MAX_VALID_DEPTH),
    depth,
    np.nan
)


# ============================================================
# CAMERA REGIONS
# ============================================================

height, width = valid_depth.shape

# Ignore the extreme top and bottom of the image.
# This reduces influence from sky/ceiling and very close ground.

y_start = int(height * 0.25)
y_end = int(height * 0.85)

x_left_end = int(width * 0.33)
x_center_end = int(width * 0.66)


left_region = valid_depth[
    y_start:y_end,
    0:x_left_end
]

center_region = valid_depth[
    y_start:y_end,
    x_left_end:x_center_end
]

right_region = valid_depth[
    y_start:y_end,
    x_center_end:width
]


# ============================================================
# ROBUST NEAREST DISTANCE
# ============================================================

def nearest_distance(region):
    """
    Return a robust estimate of the nearest obstacle.

    We use the 5th percentile instead of the absolute minimum
    so that a single noisy pixel does not dominate the result.
    """

    values = region[np.isfinite(region)]

    if len(values) == 0:
        return float("nan")

    return float(np.percentile(values, 5))


left_distance = nearest_distance(left_region)
center_distance = nearest_distance(center_region)
right_distance = nearest_distance(right_region)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("OBSTACLE DISTANCES")
print("=" * 60)

print(f"Left   : {left_distance:.2f} m")
print(f"Center : {center_distance:.2f} m")
print(f"Right  : {right_distance:.2f} m")


# ============================================================
# OVERALL NEAREST OBSTACLE
# ============================================================

distances = {
    "LEFT": left_distance,
    "CENTER": center_distance,
    "RIGHT": right_distance
}

valid_distances = {
    region: distance
    for region, distance in distances.items()
    if np.isfinite(distance)
}


if valid_distances:

    nearest_region = min(
        valid_distances,
        key=valid_distances.get
    )

    nearest_distance_value = valid_distances[
        nearest_region
    ]

    print()
    print("Nearest region :", nearest_region)
    print(
        f"Nearest distance : "
        f"{nearest_distance_value:.2f} m"
    )

else:

    print()
    print("No valid depth measurements found.")


# ============================================================
# SIMPLE NAVIGATION INTERPRETATION
# ============================================================

print()
print("=" * 60)
print("BASIC NAVIGATION OBSERVATION")
print("=" * 60)


SAFE_DISTANCE = 2.0


if np.isfinite(center_distance):

    if center_distance < SAFE_DISTANCE:

        print(
            "CENTER PATH: OBSTACLE TOO CLOSE"
        )

    else:

        print(
            "CENTER PATH: CLEAR"
        )

else:

    print(
        "CENTER PATH: UNKNOWN"
    )


print("=" * 60)
print("OBSTACLE TEST COMPLETE")
print("=" * 60)