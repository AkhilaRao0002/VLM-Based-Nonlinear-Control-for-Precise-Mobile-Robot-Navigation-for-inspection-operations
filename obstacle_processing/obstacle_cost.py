import numpy as np

print("=" * 60)
print("OBSTACLE COST / NAVIGATION DECISION")
print("=" * 60)

# ============================================================
# LOAD 3D OBSTACLE POINTS
# ============================================================

points = np.load("obstacle_points_3d.npy")

# TartanAir convention:
# points[:, 0] = forward distance
# points[:, 1] = horizontal position
# points[:, 2] = vertical position

z = points[:, 0]
x = points[:, 1]

print("[TEST] Loaded 3D obstacle points:", len(points))

# ============================================================
# SAFETY PARAMETERS
# ============================================================

SAFETY_DISTANCE = 1.0

LEFT_THRESHOLD = -0.5
RIGHT_THRESHOLD = 0.5

# ============================================================
# LEFT / CENTER / RIGHT
# ============================================================

left = x < LEFT_THRESHOLD
center = (x >= LEFT_THRESHOLD) & (x <= RIGHT_THRESHOLD)
right = x > RIGHT_THRESHOLD

def get_distance(mask):
    if np.any(mask):
        return float(np.min(z[mask]))
    return float("inf")

left_distance = get_distance(left)
center_distance = get_distance(center)
right_distance = get_distance(right)

# ============================================================
# DISPLAY DISTANCES
# ============================================================

print()
print("=" * 60)
print("OBSTACLE DISTANCES")
print("=" * 60)

print(
    "Left   :",
    "No obstacle" if np.isinf(left_distance)
    else f"{left_distance:.3f} m"
)

print(
    "Center :",
    "No obstacle" if np.isinf(center_distance)
    else f"{center_distance:.3f} m"
)

print(
    "Right  :",
    "No obstacle" if np.isinf(right_distance)
    else f"{right_distance:.3f} m"
)

# ============================================================
# NAVIGATION DECISION
# ============================================================

print()
print("=" * 60)
print("NAVIGATION DECISION")
print("=" * 60)

if center_distance < SAFETY_DISTANCE:

    print("CENTER: OBSTACLE TOO CLOSE")

    if right_distance > SAFETY_DISTANCE:
        decision = "AVOID RIGHT"

    elif left_distance > SAFETY_DISTANCE:
        decision = "AVOID LEFT"

    else:
        decision = "STOP"

else:

    if left_distance < SAFETY_DISTANCE:
        decision = "SLIGHTLY RIGHT"

    elif right_distance < SAFETY_DISTANCE:
        decision = "SLIGHTLY LEFT"

    else:
        decision = "FORWARD"

print()
print("Safety distance :", SAFETY_DISTANCE, "m")
print("Decision        :", decision)

# ============================================================
# NMPC REFERENCE
# ============================================================

print()
print("=" * 60)
print("NMPC NAVIGATION INPUT")
print("=" * 60)

if decision == "FORWARD":
    steering_bias = 0.0
    speed_scale = 1.0

elif decision == "SLIGHTLY_RIGHT":
    steering_bias = 0.25
    speed_scale = 0.8

elif decision == "SLIGHTLY_LEFT":
    steering_bias = -0.25
    speed_scale = 0.8

elif decision == "AVOID_RIGHT":
    steering_bias = 0.5
    speed_scale = 0.5

elif decision == "AVOID_LEFT":
    steering_bias = -0.5
    speed_scale = 0.5

else:
    steering_bias = 0.0
    speed_scale = 0.0

print("Steering bias :", steering_bias)
print("Speed scale   :", speed_scale)

print()
print("=" * 60)
print("OBSTACLE COST TEST COMPLETE")
print("=" * 60)