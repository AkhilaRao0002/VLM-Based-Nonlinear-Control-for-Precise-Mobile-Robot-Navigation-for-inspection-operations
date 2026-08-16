import numpy as np

print("=" * 60)
print("NMPC OBSTACLE COST TEST")
print("=" * 60)

# ============================================================
# LOAD 3D OBSTACLE POINTS
# ============================================================

points = np.load("obstacle_points_3d.npy")

# TartanAir convention
z = points[:, 0]   # forward
x = points[:, 1]   # horizontal
y = points[:, 2]   # vertical

print("[TEST] Loaded obstacle points:", len(points))

# ============================================================
# NMPC PARAMETERS
# ============================================================

SAFETY_DISTANCE = 1.0
INFLUENCE_DISTANCE = 2.0

# Weight of obstacle avoidance in NMPC objective
OBSTACLE_WEIGHT = 10.0

# ============================================================
# FIND NEAREST OBSTACLE
# ============================================================

nearest_idx = np.argmin(z)

obstacle_distance = float(z[nearest_idx])
obstacle_x = float(x[nearest_idx])
obstacle_y = float(y[nearest_idx])

print()
print("=" * 60)
print("NEAREST OBSTACLE")
print("=" * 60)

print("Distance :", round(obstacle_distance, 3), "m")
print("X        :", round(obstacle_x, 3), "m")
print("Y        :", round(obstacle_y, 3), "m")

# ============================================================
# OBSTACLE COST
# ============================================================

if obstacle_distance >= INFLUENCE_DISTANCE:

    obstacle_cost = 0.0

elif obstacle_distance > SAFETY_DISTANCE:

    # Smooth cost inside influence region
    normalized = (
        INFLUENCE_DISTANCE - obstacle_distance
    ) / (
        INFLUENCE_DISTANCE - SAFETY_DISTANCE
    )

    obstacle_cost = OBSTACLE_WEIGHT * normalized ** 2

else:

    # Strong penalty when inside safety distance
    violation = SAFETY_DISTANCE - obstacle_distance

    obstacle_cost = (
        OBSTACLE_WEIGHT
        + OBSTACLE_WEIGHT * violation ** 2
    )

# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 60)
print("NMPC OBSTACLE COST")
print("=" * 60)

print("Safety distance    :", SAFETY_DISTANCE, "m")
print("Influence distance :", INFLUENCE_DISTANCE, "m")
print("Obstacle weight    :", OBSTACLE_WEIGHT)

print()
print("Obstacle cost      :", round(obstacle_cost, 4))

# ============================================================
# LATER USED BY NMPC
# ============================================================

nmpc_input = {
    "obstacle_x": obstacle_x,
    "obstacle_y": obstacle_y,
    "obstacle_distance": obstacle_distance,
    "safety_distance": SAFETY_DISTANCE,
    "obstacle_cost": obstacle_cost
}

print()
print("=" * 60)
print("NMPC INPUT")
print("=" * 60)

for key, value in nmpc_input.items():
    print(f"{key:20s}: {value}")

print()
print("=" * 60)
print("NMPC OBSTACLE COST TEST COMPLETE")
print("=" * 60)