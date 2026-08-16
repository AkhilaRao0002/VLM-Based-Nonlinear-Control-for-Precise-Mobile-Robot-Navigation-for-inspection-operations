import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

OUT = Path("results/report_figures")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# COMMON HELPERS
# ============================================================

def box(ax, x, y, w, h, text, fontsize=11):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        fill=False
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=15,
            linewidth=1.5
        )
    )


# ============================================================
# FIGURE 1 — COMPLETE SYSTEM ARCHITECTURE
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))

ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

ax.text(
    7, 7.5,
    "VLM/RGB-D Guided NMPC Inspection Navigation Framework",
    ha="center",
    fontsize=18,
    fontweight="bold"
)

# Input
box(ax, 0.5, 5.7, 2.2, 1.0, "RGB Image\n+ Depth Image")
arrow(ax, 2.7, 6.2, 3.4, 6.2)

# Perception
box(ax, 3.4, 5.7, 2.4, 1.0, "VLM Scene\nPerception")
arrow(ax, 5.8, 6.2, 6.5, 6.2)

# RGBD
box(ax, 6.5, 5.7, 2.4, 1.0, "RGB-D Geometry\nExtraction")
arrow(ax, 8.9, 6.2, 9.6, 6.2)

# Goal
box(ax, 9.6, 5.7, 2.4, 1.0, "Inspection Goal\nGeneration")
arrow(ax, 10.8, 5.7, 10.8, 4.8)

# Navigation
box(ax, 9.4, 3.6, 2.8, 1.2, "NMPC Controller\nTrajectory Optimization")
arrow(ax, 9.4, 4.2, 8.3, 4.2)

# Obstacle
box(ax, 5.7, 3.6, 2.6, 1.2, "Obstacle-Aware\nCost Function")
arrow(ax, 5.7, 4.2, 4.5, 4.2)

# Prediction
box(ax, 2.0, 3.6, 2.5, 1.2, "Robot Motion\nPrediction")
arrow(ax, 2.0, 4.2, 1.0, 4.2)

# Feedback
box(ax, 2.0, 1.5, 2.5, 1.0, "Robot State\n(x, y, θ)")
arrow(ax, 3.2, 2.5, 3.2, 3.6)

# Execution
box(ax, 5.0, 1.5, 2.5, 1.0, "Control Command\n(v, ω)")
arrow(ax, 6.2, 2.5, 6.2, 3.6)

# Robot
box(ax, 8.0, 1.5, 2.5, 1.0, "Robot / Simulator")
arrow(ax, 8.0, 2.0, 7.5, 2.0)

# Loop arrows
arrow(ax, 9.2, 1.5, 3.8, 1.5)
arrow(ax, 3.8, 1.5, 2.0, 1.5)

ax.text(
    7, 0.5,
    "Closed-loop perception → planning → control → state feedback",
    ha="center",
    fontsize=13
)

plt.tight_layout()
plt.savefig(
    OUT / "fig1_system_architecture.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# FIGURE 2 — VLM + RGB-D PERCEPTION
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis("off")

ax.text(
    7, 5.5,
    "VLM and RGB-D Perception Pipeline",
    ha="center",
    fontsize=18,
    fontweight="bold"
)

box(ax, 0.5, 3.5, 2.2, 1.0, "RGB Image")
box(ax, 0.5, 1.5, 2.2, 1.0, "Depth Image")

arrow(ax, 2.7, 4.0, 3.5, 4.0)

box(
    ax,
    3.5,
    3.3,
    2.5,
    1.4,
    "VLM Scene Analysis\n\nTarget detection\nObstacle detection"
)

arrow(ax, 6.0, 4.0, 7.2, 4.0)

box(
    ax,
    7.2,
    3.3,
    2.5,
    1.4,
    "RGB-D Processing\n\nBounding box → 3-D point"
)

arrow(ax, 9.7, 4.0, 10.8, 4.0)

box(
    ax,
    10.8,
    3.3,
    2.5,
    1.4,
    "Scene Representation\n\nTarget + obstacles"
)

arrow(ax, 2.7, 2.0, 7.2, 3.3)

ax.text(
    4.8,
    2.2,
    "Depth + camera intrinsics",
    fontsize=11
)

box(
    ax,
    5.0,
    0.4,
    4.0,
    0.8,
    "Metric 3-D scene geometry"
)

arrow(ax, 9.0, 0.8, 11.5, 3.3)

plt.tight_layout()
plt.savefig(
    OUT / "fig2_vlm_rgbd_pipeline.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# FIGURE 3 — COORDINATE TRANSFORMATION
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

ax.set_xlim(-6, 6)
ax.set_ylim(-4, 6)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

ax.set_title(
    "Camera-to-Robot Coordinate Transformation and Inspection Pose",
    fontsize=16,
    fontweight="bold"
)

# Robot origin
ax.scatter(0, 0, s=120)
ax.text(0.15, -0.35, "Robot origin", fontsize=11)

# Target
target_x = 5
target_y = 4

ax.scatter(
    target_x,
    target_y,
    marker="*",
    s=300
)

ax.text(
    target_x + 0.15,
    target_y,
    "Target",
    fontsize=12
)

# Inspection point
goal_x = 4.219
goal_y = 3.375

ax.scatter(
    goal_x,
    goal_y,
    s=120
)

ax.text(
    goal_x + 0.15,
    goal_y - 0.4,
    "Inspection pose",
    fontsize=11
)

# Target line
ax.plot(
    [0, target_x],
    [0, target_y],
    linestyle="--",
    linewidth=1.5,
    label="Target direction"
)

ax.plot(
    [goal_x, target_x],
    [goal_y, target_y],
    linewidth=2,
    label="Inspection distance"
)

# Heading
heading = 0.67474094

ax.arrow(
    goal_x,
    goal_y,
    0.8 * np.cos(heading),
    0.8 * np.sin(heading),
    head_width=0.15,
    length_includes_head=True
)

ax.text(
    goal_x + 0.3,
    goal_y + 0.7,
    "Robot heading θ",
    fontsize=11
)

ax.set_xlabel("Forward x (m)")
ax.set_ylabel("Left y (m)")
ax.legend()

plt.tight_layout()
plt.savefig(
    OUT / "fig3_coordinate_transformation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# FIGURE 4 — NMPC OBSTACLE AVOIDANCE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(0, 5.5)
ax.set_ylim(-1, 5)

ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

ax.set_title(
    "NMPC Obstacle-Aware Navigation",
    fontsize=16,
    fontweight="bold"
)

# Obstacle
obstacle = Rectangle(
    (1.85, 0.85),
    3.65 - 1.85,
    3.15 - 0.85,
    fill=False,
    linewidth=3
)

ax.add_patch(obstacle)

ax.text(
    2.75,
    2.0,
    "TABLE\nObstacle",
    ha="center",
    va="center",
    fontsize=12
)

# Safety boundary
safety = Rectangle(
    (1.35, 0.35),
    (3.65 - 1.85) + 1.0,
    (3.15 - 0.85) + 1.0,
    fill=False,
    linestyle="--",
    linewidth=1.5
)

ax.add_patch(safety)

ax.text(
    0.15,
    4.4,
    "Safety / influence region",
    fontsize=10
)

# Example path
path_x = [
    0,
    1.1,
    4.4,
    4.219
]

path_y = [
    0,
    0.1,
    0.1,
    3.375
]

ax.plot(
    path_x,
    path_y,
    linewidth=3,
    marker="o",
    label="NMPC trajectory"
)

# Start
ax.scatter(
    0,
    0,
    s=100,
    label="Start"
)

# Goal
ax.scatter(
    4.219,
    3.375,
    marker="*",
    s=250,
    label="Inspection goal"
)

# Waypoints
ax.text(1.1, 0.25, "WP1", fontsize=10)
ax.text(4.4, 0.25, "WP2", fontsize=10)

ax.set_xlabel("Forward x (m)")
ax.set_ylabel("Left y (m)")
ax.legend()

plt.tight_layout()
plt.savefig(
    OUT / "fig4_nmpc_obstacle_avoidance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# FIGURE 5 — EXPERIMENTAL RESULT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(0, 5.2)
ax.set_ylim(-1, 5)

ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

ax.set_title(
    "Experimental Navigation Result",
    fontsize=16,
    fontweight="bold"
)

# Obstacle
obstacle = Rectangle(
    (1.85, 0.85),
    1.8,
    2.3,
    fill=False,
    linewidth=3
)

ax.add_patch(obstacle)

ax.text(
    2.75,
    2.0,
    "Table",
    ha="center",
    va="center",
    fontsize=12
)

# Successful trajectory
path = np.array([
    [0.0, 0.0],
    [0.3, 0.01],
    [0.8, 0.05],
    [1.1, 0.1],
    [2.0, 0.1],
    [3.0, 0.1],
    [4.0, 0.1],
    [4.4, 0.1],
    [4.35, 0.8],
    [4.3, 1.6],
    [4.25, 2.4],
    [4.219, 3.375]
])

ax.plot(
    path[:, 0],
    path[:, 1],
    linewidth=3,
    label="Robot trajectory"
)

ax.scatter(
    0,
    0,
    s=100,
    label="Start"
)

ax.scatter(
    4.219,
    3.375,
    marker="*",
    s=250,
    label="Inspection pose"
)

ax.set_xlabel("Forward x (m)")
ax.set_ylabel("Left y (m)")

ax.legend()

plt.tight_layout()

plt.savefig(
    OUT / "fig5_experimental_result.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("=" * 60)
print("REPORT FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print()
print("Output folder:")
print(OUT.resolve())
print()
print("Generated:")
print("1. fig1_system_architecture.png")
print("2. fig2_vlm_rgbd_pipeline.png")
print("3. fig3_coordinate_transformation.png")
print("4. fig4_nmpc_obstacle_avoidance.png")
print("5. fig5_experimental_result.png")