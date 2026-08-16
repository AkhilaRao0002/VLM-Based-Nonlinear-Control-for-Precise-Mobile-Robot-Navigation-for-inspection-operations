import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# FIGURE 7
# Complete End-to-End Perception-to-Action Pipeline
# ============================================================

OUTPUT_PNG = "Figure_7_End_to_End_Pipeline.png"
OUTPUT_PDF = "Figure_7_End_to_End_Pipeline.pdf"


fig, ax = plt.subplots(figsize=(11, 15))

ax.set_xlim(0, 11)
ax.set_ylim(0, 17)
ax.axis("off")


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def add_box(
    x,
    y,
    width,
    height,
    text,
    fontsize=11,
    linewidth=1.8
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.08",
        linewidth=linewidth,
        edgecolor="black",
        facecolor="white"
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True
    )


def add_arrow(
    x1,
    y1,
    x2,
    y2,
    linewidth=1.6
):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=linewidth,
        color="black"
    )

    ax.add_patch(arrow)


# ============================================================
# TITLE
# ============================================================

ax.text(
    5.5,
    16.5,
    "Complete VLM-Based Mobile Robot Inspection Pipeline",
    ha="center",
    va="center",
    fontsize=17,
    fontweight="bold"
)


ax.text(
    5.5,
    16.05,
    "Perception → Geometry → Planning → NMPC Control → Evaluation",
    ha="center",
    va="center",
    fontsize=11
)


# ============================================================
# STAGE 1 — INPUT
# ============================================================

add_box(
    3.0,
    14.7,
    5.0,
    0.9,
    "RGB Image + Aligned Depth Image",
    fontsize=12
)

ax.text(
    1.4,
    15.15,
    "INPUT",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold"
)


# Arrow
add_arrow(
    5.5,
    14.7,
    5.5,
    14.0
)


# ============================================================
# STAGE 2 — VLM PERCEPTION
# ============================================================

add_box(
    3.0,
    13.0,
    5.0,
    1.0,
    "VLM Scene Understanding\n"
    "Target + Relevant Obstacles + Scene Information",
    fontsize=11
)

ax.text(
    1.4,
    13.5,
    "PERCEPTION",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold"
)


add_arrow(
    5.5,
    13.0,
    5.5,
    12.3
)


# ============================================================
# STAGE 3 — 2D DETECTION
# ============================================================

add_box(
    3.0,
    11.3,
    5.0,
    1.0,
    "2-D Semantic Information\n"
    "Target Bounding Box + Obstacle Bounding Boxes",
    fontsize=11
)

add_arrow(
    5.5,
    11.3,
    5.5,
    10.6
)


# ============================================================
# STAGE 4 — RGB-D GEOMETRY
# ============================================================

add_box(
    3.0,
    9.6,
    5.0,
    1.0,
    "RGB-D Geometric Reconstruction\n"
    "Robust Depth Estimation + Pixel-to-3-D Conversion",
    fontsize=11
)

ax.text(
    1.4,
    10.1,
    "GEOMETRY",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold"
)

add_arrow(
    5.5,
    9.6,
    5.5,
    8.9
)


# ============================================================
# STAGE 5 — COORDINATE TRANSFORMATION
# ============================================================

add_box(
    3.0,
    7.9,
    5.0,
    1.0,
    "Camera-to-Robot Coordinate Transformation\n"
    "3-D Target and Obstacle Geometry in Robot Frame",
    fontsize=11
)

add_arrow(
    5.5,
    7.9,
    5.5,
    7.2
)


# ============================================================
# STAGE 6 — INSPECTION POSE
# ============================================================

add_box(
    3.0,
    6.2,
    5.0,
    1.0,
    "Inspection Pose Generation\n"
    "Preferred Distance + Target-Facing Heading",
    fontsize=11
)

ax.text(
    1.4,
    6.7,
    "PLANNING",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold"
)

add_arrow(
    5.5,
    6.2,
    5.5,
    5.5
)


# ============================================================
# STAGE 7 — OBSTACLE MAP
# ============================================================

add_box(
    3.0,
    4.5,
    5.0,
    1.0,
    "Metric Obstacle Map + Safety Constraints\n"
    "Safety Distance + Influence Distance",
    fontsize=11
)

add_arrow(
    5.5,
    4.5,
    5.5,
    3.8
)


# ============================================================
# STAGE 8 — WAYPOINT PLANNING
# ============================================================

add_box(
    3.0,
    2.8,
    5.0,
    1.0,
    "Safe Waypoint Planning\n"
    "Obstacle-Aware Intermediate Navigation Goals",
    fontsize=11
)

add_arrow(
    5.5,
    2.8,
    5.5,
    2.1
)


# ============================================================
# STAGE 9 — NMPC
# ============================================================

add_box(
    2.2,
    1.0,
    6.6,
    1.1,
    "NONLINEAR MODEL PREDICTIVE CONTROL (NMPC)\n"
    "Prediction → Cost Optimization → Optimal [v, ω]",
    fontsize=12,
    linewidth=2.2
)


# ============================================================
# SIDE INFORMATION TO NMPC
# ============================================================

add_box(
    8.6,
    9.6,
    2.0,
    1.0,
    "Robot State\n"
    "[x, y, θ]",
    fontsize=10
)

add_box(
    8.6,
    7.9,
    2.0,
    1.0,
    "Goal Pose\n"
    "[xg, yg, θg]",
    fontsize=10
)

add_box(
    8.6,
    6.2,
    2.0,
    1.0,
    "Obstacle\n"
    "Constraints",
    fontsize=10
)


# Side arrows toward NMPC region
add_arrow(
    8.6,
    10.0,
    7.8,
    2.0
)

add_arrow(
    8.6,
    8.3,
    7.8,
    2.0
)

add_arrow(
    8.6,
    6.6,
    7.8,
    2.0
)


# ============================================================
# CONTROL OUTPUT
# ============================================================

add_box(
    2.2,
    -0.7,
    3.0,
    0.8,
    "Robot Motion\n"
    "Linear v + Angular ω",
    fontsize=10
)

add_box(
    5.8,
    -0.7,
    3.0,
    0.8,
    "Closed-Loop Evaluation\n"
    "Goal + Safety + Performance",
    fontsize=10
)


# Arrows from NMPC
add_arrow(
    3.8,
    1.0,
    3.8,
    0.1
)

add_arrow(
    6.8,
    1.0,
    7.0,
    0.1
)


# ============================================================
# FEEDBACK LOOP
# ============================================================

# Feedback arrow from robot motion back toward NMPC
feedback = FancyArrowPatch(
    (2.2, -0.3),
    (1.0, 1.6),
    arrowstyle="->",
    mutation_scale=15,
    linewidth=1.6,
    color="black",
    connectionstyle="arc3,rad=0.35"
)

ax.add_patch(feedback)

ax.text(
    0.75,
    0.75,
    "State feedback\n"
    "and re-optimization",
    ha="center",
    va="center",
    fontsize=9,
    rotation=90
)


# ============================================================
# END-TO-END LABELS
# ============================================================

ax.text(
    9.5,
    15.0,
    "SEMANTIC",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold"
)

ax.text(
    9.5,
    10.0,
    "METRIC",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold"
)

ax.text(
    9.5,
    5.0,
    "PLANNING",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold"
)

ax.text(
    9.5,
    2.0,
    "CONTROL",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold"
)


# ============================================================
# FOOTNOTE
# ============================================================

ax.text(
    5.5,
    -1.45,
    "The system forms a closed perception-to-action loop in which "
    "visual information is converted into metric navigation goals "
    "and continuously optimized robot motion.",
    ha="center",
    va="center",
    fontsize=9,
    style="italic"
)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)

plt.close()


print("=" * 60)
print("FIGURE 7 GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"PNG : {OUTPUT_PNG}")
print(f"PDF : {OUTPUT_PDF}")
print("=" * 60)