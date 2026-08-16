import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 7))

ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")


def box(x, y, w, h, text, fontsize=11):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08",
        linewidth=1.8,
        edgecolor="black",
        facecolor="white"
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


def arrow(x1, y1, x2, y2, text=None):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.6
    )
    ax.add_patch(arr)

    if text:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.18,
            text,
            ha="center",
            va="center",
            fontsize=9
        )


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

ax.text(
    6,
    7.65,
    "Receding-Horizon NMPC Control Loop",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold"
)


# ---------------------------------------------------------
# Main blocks
# ---------------------------------------------------------

box(
    0.4, 5.5, 2.0, 1.0,
    "Current Robot State\n[x, y, θ]"
)

box(
    3.1, 5.5, 2.2, 1.0,
    "NMPC Prediction\nNonlinear Robot Model"
)

box(
    6.0, 5.5, 2.0, 1.0,
    "Cost Function\nEvaluation"
)

box(
    8.7, 5.5, 2.2, 1.0,
    "Optimization\nFind Optimal Controls"
)


# ---------------------------------------------------------
# Lower blocks
# ---------------------------------------------------------

box(
    8.7, 3.4, 2.2, 1.0,
    "Optimal Control\nSequence U*"
)

box(
    5.8, 3.4, 2.0, 1.0,
    "Apply First\nControl [v, ω]"
)

box(
    2.8, 3.4, 2.2, 1.0,
    "Robot Motion\nModel"
)

box(
    0.4, 3.4, 1.8, 1.0,
    "New Robot\nState"
)


# ---------------------------------------------------------
# Goal and obstacle inputs
# ---------------------------------------------------------

box(
    3.0, 1.2, 2.4, 0.9,
    "Inspection Goal\n[xg, yg, θg]"
)

box(
    6.2, 1.2, 2.4, 0.9,
    "Obstacle Map\n& Safety Constraints"
)


# ---------------------------------------------------------
# Main arrows
# ---------------------------------------------------------

arrow(2.4, 6.0, 3.1, 6.0)

arrow(5.3, 6.0, 6.0, 6.0)

arrow(8.0, 6.0, 8.7, 6.0)

arrow(9.8, 5.5, 9.8, 4.4)

arrow(8.7, 3.9, 7.8, 3.9)

arrow(5.8, 3.9, 5.0, 3.9)

arrow(2.8, 3.9, 2.2, 3.9)

# New state back to NMPC
arrow(
    1.3, 4.4,
    1.3, 6.8
)

arrow(
    1.3, 6.8,
    3.1, 6.8
)


# ---------------------------------------------------------
# Goal and obstacle connections
# ---------------------------------------------------------

arrow(
    4.2, 2.1,
    4.2, 5.5
)

arrow(
    7.4, 2.1,
    7.4, 5.5
)


# ---------------------------------------------------------
# Receding horizon annotation
# ---------------------------------------------------------

ax.text(
    6,
    0.45,
    "Only the first optimized control input is applied; "
    "the horizon then shifts forward and the optimization is repeated.",
    ha="center",
    va="center",
    fontsize=10,
    style="italic"
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

plt.tight_layout()

plt.savefig(
    "Figure_5_NMPC_Receding_Horizon.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "Figure_5_NMPC_Receding_Horizon.pdf",
    bbox_inches="tight"
)

plt.close()

print("Figure 5 generated successfully.")
print("Saved:")
print("  Figure_5_NMPC_Receding_Horizon.png")
print("  Figure_5_NMPC_Receding_Horizon.pdf")