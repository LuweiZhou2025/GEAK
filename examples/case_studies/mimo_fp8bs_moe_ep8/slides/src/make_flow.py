"""PPT-style process diagram for GEAK run 1, round 1 (MiMo fp8 MoE slides)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "round1_process.png"

SURFACE = "#ffffff"
BLUE = "#2a78d6"
BLUE_WASH = "#eaf2fd"
BOX = "#f6f6f4"
BORDER = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GOOD = "#0ca30c"
BAD = "#d03b3b"

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(13.8, 5.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 13.8)
ax.set_ylim(0.05, 5.35)
ax.axis("off")


def box(x0, y0, x1, y1, fill=BOX, edge=BORDER, lw=1.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fill, ec=edge, lw=lw, zorder=2))


def lines(x, y, rows, ha="left"):
    """rows: list of (text, size, weight, color); drawn top-down from y."""
    for text, size, weight, color in rows:
        ax.text(x, y, text, ha=ha, va="top", fontsize=size, fontweight=weight, color=color, zorder=3)
        y -= size / 72 * 1.55 * (5.3 / 5.4)  # line advance in data units (1 data unit ~= 1 inch here)


def header(x, n, title, sub=None):
    ax.add_patch(Circle((x + 0.16, 5.0), 0.16, fc=BLUE, ec="none", zorder=3))
    ax.text(x + 0.16, 5.0, str(n), ha="center", va="center", fontsize=10.5, fontweight="bold", color="white", zorder=4)
    ax.text(x + 0.42, 5.0, title, ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK, zorder=3)
    if sub:
        ax.text(x + 0.42, 4.68, sub, ha="left", va="center", fontsize=9, color=MUTED, zorder=3)


def arrow(p, q, color=MUTED, lw=1.3, style="-|>", ms=12, cs="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=cs, zorder=1, shrinkA=0, shrinkB=0))


B = dict(size=10.5, weight="bold")
T = 9.6

# 1 Profile
header(0.2, 1, "Profile")
box(0.2, 1.5, 2.45, 3.7)
lines(0.38, 3.45, [
    ("aiter asm kernel", T, "bold", INK),
    ("low MFMA efficiency", T, "normal", INK2),
    ("", 4, "normal", INK2),
    ("1 wave/SIMD", T, "bold", INK),
    ("512 VGPRs per wave", T, "normal", INK2),
    ("", 4, "normal", INK2),
    ("32-row tiles", T, "bold", INK),
    ("reload weights", T, "normal", INK2),
])

# 2 Plan
header(2.8, 2, "Plan")
box(2.8, 1.95, 4.45, 3.25)
lines(2.97, 3.0, [("Tech lead issues", T, "normal", INK2), ("3 parallel", T, "bold", INK), ("directions", T, "bold", INK)])

# Lanes: y centers for algorithm / host-runtime / compute
LANES = [3.75, 2.6, 1.45]
H = 0.9
IMPL = (4.9, 7.75)
VER = (8.15, 10.35)
impl_text = [
    ("Algorithm", "New FlyDSL fp8 2-stage kernel", "1.28× → 1.50×"),
    ("Host-runtime", "Tune the aiter asm path", "1.05×"),
    ("Compute", "Alternate FlyDSL stage-1", "1.51× (self-reported)"),
]
ver_text = [("1.49×", "best of round", True), ("1.05×", "conflicts with new path", False),
            ("No gain", "1.00× when re-measured", False)]

header(IMPL[0], 3, "Implement", "own workspace · shared 4-GPU pool")
header(VER[0], 4, "Verify", "each patch alone vs. baseline")

for i, (yc, (name, what, res), (vnum, vnote, ok)) in enumerate(zip(LANES, impl_text, ver_text)):
    win = i == 0
    fill, edge, lw = (BLUE_WASH, BLUE, 1.4) if win else (BOX, BORDER, 1.0)
    # implement box
    box(IMPL[0], yc - H / 2, IMPL[1], yc + H / 2, fill, edge, lw)
    ax.text(IMPL[0] + 0.17, yc + 0.22, name, ha="left", va="center", fontsize=10.5, fontweight="bold", color=INK, zorder=3)
    ax.text(IMPL[0] + 0.17, yc - 0.02, what, ha="left", va="center", fontsize=T, color=INK2, zorder=3)
    ax.text(IMPL[0] + 0.17, yc - 0.26, res, ha="left", va="center", fontsize=T, fontweight="bold", color=INK, zorder=3)
    # verify box
    box(VER[0], yc - H / 2, VER[1], yc + H / 2, fill, edge, lw)
    ax.text(VER[0] + 0.17, yc + 0.12, vnum, ha="left", va="center", fontsize=12, fontweight="bold", color=INK, zorder=3)
    ax.text(VER[0] + 0.17, yc - 0.2, vnote, ha="left", va="center", fontsize=8.8, color=INK2, zorder=3)
    ax.text(VER[1] - 0.2, yc + 0.12, "✔" if ok else "✘", ha="center", va="center", fontsize=15,
            fontweight="bold", color=GOOD if ok else BAD, zorder=3)
    # arrows: plan -> implement (elbow fan-out from a trunk), implement -> verify
    arrow((4.67, yc), (IMPL[0] - 0.02, yc))
    arrow((IMPL[1], yc), (VER[0] - 0.02, yc), color=BLUE if win else MUTED, lw=1.6 if win else 1.3)

# fan-out trunk: plan right edge -> vertical bus at x=4.67
ax.plot([4.45, 4.67], [2.6, 2.6], color=MUTED, lw=1.3, zorder=1)
ax.plot([4.67, 4.67], [LANES[-1], LANES[0]], color=MUTED, lw=1.3, zorder=1, solid_capstyle="round")

# 5 Commit & re-profile (only the winner flows in)
header(10.75, 5, "Commit & re-profile")
box(10.75, 1.95, 13.15, 4.3, BLUE_WASH, BLUE, 1.4)
lines(10.92, 4.05, [
    ("Commit algorithm patch", T, "bold", INK),
    ("1.49× verified", T, "normal", INK2),
    ("", 4, "normal", INK2),
    ("Re-profile", T, "bold", INK),
    ("now compute-bound", T, "normal", INK2),
    ("", 4, "normal", INK2),
    ("Power cap found", T, "bold", INK),
    ("clock ~1.1–1.4 GHz", T, "normal", INK2),
    ("(2.1 GHz max)", T, "normal", INK2),
])
arrow((VER[1], LANES[0]), (10.73, LANES[0]), color=BLUE, lw=1.6)
arrow((2.45, 2.6), (2.78, 2.6))  # profile -> plan

# Loop back: commit -> next round's plan
ax.plot([11.95, 11.95, 3.62], [1.95, 0.5, 0.5], color=MUTED, lw=1.3, zorder=1, solid_capstyle="round")
arrow((3.62, 0.5), (3.62, 1.93))
ax.text(7.8, 0.33, "next round re-plans on the new bottleneck", ha="center", va="center", fontsize=9, color=MUTED)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT, facecolor=SURFACE)
print("wrote", OUT)
