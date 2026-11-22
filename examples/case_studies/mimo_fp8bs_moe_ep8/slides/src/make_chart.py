"""Speedup progression chart for the GEAK MiMo fp8 MoE slides.

Data: verified, committed geomean speedup vs. the aiter asm baseline, per optimization round,
for two GEAK runs (run 1 from scratch, run 2 warm-started from run 1's best).
Sources: tech_lead_report.md of each run, journal.jsonl verifier results, run 1 workspace git log.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "speedup_progression.png"

# Reference palette (dataviz palette.md): slot 1 blue; chrome + ink tokens.
SURFACE = "#ffffff"
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# x positions: run 1 at 0..3, gap, run 2 at 5..9.
run1_x = [0, 1, 2, 3]
run1_y = [1.000, 1.494, 1.540, 1.540]
# Run 2's original R1 (no verified result in time) is omitted and the rest renumbered R1-R4 for the slide
# (report numbering: shown R1-R4 = report R2-R5).
run2_x = [5, 6, 7, 8, 9]
run2_y = [1.528, 1.571, 1.627, 1.627, 1.627]
xticks = run1_x + run2_x
xlabels = ["Baseline", "R1", "R2", "R3", "Warm\nstart", "R1", "R2", "R3", "R4"]

fig, ax = plt.subplots(figsize=(10, 5.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# One series: committed best per round, one line segment per run (separate measurements).
line_kw = dict(color=BLUE, lw=2.2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
mark_kw = dict(marker="o", ms=8.5, mfc=BLUE, mec=SURFACE, mew=2)
ax.plot(run1_x, run1_y, **line_kw, **mark_kw)
ax.plot(run2_x, run2_y, **line_kw, **mark_kw)

# Recessive chrome: horizontal hairline grid only.
ax.set_ylim(0.9, 1.80)
ax.set_xlim(-0.6, 9.9)
ax.set_yticks([1.0, 1.2, 1.4, 1.6])
ax.set_yticklabels(["1.0×", "1.2×", "1.4×", "1.6×"], color=MUTED)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, color=INK2)
ax.tick_params(axis="both", length=0, pad=6)
ax.grid(axis="y", color=GRID, lw=0.8, ls="-", zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.spines["bottom"].set_linewidth(0.8)
ax.set_axisbelow(True)

# Run separator + run headers.
ax.axvline(4, color=AXIS, lw=0.8, zorder=1)
hdr = dict(ha="center", va="bottom", color=INK2, fontsize=10, fontweight="bold", transform=ax.get_xaxis_transform())
ax.text(1.5, 1.015, "Run 1 · from scratch · 3 rounds", **hdr)
ax.text(7.0, 1.015, "Run 2 · warm start · 4 rounds", **hdr)

# Speedup value on every point (requested), in ink - never the series color.
def val(x, y, dy=0.03):
    ax.text(x, y + dy, f"{y:.2f}×", ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold", zorder=6)

for x, y in zip(run1_x, run1_y):
    val(x, y)
for x, y in zip(run2_x, run2_y):
    val(x, y, dy=-0.075 if x == 6 else 0.03)  # run 2 R1 label below: its leader line comes from above

# Key-optimization annotations with thin leader lines (relpos = where on the text the leader starts).
def note(xy, xytext, s, ha="left", relpos=(0.5, 0.5)):
    ax.annotate(
        s, xy=xy, xytext=xytext, ha=ha, va="center", color=INK2, fontsize=8.8, zorder=6,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=6, relpos=relpos),
    )

def flat(x, y, s):
    ax.text(x, y, s, ha="center", va="top", color=MUTED, fontsize=8.3)

note((1, 1.494), (1.35, 1.22), "New FlyDSL native-fp8\n2-stage kernel (+49%)")
note((2, 1.540), (1.75, 1.37), "New stage-2 kernel (+3%)", relpos=(0, 0.5))
flat(3, 1.505, "no commit, gain < 2%")
note((5, 1.528), (4.35, 1.33), "Warm start from Run 1's best\n(same code, re-measured)")
note((6, 1.571), (6.15, 1.745), "Fused sort (6→4 launches)\n+ stage-2 scheduling (+3%)", relpos=(0, 0.5))
note((7, 1.627), (7.25, 1.43), "Scalar-FMA rescale\n+ padding-tile skip (+4%)", relpos=(0, 0.5))
flat(8.5, 1.595, "no commit, gain < 2%")

fig.suptitle("Verified, committed geomean speedup vs. aiter asm baseline", x=0.06, y=0.985, ha="left",
             fontsize=13, fontweight="bold", color=INK)
fig.text(0.06, 0.915, "MiMo fp8 block-scale MoE (EP8) on MI300X · 2048 / 4096 tokens per rank · "
         "balanced expert routing (max/mean load ≤ 1.07)", ha="left", fontsize=9.5, color=INK2)
fig.subplots_adjust(left=0.06, right=0.985, top=0.83, bottom=0.12)
fig.savefig(OUT, facecolor=SURFACE)
print("wrote", OUT)
