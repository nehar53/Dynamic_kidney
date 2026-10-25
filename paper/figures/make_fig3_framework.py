"""Fig. 3: one round of the framework (vector PDF + 600 dpi PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "mathtext.fontset": "dejavusans"})
FILL, EDGE = "#dbe7f5", "#1f3b63"
fig, ax = plt.subplots(figsize=(7.16, 3.55))
ax.set_xlim(-1.7, 17.2); ax.set_ylim(-8.0, 1.2); ax.axis("off")
B = {}


def box(key, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.25",
                                fc=FILL, ec=EDGE, lw=0.9, zorder=2))
    ax.text(x, y, text, ha="center", va="center", zorder=3, linespacing=1.25)
    B[key] = (x, y, w, h)


def diamond(key, x, y, w, h, text):
    ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)],
                         closed=True, fc=FILL, ec=EDGE, lw=0.9, zorder=2))
    ax.text(x, y, text, ha="center", va="center", zorder=3, linespacing=1.2)
    B[key] = (x, y, w, h)


def side(key, s, off=0.0):
    x, y, w, h = B[key]
    return {"n": (x + off, y + h / 2), "s": (x + off, y - h / 2),
            "e": (x + w / 2, y + off), "w": (x - w / 2, y + off)}[s]


def arrow(p, q, dashed=False, rad=0.0):
    ax.annotate("", xy=q, xytext=p, zorder=4,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", lw=0.9, color="black",
                                ls="--" if dashed else "-", shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


def line(pts, dashed=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="black", lw=0.9, ls="--" if dashed else "-", zorder=4, solid_capstyle="butt")


def lab(x, y, s, **k):
    ax.text(x, y, s, fontsize=6.5, zorder=5, **k)


# top row
box("pool", 0.4, 0, 2.6, 1.3, "Pool $V_t$\npairs and altruists")
diamond("clear", 3.7, 0, 2.6, 1.9, "Clear in\nround $t$?")
box("build", 7.6, 0, 3.9, 1.5, "Build compatibility graph\n(ABO, crossmatch, CIT);\nenumerate cycles and chains")
box("ilp", 11.9, 0, 2.7, 1.3, "ILP: maximum\ntransplants")
diamond("small", 15.5, 0, 3.0, 1.9, "At most 200\nmatchings?")
# bottom row
box("upd", 0.4, -4.3, 2.9, 1.6, "Pool update: match,\nexit, wait,\nwithdraw, arrive")
box("sel", 4.0, -4.3, 3.4, 1.6, "Select: at least 90% of\nmax transplants; closest\nto ideal in $(F_2, F_3, F_4)$;\nrecord PoF")
box("front", 7.75, -4.3, 2.8, 1.6, "Pareto front\n(archive; contains\nILP solution)")
box("nsga", 11.9, -4.3, 3.3, 1.5, "NSGA-III: crossover,\nbit-flip mutation,\nrepair, selection")
box("band", 11.9, -6.6, 2.8, 1.2, "Bandit\n(UCB1 / LinUCB)")
box("exact", 15.5, -5.3, 2.6, 1.3, "Exact Pareto front\n(enumeration)")
# AOS dashed region
ax.add_patch(FancyBboxPatch((10.1, -7.65), 3.9, 4.25, boxstyle="round,pad=0,rounding_size=0.35",
                            fc="none", ec="black", lw=0.8, ls="--", zorder=1))
lab(11.9, -7.55, "AOS within the round", ha="center", va="bottom")

# top-row flow
arrow(side("pool", "e"), side("clear", "w"))
arrow(side("clear", "e"), side("build", "w")); lab(5.35, 0.12, "yes", ha="center", va="bottom")
arrow(side("build", "e"), side("ilp", "w"))
arrow(side("ilp", "e"), side("small", "w"))
arrow(side("clear", "s"), side("upd", "n", 0.6)); lab(2.35, -2.1, "no", ha="right")
arrow((14.75, -0.48), side("nsga", "n", 1.2)); lab(14.0, -2.5, "no", ha="left")
arrow(side("small", "s"), side("exact", "n")); lab(15.62, -2.2, "yes", ha="left")
arrow(side("ilp", "s"), side("nsga", "n"), dashed=True)
lab(11.78, -1.6, "seed and\nrepair priority", ha="right", va="center")

# bottom-row flow
arrow(side("nsga", "w"), side("front", "e")); lab(9.65, -4.18, "archive", ha="center", va="bottom")
arrow(side("front", "n"), side("nsga", "n", -1.0), dashed=True, rad=-0.45)
lab(9.15, -2.55, "warm start, next round", ha="center", va="bottom")
arrow(side("nsga", "s", -0.5), side("band", "n", -0.5)); lab(11.3, -5.45, "reward\n(HV gain)", ha="right", va="center")
arrow(side("band", "n", 0.5), side("nsga", "s", 0.5)); lab(12.5, -5.45, "mutation\nrate", ha="left", va="center")
arrow((8.6, -6.6), side("band", "w")); lab(9.2, -6.5, "pool state\n(LinUCB)", ha="center", va="bottom")
line([side("exact", "s"), (15.5, -7.85), (7.75, -7.85)])
arrow((7.75, -7.85), side("front", "s")); lab(11.9, -7.95, "exact front", ha="center", va="top")
arrow(side("front", "w"), side("sel", "e"))
arrow(side("sel", "w"), side("upd", "e")); lab(2.1, -4.18, "$M_t$", ha="center", va="bottom")
line([side("upd", "w"), (-1.35, -4.3), (-1.35, 0)])
arrow((-1.35, 0), side("pool", "w")); lab(-1.45, -2.1, "$t+1$", ha="right", va="center")

fig.tight_layout(pad=0.1)
for ext in ("pdf", "png"):
    fig.savefig(f"fig3_framework.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved fig3_framework.pdf / .png")
