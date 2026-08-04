"""Overview figure: the measured heterostructure and the logic of the Letter.

WHAT THIS FIGURE IS AND IS NOT
------------------------------
Panel (a) draws the heterostructure that all three experiments were performed
on.  It is NOT a device of ours; no device is proposed or fabricated in this
work.  Every layer, thickness, composition and polarity is taken from the
description given by Chang et al., and the drawing follows that specification
rather than inventing anything.  The quantities marked as computed, that is the
confining field, the distribution centroid and the strain, come from the
calculation reported in this Letter.  The vertical scale is broken because the
buffer is five hundred nanometres while the gas itself is half a nanometre
wide, and a linear drawing would show nothing.

Panel (b) carries our results: what each probe returns, where the three
disagree, and which of the disagreements this Letter resolves.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SERIF = "Nimbus Roman"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [SERIF, "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": SERIF, "mathtext.it": f"{SERIF}:italic",
    "mathtext.bf": f"{SERIF}:bold",
    "font.size": 8.5,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "savefig.dpi": 1000, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

C_H = "#1f4e79"
C_L = "#c1121f"
C_G = "#5a5a5a"
C_T = "#2a7f62"

DX, DY = 0.62, 0.34          # isometric depth vector


def shade(col, f):
    r, g, b = mcolors.to_rgb(col)
    if f >= 1.0:
        return (r + (1 - r) * (f - 1), g + (1 - g) * (f - 1),
                b + (1 - b) * (f - 1))
    return (r * f, g * f, b * f)


def slab(ax, y0, h, w, face, z=2, lw=0.45, edge="#3a3a3a"):
    """One layer as a shaded isometric slab, front-left corner at (0, y0)."""
    top = shade(face, 1.22)
    side = shade(face, 0.74)
    quads = [
        ([(0, y0 + h), (w, y0 + h), (w + DX, y0 + h + DY), (DX, y0 + h + DY)],
         top, z + 1),
        ([(w, y0), (w + DX, y0 + DY), (w + DX, y0 + h + DY), (w, y0 + h)],
         side, z + 1),
        ([(0, y0), (w, y0), (w, y0 + h), (0, y0 + h)], face, z),
    ]
    for pts, col, zz in quads:
        ax.add_patch(Polygon(pts, closed=True, facecolor=col, edgecolor=edge,
                             lw=lw, zorder=zz, joinstyle="miter"))


def figure():
    fig = plt.figure(figsize=(7.1, 3.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.14, 1.0], wspace=0.03)

    # ==================================================================
    # (a) the heterostructure
    #
    # The layout is banded so that nothing can overlap: the stack occupies
    # 0 < x < W+DX, the leader labels x > W+DX+0.30, and the expanded view
    # of the buried interface x < -0.35.  Everything inside that expanded
    # view is placed in its own axes in fractional coordinates, in bands
    # that are separated by construction.
    # ==================================================================
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-3.75, 7.80)
    ax.set_ylim(-1.00, 6.15)
    ax.axis("off")
    ax.set_title("(a)", loc="left", fontsize=9, x=-0.02)

    W = 3.1
    AlN = "#b6bcc4"
    AlGaN = "#7f8b99"
    GaN = "#efc98a"
    GaNMg = "#d98f3d"

    y = 0.0
    slab(ax, y, 1.15, W, shade(AlN, 0.88)); y_sub = y; y += 1.15
    slab(ax, y, 0.80, W, AlN); y_buf = y; y += 0.80
    y_sl0 = y
    per = 0.215
    for i in range(10):
        slab(ax, y, per * 0.72, W, AlN, z=2 + 3 * i)
        slab(ax, y + per * 0.72, per * 0.28, W, AlGaN, z=3 + 3 * i)
        y += per
    y_sl1 = y
    slab(ax, y, 1.30, W, GaN, z=40); y_gan = y; y += 1.30
    slab(ax, y, 0.46, W, GaNMg, z=44); y_cap = y; y += 0.46

    # broken-scale symbol across the substrate
    yb = y_sub + 0.50
    xs = np.linspace(0, W, 13)
    zig = 0.055 * np.array([1, -1] * 7)[:13]
    ax.fill_between(xs, yb + zig - 0.075, yb + zig + 0.075, color="white",
                    lw=0, zorder=30)
    ax.plot(xs, yb + zig + 0.075, color="#3a3a3a", lw=0.45, zorder=31)
    ax.plot(xs, yb + zig - 0.075, color="#3a3a3a", lw=0.45, zorder=31)

    ax.plot([0, W], [y_gan, y_gan], color=C_L, lw=2.4, zorder=90,
            solid_capstyle="butt")
    ax.plot([W, W + DX], [y_gan, y_gan + DY], color=C_L, lw=2.4, zorder=90,
            solid_capstyle="butt")

    lx = W + DX + 0.30

    def lab(yc, txt, fs=6.0, col="black"):
        ax.annotate(txt, xy=(W + DX, yc + DY / 2), xytext=(lx, yc + DY / 2),
                    fontsize=fs, va="center", ha="left", color=col,
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=C_G))

    lab(y_sub + 0.58, "bulk AlN substrate, Al-polar\n"
                      "[0001]; dislocations\n"
                      "$<10^{4}$ cm$^{-2}$")
    lab(y_buf + 0.40, "AlN buffer, 500 nm")
    lab((y_sl0 + y_sl1) / 2, "10 $\\times$ [Al$_{0.95}$Ga$_{0.05}$N,\n"
                            "2-3 ML / AlN spacer, 25 nm]")
    lab(y_gan, "2DHG, $p_{\\mathrm{s}}=4.6\\times10^{13}$ cm$^{-2}$",
        col=C_L)
    lab(y_gan + 0.80, "GaN, 15 nm\n"
                      "$\\varepsilon_\\perp=-2.42$ %,"
                      " $\\varepsilon_{zz}=+1.29$ %")
    lab(y_cap + 0.23, "GaN:Mg cap, 5 nm")
    ax.text(W / 2 + DX / 2, -0.55, "vertical scale broken", fontsize=6.0,
            color=C_G, ha="center", va="top")

    # ---- expanded view of the buried interface, to the left ------------
    d = json.load(open(os.path.join(RES, "well.json")))
    zz = np.array(d["z"])
    pp = np.array(d["p_of_z"])

    bx0, bx1 = -3.60, -0.35
    bh, f_int = 6.60, 0.72
    by0 = y_gan - f_int * bh

    cal = ax.inset_axes([bx0, by0, bx1 - bx0, bh], transform=ax.transData,
                        zorder=96)
    cal.set_xlim(0, 1)
    cal.set_ylim(0, 1)
    cal.set_xticks([])
    cal.set_yticks([])
    cal.set_facecolor("white")
    for sp in cal.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color(C_G)

    ax.plot([bx1, -0.01], [y_gan, y_gan], color=C_G, lw=0.5,
            ls=(0, (2.5, 1.8)), zorder=95)

    cal.text(0.5, 0.992, "GaN/AlN interface,\natomically sharp",
             fontsize=5.6, ha="center", va="top", color="black")

    # band 1: the two charge sheets, the interface line at f_int by design
    for xx in np.linspace(0.16, 0.56, 6):
        cal.text(xx, 0.775, "$\\oplus$", fontsize=6.9, color=C_L, ha="center",
                 va="center")
    cal.plot([0.08, 0.64], [0.720, 0.720], color="#2a2a2a", lw=0.9,
             solid_capstyle="butt")
    for xx in np.linspace(0.16, 0.56, 6):
        cal.text(xx, 0.683, "$\\ominus$", fontsize=6.9, color=C_H, ha="center",
                 va="center")
    cal.add_patch(FancyArrowPatch((0.775, 0.782), (0.775, 0.700),
                                  arrowstyle="-|>", lw=0.7, color=C_T,
                                  mutation_scale=6))

    # band 2: the computed confining field, on its own line
    cal.text(0.5, 0.632, "$E=8.0$ MV cm$^{-1}$", fontsize=6.1, color=C_T,
             ha="center", va="center")
    cal.text(0.5, 0.596, "fixed polarisation charge,\n"
                         "balanced by the hole gas",
             fontsize=5.7, ha="center", va="top", color="black")
    cal.text(0.5, 0.512, "computed hole density\n"
                         "$\\langle z\\rangle=0.57$ nm\n"
                         "rms width $0.36$ nm",
             fontsize=5.7, ha="center", va="top", color=C_L)

    # band 3: the computed distribution
    ins = cal.inset_axes([0.20, 0.115, 0.72, 0.275])
    ins.fill_between(zz, 0, pp, color=C_L, alpha=0.28, lw=0)
    ins.plot(zz, pp, color=C_L, lw=1.0)
    ins.set_xlim(0, 2.2)
    ins.set_ylim(0, 1.12 * pp.max())
    ins.set_yticks([])
    ins.set_xticks([0, 1, 2])
    ins.tick_params(labelsize=5.6, length=1.8, colors="black", direction="in",
                    pad=1.2)
    ins.set_xlabel("$z$ from interface (nm)", fontsize=5.6, labelpad=0.8)
    for sp in ins.spines.values():
        sp.set_linewidth(0.45)
    ins.axvline(0.568, color=C_L, lw=0.6, ls=":")

    # ==================================================================
    # (b) what each probe returns, and what disagrees
    # ==================================================================
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.9)
    ax.axis("off")
    ax.set_title("(b)", loc="left", fontsize=9, x=-0.01)

    def box(x, y, w, h, title, body, edge, fs=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.10,rounding_size=0.14",
                                    linewidth=0.8, edgecolor=edge,
                                    facecolor="white", zorder=2))
        ax.text(x + w / 2, y + h - 0.20, title, fontsize=6.2, ha="center",
                va="top", color=edge, zorder=3)
        ax.text(x + w / 2, y + h - 0.62, body, fontsize=fs, ha="center",
                va="top", color="black", zorder=3, linespacing=1.28)

    box(0.15, 5.90, 3.0, 1.72, "quantum oscillations",
        "to 72 T\n$m=1.92,\\ 0.53\\,m_0$\n$p_{\\mathrm{s}}$ per subband", C_H)
    box(3.50, 5.90, 3.0, 1.72, "cyclotron resonance",
        "to 31 T\n$m=2.6,\\ 0.57\\,m_0$\n$\\tau$ per subband", C_L)
    box(6.85, 5.90, 3.0, 1.72, "two-carrier Hall",
        "to 9 T\n$\\mu=1900,\\ 400$\ncm$^2$V$^{-1}$s$^{-1}$", C_T)

    dis = [
        (0.15, 3.60, 4.55, 1.75, "heavy-hole mass disagrees",
         "$1.92$ vs $2.6\\,m_0$", C_H,
         "resolved: $\\omega_{\\mathrm{c}}\\tau=0.82$ at 31 T,\n"
         "so the resonance is overdamped"),
        (5.30, 3.60, 4.55, 1.75, "light-hole mass exceeds theory",
         "$0.53$ vs $0.25$ to $0.30\\,m_0$", C_L,
         "resolved: $0.53$ is a field average;\n"
         "at $B\\rightarrow0$ theory and data agree"),
        (0.15, 1.35, 4.55, 1.75, "subband occupations disagree",
         "light: $0.45$ vs $0.80\\times10^{13}$ cm$^{-2}$", C_T,
         "open; a Rashba splitting of\n1.5 meV is predicted, not assumed"),
        (5.30, 1.35, 4.55, 1.75, "lifetime ratios not reproduced",
         "$\\tau_{\\mathrm{tr}}/\\tau_{\\mathrm{q}}$: $3.82$ vs $2.13$", C_G,
         "open; no elastic mechanism fits\nboth, which questions that fit"),
    ]
    for (x, y, w, h, title, sub, col, res) in dis:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.10,rounding_size=0.14",
                                    linewidth=0.8, edgecolor=col,
                                    facecolor="#fbfbfb", zorder=2))
        ax.text(x + w / 2, y + h - 0.20, title, fontsize=6.3, ha="center",
                va="top", color=col, zorder=3)
        ax.text(x + w / 2, y + h - 0.66, sub, fontsize=6.0, ha="center",
                va="top", color="black", zorder=3)
        ax.text(x + w / 2, y + h - 1.10, res, fontsize=5.5, ha="center",
                va="top", color=C_G, zorder=3, linespacing=1.26,
                style="italic")

    for x0, x1 in ((1.65, 2.4), (5.00, 5.4), (8.35, 7.6)):
        ax.add_patch(FancyArrowPatch((x0, 5.85), (x1, 5.42),
                                     arrowstyle="->", lw=0.7, color=C_G,
                                     mutation_scale=7, zorder=1))
    ax.text(5.0, 0.82, "this work: self-consistent six-band envelope functions\n"
                       "and two-dimensional scattering",
            fontsize=6.2, ha="center", va="top", color="black",
            linespacing=1.3)

    fig.savefig(os.path.join(FIG, "prb_fig0.png"))
    fig.savefig(os.path.join(FIG, "prb_fig0.pdf"))
    plt.close(fig)
    print("prb_fig0 written")


if __name__ == "__main__":
    figure()
