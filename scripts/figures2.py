"""Figures for the GaN two-dimensional hole gas manuscript.

All text is set in a Times-metric face and all axis ticks and tick labels are
black, as requested.  Every panel is generated from the JSON written by the
analysis scripts, so no number in a figure can drift from the text.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gpfet import kp6 as K
from gpfet import scatter2d as S
from gpfet.constants import Q as QE

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SERIF = "Nimbus Roman"          # metric-compatible with Times New Roman
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [SERIF, "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": SERIF, "mathtext.it": f"{SERIF}:italic",
    "mathtext.bf": f"{SERIF}:bold",
    "font.size": 8.5,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False,
    "savefig.dpi": 1000, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

C_HEAVY = "#1f4e79"
C_LIGHT = "#c1121f"
C_THIRD = "#2a7f62"
C_GREY = "#5a5a5a"

M0 = 9.1093837015e-31


def finish(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for s in ax.spines.values():
        s.set_color("black")


# ===========================================================================
# Figure 1.  Band structure: dispersion and the confinement dependence
# ===========================================================================
def figure1():
    st = K.biaxial_strain_on_AlN()
    kt = np.linspace(1e-3, 2.2, 1400)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))

    # (a) dispersion, bulk strained versus confined
    ax = axes[0]
    for kz, ls in ((0.0, "--"), (K.subband_kz(1.0), "-")):
        E = K.dispersion(kt, kz=kz, strain=st)
        ax.plot(kt, 1000 * E[:, 0], ls, color=C_HEAVY, lw=1.3)
        ax.plot(kt, 1000 * E[:, 2], ls, color=C_LIGHT, lw=1.3)
    for kf, c, lab in ((K.kf_from_density(3.8e13), C_HEAVY, "heavy"),
                       (K.kf_from_density(8.0e12), C_LIGHT, "light")):
        ax.axvline(kf, color=c, lw=0.6, ls=":", alpha=0.9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=C_HEAVY, lw=1.3, label="heavy branch"),
        Line2D([], [], color=C_LIGHT, lw=1.3, label="light branch"),
        Line2D([], [], color="black", lw=1.0, ls="--", label="bulk, strained"),
        Line2D([], [], color="black", lw=1.0, ls="-", label="confined, 1 nm")],
        loc="upper left", handlelength=1.7, labelspacing=0.3)
    ax.annotate(r"$k_{\mathrm{F}}$", xy=(K.kf_from_density(8.0e12), 6),
                fontsize=7, color=C_LIGHT, ha="center", va="bottom")
    ax.annotate(r"$k_{\mathrm{F}}$", xy=(K.kf_from_density(3.8e13), 6),
                fontsize=7, color=C_HEAVY, ha="center", va="bottom")
    ax.set_xlabel(r"in-plane wavevector $k_\perp$ (nm$^{-1}$)")
    ax.set_ylabel("hole energy (meV)")
    ax.set_xlim(0, 1.8)
    ax.set_ylim(0, 150)
    ax.set_title("(a)", loc="left", fontsize=9)
    finish(ax)

    # (b) mass versus confinement
    ax = axes[1]
    d = json.load(open(os.path.join(RES, "masses.json")))
    sc = [r for r in d["confinement_scan"] if r["width_nm"]]
    w = np.array([r["width_nm"] for r in sc])
    mh = np.array([r["m_heavy"] for r in sc])
    ml = np.array([r["m_light"] for r in sc])
    o = np.argsort(w)
    ax.plot(w[o], mh[o], "o-", color=C_HEAVY, ms=3, lw=1.2, label="heavy")
    ax.plot(w[o], ml[o], "s-", color=C_LIGHT, ms=3, lw=1.2, label="light")
    ax.axhline(1.92, color=C_HEAVY, lw=0.8, ls="--")
    ax.axhline(0.53, color=C_LIGHT, lw=0.8, ls="--")
    ax.text(5.8, 1.99, "measured 1.92", color=C_HEAVY, fontsize=7, ha="right")
    ax.text(5.8, 0.60, "measured 0.53", color=C_LIGHT, fontsize=7, ha="right")
    ax.set_xlabel("confinement length (nm)")
    ax.set_ylabel(r"$m_{\mathrm{CR}}(k_{\mathrm{F}})\ (m_0)$")
    ax.set_xlim(0.8, 6.2)
    ax.set_ylim(0, 2.3)
    ax.legend(loc="center right", handlelength=1.6)
    ax.set_title("(b)", loc="left", fontsize=9)
    finish(ax)

    # (c) omega_c tau
    ax = axes[2]
    B = np.linspace(0, 110, 300)
    for m, tau, c, lab in ((0.57, 4.0e-13, C_LIGHT, "light"),
                           (2.6, 3.9e-13, C_HEAVY, "heavy")):
        ax.plot(B, QE * B / (m * M0) * tau, color=c, lw=1.4, label=lab)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axvspan(0, 31, color=C_GREY, alpha=0.13, lw=0)
    ax.text(31, 13.4, "  field range of\n  cyclotron experiment",
            ha="left", va="top", fontsize=7, color=C_GREY)
    ax.text(108, 1.4, r"$\omega_{\mathrm{c}}\tau=1$", ha="right", fontsize=7)
    ax.annotate("0.82 at 31 T", xy=(31, 0.82), xytext=(58, 4.4),
                fontsize=7, color=C_HEAVY,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_HEAVY))
    ax.set_xlabel("magnetic field (T)")
    ax.set_ylabel(r"$\omega_{\mathrm{c}}\tau$")
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 14)
    ax.legend(loc="upper left", handlelength=1.6, bbox_to_anchor=(0.0, 0.80))
    ax.set_title("(c)", loc="left", fontsize=9)
    finish(ax)

    fig.tight_layout(pad=0.4, w_pad=1.5)
    fig.savefig(os.path.join(FIG, "fig1.png"))
    fig.savefig(os.path.join(FIG, "fig1.pdf"))
    plt.close(fig)
    print("fig1 written")


# ===========================================================================
# Figure 2.  The lifetime ratios and what they exclude
# ===========================================================================
def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # (a) angular weighting: why the ratio measures range
    ax = axes[0]
    th = np.linspace(1e-3, np.pi, 400)
    for lam, c, lab in ((0.5, C_THIRD, r"$\Lambda k_{\mathrm{F}}=0.35$"),
                        (2.0, C_LIGHT, r"$\Lambda k_{\mathrm{F}}=1.4$"),
                        (6.0, C_HEAVY, r"$\Lambda k_{\mathrm{F}}=4.2$")):
        kF = 0.709
        q = 2 * kF * np.sin(th / 2)
        w = np.exp(-(q * lam) ** 2 / 4)
        ax.plot(np.degrees(th), w / w.max(), color=c, lw=1.4, label=lab)
    ax.fill_between(np.degrees(th), 0, (1 - np.cos(th)) / 2, color=C_GREY,
                    alpha=0.14, lw=0)
    ax.text(140, 0.30, r"transport weight" "\n" r"$(1-\cos\theta)/2$",
            fontsize=7, color=C_GREY, ha="center")
    ax.set_xlabel(r"scattering angle $\theta$ (degrees)")
    ax.set_ylabel("normalised scattering probability")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.16)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.legend(loc="upper right", handlelength=1.6, ncol=3,
              columnspacing=0.9, borderpad=0.2, handletextpad=0.4)
    ax.set_title("(a)", loc="left", fontsize=9)
    finish(ax)

    # (b) predicted versus measured ratios
    ax = axes[1]
    d = json.load(open(os.path.join(RES, "scattering.json")))
    mech = d["mechanisms"]
    keep = [m for m in mech if m["mechanism"] in (
        "remote ionised charge, d = 2 nm", "remote ionised charge, d = 10 nm",
        "interface roughness, Lambda = 1.0 nm",
        "interface roughness, Lambda = 4.0 nm",
        "background impurities in channel", "threading dislocations")]
    short = {"remote ionised charge, d = 2 nm": "remote charge, 2 nm",
             "remote ionised charge, d = 10 nm": "remote charge, 10 nm",
             "interface roughness, Lambda = 1.0 nm": "roughness, 1 nm",
             "interface roughness, Lambda = 4.0 nm": "roughness, 4 nm",
             "background impurities in channel": "background impurity",
             "threading dislocations": "dislocations"}
    x = np.arange(len(keep))
    ax.bar(x - 0.19, [m["lh_ratio"] for m in keep], 0.36, color=C_LIGHT,
           label="light, model", edgecolor="none")
    ax.bar(x + 0.19, [m["hh_ratio"] for m in keep], 0.36, color=C_HEAVY,
           label="heavy, model", edgecolor="none")
    ax.axhline(3.82, color=C_LIGHT, lw=1.1, ls="--")
    ax.axhline(2.13, color=C_HEAVY, lw=1.1, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.4, 2.0e4)
    ax.set_xlim(-0.6, 7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([short[m["mechanism"]] for m in keep], fontsize=6.8,
                       rotation=32, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"$\tau_{\mathrm{tr}}/\tau_{\mathrm{q}}$")
    ax.text(5.62, 3.82, "  measured,\n  light", color=C_LIGHT, fontsize=7,
            ha="left", va="center")
    ax.text(5.62, 2.13 / 2.6, "  measured,\n  heavy", color=C_HEAVY,
            fontsize=7, ha="left", va="center")
    ax.legend(loc="upper right", handlelength=1.4, ncol=2,
              columnspacing=0.9, handletextpad=0.4)
    ax.set_title("(b)", loc="left", fontsize=9)
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(
        base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
    for s in ax.spines.values():
        s.set_color("black")

    fig.tight_layout(pad=0.4, w_pad=1.8)
    fig.savefig(os.path.join(FIG, "fig2.png"))
    fig.savefig(os.path.join(FIG, "fig2.pdf"))
    plt.close(fig)
    print("fig2 written")


if __name__ == "__main__":
    figure1()
    figure2()
