"""Figures for the Physical Review B Letter.

Times-metric serif throughout, black ticks and tick labels, 1000 dpi.
Every panel reads from the JSON written by the analysis scripts.
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
from gpfet.constants import Q as QE

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
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
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

C_H = "#1f4e79"       # heavy
C_L = "#c1121f"       # light
C_G = "#5a5a5a"
C_T = "#2a7f62"
M0 = 9.1093837015e-31


def finish(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for s in ax.spines.values():
        s.set_color("black")


def figure1():
    d = json.load(open(os.path.join(RES, "well.json")))
    z = np.array(d["z"])
    V = np.array(d["V"])
    p = np.array(d["p_of_z"])
    kt = np.array(d["kt"])
    E = np.array(d["E_of_k"])
    EF = d["EF"]
    E0 = E[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))

    # (a) self-consistent well and hole distribution
    ax = axes[0]
    ax.plot(z, 1000 * (V - V.min()), color="black", lw=1.4,
            label="self-consistent $V(z)$")
    for b, c, lab in ((0, C_H, "heavy"), (2, C_L, "light")):
        ax.axhline(1000 * (E[0, b] - E0), color=c, lw=0.9, ls="--")
    ax.axhline(1000 * (EF - E0), color=C_T, lw=0.9, ls=":")
    ax.text(2.42, 1000 * (E[0, 0] - E0) + 2.5, "heavy edge", color=C_H,
            fontsize=7, ha="right")
    ax.text(2.42, 1000 * (E[0, 2] - E0) + 2.5, "light edge", color=C_L,
            fontsize=7, ha="right")
    ax.text(2.42, 1000 * (EF - E0) + 2.5, r"$E_{\mathrm{F}}$", color=C_T,
            fontsize=7, ha="right")
    ax2 = ax.twinx()
    ax2.fill_between(z, 0, p, color=C_G, alpha=0.22, lw=0)
    ax2.plot(z, p, color=C_G, lw=1.0)
    ax2.set_ylabel(r"hole density (nm$^{-3}$)", color=C_G)
    ax2.tick_params(axis="y", colors="black", direction="in")
    ax2.set_ylim(0, 1.15 * p.max())
    ax2.spines["right"].set_color("black")
    ax.annotate(r"$V(z)$", xy=(0.13, 62), xytext=(0.60, 68), fontsize=7.5,
                color="black",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))
    ax.set_xlabel(r"distance from GaN/AlN interface (nm)")
    ax.set_ylabel("hole energy (meV)")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 90)
    ax.set_title("(a)", loc="left", fontsize=9)
    finish(ax)

    # (b) in-plane dispersion
    ax = axes[1]
    for b, c, lab in ((0, C_H, "heavy"), (2, C_L, "light")):
        ax.plot(kt, 1000 * (E[:, b] - E0), color=c, lw=1.4, label=lab)
    ax.axhline(1000 * (EF - E0), color=C_T, lw=0.9, ls=":")
    ax.text(2.02, 1000 * (EF - E0) + 3, r"$E_{\mathrm{F}}$", color=C_T,
            fontsize=7, ha="right")
    for b, c in ((0, C_H), (2, C_L)):
        n = d["masses"][0]["n_cm2"] if b == 0 else d["masses"][2]["n_cm2"]
        kF = np.sqrt(2 * np.pi * n * 1e-14)
        ax.plot([kF], [1000 * (EF - E0)], "o", color=c, ms=3.5, zorder=5)
    ax.set_xlabel(r"in-plane wavevector $k_\perp$ (nm$^{-1}$)")
    ax.set_ylabel("hole energy (meV)")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, 90)
    ax.legend(loc="upper left", handlelength=1.6)
    ax.set_title("(b)", loc="left", fontsize=9)
    finish(ax)

    # (c) mass against sheet density: the prediction
    ax = axes[2]
    sp = os.path.join(RES, "well_sweep.json")
    if os.path.exists(sp):
        sw = json.load(open(sp))
        ns, mh, ml = [], [], []
        for r in sw:
            hv = [b["m"] for b in r["bands"] if b["m"] > 1.0]
            lv = [b["m"] for b in r["bands"] if b["m"] <= 1.0]
            if not hv:
                continue
            ns.append(r["p_s_cm2"] / 1e13)
            mh.append(np.mean(hv))
            ml.append(np.mean(lv) if lv else np.nan)
        o = np.argsort(ns)
        ns = np.array(ns)[o]
        ax.plot(ns, np.array(mh)[o], "o-", color=C_H, ms=3.2, lw=1.2,
                label="heavy")
        ax.plot(ns, np.array(ml)[o], "s-", color=C_L, ms=3.2, lw=1.2,
                label="light")
    ax.errorbar([4.6], [1.92], yerr=[0.16], fmt="D", color=C_H, ms=4.0,
                mfc="white", mew=1.0, capsize=2.5, lw=1.0,
                label="measured, heavy")
    ax.errorbar([4.6], [0.30], yerr=[0.03], fmt="D", color=C_L, ms=4.0,
                mfc="white", mew=1.0, capsize=2.5, lw=1.0,
                label=r"measured, light ($B\!\to\!0$)")
    ax.set_xlabel(r"sheet density $p_{\mathrm{s}}$ ($10^{13}$ cm$^{-2}$)")
    ax.set_ylabel(r"$m_{\mathrm{CR}}(k_{\mathrm{F}})\ (m_0)$")
    ax.set_ylim(0, 2.4)
    ax.legend(loc="center left", handlelength=1.5, labelspacing=0.28,
              bbox_to_anchor=(0.0, 0.42))
    ax.set_title("(c)", loc="left", fontsize=9)
    finish(ax)

    fig.tight_layout(pad=0.4, w_pad=2.4)
    fig.savefig(os.path.join(FIG, "prb_fig1.png"))
    fig.savefig(os.path.join(FIG, "prb_fig1.pdf"))
    plt.close(fig)
    print("prb_fig1 written")


def figure2():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))

    # (a) omega_c tau
    ax = axes[0]
    B = np.linspace(0, 110, 300)
    for m, tau, c, lab in ((0.57, 4.0e-13, C_L, "light"),
                           (2.6, 3.9e-13, C_H, "heavy")):
        ax.plot(B, QE * B / (m * M0) * tau, color=c, lw=1.4, label=lab)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axvspan(0, 31, color=C_G, alpha=0.13, lw=0)
    ax.text(33, 13.2, "field range of the\ncyclotron experiment",
            ha="left", va="top", fontsize=7, color=C_G)
    ax.text(108, 1.45, r"$\omega_{\mathrm{c}}\tau=1$", ha="right", fontsize=7)
    ax.annotate("0.82 at 31 T", xy=(31, 0.82), xytext=(56, 4.2), fontsize=7,
                color=C_H, arrowprops=dict(arrowstyle="->", lw=0.7, color=C_H))
    ax.set_xlabel("magnetic field (T)")
    ax.set_ylabel(r"$\omega_{\mathrm{c}}\tau$")
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 14)
    ax.legend(loc="upper left", handlelength=1.6, bbox_to_anchor=(0.0, 0.72))
    ax.set_title("(a)", loc="left", fontsize=9)
    finish(ax)

    # (b) angular character
    ax = axes[1]
    th = np.linspace(1e-3, np.pi, 400)
    kF = 0.709
    for lam, c, lab in ((0.5, C_T, r"$\Lambda k_{\mathrm{F}}=0.35$"),
                        (2.0, C_L, r"$\Lambda k_{\mathrm{F}}=1.4$"),
                        (6.0, C_H, r"$\Lambda k_{\mathrm{F}}=4.2$")):
        q = 2 * kF * np.sin(th / 2)
        w = np.exp(-(q * lam) ** 2 / 4)
        ax.plot(np.degrees(th), w / w.max(), color=c, lw=1.4, label=lab)
    ax.fill_between(np.degrees(th), 0, (1 - np.cos(th)) / 2, color=C_G,
                    alpha=0.14, lw=0)
    ax.text(138, 0.30, "transport weight\n" r"$(1-\cos\theta)/2$", fontsize=7,
            color=C_G, ha="center")
    ax.set_xlabel(r"scattering angle $\theta$ (deg)")
    ax.set_ylabel("normalised probability")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.18)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.legend(loc="upper right", handlelength=1.4, labelspacing=0.25)
    ax.set_title("(b)", loc="left", fontsize=9)
    finish(ax)

    # (c) predicted versus measured ratios
    ax = axes[2]
    d = json.load(open(os.path.join(RES, "scattering.json")))
    keep_names = ["remote ionised charge, d = 2 nm",
                  "remote ionised charge, d = 10 nm",
                  "interface roughness, Lambda = 1.0 nm",
                  "interface roughness, Lambda = 4.0 nm",
                  "background impurities in channel",
                  "threading dislocations"]
    short = ["remote 2 nm", "remote 10 nm", "rough 1 nm", "rough 4 nm",
             "background", "dislocation"]
    keep = [m for n in keep_names for m in d["mechanisms"]
            if m["mechanism"] == n]
    x = np.arange(len(keep))
    ax.bar(x - 0.19, [m["lh_ratio"] for m in keep], 0.36, color=C_L,
           label="light", edgecolor="none")
    ax.bar(x + 0.19, [m["hh_ratio"] for m in keep], 0.36, color=C_H,
           label="heavy", edgecolor="none")
    ax.axhline(3.82, color=C_L, lw=1.1, ls="--")
    ax.axhline(2.13, color=C_H, lw=1.1, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.4, 4.0e4)
    ax.set_xlim(-0.6, 5.6)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6.6, rotation=34, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(r"$\tau_{\mathrm{tr}}/\tau_{\mathrm{q}}$")
    ax.text(-0.45, 5.6, "measured, light", color=C_L, fontsize=6.8)
    ax.text(-0.45, 1.30, "measured, heavy", color=C_H, fontsize=6.8)
    ax.legend(loc="upper right", handlelength=1.3, ncol=2, columnspacing=0.8)
    ax.set_title("(c)", loc="left", fontsize=9)
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(
        base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
    for s in ax.spines.values():
        s.set_color("black")

    fig.tight_layout(pad=0.4, w_pad=2.0)
    fig.savefig(os.path.join(FIG, "prb_fig2.png"))
    fig.savefig(os.path.join(FIG, "prb_fig2.pdf"))
    plt.close(fig)
    print("prb_fig2 written")


if __name__ == "__main__":
    figure1()
    figure2()
