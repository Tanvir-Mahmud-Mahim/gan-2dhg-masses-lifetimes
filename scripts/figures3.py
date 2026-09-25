"""Figures for the Physical Review B Letter.

Times-metric serif throughout, black ticks and tick labels, 1000 dpi.
Every panel reads from the JSON written by the analysis scripts.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gan2dhg.constants import Q as QE

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
    # Panels (a) and (b) show the finite-barrier solution, which is the
    # calculation the Letter reports; the hard-wall file is kept for the
    # comparison quoted in the Supplemental Material.
    het = os.path.join(RES, "well_het.json")
    d = json.load(open(het if os.path.exists(het)
                       else os.path.join(RES, "well.json")))
    z = np.array(d["z"])
    V = np.array(d["V"])
    p = np.array(d["p_of_z"])
    kt = np.array(d["kt"])
    E = np.array(d["E_of_k"])
    EF = d["EF"]
    E0 = E[0, 0]

    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.05))

    # (a) self-consistent well and hole distribution
    ax = axes[0]
    ax.plot(z, 1000 * (V - V.min()), color="black", lw=1.4)
    for b, c in ((0, C_H), (2, C_L)):
        ax.axhline(1000 * (E[0, b] - E0), color=c, lw=0.9, ls="--")
    ax.axhline(1000 * (EF - E0), color=C_T, lw=0.9, ls=":")
    ax.text(2.42, 1000 * (EF - E0) + 2.5, r"$E_{\mathrm{F}}$", color=C_T,
            fontsize=7, ha="right", va="bottom")
    ax2 = ax.twinx()
    ax2.fill_between(z, 0, p, color=C_G, alpha=0.22, lw=0)
    ax2.plot(z, p, color=C_G, lw=1.0)
    ax2.set_ylabel(r"hole density (nm$^{-3}$)", color="black")
    ax2.tick_params(axis="y", colors="black", direction="in")
    ax2.set_ylim(-0.05 / 0.95 * 1.15 * p.max(), 1.15 * p.max())
    ax2.spines["right"].set_color("black")
    if z.min() < 0:
        ax.axvspan(z.min(), 0.0, color="#c8cdd4", alpha=0.55, lw=0, zorder=0)
        ax.text(-0.45, 72.0, "AlN", fontsize=7, color="#40474f",
                ha="center", va="center")
        ax.text(1.75, 72.0, "GaN", fontsize=7, color="#40474f",
                ha="center", va="center")
    ax.set_xlabel(r"$z$ from interface (nm)")
    ax.set_ylabel("hole energy (meV)")
    ax.set_xlim(max(z.min(), -0.9), 2.5)
    ax.set_ylim(-5, 90)
    ax.set_title("(a)", loc="left", fontsize=9)
    finish(ax)

    # (b) in-plane dispersion
    ax = axes[1]
    dj = json.load(open(os.path.join(RES, "dispersion.json")))
    kd = np.array(dj["k_per_nm"])
    Ed = np.array(dj["E_eV"])
    for b, c, lab in ((0, C_H, "heavy"), (2, C_L, "light")):
        ax.plot(kd, 1000 * (Ed[b] - dj["E0_eV"]), color=c, lw=1.4, label=lab)
    ax.axhline(1000 * (EF - E0), color=C_T, lw=0.9, ls=":")
    ax.text(0.06, 1000 * (EF - E0) + 3, r"$E_{\mathrm{F}}$", color=C_T,
            fontsize=7, ha="left")
    # Fermi wavevector of a SINGLE spin-resolved branch: k_F = sqrt(4 pi n).
    for b, c in ((0, C_H), (2, C_L)):
        n = d["masses"][0]["n_cm2"] if b == 0 else d["masses"][2]["n_cm2"]
        kF = np.sqrt(4 * np.pi * n * 1e-14)
        ax.plot([kF], [1000 * (EF - E0)], "o", color=c, ms=3.5, zorder=5)
    ax.set_xlabel(r"$k_\perp$ (nm$^{-1}$)")
    ax.set_ylabel("hole energy (meV)")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-5, 90)
    ax.legend(loc="lower right", handlelength=1.1, fontsize=6.0,
              labelspacing=0.18, borderpad=0.30, handletextpad=0.4,
              borderaxespad=0.45)
    ax.set_title("(b)", loc="left", fontsize=9)
    finish(ax)

    # (c) mass against sheet density: the prediction, finite barrier at two
    #     valence band offsets
    ax = axes[2]
    sw = json.load(open(os.path.join(RES, "well_sweep.json")))
    for idx, c, mk, lab in (((0, 1), C_H, "o", "heavy"),
                            ((2, 3), C_L, "s", "light")):
        curves = {}
        for r in sw:
            ms = [b["m"] for b in r["bands"] if b["index"] in idx]
            if ms:
                curves.setdefault(r["vbo_eV"], []).append(
                    (r["p_s_cm2"] / 1e13, float(np.mean(ms))))
        vbos = sorted(curves)
        xs = [np.array(sorted(curves[v]))[:, 0] for v in vbos]
        ys = [np.array(sorted(curves[v]))[:, 1] for v in vbos]
        common = xs[0]
        lo = np.min([np.interp(common, x, y) for x, y in zip(xs, ys)], axis=0)
        hi = np.max([np.interp(common, x, y) for x, y in zip(xs, ys)], axis=0)
        ax.fill_between(common, lo, hi, color=c, alpha=0.25, lw=0)
        ax.plot(common, 0.5 * (lo + hi), mk + "-", color=c, ms=2.8, lw=1.1,
                label=lab)
    ax.errorbar([4.6], [1.92], yerr=[0.16], fmt="D", color=C_H, ms=3.6,
                mfc="white", mew=1.0, capsize=2.2, lw=1.0,
                label="measured, heavy")
    ax.errorbar([4.6], [0.53], yerr=[0.01], fmt="D", color=C_L, ms=3.6,
                mfc="white", mew=1.0, capsize=2.2, lw=1.0,
                label="measured, light")
    fits = json.load(open(os.path.join(RES, "a6_apply.json")))[
        "fits_to_measured_light_occupation"]
    mf = [f["m_lh"] for f in fits.values() if f]
    ax.errorbar([4.85], [np.mean(mf)], yerr=[[np.mean(mf) - min(mf)],
                                             [max(mf) - np.mean(mf)]],
                fmt="^", color=C_L, ms=3.6, capsize=2.0, lw=1.0,
                label=r"light, rescaled $A_6$")
    ax.set_xlabel(r"sheet density ($10^{13}$ cm$^{-2}$)")
    ax.set_ylabel(r"$m_{\mathrm{CR}}(k_{\mathrm{F}})\ (m_0)$")
    ax.set_xlim(1.6, 6.9)
    ax.set_ylim(0, 2.4)
    ax.legend(loc="center left", handlelength=1.2, labelspacing=0.2,
              fontsize=5.6, bbox_to_anchor=(0.0, 0.47), borderaxespad=0.3)
    ax.set_title("(c)", loc="left", fontsize=9)
    finish(ax)

    # (d) light-hole mass from a Lifshitz-Kosevich analysis of the computed
    #     Landau levels (level width fixed by the measured Dingle slope,
    #     scripts/run_landau_dingle.py), with the published A6 and with A6
    #     rescaled to the measured light-hole occupation, against the reported
    #     field dependence and the cyclotron-resonance mass
    ax = axes[3]
    ld = json.load(open(os.path.join(RES, "landau_dingle.json")))
    fields = ["32", "40", "48", "56", "64", "72"]
    Bc = np.array([float(f) for f in fields])
    for well, c, lab in (("well_het_pol03.json", C_G, "published $A_6$"),
                         ("well_het_pol03_A6.json", C_L, r"rescaled $A_6$")):
        rec = ld[well]
        rows = [np.array([k["LK_fixed_density"][f] for f in fields])
                for k in rec["kappa"].values()]
        rows = np.array(rows)
        ax.fill_between(Bc, rows.min(0), rows.max(0), color=c, alpha=0.25,
                        lw=0)
        k0 = rec["kappa"]["0.0"]["LK_fixed_density"]
        ax.plot(Bc, [k0[f] for f in fields], "s-", color=c, ms=2.6, lw=1.0,
                label=lab)
        ax.plot([1.5], [rec["m_light_zero_field"]], "o", color=c, ms=3.4,
                mfc=c, zorder=5)
    meas = json.load(open(os.path.join(RES, "landau.json")))["measured"]
    slope = (meas["m_72T"] - meas["m_32T"]) / 40.0
    ax.plot([32, 72], [meas["m_32T"], meas["m_72T"]], color="black", lw=1.1,
            ls="--", label="reported")
    ax.plot([0, 32], [meas["m_32T"] - 32 * slope, meas["m_32T"]],
            color="black", lw=0.8, ls=":")
    ax.plot([31.0], [0.57], "^", color="black", ms=3.8, mfc="white", mew=0.9,
            label="cyclotron resonance")
    ax.set_xlabel("magnetic field (T)")
    ax.set_ylabel(r"light-hole mass ($m_0$)")
    ax.set_xlim(0, 80)
    ax.set_ylim(0.0, 0.8)
    ax.legend(loc="lower right", handlelength=1.4, labelspacing=0.15,
              fontsize=5.4, borderaxespad=0.25, handletextpad=0.4)
    ax.set_title("(d)", loc="left", fontsize=9)
    finish(ax)

    fig.tight_layout(pad=0.4, w_pad=1.2)
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
    ax.text(104, 1.45, r"$\omega_{\mathrm{c}}\tau=1$", ha="right", fontsize=7)
    ax.annotate("0.82 at 31 T", xy=(31.0, 0.82), xytext=(50, 5.2), fontsize=7,
                color=C_H, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_H,
                                shrinkA=0, shrinkB=1))
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
    ax.set_ylabel("normalized probability")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.46)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.legend(loc="upper right", handlelength=1.2, labelspacing=0.18,
              fontsize=6.0, borderpad=0.25, handletextpad=0.4,
              borderaxespad=0.35)
    ax.set_title("(b)", loc="left", fontsize=9)
    finish(ax)

    # (c) the two lifetime ratios, computed against measured.  Each curve is
    #     one mechanism with its shape parameter scanned continuously,
    #     solved with the coupled two-subband Boltzmann equation and the
    #     computed Bloch overlap.  No curve reaches the measured region.
    ax = axes[2]
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from gan2dhg import measured as MS
    d = json.load(open(os.path.join(RES, "tension.json")))
    style = {"interface roughness": (C_L, "-", "roughness"),
             "remote ionised charge": (C_H, "-", "remote charge"),
             "charged dislocations": (C_T, "-", "dislocations"),
             "background impurities": ("black", "o", "background")}
    for f in d["figure"]:
        if f["name"] == "charged dislocations":
            continue          # ratios above 300 for both subbands, off scale
        c, ls, lab = style[f["name"]]
        x, y = np.array(f["ratio_light"]), np.array(f["ratio_heavy"])
        if ls == "o":
            ax.plot(x, y, "o", color=c, ms=3.2, mfc="white", mew=0.9,
                    label=lab, zorder=4)
        else:
            ax.plot(x, y, ls, color=c, lw=1.3, label=lab)
    mx = json.load(open(os.path.join(RES, "mixtures.json")))
    tr = mx["closest_pair_trajectory"]
    x = np.array([p["ratios"][0] for p in tr["points"]])
    y = np.array([p["ratios"][1] for p in tr["points"]])
    ax.plot(x, y, color="#6a3d9a", lw=1.1, ls="-.", label="two mechanisms")
    (l_lo, l_hi), (h_lo, h_hi) = MS.TARGET_BOX
    ax.add_patch(matplotlib.patches.Rectangle(
        (l_lo, h_lo), l_hi - l_lo, h_hi - h_lo, facecolor=C_G, alpha=0.20,
        edgecolor="none", zorder=1))
    ax.errorbar([MS.R_L], [MS.R_H],
                xerr=[[MS.R_L - MS.R_L_RANGE[0]], [MS.R_L_RANGE[1] - MS.R_L]],
                yerr=[[MS.R_H - MS.R_H_RANGE[0]], [MS.R_H_RANGE[1] - MS.R_H]],
                fmt="D", color="black", ms=4.0, mfc="white", mew=1.0,
                capsize=2.0, lw=0.9, zorder=6, label="measured")
    ax.plot([0.3, 3e3], [0.3, 3e3], color=C_G, lw=0.6, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1.0e3)
    ax.set_ylim(0.5, 1.0e3)
    ax.set_xlabel(r"$\tau_{\mathrm{tr}}/\tau_{\mathrm{q}}$, light subband")
    ax.set_ylabel(r"$\tau_{\mathrm{tr}}/\tau_{\mathrm{q}}$, heavy subband")
    ax.legend(loc="upper left", handlelength=1.3, labelspacing=0.2,
              fontsize=6.0, borderaxespad=0.35)
    ax.set_title("(c)", loc="left", fontsize=9)
    for s_ in ax.spines.values():
        s_.set_color("black")

    fig.tight_layout(pad=0.4, w_pad=2.0)
    fig.savefig(os.path.join(FIG, "prb_fig2.png"))
    fig.savefig(os.path.join(FIG, "prb_fig2.pdf"))
    plt.close(fig)
    print("prb_fig2 written")


if __name__ == "__main__":
    figure1()
    figure2()
