"""Figure: the EMD4 cascade, simulated.

Follows the conventions established in emd1_simulation/figure.py and
figures/regenerate_figures.ipynb: exact 183 mm double-column canvas, one shared
type scale with a 6 pt floor, Type 42 embedded fonts, colour assigned by role,
solid tints rather than alpha (EPS has no real transparency), vector + 600 dpi
raster export.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# Must precede the pyplot import. The interactive macOS backend snaps figure
# width to whole screen pixels, which silently costs 0.34 pt and breaks the
# "the PDF measures exactly 183 mm" guarantee the other figures hold to.
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .conditions import BY_KEY, PLOTTED, T_OFF
from .model import MARKERS

MM = 1 / 25.4
W_DOUBLE = 183 * MM

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans",
                        "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "mathtext.default": "regular",
    "figure.dpi": 160, "savefig.dpi": 600,
    "savefig.bbox": None, "savefig.pad_inches": 0,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
})

INK, INK2, MUTED, RULE = "#12253a", "#3d4a58", "#6b7785", "#c9d1da"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bbf"]
FLAG = "#b3323f"
TS = {"fignote": 6.6, "body": 7.2, "label": 7.6, "head": 8.6, "title": 9.4}

COND_COLOR = {"control": MUTED, "as": CAT[0], "as_low": CAT[2], "as_sasp": CAT[3]}


def tint(hex_color: str, frac: float) -> str:
    """Blend toward white, returning a SOLID hex (EPS-safe; no alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * (1 - frac)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _style(ax, title, ylabel, tag, xlabel=None):
    ax.set_title(title, fontsize=TS["label"], color=INK, pad=3.5, loc="left")
    ax.set_ylabel(ylabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    ax.tick_params(labelsize=TS["fignote"], colors=INK2, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.text(-0.22, 1.16, tag, transform=ax.transAxes, fontsize=TS["head"],
            fontweight="bold", color=INK, va="top", ha="left")


def _withdrawal_marker(ax, ymax=1.0):
    ax.axvline(T_OFF, color=RULE, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    ax.text(T_OFF + 4, ymax * 0.97, "withdrawal", fontsize=TS["fignote"] - 0.6,
            color=MUTED, va="top", ha="left")


def make_figure(runs: dict, t: np.ndarray, ctx: dict, outdir: Path) -> list[Path]:
    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.66))
    gs = fig.add_gridspec(2, 4, left=0.055, right=0.978, top=0.855, bottom=0.080,
                          wspace=0.50, hspace=0.60)

    # --- a: latent trajectories ------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for c in PLOTTED:
        ax.plot(t, runs[c.key]["S"], color=COND_COLOR[c.key], linewidth=1.1,
                label=c.label.replace("Arsenite", "As")
                              .replace(", withdrawn d14", "")
                              .replace(" neutralisation d60", " neut. d60"))
    _withdrawal_marker(ax)
    _style(ax, "Latent senescent state", "S", "a", "day")
    ax.set_xlim(0, 400); ax.set_ylim(-0.03, 1.06)
    ax.legend(fontsize=TS["fignote"] - 1.0, frameon=False, loc="upper right",
              bbox_to_anchor=(1.02, 0.86), handlelength=1.2,
              borderaxespad=0.0, labelspacing=0.22)

    # --- b: markers vs latent state (construct validity) ------------------
    ax = fig.add_subplot(gs[0, 1])
    tr = runs["as_low"]
    ax.plot(t, tr["S"], color=INK, linewidth=1.4, zorder=5, label="latent S")
    for m, col in zip([MARKERS[3], MARKERS[4], MARKERS[0]], [FLAG, tint(FLAG, 0.55), CAT[0]]):
        ax.plot(t, tr[m.key], color=col, linewidth=0.9, label=m.label)
    _withdrawal_marker(ax, 0.26)
    _style(ax, "Markers overstate the state", "fraction positive", "b", "day")
    ax.set_xlim(0, 120); ax.set_ylim(-0.01, 0.26)
    ax.legend(fontsize=TS["fignote"] - 1.0, frameon=False, loc="center right",
              handlelength=1.2, borderaxespad=0.3, labelspacing=0.22)

    # --- c: forward dose-response, the null matches -----------------------
    ax = fig.add_subplot(gs[0, 2])
    n = ctx["null"]
    from .nulls import forward_dose_response, simulate_null
    from .model import Intervention, Params
    doses = np.linspace(0.0, 0.4, 33)
    p = Params()
    mech = forward_dose_response(p, Intervention(), doses, t_end=60.0)
    tf = np.linspace(0.0, 60.0, 121)
    nullv = np.array([simulate_null(n["endpoint_fit"], float(E), tf)[-1] for E in doses])
    ax.plot(doses, mech, color=INK, linewidth=1.4, label="mechanistic", zorder=4)
    ax.plot(doses, nullv, color=FLAG, linewidth=1.0, linestyle=(0, (3, 2)),
            label="no-feedback null")
    _style(ax, f"Forward curve: R$^2$ = {n['endpoint_r2']:.3f}",
           "S at day 60", "c", "dose E")
    ax.set_ylim(-0.03, 1.0)
    ax.legend(fontsize=TS["fignote"] - 0.8, frameon=False, loc="lower right",
              handlelength=1.4, borderaxespad=0.2, labelspacing=0.25)

    # --- d: withdrawal, the null fails ------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    from .nulls import compare_withdrawal, cascade_withdrawal
    w = compare_withdrawal(p, Intervention(), n["timecourse_fit"], 1.0, T_OFF, 400.0)
    wc = cascade_withdrawal(p, Intervention(), n["cascade_fit"], 1.0, T_OFF, 400.0)
    ax.plot(w["t"], w["mech"], color=INK, linewidth=1.4, label="mechanistic")
    ax.plot(w["t"], w["null"], color=FLAG, linewidth=1.0, linestyle=(0, (3, 2)),
            label="simple null")
    ax.plot(wc["t"], wc["null"], color=tint(FLAG, 0.55), linewidth=1.0,
            linestyle=(0, (1, 1.5)), label="cascade null")
    _withdrawal_marker(ax)
    _style(ax, "Withdrawal separates them", "S", "d", "day")
    ax.set_xlim(0, 400); ax.set_ylim(-0.03, 1.0)
    ax.legend(fontsize=TS["fignote"] - 0.8, frameon=False, loc="center right",
              handlelength=1.4, borderaxespad=0.2, labelspacing=0.25)

    # --- e: hysteresis ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    h = ctx["bif"]["hyst"]
    ax.plot(h["E"], h["up"], color=CAT[0], linewidth=1.2, marker="o",
            markersize=1.8, label="up-sweep")
    ax.plot(h["E"], h["down"], color=CAT[1], linewidth=1.2, marker="s",
            markersize=1.8, label="down-sweep")
    if ctx["bif"]["fold"] is not None:
        ax.axvline(ctx["bif"]["fold"], color=RULE, linewidth=0.6,
                   linestyle=(0, (2, 2)), zorder=0)
    _style(ax, f"Hysteresis, area {h['width']:.3f}", "S at equilibrium", "e", "dose E")
    ax.set_ylim(-0.03, 1.0)
    ax.legend(fontsize=TS["fignote"] - 0.8, frameon=False, loc="center right",
              handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)

    # --- f: critical slowing ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    fold = ctx["bif"]["fold"]
    dE = fold - ctx["bif"]["rt_E"]
    good = dE > 0
    ax.loglog(dE[good], ctx["bif"]["rt_tau"][good], color=CAT[0], linewidth=1.2,
              marker="o", markersize=2.0)
    ref = dE[good][dE[good] < 6e-3]        # near-fold regime only
    tau_g = ctx["bif"]["rt_tau"][good]
    ax.loglog(ref, tau_g[good.sum() - 1] * np.sqrt(dE[good][-1] / ref),
              color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)),
              label=r"$\propto (E^*-E)^{-1/2}$")
    _style(ax, "Critical slowing", "recovery time (d)", "f", "$E^* - E$")
    ax.legend(fontsize=TS["fignote"] - 0.8, frameon=False, loc="upper right",
              handlelength=1.4, borderaxespad=0.2)

    # --- g: spatial containment -------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    if ctx["spatial"] is not None:
        sc = ctx["spatial"]["scan"]
        # Escape PROBABILITY with its Wilson interval, and nothing else. The
        # cluster-size and radius columns this panel used to carry are the
        # saturation value times this same probability (r = 1.000), so plotting
        # them alongside drew one measurement as two.
        esc = sc["escaped"] * 100
        ax.fill_between(sc["q"], sc["ci_lo"] * 100, sc["ci_hi"] * 100,
                        color=tint(CAT[1], 0.22), linewidth=0, zorder=1)
        ax.plot(sc["q"], esc, color=CAT[1], linewidth=1.2, marker="s",
                markersize=2.0, zorder=3)
        ax.set_ylim(-4, 108)
        ax.axvline(ctx["bif"]["q_crit"], color=FLAG, linewidth=0.8,
                   linestyle=(0, (2, 2)), zorder=0)
        ax.text(ctx["bif"]["q_crit"] - 0.015, 82,
                f"mean-field\n$q^*$={ctx['bif']['q_crit']:.2f}",
                fontsize=TS["fignote"] - 1.0, color=FLAG, va="center", ha="right")
        ax.axhline(50, color=RULE, linewidth=0.6, linestyle=(0, (1, 2)), zorder=0)
        ax.legend(handles=[Line2D([], [], color=CAT[1], lw=1.2, marker="s",
                                  markersize=2, label="escaping"),
                           Line2D([], [], color=tint(CAT[1], 0.22), lw=4,
                                  label=f"95% CI, n={sc['n_rep']}")],
                  fontsize=TS["fignote"] - 1.0, frameon=False, loc="upper left",
                  handlelength=1.4, borderaxespad=0.2, labelspacing=0.2)
    _style(ax, "Spatial propagation", "% of replicates escaping", "g", "$q_{sec}$")

    # --- h: ensemble sign robustness --------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    ens = ctx.get("ensemble")
    n_draws = int(ens["n"]) if ens is not None else 0
    if ens is not None:
        # "bistable" previously pointed at frac_bistable, which is a
        # finite-time RETENTION count, not a count of two stable equilibria.
        # The bar now shows the equilibrium quantity and says so; the paired
        # control contrast replaces the unpaired persistence count, which did
        # not require the matched control to stay below threshold.
        labels = ["2 stable\nequilibria", "persists\n(vs control)",
                  "low dose\n<0.30", "SA-βgal over\n(net of ctrl)",
                  "SASP neut.", "marker\nfalls"]
        vals = [ens["frac_two_stable_equilibria"],
                ens["frac_persist_exposure_attributable"],
                ens["frac_lowdose_resolves"],
                ens["frac_v3_overstate_net_positive"],
                ens["frac_v7_sasp_collapses"],
                ens["frac_v3_understate"]]
        cols = [CAT[2] if v >= 0.90 else (CAT[1] if v >= 0.60 else FLAG) for v in vals]
        y = np.arange(len(vals))[::-1]
        # Counts, not percentages -- the same rule the text follows. The draw
        # widths are judgement-based, so a percentage would imply a calibration
        # this procedure does not have.
        counts = [int(round(v * n_draws)) for v in vals]
        ax.barh(y, counts, color=cols, height=0.62, edgecolor="none")
        ax.axvline(0.90 * n_draws, color=RULE, linewidth=0.6,
                   linestyle=(0, (2, 2)), zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=TS["fignote"] - 1.0)
        ax.set_xlim(0, n_draws)
        for yy, c_ in zip(y, counts):
            inside = c_ > 0.72 * n_draws
            ax.text(c_ - 0.02 * n_draws if inside else c_ + 0.02 * n_draws, yy,
                    f"{c_}/{n_draws}", fontsize=TS["fignote"] - 0.8,
                    color="white" if inside else INK2,
                    va="center", ha="right" if inside else "left")
    _style(ax, f"Sign robustness ({n_draws} draws)",
           "", "h", "draws holding")

    fig.text(0.055, 0.983,
             "EMD4: persistent senescence and SASP   "
             "KCC5 → EMD4/KCC6 → KCC7, KCC10   (KCC9 opposing)",
             fontsize=TS["title"], fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.938,
             "Arsenite exemplar. Relative effect sizes, not physiological rate "
             "constants. Model output is not an independent KCC or EMD positive.",
             fontsize=TS["fignote"], color=MUTED, va="top")

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png", "eps", "tif"):
        pth = outdir / f"figure2c_emd4_simulation.{ext}"
        fig.savefig(pth, format=ext)
        paths.append(pth)
    plt.close(fig)
    print(f"  wrote {paths[0].with_suffix('.{pdf,svg,png,eps,tif}')}")
    return paths
