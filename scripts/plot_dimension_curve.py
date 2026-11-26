"""fig11: routing gain vs effective action-space dimension.

  python scripts/plot_dimension_curve.py

Three things to read off this figure:
  1. Within one editor family, gain rises monotonically with dimension (7 points).
  2. The alpha=1 point is the null control: the four editors are identical by
     construction, so any router can only match random, and it does land on 0.
  3. The generative branch (a different family) lands where its dimension predicts.

Error bars are the range over 3 training seeds, not a confidence interval: n = 3
cannot support a credible CI, and drawing one would read as more certain than the
data is. The generative point has only one seed, so its bars are that seed's
bootstrap CI, labelled as such on the figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.figstyle import (  # noqa: E402
    DISPLAY, GRID, INK, INK_SECONDARY, MARKER, PALETTE, apply_style,
)

SHOW = ["greedy", "bandit", "dqn"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results_sweep/dimension_curve.json")
    ap.add_argument("--out", default="figures/fig11_dimension_curve.png")
    ap.add_argument("--no-caption", action="store_true",
                    help="omit the caption embedded in the figure")
    args = ap.parse_args()

    blob = json.loads(Path(args.curve).read_text())
    tab = blob["table"]
    fam = {k: v for k, v in tab.items() if k != "generative"}
    gen = tab.get("generative")

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Reference line for "equal to random". Grey is a chrome color, not a series
    # color (see figstyle).
    ax.axhline(0, color=PALETTE["random"], linestyle=(0, (4, 2)),
               linewidth=1.2, zorder=1)
    ax.annotate("equal to random", xy=(2.59, 0), xytext=(2.59, -0.00012),
                color=INK_SECONDARY, fontsize=8, ha="right")

    xs_all = sorted(v["eff_dim"] for v in fam.values())
    for name in SHOW:
        pts = sorted((v["eff_dim"], [s[name][0] for s in v["seeds"]])
                     for v in fam.values() if name in v["seeds"][0])
        x = np.array([p for p, _ in pts])
        ys = [np.array(v) for _, v in pts]
        mu = np.array([v.mean() for v in ys])
        lo = np.array([v.min() for v in ys])
        hi = np.array([v.max() for v in ys])
        c = PALETTE[name]

        # OLS fit: the slope is the claim itself, how much HV one more action
        # dimension is worth
        flatx = np.repeat(x, [len(v) for v in ys])
        flaty = np.concatenate(ys)
        b, a = np.polyfit(flatx, flaty, 1)
        gx = np.linspace(min(xs_all) - 0.04, max(xs_all) + 0.04, 50)
        ax.plot(gx, a + b * gx, color=c, linewidth=1.0, alpha=0.45,
                linestyle="-", zorder=2)

        ax.vlines(x, lo, hi, color=c, linewidth=1.6, alpha=0.75, zorder=3)
        ax.plot(x, mu, marker=MARKER[name], markersize=7.5, color=c,
                linewidth=0, markeredgecolor="white", markeredgewidth=1.0,
                label=f"{DISPLAY[name]}  ({b:+.5f}/dim)", zorder=4)

    # Generative branch: a different editor family with only one seed, so it is
    # deliberately drawn as hollow black markers with no line and no series color,
    # to avoid reading as a fourth series on the curve.
    if gen is not None and gen["eff_dim"]:
        gx = gen["eff_dim"]
        # Offset the three policies slightly in x; otherwise the three single-seed
        # CIs stack into one full-height vertical line that reads as a divider
        # rather than data.
        for off, name in zip((-0.035, 0.0, 0.035), SHOW):
            if name not in gen["seeds"][0]:
                continue
            m, clo, chi = gen["seeds"][0][name]
            ax.vlines(gx + off, clo, chi, color=INK, linewidth=0.9,
                      alpha=0.30, zorder=3)
            ax.plot([gx + off], [m], marker=MARKER[name], markersize=7.0,
                    markerfacecolor="none", markeredgecolor=INK,
                    markeredgewidth=1.2, linewidth=0, zorder=5)
        ax.annotate("generative backbone\n(other editor family, 1 seed;\n"
                    "bars are that seed's CI)",
                    xy=(gx + 0.05, -0.00075), xytext=(2.03, -0.00155),
                    color=INK, fontsize=8, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK_SECONDARY,
                                    linewidth=0.8, shrinkA=4, shrinkB=3))

    # Note on the null control: this point landing on 0 tests the measurement
    # pipeline rather than being a finding
    x0 = min(xs_all)
    # Text goes top-right with a long arrow; a short arrow would appear to point
    # at the cluster near 1.28
    ax.annotate("null control: four editors\nidentical by construction",
                xy=(x0, 0.00006), xytext=(1.16, 0.00128),
                color=INK_SECONDARY, fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK_SECONDARY,
                                linewidth=0.8, shrinkA=4, shrinkB=4))

    ax.set_xlabel("Effective action dimension  "
                  "(participation ratio, per-objective normalised)")
    ax.set_ylabel("Hypervolume gain vs random\n"
                  "(paired, at matched candidate budget x = 5)")
    ax.set_title("Routing gain tracks action-space dimensionality, "
                 "not router sophistication", fontsize=11, loc="left",
                 pad=12)
    ax.set_xlim(0.92, 2.62)
    # The lower limit must fit the generative point's single-seed CI (down to about
    # -0.0018); a clipped error bar would understate the uncertainty.
    ax.set_ylim(-0.00215, 0.00272)
    ax.legend(loc="upper left", ncol=1, handletextpad=0.6)

    if not args.no_caption:
        _caption(fig)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


def _caption(fig) -> None:
    cap = ("Error bars are the range over 3 training seeds (not a CI: n = 3). RDKit pool\n"
           "overlap swept via src/editors.py:_mix; points placed by measured dimension,\n"
           "not knob setting.")
    fig.text(0.0, -0.022, cap, fontsize=7.5, color=INK_SECONDARY, ha="left",
             va="top", linespacing=1.45)


if __name__ == "__main__":
    raise SystemExit(main())
