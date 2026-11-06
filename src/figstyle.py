"""Shared figure style: fixed palette, readable in greyscale.

Colors come from a validated categorical palette (fixed order, no cycling).
`random` is deliberately a grey dashed line: it is the baseline for every paired
comparison, not one of six peer series, and should read that way.

Each policy also carries a linestyle and marker as a second encoding, since the
report may be printed in greyscale.
"""
from __future__ import annotations

# Fixed slots of the categorical palette; random uses a neutral grey
PALETTE = {
    "random":      "#8a8a85",   # baseline: neutral grey
    "round_robin": "#2a78d6",   # slot 1 blue
    "gap":         "#eb6834",   # slot 2 orange
    "greedy":      "#1baf7a",   # slot 3 aqua
    "bandit":      "#eda100",   # slot 4 yellow
    "dqn":         "#e87ba4",   # slot 5 magenta
}

LINESTYLE = {
    "random": (0, (4, 2)), "round_robin": "-", "gap": (0, (1, 1.5)),
    "greedy": "-", "bandit": (0, (5, 1.5, 1, 1.5)), "dqn": (0, (3, 1, 1, 1)),
}

MARKER = {
    "random": None, "round_robin": "o", "gap": "s",
    "greedy": "^", "bandit": "D", "dqn": "v",
}

# Names shown in the figures
DISPLAY = {
    "random": "Random", "round_robin": "Round-robin", "gap": "Target-gap",
    "greedy": "Greedy (1-step)", "bandit": "LinUCB bandit", "dqn": "DQN",
}

OBJ_LABEL = {"gsk3b": "GSK3B", "jnk3": "JNK3", "qed": "QED", "sa": "SA"}

ORDER = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]

# Neutral colors: text and grid use ink, never a series color
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d8d7d2"


def apply_style() -> None:
    """Shared rcParams: thin lines, restrained grid, no top/right spines."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.8,
        "font.family": "DejaVu Sans",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def style_for(name: str) -> dict:
    """Full line parameters for one policy."""
    return {
        "color": PALETTE.get(name, INK_SECONDARY),
        "linestyle": LINESTYLE.get(name, "-"),
        "label": DISPLAY.get(name, name),
    }
