"""Shared matplotlib styling: recessive axes, fixed categorical colours."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GENERATED = "#2a78d6"     # blue: LLM-generated rubrics / r_hat
HANDCRAFTED = "#eb6834"   # orange: handcrafted prompts
BOUND = "#1baf7a"         # aqua: theoretical bounds
REFERENCE = "#6b6b66"     # gray: reference lines (Delta, ...)
TEXT = "#3d3d3a"

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ad",
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "lines.linewidth": 2,
    "legend.frameon": False,
})
