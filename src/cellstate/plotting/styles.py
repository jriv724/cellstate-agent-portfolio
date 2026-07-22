"""Frozen visual constants for deterministic AtlasLODO figures."""
from __future__ import annotations

from contextlib import contextmanager
import matplotlib as mpl

GROUP_A_COLOR = "#D55E00"
GROUP_B_COLOR = "#0072B2"
NEUTRAL_COLOR = "#9A9A9A"
ELIGIBLE_COLOR = "#009E73"
INELIGIBLE_COLOR = "#B8B8B8"
REFERENCE_COLOR = "#4D4D4D"

RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.hashsalt": "CAP-LODO-001-v1",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextmanager
def atlas_lodo_style():
    with mpl.rc_context(RC_PARAMS):
        yield
