"""
The look of every figure in one place: colours, sizes, fonts. Nothing else in vizual/ holds a colour or a size.

    from vizual import style
    style.use()                                   # defaults
    style.use(scale=1.3, dpi=160)                 # larger figures for slides
    style.use(colors={"top": "#4a3aa7"}, cmap_score="RdYlGn")

The colour of a series follows what the series is (the kind of the mask), not its order on the figure.
"""
from dataclasses import dataclass, field, replace

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# one hue, light to dark: magnitude (heatmaps) and ordered series (the number of masked heads)
RAMP = ["#fde3df", "#fbcdc6", "#f8b3aa", "#f4978d", "#ee7a72", "#e55d59", "#d94145", "#c62a36", "#ad1c2c", "#901424",
        "#720f1d"]


@dataclass
class Theme:
    # series colours by meaning
    colors: dict = field(default_factory=lambda: {
        "top": "#2a78d6",            # the best heads of the ranking are masked
        "random": "#eb6834",         # random single heads
        "random_groups": "#1baf7a",  # random whole groups of heads
        "none": "#52514e",           # no mask: the reference
        "accent": "#2a78d6",         # a figure with one series
        "second": "#eb6834",         # the second series of a figure with two
        "muted": "#c3c2b7",          # what steps back: the rest of the heads, an empty class
    })
    cmap_score: object = "ramp"      # answer quality over length and depth; "ramp" is the ramp above, any matplotlib name works
    cmap_heads: object = "ramp"      # head scores over layer and head
    cmap_corr: object = "ramp"       # correlation matrices
    surface: str = "#fcfcfb"
    ink: str = "#0b0b0b"
    ink_secondary: str = "#52514e"
    ink_muted: str = "#898781"
    grid: str = "#e1e0d9"
    axis: str = "#c3c2b7"
    scale: float = 1.0               # multiplies every figure size
    font: float = 11.5               # base font size, points
    dpi: int = 120
    line: float = 2.0
    marker: float = 6.0


THEME = Theme()

LABELS = {"top": "лучшие головы", "random": "случайные головы", "random_groups": "случайные группы", "none": "без маски"}


def cmap(name):
    """The colormap of a theme field: "ramp" is the one-hue ramp, anything else goes to matplotlib."""
    if name == "ramp":
        return LinearSegmentedColormap.from_list("ramp", [THEME.surface] + RAMP)
    return mpl.colormaps[name] if isinstance(name, str) else name


def ordered(n):
    """n colours of the ramp for ordered series, the lightest still readable on the surface."""
    steps = RAMP[3:]
    return [steps[round(i * (len(steps) - 1) / max(n - 1, 1))] for i in range(n)]


def size(width, height):
    """Figure size in inches under the scale of the theme."""
    return (width * THEME.scale, height * THEME.scale)


def use(colors=None, **changes):
    """Sets the theme and the matplotlib defaults; colors updates single series colours, the rest are Theme fields."""
    global THEME
    THEME = replace(Theme(), **changes)
    THEME.colors.update(colors or {})
    t = THEME
    mpl.rcParams.update({
        "figure.dpi": t.dpi, "savefig.dpi": max(t.dpi, 200), "savefig.bbox": "tight",
        "figure.facecolor": t.surface, "axes.facecolor": t.surface, "savefig.facecolor": t.surface,
        "font.size": t.font, "axes.titlesize": t.font + 3, "axes.labelsize": t.font + 1, "figure.titlesize": t.font + 4,
        "xtick.labelsize": t.font, "ytick.labelsize": t.font, "legend.fontsize": t.font, "legend.title_fontsize": t.font,
        "axes.labelweight": "semibold", "axes.labelpad": 7,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": t.ink, "axes.labelcolor": t.ink, "axes.titlecolor": t.ink,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "xtick.color": t.ink_muted, "ytick.color": t.ink_muted, "xtick.labelcolor": t.ink, "ytick.labelcolor": t.ink,
        "axes.edgecolor": t.axis, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": t.grid, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": t.line, "lines.markersize": t.marker, "lines.markeredgewidth": 1.5, "lines.markeredgecolor": t.surface,
        "legend.frameon": False, "figure.constrained_layout.use": True,
    })
    return THEME
