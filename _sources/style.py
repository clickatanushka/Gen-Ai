"""Shared look for the board-replacement diagrams (matches the module decks)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon, Rectangle

plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 18, "text.color": "#1B2433",
                     "axes.edgecolor": "#1B2433", "axes.labelcolor": "#1B2433",
                     "xtick.color": "#1B2433", "ytick.color": "#1B2433"})

INK = "#1B2433"; MUTED = "#5A6474"; LINE = "#9AA3B2"
TEAL, ORANGE, CORAL, BLUE, GREEN, PURPLE = "#2A9D8F", "#E9A23B", "#E07A5F", "#5B7DB1", "#7FA86A", "#9C7A9E"
TINT = {TEAL: "#E3F3F1", ORANGE: "#FCF1DF", CORAL: "#FBE7E1", BLUE: "#E6ECF6", GREEN: "#EDF4EA", PURPLE: "#F1ECF2"}
RED = "#D1495B"

W, H = 16, 6.4   # canvas units = inches; image fills the slide body area (~11.6 x 4.65 in at 1.38x)


def canvas():
    """Blank 16 x 6.4 drawing area with data coords = inches."""
    fig = plt.figure(figsize=(W, H), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", color=BLUE, fill=None, fs=18, bold=False, tc=INK, lw=2, r=0.12, ha="center", ls="-"):
    """Rounded box centred at (x, y)."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=TINT.get(color, "white") if fill is None else fill, ec=color, lw=lw, ls=ls))
    if text:
        tx = x if ha == "center" else x - w / 2 + 0.15
        ax.text(tx, y, text, ha=ha, va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc,
                linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=2, style="-|>", rad=0, ls="-", ms=18):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, fs=18, color=INK, bold=False, ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, fontweight="bold" if bold else "normal", ha=ha, va=va,
                   linespacing=1.35, **kw)


def pill(ax, x, y, s, color=CORAL, fs=17):
    """Takeaway callout: solid coloured pill with white bold text."""
    ax.text(x, y, s, fontsize=fs, color="white", fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.5", fc=color, ec="none"))


def table(ax, x0, y0, cols, rows, cw, rh=0.46, color=BLUE, fs=15, highlight=None):
    """Simple grid table; top-left at (x0, y0). cw = list of column widths. highlight = {(r, c): colour}."""
    highlight = highlight or {}
    x = x0
    for j, (c, w) in enumerate(zip(cols, cw)):
        ax.add_patch(Rectangle((x, y0 - rh), w, rh, fc=color, ec="white", lw=1.5))
        ax.text(x + w / 2, y0 - rh / 2, c, ha="center", va="center", fontsize=fs, color="white", fontweight="bold")
        for i, row in enumerate(rows):
            fc = highlight.get((i, j), highlight.get((i, None), TINT.get(color, "white") if i % 2 == 0 else "white"))
            ax.add_patch(Rectangle((x, y0 - rh * (i + 2)), w, rh, fc=fc, ec=LINE, lw=0.8))
            ax.text(x + w / 2, y0 - rh * (i + 1.5), str(row[j]), ha="center", va="center", fontsize=fs)
        x += w
    return x, y0 - rh * (len(rows) + 1)   # bottom-right corner


def save(fig, path):
    fig.savefig(path, dpi=150, transparent=True)
    plt.close(fig)
