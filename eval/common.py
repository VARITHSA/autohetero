"""What the eval/ scripts share: one colour per entity, axis styling, figure headers, and finding data files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "out"

# Reference palette slots 1-3 (validated colour-blind safe as a set, all pairs) and slot 7 for Hybrid, which only
# appears alone in its own chart. AutoHetero's aqua is under 3:1 contrast, so every figure ships with its table.
COLORS = {"cpu": "#2a78d6", "gpu": "#eb6834", "autohetero": "#1baf7a", "hybrid": "#6250d6"}
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def newest(folder):
    """The newest CSV in ROOT/folder, or None."""
    found = sorted((ROOT / folder).glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def machine(data):
    """'CPU · GPU · torch · file' from the JSON written next to a data CSV."""
    info_path = Path(data).with_suffix(".json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return f"CPU: {info.get('cpu', '?')} · GPU: {info.get('gpu') or 'none'} · torch {info.get('torch', '?')} · {Path(data).name}"


def style(ax):
    """Recessive chrome: hairline grid, muted ticks, no top or right spine."""
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)


def header(fig, title, subtitle):
    """Title and subtitle pinned to the top edge, panels laid out below them."""
    top = lambda inches: 1 - inches / fig.get_figheight()
    fig.suptitle(title, x=0.02, y=top(0.15), ha="left", va="top", fontsize=12, color=INK)
    fig.text(0.02, top(0.42), subtitle, va="top", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, top(0.65)))


def write_csv(rows, path):
    """rows (dicts) as CSV; floats to 4 significant digits."""
    import csv

    columns = list(dict.fromkeys(c for r in rows for c in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: f"{v:.4g}" if isinstance(v, float) else v for c, v in r.items()} for r in rows)
