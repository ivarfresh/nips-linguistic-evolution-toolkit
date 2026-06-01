#!/usr/bin/env python3
"""Claude trajectories split by whether agents were told about the noise.

Output:
  figures/claude_noise_awareness_trajectories.png
  figures/claude_noise_awareness_trajectories.pdf
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
JSON_ROOT = REPO_ROOT / "data" / "json" / "noise_experiments" / "v4_direct_provider"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODEL = "claude-sonnet-4.5"
SPECS = [
    ("Positive k=5 noise", "noise_positive_mem3_claude_sonnet_45", "noisy_positive_5"),
    ("Negative k=5 noise", "noise_negative_mem3_claude_sonnet_45", "noisy_negative_5"),
]
TASK_ORDER_LABELS = {
    "game": "Game only",
    "game_myth": "Game -> myth",
    "myth_game": "Myth -> game",
}
TASK_ORDER_COLORS = {
    "game": (75, 75, 75),
    "game_myth": (31, 119, 180),
    "myth_game": (214, 39, 40),
}


def load_balance_trajectory(path: Path) -> np.ndarray | None:
    """Return round-by-round mean cumulative balance across the two agents."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    values = []
    for entry in sorted(data.get("conversation_history", []), key=lambda e: e.get("round", 0)):
        balances = entry.get("balances") or {}
        a1 = balances.get("Agent_1")
        a2 = balances.get("Agent_2")
        if a1 is None or a2 is None:
            continue
        values.append(0.5 * (float(a1) + float(a2)))
    if not values:
        return None
    return np.array(values, dtype=float)


def collect():
    """Return data[noise_label][awareness][task_order] -> list[np.ndarray]."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for noise_label, experiment, base_condition in SPECS:
        for task_order in TASK_ORDER_LABELS:
            for informed in (False, True):
                condition = base_condition + ("_informed" if informed else "")
                root = JSON_ROOT / experiment / MODEL / task_order / condition
                if not root.exists():
                    continue
                for path in root.glob("*.json"):
                    name = path.name
                    if ".results" in name or ".checkpoint" in name or ".error" in name:
                        continue
                    traj = load_balance_trajectory(path)
                    if traj is None or len(traj) < 2:
                        continue
                    awareness = "Informed" if informed else "Uninformed"
                    data[noise_label][awareness][task_order].append(traj)
    return data


def mean_and_sem(trajectories: list[np.ndarray]):
    """Pad trajectories and return rounds, mean, SEM."""
    max_len = max(len(t) for t in trajectories)
    arr = np.full((len(trajectories), max_len), np.nan)
    for i, traj in enumerate(trajectories):
        arr[i, : len(traj)] = traj
    mean = np.nanmean(arr, axis=0)
    counts = np.sum(~np.isnan(arr), axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    sem = std / np.sqrt(np.maximum(counts, 1))
    return np.arange(1, max_len + 1), mean, sem


def rgba(color: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return color[0], color[1], color[2], alpha


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = xy[0] - (bbox[2] - bbox[0]) / 2
    y = xy[1] - (bbox[3] - bbox[1]) / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_right_aligned(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]), xy[1]), text, font=font, fill=fill)


def draw_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    cell: dict[str, list[np.ndarray]],
    show_ylabel: bool,
    show_xlabel: bool,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    left, top, right, bottom = rect
    grid_color = (222, 222, 222)
    axis_color = (25, 25, 25)
    text_color = (45, 45, 45)
    x_min, x_max = 1.0, 10.6
    y_min, y_max = 0.0, 77.0

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    draw.rectangle(rect, fill=(255, 255, 255))

    for y in [0, 15, 30, 45, 60, 75]:
        yy = py(y)
        draw.line([(left, yy), (right, yy)], fill=grid_color, width=2)
        draw_right_aligned(draw, (left - 14, yy - 13), f"{y}", fonts["tick"], fill=text_color)
    for x in [1, 5, 10]:
        xx = px(x)
        draw.line([(xx, top), (xx, bottom)], fill=(238, 238, 238), width=1)
        draw_centered(draw, (xx, bottom + 28), f"{x}", fonts["tick"], fill=text_color)

    draw.line([(left, bottom), (right, bottom)], fill=axis_color, width=3)
    draw.line([(left, top), (left, bottom)], fill=axis_color, width=3)

    for task_order, label in TASK_ORDER_LABELS.items():
        trajectories = cell.get(task_order, [])
        if not trajectories:
            continue
        rounds, mean, sem = mean_and_sem(trajectories)
        color = TASK_ORDER_COLORS[task_order]

        upper = [(px(float(x)), py(float(y))) for x, y in zip(rounds, mean + sem)]
        lower = [(px(float(x)), py(float(y))) for x, y in zip(rounds[::-1], (mean - sem)[::-1])]
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.polygon(upper + lower, fill=rgba(color, 45))
        image.alpha_composite(overlay)

        points = [(px(float(x)), py(float(y))) for x, y in zip(rounds, mean)]
        draw.line(points, fill=color, width=6, joint="curve")

    title_lines = title.split("\n")
    draw_centered(draw, ((left + right) / 2, top - 58), title_lines[0], fonts["panel"], fill=(0, 0, 0))
    draw_centered(draw, ((left + right) / 2, top - 22), title_lines[1], fonts["panel_sub"], fill=text_color)
    if show_ylabel:
        ylabel = "Mean cumulative balance"
        ylabel_img = Image.new("RGBA", (360, 42), (255, 255, 255, 0))
        ylabel_draw = ImageDraw.Draw(ylabel_img)
        draw_centered(ylabel_draw, (180, 21), ylabel, fonts["axis"], fill=(0, 0, 0))
        ylabel_img = ylabel_img.rotate(90, expand=True)
        image.alpha_composite(ylabel_img, (left - 118, int((top + bottom) / 2 - ylabel_img.height / 2)))
    if show_xlabel:
        draw_centered(draw, ((left + right) / 2, bottom + 74), "Round", fonts["axis"], fill=(0, 0, 0))


def main():
    data = collect()
    awareness_order = ["Uninformed", "Informed"]

    width, height = 2400, 1500
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": load_font(60, bold=True),
        "subtitle": load_font(30),
        "panel": load_font(38, bold=True),
        "panel_sub": load_font(30),
        "axis": load_font(31),
        "tick": load_font(27),
        "legend": load_font(30),
        "small": load_font(24),
    }

    draw_centered(
        draw,
        (width / 2, 68),
        "Claude Sonnet 4.5: trajectories by noise awareness condition",
        fonts["title"],
    )
    draw_centered(
        draw,
        (width / 2, 126),
        "Lines are means; bands are +/-1 SEM across runs. n=15 per task-order line in every panel.",
        fonts["subtitle"],
        fill=(70, 70, 70),
    )

    left_margin, right_margin = 190, 145
    top_margin, bottom_margin = 275, 190
    gap_x, gap_y = 105, 160
    panel_w = int((width - left_margin - right_margin - gap_x) / 2)
    panel_h = int((height - top_margin - bottom_margin - gap_y) / 2)

    for row_idx, (noise_label, _experiment, _condition) in enumerate(SPECS):
        for col_idx, awareness in enumerate(awareness_order):
            left = left_margin + col_idx * (panel_w + gap_x)
            top = top_margin + row_idx * (panel_h + gap_y)
            rect = (left, top, left + panel_w, top + panel_h)
            cell = data.get(noise_label, {}).get(awareness, {})
            draw_panel(
                image,
                draw,
                rect,
                f"{noise_label}\n{awareness}",
                cell,
                show_ylabel=col_idx == 0,
                show_xlabel=row_idx == len(SPECS) - 1,
                fonts=fonts,
            )

    legend_y = height - 65
    legend_xs = [610, 1050, 1510]
    for x, (task_order, label) in zip(legend_xs, TASK_ORDER_LABELS.items()):
        color = TASK_ORDER_COLORS[task_order]
        draw.line([(x, legend_y), (x + 95, legend_y)], fill=color, width=8)
        draw.text((x + 120, legend_y - 18), label, font=fonts["legend"], fill=(0, 0, 0))

    out_png = FIG_DIR / "claude_noise_awareness_trajectories.png"
    out_pdf = FIG_DIR / "claude_noise_awareness_trajectories.pdf"
    image.convert("RGB").save(out_png)
    image.convert("RGB").save(out_pdf, "PDF", resolution=220)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    for noise_label, _experiment, _condition in SPECS:
        for awareness in awareness_order:
            counts = {
                task_order: len(data.get(noise_label, {}).get(awareness, {}).get(task_order, []))
                for task_order in TASK_ORDER_LABELS
            }
            print(f"{noise_label} / {awareness}: {counts}")


if __name__ == "__main__":
    main()
