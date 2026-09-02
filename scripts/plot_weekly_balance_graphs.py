#!/usr/bin/env python3
"""Build weekly round-10 balance plots in the style of the reference slide."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TASK_ORDERS = ["game", "game_myth", "myth_game"]
TASK_LABELS = {
    "game": "Game Only",
    "game_myth": "Game -> Myth",
    "myth_game": "Myth -> Game",
}

SIGNED_NOISE_HUES = [
    "No Noise",
    "Positive +5",
    "Positive +5 Informed",
    "Negative -5",
    "Negative -5 Informed",
]
BOOTSTRAP_HUES = ["No Noise", "Bootstrap", "Bootstrap Informed"]
POSITIVE_HUES = ["No Noise", "Positive +5", "Positive +5 Informed"]
NEGATIVE_HUES = ["No Noise", "Negative -5", "Negative -5 Informed"]

PALETTE = {
    "No Noise": "#4C72B0",
    "Noise": "#DD8452",
    "Noise (Informed)": "#55A868",
    "Positive +5": "#DD8452",
    "Positive +5 Informed": "#55A868",
    "Negative -5": "#C44E52",
    "Negative -5 Informed": "#8172B2",
    "Bootstrap": "#937860",
    "Bootstrap Informed": "#DA8BC3",
    "Deterministic Max": "#8C8C8C",
    "Deterministic Max Informed": "#64B5CD",
    "Targeted +1": "#DD8452",
    "Targeted -1": "#C44E52",
    "Targeted +2": "#F0A35E",
    "Targeted -2": "#B64B5B",
}

FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

BASELINES = {
    "gpt-5-nano": PROJECT_ROOT
    / "data/json/noise_experiments/v4_direct_provider_baseline/baseline_v4_mem3_direct/gpt-5-nano",
    "claude-sonnet-4.5": PROJECT_ROOT
    / "data/json/noise_experiments/v4_direct_provider_baseline/baseline_v4_mem3_direct/claude-sonnet-4.5",
    "gemini-3.1-pro-preview": PROJECT_ROOT / "data/json/baseline/gemini-3.1-pro-preview",
}

REFERENCE_RUNS = [
    {
        "name": "main_signed_noise_gpt5nano",
        "title": "Cumulative Balance at Round 10: gpt-5-nano (signed noise)",
        "model": "gpt-5-nano",
        "roots": [
            "data/json/noise_experiments/v4_direct_provider/noise_positive_mem3_gpt5_nano/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider/noise_negative_mem3_gpt5_nano/gpt-5-nano",
        ],
        "hues": SIGNED_NOISE_HUES,
        "add_stats": False,
    },
    {
        "name": "main_signed_noise_claude",
        "title": "Cumulative Balance at Round 10: claude-sonnet-4.5 (signed noise)",
        "model": "claude-sonnet-4.5",
        "roots": [
            "data/json/noise_experiments/v4_direct_provider/noise_positive_mem3_claude_sonnet_45/claude-sonnet-4.5",
            "data/json/noise_experiments/v4_direct_provider/noise_negative_mem3_claude_sonnet_45/claude-sonnet-4.5",
        ],
        "hues": SIGNED_NOISE_HUES,
        "add_stats": False,
    },
    {
        "name": "main_signed_noise_gemini",
        "title": "Cumulative Balance at Round 10: gemini-3.1-pro-preview (signed noise)",
        "model": "gemini-3.1-pro-preview",
        "roots": [
            "data/json/noise_experiments/v4_direct_provider/noise_positive_mem3_gemini_3_1_pro/gemini-3.1-pro-preview",
            "data/json/noise_experiments/v4_direct_provider/noise_negative_mem3_gemini_3_1_pro/gemini-3.1-pro-preview",
        ],
        "hues": SIGNED_NOISE_HUES,
        "add_stats": False,
    },
    {
        "name": "main_positive_gpt5nano",
        "title": "Cumulative Balance at Round 10: gpt-5-nano (positive +5)",
        "model": "gpt-5-nano",
        "root": "data/json/noise_experiments/v4_direct_provider/noise_positive_mem3_gpt5_nano/gpt-5-nano",
        "hues": POSITIVE_HUES,
    },
    {
        "name": "main_negative_gpt5nano",
        "title": "Cumulative Balance at Round 10: gpt-5-nano (negative -5)",
        "model": "gpt-5-nano",
        "root": "data/json/noise_experiments/v4_direct_provider/noise_negative_mem3_gpt5_nano/gpt-5-nano",
        "hues": NEGATIVE_HUES,
    },
    {
        "name": "main_bootstrap_gpt5nano",
        "title": "Cumulative Balance at Round 10: gpt-5-nano (bootstrap)",
        "model": "gpt-5-nano",
        "root": "data/json/noise_experiments/v4_direct_provider/noise_bootstrap_mem3/gpt-5-nano",
        "hues": BOOTSTRAP_HUES,
    },
    {
        "name": "main_positive_claude",
        "title": "Cumulative Balance at Round 10: claude-sonnet-4.5 (positive +5)",
        "model": "claude-sonnet-4.5",
        "root": "data/json/noise_experiments/v4_direct_provider/noise_positive_mem3_claude_sonnet_45/claude-sonnet-4.5",
        "hues": POSITIVE_HUES,
    },
    {
        "name": "main_negative_claude",
        "title": "Cumulative Balance at Round 10: claude-sonnet-4.5 (negative -5)",
        "model": "claude-sonnet-4.5",
        "root": "data/json/noise_experiments/v4_direct_provider/noise_negative_mem3_claude_sonnet_45/claude-sonnet-4.5",
        "hues": NEGATIVE_HUES,
    },
    {
        "name": "main_positive_gemini",
        "title": "Cumulative Balance at Round 10: gemini-3.1-pro-preview (positive +5)",
        "model": "gemini-3.1-pro-preview",
        "root": "data/json/noise_experiments/v4_direct_provider/noise_positive_mem3_gemini_3_1_pro/gemini-3.1-pro-preview",
        "hues": POSITIVE_HUES,
    },
    {
        "name": "main_negative_gemini",
        "title": "Cumulative Balance at Round 10: gemini-3.1-pro-preview (negative -5)",
        "model": "gemini-3.1-pro-preview",
        "root": "data/json/noise_experiments/v4_direct_provider/noise_negative_mem3_gemini_3_1_pro/gemini-3.1-pro-preview",
        "hues": NEGATIVE_HUES,
    },
    {
        "name": "main_bootstrap_gemini",
        "title": "Cumulative Balance at Round 10: gemini-3.1-pro-preview (bootstrap)",
        "model": "gemini-3.1-pro-preview",
        "root": "data/json/noise_experiments/v4_direct_provider/noise_bootstrap_mem3_gemini_3_1_pro/gemini-3.1-pro-preview",
        "hues": BOOTSTRAP_HUES,
    },
    {
        "name": "shared_context_bootstrap",
        "title": "Cumulative Balance at Round 10: shared context bootstrap",
        "model": "gpt-5-nano",
        "root": "data/json/noise_experiments/v4_direct_provider_shared_context/gpt5nano_shared_context_bootstrap/gpt-5-nano",
        "hues": BOOTSTRAP_HUES,
    },
    {
        "name": "targeted_bootstrap",
        "title": "Cumulative Balance at Round 10: targeted myth bootstrap",
        "model": "gpt-5-nano",
        "root": "data/json/noise_experiments/v4_direct_provider_targeted_bootstrap/targeted_myth_bootstrap_gpt5nano/gpt-5-nano",
        "hues": BOOTSTRAP_HUES,
    },
    {
        "name": "a1_targeted_bootstrap",
        "title": "Cumulative Balance at Round 10: A1 targeted bootstrap",
        "model": "gpt-5-nano",
        "root": "data/json/noise_experiments/v4_direct_provider_A1_targeted_bootstrap/gpt5nano_partner_myth_targeted_bootstrap/gpt-5-nano",
        "hues": BOOTSTRAP_HUES,
    },
]

KNOWN_SKIPS = [
    {
        "plot": "gpt5_5_main_sweeps",
        "status": "skipped",
        "reason": "no gpt-5.5 no-noise baseline and negative/bootstrap cells are incomplete",
        "path": "",
    },
    {
        "plot": "prompt_variants",
        "status": "skipped",
        "reason": "prompt variant runs are still partial; only the unconstrained bootstrap has more than smoke-test coverage",
        "path": "",
    },
    {
        "plot": "deterministic_max_gpt5nano",
        "status": "skipped",
        "reason": "deterministic-max task-order cells are incomplete",
        "path": "",
    },
]


def is_final_json(path: Path) -> bool:
    name = path.name
    return (
        path.suffix == ".json"
        and ".results" not in name
        and ".checkpoint" not in name
        and not name.endswith(".error.json")
    )


def final_json_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*.json") if is_final_json(p))


def load_balance_at_round(path: Path, target_round: int) -> float | None:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    game_rounds = [
        row for row in data.get("conversation_history", []) if row.get("sent") is not None
    ]
    if len(game_rounds) < target_round:
        return None

    balances = game_rounds[target_round - 1].get("balances", {})
    try:
        return (float(balances["Agent_1"]) + float(balances["Agent_2"])) / 2.0
    except (KeyError, TypeError, ValueError):
        return None


def generic_condition_label(condition_dir: str) -> str | None:
    if condition_dir == "default":
        return "No Noise"
    if condition_dir.endswith("_informed"):
        return "Noise (Informed)"
    return "Noise"


def specific_condition_label(condition_dir: str) -> str | None:
    labels = {
        "default": "No Noise",
        "no_noise": "No Noise",
        "noisy_positive_5": "Positive +5",
        "noisy_positive_5_informed": "Positive +5 Informed",
        "noisy_negative_5": "Negative -5",
        "noisy_negative_5_informed": "Negative -5 Informed",
        "noisy_bootstrap_cooperation": "Bootstrap",
        "noisy_bootstrap_cooperation_informed": "Bootstrap Informed",
        "noisy_deterministic_max": "Deterministic Max",
        "noisy_deterministic_max_informed": "Deterministic Max Informed",
        "noisy_positive_1": "Targeted +1",
        "noisy_negative_1": "Targeted -1",
        "noisy_positive_2": "Targeted +2",
        "noisy_negative_2": "Targeted -2",
    }
    return labels.get(condition_dir)


def collect_records(
    root: Path,
    target_round: int,
    *,
    plot_name: str,
    run_label: str,
    model: str,
    source: str | None = None,
    condition_mode: str = "specific",
    forced_condition: str | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    labeler = generic_condition_label if condition_mode == "generic" else specific_condition_label

    for task_order in TASK_ORDERS:
        task_root = root / task_order
        if not task_root.exists():
            continue

        if forced_condition is not None:
            condition_roots = [(task_root, forced_condition, "forced")]
        else:
            condition_roots = []
            for child in sorted(task_root.iterdir()):
                if child.is_dir():
                    condition_label = labeler(child.name)
                    if condition_label is not None:
                        condition_roots.append((child, condition_label, child.name))
            if not condition_roots:
                condition_label = labeler(task_root.name)
                if condition_label is not None:
                    condition_roots.append((task_root, condition_label, task_root.name))

        for condition_root, condition_label, condition_key in condition_roots:
            for path in final_json_files(condition_root):
                value = load_balance_at_round(path, target_round)
                if value is None:
                    continue
                records.append(
                    {
                        "plot": plot_name,
                        "run": run_label,
                        "model": model,
                        "source": source or run_label,
                        "task_order": task_order,
                        "Task Order": TASK_LABELS[task_order],
                        "condition": condition_label,
                        "condition_key": condition_key,
                        "mean_balance_r10": value,
                        "path": str(path.relative_to(PROJECT_ROOT)),
                    }
                )
    return records


def collect_baseline_records(
    baseline_root: Path,
    target_round: int,
    *,
    plot_name: str,
    run_label: str,
    model: str,
    source: str | None = None,
) -> list[dict[str, object]]:
    return collect_records(
        baseline_root,
        target_round,
        plot_name=plot_name,
        run_label=run_label,
        model=model,
        source=source,
        condition_mode="specific",
        forced_condition="No Noise",
    )


def cell_counts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=columns + ["n"])
    return df.groupby(columns, dropna=False).size().reset_index(name="n")


def is_complete_reference_plot(df: pd.DataFrame, min_n: int, hue_order: list[str]) -> bool:
    counts = cell_counts(df, ["Task Order", "condition"])
    lookup = {
        (row["Task Order"], row["condition"]): int(row["n"])
        for _, row in counts.iterrows()
    }
    for task in [TASK_LABELS[t] for t in TASK_ORDERS]:
        for condition in hue_order:
            if lookup.get((task, condition), 0) < min_n:
                return False
    return True


def compute_pairwise(df: pd.DataFrame, x_order: list[str], hue_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task in x_order:
        task_df = df[df["Task Order"] == task]
        for cond_a, cond_b in combinations(hue_order, 2):
            vals_a = task_df[task_df["condition"] == cond_a]["mean_balance_r10"].to_numpy()
            vals_b = task_df[task_df["condition"] == cond_b]["mean_balance_r10"].to_numpy()
            if len(vals_a) < 2 or len(vals_b) < 2:
                continue
            result = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            rows.append(
                {
                    "Task Order": task,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n_a": len(vals_a),
                    "n_b": len(vals_b),
                    "median_a": float(np.median(vals_a)),
                    "median_b": float(np.median(vals_b)),
                    "p_raw": float(result.pvalue),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_corrected"] = np.minimum(out["p_raw"] * len(out), 1.0)
    out["significant"] = out["p_corrected"] < 0.05
    return out


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def hex_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str | tuple[int, int, int] = "#222222",
) -> None:
    width, height = text_size(draw, text, font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def draw_rotated_label(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "#222222",
) -> None:
    probe = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    width, height = text_size(probe_draw, text, font)
    label = Image.new("RGBA", (width + 10, height + 10), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((5, 5), text, font=font, fill=fill)
    rotated = label.rotate(90, expand=True)
    image.alpha_composite(rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2))


def nice_axis_limits(values: pd.Series, top_extra_fraction: float = 0.12) -> tuple[float, float]:
    data_min = float(values.min())
    data_max = float(values.max())
    if data_min == data_max:
        data_min -= 1
        data_max += 1
    data_range = data_max - data_min
    y_min = np.floor((data_min - data_range * 0.05) / 5.0) * 5.0
    y_max = np.ceil((data_max + data_range * top_extra_fraction) / 5.0) * 5.0
    if y_max - y_min < 10:
        y_max = y_min + 10
    return float(y_min), float(y_max)


def make_y_mapper(plot_bottom: int, plot_top: int, y_min: float, y_max: float):
    def y_to_px(value: float) -> float:
        return plot_bottom - ((value - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)

    return y_to_px


def box_stats(values: np.ndarray) -> dict[str, float]:
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    inlier_values = values[(values >= lower_bound) & (values <= upper_bound)]
    if len(inlier_values) == 0:
        inlier_values = values
    return {
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "whisker_low": float(inlier_values.min()),
        "whisker_high": float(inlier_values.max()),
    }


def draw_legend(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    hue_order: list[str],
    title: str,
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
) -> None:
    swatch = 18
    row_h = 28
    pad = 10
    title_w, title_h = text_size(draw, title, title_font)
    item_widths = [text_size(draw, label, font)[0] for label in hue_order]
    width = max([title_w] + item_widths) + swatch + pad * 4
    height = title_h + len(hue_order) * row_h + pad * 3
    x, y = origin
    draw.rounded_rectangle(
        [x, y, x + width, y + height],
        radius=3,
        fill=(255, 255, 255, 235),
        outline="#DDDDDD",
        width=2,
    )
    draw.text((x + pad, y + pad), title, font=title_font, fill="#222222")
    y_cursor = y + pad * 2 + title_h
    for label in hue_order:
        color = hex_to_rgba(PALETTE.get(label, "#777777"), 255)
        draw.rectangle(
            [x + pad, y_cursor + 5, x + pad + swatch, y_cursor + 5 + swatch],
            fill=color,
            outline="#444444",
            width=1,
        )
        draw.text((x + pad * 2 + swatch, y_cursor + 3), label, font=font, fill="#222222")
        y_cursor += row_h


def plot_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    x_col: str,
    x_label: str,
    hue_order: list[str],
    x_order: list[str],
    add_stats: bool,
    legend_title: str = "Noise Condition",
) -> pd.DataFrame:
    plot_df = df[df["condition"].isin(hue_order)].copy()
    plot_df = plot_df[plot_df[x_col].isin(x_order)]
    if plot_df.empty:
        return pd.DataFrame()

    pairwise = pd.DataFrame()
    if add_stats:
        pairwise = compute_pairwise(plot_df, x_order=x_order, hue_order=hue_order)

    significant = pairwise[pairwise["significant"]] if not pairwise.empty else pd.DataFrame()
    extra_fraction = 0.12 if significant.empty else min(0.58, 0.18 + 0.07 * len(significant))

    legend_outside = len(hue_order) > 4
    legend_margin = 430 if legend_outside else 90
    image_width = max(1600, 240 + 245 * len(x_order) + 55 * len(hue_order) + (330 if legend_outside else 0))
    image_height = 950
    left = 140
    right = image_width - legend_margin
    top = 100
    bottom = image_height - 135
    plot_width = right - left
    plot_height = bottom - top

    title_font = load_font(34, bold=True)
    axis_font = load_font(24)
    tick_font = load_font(20)
    legend_font = load_font(20)
    legend_title_font = load_font(21)
    star_font = load_font(20, bold=True)

    y_min, y_max = nice_axis_limits(plot_df["mean_balance_r10"], top_extra_fraction=extra_fraction)
    y_to_px = make_y_mapper(bottom, top, y_min, y_max)

    image = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    draw.rectangle([left, top, right, bottom], outline="#DDDDDD", width=2)
    tick_step = 10 if y_max - y_min >= 30 else 5
    first_tick = int(np.ceil(y_min / tick_step) * tick_step)
    for tick in range(first_tick, int(y_max) + 1, tick_step):
        y = y_to_px(tick)
        draw.line([left, y, right, y], fill="#E2E2E2", width=2)
        label = str(tick)
        label_w, label_h = text_size(draw, label, tick_font)
        draw.text((left - label_w - 18, y - label_h / 2), label, font=tick_font, fill="#333333")

    draw_centered_text(draw, (image_width / 2, 44), title, title_font)
    draw_centered_text(draw, ((left + right) / 2, image_height - 44), x_label, axis_font)
    draw_rotated_label(
        image,
        (48, int((top + bottom) / 2)),
        "Mean Cumulative Balance (avg of both agents)",
        axis_font,
    )

    rng = np.random.default_rng(7)
    group_width = plot_width / max(len(x_order), 1)
    total_hue_width = min(group_width * 0.72, 260)
    hue_step = total_hue_width / max(len(hue_order), 1)
    box_width = min(hue_step * 0.68, 52)
    x_positions: dict[tuple[str, str], float] = {}

    # Draw boxes and whiskers first.
    for x_index, x_value in enumerate(x_order):
        group_center = left + group_width * (x_index + 0.5)
        draw_centered_text(draw, (group_center, bottom + 45), x_value, tick_font)
        for hue_index, hue_value in enumerate(hue_order):
            cell = plot_df[(plot_df[x_col] == x_value) & (plot_df["condition"] == hue_value)]
            if cell.empty:
                continue
            values = cell["mean_balance_r10"].to_numpy(dtype=float)
            stats = box_stats(values)
            x = group_center - total_hue_width / 2 + hue_step * (hue_index + 0.5)
            x_positions[(x_value, hue_value)] = x
            color = hex_to_rgba(PALETTE.get(hue_value, "#777777"), 205)
            outline = "#4A4A4A"

            y_q1 = y_to_px(stats["q1"])
            y_q3 = y_to_px(stats["q3"])
            y_med = y_to_px(stats["median"])
            y_low = y_to_px(stats["whisker_low"])
            y_high = y_to_px(stats["whisker_high"])

            draw.line([x, y_high, x, y_q3], fill=outline, width=2)
            draw.line([x, y_q1, x, y_low], fill=outline, width=2)
            cap = box_width * 0.45
            draw.line([x - cap, y_high, x + cap, y_high], fill=outline, width=2)
            draw.line([x - cap, y_low, x + cap, y_low], fill=outline, width=2)
            draw.rectangle([x - box_width / 2, y_q3, x + box_width / 2, y_q1], fill=color, outline=outline, width=2)
            draw.line([x - box_width / 2, y_med, x + box_width / 2, y_med], fill=outline, width=3)

    # Draw jittered raw observations on top.
    for x_index, x_value in enumerate(x_order):
        group_center = left + group_width * (x_index + 0.5)
        for hue_index, hue_value in enumerate(hue_order):
            cell = plot_df[(plot_df[x_col] == x_value) & (plot_df["condition"] == hue_value)]
            if cell.empty:
                continue
            x = group_center - total_hue_width / 2 + hue_step * (hue_index + 0.5)
            point_color = hex_to_rgba(PALETTE.get(hue_value, "#777777"), 155)
            jitter = max(5.0, min(box_width * 0.28, 14.0))
            for value in cell["mean_balance_r10"].to_numpy(dtype=float):
                px = x + float(rng.uniform(-jitter, jitter))
                py = y_to_px(float(value))
                r = 4.2
                draw.ellipse([px - r, py - r, px + r, py + r], fill=point_color)

    if add_stats and not significant.empty:
        bracket_height = (y_max - y_min) * 0.018
        y_cursor = float(plot_df["mean_balance_r10"].max()) + (y_max - y_min) * 0.045
        for _, row in significant.iterrows():
            task = str(row["Task Order"])
            cond_a = str(row["condition_a"])
            cond_b = str(row["condition_b"])
            x_a = x_positions.get((task, cond_a))
            x_b = x_positions.get((task, cond_b))
            if x_a is None or x_b is None:
                continue
            y = y_to_px(y_cursor)
            y_top = y_to_px(y_cursor + bracket_height)
            draw.line([x_a, y, x_a, y_top, x_b, y_top, x_b, y], fill="#222222", width=2)
            p = float(row["p_corrected"])
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            draw_centered_text(draw, ((x_a + x_b) / 2, y_top - 12), stars, star_font)
            y_cursor += (y_max - y_min) * 0.055

    legend_x = left + 12 if not legend_outside else right + 30
    legend_y = top + 12
    draw_legend(draw, (int(legend_x), int(legend_y)), hue_order, legend_title, legend_font, legend_title_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    return pairwise


def build_reference_plots(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    index_rows: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []

    for run in REFERENCE_RUNS:
        model = str(run["model"])
        baseline = BASELINES.get(model)
        roots = [str(root) for root in run.get("roots", [run.get("root")])]
        hue_order = list(run.get("hues", POSITIVE_HUES))
        add_stats = bool(run.get("add_stats", len(hue_order) <= 3))
        records: list[dict[str, object]] = []
        if baseline is not None:
            records.extend(
                collect_baseline_records(
                    baseline,
                    args.target_round,
                    plot_name=str(run["name"]),
                    run_label=str(run["title"]),
                    model=model,
                    source="Baseline",
                )
            )
        for root in roots:
            if not root:
                continue
            records.extend(
                collect_records(
                    PROJECT_ROOT / root,
                    args.target_round,
                    plot_name=str(run["name"]),
                    run_label=str(run["title"]),
                    model=model,
                    source=str(run["name"]),
                    condition_mode="specific",
                )
            )

        df = pd.DataFrame(records)
        if df.empty:
            index_rows.append(
                {
                    "plot": run["name"],
                    "status": "skipped",
                    "reason": "no usable records",
                    "path": "",
                }
            )
            continue

        all_frames.append(df)
        complete = is_complete_reference_plot(df, args.min_n, hue_order)
        if not complete and args.skip_incomplete:
            index_rows.append(
                {
                    "plot": run["name"],
                    "status": "skipped",
                    "reason": f"incomplete reference cells at min_n={args.min_n}",
                    "path": "",
                }
            )
            continue

        rel_path = Path("reference_like") / f"{run['name']}.png"
        pairwise = plot_boxplot(
            df,
            output_dir / rel_path,
            title=str(run["title"]),
            x_col="Task Order",
            x_label="Task Order",
            hue_order=hue_order,
            x_order=[TASK_LABELS[t] for t in TASK_ORDERS],
            add_stats=add_stats,
        )
        if not pairwise.empty:
            pairwise.to_csv(output_dir / "reference_like" / f"{run['name']}_pairwise.csv", index=False)
        index_rows.append(
            {
                "plot": run["name"],
                "status": "written",
                "reason": "complete" if complete else "incomplete but plotted",
                "path": str(rel_path),
            }
        )

    return index_rows, all_frames


def collect_source_roots(
    target_round: int,
    plot_name: str,
    source: str,
    roots: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in roots:
        rows.extend(
            collect_records(
                PROJECT_ROOT / root,
                target_round,
                plot_name=plot_name,
                run_label=plot_name,
                model="gpt-5-nano",
                source=source,
                condition_mode="specific",
            )
        )
    return rows


def build_controls_dataset(target_round: int) -> pd.DataFrame:
    roots_by_source = {
        "Partner": [
            "data/json/noise_experiments/v4_direct_provider_A1_no_noise/gpt5nano_partner_myth_no_noise/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_A1_partner_myth/gpt5nano_partner_myth_injection/gpt-5-nano",
        ],
        "Own": [
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_own_no_noise/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_own_positive_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_own_negative_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_own_bootstrap/gpt-5-nano",
        ],
        "Shuffled": [
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_shuffled_no_noise/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_shuffled_positive_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_shuffled_negative_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_shuffled_bootstrap/gpt-5-nano",
        ],
        "Filler": [
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_filler_no_noise/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_filler_positive_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_filler_negative_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_filler_bootstrap/gpt-5-nano",
        ],
        "Adversarial": [
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_adversarial_no_noise/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_adversarial_positive_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_adversarial_negative_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_A1_adversarial_bootstrap/gpt5nano_partner_myth_adversarial_bootstrap/gpt-5-nano",
        ],
        "Targeted": [
            "data/json/noise_experiments/v4_direct_provider_targeted_neutral_gpt5nano/targeted_myth_neutral_gpt5nano/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_targeted_positive_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_controls/gpt5nano_partner_myth_targeted_negative_5/gpt-5-nano",
            "data/json/noise_experiments/v4_direct_provider_A1_targeted_bootstrap/gpt5nano_partner_myth_targeted_bootstrap/gpt-5-nano",
        ],
    }

    rows: list[dict[str, object]] = []
    for source, roots in roots_by_source.items():
        rows.extend(collect_source_roots(target_round, "control_sources", source, roots))
    return pd.DataFrame(rows)


def build_multi_condition_plots(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    index_rows: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []

    multi_runs = [
        {
            "name": "a1_partner_myth_multi_condition",
            "title": "Cumulative Balance at Round 10: partner myth intervention",
            "roots": [
                "data/json/noise_experiments/v4_direct_provider_A1_no_noise/gpt5nano_partner_myth_no_noise/gpt-5-nano",
                "data/json/noise_experiments/v4_direct_provider_A1_partner_myth/gpt5nano_partner_myth_injection/gpt-5-nano",
            ],
            "include_baseline_game_only": True,
            "hues": ["No Noise", "Negative -5", "Positive +5", "Positive +5 Informed", "Bootstrap"],
        },
        {
            "name": "a3_forced_reasoning_multi_condition",
            "title": "Cumulative Balance at Round 10: forced reasoning",
            "roots": [
                "data/json/noise_experiments/v4_direct_provider_A3_forced_reasoning/gpt5nano_forced_reasoning/gpt-5-nano",
            ],
            "include_baseline_game_only": True,
            "hues": ["No Noise", "Negative -5", "Positive +5", "Positive +5 Informed", "Bootstrap"],
        },
        {
            "name": "a1a3_combined_multi_condition",
            "title": "Cumulative Balance at Round 10: partner myth + forced reasoning",
            "roots": [
                "data/json/noise_experiments/v4_direct_provider_A1A3_combined/gpt5nano_partner_myth_plus_reasoning/gpt-5-nano",
            ],
            "include_baseline_game_only": True,
            "hues": ["No Noise", "Negative -5", "Positive +5", "Positive +5 Informed", "Bootstrap"],
        },
        {
            "name": "targeted_k_sweep",
            "title": "Cumulative Balance at Round 10: targeted k sweep",
            "roots": [
                "data/json/noise_experiments/v4_direct_provider_targeted_neutral_gpt5nano/targeted_myth_neutral_gpt5nano/gpt-5-nano",
                "data/json/noise_experiments/v4_direct_provider_targeted_k1_gpt5nano/targeted_myth_k1_gpt5nano/gpt-5-nano",
                "data/json/noise_experiments/v4_direct_provider_targeted_k2_gpt5nano/targeted_myth_k2_gpt5nano/gpt-5-nano",
            ],
            "include_baseline_game_only": False,
            "hues": ["No Noise", "Targeted -1", "Targeted +1", "Targeted -2", "Targeted +2"],
        },
    ]

    for run in multi_runs:
        records: list[dict[str, object]] = []
        if run["include_baseline_game_only"]:
            records.extend(
                collect_baseline_records(
                    BASELINES["gpt-5-nano"],
                    args.target_round,
                    plot_name=str(run["name"]),
                    run_label=str(run["title"]),
                    model="gpt-5-nano",
                    source="Baseline",
                )
            )
        for root in run["roots"]:
            records.extend(
                collect_records(
                    PROJECT_ROOT / str(root),
                    args.target_round,
                    plot_name=str(run["name"]),
                    run_label=str(run["title"]),
                    model="gpt-5-nano",
                    source=str(run["name"]),
                    condition_mode="specific",
                )
            )

        df = pd.DataFrame(records)
        if df.empty:
            index_rows.append({"plot": run["name"], "status": "skipped", "reason": "no usable records", "path": ""})
            continue

        all_frames.append(df)
        rel_path = Path("multi_condition") / f"{run['name']}.png"
        plot_boxplot(
            df,
            output_dir / rel_path,
            title=str(run["title"]),
            x_col="Task Order",
            x_label="Task Order",
            hue_order=list(run["hues"]),
            x_order=[TASK_LABELS[t] for t in TASK_ORDERS],
            add_stats=False,
        )
        index_rows.append({"plot": run["name"], "status": "written", "reason": "multi-condition", "path": str(rel_path)})

    controls = build_controls_dataset(args.target_round)
    if not controls.empty:
        all_frames.append(controls)
        source_order = ["Partner", "Own", "Shuffled", "Filler", "Adversarial", "Targeted"]
        control_hues = [
            "No Noise",
            "Positive +5",
            "Positive +5 Informed",
            "Negative -5",
            "Negative -5 Informed",
            "Bootstrap",
            "Bootstrap Informed",
        ]
        for task_order in ["game_myth", "myth_game"]:
            task_label = TASK_LABELS[task_order]
            task_df = controls[controls["task_order"] == task_order].copy()
            rel_path = Path("controls") / f"control_sources_{task_order}.png"
            plot_boxplot(
                task_df,
                output_dir / rel_path,
                title=f"Cumulative Balance at Round 10: controls ({task_label})",
                x_col="source",
                x_label="Myth Source",
                hue_order=control_hues,
                x_order=source_order,
                add_stats=False,
            )
            index_rows.append(
                {
                    "plot": f"control_sources_{task_order}",
                    "status": "written",
                    "reason": "control source matrix",
                    "path": str(rel_path),
                }
            )

    return index_rows, all_frames


def write_index(output_dir: Path, index_rows: list[dict[str, object]], all_data: pd.DataFrame) -> None:
    counts = cell_counts(all_data, ["plot", "model", "source", "Task Order", "condition"])
    counts.to_csv(output_dir / "cell_counts.csv", index=False)
    all_data.to_csv(output_dir / "plot_data.csv", index=False)
    pd.DataFrame(index_rows).to_csv(output_dir / "plot_index.csv", index=False)

    lines = [
        "# Weekly Balance Graphs",
        "",
        "Generated round-10 cumulative balance plots in the reference boxplot/stripplot style.",
        "",
        "## Plots",
        "",
    ]
    for row in index_rows:
        path = row.get("path") or ""
        if path:
            lines.append(f"- `{path}` - {row['status']} ({row['reason']})")
        else:
            lines.append(f"- `{row['plot']}` - {row['status']} ({row['reason']})")
    lines.extend(
        [
            "",
            "## Tables",
            "",
            "- `plot_data.csv` - one row per run file used in the plots.",
            "- `cell_counts.csv` - counts per plotted cell.",
            "- `plot_index.csv` - generated and skipped plot registry.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/plots/weekly_2026_05_04_balance_graphs",
        help="Output directory relative to project root.",
    )
    parser.add_argument("--target-round", type=int, default=10)
    parser.add_argument("--min-n", type=int, default=5, help="Minimum n per cell for strict reference plots.")
    parser.add_argument(
        "--skip-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip strict reference plots with incomplete task/condition cells.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []

    reference_rows, reference_frames = build_reference_plots(args, output_dir)
    index_rows.extend(reference_rows)
    all_frames.extend(reference_frames)

    multi_rows, multi_frames = build_multi_condition_plots(args, output_dir)
    index_rows.extend(multi_rows)
    all_frames.extend(multi_frames)
    index_rows.extend(KNOWN_SKIPS)

    all_data = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    write_index(output_dir, index_rows, all_data)

    written = sum(1 for row in index_rows if row["status"] == "written")
    skipped = sum(1 for row in index_rows if row["status"] == "skipped")
    print(f"wrote={written} skipped={skipped} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
