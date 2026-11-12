#!/usr/bin/env python3
"""Render a stable Gene Agent task-stage SVG from a JSON specification."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


COLORS = {
    "ink": "#26384e",
    "muted": "#57708c",
    "line": "#c8d8ea",
    "row_line": "#dce5ef",
    "blue": "#eef4fb",
    "green": "#eaf4ee",
    "white": "#ffffff",
    "success": "#edf7f0",
}


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def wrap(value: Any, width: int) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    return textwrap.wrap(
        text,
        width=max(8, width),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [text]


def text_element(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    weight: int = 400,
    color: str = COLORS["ink"],
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("input root must be a JSON object")
    if not str(spec.get("title", "")).strip():
        raise ValueError("title is required")
    stages = require_list(spec.get("stages"), "stages")
    if not stages:
        raise ValueError("at least one stage is required")
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"stages[{index}] must be an object")
        for key in ("id", "label", "operation"):
            if not str(stage.get(key, "")).strip():
                raise ValueError(f"stages[{index}].{key} is required")
        require_list(stage.get("phases", []), f"stages[{index}].phases")
        require_list(
            stage.get("pass_conditions", []),
            f"stages[{index}].pass_conditions",
        )
    require_list(spec.get("sources", []), "sources")
    require_list(spec.get("task_success", []), "task_success")
    return spec


def render(spec: dict[str, Any], width: int) -> str:
    stages = spec["stages"]
    margin = 52
    usable = width - 2 * margin
    overview_cols = min(6, len(stages))
    overview_rows = math.ceil(len(stages) / overview_cols)
    gap = 18
    card_width = (usable - gap * (overview_cols - 1)) / overview_cols
    card_height = 100
    title_y = 66
    subtitle_y = 104
    overview_y = 145
    detail_y = overview_y + overview_rows * (card_height + 28) + 42

    detail_heights: list[int] = []
    for stage in stages:
        phase_lines = sum(max(1, len(wrap(item, 64))) for item in stage.get("phases", []))
        pass_lines = sum(
            max(1, len(wrap(item, 38))) for item in stage.get("pass_conditions", [])
        )
        detail_heights.append(max(126, 62 + 25 * max(phase_lines, pass_lines)))

    success_lines: list[str] = []
    for item in spec.get("task_success", []):
        success_lines.extend(wrap(item, 100))
    success_height = 78 + 25 * max(1, len(success_lines))
    footer_height = 78
    height = detail_y + sum(detail_heights) + 24 + success_height + footer_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<title>{escape(str(spec["title"]))}</title>',
        text_element(margin, title_y, str(spec["title"]), size=34, weight=700),
    ]
    subtitle = str(spec.get("subtitle", "")).strip()
    if subtitle:
        parts.append(text_element(margin, subtitle_y, subtitle, size=20, color=COLORS["muted"]))

    for index, stage in enumerate(stages):
        row = index // overview_cols
        col = index % overview_cols
        x = margin + col * (card_width + gap)
        y = overview_y + row * (card_height + 28)
        fill = COLORS["green"] if index % 2 else COLORS["blue"]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{card_width:.1f}" '
            f'height="{card_height}" rx="16" fill="{fill}" '
            f'stroke="{COLORS["line"]}" stroke-width="2"/>'
        )
        parts.append(
            text_element(
                x + 18,
                y + 39,
                f'{stage["id"]}  {stage["label"]}',
                size=22,
                weight=700,
            )
        )
        parts.append(
            text_element(
                x + 18,
                y + 73,
                str(stage["operation"]),
                size=18,
                color=COLORS["muted"],
            )
        )
        if col < overview_cols - 1 and index + 1 < len(stages):
            arrow_y = y + card_height / 2
            x1 = x + card_width
            x2 = x + card_width + gap
            parts.append(
                f'<line x1="{x1:.1f}" y1="{arrow_y:.1f}" x2="{x2 - 6:.1f}" '
                f'y2="{arrow_y:.1f}" stroke="{COLORS["muted"]}" stroke-width="2"/>'
            )
            parts.append(
                f'<polygon points="{x2 - 6:.1f},{arrow_y - 5:.1f} '
                f'{x2:.1f},{arrow_y:.1f} {x2 - 6:.1f},{arrow_y + 5:.1f}" '
                f'fill="{COLORS["muted"]}"/>'
            )

    parts.extend(
        [
            text_element(margin + 18, detail_y - 14, "Stage / operation", size=20, weight=700),
            text_element(margin + 350, detail_y - 14, "Executed phases", size=20, weight=700),
            text_element(width - margin - 430, detail_y - 14, "Pass conditions", size=20, weight=700),
        ]
    )

    y = detail_y
    for index, (stage, row_height) in enumerate(zip(stages, detail_heights)):
        fill = COLORS["blue"] if index % 2 == 0 else COLORS["white"]
        parts.append(
            f'<rect x="{margin}" y="{y}" width="{usable}" height="{row_height}" '
            f'rx="14" fill="{fill}" stroke="{COLORS["row_line"]}" stroke-width="2"/>'
        )
        parts.append(
            text_element(
                margin + 18,
                y + 36,
                f'{stage["id"]}  {stage["label"]}',
                size=21,
                weight=700,
            )
        )
        parts.append(
            text_element(
                margin + 18,
                y + 67,
                str(stage["operation"]),
                size=17,
                color=COLORS["muted"],
            )
        )

        phase_y = y + 34
        for phase in stage.get("phases", []):
            for line in wrap(phase, 64):
                parts.append(text_element(margin + 350, phase_y, line, size=18))
                phase_y += 25

        pass_y = y + 34
        for condition in stage.get("pass_conditions", []):
            condition_lines = wrap(condition, 38)
            for line_index, line in enumerate(condition_lines):
                prefix = "- " if line_index == 0 else "  "
                parts.append(
                    text_element(width - margin - 430, pass_y, prefix + line, size=18)
                )
                pass_y += 25
        y += row_height

    success_y = y + 24
    parts.append(
        f'<rect x="{margin}" y="{success_y}" width="{usable}" height="{success_height}" '
        f'rx="14" fill="{COLORS["success"]}" stroke="#b0d2bd" stroke-width="2"/>'
    )
    parts.append(
        text_element(
            margin + 18,
            success_y + 34,
            "Complete task success",
            size=21,
            weight=700,
        )
    )
    line_y = success_y + 64
    for line in success_lines or ["All declared stages and final acceptance conditions pass."]:
        parts.append(text_element(margin + 250, line_y, line, size=18))
        line_y += 25

    footer_y = success_y + success_height + 32
    version = str(spec.get("version", "unversioned"))
    sources = "; ".join(str(item) for item in spec.get("sources", [])) or "not recorded"
    parts.append(
        text_element(
            margin,
            footer_y,
            f"Version: {version}",
            size=15,
            color=COLORS["muted"],
        )
    )
    for offset, line in enumerate(wrap(f"Sources: {sources}", 150)):
        parts.append(
            text_element(
                margin,
                footer_y + 23 + 20 * offset,
                line,
                size=15,
                color=COLORS["muted"],
            )
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Task-stage JSON specification")
    parser.add_argument("--output", required=True, type=Path, help="Output SVG path")
    parser.add_argument("--width", type=int, default=1600, help="SVG width in pixels")
    args = parser.parse_args()
    if args.width < 1000:
        parser.error("--width must be at least 1000")
    spec = load_spec(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(spec, args.width), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
