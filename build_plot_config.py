#!/usr/bin/env python3
"""Build plot_config.json from three method directories.

Each directory contains <video_id>/budget.json with a "scales" array.
The output JSON merges all three into a 4-curve config (Uniform + 3 methods).

Usage:
    python build_plot_config.py \
        --root_g ./videomme_64f_g \
        --root_l ./videomme_64f_l \
        --root_gl ./videomme_64f_g_l_norm \
        --picture_root ./budget_data_64f \
        --out ./plot_config_4curve.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def video_ids_in(root: Path) -> set[str]:
    """返回目录下所有含 budget.json 的子目录名（即 video ID）。"""
    return {p.parent.name for p in root.glob("*/budget.json")}


def load_scales(root: Path, video_id: str) -> list[float]:
    """从指定目录的 video_id/budget.json 中读取 scales 数组。"""
    data = json.loads((root / video_id / "budget.json").read_text())
    return data["scales"]


def build_case(
    video_id: str,
    root_g: Path,
    root_l: Path,
    root_gl: Path,
) -> dict:
    """为一个视频构建 case 配置，包含三条方法曲线。"""
    return {
        "id": video_id,
        "image_dir": video_id,
        "curves": {
            "Global Uniqueness": load_scales(root_g, video_id),
            "Local Variation": load_scales(root_l, video_id),
            "Global+Local": load_scales(root_gl, video_id),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="从三个方法目录生成 4 曲线 plot_config.json")
    ap.add_argument("--root_g", type=Path, required=True, help="Global Uniqueness 目录")
    ap.add_argument("--root_l", type=Path, required=True, help="Local Variation 目录")
    ap.add_argument("--root_gl", type=Path, required=True, help="Global+Local 目录")
    ap.add_argument("--picture_root", type=Path, default=None, help="图片帧根目录（写入 JSON 供绘图脚本使用）")
    ap.add_argument("--out", type=Path, required=True, help="输出 plot_config.json 路径")
    args = ap.parse_args()

    ids_g = video_ids_in(args.root_g)
    ids_l = video_ids_in(args.root_l)
    ids_gl = video_ids_in(args.root_gl)
    common = sorted(ids_g & ids_l & ids_gl)

    if not common:
        raise SystemExit(
            f"No common video IDs found.\n"
            f"  root_g: {len(ids_g)} videos\n"
            f"  root_l: {len(ids_l)} videos\n"
            f"  root_gl: {len(ids_gl)} videos"
        )

    cases = [build_case(vid, args.root_g, args.root_l, args.root_gl) for vid in common]

    cfg = {
        "figure": {"width": 32.0, "curve_strip_height": 0.53, "image_stretch_h": 1.2},
        "curve_order": ["Uniform", "Global Uniqueness", "Local Variation", "Global+Local"],
        "colors": {
            "Uniform": "#A9A9A9",
            "Global Uniqueness": "#6BB983",
            "Local Variation": "#E8845C",
            "Global+Local": "#5B8BD4",
        },
        "cases": cases,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {args.out} with {len(cases)} case(s)")


if __name__ == "__main__":
    main()
