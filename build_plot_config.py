#!/usr/bin/env python3
"""Build plot_config.json from a set of <video_id>/budget.json directories.

Usage:
    python build_plot_config.py --root ./0ay2Qy3wBe8 --out ./plot_config.json
    python build_plot_config.py --root ./budget_data --out ./budget_data/plot_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_case_dirs(root: Path) -> list[Path]:
    # 如果 root 本身就是一个 case（含 budget.json），就只处理这一个；否则遍历子目录
    if (root / "budget.json").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "budget.json").exists())


def build_case(case_dir: Path, picture_root: Path) -> dict:
    # 读每个视频的 scales，写进 curves.VidCom2；Uniform 脚本会自动补 0.25
    data = json.loads((case_dir / "budget.json").read_text())
    scales = data["scales"]
    # image_dir 写成相对 picture_root 的相对路径，方便迁到服务器
    rel = case_dir.relative_to(picture_root) if case_dir != picture_root else Path(case_dir.name)
    return {
        "id": case_dir.name,
        "image_dir": str(rel),
        "curves": {"VidCom2": scales},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="包含多个 <id>/budget.json 的目录，或单个 case 目录")
    ap.add_argument("--out", type=Path, required=True, help="输出 plot_config.json 路径")
    ap.add_argument("--picture_root", type=Path, default=None, help="画图时的 picture_root；默认等于 --root 的父目录")
    args = ap.parse_args()

    case_dirs = iter_case_dirs(args.root)
    if not case_dirs:
        raise SystemExit(f"No budget.json found under {args.root}")

    # 单 case 模式下 picture_root 用 root 自身的父目录
    picture_root = args.picture_root or (args.root if args.root != case_dirs[0] else args.root.parent)
    cases = [build_case(cd, picture_root) for cd in case_dirs]

    cfg = {
        "figure": {"width": 32.0, "curve_strip_height": 0.53, "image_stretch_h": 1.2},
        "curve_order": ["Uniform", "VidCom2"],
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {args.out} with {len(cases)} case(s)")


if __name__ == "__main__":
    main()
