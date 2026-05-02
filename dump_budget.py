#!/usr/bin/env python3
"""Dump per-frame token budget data for VidCom2 on Video-MME samples.

Loads the full Qwen3-VL model, processes specified videos with the same
preprocessing as the evaluation pipeline, runs VidCom2 scoring, and saves
per-frame budget data (scales, kept token counts) plus extracted frames.

Usage:
    python dump_budget.py --video_ids v1 v2 v3 --output_dir ./budget_data/
    python dump_budget.py --video_ids_file ids.txt --output_dir ./budget_data/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from token_compressor.vidcom2 import (
    select_low_var_channels,
    compute_gaussian_scores,
    compute_scales,
    select_outlier_indices,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils")


# ---------- 实验参数，和 examples/models/qwen3vl.sh 保持一致 ----------
PRETRAINED = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NUM_FRAMES = 32
MAX_PIXELS = 1605632
MIN_PIXELS = 256 * 28 * 28
R_RATIO = 0.25


def find_video_path(video_id: str, cache_dir: Path) -> Path:
    """在 Video-MME 缓存目录中查找视频文件，和 videomme utils 逻辑一致。"""
    data_dir = cache_dir / "data"
    for ext in ("mp4", "MP4", "mkv"):
        p = data_dir / f"{video_id}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Video not found for {video_id} in {data_dir}")


def extract_and_save_frames(
    video_tensor: torch.Tensor, output_dir: Path
) -> List[Path]:
    """把 process_vision_info 输出的 video tensor (T, C, H, W) 转成 PIL 图片并保存。

    process_vision_info 返回的 tensor 值域是 [0, 1]，不需要反归一化。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(video_tensor.shape[0]):
        frame = video_tensor[i].cpu().clamp(0, 255)
        img = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
        p = output_dir / f"frame_{i:03d}.jpg"
        img.save(p, quality=95)
        paths.append(p)
    return paths


def process_single_video(
    video_path: Path,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
) -> dict:
    """处理单个视频，返回 budget 数据。"""
    # 构造和推理时一样的 message 格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": MAX_PIXELS,
                    "min_pixels": MIN_PIXELS,
                },
                {"type": "text", "text": "placeholder"},
            ],
        }
    ]

    # 用 qwen_vl_utils 处理视频，和推理管线一致
    image_inputs, video_inputs = process_vision_info(messages)

    if video_inputs is None or len(video_inputs) == 0:
        raise RuntimeError(f"No video frames extracted from {video_path}")

    # 均匀采样到 MAX_NUM_FRAMES 帧，和 simple/qwen3_vl.py 一致
    total_frames = video_inputs[0].shape[0]
    indices = np.linspace(0, total_frames - 1, MAX_NUM_FRAMES, dtype=int)
    indices = np.unique(indices)
    if total_frames - 1 not in indices:
        indices = np.append(indices, total_frames - 1)
        indices = np.unique(indices)
    video_inputs[0] = video_inputs[0][indices]
    actual_frames = video_inputs[0].shape[0]

    # 保存原始帧图片（在 processor 处理之前）
    raw_video_tensor = video_inputs[0].clone()

    # 通过 processor 得到模型输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 跑 visual encoder 拿 video embeddings
    with torch.no_grad():
        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")

        if pixel_values_videos is None or video_grid_thw is None:
            raise RuntimeError("No video data in processor output")

        video_embeds, _ = model.model.get_video_features(
            pixel_values_videos, video_grid_thw
        )
        video_embeds = torch.cat(video_embeds, dim=0)

    # 跑 VidCom2 scoring
    merge_size = model.model.visual.spatial_merge_size
    split_sizes = (video_grid_thw.prod(-1) // merge_size**2).tolist()
    video_splits = torch.split(video_embeds, split_sizes)

    all_scales = []
    all_ks = []
    all_frame_tokens = []

    for grid, feat in zip(video_grid_thw, video_splits):
        t, h, w = grid.tolist()
        frame_tokens = (h * w) // (merge_size**2)
        all_frame_tokens.append(frame_tokens)

        if frame_tokens <= 0 or feat.numel() == 0:
            # 无法压缩，保留全部
            all_scales.append([1.0] * t)
            all_ks.append([frame_tokens] * t)
            continue

        with torch.no_grad():
            sel_feat = select_low_var_channels(feat)
            vid_score, frame_score = compute_gaussian_scores(sel_feat, frame_tokens)
            scales = compute_scales(-vid_score.mean(dim=-1), R_RATIO)
            ks = (scales * frame_tokens).round().long().clamp(min=1).tolist()

        all_scales.append(scales.cpu().tolist())
        all_ks.append(ks)

    return {
        "actual_frames": actual_frames,
        "frame_tokens": all_frame_tokens[0] if all_frame_tokens else 0,
        "scales": all_scales[0] if all_scales else [],
        "ks": all_ks[0] if all_ks else [],
        "r_ratio": R_RATIO,
        "raw_video_tensor": raw_video_tensor,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video_ids",
        nargs="+",
        default=None,
        help="Video-MME videoID list",
    )
    ap.add_argument(
        "--video_ids_file",
        type=Path,
        default=None,
        help="Text file with one videoID per line",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./budget_data"),
    )
    ap.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Video-MME cache dir. Defaults to $HF_HOME/videomme",
    )
    ap.add_argument(
        "--pretrained",
        type=str,
        default=PRETRAINED,
    )
    args = ap.parse_args()

    # 收集 video_ids
    video_ids: List[str] = []
    if args.video_ids:
        video_ids.extend(args.video_ids)
    if args.video_ids_file and args.video_ids_file.exists():
        video_ids.extend(
            line.strip()
            for line in args.video_ids_file.read_text().splitlines()
            if line.strip()
        )
    if not video_ids:
        ap.error("Provide --video_ids or --video_ids_file")

    # 确定 Video-MME 缓存目录
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        cache_dir = Path(hf_home) / "videomme"

    print(f"Video-MME cache dir: {cache_dir}")
    print(f"Processing {len(video_ids)} videos")

    # 加载模型
    print(f"Loading model: {args.pretrained}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.pretrained,
        dtype="bfloat16",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.pretrained,
        max_pixels=MAX_PIXELS,
        min_pixels=MIN_PIXELS,
    )

    # 批量处理
    for vid in tqdm(video_ids, desc="Dumping budgets"):
        try:
            video_path = find_video_path(vid, cache_dir)
        except FileNotFoundError as e:
            print(f"SKIP {vid}: {e}")
            continue

        try:
            result = process_single_video(video_path, model, processor)
        except Exception as e:
            print(f"ERROR {vid}: {e}")
            continue

        # 保存帧图片
        out_dir = args.output_dir / vid
        raw_tensor = result.pop("raw_video_tensor")
        extract_and_save_frames(raw_tensor, out_dir)

        # 保存 budget JSON
        budget_path = out_dir / "budget.json"
        with budget_path.open("w") as f:
            json.dump(result, f, indent=2)

        print(f"  {vid}: {result['actual_frames']} frames, "
              f"frame_tokens={result['frame_tokens']}, "
              f"mean_scale={np.mean(result['scales']):.3f}")

    # 生成可视化用的 config JSON
    generate_plot_config(args.output_dir, video_ids)
    print("Done.")


def generate_plot_config(output_dir: Path, video_ids: List[str]) -> None:
    """从 dump 结果生成 plot_budget_overlay_template.py 需要的 config JSON。"""
    cases = []
    for vid in video_ids:
        budget_path = output_dir / vid / "budget.json"
        if not budget_path.exists():
            continue
        with budget_path.open() as f:
            data = json.load(f)

        num_frames = len(data["scales"])
        frame_tokens = data["frame_tokens"]

        # scales 就是保留比例，直接作为 VidCom2 曲线
        cases.append(
            {
                "id": vid,
                "image_dir": vid,
                "curves": {
                    "VidCom2": data["scales"],
                    "Uniform": [data["r_ratio"]] * num_frames,
                },
            }
        )

    config = {
        "figure": {
            "width": 32.0,
            "height": 7.9 * len(cases) if cases else 7.9,
            "curve_strip_height": 0.53,
            "block_gap": -0.1,
            "image_stretch_h": 1.2,
        },
        "colors": {
            "VidCom2": "#6BB983",
            "Uniform": "#A9A9A9",
        },
        "curve_order": ["Uniform", "VidCom2"],
        "cases": cases,
    }

    config_path = output_dir / "plot_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    print(f"Plot config saved to: {config_path}")


if __name__ == "__main__":
    main()
