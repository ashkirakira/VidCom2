#!/usr/bin/env python3
"""Append a question/answer page to each budget visualization PDF.

Reads the mapping file (degraded_video_ids_32f.txt), groups question blocks
by video_id, and writes new PDFs to OUT_ROOT preserving the short/medium/long
subdirectory layout of IN_ROOT.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF


MAPPING_FILE = Path("/home/gaofei/keti/vidcom2/VidCom2/degraded_video_ids_32f.txt")
IN_ROOT = Path("/home/gaofei/keti/vidcom2/VidCom2/vis_out_32f")
OUT_ROOT = Path("/home/gaofei/keti/vidcom2/VidCom2/vis_out_32f_with_q")
SUBDIRS = ("short", "medium", "long")


def parse_mapping(path: Path) -> Dict[str, List[str]]:
    """Return {video_id: [question_block_text, ...]} preserving file order."""
    out: Dict[str, List[str]] = {}
    current_video: str | None = None
    current_q_lines: List[str] = []

    def flush_question() -> None:
        # 收尾：把当前 question 块拼成字符串放进所属 video 的列表
        if current_video and current_q_lines:
            out.setdefault(current_video, []).append("\n".join(current_q_lines).rstrip())
        current_q_lines.clear()

    for raw in path.read_text().splitlines():
        if not raw.strip():
            # 空行：结束当前 question 块（但可能还有同视频的下一个 question）
            flush_question()
            continue
        if not raw.startswith(" "):
            # 顶格行 = 新视频开始
            flush_question()
            current_video = raw.strip()
            continue
        if raw.lstrip().startswith("question_id="):
            # 缩进 2 空格、以 question_id= 开头 = 新 question 块开始
            flush_question()
            current_q_lines.append(raw.strip())
        else:
            # 其余缩进行（Question/Options/A.B.C.D./Answer）属于当前 question 块
            current_q_lines.append(raw.strip())
    flush_question()
    return out


def build_question_page(pdf: fitz.Document, blocks: List[str]) -> None:
    """Insert one new page at the end of `pdf`, with all question blocks rendered."""
    # 用第一页的尺寸做新页，保持视觉一致
    first = pdf[0]
    page_w = first.rect.width
    page_h = first.rect.height
    page = pdf.new_page(width=page_w, height=page_h)

    margin_x = 36.0
    margin_y = 36.0
    available_w = page_w - 2 * margin_x

    # 字号按内容总行数自适应：行多就缩小，避免越界
    total_lines = sum(b.count("\n") + 2 for b in blocks)  # +2 给每块上下留白
    fontsize = 14.0 if total_lines <= 20 else (12.0 if total_lines <= 32 else 10.0)
    line_h = fontsize * 1.45

    y = margin_y
    for idx, block in enumerate(blocks):
        if idx > 0:
            # 块之间画一条灰色分隔线
            page.draw_line(
                (margin_x, y - line_h * 0.4),
                (page_w - margin_x, y - line_h * 0.4),
                color=(0.7, 0.7, 0.7),
                width=0.6,
            )
            y += line_h * 0.4

        rc = page.insert_textbox(
            fitz.Rect(margin_x, y, margin_x + available_w, page_h - margin_y),
            block,
            fontsize=fontsize,
            fontname="helv",
            align=fitz.TEXT_ALIGN_LEFT,
        )
        # insert_textbox 返回剩余高度；负数表示装不下，但页面已尽量塞
        consumed = block.count("\n") + 1
        y += consumed * line_h + line_h * 0.6


def main() -> None:
    mapping = parse_mapping(MAPPING_FILE)
    print(f"parsed {len(mapping)} videos, "
          f"{sum(len(v) for v in mapping.values())} question blocks")

    written = 0
    missed: List[str] = []
    for sub in SUBDIRS:
        in_dir = IN_ROOT / sub
        out_dir = OUT_ROOT / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(in_dir.glob("*.pdf")):
            video_id = src.stem
            blocks = mapping.get(video_id)
            if not blocks:
                missed.append(video_id)
                continue
            doc = fitz.open(src)
            build_question_page(doc, blocks)
            doc.save(out_dir / src.name)
            doc.close()
            written += 1

    print(f"wrote {written} PDFs to {OUT_ROOT}")
    if missed:
        print(f"MISS ({len(missed)}): {missed[:10]}{'...' if len(missed) > 10 else ''}")


if __name__ == "__main__":
    main()
