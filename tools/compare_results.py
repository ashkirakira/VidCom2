"""Compare two lmms-eval Video-MME runs and dump the degraded questions.

Usage:
    python tools/compare_results.py BASELINE_JSONL VIDCOM2_JSONL [--out PATH]

A "degraded" question is one the baseline answered correctly but the
vidcom2 run answered incorrectly. The script writes a per-video report:
each block lists the videoID followed by the failing questions
(question_id, doc_id, target, vidcom2 prediction).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_predictions(jsonl_path: Path) -> dict[int, dict]:
    """Read an lmms-eval samples jsonl and return {doc_id: {target, pred, question_id}}."""
    out: dict[int, dict] = {}
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            doc_id = int(rec["doc_id"])
            score = rec["videomme_perception_score"]
            out[doc_id] = {
                "target": rec["target"],
                "pred": score["pred_answer"],
                "question_id": score["question_id"],
                "input": rec["input"],
            }
    return out


def find_degraded(
    baseline: dict[int, dict],
    vidcom2: dict[int, dict],
) -> list[dict]:
    """Return one record per question where baseline is correct and vidcom2 is wrong."""
    degraded: list[dict] = []
    for doc_id, b in baseline.items():
        v = vidcom2.get(doc_id)
        if v is None:
            continue
        # sanity: both runs should agree on the gold answer
        assert b["target"] == v["target"], f"target mismatch at doc_id={doc_id}"
        if b["pred"] == b["target"] and v["pred"] != v["target"]:
            degraded.append(
                {
                    "doc_id": doc_id,
                    "question_id": b["question_id"],
                    "target": b["target"],
                    "vidcom2_pred": v["pred"],
                    "input": b["input"],
                }
            )
    return degraded


def attach_video_ids(degraded: list[dict]) -> None:
    """Look up videoID from the videomme test split and add it in-place."""
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/Video-MME", split="test")
    for item in degraded:
        item["videoID"] = ds[item["doc_id"]]["videoID"]


def write_report(degraded: list[dict], out_path: Path) -> None:
    """Write one block per videoID with its failing questions."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for item in degraded:
        by_video[item["videoID"]].append(item)

    lines: list[str] = []
    for vid in sorted(by_video):
        items = sorted(by_video[vid], key=lambda x: x["doc_id"])
        lines.append(vid)
        for it in items:
            lines.append(
                f"  question_id={it['question_id']}  doc_id={it['doc_id']}  "
                f"target={it['target']}  vidcom2={it['vidcom2_pred']}"
            )
            # indent the multi-line input block for readability
            for q_line in it["input"].splitlines():
                lines.append(f"    {q_line}")
            lines.append("")  # blank line between questions
        lines.append("")  # extra blank line between videos

    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path, help="baseline samples_videomme.jsonl")
    ap.add_argument("vidcom2", type=Path, help="vidcom2 samples_videomme.jsonl")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("./degraded_video_ids_32f.txt"),
        help="output report file",
    )
    args = ap.parse_args()

    baseline = load_predictions(args.baseline)
    vidcom2 = load_predictions(args.vidcom2)
    print(f"baseline samples: {len(baseline)}, vidcom2 samples: {len(vidcom2)}")

    degraded = find_degraded(baseline, vidcom2)
    print(f"degraded questions (baseline✓, vidcom2✗): {len(degraded)}")

    attach_video_ids(degraded)
    unique_videos = len({d["videoID"] for d in degraded})
    print(f"unique videoIDs: {unique_videos}")

    write_report(degraded, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
