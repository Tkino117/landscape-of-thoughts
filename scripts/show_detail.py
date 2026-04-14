"""detail.json の内容を CSV で出力する。

Usage:
    python scripts/show_detail.py figures/same_ans1/evaluation/Qwen2.5-3B-Instruct-same_ans1-cot-detail.json
    python scripts/show_detail.py <path> -q 1
    python scripts/show_detail.py <path> -q 1 -b "60-80%"

出力: 入力と同じディレクトリに detail.csv (またはフィルタ付きの名前) を生成
"""
import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="detail.json のパス")
    parser.add_argument("--question", "-q", type=int, default=None, help="表示する質問番号")
    parser.add_argument("--bin", "-b", type=str, default=None, help="表示するビン (例: 60-80%%)")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 出力ファイル名
    base = os.path.splitext(args.path)[0]
    suffix = ""
    if args.question is not None:
        suffix += f"-q{args.question}"
    if args.bin:
        suffix += f"-{args.bin.replace('%', '')}"
    out_path = f"{base}{suffix}.csv"

    rows = []
    for q_idx, q in sorted(data["questions"].items(), key=lambda kv: int(kv[0])):
        if args.question is not None and int(q_idx) != args.question:
            continue
        for trial in q["trials"]:
            for step in trial["steps"]:
                if args.bin and step["bin"] != args.bin:
                    continue
                rows.append({
                    "question": int(q_idx),
                    "trial": trial["trial_idx"],
                    "correct": trial["correct"],
                    "num_steps": trial["num_steps"],
                    "step": step["step_idx"],
                    "bin": step["bin"],
                    "weight": step["weight"],
                    "d_gt_norm": step["dist_normalized"][0],
                    "d_gt_raw": step["dist_raw"][0],
                    "text": step["text"],
                })

    fieldnames = ["question", "trial", "correct", "num_steps", "step", "bin", "weight",
                  "d_gt_norm", "d_gt_raw", "text"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"==> {len(rows)} rows written to: {out_path}")


if __name__ == "__main__":
    main()
