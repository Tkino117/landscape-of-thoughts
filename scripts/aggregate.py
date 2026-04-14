"""same_ans1-5 の評価結果を平均して figures/same_ans_avg に保存する。

Usage:
    python scripts/aggregate.py
    python scripts/aggregate.py --model Qwen2.5-3B-Instruct --method cot
    python scripts/aggregate.py --datasets same_ans1 same_ans2 same_ans3
    python scripts/aggregate.py --input-dir figures --output-dir figures/same_ans_avg

出力: output_dir 以下に evaluation/ と evaluation_raw/ を生成
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lot.aggregate import aggregate_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        default=["same_ans1", "same_ans2", "same_ans3", "same_ans4", "same_ans5"],
        help="平均するデータセット名 (default: same_ans1-5)",
    )
    parser.add_argument("--model", default="Qwen2.5-3B-Instruct", help="モデル名")
    parser.add_argument("--method", default="cot", help="推論手法")
    parser.add_argument("--input-dir", default="figures", help="入力ディレクトリ")
    parser.add_argument("--output-dir", default="figures/same_ans_avg", help="出力ディレクトリ")
    args = parser.parse_args()

    aggregate_evaluate(
        dataset_names=args.datasets,
        model_name=args.model,
        method=args.method,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
