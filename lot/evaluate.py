import json
import os
from typing import Dict

import numpy as np

from .visualization_utils import load_landscape_data
from .visualization_utils.utils import split_list


def evaluate(
    model_name: str = "Meta-Llama-3-8B-Instruct-Lite",
    dataset_name: str = "aqua",
    method: str = "cot",
    save_root: str = "exp-data",
    output_dir: str = "figures",
) -> dict:
    """
    距離行列と思考トレースから定量指標を算出し、JSON で保存する。

    入力:
        save_root/{dataset_name}/thoughts/{model}--{method}--{dataset}--{i}.json
        save_root/{dataset_name}/distance_matrix/{model}--{method}--{dataset}--{i}.pkl

    出力:
        output_dir/evaluation/{model}-{dataset}-{method}-eval.json
    """
    # --- 入力: visualization と同じ前処理で読み込み ---
    # distance_matrices: (N, num_answers) L1正規化済み、正解列が先頭
    # num_all_thoughts_w_start_list: 質問ごとの行数 (思考ステップ数+1)
    # plot_datas: {sample_idx: {num_thoughts_each_chain, num_chains, num_all_thoughts, all_answers, answer_gt_short}}
    distance_matrices, num_all_thoughts_w_start_list, plot_datas = load_landscape_data(
        model=model_name,
        dataset=dataset_name,
        method=method,
        ROOT=save_root,
    )

    # 質問ごとに分割（各要素: (num_all_thoughts+1, num_answers)）
    per_question_matrices = split_list(num_all_thoughts_w_start_list, distance_matrices)

    print(f"==> Loaded {len(plot_datas)} questions")
    print(f"==> Distance matrix shape (concatenated): {distance_matrices.shape}")

    # --- TODO: 定量指標の算出 ---
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "method": method,
        "num_questions": len(plot_datas),
    }

    # --- 出力: JSON の保存 ---
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    save_path = os.path.join(eval_dir, f"{model_name}-{dataset_name}-{method}-eval.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"==> Evaluation saved to: {save_path}")

    return result
