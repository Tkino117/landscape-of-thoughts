import json
import os

import matplotlib.pyplot as plt
import numpy as np

from .visualization_utils import load_landscape_data
from .visualization_utils.utils import split_list


def _extract_stages_gt(rows_list, weights_list):
    """正解次元方向の距離について、パーセンタイル5段階の mean/var を返す。
    rows_list, weights_list が空なら None を返す。"""
    if not rows_list:
        return None

    rows = np.concatenate(rows_list)         # (M, num_answers)
    weights = np.concatenate(weights_list)    # (M,)
    dist_to_gt = rows[:, 0]                   # 正解アンカーへの距離

    thresholds = np.percentile(weights, [20, 40, 60, 80])
    stages = []
    for stage_idx in range(5):
        if stage_idx == 0:
            mask = weights <= thresholds[0]
        elif stage_idx == 4:
            mask = weights > thresholds[3]
        else:
            mask = (weights > thresholds[stage_idx - 1]) & (weights <= thresholds[stage_idx])

        values = dist_to_gt[mask]
        stages.append({
            "mean": round(float(np.mean(values)), 6) if len(values) > 0 else None,
            "var": round(float(np.var(values)), 6) if len(values) > 0 else None,
            "count": int(len(values)),
        })
    return stages


def _extract_stages_full(rows_list, weights_list):
    """5次元空間全体の平均ベクトルと分散スカラーを、パーセンタイル5段階で返す。
    rows_list, weights_list が空なら None を返す。"""
    if not rows_list:
        return None

    rows = np.concatenate(rows_list)         # (M, num_answers)
    weights = np.concatenate(weights_list)    # (M,)

    thresholds = np.percentile(weights, [20, 40, 60, 80])
    stages = []
    for stage_idx in range(5):
        if stage_idx == 0:
            mask = weights <= thresholds[0]
        elif stage_idx == 4:
            mask = weights > thresholds[3]
        else:
            mask = (weights > thresholds[stage_idx - 1]) & (weights <= thresholds[stage_idx])

        vecs = rows[mask]  # (n, num_answers)
        if len(vecs) > 0:
            mean_vec = np.mean(vecs, axis=0)  # (num_answers,)
            dists = np.linalg.norm(vecs - mean_vec, axis=1)  # 各点から平均までの距離
            var_scalar = float(np.var(dists))  # ユークリッド距離の分散
            stages.append({
                "mean": [round(float(v), 6) for v in mean_vec],
                "var": round(var_scalar, 6),
                "count": int(len(vecs)),
            })
        else:
            stages.append({"mean": None, "var": None, "count": 0})
    return stages


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
        output_dir/evaluation/{model}-{dataset}-{method}-eval-{correct|incorrect|all}.png
    """
    # --- 入力: visualization と同じ前処理で読み込み ---
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

    # --- 定量指標の算出 ---
    per_question = {}
    for i, (sample_idx, plot_data) in enumerate(sorted(plot_datas.items())):
        dm = np.array(per_question_matrices[i])
        thoughts_dm = dm[:-1]  # 最後の1行（Start）を除く

        num_thoughts_each_chain = plot_data["num_thoughts_each_chain"]
        all_answers = plot_data["all_answers"]
        answer_gt = plot_data["answer_gt_short"]

        # チェーンを正解/不正解に振り分け、進行度を付与
        correct_rows, correct_weights = [], []
        incorrect_rows, incorrect_weights = [], []
        all_rows, all_weights = [], []
        offset = 0
        for chain_idx, length in enumerate(num_thoughts_each_chain):
            chain_rows = thoughts_dm[offset:offset + length]
            weights = np.linspace(0, 1, length)
            all_rows.append(chain_rows)
            all_weights.append(weights)
            if all_answers[chain_idx] == answer_gt:
                correct_rows.append(chain_rows)
                correct_weights.append(weights)
            else:
                incorrect_rows.append(chain_rows)
                incorrect_weights.append(weights)
            offset += length

        num_correct = sum(1 for a in all_answers if a == answer_gt)
        num_incorrect = len(all_answers) - num_correct

        per_question[str(sample_idx)] = {
            "correct": {
                "stages_gt": _extract_stages_gt(correct_rows, correct_weights),
                "stages_full": _extract_stages_full(correct_rows, correct_weights),
                "num_chains": num_correct,
            },
            "incorrect": {
                "stages_gt": _extract_stages_gt(incorrect_rows, incorrect_weights),
                "stages_full": _extract_stages_full(incorrect_rows, incorrect_weights),
                "num_chains": num_incorrect,
            },
            "all": {
                "stages_gt": _extract_stages_gt(all_rows, all_weights),
                "stages_full": _extract_stages_full(all_rows, all_weights),
                "num_chains": len(all_answers),
            },
        }

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "method": method,
        "num_questions": len(plot_datas),
        "per_question": per_question,
    }

    # --- 出力: JSON の保存 ---
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    save_path = os.path.join(eval_dir, f"{model_name}-{dataset_name}-{method}-eval.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"==> Evaluation saved to: {save_path}")

    # --- プロット: 3種類それぞれ別ファイル ---
    for category in ("correct", "incorrect", "all"):
        _plot_stages(per_question, category, eval_dir, model_name, dataset_name, method)
        _plot_stages_var(per_question, category, eval_dir, model_name, dataset_name, method)
        _plot_stages_var_gt(per_question, category, eval_dir, model_name, dataset_name, method)

    return result


def _plot_stages(per_question: dict, category: str, eval_dir: str,
                 model_name: str, dataset_name: str, method: str):
    """各質問の5段階の mean ± std を折れ線グラフでプロットする。"""
    stage_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    x = np.arange(len(stage_labels))

    fig, ax = plt.subplots(figsize=(8, 5))

    for q_idx, q_data in sorted(per_question.items(), key=lambda kv: int(kv[0])):
        stages = q_data[category]["stages_gt"]
        if stages is None:
            continue
        means = [s["mean"] if s["mean"] is not None else 0 for s in stages]
        stds = [np.sqrt(s["var"]) if s["var"] is not None else 0 for s in stages]
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=3, label=f"Q{q_idx}")

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_xlabel("Reasoning progress")
    ax.set_ylabel("Distance to correct answer")
    ax.set_title(f"{model_name} / {dataset_name} / {method}  [{category}]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(eval_dir, f"{model_name}-{dataset_name}-{method}-eval-{category}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"==> Plot saved to: {save_path}")


def _plot_stages_var(per_question: dict, category: str, eval_dir: str,
                     model_name: str, dataset_name: str, method: str):
    """各質問の5段階の分散（点群の広がり）を折れ線グラフでプロットする。"""
    stage_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    x = np.arange(len(stage_labels))

    fig, ax = plt.subplots(figsize=(8, 5))

    for q_idx, q_data in sorted(per_question.items(), key=lambda kv: int(kv[0])):
        stages = q_data[category]["stages_full"]
        if stages is None:
            continue
        vars_ = [s["var"] if s["var"] is not None else 0 for s in stages]
        ax.plot(x, vars_, marker="o", label=f"Q{q_idx}")

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_xlabel("Reasoning progress")
    ax.set_ylabel("Variance of distance to centroid")
    ax.set_title(f"{model_name} / {dataset_name} / {method}  [{category}] (spread)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(eval_dir, f"{model_name}-{dataset_name}-{method}-eval-{category}-var.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"==> Plot saved to: {save_path}")


def _plot_stages_var_gt(per_question: dict, category: str, eval_dir: str,
                        model_name: str, dataset_name: str, method: str):
    """各質問の5段階の正解次元方向の分散を折れ線グラフでプロットする。"""
    stage_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    x = np.arange(len(stage_labels))

    fig, ax = plt.subplots(figsize=(8, 5))

    for q_idx, q_data in sorted(per_question.items(), key=lambda kv: int(kv[0])):
        stages = q_data[category]["stages_gt"]
        if stages is None:
            continue
        vars_ = [s["var"] if s["var"] is not None else 0 for s in stages]
        ax.plot(x, vars_, marker="o", label=f"Q{q_idx}")

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_xlabel("Reasoning progress")
    ax.set_ylabel("Variance (ground-truth direction)")
    ax.set_title(f"{model_name} / {dataset_name} / {method}  [{category}] (gt spread)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(eval_dir, f"{model_name}-{dataset_name}-{method}-eval-{category}-var-gt.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"==> Plot saved to: {save_path}")
