import json
import os

import numpy as np

from .evaluate import _plot_stages, _plot_stages_var, _plot_stages_var_gt


def _avg_stages(all_stages):
    """複数の stages リスト (各5段階) を平均する。None は除外。"""
    valid = [s for s in all_stages if s is not None]
    if not valid:
        return None

    num_stages = len(valid[0])
    result = []
    for si in range(num_stages):
        entries = [s[si] for s in valid]

        # mean: スカラーまたはベクトル
        means = [e["mean"] for e in entries if e["mean"] is not None]
        if not means:
            avg_mean = None
        elif isinstance(means[0], list):
            avg_mean = [round(float(np.mean([m[d] for m in means])), 6)
                        for d in range(len(means[0]))]
        else:
            avg_mean = round(float(np.mean(means)), 6)

        # var: スカラー
        vars_ = [e["var"] for e in entries if e["var"] is not None]
        avg_var = round(float(np.mean(vars_)), 6) if vars_ else None

        # count: 合計
        total_count = sum(e["count"] for e in entries)

        result.append({"mean": avg_mean, "var": avg_var, "count": total_count})
    return result


def aggregate_evaluate(
    dataset_names: list[str],
    model_name: str = "Qwen2.5-3B-Instruct",
    method: str = "cot",
    input_dir: str = "figures",
    output_dir: str = "figures/same_ans_avg",
):
    """複数データセットの eval JSON を読み込み、平均して保存・プロットする。"""

    for normalize in (True, False):
        sub = "evaluation" if normalize else "evaluation_raw"

        # --- 読み込み ---
        all_data = []
        for ds in dataset_names:
            path = os.path.join(input_dir, ds, sub,
                                f"{model_name}-{ds}-{method}-eval.json")
            with open(path, encoding="utf-8") as f:
                all_data.append(json.load(f))

        # --- question index ごとに平均 ---
        q_keys = sorted(all_data[0]["per_question"].keys(), key=int)
        avg_per_question = {}

        for qi in q_keys:
            avg_per_question[qi] = {}
            for category in ("correct", "incorrect", "all"):
                stages_gt_list = [d["per_question"][qi][category]["stages_gt"]
                                  for d in all_data]
                stages_full_list = [d["per_question"][qi][category]["stages_full"]
                                    for d in all_data]
                chains_list = [d["per_question"][qi][category]["num_chains"]
                               for d in all_data]

                avg_per_question[qi][category] = {
                    "stages_gt": _avg_stages(stages_gt_list),
                    "stages_full": _avg_stages(stages_full_list),
                    "num_chains": sum(chains_list),
                }

        result = {
            "model": model_name,
            "dataset": "same_ans_avg",
            "method": method,
            "num_questions": len(q_keys),
            "source_datasets": dataset_names,
            "per_question": avg_per_question,
        }

        # --- 保存 ---
        eval_dir = os.path.join(output_dir, sub)
        os.makedirs(eval_dir, exist_ok=True)
        save_path = os.path.join(eval_dir,
                                 f"{model_name}-same_ans_avg-{method}-eval.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"==> Aggregated evaluation saved to: {save_path}")

        # --- プロット ---
        for category in ("correct", "incorrect", "all"):
            _plot_stages(avg_per_question, category, eval_dir,
                         model_name, "same_ans_avg", method)
            _plot_stages_var(avg_per_question, category, eval_dir,
                             model_name, "same_ans_avg", method)
            _plot_stages_var_gt(avg_per_question, category, eval_dir,
                                model_name, "same_ans_avg", method)

    return result
