# 作業ログ

## 可視化の拡張
- visualization.py
  - 問題ごとの landscape of thoughts を見れる関数を追加
  - 自動的に出力される

## データセットの追加
- quadratic
  - 二次方程式のデータセットを追加
- same_ans
  - 違う質問・同じ答えのデータセットを追加
  - 5つの質問の lot 空間を同じものとするため

## 定量評価の追加
- figures/{dataset}/evaluation に定量評価を導入
  - -correct.png
    - 正解次元方向の距離の変化
  - -correct-var.png 
    - 5次元空間の分散の変化
  - -correct-var-gt.png
    - 正解次元方向の分散の変化

## 問題文の与え方の修正
- スペース区切りだとなぜか選択肢勘違いミスが多発するので、\n 区切りに変更

## 複数データの平均を取る
- same_ans1-5 の平均を取る関数、aggregate_evaluate() を lot/aggregate.py に作成
- 実行法：
```
from lot.aggregate import aggregate_evaluate

aggregate_evaluate(
    dataset_names=["same_ans1", "same_ans2", "same_ans3", "same_ans4", "same_ans5"],
    model_name="Qwen2.5-3B-Instruct",
    method="cot",
    input_dir="figures",
    output_dir="figures/same_ans_avg",
)
```

## 出力の詳細を追えるようにする
- evaluate() に figures/{dataset}/evaluation/*detail.json の出力を追加
  - 全出力と、そのパーセンタイル、各答えへの距離がまとまっている
- scripts/show_detail.py
  - 上をみやすく整形して csv で出力

