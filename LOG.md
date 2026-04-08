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
    - 正解次元方向の分散の変化
  - -correct-var-gt.png
    - 5次元空間での分散の変化
