# 作業ログ

## 可視化の拡張

- **質問別プロット** (`1dc337d`, `af2f627`): `visualize_per_question` を追加。`draw_landscape` を再利用。`--plot_mode` オプション（all / per_question / both）で出力を切り替え。
  - `{model}-{dataset}-{method}-q{i}.png` — 各質問の landscape
  - `{model}-{dataset}-{method}-correct.png` — 正解 chain のみ結合
- **描画空間の統一** (`3c54981`): 質問別プロットの軸範囲を揃えた。

## データセットの追加

- **quadratic** (`38dabd8`, `3948d02`): 二次方程式のデータセットを作成（AQuA と同じスタイル）。
- **same_ans** (`81e19f5`): 同一回答のデータセットを追加。
- データセット追加手順をメモ (`b860cb8`)。

## 定量評価 (evaluate.py)

- **evaluate.py 新規作成** (`3c9d81f`): 定量指標の算出を開始。
- **正解次元方向の評価** (`605ebb3`): `stages_gt` を追加。正解次元方向のみの距離・分散を出力。
- **分散の変化プロット** (`24b0145`): 5次元空間の分散変化、正解次元方向の分散変化をプロット。
- **正規化スキップ** (`fb913aa`): `load_landscape_data` に正規化をスキップするオプションを追加。

### evaluate の出力ファイル

`{output_dir}/evaluation/` (正規化あり) / `evaluation_raw/` (正規化なし) 以下に生成：

| ファイル名 | 内容 |
|---|---|
| `{model}-{dataset}-{method}-eval.json` | 定量指標（ステージ別の距離統計、correct/incorrect 別） |
| `...-eval-{cat}.png` | 正解 anchor への距離 vs 推論進捗（cat: correct/incorrect/all） |
| `...-eval-{cat}-var.png` | 5次元空間での分散 vs 推論進捗 |
| `...-eval-{cat}-var-gt.png` | 正解次元方向の分散 vs 推論進捗 |

### plot の出力ファイル

`{output_dir}/` 以下に生成：

| ファイル名 | 内容 |
|---|---|
| `{model}-{dataset}-{method}.png` | 全質問まとめた landscape（t-SNE 2D） |
| `{model}-{dataset}-{method}-q{i}.png` | 質問別 landscape |
| `{model}-{dataset}-{method}-correct.png` | 正解 chain のみ結合表示 |
