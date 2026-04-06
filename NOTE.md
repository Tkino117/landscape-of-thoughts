## 実行法
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --api-key "token-api-123" \
  --port 8000
  --download-dir YOUR_MODEL_PATH \

python main.py \
  --task all \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --dataset_name quadratic \
  --data_path ./lot/data/quadratic.jsonl \
  --method cot \
  --num_samples 10 \
  --start_index 0 \
  --end_index 5 \
  --plot_type method \
  --output_dir figures/landscape \
  --local \
  --local_api_key token-abc123

## データセットの追加法

新しいデータセットを追加するには、以下の3ステップが必要。

### 1. データファイルの作成

`lot/data/` に JSON または JSONL ファイルを配置する。
aqua と同じフラット形式なら以下の構造:

```json
{"question": "問題文", "options": ["A) ...", "B) ...", "C) ...", "D) ...", "E) ..."], "rationale": "解説", "correct": "A"}
```

### 2. `lot/datasets/dataset_loader.py` の編集

2箇所に追記する:

- `DATASET_TYPES` にデータセット名とタイプを追加
- `DATASET_FIELDS` にフィールドマッピングを追加

```python
# DATASET_TYPES
'新データセット名': 'json',

# DATASET_FIELDS（aqua と同じ形式の場合）
'新データセット名': {
    'question_field': 'question',
    'options_field': 'options',
    'answer_field': 'correct',
    'explanation_field': 'rationale',
},
```

### 3. `lot/datasets/prompt.py` の編集

2箇所に追記する:

- `DATASET_PROMPTS` に各推論手法 (cot, zero-shot-cot, l2m, mcts, tot) のプロンプトテンプレートを追加
  - cot, l2m: few-shot 例題 + `{question}` プレースホルダー
  - zero-shot-cot, mcts, tot: `{question}` プレースホルダーのみ（例題不要）
- `DATASET_PATTERNS` に回答抽出用の正規表現を追加（例: `r'A|B|C|D|E'`）

### 注意事項

- 選択肢数が既存と異なる場合は `lot/visualization_utils/landscape.py` の `expected_dims` への追記も必要
- 選択肢が5つ (A-E) なら `landscape.py` の t-SNE アンカー行列は `else` ブランチがそのまま使える
- 選択肢が4つや2つの場合はアンカー行列の分岐 (480行目付近) も要確認