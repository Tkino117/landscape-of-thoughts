"""
Landscape visualization utilities for LOT.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from .utils import process_chain_points, split_list


def draw_landscape(
    dataset_name: str,  # データセット名（正解アンカー情報を知るため）
    plot_datas: Dict[int, Dict[str, Any]], # 質問ごとのメタ情報
    splited_T_2D: List[np.ndarray], # t-SNE後の2D座標を質問ごとに分割したリスト
    A_matrix_2D: np.ndarray, # 固定アンカー行列(A-E)をt-SNEした後の2D座標
    num_all_thoughts_w_start_list: List[int]  # 全結合から質問ごとに分割するための長さ
) -> go.Figure:
    """
    Draw a landscape visualization of reasoning traces.
    
    Args:
        dataset_name (str): Name of the dataset.
        plot_datas (Dict[int, Dict[str, Any]]): Data for plotting.
        splited_T_2D (List[np.ndarray]): Split T matrix in 2D.
        A_matrix_2D (np.ndarray): A matrix in 2D.
        num_all_thoughts_w_start_list (List[int]): List of number of thoughts with start.
        
    Returns:
        go.Figure: Plotly figure object.
    """
    # t-SNE後の2D座標を質問ごとのリストに分割する
    all_T_with_start_coordinate_matrix = split_list(num_all_thoughts_w_start_list, splited_T_2D)

    # 2行×5列のサブプロットを作成（列: 推論の進行度5段階、行: 上段=不正解、下段=正解）
    column_titles = [r'0-20% states', r'20-40% states', r'40-60% states', r'60-80% states', r'80-100% states']
    fig = make_subplots(rows=2, cols=5,
                        vertical_spacing=0.01, horizontal_spacing=0.005,
                        column_titles=column_titles)

    # 全質問の全チェーンを正解/不正解に振り分ける
    wrong_chain_points = []
    correct_chain_points = []
    all_start_coordinates = []
    for sample_idx, plot_data in plot_datas.items():

        num_thoughts_each_chain, num_chains, _, all_answers, answer_gt_short = plot_data.values()
        try:
            temp_distance_matrix = all_T_with_start_coordinate_matrix[sample_idx]
        except:
            print(len(all_T_with_start_coordinate_matrix), sample_idx)
        # 末尾1行はStart（質問アンカー）の座標、それ以外が思考ステップの座標
        thoughts_coordinates = np.array(temp_distance_matrix[:-1])
        start_coordinate = temp_distance_matrix[-1]
        all_start_coordinates.append(start_coordinate)

        # 各チェーンの2D座標列を、最終回答の正誤で振り分ける
        for chain_idx in range(num_chains):
            start_idx = sum(num_thoughts_each_chain[:chain_idx])
            end_idx = sum(num_thoughts_each_chain[:chain_idx+1])

            if end_idx <= start_idx:
                continue

            chain_points = thoughts_coordinates[start_idx:end_idx]

            if len(chain_points) <= 1:
                continue

            chain_data = {
                'points': chain_points,
                'start': start_coordinate
            }

            if all_answers[chain_idx] == answer_gt_short:
                correct_chain_points.append(chain_data)
            else:
                wrong_chain_points.append(chain_data)

    # 全チェーンの全ステップをフラット化し、各ステップに進行度(0→1)の重みを付ける
    wrong_x, wrong_y, wrong_weights, _, _ = process_chain_points(wrong_chain_points)
    correct_x, correct_y, correct_weights, _, _ = process_chain_points(correct_chain_points)

    # 進行度の重みを5段階に分割するための閾値を計算（パーセンタイル）
    wrong_thresholds = np.percentile(wrong_weights, [20, 40, 60, 80]) if len(wrong_weights) > 0 else np.array([0.2, 0.4, 0.6, 0.8])

    # 正解チェーンが1つもない場合のフォールバック
    if len(correct_weights) > 0:
        correct_thresholds = np.percentile(correct_weights, [20, 40, 60, 80])
    else:
        print("Warning: No correct answers found. Using default thresholds for correct answers.")
        correct_thresholds = np.array([0.2, 0.4, 0.6, 0.8])
        correct_x = np.array([])
        correct_y = np.array([])

    # 進行度の閾値でステップを5グループに分割する
    wrong_segments = []
    correct_segments = []
    for i in range(5):
        if i == 0:
            wrong_mask = wrong_weights <= wrong_thresholds[0] if len(wrong_weights) > 0 else np.array([], dtype=bool)
            correct_mask = correct_weights <= correct_thresholds[0] if len(correct_weights) > 0 else np.array([], dtype=bool)
        elif i == 4:
            wrong_mask = wrong_weights > wrong_thresholds[3] if len(wrong_weights) > 0 else np.array([], dtype=bool)
            correct_mask = correct_weights > correct_thresholds[3] if len(correct_weights) > 0 else np.array([], dtype=bool)
        else:
            if len(wrong_weights) > 0:
                wrong_mask = (wrong_weights > wrong_thresholds[i-1]) & (wrong_weights <= wrong_thresholds[i])
            else:
                wrong_mask = np.array([], dtype=bool)

            if len(correct_weights) > 0:
                correct_mask = (correct_weights > correct_thresholds[i-1]) & (correct_weights <= correct_thresholds[i])
            else:
                correct_mask = np.array([], dtype=bool)

        # マスクを適用して各グループの(x, y)座標を取り出す
        if len(wrong_weights) > 0:
            wrong_x_segment = np.array(wrong_x)[wrong_mask]
            wrong_y_segment = np.array(wrong_y)[wrong_mask]
        else:
            wrong_x_segment = np.array([])
            wrong_y_segment = np.array([])

        if len(correct_weights) > 0:
            correct_x_segment = np.array(correct_x)[correct_mask]
            correct_y_segment = np.array(correct_y)[correct_mask]
        else:
            correct_x_segment = np.array([])
            correct_y_segment = np.array([])

        wrong_segments.append((wrong_x_segment, wrong_y_segment))
        correct_segments.append((correct_x_segment, correct_y_segment))

    # 各グループを2Dヒストグラム等高線でプロット（上段=不正解(赤)、下段=正解(青)）
    for i in range(5):
        wrong_x_segment, wrong_y_segment = wrong_segments[i]
        correct_x_segment, correct_y_segment = correct_segments[i]

        # 不正解チェーンの等高線（上段）
        if len(wrong_x_segment) > 0:
            fig.add_trace(
                go.Histogram2dContour(
                    x=wrong_x_segment,
                    y=wrong_y_segment,
                    colorscale="Reds",
                    showscale=False,
                    histfunc='count',
                    contours=dict(
                        showlines=True,
                        coloring='fill'
                    ),
                    autocontour=True,
                    opacity=0.6,
                    name=f'Wrong Range {i+1}'
                ),
                row=1, col=i+1
            )
        else:
            # 点がないグループでもサブプロットの構造を維持するため空トレースを追加
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode='markers',
                    showlegend=False
                ),
                row=1, col=i+1
            )

        # 正解チェーンの等高線（下段）
        if len(correct_x_segment) > 0:
            fig.add_trace(
                go.Histogram2dContour(
                    x=correct_x_segment,
                    y=correct_y_segment,
                    colorscale="Blues",
                    showscale=False,
                    histfunc='count',
                    contours=dict(
                        showlines=True,
                        coloring='fill'
                    ),
                    autocontour=True,
                    opacity=0.6,
                    name=f'Correct Range {i+1}'
                ),
                row=2, col=i+1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode='markers',
                    showlegend=False
                ),
                row=2, col=i+1
            )

    # 回答アンカー(A〜E)のラベルをデータセットに応じて決定
    if dataset_name == "mmlu":
        labels_anchors = ['A', 'B', 'C', 'D']
    elif dataset_name == "strategyqa":
        labels_anchors = ['A', 'B']
    else:
        labels_anchors = ['A', 'B', 'C', 'D', 'E']

    # 全サブプロットに回答アンカーをプロット（★=正解(緑)、×=不正解(赤)）
    for idx, anchor_name in enumerate(labels_anchors):
        if idx == 0:  # rearrange済みなので先頭が常に正解
            marker_symbol = 'star'
            marker_color = "green"
        else:
            marker_symbol = 'x'
            marker_color = "red"

        # 上段の全5列に配置
        for col_idx in range(5):
            fig.add_trace(
                go.Scatter(
                    x=[A_matrix_2D[idx, 0]],
                    y=[A_matrix_2D[idx, 1]],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=18,
                        line_width=0.5,
                        color=marker_color,
                        opacity=0.8,
                    ),
                    showlegend=False,
                ),
                row=1, col=col_idx+1
            )

        # 下段の全5列に配置
        for col_idx in range(5):
            fig.add_trace(
                go.Scatter(
                    x=[A_matrix_2D[idx, 0]],
                    y=[A_matrix_2D[idx, 1]],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=18,
                        line_width=0.5,
                        color=marker_color,
                        opacity=0.8,
                    ),
                    showlegend=False,
                ),
                row=2, col=col_idx+1
            )

    # サブプロットのタイトルを下部に移動
    fig = move_titles_to_bottom(fig, column_titles=column_titles, y_position=-0.12)

    # 全サブプロットの軸設定（目盛り非表示、グリッド表示）
    for row in [1, 2]:
        for i in range(1, 6):
            fig.update_xaxes(
                row=row,
                col=i,
                showticklabels=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showline=False,
                linewidth=1,
                linecolor='black',
                mirror=True,
            )
            fig.update_yaxes(
                row=row,
                col=i,
                showticklabels=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showline=False,
                linewidth=1,
                linecolor='black',
                mirror=True,
            )

    # レイアウト設定（白背景、余白最小化）
    fig.update_layout(
        height=350,
        width=1500,
        margin=dict(l=5, r=5, t=5, b=37),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    template = dict(
        layout=dict(
            font_color="black",
            paper_bgcolor="white",
            plot_bgcolor="white",
            title_font_color="black",
            legend_font_color="black",

            xaxis=dict(
                title_font_color="black",
                tickfont_color="black",
                linecolor="black",
                gridcolor="black",
                zerolinecolor="black",
            ),

            yaxis=dict(
                title_font_color="black",
                tickfont_color="black",
                linecolor="black",
                gridcolor="black",
                zerolinecolor="black",
            ),

            hoverlabel=dict(
                font_color="black",
                bgcolor="white"
            ),

            annotations=[dict(font_color="black")],
            shapes=[dict(line_color="black")],

            coloraxis=dict(
                colorbar_tickfont_color="black",
                colorbar_title_font_color="black"
            ),
        )
    )

    fig.update_layout(template=template)

    return fig


def draw_landscape_per_question(
    dataset_name: str,
    plot_datas: Dict[int, Dict[str, Any]],
    splited_T_2D: List[np.ndarray],
    A_matrix_2D: np.ndarray,
    num_all_thoughts_w_start_list: List[int]
) -> Dict[int, go.Figure]:
    """質問ごとに個別のlandscape図を生成する。draw_landscapeと同じ2D座標空間を使用。"""
    # t-SNE後の2D座標を質問ごとに分割
    all_T = split_list(num_all_thoughts_w_start_list, splited_T_2D)

    figures = {}
    for i, sample_idx in enumerate(sorted(plot_datas.keys())):
        single_T = np.array(all_T[i])
        fig = draw_landscape(
            dataset_name=dataset_name,
            plot_datas={0: plot_datas[sample_idx]},
            splited_T_2D=single_T,
            A_matrix_2D=A_matrix_2D,
            num_all_thoughts_w_start_list=[num_all_thoughts_w_start_list[i]],
        )
        figures[sample_idx] = fig

    return figures


def move_titles_to_bottom(fig, column_titles, y_position=-0.12):
    """
    Move column titles to the bottom of the figure.
    
    Args:
        fig (go.Figure): Plotly figure object.
        column_titles (List[str]): List of column titles.
        y_position (float): Y position for the titles.
        
    Returns:
        go.Figure: Updated figure.
    """
    for i, title in enumerate(column_titles):
        fig.add_annotation(
            x=fig.layout.annotations[i].x,
            y=y_position,
            text=title,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=22),
            xanchor="center"  # Center the text horizontally
        )
    
    # Remove the original titles
    fig.update_layout(annotations=fig.layout.annotations)
    
    return fig

def rearrange_columns(matrix, k):
    """
    Rearranges the columns of a given N x C matrix, 
    placing the k-th column first and shifting the others.
    
    Parameters:
    - matrix: a numpy array of shape (N, C).
    - k: the index (0-based) of the column to move to the first position.
    
    Returns:
    - A new numpy array with columns rearranged.
    
    This function is only used in load_data()
    """
    # Determine the new order of columns
    new_order = [k] + [i for i in range(matrix.shape[1]) if i != k]
    
    # Rearrange and return the new matrix
    return matrix[:, new_order]

def process_landscape_data(
    model: str,
    dataset: str,
    models: List[str] = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Meta-Llama-3.1-8B-Instruct-Turbo", "Meta-Llama-3.1-70B-Instruct-Turbo"],
    methods: List[str] = ["cot", "l2m", "mcts", "tot"],
    plot_type: str = 'method',
    ROOT: str = "./exp-data"
) -> Tuple[List[np.ndarray], np.ndarray, List[Dict[int, Dict[str, Any]]], List[List[int]]]:
    """
    Process data for landscape visualization.
    
    Args:
        model (str): Model name.
        dataset (str): Dataset name.
        methods (List[str]): List of methods to process.
        plot_type (str): Type of plot ('method' or 'model').
        ROOT (str): Root directory for data.
        
    Returns:
        Tuple containing:
            - list_all_T_2D: List of T matrices in 2D.
            - A_matrix_2D: A matrix in 2D.
            - list_plot_data: List of plot data.
            - list_num_all_thoughts_w_start_list: List of number of thoughts with start.
    """
    from sklearn.manifold import TSNE
    
    distance_matrix_shape = []
    list_distance_matrix = []
    list_num_all_thoughts_w_start_list = []
    list_plot_data = []

    # plot_typeに応じて比較軸を変えてデータを集約する
    if plot_type == "model":
        # モデル間比較: 同一手法で複数モデルのデータを読み込む
        for model_name in models:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = load_landscape_data(model=model_name, dataset=dataset, method=methods[0], ROOT=ROOT)
            list_distance_matrix.append(distance_matries)
            list_plot_data.append(plot_datas)
            list_num_all_thoughts_w_start_list.append(num_all_thoughts_w_start_list)
            distance_matrix_shape.append(distance_matries.shape)

    elif plot_type == "dataset":
        # データセット間比較: 回答選択肢数が異なるため未実装
        raise NotImplementedError

    elif plot_type == "method":
        # 手法間比較: 同一モデルで複数手法のデータを読み込む
        for method in methods:
            distance_matries, num_all_thoughts_w_start_list, plot_datas = load_landscape_data(model=model, dataset=dataset, method=method, ROOT=ROOT)
            list_distance_matrix.append(distance_matries)
            list_plot_data.append(plot_datas)
            list_num_all_thoughts_w_start_list.append(num_all_thoughts_w_start_list)
            distance_matrix_shape.append(distance_matries.shape)
    else:
        raise NotImplementedError

    # 全データを縦に結合
    fig_data = np.concatenate(list_distance_matrix)

    # 回答アンカー用の固定距離行列を作成（各回答が互いに等距離 = 正単体の頂点）
    if dataset == "mmlu":
        target_A_matrix = np.ones((4,4)) * (1/4)
    elif dataset == "strategyqa":
        target_A_matrix = np.ones((2,2)) * (1/3)
    else:
        target_A_matrix = np.ones((5,5)) * (1/4)
    target_A_matrix[np.diag_indices(target_A_matrix.shape[0])] = 0

    # 全思考ステップ + 固定アンカー行列をまとめてt-SNEで2Dに次元削減
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    all_T_constant_A_distance_matrix = tsne.fit_transform(np.concatenate([fig_data, target_A_matrix]))

    # t-SNE結果を「思考ステップの2D座標」と「回答アンカーの2D座標」に分離
    if dataset == "mmlu":
        index = -4
    elif dataset == "strategyqa":
        index = -2
    else:
        index = -5
    all_T_2D, A_matrix_2D = all_T_constant_A_distance_matrix[:index, :], all_T_constant_A_distance_matrix[index:, :]
    # 思考ステップの2D座標を比較軸（モデルまたは手法）ごとに分割
    list_all_T_2D = split_array(distance_matrix_shape, all_T_2D)

    return list_all_T_2D, A_matrix_2D, list_plot_data, list_num_all_thoughts_w_start_list


def load_landscape_data(
    model: str,
    dataset: str,
    method: str = "cot",
    ROOT: str = "./exp-data"
) -> Tuple[np.ndarray, List[int], Dict[int, Dict[str, Any]]]:
    """
    Load data for landscape visualization.
    
    Args:
        model (str): Model name.
        dataset (str): Dataset name.
        method (str): Method name.
        ROOT (str): Root directory for data.
        
    Returns:
        Tuple containing:
            - distance_matrices: Concatenated distance matrices.
            - num_all_thoughts_w_start_list: List of number of thoughts with start.
            - plot_datas: Dictionary of plot data.
    """
    import json
    import pickle as pkl

    plot_datas = {}
    distance_matrices = []
    num_all_thoughts_w_start_list = []

    # 距離行列ディレクトリから対象ファイルを取得
    distance_matrix_dir = f'{ROOT}/{dataset}/distance_matrix/'
    if not os.path.exists(distance_matrix_dir):
        raise FileNotFoundError(f"Directory not found: {distance_matrix_dir}")

    # 指定モデル・手法に該当するファイルのみ抽出し、インデックス順にソート
    files = [f for f in os.listdir(distance_matrix_dir) if f.startswith(f"{model}--{method}--{dataset}--") and f.endswith(".pkl")]
    files.sort(key=lambda x: int(x.split("--")[-1].split(".")[0]))

    for file_name in tqdm(files, desc=f"Loading plot data for {model} {method} {dataset}"):
        sample_idx = int(file_name.split("--")[-1].split(".")[0])

        # 対応するJSONファイル（思考トレース）のパスを構築
        thoughts_file = f'{ROOT}/{dataset}/thoughts/{model}--{method}--{dataset}--{sample_idx}.json'

        if not os.path.exists(thoughts_file):
            print(f"Thoughts file not found: {thoughts_file}")
            continue

        distance_matrix_file = f'{ROOT}/{dataset}/distance_matrix/{file_name}'
        if not os.path.exists(distance_matrix_file):
            print(f"Distance matrix file not found: {distance_matrix_file}")
            continue

        # JSONから推論結果のメタ情報を読み込む
        with open(thoughts_file, 'r', encoding='utf-8') as f:
            trial_data = json.load(f)

        trial_thoughts = trial_data["trial_thoughts"]
        all_answers = [answer for _, answer, _ in trial_thoughts]
        answer_gt_short = trial_data["answer_gt_short"]

        num_thoughts_each_chain = [len(thoughts) for thoughts, _, _ in trial_thoughts]
        num_chains = len(trial_thoughts)
        num_all_thoughts = sum(num_thoughts_each_chain)

        # PKLから距離行列を読み込む
        with open(distance_matrix_file, 'rb') as f:
            distance_matrix = pkl.load(f)

        # データセットに応じてアンカーラベルと正解インデックスを決定
        if "strategyqa" in thoughts_file:
            labels_anchors = ["Start", 'A', 'B']
            gt_idx = labels_anchors.index(answer_gt_short)
        elif "mmlu" in thoughts_file:
            labels_anchors = ["Start", 'A', 'B', 'C', 'D']
            gt_idx = labels_anchors.index(answer_gt_short)
        else:
            labels_anchors = ["Start", 'A', 'B', 'C', 'D', 'E']
            gt_idx = labels_anchors.index(answer_gt_short)

        # 列数が期待値と一致しない壊れた距離行列をスキップ
        expected_dims = {
            "commonsenseqa": 6,
            "aqua": 6,
            "mmlu": 5,
            "strategyqa": 3,
            "dummy": 6,
            "quadratic": 6,
        }
        if distance_matrix.shape[1] != expected_dims.get(dataset):
            continue

        # 列0（質問との距離）を除外し、思考ステップ+質問アンカー1行だけ取り出す
        distance_matrix = distance_matrix[:num_all_thoughts+1, 1:]
        # NaN を含む行を 0 で補完（パープレキシティ計算が失敗した行）
        distance_matrix = np.nan_to_num(distance_matrix, nan=0.0)
        # 各行をL1正規化（回答への距離の合計を1にする）
        row_norms = np.linalg.norm(distance_matrix, axis=1, ord=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0  # 全要素 0 の行でゼロ除算を防止
        distance_matrix = distance_matrix / row_norms
        # 正解の列を先頭に移動（後段の処理で先頭列=正解として扱うため）
        distance_matrix = rearrange_columns(distance_matrix, gt_idx-1)

        # 描画用メタ情報を保存
        plot_datas[sample_idx] = {
            "num_thoughts_each_chain": num_thoughts_each_chain,
            "num_chains": num_chains,
            "num_all_thoughts": num_all_thoughts,
            "all_answers": all_answers,
            "answer_gt_short": answer_gt_short
        }

        distance_matrices.append(distance_matrix)
        # +1はStart（質問アンカー）の行分
        num_all_thoughts_w_start_list.append(num_all_thoughts+1)

    if len(distance_matrices) == 0:
        raise ValueError(f"No data found for {model} {method} {dataset}")

    # 全質問の距離行列を縦に結合
    distance_matrices = np.concatenate(distance_matrices)

    return distance_matrices, num_all_thoughts_w_start_list, plot_datas


def split_array(shapes, array):
    """
    Split an array according to the given shapes.
    
    Args:
        shapes (List[Tuple[int, int]]): List of shapes.
        array (np.ndarray): Array to split.
        
    Returns:
        List[np.ndarray]: List of split arrays.
    """
    result = []
    start_idx = 0
    for shape in shapes:
        end_idx = start_idx + shape[0]
        result.append(array[start_idx:end_idx])
        start_idx = end_idx
    return result 