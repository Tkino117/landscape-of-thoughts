import io
import os
from PIL import Image
import plotly.io as pio

from .visualization_utils import draw_landscape, draw_landscape_per_question, process_landscape_data

def plot(
    model_name: str = 'Meta-Llama-3-8B-Instruct-Lite',
    dataset_name: str = 'aqua',
    method: str = 'cot',
    plot_type: str = 'method',
    save_root: str = "exp-data",
    output_dir: str = "figures/landscape"
) -> bool:
    """
    Main function to plot landscape visualizations of reasoning traces.
    
    Args:
        model_name (str): Name of the model to use.
        dataset_name (str): Name of the dataset to use.
        method (str): The reasoning method used (e.g., 'cot', 'standard').
        plot_type (str): Type of plot ('method' or 'model').
        save_root (str): Root directory where data is stored.
        output_dir (str): Directory to save output figures.
        
    Returns:
        bool: True if plotting was successful.
    """
    print(f"==> model_name: {model_name}")
    print(f"==> dataset_name: {dataset_name}")
    print(f"==> method: {method}")
    print(f"==> plot_type: {plot_type}")
    print(f"==> save_root: {save_root}")
    print(f"==> output_dir: {output_dir}")
    
    # Create methods list
    methods = [method] if method else ['cot', 'l2m', 'mcts', 'tot']
    
    # Process data for landscape visualization
    list_all_T_2D, A_matrix_2D, list_plot_data, list_num_all_thoughts_w_start_list = process_landscape_data(
        model=model_name,
        dataset=dataset_name,
        methods=methods,
        plot_type=plot_type,
        ROOT=save_root
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save plots
    method_idx = 0
    for plot_datas, splited_T_2D, num_all_thoughts_w_start_list in zip(list_plot_data, list_all_T_2D, list_num_all_thoughts_w_start_list):
        # Create the figure
        fig = draw_landscape(
            dataset_name=dataset_name,
            plot_datas=plot_datas,
            splited_T_2D=splited_T_2D,
            A_matrix_2D=A_matrix_2D,
            num_all_thoughts_w_start_list=num_all_thoughts_w_start_list
        )
        
        # Define save path
        save_path = os.path.join(output_dir, f"{model_name}-{dataset_name}-{methods[method_idx]}.png")
        
        # Increment method index if not specific method
        if not method:
            method_idx += 1
        
        # Save the figure
        print(f"==> Saving figure to: {save_path}")
        pio.write_image(fig, save_path, scale=6, width=1500, height=350)
    
    print("==> Plotting complete!")
    return True


def plot_per_question(
    model_name: str = 'Meta-Llama-3-8B-Instruct-Lite',
    dataset_name: str = 'aqua',
    method: str = 'cot',
    plot_type: str = 'method',
    save_root: str = "exp-data",
    output_dir: str = "figures/landscape_per_question"
) -> bool:
    """質問ごとに個別のlandscape図を生成する。t-SNEの座標空間はplotと同一。"""
    print(f"==> model_name: {model_name}")
    print(f"==> dataset_name: {dataset_name}")
    print(f"==> method: {method}")
    print(f"==> save_root: {save_root}")
    print(f"==> output_dir: {output_dir}")

    methods = [method] if method else ['cot', 'l2m', 'mcts', 'tot']

    # plotと同じprocess_landscape_dataで共通の2D座標空間を構築
    list_all_T_2D, A_matrix_2D, list_plot_data, list_num_all_thoughts_w_start_list = process_landscape_data(
        model=model_name,
        dataset=dataset_name,
        methods=methods,
        plot_type=plot_type,
        ROOT=save_root
    )

    os.makedirs(output_dir, exist_ok=True)

    method_idx = 0
    for plot_datas, splited_T_2D, num_all_thoughts_w_start_list in zip(list_plot_data, list_all_T_2D, list_num_all_thoughts_w_start_list):
        # 質問ごとに個別の図を生成
        figures = draw_landscape_per_question(
            dataset_name=dataset_name,
            plot_datas=plot_datas,
            splited_T_2D=splited_T_2D,
            A_matrix_2D=A_matrix_2D,
            num_all_thoughts_w_start_list=num_all_thoughts_w_start_list
        )

        current_method = methods[method_idx]
        if not method:
            method_idx += 1

        # 個別の図を保存しつつ、画像を収集
        images = []
        for sample_idx, fig in figures.items():
            save_path = os.path.join(output_dir, f"{model_name}-{dataset_name}-{current_method}-q{sample_idx}.png")
            print(f"==> Saving figure to: {save_path}")
            img_bytes = pio.to_image(fig, format="png", scale=6, width=1500, height=350)
            with open(save_path, "wb") as f:
                f.write(img_bytes)
            images.append(Image.open(io.BytesIO(img_bytes)))

        # 全質問の図を縦に結合して1枚にまとめる
        if images:
            total_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)
            combined = Image.new("RGB", (total_width, total_height), "white")
            y_offset = 0
            for img in images:
                combined.paste(img, (0, y_offset))
                y_offset += img.height
            correct_save_path = os.path.join(output_dir, f"{model_name}-{dataset_name}-{current_method}-correct.png")
            print(f"==> Saving combined figure to: {correct_save_path}")
            combined.save(correct_save_path)

    print("==> Plotting complete!")
    return True