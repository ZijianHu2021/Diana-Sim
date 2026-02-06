#!/usr/bin/env python3
"""
Edge-Weighted GCN Inference Example (5层 + 128维 + 边权重版本)
Inference and evaluation on test set

🔑 关键改进（相比gcn5d）:
  ✅ 在推理时传入edge_attr
  ✅ 利用Jacobian边权重进行更准确的预测
"""
import pickle
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from gcn_model import GCNNet
from gcn_dataset import build_dataset
from config_utils import load_config, resolve_data_root, resolve_output_dir, copy_config_snapshot


def get_latest_timestamp(output_dir: Path, device: str = None, data_timestamp: str = None) -> tuple[str, str]:
    """
    自动获取最新的训练时间戳
    新结构: logs/<device_data_timestamp>/<training_timestamp>/
    
    Args:
        output_dir: 模型保存目录
        device: 可选，设备类型（nmos/pmos），用于筛选特定设备的训练记录
        data_timestamp: 可选，数据时间戳，用于筛选特定数据的训练记录
    
    Returns:
        (data_tag, training_timestamp) 例如 ("nmos_20260204_130041", "20260204_140000")
    """
    logs_dir = output_dir / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError("未找到 logs/ 目录")
    
    # 查找所有数据标签目录 (device_data_timestamp)
    data_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    
    # 筛选符合条件的数据目录
    valid_data_dirs = []
    for d in data_dirs:
        name = d.name
        # 格式: device_data_timestamp (e.g., "nmos_20260204_130041")
        if '_' in name:
            parts = name.split('_')
            if len(parts) >= 2:
                dev = parts[0]
                ts = '_'.join(parts[1:])  # 时间戳可能包含多个_
                # 筛选
                if device is None or dev == device:
                    if data_timestamp is None or ts == data_timestamp:
                        valid_data_dirs.append(d)
    
    if not valid_data_dirs:
        raise FileNotFoundError(
            f"logs/ 目录下未找到符合条件的数据目录 (device={device}, data_timestamp={data_timestamp})"
        )
    
    # 在每个数据目录下查找最新的训练时间戳
    latest_training_dir = None
    latest_data_tag = None
    
    for data_dir in valid_data_dirs:
        training_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
        if training_dirs:
            latest_train = max(training_dirs, key=lambda p: p.name)
            if latest_training_dir is None or latest_train.name > latest_training_dir.name:
                latest_training_dir = latest_train
                latest_data_tag = data_dir.name
    
    if latest_training_dir is None:
        raise FileNotFoundError("未找到任何训练目录")
    
    return latest_data_tag, latest_training_dir.name


def load_model_and_scaler(output_dir: Path, data_tag: str, training_timestamp: str, model_path: Path = None, scaler_path: Path = None):
    """
    Load trained model and feature scaler
    
    Args:
        output_dir: Model save directory
        data_tag: Data directory name (e.g., "nmos_20260204_123456")
        training_timestamp: Training timestamp (e.g., "20260130_063915")
    
    Returns:
        (model, scaler) - Trained GCNNet and StandardScaler
    """
    # Load model
    model = GCNNet(in_channels=3, hidden_channels=128, out_channels=1)
    if model_path is None:
        model_path = output_dir / "logs" / data_tag / training_timestamp / "model.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()  # Set to evaluation mode
    
    # Load Scaler
    if scaler_path is None:
        scaler_path = output_dir / "logs" / data_tag / training_timestamp / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler


def load_test_indices(output_dir: Path, data_tag: str, training_timestamp: str, metrics_path: Path = None) -> list:
    """
    Load test set indices from metrics.json
    
    Args:
        output_dir: Model save directory
        data_tag: Data directory name (e.g., "nmos_20260204_123456")
        training_timestamp: Training timestamp (e.g., "20260130_063915")
        metrics_path: Optional custom path to metrics.json (for using different dataset's test_indices)
    
    Returns:
        test_indices - List of test sample indices
    """
    if metrics_path is None:
        metrics_path = output_dir / "logs" / data_tag / training_timestamp / f"metrics_{training_timestamp}.json"
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    
    test_indices = metrics_data.get("test_indices", [])
    return test_indices


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算MSE, MAE, RMSE, R²等指标"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def main(data_tag: str = None, training_timestamp: str = None, suffix: str = "", model_path: Path = None, scaler_path: Path = None, config: dict = None, config_path: Path = None, metrics_path: Path = None):
    """
    Inference and evaluation on test set
    
    Args:
        data_tag: Data directory name (e.g., "nmos_20260204_130041")
                 If None, automatically use the latest
        training_timestamp: Training timestamp (e.g., "20260130_063915")
                           If None, automatically use the latest
        suffix: Output file suffix (e.g., "nmos", "pmos")
               Will be appended as "_suffix" before file extension
    """
    
    # Configuration
    output_dir = resolve_output_dir(config) if config else Path("/home/hu/Diana-Sim/gcn5")
    
    # Determine data_root based on metrics_path or data_tag
    # Priority: metrics_path > data_tag > config
    if metrics_path:
        # Extract device and data_timestamp from metrics_path
        # Format: /home/hu/Diana-Sim/gcn5/logs/<device_data_timestamp>/<training_timestamp>/metrics_*.json
        metrics_parent = Path(metrics_path).parent.parent.name  # Get <device_data_timestamp>
        if '_' in metrics_parent:
            parts = metrics_parent.split('_')
            data_device = parts[0]  # e.g., "nmos"
            data_timestamp = '_'.join(parts[1:])  # e.g., "20260204_150830"
        else:
            # Fallback
            data_timestamp = "unknown"
            data_device = "unknown"
    elif data_tag and '_' in data_tag:
        # Extract device and data_timestamp from data_tag
        # data_tag format: "device_data_timestamp" (e.g., "nmos_20260204_130041")
        parts = data_tag.split('_')
        data_device = parts[0]  # e.g., "nmos"
        data_timestamp = '_'.join(parts[1:])  # e.g., "20260204_130041"
    else:
        # Fallback to config if both metrics_path and data_tag parsing fail
        _, data_timestamp, data_device = resolve_data_root(config) if config else (Path("/home/hu/Diana-Sim/gdata"), "unknown", "unknown")
    
    # Now construct data_root with the correct timestamp
    data_root = Path("/home/hu/Diana-Sim/gdata") / data_timestamp / data_device / "gnn_data"
    
    # If timestamp not provided, auto-detect or ask user
    if data_tag is None or training_timestamp is None:
        try:
            # 使用config中的device和data_timestamp信息筛选最新训练
            device_filter = data_device if data_device != "unknown" else None
            data_ts_filter = data_timestamp if data_timestamp != "unknown" else None
            data_tag, training_timestamp = get_latest_timestamp(output_dir, device=device_filter, data_timestamp=data_ts_filter)
            print(f"✓ Auto-detected latest: data_tag={data_tag}, training_timestamp={training_timestamp}\n")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("\nPlease enter data tag (e.g., nmos_20260204_130041) and training timestamp (e.g., 20260130_063915):")
            data_tag = input("Data tag: ").strip()
            training_timestamp = input("Training timestamp: ").strip()
    
    # 创建推理结果目录
    run_dir = output_dir / "logs" / data_tag / training_timestamp
    if not run_dir.exists():
        print(f"❌ 训练目录不存在: {run_dir}")
        return
    
    # Create separate inference subdirectory with timestamp
    suffix_str = f"_{suffix}" if suffix else ""
    inference_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    infer_dir = run_dir / inference_timestamp
    infer_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file in infer_dir
    log_path = infer_dir / f"inference_log_{inference_timestamp}{suffix_str}.txt"
    
    # 清空日志文件
    if log_path.exists():
        log_path.unlink()
    
    def log_msg(msg: str):
        """Print and save to log file simultaneously"""
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    
    log_msg("="*70)
    log_msg("Edge-Weighted GCN Inference Log (5-Layer + 128D + Edge Weights)")
    log_msg("="*70)
    log_msg("")
    
    log_msg("="*70)
    log_msg("[Step 0] Configuration")
    log_msg("="*70)
    log_msg(f"Output directory: {output_dir}")
    log_msg(f"Data directory: {data_root}")
    log_msg(f"Data tag: {data_tag}")
    log_msg(f"Training timestamp: {training_timestamp}")
    log_msg(f"Data root: {data_root}")
    log_msg(f"Data timestamp: {data_timestamp}")
    log_msg(f"Data device: {data_device}")
    if config_path:
        log_msg(f"Config: {config_path}")
    # Show which model path will be used
    actual_model_path = model_path if model_path else (output_dir / "logs" / data_tag / training_timestamp / "model.pt")
    log_msg(f"Model path: {actual_model_path}")
    # Show which scaler path will be used
    actual_scaler_path = scaler_path if scaler_path else (output_dir / "logs" / data_tag / training_timestamp / "scaler.pkl")
    log_msg(f"Scaler path: {actual_scaler_path}")
    # Show which metrics path will be used
    actual_metrics_path = metrics_path if metrics_path else (output_dir / "logs" / data_tag / training_timestamp / f"metrics_{training_timestamp}.json")
    log_msg(f"Metrics path: {actual_metrics_path}")
    log_msg("")
    
    # Step 1: Load model and Scaler
    log_msg("="*70)
    log_msg("[Step 1] Load Model and Scaler")
    log_msg("="*70)
    model, scaler = load_model_and_scaler(output_dir, data_tag, training_timestamp, model_path=model_path, scaler_path=scaler_path)
    log_msg(f"✓ Model loaded: {model_path if model_path else (output_dir / 'logs' / data_tag / training_timestamp / 'model.pt')}")
    log_msg(f"✓ Scaler loaded: {scaler_path if scaler_path else (output_dir / 'logs' / data_tag / training_timestamp / 'scaler.pkl')}")
    log_msg(f"  - Scaler mean: {scaler.mean_}")
    log_msg(f"  - Scaler std: {scaler.scale_}")
    log_msg("")
    
    # Step 2: Load test set indices
    log_msg("="*70)
    log_msg("[Step 2] Load Test Set")
    log_msg("="*70)
    test_indices = load_test_indices(output_dir, data_tag, training_timestamp, metrics_path=metrics_path if metrics_path else None)
    log_msg(f"✓ Test set indices loaded: {len(test_indices)} samples")
    log_msg(f"  - Indices: {test_indices}")
    log_msg("")
    
    # Step 3: Load dataset
    log_msg("="*70)
    log_msg("[Step 3] Load Dataset")
    log_msg("="*70)
    data_list, _ = build_dataset(data_root, normalize=True, scaler=scaler)
    log_msg(f"✓ Loaded {len(data_list)} graphs (need all data to apply training scaler)")
    log_msg(f"  - Test set has {len(test_indices)} graphs to evaluate")
    
    # 验证edge_attr存在
    sample_data = data_list[0]
    log_msg(f"✓ 数据检查:")
    log_msg(f"  - 节点特征: {sample_data.x.shape}")
    log_msg(f"  - 边索引: {sample_data.edge_index.shape}")
    log_msg(f"  - 边权重: {sample_data.edge_attr.shape} ✅")
    log_msg("")
    
    # Step 4: Perform inference
    log_msg("="*70)
    log_msg("[Step 4] Inference on Test Set (with Edge Weights)")
    log_msg("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log_msg(f"Device: {device}\n")
    
    test_predictions = []
    test_labels = []
    test_details = []
    test_current_voltages = []  # Store current node voltages
    
    with torch.no_grad():
        for idx in test_indices:
            data = data_list[idx].to(device)
            # 🔑 关键修改：传入edge_attr
            pred = model(data.x, data.edge_index, data.edge_attr)
            
            pred_flat = pred.squeeze().cpu().numpy()
            label_flat = data.y.squeeze().cpu().numpy()
            current_V = data.x[:, 0].cpu().numpy()  # First feature is current voltage
            
            test_predictions.extend(pred_flat)
            test_labels.extend(label_flat)
            test_current_voltages.extend(current_V)
            
            # Record detailed information
            for node_idx in range(len(label_flat)):
                test_details.append({
                    "graph_idx": idx,
                    "node_idx": node_idx,
                    "current_voltage": float(current_V[node_idx]),
                    "prediction": float(pred_flat[node_idx]),
                    "label": float(label_flat[node_idx]),
                    "predicted_final": float(current_V[node_idx] + pred_flat[node_idx]),
                    "actual_final": float(current_V[node_idx] + label_flat[node_idx]),
                    "error": float(label_flat[node_idx] - pred_flat[node_idx])
                })
    
    test_predictions = np.array(test_predictions)
    test_labels = np.array(test_labels)
    test_current_voltages = np.array(test_current_voltages)
    
    log_msg(f"✓ Inference completed, {len(test_predictions)} predictions\n")
    
    # Step 5: Calculate metrics
    log_msg("="*70)
    log_msg("[Step 5] Calculate Metrics")
    log_msg("="*70)
    
    metrics = calculate_metrics(test_labels, test_predictions)
    
    log_msg(f"\nTest Set Performance Metrics:")
    log_msg(f"  - MSE:  {metrics['mse']:.6f}")
    log_msg(f"  - MAE:  {metrics['mae']:.6f}")
    log_msg(f"  - RMSE: {metrics['rmse']:.6f}")
    log_msg(f"  - R²:   {metrics['r2']:.6f}")
    log_msg("")
    
    # Step 6: Detailed analysis
    log_msg("="*70)
    log_msg("[Step 6] Detailed Prediction Analysis")
    log_msg("="*70)
    
    log_msg(f"\nPrediction Statistics:")
    log_msg(f"  - Min:  {test_predictions.min():.6f} V")
    log_msg(f"  - Max:  {test_predictions.max():.6f} V")
    log_msg(f"  - Mean: {test_predictions.mean():.6f} V")
    log_msg(f"  - Std:  {test_predictions.std():.6f} V")
    
    log_msg(f"\nTrue Label Statistics:")
    log_msg(f"  - Min:  {test_labels.min():.6f} V")
    log_msg(f"  - Max:  {test_labels.max():.6f} V")
    log_msg(f"  - Mean: {test_labels.mean():.6f} V")
    log_msg(f"  - Std:  {test_labels.std():.6f} V")
    
    residuals = test_labels - test_predictions
    log_msg(f"\nResidual Analysis:")
    log_msg(f"  - Mean Residual: {residuals.mean():.6f} V")
    log_msg(f"  - Residual Std:  {residuals.std():.6f} V")
    log_msg(f"  - Max Abs Error: {np.abs(residuals).max():.6f} V")
    log_msg("")
    
    # Step 7: Per-graph analysis
    log_msg("="*70)
    log_msg("[Step 7] Per-Graph Analysis")
    log_msg("="*70)
    
    # Analyze prediction for each graph
    predictions_by_graph = {}
    labels_by_graph = {}
    
    for detail in test_details:
        graph_idx = detail['graph_idx']
        if graph_idx not in predictions_by_graph:
            predictions_by_graph[graph_idx] = []
            labels_by_graph[graph_idx] = []
        
        predictions_by_graph[graph_idx].append(detail['prediction'])
        labels_by_graph[graph_idx].append(detail['label'])
    
    log_msg(f"\nPer-Graph MAE and R²:")
    for graph_idx in sorted(predictions_by_graph.keys()):
        preds = np.array(predictions_by_graph[graph_idx])
        labels = np.array(labels_by_graph[graph_idx])
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)
        log_msg(f"  - iteration_{graph_idx}: MAE={mae:.6f}, R²={r2:.4f}")
    log_msg("")
    
    # Step 8: Prediction result classification
    log_msg("="*70)
    log_msg("[Step 8] Prediction Result Classification")
    log_msg("="*70)
    
    # Classify based on prediction values
    needs_decrease = (test_predictions > 0.1).sum()
    nearly_converged = (np.abs(test_predictions) <= 0.1).sum()
    needs_increase = (test_predictions < -0.1).sum()
    
    log_msg(f"\nNode Convergence Status Distribution:")
    log_msg(f"  - Needs decrease (>0.1V): {needs_decrease} nodes ({100*needs_decrease/len(test_predictions):.1f}%)")
    log_msg(f"  - Nearly converged ([-0.1, 0.1]V): {nearly_converged} nodes ({100*nearly_converged/len(test_predictions):.1f}%)")
    log_msg(f"  - Needs increase (<-0.1V): {needs_increase} nodes ({100*needs_increase/len(test_predictions):.1f}%)")
    log_msg("")
    
    # Step 9: Visualization
    log_msg("="*70)
    log_msg("[Step 9] Visualization")
    log_msg("="*70)
    
    # Extract node names from metadata file
    first_test_graph_idx = sorted(predictions_by_graph.keys())[0]
    first_data = data_list[first_test_graph_idx]
    
    # Try to read actual node names from labels_metadata.json
    node_names = []
    try:
        metadata_file = data_root / "labels_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if "node_names" in metadata:
                    node_names = metadata["node_names"]
                else:
                    node_names = [f"Node_{i}" for i in range(first_data.x.shape[0])]
        else:
            node_names = [f"Node_{i}" for i in range(first_data.x.shape[0])]
    except:
        # Fallback: use simple numbering
        node_names = [f"Node_{i}" for i in range(first_data.x.shape[0])]
    
    # Create visualization: PART 1 - Predictions vs True Labels (Convergence Distance)
    # 注释掉inference_curves.png的生成，不需要这张图
    # num_iterations = len(test_indices)
    # # Fix random seed for reproducible graph sampling
    # np.random.seed(42)
    # sampled_graphs = sorted(np.random.choice(list(predictions_by_graph.keys()), size=min(3, len(predictions_by_graph)), replace=False))
    # fig, axes = plt.subplots(1, len(sampled_graphs), figsize=(6 * len(sampled_graphs), 4.5))
    # fig.suptitle(f"Edge-Weighted GCN Test Set: Predictions vs True Labels [5-Layer + 128D + Edge Weights]\n" + 
    #              "Node Mapping: " + ", ".join([f"N{i}={node_names[i]}" for i in range(len(node_names))]), 
    #              fontsize=12, fontweight="bold")
    # if len(sampled_graphs) == 1:
    #     axes = [axes]
    # 
    # for plot_idx, graph_idx in enumerate(sampled_graphs):
    #     ax = axes[plot_idx]
    #     preds = np.array(predictions_by_graph[graph_idx])
    #     labels = np.array(labels_by_graph[graph_idx])
    #     
    #     # X-axis: node indices (0-9)
    #     node_indices = np.arange(len(preds))
    #     
    #     # Plot predictions and true labels
    #     ax.plot(node_indices, preds, marker="o", linestyle="-", linewidth=2, markersize=8, 
    #             label="Prediction", color="steelblue", alpha=0.7)
    #     ax.plot(node_indices, labels, marker="s", linestyle="--", linewidth=2, markersize=8, 
    #             label="True Label", color="coral", alpha=0.7)
    #     
    #     # Calculate metrics for this iteration
    #     mae = mean_absolute_error(labels, preds)
    #     r2 = r2_score(labels, preds)
    #     
    #     ax.set_xlabel("Node Index")
    #     ax.set_ylabel("Convergence Distance (V)")
    #     ax.set_title(f"Iteration {graph_idx}\n(MAE={mae:.4f}V, R²={r2:.4f})")
    #     ax.set_xticks(node_indices)
    #     ax.set_xticklabels([f"N{i}" for i in node_indices])
    #     ax.legend(loc="best")
    #     ax.grid(True, alpha=0.3)
    # 
    # plt.tight_layout()
    # plot_path1 = run_dir / f"inference_curves{suffix_str}.png"
    # plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    # log_msg(f"✓ Visualization 1 (Convergence Distance) saved: {plot_path1}")
    
    log_msg(f"✓ Skipping inference_curves.png generation (not needed)")
    
    # Aggregate per-node averages across all test samples
    node_count = len(node_names)
    avg_pred_final = []
    avg_input_voltage = []
    avg_final_label = []
    for node_idx in range(node_count):
        node_inputs = []
        node_pred_final = []
        node_final_labels = []
        for detail in test_details:
            if detail["node_idx"] == node_idx:
                current_v = detail["current_voltage"]
                pred_delta = detail["prediction"]
                label_delta = detail["label"]
                node_inputs.append(current_v)
                node_pred_final.append(current_v + pred_delta)
                node_final_labels.append(current_v + label_delta)
        avg_input_voltage.append(float(np.mean(node_inputs)))
        avg_pred_final.append(float(np.mean(node_pred_final)))
        avg_final_label.append(float(np.mean(node_final_labels)))

    node_indices = np.arange(node_count)

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    fig2.suptitle(
        "Edge-Weighted GCN Test Set: Input → Prediction → Label (Averaged per Node)\n"
        + "Node Mapping: "
        + ", ".join([f"N{i}={node_names[i]}" for i in range(node_count)]),
        fontsize=12,
        fontweight="bold",
    )
    ax2.plot(node_indices, avg_input_voltage, marker="^", linestyle=":", linewidth=2, markersize=7,
             label="Avg Input Voltage", color="gray", alpha=0.7)
    ax2.plot(node_indices, avg_pred_final, marker="o", linestyle="-", linewidth=2, markersize=8,
             label="Avg Predicted Final (V)", color="steelblue", alpha=0.7)
    ax2.plot(node_indices, avg_final_label, marker="s", linestyle="--", linewidth=2, markersize=8,
             label="Avg Final Label (V)", color="coral", alpha=0.7)

    ax2.set_xlabel("Node Index")
    ax2.set_ylabel("Voltage (V)")
    ax2.set_xticks(node_indices)
    ax2.set_xticklabels([f"N{i}" for i in node_indices])
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path2 = infer_dir / f"inference_final_voltage_{inference_timestamp}{suffix_str}.png"
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    log_msg(f"✓ Visualization (Final Voltage) saved: {plot_path2}")
    
    log_msg(f"\nNode Mapping:")
    for i, name in enumerate(node_names):
        log_msg(f"  - N{i} = {name}")
    log_msg("")
    
    # Step 10: Calculate averaged initial voltage prediction
    log_msg("="*70)
    log_msg("[Step 10] Calculate Averaged Initial Voltage Prediction")
    log_msg("="*70)
    
    # Group predictions by node index across all test samples
    node_predictions = {}
    for detail in test_details:
        node_idx = detail["node_idx"]
        if node_idx not in node_predictions:
            node_predictions[node_idx] = {
                "predicted_final_voltages": [],
                "node_name": None
            }
        node_predictions[node_idx]["predicted_final_voltages"].append(detail["predicted_final"])
        if node_predictions[node_idx]["node_name"] is None:
            # Use node name from first occurrence
            node_predictions[node_idx]["node_name"] = node_names[node_idx]
    
    # Calculate average predicted final voltage for each node
    averaged_predictions = {}
    log_msg(f"\n对所有 {len(test_indices)} 个测试样本求平均（作为初始电压预测）:")
    log_msg("-"*70)
    log_msg(f"{'节点名称':<20} {'平均预测电压(V)':<20} {'标准差(V)':<15}")
    log_msg("-"*70)
    
    for node_idx in sorted(node_predictions.keys()):
        voltages = np.array(node_predictions[node_idx]["predicted_final_voltages"])
        avg_voltage = voltages.mean()
        std_voltage = voltages.std()
        node_name = node_predictions[node_idx]["node_name"]
        
        averaged_predictions[node_name] = float(avg_voltage)
        
        log_msg(f"{node_name:<20} {avg_voltage:>19.6f} {std_voltage:>14.6f}")
    
    log_msg("-"*70)
    log_msg("")
    
    # Save averaged predictions to JSON file (for Newton solver)
    initial_guess_path = infer_dir / f"gnn_initial_guess_{inference_timestamp}{suffix_str}.json"
    initial_guess_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "based_on_training": training_timestamp,
        "model_type": "edge-weighted-gcn",
        "num_test_samples": len(test_indices),
        "description": "Averaged predicted final voltages across all test samples (Edge-Weighted GCN), to be used as initial guess for Newton iteration",
        "initial_voltages": averaged_predictions
    }
    
    with open(initial_guess_path, "w") as f:
        json.dump(initial_guess_data, f, indent=2)
    
    log_msg(f"✓ 初始电压预测已保存到: {initial_guess_path}")
    log_msg(f"  - 格式: JSON，包含节点名称到平均预测电压的映射")
    log_msg(f"  - 模型: Edge-Weighted GCN (利用Jacobian边权重)")
    log_msg(f"  - 用途: 作为牛顿迭代的初始猜测，预期可将迭代次数降低")
    log_msg("")
    
    log_msg("="*70)
    if config and config.get("logging", {}).get("save_config", True) and config_path:
        snapshot_name = f"config_used_inference_{inference_timestamp}{suffix_str}.yaml"
        snapshot_path = copy_config_snapshot(config_path, infer_dir, snapshot_name)
        log_msg(f"✓ Config snapshot saved: {snapshot_path}")

    log_msg("✅ Inference Complete!")
    log_msg("="*70)
    log_msg(f"\nLog saved to: {log_path}")
    log_msg(f"Initial guess saved to: {initial_guess_path}")
    
    return model, scaler, metrics, test_details, averaged_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-Weighted GCN Inference - Evaluate model on test set")
    parser.add_argument("data_tag", nargs="?", default=None,
                       help="Data tag (e.g., nmos_20260204_130041). If not provided, auto-detect latest.")
    parser.add_argument("training_timestamp", nargs="?", default=None,
                       help="Training timestamp (e.g., 20260130_063915). If not provided, auto-detect latest.")
    parser.add_argument("--data-tag", dest="data_tag_opt", type=str, default=None,
                       help="Data tag (named option). Overrides positional data_tag if provided.")
    parser.add_argument("--training-timestamp", dest="training_timestamp_opt", type=str, default=None,
                       help="Training timestamp (named option). Overrides positional training_timestamp if provided.")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config.yaml. If not provided, will use training's config_used.yaml")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model .pt file. If provided, overrides default model.pt.")
    parser.add_argument("--scaler", type=str, default=None,
                       help="Path to scaler .pkl file. If not provided, auto-detect near model.")
    parser.add_argument("--suffix", type=str, default=None,
                       help="Output file suffix (e.g., 'nmos', 'pmos'). Will append as '_suffix' to filenames.")
    parser.add_argument("--metrics-path", type=str, default=None,
                       help="Path to metrics.json file for test_indices. Allows using test_indices from different dataset.")

    args = parser.parse_args()

    # 智能配置加载：优先使用训练时的配置快照
    config_to_load = args.config
    output_dir = Path("/home/hu/Diana-Sim/gcn5")  # 默认输出目录
    
    # 如果没有指定配置文件，尝试从训练目录加载
    if config_to_load is None:
        # 先确定数据标签和训练时间戳
        data_tag_arg = args.data_tag_opt if args.data_tag_opt else args.data_tag
        training_ts_arg = args.training_timestamp_opt if args.training_timestamp_opt else args.training_timestamp
        if data_tag_arg and training_ts_arg:
            data_tag_check = data_tag_arg
            training_ts_check = training_ts_arg
        else:
            try:
                # 不指定device，获取最新的任何训练
                data_tag_check, training_ts_check = get_latest_timestamp(output_dir, device=None)
            except:
                data_tag_check, training_ts_check = None, None
        
        # 尝试加载训练时的配置快照
        if data_tag_check and training_ts_check:
            training_config = output_dir / "logs" / data_tag_check / training_ts_check / "config_used.yaml"
            if training_config.exists():
                config_to_load = str(training_config)
                print(f"✓ 使用训练时的配置快照: {training_config}")
            else:
                print(f"⚠️  未找到训练配置快照: {training_config}")
                print(f"   回退到主配置文件")
    
    config, config_path = load_config(config_to_load)
    inference_cfg = config.get("inference", {})

    data_tag = args.data_tag_opt if args.data_tag_opt else args.data_tag
    training_timestamp = args.training_timestamp_opt if args.training_timestamp_opt else args.training_timestamp
    suffix = args.suffix if args.suffix is not None else inference_cfg.get("suffix", "")

    model_path = Path(args.model) if args.model else None
    if model_path is None and inference_cfg.get("model_path"):
        model_path = Path(inference_cfg.get("model_path"))

    scaler_path = Path(args.scaler) if args.scaler else None
    if scaler_path is None and inference_cfg.get("scaler_path"):
        scaler_path = Path(inference_cfg.get("scaler_path"))

    metrics_path = Path(args.metrics_path) if args.metrics_path else None
    if metrics_path is None and inference_cfg.get("metrics_path"):
        metrics_path = Path(inference_cfg.get("metrics_path"))
    main(data_tag, training_timestamp, suffix, model_path=model_path, scaler_path=scaler_path, config=config, config_path=config_path, metrics_path=metrics_path)
