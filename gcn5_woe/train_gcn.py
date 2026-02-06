#!/usr/bin/env python3
"""
训练标准GCN模型（5层深层 + 128维隐层，不使用边权重）
包含特征归一化、详细日志、评价指标、预测结果保存和loss曲线可视化

标准GCN特点:
  ✅ 使用对称归一化 D^(-0.5) @ A @ D^(-0.5)
  ✅ 不使用边权重，所有边平等对待
  ✅ 代码简洁高效
"""
import argparse
import random
import json
from pathlib import Path
from datetime import datetime
import pickle
import sys

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 动态调整导入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "graph"))

from gcn_dataset import build_dataset, load_train_val_split
from gcn_model import GCNNet
from config_utils import load_config, resolve_data_root, resolve_output_dir, copy_config_snapshot


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(num_graphs: int, train_ratio: float = 0.7, val_ratio: float = 0.2):
    """
    三分割：训练集、验证集、测试集
    
    Args:
        num_graphs: 总图数量
        train_ratio: 训练集比例 (默认 0.7)
        val_ratio: 验证集比例 (默认 0.2)
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    indices = list(range(num_graphs))
    random.shuffle(indices)
    
    train_split = int(train_ratio * num_graphs)
    val_split = train_split + int(val_ratio * num_graphs)
    
    return (indices[:train_split], 
            indices[train_split:val_split], 
            indices[val_split:])


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算MSE, MAE, R²等指标"""
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


def main(config_path: str | None = None):
    config, resolved_config_path = load_config(config_path)
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    log_cfg = config.get("logging", {})

    set_seed(train_cfg.get("seed", 42))

    data_root, data_timestamp, data_device = resolve_data_root(config)
    output_dir = resolve_output_dir(config)
    logs_dir = output_dir / "logs"
    data_tag = f"{data_device}_{data_timestamp}"
    data_dir = logs_dir / data_tag
    
    # 生成训练时刻的时间戳
    training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = data_dir / training_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志文件
    log_prefix = log_cfg.get("training_log_prefix", "training_log_")
    metrics_prefix = log_cfg.get("metrics_prefix", "metrics_")
    predictions_prefix = log_cfg.get("predictions_prefix", "predictions_")
    plot_prefix = log_cfg.get("plot_prefix", "training_curves_")

    log_path = run_dir / f"{log_prefix}{training_timestamp}.txt"
    metrics_path = run_dir / f"{metrics_prefix}{training_timestamp}.json"
    predictions_path = run_dir / f"{predictions_prefix}{training_timestamp}.json"
    
    def log_msg(msg: str):
        """同时写入日志文件和打印"""
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log_msg(f"{'='*70}")
    log_msg(f"标准GCN训练日志（5层 + 128维隐层，无边权重）- {training_timestamp}")
    log_msg(f"Data: {data_tag}")
    log_msg(f"{'='*70}")

    # 加载数据并进行特征归一化
    log_msg("[1] 加载数据集...")
    data_list, scaler = build_dataset(data_root, normalize=True)
    log_msg(f"✓ 加载完成，共{len(data_list)}个图")
    
    if scaler is not None:
        log_msg(f"✓ 特征已归一化")
        log_msg(f"  - Scaler统计: mean={scaler.mean_}, var={scaler.var_}")

    # 数据检查
    sample_data = data_list[0]
    log_msg(f"✓ 数据检查:")
    log_msg(f"  - 节点特征: {sample_data.x.shape}")
    log_msg(f"  - 边索引: {sample_data.edge_index.shape}")
    log_msg(f"  - 标签: {sample_data.y.shape}")

    # 使用split_indices进行7:2:1分割
    train_ratio = train_cfg.get("train_ratio", 0.7)
    val_ratio = train_cfg.get("val_ratio", 0.2)
    train_idx, val_idx, test_idx = split_indices(len(data_list), train_ratio=train_ratio, val_ratio=val_ratio)
    log_msg(f"✓ 创建train/val/test分割 (7:2:1)")
    
    log_msg(f"✓ 训练集: {len(train_idx)}个样本，验证集: {len(val_idx)}个样本，测试集: {len(test_idx)}个样本")

    train_set = [data_list[i] for i in train_idx]
    val_set = [data_list[i] for i in val_idx]
    test_set = [data_list[i] for i in test_idx]

    batch_size = train_cfg.get("batch_size", 8)
    shuffle = train_cfg.get("shuffle", True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg(f"✓ 设备: {device}")

    # 模型配置 - 使用128维隐层，不使用边权重
    log_msg(f"\n[2] 模型配置")
    in_channels = model_cfg.get("in_channels", 3)
    hidden_channels = model_cfg.get("hidden_channels", 128)
    out_channels = model_cfg.get("out_channels", 1)
    dropout = model_cfg.get("dropout", 0.0)

    model = GCNNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    ).to(device)
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    log_msg(f"✓ 模型架构: GCNNet({in_channels} -> {hidden_channels} -> ... -> {out_channels})")
    log_msg(f"  【5层深层 + 128维隐层，标准GCN（无边权重）】")
    log_msg(f"✓ 优化器: Adam, lr={learning_rate}")
    log_msg(f"✓ 损失函数: MSELoss")
    log_msg(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    epochs = train_cfg.get("epochs", 200)
    train_losses = []
    val_losses = []
    
    log_msg(f"\n[3] 开始训练 ({epochs} epochs)")
    log_msg(f"{'-'*70}")

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # 标准GCN：只传入x和edge_index，不使用edge_attr
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            train_batches += batch.num_graphs

        train_loss = train_loss / max(1, train_batches)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # 标准GCN：只传入x和edge_index
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                val_batches += batch.num_graphs

        val_loss = val_loss / max(1, val_batches)
        val_losses.append(val_loss)

        # 每20个epoch打印一次
        if epoch == 1 or epoch % 20 == 0:
            log_msg(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    log_msg(f"{'-'*70}")
    log_msg(f"✓ 训练完成")

    # 计算最终指标
    log_msg(f"\n[4] 最终指标")
    log_msg(f"✓ 初始训练损失: {train_losses[0]:.6f}")
    log_msg(f"✓ 最终训练损失: {train_losses[-1]:.6f}")
    log_msg(f"✓ 最佳验证损失: {min(val_losses):.6f} (Epoch {np.argmin(val_losses)+1})")
    log_msg(f"✓ 损失下降: {(1 - train_losses[-1]/train_losses[0])*100:.2f}%")

    # 计算预测结果和详细指标
    log_msg(f"\n[5] 计算评价指标")
    model.eval()
    
    # 在验证集上的预测
    val_predictions = []
    val_labels = []
    val_current_voltages = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # 标准GCN：只传入x和edge_index
            out = model(batch.x, batch.edge_index)
            val_predictions.extend(out.cpu().numpy().flatten())
            val_labels.extend(batch.y.cpu().numpy().flatten())
            # Extract current voltages (first feature)
            val_current_voltages.extend(batch.x[:, 0].cpu().numpy().flatten())
    
    val_predictions = np.array(val_predictions)
    val_labels = np.array(val_labels)
    val_current_voltages = np.array(val_current_voltages)
    val_metrics = calculate_metrics(val_labels, val_predictions)
    
    # Calculate V_final differences
    val_predicted_final = val_current_voltages + val_predictions
    val_actual_final = val_current_voltages + val_labels
    val_final_diff = np.abs(val_actual_final - val_predicted_final)
    
    log_msg(f"✓ 验证集评价指标:")
    log_msg(f"  - MSE:  {val_metrics['mse']:.6f}")
    log_msg(f"  - MAE:  {val_metrics['mae']:.6f}")
    log_msg(f"  - RMSE: {val_metrics['rmse']:.6f}")
    log_msg(f"  - R²:   {val_metrics['r2']:.6f}")
    
    # 在测试集上的预测
    test_predictions = []
    test_labels = []
    if len(test_loader) > 0:
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                # 标准GCN：只传入x和edge_index
                out = model(batch.x, batch.edge_index)
                test_predictions.extend(out.cpu().numpy().flatten())
                test_labels.extend(batch.y.cpu().numpy().flatten())
        
        test_predictions = np.array(test_predictions)
        test_labels = np.array(test_labels)
        test_metrics = calculate_metrics(test_labels, test_predictions)
        
        log_msg(f"\n✓ 测试集评价指标:")
        log_msg(f"  - MSE:  {test_metrics['mse']:.6f}")
        log_msg(f"  - MAE:  {test_metrics['mae']:.6f}")
        log_msg(f"  - RMSE: {test_metrics['rmse']:.6f}")
        log_msg(f"  - R²:   {test_metrics['r2']:.6f}")
    else:
        log_msg(f"\n⚠ 测试集为空，跳过测试集评估")
        test_predictions = np.array([])
        test_labels = np.array([])
        test_metrics = {
            "mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0
        }

    # 保存模型
    log_msg(f"\n[6] 保存结果")
    model_path = run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    log_msg(f"✓ 模型已保存: {model_path}")

    # 保存scaler
    if scaler is not None:
        scaler_path = run_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        log_msg(f"✓ Scaler已保存: {scaler_path}")

    # 保存训练指标
    metrics_data = {
        "timestamp": training_timestamp,
        "data_tag": data_tag,
        "data_timestamp": data_timestamp,
        "data_device": data_device,
        "model_version": "5-layer-deep-128d-standard-gcn",
        "edge_weighted": False,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_channels": hidden_channels,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "initial_train_loss": float(train_losses[0]),
        "final_train_loss": float(train_losses[-1]),
        "best_val_loss": float(min(val_losses)),
        "best_val_epoch": int(np.argmin(val_losses) + 1),
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "validation_metrics": {
            "mse": float(val_metrics["mse"]),
            "mae": float(val_metrics["mae"]),
            "rmse": float(val_metrics["rmse"]),
            "r2": float(val_metrics["r2"])
        },
        "test_metrics": {
            "mse": float(test_metrics["mse"]),
            "mae": float(test_metrics["mae"]),
            "rmse": float(test_metrics["rmse"]),
            "r2": float(test_metrics["r2"])
        }
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    log_msg(f"✓ 训练指标已保存: {metrics_path}")

    # 保存预测结果
    predictions_data = {
        "timestamp": training_timestamp,
        "data_tag": data_tag,
        "validation_predictions": val_predictions.tolist(),
        "validation_labels": val_labels.tolist(),
        "test_predictions": test_predictions.tolist(),
        "test_labels": test_labels.tolist()
    }
    
    with open(predictions_path, "w") as f:
        json.dump(predictions_data, f, indent=2)
    log_msg(f"✓ 预测结果已保存: {predictions_path}")

    # Draw loss curves and validation final voltage difference
    log_msg(f"\n[7] Plotting visualization")
    
    # Load node names from metadata
    node_names = []
    try:
        metadata_file = data_root / "labels_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if "node_names" in metadata:
                    node_names = metadata["node_names"]
                else:
                    node_names = [f"Node_{i}" for i in range(10)]
        else:
            node_names = [f"Node_{i}" for i in range(10)]
    except:
        node_names = [f"Node_{i}" for i in range(10)]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Standard GCN Training Results (5-Layer + 128D, No Edge Weights) ({data_tag})\n" + 
                 "Node Mapping: " + ", ".join([f"N{i}={node_names[i]}" for i in range(len(node_names))]), 
                 fontsize=12, fontweight="bold")
    
    # Loss curve (linear scale)
    ax = axes[0]
    ax.plot(train_losses, label="Train Loss", linewidth=2, marker="o", markersize=3, alpha=0.7)
    ax.plot(val_losses, label="Val Loss", linewidth=2, marker="s", markersize=3, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Loss Curve (Linear Scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss curve (log scale)
    ax = axes[1]
    ax.semilogy(train_losses, label="Train Loss", linewidth=2, marker="o", markersize=3, alpha=0.7)
    ax.semilogy(val_losses, label="Val Loss", linewidth=2, marker="s", markersize=3, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, log scale)")
    ax.set_title("Loss Curve (Log Scale)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    
    # Validation: per-node average input, predicted final, and actual final
    ax = axes[2]
    val_input_per_node = []
    val_pred_final_per_node = []
    val_actual_final_per_node = []
    for node_idx in range(10):  # 10 nodes
        node_inputs = []
        node_pred_final = []
        node_actual_final = []
        for i in range(node_idx, len(val_predictions), 10):
            current_v = val_current_voltages[i]
            pred_delta = val_predictions[i]
            label_delta = val_labels[i]
            node_inputs.append(current_v)
            node_pred_final.append(current_v + pred_delta)
            node_actual_final.append(current_v + label_delta)
        if node_inputs:
            val_input_per_node.append(np.mean(node_inputs))
            val_pred_final_per_node.append(np.mean(node_pred_final))
            val_actual_final_per_node.append(np.mean(node_actual_final))
    
    node_indices = np.arange(len(val_input_per_node))
    
    ax.plot(node_indices, val_input_per_node, marker="^", linestyle=":", linewidth=2, markersize=7,
            label="Avg Input Voltage", color="gray", alpha=0.7)
    ax.plot(node_indices, val_pred_final_per_node, marker="o", linestyle="-", linewidth=2, markersize=8,
            label="Avg Predicted Final (V)", color="steelblue", alpha=0.7)
    ax.plot(node_indices, val_actual_final_per_node, marker="s", linestyle="--", linewidth=2, markersize=8,
            label="Avg Final Label (V)", color="coral", alpha=0.7)
    
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Validation: Input vs Predicted Final vs Label")
    ax.set_xticks(node_indices)
    ax.set_xticklabels([f"N{i}" for i in node_indices])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = run_dir / f"{plot_prefix}{training_timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    log_msg(f"✓ Visualization saved: {plot_path}")
    
    log_msg(f"\n{'='*70}")
    if log_cfg.get("save_config", True):
        snapshot_path = copy_config_snapshot(resolved_config_path, run_dir, "config_used.yaml")
        log_msg(f"✓ Config snapshot saved: {snapshot_path}")

    log_msg(f"✅ Training Complete! All results saved to {run_dir}")
    log_msg(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train standard GCN with unified YAML config")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    main(args.config)
