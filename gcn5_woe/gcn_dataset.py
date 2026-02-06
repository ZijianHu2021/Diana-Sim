#!/usr/bin/env python3
"""
GNN数据集加载 - 完全按照规范使用数据

📋 数据使用规范:
  1. node_features.npy 作为GNN输入 (形状: 10, 3)
     - [0]: 电压V (约0-3V)
     - [1]: 残差f (约1e-15到1e-3)
     - [2]: 节点类型 (0-5)
  
  2. edge_index.npy + edge_attr.npy 构建图结构
     - edge_index: (2, E) - 边的连接关系
     - edge_attr: (E, 1) - 边的属性（加载但不用于标准GCN）
  
  3. actual_changes.npy 作为训练标签 (形状: 55, 10)
     - 标签定义: 最终收敛值(V) - 本次迭代后值(V)
     - actual_changes[i] 对应 iteration_i/ 目录的标签

特征归一化:
  - 使用StandardScaler对所有节点特征进行归一化
  - 在训练集上拟合scaler，然后应用到所有数据
  - 解决特征量纲不匹配问题 (V vs f 量级差异大)
"""
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def _sorted_iteration_dirs(numpy_dir: Path) -> List[Path]:
    """
    获取排序后的迭代目录列表
    
    Args:
        numpy_dir: numpy目录路径 (e.g., /path/to/gnn_data/numpy)
    
    Returns:
        按迭代编号排序的目录列表 [iteration_0, iteration_1, ..., iteration_54]
    """
    iteration_dirs = [d for d in numpy_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")]
    iteration_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))
    return iteration_dirs


def load_actual_changes(data_root: Path) -> np.ndarray:
    """
    加载标签数据
    
    Args:
        data_root: 数据根目录
    
    Returns:
        actual_changes: (55, 10) 数组
          - 55: 迭代数
          - 10: 每个迭代的节点数
          - 值: 最终收敛值(V) - 本次迭代后值(V)
    """
    changes_path = data_root / "actual_changes.npy"
    if not changes_path.exists():
        raise FileNotFoundError(f"未找到actual_changes.npy: {changes_path}")
    return np.load(changes_path)


def load_train_val_split(data_root: Path) -> Tuple[List[int], List[int]]:
    """
    加载预定义的训练/验证集分割
    
    Args:
        data_root: 数据根目录
    
    Returns:
        (train_indices, val_indices) - 若文件不存在返回 ([], [])
    """
    split_path = data_root / "train_val_split.json"
    if not split_path.exists():
        return [], []
    with open(split_path, "r") as f:
        split = json.load(f)
    return split.get("train_indices", []), split.get("val_indices", [])


def build_dataset(data_root: Path, normalize: bool = True, scaler: Optional[StandardScaler] = None) -> Tuple[List[Data], Optional[StandardScaler]]:
    """
    构建PyTorch Geometric数据集，支持特征归一化
    
    从numpy文件加载图数据，构造PyTorch Geometric的Data对象列表
    
    参数:
        data_root: 数据根目录 (包含numpy/和actual_changes.npy)
        normalize: 是否对节点特征进行归一化 (默认True)
        scaler: 已拟合的StandardScaler对象
                若为None，则对数据进行拟合（训练集模式）
                若不为None，则仅使用该scaler进行变换（测试集模式）
    
    返回:
        (data_list, scaler)
        - data_list: List[Data], 长度为55 (55个迭代)
        - scaler: 使用的StandardScaler对象（若normalize=True）或None
    
    数据结构 (每个Data对象):
        - x: (10, 3) 节点特征 [电压V, 残差f, 节点类型]
        - edge_index: (2, E) 边连接 (通常E≈24)
        - edge_attr: (E, 1) 边属性（加载但不必使用）
        - y: (10, 1) 标签 - actual_changes[i]
    """
    numpy_dir = data_root / "numpy"
    if not numpy_dir.exists():
        raise FileNotFoundError(f"未找到numpy目录: {numpy_dir}")

    # 获取所有迭代目录（已排序）
    iteration_dirs = _sorted_iteration_dirs(numpy_dir)
    actual_changes = load_actual_changes(data_root)

    # 验证数据一致性
    if len(iteration_dirs) != actual_changes.shape[0]:
        raise ValueError(
            f"迭代目录数量({len(iteration_dirs)})与actual_changes行数({actual_changes.shape[0]})不一致"
        )

    data_list: List[Data] = []
    
    # 步骤1: 收集所有特征用于归一化
    all_features = []
    print(f"[数据加载] 读取{len(iteration_dirs)}个迭代的特征...")
    for it_dir in iteration_dirs:
        node_features = np.load(it_dir / "node_features.npy").astype(np.float32)
        # node_features 形状: (10, 3)
        #   维度0: 10个节点
        #   维度1: [V (0-3V), f (1e-15~1e-3), 节点类型 (0-5)]
        all_features.append(node_features)
    
    # 步骤2: 拟合或使用已有的scaler进行特征归一化
    if normalize:
        if scaler is None:
            # 训练模式: 拟合scaler
            # 将所有特征连接: (55*10, 3) = (550, 3)
            all_features_concat = np.vstack(all_features)
            scaler = StandardScaler()
            scaler.fit(all_features_concat)
            print(f"[特征归一化] 已拟合StandardScaler")
            print(f"  - mean: {scaler.mean_}")
            print(f"  - scale (std): {scaler.scale_}")
        
        # 对每个迭代应用归一化
        normalized_features = []
        for feat in all_features:
            feat_norm = scaler.transform(feat)  # (10, 3)
            normalized_features.append(feat_norm)
        print(f"[特征归一化] 已应用StandardScaler到所有特征")
    else:
        normalized_features = all_features
        scaler = None

    # 步骤3: 构造Data对象列表
    print(f"[构造数据集] 创建PyTorch Geometric Data对象...")
    for idx, it_dir in enumerate(iteration_dirs):
        # 加载边信息
        edge_index = np.load(it_dir / "edge_index.npy").astype(np.int64)  # (2, E)
        edge_attr = np.load(it_dir / "edge_attr.npy").astype(np.float32)   # (E, 1)

        # 加载标签
        y = actual_changes[idx].astype(np.float32)  # (10,)

        # 转换为PyTorch张量
        x = torch.from_numpy(normalized_features[idx])       # (10, 3)
        edge_index = torch.from_numpy(edge_index)            # (2, E)
        edge_attr = torch.from_numpy(edge_attr)              # (E, 1)
        y = torch.from_numpy(y).view(-1, 1)                  # (10, 1)

        # 构造Data对象（不使用edge_attr，但仍然加载以保持兼容性）
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data)

    print(f"[完成] 共{len(data_list)}个Data对象")
    return data_list, scaler
