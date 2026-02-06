#!/usr/bin/env python3
"""
标准GCN模型定义（5层深层网络 + 128维隐层，不使用边权重）

📊 模型架构: GCNNet(3 → 128 → 128 → 64 → 64 → 1)
  - in_channels=3: 对应node_features的3维 [电压V, 残差f, 节点类型]
  - hidden_channels=[128, 128, 64, 64]: 逐层特征维度
  - out_channels=1: 单个预测值（收敛距离）
  - 无edge_attr: 标准对称归一化GCN

对比edge-weighted版本的简化:
  ✅ 移除edge_attr处理逻辑
  ✅ 使用标准 D^(-0.5) @ A @ D^(-0.5) 对称归一化
  ✅ 代码更简洁，计算更高效

数据流向:
  输入: x ∈ ℝ^(N×3), edge_index ∈ ℤ^(2×E)
    ↓
  [GCNConv 3→128] + ReLU
    ↓
  [GCNConv 128→128] + ReLU
    ↓
  [GCNConv 128→64] + ReLU
    ↓
  [GCNConv 64→64] + ReLU
    ↓
  [GCNConv 64→1]
    ↓
  输出: pred ∈ ℝ^(N×1)  (N个节点的预测值)
"""
import torch
from torch import nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    """
    标准图卷积层 (Graph Convolutional Layer)，不使用边权重
    
    实现: A_norm @ X @ W，其中
      - A_norm = D^(-0.5) @ A @ D^(-0.5)
      - A: 邻接矩阵
      - X: 节点特征矩阵
      - W: 可学习的权重矩阵
    
    特点:
      1. 使用对称归一化系数
      2. 无边权重，所有边平等对待
      3. 简洁高效的标准GCN实现
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')  # 邻域消息聚合方式：求和
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        """
        前向传播（标准GCN，无边权重）
        
        Args:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
        
        Returns:
            out: 输出特征 [N, out_channels]
        """
        # 步骤1: 添加自环 (确保节点考虑自身特征)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 步骤2: 特征线性变换 X @ W
        x = self.lin(x)

        # 步骤3: 计算对称归一化系数 D^(-0.5)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)  # 出度
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 处理孤立节点
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # D^(-0.5)[row] * D^(-0.5)[col]

        # 步骤4: 消息传递和邻域聚合
        out = self.propagate(edge_index, x=x, norm=norm)
        
        # 步骤5: 添加偏置
        out += self.bias
        return out

    def message(self, x_j, norm):
        """
        定义邻域j到节点i的消息形式
        
        消息 = norm[i,j] * x[j]
        其中 norm[i,j] = D^(-0.5)[i] * D^(-0.5)[j]
        
        Args:
            x_j: 邻居节点特征 [E, out_channels]
            norm: 归一化系数 [E]
        """
        return norm.view(-1, 1) * x_j


class GCNNet(nn.Module):
    """
    完整标准GCN网络（5层 + 128维隐层，不使用边权重），用于节点级预测
    
    输入:
      - x: 节点特征 (N, 3) - [电压V, 残差f, 节点类型]
      - edge_index: 边连接 (2, E)
    
    输出:
      - pred: 节点预测值 (N, 1) - 每个节点的收敛距离
    
    标签 (training):
      - y: (N, 1) - actual_changes中对应迭代的标签值
           = 最终收敛电压 - 本次迭代后电压
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 128, 
                 out_channels: int = 1, dropout: float = 0.0):
        """
        参数:
            in_channels: 输入特征维度 (固定=3)
            hidden_channels: 初始隐层维度 (默认=128)
            out_channels: 输出维度 (固定=1用于回归)
            dropout: dropout比率 (默认=0.0)
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)              # 3 → 128
        self.conv2 = GCNConv(hidden_channels, hidden_channels)          # 128 → 128
        self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)     # 128 → 64
        self.conv4 = GCNConv(hidden_channels // 2, hidden_channels // 2)  # 64 → 64
        self.conv5 = GCNConv(hidden_channels // 2, out_channels)        # 64 → 1
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """
        前向传播（标准GCN，无边权重）
        
        数据流:
          x (N, 3)
            ↓ conv1
          x (N, 128) → ReLU → Dropout
            ↓ conv2
          x (N, 128) → ReLU → Dropout
            ↓ conv3
          x (N, 64) → ReLU → Dropout
            ↓ conv4
          x (N, 64) → ReLU → Dropout
            ↓ conv5
          out (N, 1)
        
        Args:
            x: 节点特征 [N, 3]
            edge_index: 边索引 [2, E]
        """
        # 层1: 初步特征提取 (3 → 128维)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # 层2: 邻域交互学习 (128 → 128维)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # 层3: 特征压缩 (128 → 64维)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # 层4: 深层特征融合 (64 → 64维)
        x = self.conv4(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # 层5: 最终预测 (64 → 1维)
        x = self.conv5(x, edge_index)
        return x
