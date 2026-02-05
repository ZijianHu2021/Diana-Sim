#!/usr/bin/env python3
"""
Edge-Weighted GCN模型定义（5层深层网络 + 128维隐层 + 边权重版本）

📊 模型架构: GCNNet5D_EW(3 → 128 → 128 → 64 → 64 → 1) + Edge Weights
  - in_channels=3: 对应node_features的3维 [电压V, 残差f, 节点类型]
  - hidden_channels=[128, 128, 64, 64]: 逐层特征维度
  - out_channels=1: 单个预测值（收敛距离）
  - edge_attr: 边权重（Jacobian矩阵值），参与消息传递计算

关键改进（相比gcn5d）:
  ✅ 在消息传递中使用edge_attr（边权重）
  ✅ 保留符号信息：通过tanh保留Jacobian正负相关性
  ✅ 软化极端值：避免权重差异过大破坏学习
  ✅ 范围控制：edge_weight ∈ [0.1, 1.1]，避免完全抑制或过度增强

数据流向:
  输入: x ∈ ℝ^(N×3), edge_index ∈ ℤ^(2×E), edge_attr ∈ ℝ^(E×1)
    ↓
  [GCNConv 3→128 + edge_attr] + ReLU
    ↓
  [GCNConv 128→128 + edge_attr] + ReLU
    ↓
  [GCNConv 128→64 + edge_attr] + ReLU
    ↓
  [GCNConv 64→64 + edge_attr] + ReLU
    ↓
  [GCNConv 64→1 + edge_attr]
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
    Edge-Weighted 图卷积层 (Edge-Weighted Graph Convolutional Layer)
    
    实现: A_weighted @ X @ W，其中
      - A_weighted = EdgeWeight ⊙ (D^(-0.5) @ A @ D^(-0.5))
      - EdgeWeight: 边权重矩阵（来自edge_attr）
      - A: 邻接矩阵
      - X: 节点特征矩阵
      - W: 可学习的权重矩阵
    
    相比标准GCN的改进:
      1. 保留Jacobian符号：正值=正相关，负值=负相关
      2. 软化极端值：通过tanh(edge_attr*2)压缩到[-1,1]
      3. 映射到安全范围：[0.1, 1.1]，避免不稳定或完全抑制
      消息计算 = tanh_softened(edge_attr) * norm * x_j
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')  # 邻域消息聚合方式：求和
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播（支持边权重）
        
        Args:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
            edge_attr: 边权重 [E, 1] (可选，若为None则退化为标准GCN)
        
        Returns:
            out: 输出特征 [N, out_channels]
        """
        # 步骤1: 添加自环 (确保节点考虑自身特征)
        edge_index, edge_attr = add_self_loops(
            edge_index, 
            edge_attr=edge_attr,
            fill_value=1.0,  # 自环的边权重设为1.0
            num_nodes=x.size(0)
        )
        
        # 步骤2: 特征线性变换 X @ W
        x = self.lin(x)

        # 步骤3: 计算对称归一化系数 D^(-0.5)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)  # 出度
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 处理孤立节点
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # D^(-0.5)[row] * D^(-0.5)[col]

        # 步骤4: 如果提供了edge_attr，将其融入归一化系数
        if edge_attr is not None:
            # 🔑 改进方案：保留符号 + 软化极端值
            edge_weight = edge_attr.squeeze(-1)  # [E, 1] -> [E]，保留正负号
            
            # 使用tanh软化极端值，同时保留符号信息
            # tanh将极端值压缩到[-1, 1]，对中等值敏感
            edge_weight = torch.tanh(edge_weight * 2)  # 乘以2增加敏感度
            
            # 映射到[0.1, 1.1]范围：
            # - 避免负权重导致的不稳定（GCN通常假设非负邻接矩阵）
            # - 避免完全抑制弱边（最小0.1）
            # - 避免过度增强强边（最大1.1）
            edge_weight = edge_weight * 0.5 + 0.6  # [-1, 1] -> [0.1, 1.1]
            
            norm = norm * edge_weight  # 边权重调制归一化系数

        # 步骤5: 消息传递和邻域聚合
        out = self.propagate(edge_index, x=x, norm=norm)
        
        # 步骤6: 添加偏置
        out += self.bias
        return out

    def message(self, x_j, norm):
        """
        定义邻域j到节点i的消息形式
        
        消息 = norm[i,j] * x[j]
        其中 norm[i,j] = edge_weight[i,j] * D^(-0.5)[i] * D^(-0.5)[j]
        
        改进版edge_weight处理：
        - 保留Jacobian的符号信息（通过tanh软化后映射到正值）
        - 软化极端值的影响（tanh压缩）
        - 范围控制在[0.1, 1.1]，避免完全抑制或过度增强
        
        Args:
            x_j: 邻居节点特征 [E, out_channels]
            norm: 归一化系数（已包含边权重）[E]
        """
        return norm.view(-1, 1) * x_j


class GCNNet(nn.Module):
    """
    完整Edge-Weighted GCN网络（5层 + 128维隐层 + 边权重），用于节点级预测
    
    输入:
      - x: 节点特征 (N, 3) - [电压V, 残差f, 节点类型]
      - edge_index: 边连接 (2, E)
      - edge_attr: 边权重 (E, 1) - Jacobian矩阵值
    
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

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播（支持边权重）
        
        数据流:
          x (N, 3), edge_attr (E, 1)
            ↓ conv1 + edge_attr
          x (N, 128) → ReLU → Dropout
            ↓ conv2 + edge_attr
          x (N, 128) → ReLU → Dropout
            ↓ conv3 + edge_attr
          x (N, 64) → ReLU → Dropout
            ↓ conv4 + edge_attr
          x (N, 64) → ReLU → Dropout
            ↓ conv5 + edge_attr
          out (N, 1)
        
        Args:
            x: 节点特征 [N, 3]
            edge_index: 边索引 [2, E]
            edge_attr: 边权重 [E, 1] (可选，若为None则退化为标准GCN)
        """
        # 层1: 初步特征提取 (3 → 128维) + 边权重
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)

        # 层2: 邻域交互学习 (128 → 128维) + 边权重
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)

        # 层3: 特征压缩 (128 → 64维) + 边权重
        x = self.conv3(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)

        # 层4: 深层特征融合 (64 → 64维) + 边权重
        x = self.conv4(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)

        # 层5: 最终预测 (64 → 1维) + 边权重
        x = self.conv5(x, edge_index, edge_attr)
        return x
