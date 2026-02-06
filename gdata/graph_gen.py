#!/usr/bin/env python3
"""
从 /home/hu/Diana-Sim/dc/logs 按时间戳提取Newton数据，
构建图、生成可视化、并保存为GNN训练用.npy数据。

输出全部写到 /home/hu/Diana-Sim/gdata 下，并按时间戳分目录。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def infer_node_type(node_name: str) -> int:
    name = node_name.upper()
    if name.startswith("VSRC_"):
        return 0
    if name in {"VDD", "VSS"}:
        return 1
    if name.startswith("INTERNAL_"):
        return 2
    if "VIN" in name:
        return 3
    if "VOUT" in name:
        return 4
    return 5


class GraphData:
    def __init__(
        self,
        jacobian: np.ndarray,
        voltages: np.ndarray,
        residual: np.ndarray,
        node_names: List[str],
        iteration: int,
        source_factor: float,
        jacobian_condition_number: float | None,
    ):
        self.jacobian = jacobian
        self.voltages = voltages
        self.residual = residual
        self.node_names = node_names
        self.node_types = [infer_node_type(name) for name in node_names]
        self.node_attrs = {
            i: {"name": name, "node_type": self.node_types[i]}
            for i, name in enumerate(node_names)
        }
        self.graph = {
            "source_factor": source_factor,
            "iteration": iteration,
            "jacobian_condition_number": jacobian_condition_number,
        }
        self.edge_index, self.edge_attr = self._build_edges(jacobian)

    @staticmethod
    def _build_edges(jacobian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = np.nonzero(jacobian)
        order = np.lexsort((rows, cols))
        rows = rows[order]
        cols = cols[order]
        edge_index = np.vstack([rows, cols]).astype(np.int64)
        edge_attr = jacobian[rows, cols].astype(np.float32).reshape(-1, 1)
        return edge_index, edge_attr

    def node_ids(self) -> List[int]:
        return list(self.node_attrs.keys())

    def edge_list(self) -> List[Tuple[int, int]]:
        return list(zip(self.edge_index[0], self.edge_index[1]))


class JacobianGraphBuilder:
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.jacobians: List[np.ndarray] = []
        self.voltages: List[np.ndarray] = []
        self.residuals: List[np.ndarray] = []
        self.node_names: List[str] | None = None

    def load_json_and_build_graphs(self) -> List[GraphData]:
        with open(self.json_path, "r") as f:
            data = json.load(f)

        graphs: List[GraphData] = []
        for item in data.get("iterations", []):
            if item.get("jacobian") is None:
                continue

            jacobian = np.array(item["jacobian"], dtype=float)
            voltages = np.array(item["x"], dtype=float)
            residual = np.array(item["residual"], dtype=float)
            node_names = item["node_names"]

            self.jacobians.append(jacobian)
            self.voltages.append(voltages)
            self.residuals.append(residual)
            self.node_names = node_names

            graphs.append(
                GraphData(
                    jacobian=jacobian,
                    voltages=voltages,
                    residual=residual,
                    node_names=node_names,
                    iteration=item["iteration"],
                    source_factor=item["source_factor"],
                    jacobian_condition_number=item.get("jacobian_condition_number"),
                )
            )

        return graphs


class CircuitGraphVisualizer:
    def __init__(self):
        self.available = plt is not None

    def _plot_graph(self, graph: GraphData, ax, title: str | None = None):
        if not self.available:
            return

        node_ids = graph.node_ids()
        count = len(node_ids)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        positions = {
            node_id: (np.cos(angle), np.sin(angle))
            for node_id, angle in zip(node_ids, angles)
        }

        color_map = {
            0: "#1f77b4",
            1: "#ff7f0e",
            2: "#2ca02c",
            3: "#d62728",
            4: "#9467bd",
            5: "#8c564b",
        }

        for src, dst in graph.edge_list():
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            ax.plot([x0, x1], [y0, y1], color="#888888", linewidth=0.8, alpha=0.7)

        for node_id in node_ids:
            x, y = positions[node_id]
            node_type = graph.node_attrs[node_id]["node_type"]
            ax.scatter(x, y, s=150, color=color_map.get(node_type, "#333333"))
            ax.text(x, y, graph.node_attrs[node_id]["name"], fontsize=7, ha="center", va="center")

        if title:
            ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    def visualize_graph(
        self,
        graph: GraphData,
        layout: str = "spring",
        show_edge_labels: bool = False,
        save_path: str | None = None,
    ):
        if not self.available:
            print("⚠️  matplotlib未安装，跳过可视化")
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        title = f"Iter {graph.graph['iteration']} | SF {graph.graph['source_factor']:.3f}"
        self._plot_graph(graph, ax, title=title)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=160)
        plt.close(fig)

    def visualize_comparison(self, graphs: List[GraphData], iterations: List[int], layout: str, save_dir: str):
        if not self.available:
            print("⚠️  matplotlib未安装，跳过对比图")
            return

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        for ax, idx in zip(axes, iterations):
            graph = graphs[idx]
            title = f"Idx {idx} | Iter {graph.graph['iteration']}"
            self._plot_graph(graph, ax, title=title)
        fig.tight_layout()
        fig.savefig(Path(save_dir) / "comparison_4iterations.png", dpi=160)
        plt.close(fig)

    def visualize_node_evolution(self, graphs: List[GraphData], node_id: int, save_dir: str):
        if not self.available:
            print("⚠️  matplotlib未安装，跳过节点演化图")
            return

        values = [graph.voltages[node_id] for graph in graphs]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(values, linewidth=1.2)
        ax.set_xlabel("Iteration index")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"Node {node_id} evolution")
        fig.tight_layout()
        fig.savefig(Path(save_dir) / f"node_{node_id}_evolution.png", dpi=160)
        plt.close(fig)


class GNNDataset:
    def __init__(self, graphs: List[GraphData]):
        self.graphs = graphs

    def get_statistics(self, labels: np.ndarray | None = None) -> Dict:
        num_iterations = len(self.graphs)
        num_nodes = len(self.graphs[0].node_ids()) if self.graphs else 0
        edge_counts = [len(graph.edge_list()) for graph in self.graphs]
        avg_edges = float(np.mean(edge_counts)) if edge_counts else 0.0

        node_type_counts: Dict[int, int] = {}
        residual_values: List[float] = []
        for graph in self.graphs:
            residual_values.extend(graph.residual.tolist())
            for node_type in graph.node_types:
                node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        stats = {
            "num_iterations": num_iterations,
            "num_nodes": num_nodes,
            "node_feature_dim": 3,
            "edge_feature_dim": 1,
            "avg_edges_per_graph": avg_edges,
            "node_type_distribution": {str(k): v for k, v in sorted(node_type_counts.items())},
            "residual_range": [float(min(residual_values)), float(max(residual_values))] if residual_values else [0.0, 0.0],
        }

        if labels is not None and labels.size:
            stats["prediction_target_range"] = [float(labels.min()), float(labels.max())]

        return stats

    def _train_val_split(self, num_iterations: int, seed: int = 123) -> Dict:
        indices = np.arange(num_iterations)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        split_point = int(num_iterations * 0.8)
        train_indices = np.sort(indices[:split_point]).tolist()
        val_indices = np.sort(indices[split_point:]).tolist()
        return {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_size": len(train_indices),
            "val_size": len(val_indices),
        }

    def save_numpy(self, numpy_dir: str, split_path: Path | None = None):
        numpy_dir = Path(numpy_dir)
        numpy_dir.mkdir(parents=True, exist_ok=True)

        for idx, graph in enumerate(self.graphs):
            iter_dir = numpy_dir / f"iteration_{idx}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            node_features = np.column_stack([graph.voltages, graph.residual, graph.node_types]).astype(float)
            adjacency = graph.jacobian.T.astype(float)

            np.save(iter_dir / "node_features.npy", node_features)
            np.save(iter_dir / "edge_index.npy", graph.edge_index)
            np.save(iter_dir / "edge_attr.npy", graph.edge_attr)
            np.save(iter_dir / "adjacency.npy", adjacency)

        if split_path is not None:
            split = self._train_val_split(len(self.graphs))
            with open(split_path, "w") as f:
                json.dump(split, f, indent=2)


def print_header(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def find_latest_timestamp(logs_root: Path) -> str:
    if not logs_root.exists():
        raise FileNotFoundError(f"logs目录不存在: {logs_root}")
    candidates = [p.name for p in logs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"logs目录为空: {logs_root}")
    return sorted(candidates)[-1]


def collect_newton_jsons(run_dir: Path):
    return sorted(run_dir.glob("*/newton_analysis/newton_dc_*.json"))


def find_iteration_log(json_file: Path) -> Path:
    """在同一newton_analysis目录中找到iteration_tracking日志"""
    log_dir = json_file.parent
    candidates = list(log_dir.glob("iteration_tracking*.log"))
    if not candidates:
        raise FileNotFoundError(f"未找到iteration_tracking日志: {log_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_iteration_log(log_file: Path) -> Dict:
    """解析迭代追踪日志，提取actual_changes与节点名称"""
    print(f"📂 解析日志文件: {log_file.name}")

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取最终收敛节点电压（最终值）
    final_section_pattern = r"✅ 最终收敛节点电压:.*?(?=\n=+|$)"
    final_section_match = re.search(final_section_pattern, content, re.DOTALL)
    final_voltage_map: Dict[str, float] = {}
    if final_section_match:
        final_section = final_section_match.group(0)
        final_line_pattern = r"(\w+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)"
        for line_match in re.finditer(final_line_pattern, final_section):
            node_name = line_match.group(1)
            final_v = float(line_match.group(2))
            final_voltage_map[node_name] = final_v
    else:
        print("⚠️  未找到最终收敛节点电压段，无法计算到最终值的差值")

    # 提取迭代块
    iteration_pattern = r"📍 迭代 #(\d+).*?Source Factor: ([\d.]+).*?(?=📍|$)"
    iterations = re.finditer(iteration_pattern, content, re.DOTALL)

    global_iterations: List[int] = []
    source_factors: List[float] = []
    actual_changes_list: List[List[float]] = []
    node_names: List[str] | None = None

    # 提取节点数据行（在表格中）
    # 格式: 节点名 残差 更新量 迭代前 迭代后 实际变化
    line_pattern = r"(\w+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)"

    for match in iterations:
        iter_num = int(match.group(1))
        sf = float(match.group(2))

        global_iterations.append(iter_num)
        source_factors.append(sf)

        # 在这个迭代块中找"迭代后 (V)"列
        iter_block = match.group(0)

        lines = re.finditer(line_pattern, iter_block)

        changes: List[float] = []
        names: List[str] = []
        for line_match in lines:
            node_name = line_match.group(1)
            iter_after_v = float(line_match.group(5))  # 第5列是迭代后 (V)
            final_v = final_voltage_map.get(node_name)
            if final_v is None:
                print(f"⚠️  未找到节点 {node_name} 的最终值，跳过该节点")
                continue
            # 标签：最终值 - 本次迭代后值
            change_to_final = final_v - iter_after_v
            changes.append(change_to_final)
            names.append(node_name)

        # 确保有10个节点
        if len(changes) == 10:
            actual_changes_list.append(changes)
            if node_names is None:
                node_names = names
        else:
            print(f"⚠️  迭代 #{iter_num} 只找到 {len(changes)} 个节点，跳过")

    if not actual_changes_list:
        raise ValueError("未提取到有效的actual_changes数据")

    result = {
        'global_iteration': global_iterations,
        'source_factor': source_factors,
        'actual_changes': np.array(actual_changes_list),  # shape: (num_iters, 10)
        'node_names': node_names,
        'num_iterations': len(actual_changes_list)
    }

    return result


def save_labels_from_log(log_file: Path, output_dir: Path) -> Dict:
    """从日志中提取并保存actual_changes.npy与labels_metadata.json"""
    log_data = parse_iteration_log(log_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_file = output_dir / "actual_changes.npy"
    np.save(labels_file, log_data['actual_changes'])

    metadata = {
        'global_iterations': log_data['global_iteration'],
        'source_factors': log_data['source_factor'],
        'node_names': log_data['node_names'],
        'shape': log_data['actual_changes'].shape,
        'description': '标签=最终收敛值(V)-本次迭代后值(V)，用于GNN训练',
        'log_file': str(log_file),
    }

    metadata_file = output_dir / "labels_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ 标签已保存: {labels_file}")
    print(f"✅ 元数据已保存: {metadata_file}")
    return log_data


def process_single_json(json_file: Path, output_root: Path, timestamp: str):
    # device label: pmos/nmos (from parent folder)
    device = json_file.parent.parent.name

    output_base = output_root / timestamp / device / "output"
    gnn_output = output_root / timestamp / device / "gnn_data"

    output_base.mkdir(parents=True, exist_ok=True)
    gnn_output.mkdir(parents=True, exist_ok=True)

    print_header(f"🚀 处理 {device.upper()} - {json_file.name}")

    # ========== 第1阶段：数据加载 ==========
    print_header("第1阶段：数据加载")
    print(f"📂 加载JSON文件: {json_file}")

    builder = JacobianGraphBuilder(str(json_file))
    print("🔨 构建所有迭代的图...")
    graphs = builder.load_json_and_build_graphs()

    print(f"✅ JSON数据已加载")
    print(f"   - 迭代次数: {len(graphs)}")

    if len(builder.jacobians) > 0:
        print(f"\n📋 第一次迭代信息:")
        print(f"   - 源因子: {graphs[0].graph['source_factor']:.6f}")
        print(f"   - 节点电压: {builder.voltages[0][:3]}...")
        print(f"   - Jacobian矩阵大小: {builder.jacobians[0].shape}")
        cond = graphs[0].graph.get("jacobian_condition_number")
        cond_str = f"{cond:.3e}" if cond is not None else "N/A"
        print(f"   - Jacobian条件数: {cond_str}")

    # ========== 第2阶段：图构建统计 ==========
    print_header("第2阶段：图构建统计")
    print(f"✅ 已构建 {len(graphs)} 个图")

    if graphs:
        print(f"\n📊 图统计信息:")
        print(f"   - 每个图的节点数: {len(graphs[0].node_ids())}")
        print(f"   - 第一个图的边数: {len(graphs[0].edge_list())}")

        node_types = {}
        for node in graphs[0].node_ids():
            node_type = graphs[0].node_attrs[node]["node_type"]
            node_name = graphs[0].node_attrs[node]["name"]
            node_types.setdefault(node_type, []).append(node_name)

        print(f"\n   节点类型分布:")
        type_names = {0: '电压源', 1: '电源', 2: '内部', 3: '输入', 4: '输出', 5: '普通'}
        for t_id in sorted(node_types.keys()):
            print(f"      - {type_names.get(t_id, 'Unknown')}: {node_types[t_id]}")

        print(f"\n   Jacobian矩阵稀疏性分析:")
        num_nonzero = [len(G.edge_list()) for G in graphs]
        avg_nonzero = np.mean(num_nonzero) if num_nonzero else 0
        sparsity = 1 - avg_nonzero / (10 * 10) if graphs else 0
        print(f"      - 平均非零元素数: {avg_nonzero:.1f}")
        print(f"      - 稀疏度: {sparsity*100:.1f}%")

    # ========== 第3阶段：可视化 ==========
    print_header("第3阶段：图可视化")
    visualizer = CircuitGraphVisualizer()

    if graphs:
        print("🎨 可视化关键迭代...")
        key_iterations = [0, len(graphs)//4, len(graphs)//2, 3*len(graphs)//4, len(graphs)-1]

        for idx in key_iterations:
            if 0 <= idx < len(graphs):
                sf = graphs[idx].graph['source_factor']
                local_it = graphs[idx].graph['iteration']
                sf_str = f"{sf:.2f}".replace('.', 'p')
                save_path = output_base / f"index{idx}_iter{local_it}_SF{sf_str}.png"
                print(f"   - Index #{idx} (LocalIter={local_it}, SF={sf:.4f})...", end=" ")
                visualizer.visualize_graph(
                    graphs[idx],
                    layout='spring',
                    show_edge_labels=False,
                    save_path=str(save_path)
                )
                print("✅")

        print("\n🎨 创建对比图（4个关键迭代）...")
        comparison_iterations = [0, len(graphs)//3, 2*len(graphs)//3, len(graphs)-1]
        visualizer.visualize_comparison(
            graphs,
            iterations=comparison_iterations,
            layout='spring',
            save_dir=str(output_base)
        )
        print("✅ 对比图已完成")

        print("\n📈 可视化节点演化(VOUT节点)...")
        visualizer.visualize_node_evolution(graphs, node_id=3, save_dir=str(output_base))
        print("✅ 节点演化图已完成")

    # ========== 第4阶段：GNN数据准备 ==========
    print_header("第4阶段：GNN数据准备")
    print("🔨 创建GNN数据集...")
    dataset = GNNDataset(graphs)

    stats = dataset.get_statistics()
    print(f"\n📊 数据集统计:")
    print(f"   - 总迭代数: {stats['num_iterations']}")
    print(f"   - 节点数: {stats['num_nodes']}")
    print(f"   - 节点特征维度: {stats['node_feature_dim']} (电压V, 残差f, 节点类型)")
    print(f"   - 边特征维度: {stats['edge_feature_dim']}")
    print(f"   - 平均边数: {stats['avg_edges_per_graph']:.1f}")

    print(f"\n💾 保存数据到 {gnn_output}...")
    dataset.save_numpy(str(gnn_output / "numpy"), split_path=gnn_output / "train_val_split.json")

    # 从iteration_tracking日志提取标签
    print(f"\n🏷️  提取标签 (actual_changes.npy)...")
    tracking_log = find_iteration_log(json_file)
    labels_info = save_labels_from_log(tracking_log, gnn_output)

    stats = dataset.get_statistics(labels_info["actual_changes"])
    stats_file = gnn_output / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ 统计信息已保存: {stats_file}")

    # ========== 第5阶段：生成摘要 ==========
    print_header("第5阶段：生成摘要")

    summary = {
        'json_file': str(json_file),
        'iteration_log': str(tracking_log),
        'timestamp': timestamp,
        'device': device,
        'total_iterations': len(graphs),
        'num_nodes': stats['num_nodes'],
        'node_feature_dim': stats['node_feature_dim'],
        'edge_feature_dim': stats['edge_feature_dim'],
        'avg_edges_per_graph': stats['avg_edges_per_graph'],
        'labels_shape': list(labels_info['actual_changes'].shape),
        'output_directories': {
            'visualizations': str(output_base),
            'gnn_data': str(gnn_output),
            'numpy_format': str(gnn_output / "numpy"),
        },
    }

    summary_file = output_root / timestamp / device / "PIPELINE_SUMMARY.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("📝 生成摘要...")
    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"\n✅ 摘要已保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="从dc/logs按时间戳提取GNN数据")
    parser.add_argument("--logs-root", type=str, default="/home/hu/Diana-Sim/dc/logs",
                        help="dc日志目录")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="指定时间戳目录（如 20260204_053030），不指定则使用最新")
    parser.add_argument("--output-root", type=str, default="/home/hu/Diana-Sim/gdata",
                        help="输出目录")
    args = parser.parse_args()

    logs_root = Path(args.logs_root)
    output_root = Path(args.output_root)

    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = find_latest_timestamp(logs_root)

    run_dir = logs_root / timestamp
    if not run_dir.exists():
        raise FileNotFoundError(f"时间戳目录不存在: {run_dir}")

    print_header(f"📌 选择时间戳: {timestamp}")

    json_files = collect_newton_jsons(run_dir)
    if not json_files:
        raise FileNotFoundError(f"未找到Newton JSON文件: {run_dir}/*/newton_analysis/newton_dc_*.json")

    for json_file in json_files:
        process_single_json(json_file, output_root, timestamp)

    print_header("✨ 全部完成")
    print(f"输出目录: {output_root / timestamp}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
