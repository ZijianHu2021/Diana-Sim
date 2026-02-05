#!/usr/bin/env python3
"""
从 /home/hu/saratoga/dc/logs 按时间戳提取Newton数据，
构建图、生成可视化、并保存为GNN训练用.npy数据。

输出全部写到 /home/hu/saratoga/gdata 下，并按时间戳分目录。
"""

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

# 允许从 /home/hu/saratoga/graph 目录导入
GRAPH_DIR = Path("/home/hu/saratoga/graph")
sys.path.insert(0, str(GRAPH_DIR))

from graph_builder import JacobianGraphBuilder
from graph_visualizer import CircuitGraphVisualizer
from gnn_data_preparation import GNNDataset


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
        print(f"   - Jacobian条件数: {graphs[0].graph.get('jacobian_condition_number', 'N/A'):.3e}")

    # ========== 第2阶段：图构建统计 ==========
    print_header("第2阶段：图构建统计")
    print(f"✅ 已构建 {len(graphs)} 个图")

    if graphs:
        print(f"\n📊 图统计信息:")
        print(f"   - 每个图的节点数: {len(graphs[0].nodes())}")
        print(f"   - 第一个图的边数: {len(graphs[0].edges())}")

        node_types = {}
        for node in graphs[0].nodes():
            node_type = graphs[0].nodes[node]["node_type"]
            node_name = graphs[0].nodes[node]["name"]
            node_types.setdefault(node_type, []).append(node_name)

        print(f"\n   节点类型分布:")
        type_names = {0: '电压源', 1: '电源', 2: '内部', 3: '输入', 4: '输出', 5: '普通'}
        for t_id in sorted(node_types.keys()):
            print(f"      - {type_names.get(t_id, 'Unknown')}: {node_types[t_id]}")

        print(f"\n   Jacobian矩阵稀疏性分析:")
        import numpy as np
        num_nonzero = [len(G.edges()) for G in graphs]
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
    dataset.save_numpy(str(gnn_output / "numpy"))

    # 从iteration_tracking日志提取标签
    print(f"\n🏷️  提取标签 (actual_changes.npy)...")
    tracking_log = find_iteration_log(json_file)
    labels_info = save_labels_from_log(tracking_log, gnn_output)

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
    parser.add_argument("--logs-root", type=str, default="/home/hu/saratoga/dc/logs",
                        help="dc日志目录")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="指定时间戳目录（如 20260204_053030），不指定则使用最新")
    parser.add_argument("--output-root", type=str, default="/home/hu/saratoga/gdata",
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
