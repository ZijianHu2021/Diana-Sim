#!/usr/bin/env python3
"""
Newton迭代数据收集器
======================
从DC/TRAN解析中收集Newton-Raphson迭代的详细数据：
- Jacobian矩阵
- 残差向量（residual）
- 更新向量（delta）
- 收敛指标

数据保存为JSON格式，便于后续分析和可视化
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，确保numpy类型和数组被正确序列化为高精度JSON"""
    def default(self, obj):
        # numpy 数组转为列表（JSON 支持）
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy 整数类型转为 Python int
        elif isinstance(obj, (np.integer, np.intp)):
            return int(obj)
        # numpy 浮点类型转为 Python float（保持 float64 精度）
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        # numpy bool 转为 Python bool
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # 其他类型调用父类处理
        return super().default(obj)



class NewtonIterationData:
    """单步Newton迭代的数据"""
    
    def __init__(self, iteration: int, time: float = None, source_factor: float = None, nlscale: float = None):
        self.iteration = iteration
        self.time = time  # TRAN分析时的仿真时间
        self.source_factor = source_factor  # DC分析时的源步进因子（0.01 ~ 1.0）
        self.nlscale = nlscale  # 非线性缩放因子（0.3, 0.6, 1.0）
        self.x: Optional[np.ndarray] = None  # 当前节点电压向量
        self.jacobian: Optional[np.ndarray] = None
        self.residual: Optional[np.ndarray] = None
        self.delta_x: Optional[np.ndarray] = None
        self.node_names: List[str] = []
        self.max_residual = None
        self.max_delta = None
        self.l2_residual = None
        self.l2_delta = None
        self.convergence_metrics: Dict[str, float] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可JSON序列化的字典"""
        return {
            'iteration': self.iteration,
            'time': self.time,
            'source_factor': float(self.source_factor) if self.source_factor is not None else None,
            'nlscale': float(self.nlscale) if self.nlscale is not None else None,
            'x': self.x.tolist() if self.x is not None else None,
            'jacobian': self.jacobian.tolist() if self.jacobian is not None else None,
            'jacobian_shape': self.jacobian.shape if self.jacobian is not None else None,
            'jacobian_condition_number': float(np.linalg.cond(self.jacobian)) 
                                         if self.jacobian is not None and self.jacobian.shape[0] > 0 
                                         else None,
            'residual': self.residual.tolist() if self.residual is not None else None,
            'delta_x': self.delta_x.tolist() if self.delta_x is not None else None,
            'node_names': self.node_names,
            'max_residual': float(self.max_residual) if self.max_residual is not None else None,
            'max_delta': float(self.max_delta) if self.max_delta is not None else None,
            'l2_residual': float(self.l2_residual) if self.l2_residual is not None else None,
            'l2_delta': float(self.l2_delta) if self.l2_delta is not None else None,
            'convergence_metrics': {k: float(v) for k, v in self.convergence_metrics.items()},
        }


class NewtonDataCollector:
    """收集多步Newton迭代数据"""
    
    def __init__(self, analysis_type: str = "DC", verbose: bool = False):
        """
        Args:
            analysis_type: "DC" 或 "TRAN"
            verbose: 是否打印调试信息
        """
        self.analysis_type = analysis_type
        self.verbose = verbose
        self.iterations: List[NewtonIterationData] = []
        self.metadata: Dict[str, Any] = {
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'total_iterations': 0,
            'converged': False,
            'convergence_reason': None,
        }
    
    def add_iteration(self, data: NewtonIterationData):
        """添加一步迭代数据"""
        self.iterations.append(data)
        if self.verbose:
            print(f"  Iter {data.iteration}: max_res={data.max_residual:.3e}, "
                  f"max_delta={data.max_delta:.3e}")
    
    def add_jacobian(self, J: np.ndarray, node_names: List[str]):
        """为最后一步迭代添加Jacobian矩阵"""
        if self.iterations:
            self.iterations[-1].jacobian = J.copy()
            self.iterations[-1].node_names = node_names.copy()
    
    def add_residual(self, residual: np.ndarray):
        """为最后一步迭代添加残差"""
        if self.iterations:
            self.iterations[-1].residual = residual.copy()
            self.iterations[-1].max_residual = np.max(np.abs(residual))
            self.iterations[-1].l2_residual = np.linalg.norm(residual)
    
    def add_delta(self, delta_x: np.ndarray):
        """为最后一步迭代添加更新向量"""
        if self.iterations:
            self.iterations[-1].delta_x = delta_x.copy()
            self.iterations[-1].max_delta = np.max(np.abs(delta_x))
            self.iterations[-1].l2_delta = np.linalg.norm(delta_x)
    
    def set_convergence(self, converged: bool, reason: str = ""):
        """设置收敛状态"""
        self.metadata['converged'] = converged
        self.metadata['convergence_reason'] = reason
        self.metadata['total_iterations'] = len(self.iterations)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'metadata': self.metadata,
            'iterations': [it.to_dict() for it in self.iterations],
        }
    
    def save_to_json(self, filepath: str, compact: bool = False):
        """保存到JSON文件
        
        Args:
            filepath: 输出文件路径
            compact: 如果True，不保存完整的Jacobian和残差（节省空间）
        """
        data = self.to_dict()
        
        if compact:
            # 移除大型矩阵以减少文件大小
            for it in data['iterations']:
                it['jacobian'] = None
                it['residual'] = None
                it['delta_x'] = None
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
    def save_to_json(self, filepath: str, compact: bool = False):
        """保存到JSON文件，保持高精度
        
        Args:
            filepath: 输出文件路径
            compact: 如果True，不保存完整的Jacobian和残差（节省空间）
        """
        data = self.to_dict()
        
        if compact:
            # 移除大型矩阵以减少文件大小
            for it in data['iterations']:
                it['jacobian'] = None
                it['residual'] = None
                it['delta_x'] = None
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 使用自定义编码器并禁用 allow_nan 以保证精度
        # 使用较小的 separators 减小文件大小，但保留精度
        with open(filepath, 'w') as f:
            # indent=2 用于可读性，允许 NaN/Infinity 以保留原始值
            json.dump(data, f, indent=2, cls=NumpyEncoder, 
                     separators=(',', ': '), ensure_ascii=True)
        
        if self.verbose:
            print(f"\n✅ Newton data saved to: {filepath}")
            print(f"   File size: {Path(filepath).stat().st_size / 1024:.1f} KB")
            print(f"   Precision: Full float64 precision retained")
        
        if self.verbose:
            print(f"\n✅ Newton data saved to: {filepath}")
            print(f"   File size: {Path(filepath).stat().st_size / 1024:.1f} KB")


class NewtonDataAnalyzer:
    """分析Newton迭代数据"""
    
    def __init__(self, json_file: str):
        """从JSON文件加载数据"""
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.metadata = self.data['metadata']
        self.iterations = self.data['iterations']
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*80)
        print("📊 Newton Iteration Analysis Summary")
        print("="*80)
        
        print(f"\nAnalysis Type: {self.metadata['analysis_type']}")
        print(f"Converged: {self.metadata['converged']}")
        print(f"Total Iterations: {self.metadata['total_iterations']}")
        print(f"Convergence Reason: {self.metadata['convergence_reason']}")
        
        if not self.iterations:
            print("\n⚠️  No iteration data available")
            return
        
        print(f"\n{'Iter':<6} {'Max Res':<12} {'Max Δx':<12} {'L2 Res':<12} {'L2 Δx':<12} {'Cond Num':<12}")
        print("-" * 80)
        
        for it in self.iterations:
            max_res = it['max_residual'] or 0.0
            max_delta = it['max_delta'] or 0.0
            l2_res = it['l2_residual'] or 0.0
            l2_delta = it['l2_delta'] or 0.0
            cond_num = it['jacobian_condition_number'] or 0.0
            
            print(f"{it['iteration']:<6} {max_res:<12.3e} {max_delta:<12.3e} "
                  f"{l2_res:<12.3e} {l2_delta:<12.3e} {cond_num:<12.3e}")
        
        # 收敛趋势分析
        print(f"\n{'='*80}")
        print("📈 Convergence Trend Analysis")
        print("="*80)
        
        max_residuals = [it['max_residual'] for it in self.iterations if it['max_residual'] is not None]
        max_deltas = [it['max_delta'] for it in self.iterations if it['max_delta'] is not None]
        
        if max_residuals:
            print(f"\nMax Residual Trend:")
            print(f"  First: {max_residuals[0]:.3e}")
            print(f"  Last:  {max_residuals[-1]:.3e}")
            print(f"  Reduction: {max_residuals[0]/max_residuals[-1]:.3e}x" 
                  if max_residuals[-1] > 0 else "  Reduction: infinite")
            
            # 计算平均减少率
            reductions = []
            for i in range(1, len(max_residuals)):
                if max_residuals[i-1] > 0:
                    reductions.append(max_residuals[i-1] / max_residuals[i])
            
            if reductions:
                print(f"  Avg reduction per iter: {np.mean(reductions):.3f}x")
        
        if max_deltas:
            print(f"\nMax Delta Trend:")
            print(f"  First: {max_deltas[0]:.3e}")
            print(f"  Last:  {max_deltas[-1]:.3e}")
    
    def export_convergence_csv(self, filepath: str):
        """导出收敛数据为CSV"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Max_Residual', 'Max_Delta', 'L2_Residual', 'L2_Delta', 'Jacobian_Condition_Num'])
            
            for it in self.iterations:
                writer.writerow([
                    it['iteration'],
                    it['max_residual'] or '',
                    it['max_delta'] or '',
                    it['l2_residual'] or '',
                    it['l2_delta'] or '',
                    it['jacobian_condition_number'] or '',
                ])
        
        print(f"\n✅ Convergence data exported to: {filepath}")
    
    def print_jacobian_info(self, iteration: int = None):
        """打印指定迭代步的Jacobian矩阵详细信息
        
        Args:
            iteration: 迭代步数（如果为None，打印最后一步）
        """
        if iteration is None:
            it = self.iterations[-1] if self.iterations else None
        else:
            it = next((i for i in self.iterations if i['iteration'] == iteration), None)
        
        if it is None:
            print("⚠️  Iteration not found")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 Jacobian Matrix Analysis - Iteration {it['iteration']}")
        print(f"{'='*80}\n")
        
        if it['jacobian'] is not None:
            print(f"Shape: {it['jacobian_shape']}")
            print(f"Condition Number: {it['jacobian_condition_number']:.3e}")
            print(f"\nMatrix:")
            # 显示矩阵，每行限制宽度
            jacobian = np.array(it['jacobian']) if isinstance(it['jacobian'], list) else it['jacobian']
            with np.printoptions(precision=3, suppress=True, threshold=50, edgeitems=3):
                print(jacobian)
        else:
            print("⚠️  Jacobian matrix not available in this iteration")
    
    def print_residual_info(self, iteration: int = None):
        """打印指定迭代步的残差向量详细信息
        
        Args:
            iteration: 迭代步数（如果为None，打印最后一步）
        """
        if iteration is None:
            it = self.iterations[-1] if self.iterations else None
        else:
            it = next((i for i in self.iterations if i['iteration'] == iteration), None)
        
        if it is None:
            print("⚠️  Iteration not found")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 Residual Vector Analysis - Iteration {it['iteration']}")
        print(f"{'='*80}\n")
        
        if it['residual'] is not None:
            residual = np.array(it['residual'])
            node_names = it['node_names']
            
            print(f"Vector Size: {len(residual)}")
            print(f"Max Residual: {it['max_residual']:.3e}")
            print(f"L2 Norm: {it['l2_residual']:.3e}")
            print(f"\nDetailed Residual by Node:")
            print(f"{'Node':<15s} {'Index':<6s} {'Residual Value':<15s} {'Abs Value':<15s}")
            print("-" * 60)
            
            for idx, (name, val) in enumerate(zip(node_names, residual)):
                print(f"{name:<15s} {idx:<6d} {val:+.6e}  {abs(val):.6e}")
        else:
            print("⚠️  Residual vector not available in this iteration")
    
    def print_iteration_detail(self, iteration: int = None):
        """打印一个迭代步的完整详细信息
        
        Args:
            iteration: 迭代步数（如果为None，打印最后一步）
        """
        if iteration is None:
            it = self.iterations[-1] if self.iterations else None
        else:
            it = next((i for i in self.iterations if i['iteration'] == iteration), None)
        
        if it is None:
            print("⚠️  Iteration not found")
            return
        
        print(f"\n{'='*80}")
        print(f"🔍 Complete Iteration Details - Iteration {it['iteration']}")
        print(f"{'='*80}\n")
        
        print(f"Time: {it['time']}")
        print(f"Jacobian Shape: {it['jacobian_shape']}")
        print(f"Jacobian Condition Number: {it['jacobian_condition_number']:.3e}")
        print(f"\nConvergence Metrics:")
        print(f"  Max Residual: {it['max_residual']:.3e}")
        print(f"  Max Delta:    {it['max_delta']:.3e}")
        print(f"  L2 Residual:  {it['l2_residual']:.3e}")
        print(f"  L2 Delta:     {it['l2_delta']:.3e}")
        
        if it['convergence_metrics']:
            print(f"\nAdditional Metrics:")
            for key, val in it['convergence_metrics'].items():
                print(f"  {key}: {val:.3e}")
        
        # 显示residual向量
        self.print_residual_info(iteration)
        
        # 显示Jacobian矩阵（仅显示稀疏矩阵统计）
        if it['jacobian'] is not None:
            jacobian = np.array(it['jacobian']) if isinstance(it['jacobian'], list) else it['jacobian']
            sparsity = 1.0 - (np.count_nonzero(jacobian) / jacobian.size)
            print(f"\nJacobian Sparsity: {sparsity*100:.1f}%")
            print(f"Non-zero Elements: {np.count_nonzero(jacobian)} / {jacobian.size}")
    
    def export_jacobian_matrix(self, iteration: int, filepath: str):
        """导出指定迭代步的Jacobian矩阵为文本文件
        
        Args:
            iteration: 迭代步数
            filepath: 输出文件路径
        """
        it = next((i for i in self.iterations if i['iteration'] == iteration), None)
        
        if it is None or it['jacobian'] is None:
            print(f"⚠️  Cannot export: Iteration {iteration} or Jacobian not found")
            return
        
        jacobian = np.array(it['jacobian']) if isinstance(it['jacobian'], list) else it['jacobian']
        
        with open(filepath, 'w') as f:
            f.write(f"Jacobian Matrix - Iteration {iteration}\n")
            f.write(f"Shape: {jacobian.shape}\n")
            f.write(f"Condition Number: {it['jacobian_condition_number']:.3e}\n")
            f.write(f"\nMatrix:\n")
            np.savetxt(f, jacobian, fmt='%.6e')
        
        print(f"✅ Jacobian matrix exported to: {filepath}")
    
    def export_residual_vector(self, iteration: int, filepath: str):
        """导出指定迭代步的残差向量为文本文件
        
        Args:
            iteration: 迭代步数
            filepath: 输出文件路径
        """
        it = next((i for i in self.iterations if i['iteration'] == iteration), None)
        
        if it is None or it['residual'] is None:
            print(f"⚠️  Cannot export: Iteration {iteration} or Residual not found")
            return
        
        residual = np.array(it['residual'])
        node_names = it['node_names']
        
        with open(filepath, 'w') as f:
            f.write(f"Residual Vector - Iteration {iteration}\n")
            f.write(f"Size: {len(residual)}\n")
            f.write(f"Max Residual: {it['max_residual']:.3e}\n")
            f.write(f"L2 Norm: {it['l2_residual']:.3e}\n")
            f.write(f"\n{'Node':<20s} {'Index':<8s} {'Residual Value':<20s} {'Abs Value':<20s}\n")
            f.write("-" * 70 + "\n")
            
            for idx, (name, val) in enumerate(zip(node_names, residual)):
                f.write(f"{name:<20s} {idx:<8d} {val:+.12e}  {abs(val):.12e}\n")
        
        print(f"✅ Residual vector exported to: {filepath}")
    
    def generate_iteration_tracking_log(self, filepath: str = None) -> str:
        """生成详细的迭代追踪日志，显示每次迭代中节点电压的变化"""
        
        log_lines = []
        log_lines.append("=" * 100)
        log_lines.append("📊 Newton迭代详细追踪日志 - 节点电压变化过程")
        log_lines.append("=" * 100)
        log_lines.append("")
        
        # 添加初始信息
        log_lines.append("分析类型: " + self.metadata.get('analysis_type', 'N/A'))
        log_lines.append("总迭代次数: " + str(self.metadata.get('total_iterations', 0)))
        log_lines.append("收敛状态: " + str(self.metadata.get('converged', False)))
        log_lines.append("")
        
        if not self.iterations:
            log_lines.append("⚠️  没有迭代数据")
            log_content = "\n".join(log_lines)
            if filepath:
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(log_content)
            return log_content
        
        # 获取初始值（第一次迭代的x向量）
        first_x = self.iterations[0].get('x')
        node_names = self.iterations[0].get('node_names', [])
        
        if not first_x or not node_names:
            log_lines.append("⚠️  缺少初始节点值或节点名称信息")
            log_content = "\n".join(log_lines)
            if filepath:
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(log_content)
            return log_content
        
        # 查找初始状态（iteration=-1）或使用第一次迭代作为初始值
        init_x = None
        start_idx = 0
        
        if self.iterations[0].get('iteration', 0) == -1:
            # 找到了初始状态记录
            init_x = np.array(first_x) if isinstance(first_x, list) else first_x
            start_idx = 1  # 从第二个记录开始处理（第一个是初始状态）
            log_lines.append("🎯 初始节点电压（迭代开始前）:")
        else:
            # 没有初始状态记录，使用第一次迭代的x作为参考
            init_x = np.array(first_x) if isinstance(first_x, list) else first_x
            start_idx = 0
            log_lines.append("🎯 第一次迭代的节点电压:")
        
        log_lines.append("-" * 100)
        log_lines.append(f"{'节点名称':<20} {'电压值 (V)':<15} {'变化量 (V)':<15} {'说明':<50}")
        log_lines.append("-" * 100)
        for i, (node_name, v_init) in enumerate(zip(node_names, init_x)):
            log_lines.append(f"{node_name:<20} {v_init:>14.10f} {0.0:>14.10f} {'初始猜测值':<50}")
        log_lines.append("")
        
        # 遍历每个迭代步骤（从start_idx开始，跳过初始状态记录）
        # 注意：现在JSON中的x是更新后的值
        prev_x = init_x.copy()
        
        for iter_idx in range(start_idx, len(self.iterations)):
            iteration = self.iterations[iter_idx]
            iter_num = iteration.get('iteration', iter_idx)
            source_factor = iteration.get('source_factor', 'N/A')
            nlscale = iteration.get('nlscale', 'N/A')
            max_res = iteration.get('max_residual', 0)
            max_delta = iteration.get('max_delta', 0)
            l2_res = iteration.get('l2_residual', 0)
            l2_delta = iteration.get('l2_delta', 0)
            
            # 当前迭代的x值（这是更新后的值）
            current_x = iteration.get('x')
            residual = iteration.get('residual', [])
            delta_x_vec = iteration.get('delta_x', [])
            
            if current_x is None:
                continue
            
            current_x = np.array(current_x) if isinstance(current_x, list) else current_x
            residual = np.array(residual) if isinstance(residual, list) else residual
            delta_x_vec = np.array(delta_x_vec) if isinstance(delta_x_vec, list) else delta_x_vec
            
            # 检查是否是新的source_factor步骤的第一次迭代
            # 如果是，需要从上一步的最后x值或当前x获取旧值
            if iter_idx > 0:
                prev_iter = self.iterations[iter_idx - 1]
                prev_sf = prev_iter.get('source_factor', None)
                curr_sf = source_factor
                
                # 如果source_factor改变了，说明开始新的stepping阶段
                # 此时"旧值"应该是上一阶段的最终值（即上一次迭代的x）
                if prev_sf != curr_sf and prev_sf is not None:
                    # Source factor变化，prev_x已经是正确的（上一阶段的最终值）
                    pass
            
            log_lines.append("=" * 100)
            log_lines.append(f"📍 迭代 #{iter_num} (Source Factor: {source_factor}, NL Scale: {nlscale})")
            log_lines.append("=" * 100)
            
            # 收敛指标
            log_lines.append(f"收敛指标:")
            log_lines.append(f"  Max Residual: {max_res:.3e} V/A")
            log_lines.append(f"  Max Δx:       {max_delta:.3e} V/A")
            log_lines.append(f"  L2 Residual:  {l2_res:.3e}")
            log_lines.append(f"  L2 Δx:        {l2_delta:.3e}")
            log_lines.append("")
            
            # 详细的节点变化
            log_lines.append(f"{'节点名称':<20} {'残差':<15} {'更新量':<15} {'迭代前 (V)':<15} {'迭代后 (V)':<15} {'实际变化':<15}")
            log_lines.append("-" * 100)
            
            for i, node_name in enumerate(node_names):
                if i < len(residual):
                    res = residual[i]
                else:
                    res = 0.0
                
                if i < len(delta_x_vec):
                    delta = delta_x_vec[i]
                else:
                    delta = 0.0
                
                # 旧值 = 前一次迭代的结果
                old_val = prev_x[i]
                # 新值 = 当前迭代的结果
                new_val = current_x[i]
                # 实际变化（包含了damping效果）
                change = new_val - old_val
                
                # 格式化输出
                log_lines.append(
                    f"{node_name:<20} {res:>14.3e} {delta:>14.3e} {old_val:>14.10f} {new_val:>14.10f} {change:>14.10f}"
                )
            
            log_lines.append("")
            
            # 更新prev_x为当前迭代的结果，供下次使用
            prev_x = current_x.copy()
        
        # 显示最终值
        log_lines.append("=" * 100)
        log_lines.append("✅ 最终收敛节点电压:")
        log_lines.append("-" * 100)
        log_lines.append(f"{'节点名称':<20} {'最终值 (V)':<15} {'初始值 (V)':<15} {'总变化 (V)':<15}")
        log_lines.append("-" * 100)
        
        final_x = np.array(self.iterations[-1].get('x', first_x)) if self.iterations else first_x
        for node_name, v_final, v_init in zip(node_names, final_x, first_x):
            total_change = v_final - v_init
            log_lines.append(f"{node_name:<20} {v_final:>14.10f} {v_init:>14.10f} {total_change:>14.10f}")
        
        log_lines.append("")
        log_lines.append("=" * 100)
        
        log_content = "\n".join(log_lines)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(log_content)
            print(f"\n✅ 迭代追踪日志已保存到: {filepath}")
        
        return log_content
    
    def plot_convergence(self, filepath: str = None):
        """绘制收敛曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib not available, skipping plot")
            return
        
        iterations = [it['iteration'] for it in self.iterations]
        max_residuals = [it['max_residual'] or 0 for it in self.iterations]
        max_deltas = [it['max_delta'] or 0 for it in self.iterations]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Max residual
        ax1.semilogy(iterations, max_residuals, 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Max Residual (V or A)')
        ax1.set_title(f'{self.metadata["analysis_type"]} Analysis - Max Residual')
        ax1.grid(True, which='both', alpha=0.3)
        
        # Max delta
        ax2.semilogy(iterations, max_deltas, 's-', linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Max Δx (V or A)')
        ax2.set_title(f'{self.metadata["analysis_type"]} Analysis - Max Delta')
        ax2.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=150)
            print(f"\n✅ Convergence plot saved to: {filepath}")
        else:
            plt.show()


if __name__ == "__main__":
    # 测试示例
    collector = NewtonDataCollector("DC", verbose=True)
    
    # 模拟3步迭代
    for i in range(3):
        data = NewtonIterationData(i + 1)
        data.residual = np.random.randn(4) * 10**(-(i+1))
        data.delta_x = np.random.randn(4) * 10**(-(i+1))
        data.jacobian = np.random.randn(4, 4)
        data.node_names = ['VDD', 'VOUT', 'VSS', 'GND']
        data.max_residual = np.max(np.abs(data.residual))
        data.max_delta = np.max(np.abs(data.delta_x))
        data.l2_residual = np.linalg.norm(data.residual)
        data.l2_delta = np.linalg.norm(data.delta_x)
        
        collector.add_iteration(data)
    
    collector.set_convergence(True, "Converged to tolerance")
    
    # 保存
    output_file = "/tmp/test_newton_data.json"
    collector.save_to_json(output_file)
    
    # 分析
    analyzer = NewtonDataAnalyzer(output_file)
    analyzer.print_summary()
