#!/usr/bin/env python3
"""
DC Analysis Debug Script - Separated from analog_design

This script performs detailed DC and TRAN analysis with Newton iteration tracking,
completely separated from the analog_design directory. All logs and outputs are
saved to /home/hu/Diana-Sim/dc/logs/ instead.

Usage:
    python dc/debug_dc_analysis.py [--config config.yaml]
"""

import sys
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# 添加 analog_design 到路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent / "analog_design"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analog_design" / "python_sim" / "src"))

from tests.test_python_bsim4 import TestGoldenComparison


class TeeOutput:
    """同时输出到终端和文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def load_config(config_path):
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"⚠️  Warning: Config file not found: {config_path}")
        print(f"   Using default configuration")
        return None
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded from: {config_path.name}")
        return config
    except ImportError:
        print(f"⚠️  Warning: PyYAML not installed, skipping config file")
        print(f"   Install with: pip install pyyaml")
        return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def apply_config_to_solver(dc_analyzer, config):
    """根据配置应用参数到求解器
    
    Args:
        dc_analyzer: BSIM4DCAnalyzer实例
        config: 配置字典
    """
    if config is None:
        return
    
    solver_cfg = config.get('solver', {})
    
    # 应用求解器参数
    if 'gmin' in solver_cfg:
        dc_analyzer.gmin = solver_cfg['gmin']
    if 'enable_bias_limiting' in solver_cfg:
        dc_analyzer.enable_bias_limiting = solver_cfg['enable_bias_limiting']
    if 'enable_ieq' in solver_cfg:
        dc_analyzer.enable_ieq = solver_cfg['enable_ieq']
    if 'enable_nlscale' in solver_cfg:
        dc_analyzer.enable_nlscale = solver_cfg['enable_nlscale']
    if 'enable_kcl_residual' in solver_cfg:
        dc_analyzer.enable_kcl_residual = solver_cfg['enable_kcl_residual']
    if 'enable_polish' in solver_cfg:
        dc_analyzer.enable_polish = solver_cfg['enable_polish']


def get_newton_params_from_config(config):
    """从配置中获取牛顿方法参数
    
    Args:
        config: 配置字典
    
    Returns:
        dict: 牛顿方法参数
    """
    if config is None:
        # 返回默认参数
        return {
            'tol': 1e-9,
            'max_iter': 100,
            'polish_iters': 0,
            'source_factors': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0],
            'nlscale_factors': [1.0],
            'min_iter_per_step': 1,
            'force_full_iters': False,
        }
    
    newton_cfg = config.get('newton', {})
    return {
        'tol': newton_cfg.get('tol', 1e-9),
        'max_iter': newton_cfg.get('max_iter', 100),
        'polish_iters': newton_cfg.get('polish_iters', 0),
        'source_factors': newton_cfg.get('source_factors', [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]),
        'nlscale_factors': newton_cfg.get('nlscale_factors', [1.0]),
        'min_iter_per_step': newton_cfg.get('min_iter_per_step', 1),
        'force_full_iters': newton_cfg.get('force_full_iters', False),
    }


def print_config_summary(config):
    """打印配置摘要
    
    Args:
        config: 配置字典
    """
    if config is None:
        print(f"ℹ️  Using default configuration (no config file)")
        return
    
    print(f"\n📋 Configuration Summary:")
    print(f"{'─'*80}")
    
    solver_cfg = config.get('solver', {})
    print(f"Solver Options:")
    print(f"  GMIN:              {solver_cfg.get('gmin', 1e-10)}")
    print(f"  Bias limiting:     {solver_cfg.get('enable_bias_limiting', False)}")
    print(f"  IEQ mode:          {solver_cfg.get('enable_ieq', False)}")
    print(f"  NL scaling:        {solver_cfg.get('enable_nlscale', True)}")
    print(f"  KCL residual:      {solver_cfg.get('enable_kcl_residual', False)}")
    print(f"  Polish:            {solver_cfg.get('enable_polish', True)}")
    
    newton_cfg = config.get('newton', {})
    print(f"\nNewton-Raphson Options:")
    print(f"  Convergence tol:   {newton_cfg.get('tol', 1e-9)}")
    print(f"  Max iterations:    {newton_cfg.get('max_iter', 100)}")
    print(f"  Polish iters:      {newton_cfg.get('polish_iters', 0)}")
    print(f"  Source factors:    {newton_cfg.get('source_factors', [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0])}")
    print(f"  Min iters/step:    {newton_cfg.get('min_iter_per_step', 1)}")
    print(f"  Force full iters:  {newton_cfg.get('force_full_iters', False)}")
    
    collection_cfg = config.get('collection', {})
    print(f"\nCollection Options:")
    print(f"  Enable collection: {collection_cfg.get('enable_newton_collection', True)}")
    print(f"  Target iterations: {collection_cfg.get('target_iterations', 'None (use convergence)')}")
    
    print(f"{'─'*80}\n")


def detailed_golden_test(tc, test_name, netlist, golden_path, gnn_initial_guess=None, output_dir=None, config=None):
    """详细的黄金数据比较测试，包含计算过程输出
    
    Args:
        tc: 测试用例实例
        test_name: 测试名称
        netlist: 电路网表
        golden_path: Golden数据路径
        gnn_initial_guess: GNN预测的初始电压字典 (可选)
        output_dir: 输出目录路径 (如果为None，使用当前脚本所在目录下的logs)
        config: 配置字典 (可选)
    """
    
    print(f"\n{'='*80}")
    print(f"🔍 Detailed Analysis: {test_name}")
    print(f"{'='*80}\n")
    
    # 从test_name中提取测试类型 (PMOS 或 NMOS)
    test_type = "pmos" if "PMOS" in test_name else ("nmos" if "NMOS" in test_name else "unknown")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = Path(__file__).parent / "logs"
    else:
        output_dir = Path(output_dir)
    
    # 为每个测试类型创建子目录
    output_dir = output_dir / test_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导入必要的模块（已经在脚本开头添加到sys.path了）
    from analog_design.simulator import parse_netlist, BSIM4DCAnalyzer, BSIM4TRANAnalyzer
    from newton_data_collector import NewtonDataCollector, NewtonIterationData, NewtonDataAnalyzer
    
    # 解析netlist
    circuit = parse_netlist(netlist)
    
    print("📋 Circuit Components:")
    for name, comp in circuit.components.items():
        comp_type = type(comp).__name__
        print(f"  {name:15s} : {comp_type}")
        if hasattr(comp, 'nodes'):
            print(f"    Nodes: {comp.nodes}")
        if hasattr(comp, 'w') and hasattr(comp, 'l'):
            print(f"    W/L: {comp.w*1e6:.2f}um / {comp.l*1e9:.1f}nm")
    
    # === DC 解析 ===
    print(f"\n{'='*80}")
    print("⚡ DC Analysis (Detailed)")
    print(f"{'='*80}\n")
    
    # 如果提供了GNN初始猜测，显示信息
    if gnn_initial_guess is not None:
        print(f"🤖 Using GNN-predicted initial voltages:")
        print(f"  {'Node':<15s} {'Initial Voltage (V)':>20s}")
        print(f"  {'-'*35}")
        for node_name in sorted(gnn_initial_guess.keys()):
            print(f"  {node_name:<15s} {gnn_initial_guess[node_name]:>20.6f}")
        print()
    else:
        print(f"⚠️  No GNN initial guess provided, using default initial values")
        print()
    
    # 启用详细输出
    os.environ['DEBUG_DC'] = '1'
    
    # 从配置获取求解器参数
    solver_cfg = config.get('solver', {}) if config else {}
    
    dc_analyzer = BSIM4DCAnalyzer(
        circuit,
        gmin=solver_cfg.get('gmin', 1e-10),
        enable_bias_limiting=solver_cfg.get('enable_bias_limiting', False),
        enable_ieq=solver_cfg.get('enable_ieq', False),
        enable_nlscale=solver_cfg.get('enable_nlscale', True),
        enable_kcl_residual=solver_cfg.get('enable_kcl_residual', False),
        enable_polish=solver_cfg.get('enable_polish', True),
    )
    
    # 从配置获取数据收集参数
    collection_cfg = config.get('collection', {}) if config else {}
    enable_collection = collection_cfg.get('enable_newton_collection', True)
    
    # 启用Newton数据收集
    dc_analyzer.enable_newton_collection(enable=enable_collection, verbose=collection_cfg.get('verbose', True))
    
    # 从配置获取牛顿方法参数
    newton_cfg = config.get('newton', {}) if config else {}
    source_factors = newton_cfg.get('source_factors', [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0])
    
    # 如果提供了GNN初始值且配置中启用快速收敛，使用单步
    if gnn_initial_guess is not None and config and config.get('gnn', {}).get('use_gnn_for_fast_convergence', False):
        source_factors = [1.0]

    node_voltages, mos_currents = dc_analyzer.solve(
        verbose=True,
        tol=newton_cfg.get('tol', 1e-9),
        max_iter=newton_cfg.get('max_iter', 100),
        polish_iters=newton_cfg.get('polish_iters', 0),
        source_factors=source_factors,
        nlscale_factors=newton_cfg.get('nlscale_factors', [1.0]),
        min_iter_per_step=newton_cfg.get('min_iter_per_step', 1),
        force_full_iters=newton_cfg.get('force_full_iters', False),
        initial_guess=gnn_initial_guess,
    )
    
    # 保存收集的Newton迭代数据到dc目录
    if dc_analyzer._newton_collector is not None:
        print(f"\n{'='*80}")
        print("💾 Saving Newton Iteration Data")
        print(f"{'='*80}\n")
        
        # 获取收集器
        collector = dc_analyzer._newton_collector
        
        # 标记求解完成
        collector.set_convergence(True, "DC analysis completed successfully")
        
        # 创建输出目录（在dc/logs/{type}/newton_analysis下）
        log_dir = output_dir / "newton_analysis"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳和类型标签的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        type_label = f"_{test_type}" if test_type != "unknown" else ""
        
        # 保存JSON数据
        json_file = log_dir / f"newton_dc{type_label}_{timestamp}.json"
        collector.save_to_json(str(json_file))
        print(f"✅ Newton iteration data saved to:")
        print(f"   {json_file}")
        print(f"   (包含 {len(collector.iterations)} 个迭代步)")
        
        # 分析并显示摘要
        try:
            analyzer = NewtonDataAnalyzer(str(json_file))
            
            # 打印摘要
            print(f"\n{'─'*80}")
            print("📊 Newton Convergence Summary:")
            print(f"{'─'*80}")
            analyzer.print_summary()
            
            # 打印最后一个迭代的详细信息
            print(f"\n{'─'*80}")
            print("📈 Last Iteration Details:")
            print(f"{'─'*80}")
            analyzer.print_iteration_detail()
            
            # 导出Jacobian和残差
            if collector.iterations:
                last_iter_num = collector.iterations[-1].iteration
                
                print(f"\n{'─'*80}")
                print("💾 Exporting Matrix and Vector Data:")
                print(f"{'─'*80}")
                
                try:
                    jac_file = log_dir / f"jacobian_iter{last_iter_num}{type_label}_{timestamp}.txt"
                    analyzer.export_jacobian_matrix(last_iter_num, str(jac_file))
                    print(f"✅ Jacobian matrix exported to: {jac_file}")
                except Exception as e:
                    print(f"⚠️  Failed to export Jacobian: {e}")
                
                try:
                    res_file = log_dir / f"residual_iter{last_iter_num}{type_label}_{timestamp}.txt"
                    analyzer.export_residual_vector(last_iter_num, str(res_file))
                    print(f"✅ Residual vector exported to: {res_file}")
                except Exception as e:
                    print(f"⚠️  Failed to export Residual: {e}")
            
            # 生成收敛曲线图
            try:
                plot_file = log_dir / f"convergence{type_label}_{timestamp}.png"
                analyzer.plot_convergence(str(plot_file))
                print(f"✅ Convergence plot saved to: {plot_file}")
            except Exception as e:
                print(f"⚠️  Failed to generate plot: {e}")
            
            # 生成详细的迭代追踪日志
            try:
                tracking_log_file = log_dir / f"iteration_tracking{type_label}_{timestamp}.log"
                analyzer.generate_iteration_tracking_log(str(tracking_log_file))
                print(f"✅ Iteration tracking log saved to: {tracking_log_file}")
            except Exception as e:
                print(f"⚠️  Failed to generate iteration tracking log: {e}")
        
        except Exception as e:
            print(f"⚠️  Warning: Failed to analyze Newton data: {e}")
    
    print("\n📊 DC Solution:")
    print(f"  {'Node':<15s} {'Voltage (V)':>12s}")
    print(f"  {'-'*27}")
    for node in sorted(node_voltages.keys()):
        if node != "0":
            print(f"  {node:<15s} {node_voltages[node]:12.6f}")
    
    print(f"\n  {'MOSFET':<15s} {'Current (uA)':>12s}")
    print(f"  {'-'*27}")
    for name in sorted(mos_currents.keys()):
        print(f"  {name:<15s} {mos_currents[name]*1e6:12.3f}")
    
    # === MOSFET 偏置点详细信息 ===
    print(f"\n{'='*80}")
    print("🔬 MOSFET Operating Points")
    print(f"{'='*80}\n")
    
    for name, comp in circuit.components.items():
        if hasattr(comp, 'mos_type'):
            nd_name = comp.nodes[0]
            ng_name = comp.nodes[1]
            ns_name = comp.nodes[2]
            nb_name = comp.nodes[3]
            
            vd = node_voltages.get(nd_name, 0.0)
            vg = node_voltages.get(ng_name, 0.0)
            vs = node_voltages.get(ns_name, 0.0)
            vb = node_voltages.get(nb_name, 0.0)
            
            vgs = vg - vs
            vds = vd - vs
            vbs = vb - vs
            
            print(f"{name} ({comp.mos_type.upper()}):")
            print(f"  Terminal Voltages:")
            print(f"    D={nd_name}: {vd:.6f} V")
            print(f"    G={ng_name}: {vg:.6f} V")
            print(f"    S={ns_name}: {vs:.6f} V")
            print(f"    B={nb_name}: {vb:.6f} V")
            print(f"  Bias:")
            print(f"    Vgs = {vgs:+.6f} V")
            print(f"    Vds = {vds:+.6f} V")
            print(f"    Vbs = {vbs:+.6f} V")
            print(f"  Current:")
            print(f"    Ids = {mos_currents.get(name, 0)*1e6:+.3f} uA")
            
            # BSIM4 评估
            if name in dc_analyzer.bsim4_devices:
                device = dc_analyzer.bsim4_devices[name]
                result = device.evaluate(vgs, vds, vbs)
                print(f"  BSIM4 Small-Signal Parameters:")
                print(f"    gm   = {result.get('gm', 0)*1e6:.3f} uS")
                print(f"    gds  = {result.get('gds', 0)*1e6:.3f} uS")
                print(f"    gmb  = {result.get('gmb', 0)*1e6:.3f} uS")
                print(f"    Gdpr = {result.get('Gdpr', 0)*1e6:.3f} uS")
                print(f"    Gspr = {result.get('Gspr', 0)*1e6:.3f} uS")
            print()
    
    # === TRAN 解析 ===
    print(f"\n{'='*80}")
    print("📈 TRAN Analysis (Detailed)")
    print(f"{'='*80}\n")
    
    os.environ['DEBUG_TRAN'] = '1'
    
    tran_analyzer = BSIM4TRANAnalyzer(
        circuit, dc_analyzer, node_voltages,
        use_mos_caps=True,
        node_cap=10e-15,
    )
    
    times, waves = tran_analyzer.solve(tstop=300e-9, dt=0.05e-9, verbose=True)
    
    print(f"\n✅ TRAN Complete:")
    print(f"  Total steps: {len(times)}")
    print(f"  Final time: {times[-1]*1e9:.3f} ns")
    print(f"  Avg dt: {np.mean(np.diff(times))*1e12:.2f} ps")
    
    # === 与Golden数据比较 ===
    print(f"\n{'='*80}")
    print("📊 Golden Data Comparison")
    print(f"{'='*80}\n")
    
    golden = pd.read_csv(golden_path)
    print(f"Golden file: {golden_path.name}")
    print(f"Golden points: {len(golden)}")
    print(f"Golden columns: {list(golden.columns)}")
    
    # 找到时间列和输出列
    t_col = None
    v_col = None
    for col in golden.columns:
        if col.lower() in ('time', 't'):
            t_col = col
        if 'VOUT' in col.upper() and v_col is None:
            v_col = col
    
    if t_col is None or v_col is None:
        print(f"❌ Cannot find time or VOUT column in golden data")
        return
    
    t_gold = golden[t_col].values
    v_gold = golden[v_col].values
    
    # 插值到golden时间点
    v_sim_interp = np.interp(t_gold, times, waves['VOUT'])
    
    # 计算误差
    abs_err = np.abs(v_sim_interp - v_gold)
    rel_err = abs_err / np.maximum(np.abs(v_gold), 1e-3)
    
    # 统计信息
    print(f"\n📈 Error Statistics:")
    print(f"  {'Metric':<25s} {'Value':>15s}")
    print(f"  {'-'*40}")
    print(f"  {'Max Absolute Error':<25s} {np.max(abs_err)*1e3:12.3f} mV")
    print(f"  {'Mean Absolute Error':<25s} {np.mean(abs_err)*1e3:12.3f} mV")
    print(f"  {'RMS Error':<25s} {np.sqrt(np.mean(abs_err**2))*1e3:12.3f} mV")
    print(f"  {'Max Relative Error':<25s} {np.max(rel_err)*100:12.2f} %")
    print(f"  {'Mean Relative Error':<25s} {np.mean(rel_err)*100:12.2f} %")
    
    # 找出最大误差点
    max_err_idx = np.argmax(abs_err)
    max_err_time = t_gold[max_err_idx]
    max_err_val = abs_err[max_err_idx]
    
    print(f"\n⚠️  Maximum Error Point:")
    print(f"  Time: {max_err_time*1e9:.3f} ns")
    print(f"  Golden: {v_gold[max_err_idx]:.6f} V")
    print(f"  Sim:    {v_sim_interp[max_err_idx]:.6f} V")
    print(f"  Error:  {max_err_val*1e3:+.3f} mV ({rel_err[max_err_idx]*100:.2f}%)")
    
    # 时间段误差分析
    print(f"\n📊 Error by Time Segments:")
    segments = [
        (0, 10e-9, "Initial (0-10ns)"),
        (10e-9, 20e-9, "Rising Edge (10-20ns)"),
        (20e-9, 120e-9, "High State (20-120ns)"),
        (120e-9, 130e-9, "Falling Edge (120-130ns)"),
        (130e-9, 300e-9, "Low State (130-300ns)"),
    ]
    
    for t_start, t_end, label in segments:
        mask = (t_gold >= t_start) & (t_gold <= t_end)
        if mask.any():
            seg_abs_err = abs_err[mask]
            seg_rel_err = rel_err[mask]
            print(f"\n  {label}:")
            print(f"    Max abs error: {np.max(seg_abs_err)*1e3:8.3f} mV")
            print(f"    Mean abs error: {np.mean(seg_abs_err)*1e3:7.3f} mV")
            print(f"    Max rel error: {np.max(seg_rel_err)*100:8.2f} %")
    
    # 清理环境变量
    os.environ.pop('DEBUG_DC', None)
    os.environ.pop('DEBUG_TRAN', None)


def load_gnn_initial_guess(gnn_guess_path):
    """从JSON文件加载GNN预测的初始电压
    
    Args:
        gnn_guess_path: GNN初始猜测JSON文件路径
    
    Returns:
        dict: 节点名称到初始电压的映射，如果文件不存在返回None
    """
    if gnn_guess_path is None:
        return None
    
    gnn_path = Path(gnn_guess_path)
    if not gnn_path.exists():
        print(f"⚠️  Warning: GNN initial guess file not found: {gnn_path}")
        return None
    
    try:
        import json
        with open(gnn_path, 'r') as f:
            data = json.load(f)
        
        initial_voltages = data.get("initial_voltages", {})
        print(f"✅ Loaded GNN initial guess from: {gnn_path.name}")
        print(f"   Based on training: {data.get('based_on_training', 'N/A')}")
        print(f"   Num test samples: {data.get('num_test_samples', 'N/A')}")
        print()
        
        return initial_voltages
    except Exception as e:
        print(f"❌ Error loading GNN initial guess: {e}")
        return None


def main():
    """执行详细的调试测试"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Debug DC analysis with optional GNN initial guess")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file (default: config.yaml in dc directory)")
    # GNN initial guess is now controlled via YAML config only
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 创建 dc/logs 目录（而不是 analog_design/logs）
    dc_root = Path(__file__).parent
    log_base_dir = dc_root / "logs"
    log_base_dir.mkdir(exist_ok=True)
    
    # 生成带时间戳的文件夹和文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 为这次执行创建一个以时间戳命名的文件夹
    execution_dir = log_base_dir / timestamp
    execution_dir.mkdir(parents=True, exist_ok=True)
    
    # 重定向输出
    main_log_file = execution_dir / f"debug_dc_detailed_{timestamp}.log"
    tee = TeeOutput(str(main_log_file))
    sys.stdout = tee
    sys.stderr = tee
    
    print(f"📝 DC Analysis Debug Log")
    print(f"{'='*80}")
    print(f"Log file: {main_log_file}")
    print(f"Timestamp: {timestamp}")
    print(f"Execution directory: {execution_dir}")
    
    # 保存配置文件到执行目录
    if config is not None:
        config_source = args.config if args.config else (dc_root / "config.yaml")
        config_dest = execution_dir / "config_used.yaml"
        try:
            import shutil
            shutil.copy(config_source, config_dest)
            print(f"✅ Configuration saved to: {config_dest.name}")
        except Exception as e:
            print(f"⚠️  Failed to save config: {e}")
    
    # 打印配置摘要
    print_config_summary(config)
    
    # 加载GNN初始猜测（仅通过YAML配置）
    gnn_initial_guess = None
    gnn_guess_source = None

    gnn_cfg = config.get('gnn', {}) if config else {}
    if gnn_cfg.get('enable_initial_guess', False):
        gnn_file = gnn_cfg.get('initial_guess_file')
        if gnn_file:
            # 处理相对路径（相对于Diana-Sim工作区目录）
            if not os.path.isabs(gnn_file):
                workspace_root = dc_root.parent
                gnn_file = workspace_root / gnn_file
            gnn_guess_source = f"YAML config: {gnn_file}"
            gnn_initial_guess = load_gnn_initial_guess(str(gnn_file))
        else:
            print("⚠️  GNN initial guess enabled but initial_guess_file is not set")

    if gnn_guess_source:
        print(f"✅ GNN initial guess source: {gnn_guess_source}\n")
    
    print(f"{'='*80}\n")
    
    try:
        # Golden数据路径仍然从analog_design读取
        analog_design_root = dc_root.parent / "analog_design"
        
        # 检查是否要运行PMOS测试
        test_cfg = config.get('test', {}) if config else {}
        run_pmos = test_cfg.get('run_pmos', True)
        run_nmos = test_cfg.get('run_nmos', True)
        
        # === Test 1: PMOS ===
        if run_pmos:
            pmos_netlist = """
            M0 (VOUT VIN VSS VOUT) mp25od33_svt l=460n w=400n multi=1 nf=1
            V2 (VIN 0) vsource dc=1.0 type=pulse val0=1 val1=1.5 period=30n delay=10n rise=10p fall=10p width=10n
            V1 (VDD 0) vsource dc=3 type=dc
            V0 (VSS 0) vsource dc=0 type=dc
            I4 (VDD I4_MINUS) isource dc=10u type=dc
            R0 (I4_MINUS VOUT) resistor r=2.2K
            C0 (VOUT VSS) capacitor c=10f
            """
            pmos_golden = analog_design_root / "data" / "golden" / "tran_sf_pmos_1V_1p5V.csv"
            
            tc = TestGoldenComparison('test_pmos_against_golden')
            tc.setUp()
            detailed_golden_test(tc, "PMOS Source Follower", pmos_netlist, pmos_golden, 
                               gnn_initial_guess, output_dir=execution_dir, config=config)
        
        # === Test 2: NMOS ===
        if run_nmos:
            if run_pmos:
                print(f"\n\n{'#'*80}\n")
            
            nmos_netlist = """
            M0 (VDD VIN VOUT VOUT) mn25od33_svt l=550n w=800n multi=1 nf=1
            V2 (VIN 0) vsource dc=3.0 type=pulse val0=3 val1=2.5 delay=10n rise=10p fall=10p width=10n period=30n
            V1 (VDD 0) vsource dc=3 type=dc
            V0 (VSS 0) vsource dc=0 type=dc
            I4 (I4_PLUS 0) isource dc=10u type=dc
            R0 (VOUT I4_PLUS) resistor r=2.2K
            Cload (VOUT 0) capacitor c=10f
            """
            nmos_golden = analog_design_root / "data" / "golden" / "tran_sf_3V_2p5V.csv"
            
            tc = TestGoldenComparison('test_nmos_against_golden')
            tc.setUp()
            detailed_golden_test(tc, "NMOS Source Follower", nmos_netlist, nmos_golden, 
                               gnn_initial_guess, output_dir=execution_dir, config=config)
        
        print(f"\n{'='*80}")
        print("🎉 All detailed tests completed!")
        print(f"{'='*80}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}", file=tee.terminal)
        import traceback
        traceback.print_exc()
    
    finally:
        # 恢复标准输出
        sys.stdout = tee.terminal
        sys.stderr = sys.__stderr__
        tee.close()
        print(f"\n✅ Execution directory: {execution_dir}")
        print(f"✅ Main log saved to: {main_log_file}")
        print(f"\n📁 Output structure:")
        print(f"   dc/logs/{timestamp}/")
        print(f"   ├── debug_dc_detailed_{timestamp}.log")
        print(f"   ├── pmos/")
        print(f"   │   └── newton_analysis/")
        print(f"   │       ├── newton_dc_pmos_*.json")
        print(f"   │       ├── jacobian_iter*_pmos_*.txt")
        print(f"   │       ├── residual_iter*_pmos_*.txt")
        print(f"   │       ├── convergence_pmos_*.png")
        print(f"   │       └── iteration_tracking_pmos_*.log")
        print(f"   └── nmos/")
        print(f"       └── newton_analysis/")
        print(f"           ├── newton_dc_nmos_*.json")
        print(f"           ├── jacobian_iter*_nmos_*.txt")
        print(f"           ├── residual_iter*_nmos_*.txt")
        print(f"           ├── convergence_nmos_*.png")
        print(f"           └── iteration_tracking_nmos_*.log")


if __name__ == "__main__":
    main()
