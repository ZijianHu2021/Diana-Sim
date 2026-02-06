#!/usr/bin/env python3
"""
BSIM4 Circuit Analyzer (DC + TRAN)
==================================
BSIM4モデルを使用した回路解析器
- Julia側のVARSTEP問題（Gdpr >> gm による内部ノード希釈）を回避
- 外部ノードのみを使用し、直接gmをスタンプ

設計方針:
- 内部ノード（dp/sp）を使用しない
- MINIMALアナライザと同様、外部端子に直接Jacobianをスタンプ
- BSIM4のGdpr/Gspr（シリーズ抵抗）は無視（精度より安定性優先）

依存ライブラリ:
- numpy
- bsim4_python_wrapper.py
"""

import numpy as np
import os
import csv
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# BSIM4ラッパーをインポート
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
default_model_dir = os.path.join(project_root, "data", "model", "SPECTRE")
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# デバイスパラメータ計算をインポート
try:
    from cal_device_param import BSIM4Calculator
    _HAS_CALCULATOR = True
except ImportError:
    _HAS_CALCULATOR = False
    print("Warning: cal_device_param.BSIM4Calculator not available")

# Newton数据收集（新增）
try:
    from newton_data_collector import NewtonDataCollector, NewtonIterationData
    NEWTON_COLLECTION_AVAILABLE = True
except ImportError:
    NEWTON_COLLECTION_AVAILABLE = False

try:
    from bsim4_python_wrapper import BSIM4Device, find_bsim4_library
except ImportError as e:
    print(f"Warning: Could not import bsim4_python_wrapper: {e}")
    BSIM4Device = None


# =============================================================================
# モデルパラメータの読み込み
# =============================================================================

def load_model_params_from_scs(model_name: str, model_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Spectre形式の_computed.scsファイルからモデルパラメータを読み込む
    
    Args:
        model_name: モデル名（例: "mn25od33_svt", "mp25od33_svt"）
        model_dir: モデルファイルのディレクトリ（Noneの場合は自動検索）
    
    Returns:
        パラメータ辞書
    """
    if model_dir is None:
        model_dir = default_model_dir
    
    # _computed.scsファイルを探す
    computed_file = os.path.join(model_dir, f"{model_name}_computed.scs")
    
    if not os.path.exists(computed_file):
        # computed.scsがない場合、元の.scsファイルを試す
        computed_file = os.path.join(model_dir, f"{model_name}.scs")
        if not os.path.exists(computed_file):
            print(f"Warning: Model file not found for {model_name}")
            return {}
    
    params = {}
    in_model_block = False
    
    try:
        with open(computed_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # コメント行をスキップ
                if line.startswith('//') or line.startswith('*'):
                    continue
                
                # model ブロックの開始
                if line.startswith('model ') and (model_name in line or 'nmos' in line or 'pmos' in line):
                    in_model_block = True
                    # nmos/pmos判定
                    if 'pmos' in line.lower():
                        params['type'] = 'pmos'
                    else:
                        params['type'] = 'nmos'
                    continue
                
                # model ブロックの終了
                if line == '}':
                    in_model_block = False
                    continue
                
                # パラメータ行
                if in_model_block and '=' in line:
                    # "param=value" 形式
                    parts = line.rstrip(',').split('=')
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        val_str = parts[1].strip()
                        try:
                            val = float(val_str)
                            params[key] = val
                        except ValueError:
                            params[key] = val_str
    
    except Exception as e:
        print(f"Warning: Failed to load model params from {computed_file}: {e}")
    
    return params


def get_default_model_params(mos_type: str) -> Dict[str, Any]:
    """デフォルトのモデルパラメータを返す"""
    # 基本的なBSIM4パラメータ
    params = {
        'type': mos_type,
        'toxe': 5.75e-9,
        'toxp': 4.65e-9,
        'toxm': 5.75e-9,
        'vth0': 0.5 if mos_type == 'nmos' else -0.5,
        'k1': 0.5,
        'k2': 0.0,
        'u0': 0.04,
        'vsat': 1.5e5,
        'rdsw': 200.0,
        'ndep': 1.7e17,
        'nsub': 1e16,
        'ngate': 1e20,
        'capmod': 2,
        'mobmod': 0,
        'rdsmod': 0,
    }
    return params


# =============================================================================
# 回路素子
# =============================================================================

class ComponentType(Enum):
    VSOURCE = "vsource"
    ISOURCE = "isource"
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    MOSFET = "mosfet"


@dataclass
class Component:
    """回路素子の基底クラス"""
    name: str
    nodes: List[str]
    comp_type: ComponentType
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VSource(Component):
    """電圧源"""
    dc_value: float = 0.0
    pulse_params: Optional[Dict[str, float]] = None
    
    def get_voltage(self, t: float) -> float:
        """時刻tでの電圧値を取得"""
        if self.pulse_params is None:
            return self.dc_value
        
        p = self.pulse_params
        val0 = p.get('val0', self.dc_value)
        val1 = p.get('val1', self.dc_value)
        delay = p.get('delay', 0.0)
        rise = p.get('rise', 1e-12)
        fall = p.get('fall', 1e-12)
        width = p.get('width', 1e-9)
        period = p.get('period', 1e-6)
        
        if t < delay:
            return val0
        
        t_rel = (t - delay) % period
        
        if t_rel < rise:
            return val0 + (val1 - val0) * (t_rel / rise)
        elif t_rel < rise + width:
            return val1
        elif t_rel < rise + width + fall:
            return val1 - (val1 - val0) * ((t_rel - rise - width) / fall)
        else:
            return val0


@dataclass
class ISource(Component):
    """電流源"""
    dc_value: float = 0.0


@dataclass
class Resistor(Component):
    """抵抗"""
    resistance: float = 1e6


@dataclass
class Capacitor(Component):
    """キャパシタ"""
    capacitance: float = 1e-15


@dataclass
class MOSFET(Component):
    """MOSFET (BSIM4)"""
    w: float = 1e-6
    l: float = 180e-9
    nf: int = 1
    mos_type: str = "nmos"  # "nmos" or "pmos"
    model_params: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 回路クラス
# =============================================================================

class Circuit:
    """回路定義"""
    
    def __init__(self, name: str = "circuit"):
        self.name = name
        self.components: Dict[str, Component] = {}
        self.nodes: set = {"0"}
        self._node_to_idx: Dict[str, int] = {}
        self._idx_to_node: Dict[int, str] = {}
    
    def add_component(self, comp: Component):
        """素子を追加"""
        self.components[comp.name] = comp
        for node in comp.nodes:
            self.nodes.add(node)
    
    def build_node_index(self) -> int:
        """ノード番号マッピングを構築（0=GND）"""
        self._node_to_idx = {"0": 0}
        self._idx_to_node = {0: "0"}
        
        idx = 1
        for node in sorted(self.nodes - {"0"}):
            self._node_to_idx[node] = idx
            self._idx_to_node[idx] = node
            idx += 1
        
        return len(self._node_to_idx)
    
    def node_idx(self, node: str) -> int:
        return self._node_to_idx.get(node, 0)
    
    def idx_node(self, idx: int) -> str:
        return self._idx_to_node.get(idx, "0")


# =============================================================================
# BSIM4 DC解析器
# =============================================================================

class BSIM4DCAnalyzer:
    """BSIM4を使用したDC解析器
    
    特徴:
    - 内部ノード（dp/sp）オプションによりGdpr/Gspr（シリーズ抵抗）をサポート
    - D/Sスワップ検出による安定した収束
    - use_internal_nodes=Trueで高精度モード、Falseで安定性優先モード
    """
    
    def __init__(self, circuit: Circuit, gmin: float = 1e-10, model_dir: Optional[str] = None,
                 use_internal_nodes: bool = True, enable_bias_limiting: bool = False,
                 enable_ieq: bool = False, enable_nlscale: bool = False,
                 enable_kcl_residual: bool = False, enable_polish: bool = True,
                 iabstol: float = 1e-12,
                 vntol: float = 1e-9,
                 param_source: str = "calculator"):
        """
        Args:
            circuit: 解析対象の回路
            gmin: 最小コンダクタンス
            model_dir: モデルファイルディレクトリ
            use_internal_nodes: Trueで内部ノード（Gdpr/Gspr）を使用、Falseで外部ノードのみ
        """
        self.circuit = circuit
        self.gmin = gmin
        self.model_dir = model_dir
        self.use_internal_nodes = use_internal_nodes
        self.enable_bias_limiting = enable_bias_limiting
        self.enable_ieq = enable_ieq
        self.enable_nlscale = enable_nlscale
        self.enable_kcl_residual = enable_kcl_residual
        self.enable_polish = enable_polish
        self.iabstol = iabstol
        self.vntol = vntol
        self.param_source = param_source
        self.num_nodes = circuit.build_node_index()
        self.bsim4_devices: Dict[str, Any] = {}

        # MOSFET数（param_source=auto 用）
        self.mos_count = sum(1 for c in self.circuit.components.values() if self._is_mosfet(c))

        # MOSFETバイアスの前回値（limiting用）
        self.prev_bias: Dict[str, Dict[str, float]] = {}

        # バイアスクリップ用レール電圧（solve内で推定）
        self.vg_lo: Optional[float] = None
        self.vg_hi: Optional[float] = None
        self.vd_lo: Optional[float] = None
        self.vd_hi: Optional[float] = None
        self.vb_lo: Optional[float] = None
        self.vb_hi: Optional[float] = None
        
        # 内部ノード管理
        # { 'M0': {'dp': idx, 'sp': idx}, ... }
        self.internal_node_map: Dict[str, Dict[str, int]] = {}
        self.num_internal_nodes = 0
        
        # Newton数据收集（新增）
        self._newton_collector = None
        
        # BSIM4デバイス初期化
        self._init_bsim4_devices()
        
        # 内部ノードを使用する場合、インデックスを割り当て
        if self.use_internal_nodes:
            self._init_internal_nodes()

    def enable_newton_collection(self, enable: bool = True, verbose: bool = False):
        '''启用或禁用Newton迭代数据收集
        
        Args:
            enable: 是否启用收集
            verbose: 是否输出调试信息
        '''
        if enable and NEWTON_COLLECTION_AVAILABLE:
            self._newton_collector = NewtonDataCollector("DC", verbose=verbose)
            if verbose:
                print("✅ Newton数据收集已启用")
        else:
            self._newton_collector = None
            if verbose:
                print("❌ Newton数据收集已禁用")

    def _get_prev_bias(self, dev_name: str) -> Dict[str, float]:
        return self.prev_bias.get(dev_name, {
            "Vgs": 0.0,
            "Vds": 0.0,
            "Vbs": 0.0,
            "Vbd": 0.0,
            "Vbs_j": 0.0,
        })

    def _set_prev_bias(self, dev_name: str, vgs: float, vds: float, vbs: float, vbd: float, vbs_j: float) -> None:
        self.prev_bias[dev_name] = {
            "Vgs": vgs,
            "Vds": vds,
            "Vbs": vbs,
            "Vbd": vbd,
            "Vbs_j": vbs_j,
        }

    @staticmethod
    def _fetlim(vnew: float, vold: float, maxstep: float = 0.25) -> float:
        if vnew > vold + maxstep:
            return vold + maxstep
        if vnew < vold - maxstep:
            return vold - maxstep
        return vnew

    @staticmethod
    def _pnjlim(vnew: float, vold: float, vt: float = 0.02585, vcrit: float = 0.6145) -> float:
        # forward-bias limiting (SPICE-like)
        if vnew > vcrit and abs(vnew - vold) > 2.0 * vt:
            if vold > 0.0:
                arg = 1.0 + (vnew - vold) / vt
                if arg > 0.0:
                    return vold + vt * np.log(arg)
                return vcrit
            return vt * np.log(max(vnew / vt, 1e-12))
        return vnew

    def _clip3(self, vgs: float, vds: float, vbs: float) -> Tuple[float, float, float]:
        if self.vg_lo is None or self.vg_hi is None:
            return vgs, vds, vbs
        vgs_lim = np.clip(vgs, self.vg_lo, self.vg_hi)
        vds_lim = np.clip(vds, self.vd_lo, self.vd_hi)
        vbs_lim = np.clip(vbs, self.vb_lo, self.vb_hi)
        return vgs_lim, vds_lim, vbs_lim
    
    def _init_internal_nodes(self):
        """MOSFETの内部ノード（drain prime, source prime）インデックスを割り当て"""
        base_idx = self.num_nodes - 1  # 外部ノード数（GND除く）
        idx = 0
        for name, comp in self.circuit.components.items():
            if self._is_mosfet(comp) and name in self.bsim4_devices:
                device = self.bsim4_devices[name]
                # BSIM4からGdpr/Gspr取得（評価結果から）
                result = device.evaluate(0.0, 0.0, 0.0)
                gdpr = result.get('Gdpr', 0.0)
                gspr = result.get('Gspr', 0.0)
                
                # Julia同等: gdpr/gspr が0でも dp/sp を常時割り当て
                self.internal_node_map[name] = {
                    'dp': base_idx + idx,
                    'sp': base_idx + idx + 1,
                    'gdpr': gdpr,
                    'gspr': gspr,
                }
                idx += 2
        
        self.num_internal_nodes = idx
    
    def _init_bsim4_devices(self):
        """BSIM4デバイスを初期化（BSIM4Calculatorでパラメータ計算）"""
        if BSIM4Device is None:
            print("Warning: BSIM4Device not available, using simple model")
            return
        
        for name, comp in self.circuit.components.items():
            if self._is_mosfet(comp):
                try:
                    # モデル名を推測
                    model_name = comp.model_params.get('model_name', None)
                    if model_name is None:
                        if comp.mos_type.lower() in ('pmos', 'p'):
                            model_name = 'mp25od33_svt'
                        else:
                            model_name = 'mn25od33_svt'
                    
                    # インスタンスパラメータを取得（netlist指定のみ上書き）
                    inst_sa = float(comp.model_params['sa']) if 'sa' in comp.model_params else None
                    inst_sb = float(comp.model_params['sb']) if 'sb' in comp.model_params else None
                    inst_sd = float(comp.model_params['sd']) if 'sd' in comp.model_params else None
                    inst_ad = float(comp.model_params['ad']) if 'ad' in comp.model_params else None
                    inst_as = float(comp.model_params['as']) if 'as' in comp.model_params else None
                    inst_pd = float(comp.model_params['pd']) if 'pd' in comp.model_params else None
                    inst_ps = float(comp.model_params['ps']) if 'ps' in comp.model_params else None
                    inst_multi = float(comp.model_params.get('multi', 1.0))
                    inst_nrd = float(comp.model_params['nrd']) if 'nrd' in comp.model_params else None
                    inst_nrs = float(comp.model_params['nrs']) if 'nrs' in comp.model_params else None
                    
                    # param_source の選択
                    param_source = self.param_source
                    if param_source == "auto":
                        param_source = "calculator"

                    loaded_params = None
                    if param_source in ("computed", "merge"):
                        loaded_params = load_model_params_from_scs(model_name, self.model_dir)

                    calc_params = None
                    if param_source in ("calculator", "merge") and _HAS_CALCULATOR:
                        # JSONディレクトリを取得
                        json_dir = self.model_dir if self.model_dir else default_model_dir
                        
                        # トランジスタパラメータを準備（未指定はcalculator側デフォルトを使用）
                        transistor_params = {
                            'w': comp.w,
                            'l': comp.l,
                            'nf': float(comp.nf),
                            'multi': inst_multi,
                            'temp': 27.0,
                        }
                        if inst_sa is not None:
                            transistor_params['sa'] = inst_sa
                        if inst_sb is not None:
                            transistor_params['sb'] = inst_sb
                        if inst_sd is not None:
                            transistor_params['sd'] = inst_sd
                        if inst_ad is not None:
                            transistor_params['ad'] = inst_ad
                        if inst_as is not None:
                            transistor_params['as'] = inst_as
                        if inst_pd is not None:
                            transistor_params['pd'] = inst_pd
                        if inst_ps is not None:
                            transistor_params['ps'] = inst_ps
                        if inst_nrd is not None:
                            transistor_params['nrd'] = inst_nrd
                        if inst_nrs is not None:
                            transistor_params['nrs'] = inst_nrs
                        
                        # BSIM4Calculatorで計算
                        calculator = BSIM4Calculator(json_dir, model_name)
                        calc_params = calculator.calculate('tt', transistor_params)

                    # パラメータの組み立て
                    merged_params = {}
                    if param_source == "computed":
                        if loaded_params:
                            merged_params.update(loaded_params)
                    elif param_source == "calculator":
                        if calc_params is not None:
                            merged_params.update(calc_params)
                    elif param_source == "merge":
                        if calc_params is not None:
                            merged_params.update(calc_params)
                        if loaded_params:
                            merged_params.update(loaded_params)

                    merged_params.update(comp.model_params)

                    if 'type' not in merged_params:
                        merged_params['type'] = comp.mos_type

                    # Julia同等: beta0 が大きい場合は削除して u0 ベースに寄せる
                    b0 = merged_params.get('beta0')
                    if isinstance(b0, (int, float)) and b0 > 1.0:
                        if os.environ.get('BSIM4SIM_DEBUG', '0') == '1':
                            print(f"[BSIM4] Remove beta0={b0} for {name}")
                        merged_params.pop('beta0', None)
                    
                    # Julia/CSVと同等の実効寸法を使用（w_si/l_si, lshift）
                    inst_w = comp.w
                    inst_l = comp.l
                    w_si = merged_params.get('w_si')
                    l_si = merged_params.get('l_si')
                    if isinstance(w_si, (int, float)):
                        inst_w = float(w_si)
                    if isinstance(l_si, (int, float)):
                        lshift = merged_params.get('lshift', 8e-9)
                        try:
                            lshift = float(lshift)
                        except Exception:
                            lshift = 8e-9
                        inst_l = float(l_si) - lshift
                        if inst_l <= 0:
                            inst_l = float(l_si)

                    # device_params側も実効寸法に揃える（computed.scs相当）
                    merged_params['w'] = inst_w
                    merged_params['l'] = inst_l

                    # インスタンスパラメータのフォールバック（model計算結果を優先）
                    if inst_sa is None:
                        inst_sa = float(merged_params.get('sa', 0.0))
                    if inst_sb is None:
                        inst_sb = float(merged_params.get('sb', 0.0))
                    if inst_sd is None:
                        inst_sd = float(merged_params.get('sd', 0.0))
                    if inst_ad is None:
                        inst_ad = float(merged_params.get('ad', 0.0))
                    if inst_as is None:
                        inst_as = float(merged_params.get('as', 0.0))
                    if inst_pd is None:
                        inst_pd = float(merged_params.get('pd', 0.0))
                    if inst_ps is None:
                        inst_ps = float(merged_params.get('ps', 0.0))
                    if inst_nrd is None:
                        inst_nrd = float(merged_params.get('nrd', 0.0))
                    if inst_nrs is None:
                        inst_nrs = float(merged_params.get('nrs', 0.0))

                    # Juliaのextract_inst_paramsと同等にsa/sb/sdをシュリンク
                    lshrink = inst_l / comp.l if comp.l > 0 else 1.0
                    wshrink = inst_w / comp.w if comp.w > 0 else 1.0
                    inst_sa *= lshrink
                    inst_sb *= wshrink
                    inst_sd *= wshrink

                    device = BSIM4Device(
                        name=name,
                        device_params=merged_params,
                        w=inst_w,
                        l=inst_l,
                        nf=float(comp.nf),
                        multi=inst_multi,
                        sa=inst_sa,
                        sb=inst_sb,
                        sd=inst_sd,
                        ad=inst_ad,
                        as_=inst_as,
                        pd=inst_pd,
                        ps=inst_ps,
                        nrd=inst_nrd,
                        nrs=inst_nrs,
                    )
                    self.bsim4_devices[name] = device
                except Exception as e:
                    print(f"Warning: Failed to create BSIM4 device {name}: {e}")

    def _is_vsource(self, comp: Component) -> bool:
        """型チェック：VSource か？（ローカル＆simple_circuit_analyzer両対応）"""
        return isinstance(comp, VSource) or type(comp).__name__ == 'VSource'

    def _is_resistor(self, comp: Component) -> bool:
        """型チェック：Resistor か？（ローカル＆simple_circuit_analyzer両対応）"""
        return isinstance(comp, Resistor) or type(comp).__name__ == 'Resistor'
    
    def _is_isource(self, comp: Component) -> bool:
        """型チェック：ISource か？（ローカル＆simple_circuit_analyzer両対応）"""
        return isinstance(comp, ISource) or type(comp).__name__ == 'ISource'
    
    def _is_mosfet(self, comp: Component) -> bool:
        """型チェック：MOSFET か？（ローカル＆simple_circuit_analyzer両対応）"""
        return isinstance(comp, MOSFET) or type(comp).__name__ == 'MOSFET'

    def _count_vsources(self) -> int:
        count = 0
        for c in self.circuit.components.values():
            # ローカルクラス OR simple_circuit_analyzer のクラス
            if isinstance(c, VSource) or type(c).__name__ == 'VSource':
                count += 1
        return count
    
    def _estimate_rail_max(self) -> float:
        """Julia同等: レール最大電圧を推定（GND対の電圧源から）
        
        Returns:
            最大レール電圧、見つからなければ5.0V（Juliaのデフォルト）
        """
        vmax = 0.0
        for name, comp in self.circuit.components.items():
            if self._is_vsource(comp):
                # GNDに接続されているか確認
                n1, n2 = comp.nodes[0], comp.nodes[1]
                if n1 == "0" or n2 == "0":
                    v = comp.get_voltage(0.0)
                    if np.isfinite(v):
                        vmax = max(vmax, abs(v))
        
        return vmax if vmax > 0 else 5.0
    
    def solve(self, max_iter: int = 200, tol: float = 1e-9, 
              verbose: bool = False, polish_iters: int = 80,
              source_factors: Optional[List[float]] = None,
              nlscale_factors: Optional[List[float]] = None,
              min_iter_per_step: int = 20,
              force_full_iters: bool = False,
              initial_guess: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """DC解析を実行。Julia同等の改善を適用：
        1. Vgs/Vds/Vbsのクリップ（レール電圧ベース）
        2. 内部ノードを常時有効化
        3. ソースステップ後のpolish段（80反復）
        
        Args:
            max_iter: 最大反復回数
            tol: 収束許容値
            verbose: デバッグ出力
            polish_iters: ポリッシュ段の反復回数（Juliaと同じ80が標準）
            source_factors: ソースステッピング係数リスト（Noneでデフォルト使用）
            nlscale_factors: NLスケール係数リスト（Noneでデフォルト使用）
            min_iter_per_step: 各ステップの最小反復回数
            force_full_iters: 収束しても各ステップの反復を最後まで回す
            initial_guess: GNN预测的初始电压字典 (节点名 -> 电压值)
        
        Returns:
            (node_voltages, mos_currents)
        """
        print("\n" + "="*80)
        print("🔧 DC SOLVER - Starting Analysis")
        print("="*80)
        
        # Julia同等のクリップ値推定（外部ノード用のみ）
        # TODO: バイアスクリップはデフォルトではスキップし、必要に応じて有効化
        Vrail_max = self._estimate_rail_max()
        self.vg_lo = -(Vrail_max + 0.5) if Vrail_max > 0 else -10.0
        self.vg_hi = Vrail_max + 0.5 if Vrail_max > 0 else 10.0
        self.vd_lo = self.vg_lo
        self.vd_hi = self.vg_hi
        self.vb_lo = self.vg_lo
        self.vb_hi = self.vg_hi
        
        print(f"\n📊 Circuit Configuration:")
        print(f"  Rail voltage (estimated): {Vrail_max:.3f} V")
        print(f"  Voltage clipping range: [{self.vg_lo:.3f}, {self.vg_hi:.3f}] V")
        
        n = self.num_nodes - 1  # GND除く
        num_vsrc = self._count_vsources()
        # 内部ノード（条件付き有効化を保持）
        n_internal = self.num_internal_nodes if self.use_internal_nodes else 0
        total_size = n + n_internal + num_vsrc
        
        print(f"\n📐 Matrix Dimensions:")
        print(f"  External nodes:  {n}")
        print(f"  Internal nodes:  {n_internal} {'(dp/sp enabled)' if self.use_internal_nodes else '(disabled)'}")
        print(f"  Voltage sources: {num_vsrc}")
        print(f"  Total variables: {total_size}")
        print(f"\n⚙️  Solver Options:")
        print(f"  GMIN:              {self.gmin:.3e}")
        print(f"  Bias limiting:     {self.enable_bias_limiting}")
        print(f"  IEQ mode:          {self.enable_ieq}")
        print(f"  NL scaling:        {self.enable_nlscale}")
        print(f"  KCL residual:      {self.enable_kcl_residual}")
        print(f"  Polish:            {self.enable_polish} ({polish_iters} iters)")
        print(f"  Convergence tol:   {tol:.3e}")
        
        if total_size == 0:
            return {}, {}
        
        # 電圧源インデックス（内部ノード後）
        vsrc_base_idx = n + n_internal
        vsrc_idx = {}
        idx = vsrc_base_idx
        for name, comp in self.circuit.components.items():
            if self._is_vsource(comp):
                vsrc_idx[name] = idx
                idx += 1
        
        # 初始值
        x = np.zeros(total_size)
        
        print(f"\n🎯 Initial Guess Setup:")
        
        # 🔴 新增：如果提供了GNN初始猜测，优先使用
        if initial_guess is not None:
            print(f"  🤖 Using GNN-predicted initial voltages for {len(initial_guess)} nodes")
            for node_name, voltage in initial_guess.items():
                node_idx = self.circuit.node_idx(node_name)
                if node_idx > 0:  # 排除GND（node_idx=0）
                    x[node_idx - 1] = voltage
                    print(f"    {node_name}: {voltage:.4f} V (GNN)")
        
        # 电圧源ノードの初期化
        for name, comp in self.circuit.components.items():
            if self._is_vsource(comp):
                n_pos = self.circuit.node_idx(comp.nodes[0]) - 1
                if n_pos >= 0:
                    # パルス電源の場合はt=0での電圧（val0）を使う
                    x[n_pos] = comp.get_voltage(0.0)
                    print(f"  {name}: node {comp.nodes[0]} = {x[n_pos]:.4f} V (voltage source)")
        
        # 浮遊ノードの初期推定（VDD/2または中間電圧）
        # 🔴 修改：只在没有GNN预测时才使用默认初始化
        if initial_guess is None:
            # 電圧源で設定されていないノードを検出
            vsrc_nodes = set()
            vdd_val = 0.0
            for name, comp in self.circuit.components.items():
                if self._is_vsource(comp):
                    n_pos = self.circuit.node_idx(comp.nodes[0]) - 1
                    if n_pos >= 0:
                        vsrc_nodes.add(n_pos)
                        v = comp.get_voltage(0.0)
                        if v > vdd_val:
                            vdd_val = v
            
            # 浮遊ノードをVDD/2で初期化（ソースフォロワ等に適した初期値）
            # ただしMOSFETのソース端子は低めの初期値を設定
            mos_source_nodes = set()
            mos_drain_nodes = set()
            for name, comp in self.circuit.components.items():
                if self._is_mosfet(comp):
                    s_node = comp.nodes[2]  # Source node
                    d_node = comp.nodes[0]  # Drain node
                    s_idx = self.circuit.node_idx(s_node) - 1
                    d_idx = self.circuit.node_idx(d_node) - 1
                    if s_idx >= 0 and comp.mos_type.lower() in ('nmos', 'n'):
                        mos_source_nodes.add(s_idx)
                    if d_idx >= 0 and comp.mos_type.lower() in ('pmos', 'p'):
                        mos_drain_nodes.add(d_idx)  # PMOS出力
            
            # 初期値を3段階で設定（convergence改善）
            # VSS: 0V, VDD/3: 浮遊節点の低め, 2*VDD/3: 浮遊節点の高め
            init_v_low = vdd_val / 3 if vdd_val > 0 else 0.5
            init_v_high = 2 * vdd_val / 3 if vdd_val > 0 else 1.5
            
            for i in range(n):
                if i not in vsrc_nodes and abs(x[i]) < 1e-12:
                    node_name = self.circuit.idx_node(i+1)
                    if i in mos_source_nodes:
                        # NMOSソース（定電流源の場合が多い）：低めだが0ではない
                        x[i] = init_v_low
                        print(f"  {node_name}: {x[i]:.4f} V (NMOS source)")
                    elif i in mos_drain_nodes:
                        # PMOS出力（通常は高めの電圧）
                        x[i] = init_v_high
                        print(f"  {node_name}: {x[i]:.4f} V (PMOS drain)")
                    else:
                        # その他のノード（中間値）
                        x[i] = (init_v_low + init_v_high) / 2
                        print(f"  {node_name}: {x[i]:.4f} V (floating)")
        else:
            print(f"  ✅ Skipping default initialization (using GNN predictions)")
        
        # 内部ノードの初期化（条件付き有効化）
        if self.use_internal_nodes:
            for name, info in self.internal_node_map.items():
                comp = self.circuit.components[name]
                dp_idx = info.get('dp', -1)
                sp_idx = info.get('sp', -1)
                if dp_idx >= 0:
                    # drain primeは外部drainと同じ初期値
                    nd = self.circuit.node_idx(comp.nodes[0]) - 1
                    x[dp_idx] = x[nd] if nd >= 0 else 0.0
                if sp_idx >= 0:
                    # source primeは外部sourceと同じ初期値
                    ns = self.circuit.node_idx(comp.nodes[2]) - 1
                    x[sp_idx] = x[ns] if ns >= 0 else 0.0
        
        # ソースステッピング + MOS寄与ホモトピー
        source_factors = source_factors or [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]
        if nlscale_factors is None:
            nlscale_factors = [0.3, 0.6, 1.0] if self.enable_nlscale else [1.0]
        
        print(f"\n🔄 Source Stepping & Homotopy:")
        print(f"  Source factors: {source_factors}")
        print(f"  NL scale factors: {nlscale_factors}")
        print("\n" + "="*80)

        # ====== 保存初始x值（迭代#0之前的状态） ======
        if self._newton_collector is not None:
            try:
                # 创建一个特殊的迭代-1记录来保存初始猜测值
                init_data = NewtonIterationData(
                    iteration=-1,  # 使用-1表示这是初始状态
                    time=None,
                    source_factor=0.0,
                    nlscale=1.0
                )
                init_data.x = x.copy()
                
                # 设置节点名
                init_data.node_names = []
                for i in range(n):
                    node_idx = i + 1
                    node_name = self.circuit.idx_node(node_idx)
                    init_data.node_names.append(node_name)
                for i in range(n_internal):
                    init_data.node_names.append(f"internal_{i}")
                for name, comp in self.circuit.components.items():
                    if self._is_vsource(comp):
                        init_data.node_names.append(f"vsrc_{name}")
                
                self._newton_collector.add_iteration(init_data)
            except Exception as e:
                if verbose:
                    print(f"⚠️  初始状态保存失败: {e}")
        # ====== 初始状态保存结束 ======

        for nlscale in nlscale_factors:
            if self.enable_nlscale:
                print(f"\n🎚️  NL Scale = {nlscale:.1f}")
            for source_factor in source_factors:
                print(f"\n{'─'*80}")
                print(f"⚡ Source Factor = {source_factor:.2f} (NL scale = {nlscale:.1f})")
                print(f"{'─'*80}")
                # 各ステップで十分な反復を行う
                iter_per_step = max(min_iter_per_step, max_iter // (len(source_factors) * len(nlscale_factors)))
                for iteration in range(iter_per_step):
                    if iteration == 0:
                        print(f"\n  🔁 Newton-Raphson Iterations (max {iter_per_step}):")
                    
                    J = np.zeros((total_size, total_size))
                    f = np.zeros(total_size)
                    
                    # GMINスタンプ（外部ノード + 内部ノード）
                    # ソースステップの初期段階ではGMINを大きくして数値安定化を強化
                    # ただし過度な制御は収束を阻害するため、指数関数的に減衰させる
                    if source_factor < 0.2:
                        # 初期段階（sf < 0.2）：GMIN を段階的に増大（より強い）
                        gmin_eff = self.gmin * (100.0 * (0.2 - source_factor) / 0.2)
                    else:
                        # 本段階（sf >= 0.1）：通常のGMIN
                        gmin_eff = self.gmin
                    
                    if iteration == 0:
                        print(f"    GMIN (effective): {gmin_eff:.3e}")
                    
                    for i in range(n + n_internal):
                        J[i, i] += gmin_eff
                        f[i] += gmin_eff * (0.0 - x[i])
                    
                    # 各素子のスタンプ
                    if verbose and iteration == 0 and source_factor == source_factors[0]:
                        self._debug_stamp = True
                        print(f"[DC] Stamping components:")
                    else:
                        self._debug_stamp = False
                        
                    for name, comp in self.circuit.components.items():
                        if self._is_resistor(comp):
                            self._stamp_resistor(J, f, comp, x)
                        elif self._is_vsource(comp):
                            if verbose and iteration == 0 and source_factor == source_factors[0]:
                                print(f"[DC] Stamping vsource '{name}': vsrc_idx={vsrc_idx.get(name, 'N/A')}")
                            self._stamp_vsource(J, f, comp, x, vsrc_idx[name], source_factor)
                        elif self._is_isource(comp):
                            self._stamp_isource(J, f, comp, source_factor)
                        elif self._is_mosfet(comp):
                            if iteration == 0 and verbose:
                                nd = self.circuit.node_idx(comp.nodes[0]) - 1
                                ng = self.circuit.node_idx(comp.nodes[1]) - 1
                                ns = self.circuit.node_idx(comp.nodes[2]) - 1
                                vd = x[nd] if nd >= 0 else 0.0
                                vg = x[ng] if ng >= 0 else 0.0
                                vs = x[ns] if ns >= 0 else 0.0
                                print(f"    MOS {name}: Vd={vd:.4f}, Vg={vg:.4f}, Vs={vs:.4f}, Vgs={vg-vs:.4f}, Vds={vd-vs:.4f}")
                            
                            if self.use_internal_nodes and comp.name in self.internal_node_map:
                                self._stamp_mosfet_bsim4_internal(J, f, comp, x, nlscale=nlscale)
                            else:
                                self._stamp_mosfet_bsim4(J, f, comp, x, nlscale=nlscale)
                    
                    # ニュートン更新
                    if verbose and iteration == 0 and source_factor == source_factors[0]:
                        print(f"[DC] Jacobian J at sf={source_factor:.2f}, iteration {iteration}:")
                        print(f"  J =\n{J}")
                        print(f"  f = {f}")
                        print(f"  J condition number: {np.linalg.cond(J):.3e}")
                    
                    try:
                        dx = np.linalg.solve(J, -f)
                    except np.linalg.LinAlgError as e:
                        if verbose:
                            print(f"[DC] Singular matrix at sf={source_factor:.2f}, iteration {iteration}: {e}")
                            print(f"[DC]   J condition number before rescue: {np.linalg.cond(J):.3e}")
                        # 対角に強い正則化項を追加
                        for i in range(total_size):
                            J[i, i] += 1e-6  # より強い正則化
                        try:
                            J_cond = np.linalg.cond(J)
                            if verbose:
                                print(f"[DC]   J condition number after rescue: {J_cond:.3e}")
                            if J_cond > 1e12:
                                if verbose:
                                    print(f"[DC]   J is still too ill-conditioned, skipping this step")
                                dx = np.zeros(total_size)  # ステップをスキップ
                            else:
                                dx = np.linalg.solve(J, -f)
                        except Exception as e2:
                            if verbose:
                                print(f"[DC]   Rescue failed: {e2}")
                            dx = np.zeros(total_size)  # ステップをスキップ
                    
                    # ====== Newton データ収集（新規） ======
                    # 注意：x は更新前の値を一時保存する
                    x_before_update = x.copy() if self._newton_collector is not None else None
                    
                    # ダンピング
                    # NOTE: 初期値が悪い場合、大きなdxが生じてダンピングが過度に制限される
                    # 対策: max_dx_v > 1.0の場合は段階的に適用
                    max_dx_v = np.max(np.abs(dx[:n])) if n > 0 else 0
                    
                    if max_dx_v > 1.0:
                        # 大きな修正の場合：段階的アプローチ
                        # ステップを小さくしながら複数回反復
                        step_damping = 0.2  # 1ステップごとに20%ずつ進む
                        x += step_damping * dx
                        damping_used = step_damping
                    else:
                        # 小さな修正の場合：通常のダンピング
                        damping = min(1.0, 0.5 / max(max_dx_v, 0.1)) if max_dx_v > 0.5 else 1.0
                        x += damping * dx
                        damping_used = damping
                    
                    # クランプ（外部ノードのみ、レール近傍に制限）
                    max_v = (Vrail_max + 0.5) if Vrail_max > 0 else 10.0
                    x[:n] = np.clip(x[:n], -max_v, max_v)
                    
                    # ====== Newton データ収集（更新後に実行） ======
                    if self._newton_collector is not None:
                        try:
                            # 反復データを作成
                            iter_data = NewtonIterationData(
                                iteration=iteration,
                                time=None,  # DC分析には時間情報なし
                                source_factor=source_factor,
                                nlscale=nlscale
                            )
                            
                            # 更新後のノード電圧を保存（これが次の迭代の初期値になる）
                            iter_data.x = x.copy()
                            
                            # Jacobian行列を保存
                            iter_data.jacobian = J.copy()
                            
                            # 残差ベクトルを保存
                            iter_data.residual = f.copy()
                            
                            # 更新ベクトルを保存
                            iter_data.delta_x = dx.copy()
                            
                            # ノード名を設定
                            iter_data.node_names = []
                            for i in range(n):
                                node_idx = i + 1  # GND=0なので+1
                                node_name = self.circuit.idx_node(node_idx)
                                iter_data.node_names.append(node_name)
                            
                            # 内部ノード名を追加
                            for i in range(n_internal):
                                iter_data.node_names.append(f"internal_{i}")
                            
                            # 電圧源ノード名を追加
                            for name, comp in self.circuit.components.items():
                                if self._is_vsource(comp):
                                    iter_data.node_names.append(f"vsrc_{name}")
                            
                            # 収束指標を計算
                            f_abs = np.abs(iter_data.residual)
                            iter_data.max_residual = float(np.max(f_abs)) if len(f_abs) > 0 else 0.0
                            iter_data.l2_residual = float(np.linalg.norm(iter_data.residual))
                            
                            dx_abs = np.abs(iter_data.delta_x)
                            iter_data.max_delta = float(np.max(dx_abs)) if len(dx_abs) > 0 else 0.0
                            iter_data.l2_delta = float(np.linalg.norm(iter_data.delta_x))
                            
                            # Jacobian条件数を計算
                            try:
                                iter_data.jacobian_condition_number = float(np.linalg.cond(J))
                            except:
                                pass  # 条件数計算に失敗した場合はスキップ
                            
                            # 収集器に追加
                            self._newton_collector.add_iteration(iter_data)
                            
                        except Exception as e:
                            if verbose:
                                print(f"⚠️  Newton データ収集エラー: {e}")
                    # ====== Newton データ収集終了 ======
                    
                    # 収束判定（Jacobian求解の実行を確認）
                    if self.enable_kcl_residual:
                        node_residual = np.max(np.abs(f[:n])) if n > 0 else 0.0
                        branch_start = n + n_internal
                        branch_residual = np.max(np.abs(f[branch_start:])) if branch_start < total_size else 0.0
                        residual_norm = max(node_residual, branch_residual)
                    else:
                        node_residual = np.max(np.abs(f[:n])) if n > 0 else 0.0
                        branch_residual = 0.0
                        residual_norm = node_residual

                    update_norm = damping_used * max_dx_v
                    
                    # 每次迭代都打印（简化版）
                    if iteration % 5 == 0 or iteration < 3:  # 前3次和每5次打印一次
                        print(f"    Iter {iteration:3d}: |res|={residual_norm:.3e}, |dx|={max_dx_v:.3e}, damp={damping_used:.3f}, |x|_max={np.max(np.abs(x[:n])):.3e}")
                    
                    if self.enable_kcl_residual:
                        converged = (update_norm < tol and
                                     node_residual < self.iabstol and
                                     branch_residual < self.iabstol)
                    else:
                        residual_tol = max(1e-9, tol)
                        converged = (update_norm < tol and residual_norm < residual_tol)

                    if converged and not force_full_iters:
                        print(f"    ✓ Converged at iteration {iteration}")
                        print(f"      Final residual: {residual_norm:.3e}, update: {update_norm:.3e}")
                        break
                
                # 如果未收敛，显示最终状态
                if not converged:
                    print(f"    ⚠ Did not converge after {iter_per_step} iterations")
                    print(f"      Final residual: {residual_norm:.3e}, update: {update_norm:.3e}")
        
        # === ポリッシュ段（任意） ===
        if self.enable_polish and polish_iters > 0:
            print(f"\n{'='*80}")
            print(f"✨ Polish Phase (refining solution with GMIN={min(self.gmin, 1e-12):.3e})")
            print(f"{'='*80}")
            gmin_polish = min(self.gmin, 1e-12)
            for iteration in range(polish_iters):
                if iteration == 0 or iteration % 20 == 0:
                    print(f"  Polish iteration {iteration}/{polish_iters}...")
                J = np.zeros((total_size, total_size))
                f = np.zeros(total_size)

                for i in range(n + n_internal):
                    J[i, i] += gmin_polish
                    f[i] += gmin_polish * (0.0 - x[i])

                for name, comp in self.circuit.components.items():
                    if self._is_resistor(comp):
                        self._stamp_resistor(J, f, comp, x)
                    elif self._is_vsource(comp):
                        self._stamp_vsource(J, f, comp, x, vsrc_idx[name], 1.0)
                    elif self._is_isource(comp):
                        self._stamp_isource(J, f, comp, 1.0)
                    elif self._is_mosfet(comp):
                        if self.use_internal_nodes and comp.name in self.internal_node_map:
                            self._stamp_mosfet_bsim4_internal(J, f, comp, x, nlscale=1.0)
                        else:
                            self._stamp_mosfet_bsim4(J, f, comp, x, nlscale=1.0)

                try:
                    dx = np.linalg.solve(J, -f)
                except np.linalg.LinAlgError:
                    break

                max_dx_v = np.max(np.abs(dx[:n])) if n > 0 else 0
                if max_dx_v > 1.0:
                    step_damping = 0.2
                    x += step_damping * dx
                    damping_used = step_damping
                else:
                    damping = min(1.0, 0.5 / max(max_dx_v, 0.1)) if max_dx_v > 0.5 else 1.0
                    x += damping * dx
                    damping_used = damping

                max_v = (Vrail_max + 0.5) if Vrail_max > 0 else 10.0
                x[:n] = np.clip(x[:n], -max_v, max_v)

                if self.enable_kcl_residual:
                    node_residual = np.max(np.abs(f[:n])) if n > 0 else 0.0
                    branch_start = n + n_internal
                    branch_residual = np.max(np.abs(f[branch_start:])) if branch_start < total_size else 0.0
                    residual_norm = max(node_residual, branch_residual)
                else:
                    node_residual = np.max(np.abs(f[:n])) if n > 0 else 0.0
                    branch_residual = 0.0
                    residual_norm = node_residual

                update_norm = damping_used * max_dx_v
                if self.enable_kcl_residual:
                    converged = (update_norm < tol and
                                 node_residual < self.iabstol and
                                 branch_residual < self.iabstol)
                else:
                    residual_tol = max(1e-9, tol)
                    converged = (update_norm < tol and residual_norm < residual_tol)

                if converged:
                    break
        
        # デバッグ用に解ベクトルを保持
        self.last_solution = x.copy()

        # 結果をノード電圧辞書に変換
        node_voltages = {"0": 0.0}
        for i in range(1, self.num_nodes):
            node = self.circuit.idx_node(i)
            node_voltages[node] = x[i-1]
        
        # MOS電流を計算
        mos_currents = self._calculate_mos_currents(x)
        
        print(f"\n{'='*80}")
        print(f"✅ DC Analysis Complete")
        print(f"{'='*80}")
        print(f"\n📊 Final Node Voltages:")
        for node in sorted(node_voltages.keys()):
            if node != "0":
                print(f"  {node:15s}: {node_voltages[node]:10.6f} V")
        
        print(f"\n⚡ MOSFET Currents:")
        for name in sorted(mos_currents.keys()):
            print(f"  {name:15s}: {mos_currents[name]*1e6:10.3f} uA")
        print("\n" + "="*80 + "\n")
        
        # ====== Newton データ収集のクリーンアップ（新規） ======
        if self._newton_collector is not None:
            converged = (iteration < max_iter - 1)  # 最大反復に達していない = 収束
            self._newton_collector.set_convergence(
                converged,
                f"DC analysis completed. Iterations: {iteration+1}/{max_iter}"
            )
        # ====== Newton データ収集のクリーンアップ終了 ======
        
        return node_voltages, mos_currents
    
    def _stamp_resistor(self, J: np.ndarray, f: np.ndarray, comp: Resistor, x: np.ndarray):
        """抵抗のスタンプ"""
        n1 = self.circuit.node_idx(comp.nodes[0]) - 1
        n2 = self.circuit.node_idx(comp.nodes[1]) - 1
        g = 1.0 / comp.resistance
        
        if hasattr(self, '_debug_stamp') and self._debug_stamp:
            print(f"[DEBUG] Stamping Resistor: nodes={comp.nodes}, n1={n1}, n2={n2}, g={g:.3e}")
        
        v1 = x[n1] if n1 >= 0 else 0.0
        v2 = x[n2] if n2 >= 0 else 0.0
        current = g * (v1 - v2)
        
        if n1 >= 0:
            J[n1, n1] -= g
            if n2 >= 0:
                J[n1, n2] += g
            f[n1] -= current
        
        if n2 >= 0:
            J[n2, n2] -= g
            if n1 >= 0:
                J[n2, n1] += g
            f[n2] += current
    
    def _stamp_vsource(self, J: np.ndarray, f: np.ndarray, comp: VSource, 
                       x: np.ndarray, vsrc_row: int, source_factor: float = 1.0):
        """電圧源のスタンプ"""
        n_pos = self.circuit.node_idx(comp.nodes[0]) - 1
        n_neg = self.circuit.node_idx(comp.nodes[1]) - 1
        
        # DC解析ではt=0の電圧を使用（パルスの場合はval0）
        # source_factor を適用してバイアスを段階的に導入
        vdc = comp.get_voltage(0.0) * source_factor
        
        v_pos = x[n_pos] if n_pos >= 0 else 0.0
        v_neg = x[n_neg] if n_neg >= 0 else 0.0
        i_src = x[vsrc_row]
        
        f[vsrc_row] = v_pos - v_neg - vdc
        if n_pos >= 0:
            J[vsrc_row, n_pos] = 1.0
        if n_neg >= 0:
            J[vsrc_row, n_neg] = -1.0
        
        if n_pos >= 0:
            f[n_pos] -= i_src
            J[n_pos, vsrc_row] -= 1.0
        if n_neg >= 0:
            f[n_neg] += i_src
            J[n_neg, vsrc_row] += 1.0
    
    def _stamp_isource(self, J: np.ndarray, f: np.ndarray, comp: ISource, source_factor: float):
        """電流源のスタンプ"""
        n1 = self.circuit.node_idx(comp.nodes[0]) - 1
        n2 = self.circuit.node_idx(comp.nodes[1]) - 1
        current = comp.dc_value * source_factor
        
        if n1 >= 0:
            f[n1] -= current
        if n2 >= 0:
            f[n2] += current
    
    def _stamp_mosfet_bsim4(self, J: np.ndarray, f: np.ndarray, comp: MOSFET, x: np.ndarray,
                            nlscale: float = 1.0):
        """BSIM4 MOSFETのスタンプ（外部ノードのみ、内部ノードなし）
        
        Julia側のVARSTEP問題を回避するため、Gdpr/Gspr（シリーズ抵抗）は使用しない。
        MINIMALアナライザと同様、外部端子に直接gmをスタンプする。
        """
        nd = self.circuit.node_idx(comp.nodes[0]) - 1  # Drain
        ng = self.circuit.node_idx(comp.nodes[1]) - 1  # Gate
        ns = self.circuit.node_idx(comp.nodes[2]) - 1  # Source
        nb = self.circuit.node_idx(comp.nodes[3]) - 1  # Bulk
        
        vd = x[nd] if nd >= 0 else 0.0
        vg = x[ng] if ng >= 0 else 0.0
        vs = x[ns] if ns >= 0 else 0.0
        vb = x[nb] if nb >= 0 else 0.0
        
        # D/Sスワップ検出
        is_pmos = comp.mos_type.lower() in ('pmos', 'p')
        vds_phys = vd - vs
        swap = (vds_phys > 0.05) if is_pmos else (vds_phys < -0.05)
        
        # 評価用端子電圧
        if swap:
            vd_eval, vs_eval = vs, vd
        else:
            vd_eval, vs_eval = vd, vs
        
        vgs_raw = vg - vs_eval
        vds_raw = vd_eval - vs_eval
        vbs_raw = vb - vs_eval
        vbd_raw = vb - vd_eval
        vbsj_raw = vb - vs_eval

        if self.enable_bias_limiting:
            prev = self._get_prev_bias(comp.name)
            vgs_lim = self._fetlim(vgs_raw, prev["Vgs"])
            vds_lim = self._fetlim(vds_raw, prev["Vds"])
            vbs_lim = self._fetlim(vbs_raw, prev["Vbs"])
            vbd_lim = self._pnjlim(vbd_raw, prev["Vbd"])
            vbsj_lim = self._pnjlim(vbsj_raw, prev["Vbs_j"])
            vbs_lim = vbsj_lim
            vgs, vds, vbs = self._clip3(vgs_lim, vds_lim, vbs_lim)
        else:
            vgs, vds, vbs = vgs_raw, vds_raw, vbs_raw
            vbd_lim = vbd_raw
            vbsj_lim = vbsj_raw
        
        # NOTE: バイアスクリップはデフォルトではスキップ（複雑な回路では不安定）
        # Julia側のクリップはvg_lo/hi = ±3.5V（Vrail_max=3V）だが、
        # Python側では外部ノードレベルでのみ±10Vクリップとし、
        # BSIM4評価時は無制限を許す（BSIM4内部で飽和処理）
        # vgs = np.clip(vgs, self.vg_lo, self.vg_hi)
        # vds = np.clip(vds, self.vd_lo, self.vd_hi)
        # vbs = np.clip(vbs, self.vb_lo, self.vb_hi)
        
        # BSIM4評価
        if comp.name in self.bsim4_devices:
            device = self.bsim4_devices[comp.name]
            
            # DEBUG: PMOS電圧確認（評価前）
            if is_pmos and os.environ.get('DEBUG_PMOS', '0') == '1':
                print(f"\nDEBUG {comp.name} BEFORE EVAL:")
                print(f"  Physical: vd={vd:.4f} vg={vg:.4f} vs={vs:.4f} vb={vb:.4f}")
                print(f"  swap={swap}, vd_eval={vd_eval:.4f}, vs_eval={vs_eval:.4f}")
                print(f"  BSIM4 Input (clipped): Vgs={vgs:.4f}, Vds={vds:.4f}, Vbs={vbs:.4f}")
            
            result = device.evaluate(vgs, vds, vbs)
            ids = result['ids'] * nlscale
            gm = max(result['gm'] * nlscale, 1e-12)
            gds = max(result['gds'] * nlscale, 1e-12)
            gmb = max(result['gmb'] * nlscale, 0.0)
            
            # DEBUG: PMOS電流確認（評価後）
            if is_pmos and os.environ.get('DEBUG_PMOS', '0') == '1':
                print(f"  BSIM4 Output: ids={ids*1e6:.4f} uA, gm={gm:.6e}, gds={gds:.6e}")
        else:
            # フォールバック: 簡易モデル
            ids, gm, gds, gmb = self._simple_mos_model(vgs, vds, vbs, is_pmos, comp)
            ids *= nlscale
            gm = max(gm * nlscale, 1e-12)
            gds = max(gds * nlscale, 1e-12)
            gmb = max(gmb * nlscale, 0.0)
        
        # スワップ時は電流方向が逆
        # Idsは「評価用ドレイン→評価用ソース」の電流
        # 物理ノードから見ると符号反転
        if swap:
            ids = -ids

        # 前回バイアス更新（limiting基準）
        if self.enable_bias_limiting:
            self._set_prev_bias(comp.name, vgs, vds, vbs, vbd_lim, vbsj_lim)
        
        # Ieq（Newton整合）: Id0 - gm*Vgs - gds*Vds - gmb*Vbs
        ieq = ids - gm * vgs - gds * vds - gmb * vbs

        # 物理ノードへのJacobianスタンプ
        if swap:
            # スワップ時: 物理ドレインが評価ソース、物理ソースが評価ドレイン
            # dId_phys/dVg = -gm
            # dId_phys/dVd = -(-(gm+gds+gmb)) = gm+gds+gmb (物理ドレインは評価ソース)
            # dId_phys/dVs = -gds (物理ソースは評価ドレイン)
            # dId_phys/dVb = -gmb
            gm_eff = -gm
            gds_d_phys = (gm + gds + gmb)
            gds_s_phys = -gds
            gmb_eff = -gmb
        else:
            # 通常時
            gm_eff = gm
            gds_d_phys = gds
            gds_s_phys = -(gm + gds + gmb)
            gmb_eff = gmb
        
        # Drain KCL: f[d] -= Ieq (電流が流出)
        if nd >= 0:
            f[nd] -= ieq if self.enable_ieq else ids
            J[nd, nd] -= gds_d_phys
            if ng >= 0:
                J[nd, ng] -= gm_eff
            if ns >= 0:
                J[nd, ns] -= gds_s_phys
            if nb >= 0:
                J[nd, nb] -= gmb_eff
        
        # Source KCL: f[s] += Ieq (電流が流入)
        if ns >= 0:
            f[ns] += ieq if self.enable_ieq else ids
            J[ns, ns] += gds_s_phys
            if ng >= 0:
                J[ns, ng] += gm_eff
            if nd >= 0:
                J[ns, nd] += gds_d_phys
            if nb >= 0:
                J[ns, nb] += gmb_eff
    
    def _stamp_mosfet_bsim4_internal(self, J: np.ndarray, f: np.ndarray, comp: MOSFET, x: np.ndarray,
                                     nlscale: float = 1.0):
        """BSIM4 MOSFETのスタンプ（内部ノードdp/sp使用、Gdpr/Gsprをスタンプ）
        
        回路構造:
          D外部 ---[Gdpr]--- dp ---[MOSFET本体]--- sp ---[Gspr]--- S外部
          
        Jacobianスタンプ:
        1. D外部-dp間にGdpr抵抗をスタンプ
        2. S外部-sp間にGspr抵抗をスタンプ  
        3. MOSFET本体（gm, gds, gmb）はdp, sp, G, B間にスタンプ
        """
        nd = self.circuit.node_idx(comp.nodes[0]) - 1  # Drain（外部）
        ng = self.circuit.node_idx(comp.nodes[1]) - 1  # Gate
        ns = self.circuit.node_idx(comp.nodes[2]) - 1  # Source（外部）
        nb = self.circuit.node_idx(comp.nodes[3]) - 1  # Bulk
        
        # 内部ノードインデックス取得
        info = self.internal_node_map.get(comp.name, {})
        ndp = info.get('dp', -1)  # drain prime
        nsp = info.get('sp', -1)  # source prime
        gdpr = info.get('gdpr', 0.0)
        gspr = info.get('gspr', 0.0)
        
        # 内部ノードがない場合は外部ノードを使用
        if ndp < 0:
            ndp = nd
        if nsp < 0:
            nsp = ns
        
        # 電圧取得
        vd = x[nd] if nd >= 0 else 0.0
        vg = x[ng] if ng >= 0 else 0.0
        vs = x[ns] if ns >= 0 else 0.0
        vb = x[nb] if nb >= 0 else 0.0
        vdp = x[ndp] if ndp >= 0 and ndp != nd else vd
        vsp = x[nsp] if nsp >= 0 and nsp != ns else vs
        
        # D/Sスワップ検出（内部ノード間電圧で判定）
        is_pmos = comp.mos_type.lower() in ('pmos', 'p')
        vds_int = vdp - vsp
        swap = (vds_int > 0.05) if is_pmos else (vds_int < -0.05)
        
        # 評価用電圧（内部ノード基準）
        if swap:
            vdp_eval, vsp_eval = vsp, vdp
        else:
            vdp_eval, vsp_eval = vdp, vsp
        
        vgs_raw = vg - vsp_eval
        vds_raw = vdp_eval - vsp_eval
        vbs_raw = vb - vsp_eval
        vbd_raw = vb - vdp_eval
        vbsj_raw = vb - vsp_eval

        if self.enable_bias_limiting:
            prev = self._get_prev_bias(comp.name)
            vgs_lim = self._fetlim(vgs_raw, prev["Vgs"])
            vds_lim = self._fetlim(vds_raw, prev["Vds"])
            vbs_lim = self._fetlim(vbs_raw, prev["Vbs"])
            vbd_lim = self._pnjlim(vbd_raw, prev["Vbd"])
            vbsj_lim = self._pnjlim(vbsj_raw, prev["Vbs_j"])
            vbs_lim = vbsj_lim
            vgs_eval, vds_eval, vbs_eval = self._clip3(vgs_lim, vds_lim, vbs_lim)
        else:
            vgs_eval, vds_eval, vbs_eval = vgs_raw, vds_raw, vbs_raw
            vbd_lim = vbd_raw
            vbsj_lim = vbsj_raw
        
        # NOTE: バイアスクリップはデフォルトではスキップ（複雑な回路では不安定）
        # vgs_eval = np.clip(vgs_eval, self.vg_lo, self.vg_hi)
        # vds_eval = np.clip(vds_eval, self.vd_lo, self.vd_hi)
        # vbs_eval = np.clip(vbs_eval, self.vb_lo, self.vb_hi)
        
        # BSIM4評価
        if comp.name in self.bsim4_devices:
            device = self.bsim4_devices[comp.name]
            result = device.evaluate(vgs_eval, vds_eval, vbs_eval)
            ids = result['ids'] * nlscale
            gm = max(result['gm'] * nlscale, 1e-12)
            gds = max(result['gds'] * nlscale, 1e-12)
            gmb = max(result['gmb'] * nlscale, 0.0)
        else:
            ids, gm, gds, gmb = self._simple_mos_model(vgs_eval, vds_eval, vbs_eval, is_pmos, comp)
            ids *= nlscale
            gm = max(gm * nlscale, 1e-12)
            gds = max(gds * nlscale, 1e-12)
            gmb = max(gmb * nlscale, 0.0)
        
        # スワップ時は電流符号反転
        if swap:
            ids = -ids

        # 前回バイアス更新（limiting基準）
        if self.enable_bias_limiting:
            self._set_prev_bias(comp.name, vgs_eval, vds_eval, vbs_eval, vbd_lim, vbsj_lim)

        # === Gdpr/Gspr は評価結果から使用（Julia同等） ===
        gdpr_eval = result.get('Gdpr', gdpr)
        gspr_eval = result.get('Gspr', gspr)
        gdpr_eff = gdpr_eval if gdpr_eval > 1e-12 else 1e12
        gspr_eff = gspr_eval if gspr_eval > 1e-12 else 1e12
        
        # ===== 1. Gdpr抵抗スタンプ（D外部 - dp間） =====
        if gdpr_eff > 0 and nd >= 0 and ndp >= 0 and nd != ndp:
            i_gdpr = gdpr_eff * (vd - vdp)
            # D外部から電流流出
            f[nd] -= i_gdpr
            J[nd, nd] -= gdpr_eff
            J[nd, ndp] += gdpr_eff
            # dpへ電流流入
            f[ndp] += i_gdpr
            J[ndp, nd] += gdpr_eff
            J[ndp, ndp] -= gdpr_eff
        
        # ===== 2. Gspr抵抗スタンプ（S外部 - sp間） =====
        if gspr_eff > 0 and ns >= 0 and nsp >= 0 and ns != nsp:
            i_gspr = gspr_eff * (vs - vsp)
            # S外部から電流流出
            f[ns] -= i_gspr
            J[ns, ns] -= gspr_eff
            J[ns, nsp] += gspr_eff
            # spへ電流流入
            f[nsp] += i_gspr
            J[nsp, ns] += gspr_eff
            J[nsp, nsp] -= gspr_eff
        
        # Ieq（Newton整合）: Id0 - gm*Vgs - gds*Vds - gmb*Vbs
        ieq = ids - gm * vgs_eval - gds * vds_eval - gmb * vbs_eval

        # ===== 3. MOSFET本体スタンプ（dp, sp, G, B間） =====
        # Jacobianスタンプ（スワップ考慮）
        if swap:
            gm_eff = -gm
            gds_dp = (gm + gds + gmb)
            gds_sp = -gds
            gmb_eff = -gmb
        else:
            gm_eff = gm
            gds_dp = gds
            gds_sp = -(gm + gds + gmb)
            gmb_eff = gmb
        
        # dp KCL: f[dp] -= Ieq
        if ndp >= 0:
            f[ndp] -= ieq if self.enable_ieq else ids
            J[ndp, ndp] -= gds_dp
            if ng >= 0:
                J[ndp, ng] -= gm_eff
            if nsp >= 0:
                J[ndp, nsp] -= gds_sp
            if nb >= 0:
                J[ndp, nb] -= gmb_eff
        
        # sp KCL: f[sp] += Ieq
        if nsp >= 0:
            f[nsp] += ieq if self.enable_ieq else ids
            J[nsp, nsp] += gds_sp
            if ng >= 0:
                J[nsp, ng] += gm_eff
            if ndp >= 0:
                J[nsp, ndp] += gds_dp
            if nb >= 0:
                J[nsp, nb] += gmb_eff
    
    def _simple_mos_model(self, vgs: float, vds: float, vbs: float, 
                          is_pmos: bool, comp: MOSFET) -> Tuple[float, float, float, float]:
        """簡易MOSFETモデル（フォールバック用）"""
        vth0 = comp.model_params.get('vth0', 0.4)
        k = comp.model_params.get('k', 100e-6) * (comp.w / comp.l)
        lambda_ = comp.model_params.get('lambda', 0.01)
        gmin = 1e-12
        
        if is_pmos:
            vgs_int = -vgs
            vds_int = -vds
            vth_int = abs(vth0)
        else:
            vgs_int = vgs
            vds_int = vds
            vth_int = vth0
        
        vov = vgs_int - vth_int
        
        if vov <= 0 or vds_int < 0:
            return (0.0, gmin, gmin, 0.0)
        elif vds_int <= vov:
            ids_int = k * (vov * vds_int - 0.5 * vds_int**2) * (1 + lambda_ * vds_int)
            gm = k * vds_int * (1 + lambda_ * vds_int)
            gds = max(k * (vov - vds_int), gmin)
        else:
            ids_int = 0.5 * k * vov**2 * (1 + lambda_ * vds_int)
            gm = k * vov * (1 + lambda_ * vds_int)
            gds = max(0.5 * k * vov**2 * lambda_, gmin)
        
        ids = -ids_int if is_pmos else ids_int
        return (ids, gm, gds, 0.0)
    
    def _calculate_mos_currents(self, x: np.ndarray) -> Dict[str, float]:
        """MOS電流を計算
        
        戻り値は物理ドレインから流れ出す電流（正の場合はドレインからソースへ）
        """
        mos_currents = {}
        
        for name, comp in self.circuit.components.items():
            if self._is_mosfet(comp):
                nd = self.circuit.node_idx(comp.nodes[0]) - 1
                ng = self.circuit.node_idx(comp.nodes[1]) - 1
                ns = self.circuit.node_idx(comp.nodes[2]) - 1
                nb = self.circuit.node_idx(comp.nodes[3]) - 1
                
                vd = x[nd] if nd >= 0 else 0.0
                vg = x[ng] if ng >= 0 else 0.0
                vs = x[ns] if ns >= 0 else 0.0
                vb = x[nb] if nb >= 0 else 0.0
                
                is_pmos = comp.mos_type.lower() in ('pmos', 'p')
                
                swap_applied = False
                if self.use_internal_nodes and name in self.internal_node_map and name in self.bsim4_devices:
                    # 外部端子から内部ノード電位を反復更新（Julia同等）
                    device = self.bsim4_devices[name]
                    vdp = vd
                    vsp = vs
                    ids_phys = 0.0
                    for _ in range(6):
                        vds_int = vdp - vsp
                        swap = (vds_int > 0.05) if is_pmos else (vds_int < -0.05)
                        if swap:
                            vdp_eval, vsp_eval = vsp, vdp
                        else:
                            vdp_eval, vsp_eval = vdp, vsp
                        vgs = vg - vsp_eval
                        vds = vdp_eval - vsp_eval
                        vbs = vb - vsp_eval
                        vgs, vds, vbs = self._clip3(vgs, vds, vbs)
                        result = device.evaluate(vgs, vds, vbs)
                        ids = result['ids']
                        ids_phys = -ids if swap else ids
                        gdpr = result.get('Gdpr', 0.0)
                        gspr = result.get('Gspr', 0.0)
                        gdpr_eff = gdpr if gdpr > 1e-12 else 1e12
                        gspr_eff = gspr if gspr > 1e-12 else 1e12
                        vdp = vd - ids_phys / gdpr_eff
                        vsp = vs + ids_phys / gspr_eff
                    ids = ids_phys
                    swap_applied = True
                else:
                    if self.use_internal_nodes and name in self.internal_node_map:
                        info = self.internal_node_map.get(name, {})
                        ndp = info.get('dp', -1)
                        nsp = info.get('sp', -1)
                        vdp = x[ndp] if ndp >= 0 else vd
                        vsp = x[nsp] if nsp >= 0 else vs
                        vds_int = vdp - vsp
                        swap = (vds_int > 0.05) if is_pmos else (vds_int < -0.05)
                        if swap:
                            vdp_eval, vsp_eval = vsp, vdp
                        else:
                            vdp_eval, vsp_eval = vdp, vsp
                        vgs = vg - vsp_eval
                        vds = vdp_eval - vsp_eval
                        vbs = vb - vsp_eval
                    else:
                        vds_phys = vd - vs
                        swap = (vds_phys > 0.05) if is_pmos else (vds_phys < -0.05)
                        if swap:
                            vd_eval, vs_eval = vs, vd
                        else:
                            vd_eval, vs_eval = vd, vs
                        vgs = vg - vs_eval
                        vds = vd_eval - vs_eval
                        vbs = vb - vs_eval
                    if name in self.bsim4_devices:
                        result = self.bsim4_devices[name].evaluate(vgs, vds, vbs)
                        ids = result['ids']
                    else:
                        ids, _, _, _ = self._simple_mos_model(vgs, vds, vbs, is_pmos, comp)
                
                # スワップ時は物理ドレイン電流の符号を反転
                if swap and not swap_applied:
                    ids = -ids
                
                mos_currents[name] = ids
        
        return mos_currents


# =============================================================================
# BSIM4 TRAN解析器
# =============================================================================

class BSIM4TRANAnalyzer:
    """BSIM4を使用したTRAN解析器
    
    特徴:
    - 後退オイラー法（Backward Euler）による安定な積分
    - 内部ノード（dp/sp）サポートによりGdpr/Gspr（シリーズ抵抗）を考慮
    - D/Sスワップ検出
    - MOSFET寄生容量サポート（オプション）
    
    安定化処理（商用シミュレータ相当）:
    - 各ノードへの最小コンダクタンス（gmin）挿入
    - 高インピーダンスノードへの安定化容量自動追加
    - 適応タイムステップ制御（パルスエッジ検出）
    - 強化されたNewton法ダンピング
    - 電圧制限（Vt clipping）
    """
    
    def __init__(self, circuit: Circuit, dc_analyzer: BSIM4DCAnalyzer,
                 dc_solution: Dict[str, float], gmin: float = 1e-9,
                 use_mos_caps: bool = True, min_drain_cap: float = 0.0,
                 node_cap: float = 10e-15, vt_limit: float = 0.5,
                 adaptive_dt: bool = True, polish_steps: int = 3):
        """
        Args:
            circuit: 回路オブジェクト
            dc_analyzer: DC解析器（内部ノード設定を継承）
            dc_solution: DC解
            gmin: 最小コンダクタンス（各ノードをGNDに接続）
            use_mos_caps: MOSFET寄生容量を使用するか
            min_drain_cap: 各MOSFETドレインに追加する最小容量（収束改善用）
            node_cap: 各ノードに追加する安定化容量 [F]（デフォルト10fF）
            vt_limit: Newton更新時の電圧制限（Vt単位、0.5 = 13mV相当）
            adaptive_dt: 適応タイムステップを有効にするか
            polish_steps: 各タイムステップ後のpolish反復回数（安定化を減らして再収束）
        """
        self.circuit = circuit
        self.dc_analyzer = dc_analyzer
        self.dc_solution = dc_solution
        self.gmin = gmin
        self.gmin_base = gmin  # 元の値を保存
        self.num_nodes = circuit.build_node_index()
        self.use_mos_caps = use_mos_caps
        self.min_drain_cap = min_drain_cap
        
        # DCアナライザから内部ノード設定を継承
        self.use_internal_nodes = dc_analyzer.use_internal_nodes
        self.internal_node_map = dc_analyzer.internal_node_map
        self.num_internal_nodes = dc_analyzer.num_internal_nodes
        
        # 安定化パラメータ
        self.node_cap = node_cap          # 各ノードの安定化容量
        self.node_cap_base = node_cap     # 元の値を保存
        self.vt_limit = vt_limit          # Newton更新の電圧制限
        self.adaptive_dt = adaptive_dt    # 適応タイムステップ
        self.polish_steps = polish_steps  # Polish反復回数
        self.vt = 0.026  # 熱電圧 (26mV at 300K)
        
        # ノード電圧履歴（安定化容量用）
        self.node_v_prev: Dict[int, float] = {}
        
        # キャパシタ電圧の履歴
        self.cap_v_prev: Dict[str, float] = {}
        # MOSFET寄生容量の電荷履歴
        self.mos_q_prev: Dict[str, Dict[str, float]] = {}
        self._init_cap_history()
    
    def _init_cap_history(self):
        """キャパシタ電圧の初期化"""
        for name, comp in self.circuit.components.items():
            if isinstance(comp, Capacitor):
                n1, n2 = comp.nodes[0], comp.nodes[1]
                v1 = self.dc_solution.get(n1, 0.0)
                v2 = self.dc_solution.get(n2, 0.0)
                self.cap_v_prev[name] = v1 - v2
        
        # MOSFET寄生容量の電圧履歴を初期化（CGS/CGD/Cbd/Cbsモデル用）
        if self.use_mos_caps:
            for name, comp in self.circuit.components.items():
                if self._is_mosfet(comp) and name in self.dc_analyzer.bsim4_devices:
                    nd, ng, ns, nb = comp.nodes
                    vd = self.dc_solution.get(nd, 0.0)
                    vg = self.dc_solution.get(ng, 0.0)
                    vs = self.dc_solution.get(ns, 0.0)
                    vb = self.dc_solution.get(nb, 0.0)
                    
                    # CGS/CGD/Cbd/Cbsの電圧履歴を初期化
                    self.mos_q_prev[name] = {
                        'vgd': vg - vd,
                        'vgs': vg - vs,
                        'vdb': vd - vb,
                        'vsb': vs - vb,
                    }

    def _is_vsource(self, comp: Component) -> bool:
        """型チェック：VSource か？（ローカル＆simple_circuit_analyzer両対応）"""
        return isinstance(comp, VSource) or type(comp).__name__ == 'VSource'

    def _is_resistor(self, comp: Component) -> bool:
        """型チェック：Resistor か？（ローカル＆simple_circuit_analyzer両対応）"""
        return isinstance(comp, Resistor) or type(comp).__name__ == 'Resistor'
    
    def _is_isource(self, comp: Component) -> bool:
        """型チェック：ISource か？（ローカル＆simple_circuit_analyzer両対応）"""
        return isinstance(comp, ISource) or type(comp).__name__ == 'ISource'
    
    def _is_mosfet(self, comp: Component) -> bool:
        """型チェック：MOSFET か？（ローカル＆simple_circuit_analyzer両対応）"""
        return isinstance(comp, MOSFET) or type(comp).__name__ == 'MOSFET'
    
    def _count_vsources(self) -> int:
        return sum(1 for c in self.circuit.components.values() if self._is_vsource(c))
    
    def _detect_pulse_edge(self, t: float, dt: float) -> float:
        """パルスエッジを検出してタイムステップを調整
        
        Returns:
            調整されたタイムステップ（エッジ付近ではdtを小さくする）
        """
        for name, comp in self.circuit.components.items():
            if self._is_vsource(comp) and hasattr(comp, 'pulse_params') and comp.pulse_params:
                p = comp.pulse_params
                # パラメータ名を両方サポート（delay/td, width/pw, period/per）
                td = p.get('delay', p.get('td', 0.0))
                tr = p.get('rise', p.get('tr', 1e-12))
                tf = p.get('fall', p.get('tf', 1e-12))
                pw = p.get('width', p.get('pw', 1e-9))
                per = p.get('period', p.get('per', 2e-9))
                
                # エッジ時刻を周期ごとに計算（最大10周期分）
                edge_times_abs = []
                for k in range(10):
                    base = td + k * per if k > 0 else td
                    if base > t + dt * 100:
                        break
                    edge_times_abs.extend([
                        base,           # 立ち上がり開始
                        base + tr,      # 立ち上がり終了
                        base + tr + pw, # 立ち下がり開始
                        base + tr + pw + tf  # 立ち下がり終了
                    ])
                
                # 各エッジへの距離を計算
                min_edge = min(tr, tf)  # 最短エッジ時間
                for edge_t in edge_times_abs:
                    if edge_t < t:
                        continue
                    dist = edge_t - t
                    
                    if dist < min_edge:
                        # エッジ遷移中：dtの1/10まで縮小（min_edge/10より細かくしない）
                        return max(dt / 10, min_edge / 5)
                    elif dist < 5 * min_edge:
                        # エッジ直前：dtの1/5まで縮小
                        return dt / 5
                    elif dist < 20 * dt:
                        # エッジ近傍：少し縮小
                        return dt / 2
        return dt
    
    def _vt_clipping(self, dx: np.ndarray, vsrc_nodes: Dict[int, str], n: int) -> np.ndarray:
        """Newton更新量を熱電圧単位で制限（収束安定化）
        
        商用シミュレータのVt clipping相当の機能
        """
        max_delta = self.vt_limit * self.vt  # 例: 0.5 * 26mV = 13mV
        
        dx_clipped = dx.copy()
        for i in range(n):
            if i not in vsrc_nodes:
                if abs(dx_clipped[i]) > max_delta:
                    dx_clipped[i] = max_delta * np.sign(dx_clipped[i])
        return dx_clipped
    
    def _stamp_node_cap(self, J: np.ndarray, f: np.ndarray, x: np.ndarray, 
                        dt: float, vsrc_nodes: Dict[int, str], n: int):
        """各ノードに安定化容量をスタンプ（商用シミュレータ相当）
        
        高インピーダンスノードの収束を改善
        """
        if self.node_cap <= 0:
            return
        
        geq = self.node_cap / dt
        
        for i in range(n):
            if i in vsrc_nodes:
                continue  # 電圧源ノードはスキップ
            
            v_now = x[i]
            v_prev = self.node_v_prev.get(i, v_now)
            current = geq * (v_now - v_prev)
            
            # KCL残差への寄与
            J[i, i] -= geq
            f[i] -= current
    
    def _newton_iteration(self, x_trial: np.ndarray, t: float, dt: float,
                          vsrc_nodes: Dict[int, str], vsrc_idx: Dict[str, int],
                          n: int, total_size: int, max_iter: int, tol: float,
                          gmin_scale: float = 1.0, cap_scale: float = 1.0,
                          verbose: bool = False) -> Tuple[bool, np.ndarray]:
        """Newton反復を実行（polish対応版）
        
        Args:
            gmin_scale: gminのスケール係数（polishで減らす）
            cap_scale: 安定化容量のスケール係数（polishで減らす）
            verbose: デバッグ出力を有効化
        
        Returns:
            (converged, x_trial)
        """
        effective_gmin = self.gmin_base * gmin_scale
        effective_node_cap = self.node_cap_base * cap_scale
        
        # 内部ノード数を取得
        n_internal = self.num_internal_nodes if self.use_internal_nodes else 0
        n_total_nodes = n + n_internal  # 外部ノード + 内部ノード
        
        for iteration in range(max_iter):
            J = np.zeros((total_size, total_size))
            f = np.zeros(total_size)
            
            # GMINスタンプ（スケール適用）- 外部ノード + 内部ノード
            for i in range(n_total_nodes):
                if i not in vsrc_nodes:
                    J[i, i] += effective_gmin
                    f[i] += effective_gmin * (0.0 - x_trial[i])
            
            # 安定化容量スタンプ（スケール適用）
            # 後退オイラー: i_C = C/dt * (v_now - v_prev)
            # 容量電流はノードから流出（GND方向）なので f[i] -= i_C
            # J[i,i] -= d(-i_C)/dv = -C/dt より J[i,i] -= -geq = J[i,i] += geq は間違い
            # 正しくは: f[i] -= current, J[i,i] -= geq
            if effective_node_cap > 0:
                geq = effective_node_cap / dt
                for i in range(n_total_nodes):
                    if i in vsrc_nodes:
                        continue
                    v_now = x_trial[i]
                    v_prev = self.node_v_prev.get(i, v_now)
                    current = geq * (v_now - v_prev)
                    J[i, i] -= geq
                    f[i] -= current
            
            # 各素子のスタンプ
            for name, comp in self.circuit.components.items():
                if self._is_resistor(comp):
                    self._stamp_resistor_nr(J, f, comp, x_trial)
                elif self._is_vsource(comp):
                    self._stamp_vsource_nr(J, f, comp, x_trial, t, vsrc_idx[name])
                elif self._is_isource(comp):
                    self._stamp_isource_nr(f, comp)
                elif isinstance(comp, Capacitor):
                    self._stamp_capacitor_nr(J, f, comp, x_trial, dt)
                elif self._is_mosfet(comp):
                    self._stamp_mosfet_nr(J, f, comp, x_trial, dt, vsrc_nodes)
            
            # Newton-Raphson: J*dx = -f
            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return False, x_trial
            
            # Vt clipping
            dx = self._vt_clipping(dx, vsrc_nodes, n)
            
            # ダンピング（外部+内部ノードを考慮）
            max_dx_v = 0.0
            for i in range(n_total_nodes):
                if i not in vsrc_nodes:
                    max_dx_v = max(max_dx_v, abs(dx[i]))
            
            if max_dx_v > 0.1:
                damping = min(1.0, 0.1 / max_dx_v)
            elif max_dx_v > 0.01:
                damping = min(1.0, 0.3 / max_dx_v)
            else:
                damping = 1.0
            
            # 更新
            for i in range(total_size):
                if i < n and i in vsrc_nodes:
                    continue
                x_trial[i] += damping * dx[i]
            
            # 収束判定（外部+内部ノードを考慮）
            max_dx = 0.0
            max_residual = 0.0
            for i in range(n_total_nodes):
                if i not in vsrc_nodes:
                    max_dx = max(max_dx, abs(damping * dx[i]))
                    max_residual = max(max_residual, abs(f[i]))
            
            if verbose and (iteration < 5 or iteration % 10 == 0):
                print(f"  [NR] iter {iteration}: max_dx={max_dx:.3e}, max_res={max_residual:.3e}, damp={damping:.2f}")
            
            if max_dx < tol and max_residual < tol * 1e6:
                if verbose:
                    print(f"  [NR] Converged at iter {iteration}")
                return True, x_trial
        
        if verbose:
            print(f"  [NR] Not converged after {max_iter} iterations, max_dx={max_dx:.3e}, max_res={max_residual:.3e}")
        return False, x_trial
    
    def solve(self, tstop: float, dt: float = 1e-12, max_iter: int = 50,
              tol: float = 1e-6, verbose: bool = False) -> Tuple[List[float], Dict[str, List[float]]]:
        """TRAN解析を実行（Newton-Raphson残差形式 + Polish）
        
        安定化処理:
        - 各ノードへのgmin挿入
        - 各ノードへの安定化容量(node_cap)挿入
        - 適応タイムステップ（パルスエッジ検出）
        - Vt clipping（電圧更新量制限）
        - 強化されたダンピング
        - **Polish処理**: 収束後に安定化を減らして再収束（商用シミュレータ相当）
        - 内部ノード（dp/sp）サポート（DCアナライザから継承）
        """
        n = self.num_nodes - 1
        num_vsrc = self._count_vsources()
        # 内部ノード数を追加
        n_internal = self.num_internal_nodes if self.use_internal_nodes else 0
        total_size = n + n_internal + num_vsrc
        
        if total_size == 0:
            return [], {}
        
        # 電圧源インデックスとノードマッピング（内部ノード後）
        vsrc_base_idx = n + n_internal
        vsrc_idx = {}
        vsrc_nodes = {}  # 電圧源ノード -> 目標電圧
        idx = vsrc_base_idx
        for name, comp in self.circuit.components.items():
            if self._is_vsource(comp):
                vsrc_idx[name] = idx
                n_pos = self.circuit.node_idx(comp.nodes[0]) - 1
                if n_pos >= 0:
                    vsrc_nodes[n_pos] = name
                idx += 1
        
        # 初期値（DC解から）
        x = np.zeros(total_size)
        for i in range(1, self.num_nodes):
            node = self.circuit.idx_node(i)
            x[i-1] = self.dc_solution.get(node, 0.0)
        
        # 内部ノードの初期化（外部ドレイン/ソースと同じ電圧）
        if self.use_internal_nodes:
            for name, info in self.internal_node_map.items():
                comp = self.circuit.components[name]
                dp_idx = info.get('dp', -1)
                sp_idx = info.get('sp', -1)
                if dp_idx >= 0:
                    nd = self.circuit.node_idx(comp.nodes[0]) - 1
                    x[dp_idx] = x[nd] if nd >= 0 else 0.0
                if sp_idx >= 0:
                    ns = self.circuit.node_idx(comp.nodes[2]) - 1
                    x[sp_idx] = x[ns] if ns >= 0 else 0.0
        
        # 安定化容量用のノード電圧履歴を初期化（外部+内部ノード）
        for i in range(n + n_internal):
            self.node_v_prev[i] = x[i]
        
        x_prev = x.copy()
        
        # 結果格納
        times = [0.0]
        waves = {self.circuit.idx_node(i): [x[i-1]] for i in range(1, self.num_nodes)}
        waves["0"] = [0.0]
        
        t = 0.0
        step_count = 0
        
        # 最小タイムステップ：パルスのrise/fall時間の1/10を基準に
        min_dt = dt / 10  # デフォルト
        for name, comp in self.circuit.components.items():
            if self._is_vsource(comp) and hasattr(comp, 'pulse_params') and comp.pulse_params:
                p = comp.pulse_params
                tr = p.get('rise', p.get('tr', 1e-12))
                tf = p.get('fall', p.get('tf', 1e-12))
                edge_min = min(tr, tf) / 5  # エッジ時間の1/5（1/10から緩和）
                min_dt = max(min(min_dt, edge_min), 1e-13)  # 0.1ps下限
        
        # 収束失敗カウンタ
        fail_count = 0
        max_fail = 10
        
        while t < tstop:
            # 適応タイムステップ：パルスエッジ検出
            if self.adaptive_dt:
                current_dt = self._detect_pulse_edge(t, dt)
            else:
                current_dt = dt
            
            step_accepted = False
            attempts = 0
            max_attempts = 20  # 最大試行回数
            
            while not step_accepted and current_dt >= min_dt and attempts < max_attempts:
                attempts += 1
                t_new = t + current_dt
                x_trial = x.copy()
                
                # 電圧源ノードを目標電圧に設定
                for node_idx, vsrc_name in vsrc_nodes.items():
                    comp = self.circuit.components[vsrc_name]
                    x_trial[node_idx] = comp.get_voltage(t_new)
            
                # Phase 1: フル安定化で収束
                # 最初の数ステップだけverbose出力
                nr_verbose = verbose and step_count < 3
                converged, x_trial = self._newton_iteration(
                    x_trial, t_new, current_dt, vsrc_nodes, vsrc_idx,
                    n, total_size, max_iter, tol,
                    gmin_scale=1.0, cap_scale=1.0, verbose=nr_verbose
                )
                
                # Phase 2: Polish処理（安定化を段階的に減らして再収束）
                if converged and self.polish_steps > 0:
                    for polish_idx in range(self.polish_steps):
                        # 安定化を指数的に減らす（1.0 -> 0.1 -> 0.01 -> 0.001）
                        scale = 10 ** (-(polish_idx + 1))
                        
                        polish_converged, x_polished = self._newton_iteration(
                            x_trial.copy(), t_new, current_dt, vsrc_nodes, vsrc_idx,
                            n, total_size, max_iter // 2, tol,
                            gmin_scale=scale, cap_scale=scale
                        )
                        
                        if polish_converged:
                            x_trial = x_polished
                        else:
                            # Polishで収束しなければ前の結果を使う
                            break
                
                if converged:
                    step_accepted = True
                    t = t_new
                    x = x_trial.copy()
                    fail_count = 0  # 成功したらリセット
                else:
                    # サブステップに分割
                    current_dt /= 2
            
            if not step_accepted:
                fail_count += 1
                if verbose:
                    print(f"[TRAN] Warning: Not converged at t={t:.3e}, fail_count={fail_count}")
                
                if fail_count >= max_fail:
                    print(f"[TRAN] ERROR: Too many convergence failures at t={t:.3e}, aborting")
                    break
                
                # 強制的に小さいステップで進める（結果は保存しない）
                t += min_dt
                continue  # 履歴更新・結果保存をスキップ
            
            step_count += 1
            
            # ノード電圧履歴を更新（安定化容量用）- 外部+内部ノード
            for i in range(n + n_internal):
                self.node_v_prev[i] = x[i]
            
            # 電荷履歴を更新
            self._update_cap_history(x)
            
            # 結果保存
            times.append(t)
            for i in range(1, self.num_nodes):
                node = self.circuit.idx_node(i)
                waves[node].append(x[i-1])
            waves["0"].append(0.0)
            
            if verbose and step_count % 10000 == 0:
                print(f"[TRAN] t = {t:.3e} s, steps = {step_count}")
        
        if verbose:
            print(f"[TRAN] Complete: {step_count} steps")
        
        return times, waves
    
    def _stamp_resistor(self, G: np.ndarray, comp: Resistor):
        """抵抗のスタンプ"""
        n1 = self.circuit.node_idx(comp.nodes[0]) - 1
        n2 = self.circuit.node_idx(comp.nodes[1]) - 1
        g = 1.0 / comp.resistance
        
        if n1 >= 0:
            G[n1, n1] += g
            if n2 >= 0:
                G[n1, n2] -= g
        if n2 >= 0:
            G[n2, n2] += g
            if n1 >= 0:
                G[n2, n1] -= g
    
    def _stamp_vsource(self, G: np.ndarray, I: np.ndarray, comp: VSource, 
                       t: float, vsrc_row: int):
        """電圧源のスタンプ"""
        n1 = self.circuit.node_idx(comp.nodes[0]) - 1
        n2 = self.circuit.node_idx(comp.nodes[1]) - 1
        
        if n1 >= 0:
            G[n1, vsrc_row] += 1.0
            G[vsrc_row, n1] += 1.0
        if n2 >= 0:
            G[n2, vsrc_row] -= 1.0
            G[vsrc_row, n2] -= 1.0
        
        I[vsrc_row] = comp.get_voltage(t)
    
    def _stamp_isource(self, I: np.ndarray, comp: ISource):
        """電流源のスタンプ"""
        n1 = self.circuit.node_idx(comp.nodes[0]) - 1
        n2 = self.circuit.node_idx(comp.nodes[1]) - 1
        
        if n1 >= 0:
            I[n1] -= comp.dc_value
        if n2 >= 0:
            I[n2] += comp.dc_value
    
    def _stamp_capacitor_be(self, G: np.ndarray, I: np.ndarray, comp: Capacitor,
                            x: np.ndarray, dt: float):
        """キャパシタのスタンプ（後退オイラー法）"""
        n1 = self.circuit.node_idx(comp.nodes[0]) - 1
        n2 = self.circuit.node_idx(comp.nodes[1]) - 1
        c = comp.capacitance
        
        # BE: i = C/dt * (v - v_prev)
        # コンパニオンモデル: Geq = C/dt, Ieq = Geq * v_prev
        geq = c / dt
        v_prev = self.cap_v_prev.get(comp.name, 0.0)
        ieq = geq * v_prev
        
        if n1 >= 0:
            G[n1, n1] += geq
            if n2 >= 0:
                G[n1, n2] -= geq
            I[n1] += ieq
        if n2 >= 0:
            G[n2, n2] += geq
            if n1 >= 0:
                G[n2, n1] -= geq
            I[n2] -= ieq
    
    def _stamp_mosfet_tran(self, G: np.ndarray, I: np.ndarray, comp: MOSFET, x: np.ndarray):
        """MOSFET TRANスタンプ（外部ノードのみ）
        
        D/Sスワップ時の処理:
        - NMOSでVd < Vs、PMOSでVd > Vsの場合にスワップ
        - スワップ時はBSIM4の評価結果（Ids, gm, gds, gmb）の符号を反転
        - 物理ノードへのスタンプは元のまま（電流の流れる方向が逆になる）
        """
        nd = self.circuit.node_idx(comp.nodes[0]) - 1
        ng = self.circuit.node_idx(comp.nodes[1]) - 1
        ns = self.circuit.node_idx(comp.nodes[2]) - 1
        nb = self.circuit.node_idx(comp.nodes[3]) - 1
        
        vd = x[nd] if nd >= 0 else 0.0
        vg = x[ng] if ng >= 0 else 0.0
        vs = x[ns] if ns >= 0 else 0.0
        vb = x[nb] if nb >= 0 else 0.0
        
        is_pmos = comp.mos_type.lower() in ('pmos', 'p')
        vds_phys = vd - vs
        swap = (vds_phys > 0.05) if is_pmos else (vds_phys < -0.05)
        
        if swap:
            # スワップ: 物理的なソースがドレインとして動作
            vd_eval, vs_eval = vs, vd
        else:
            vd_eval, vs_eval = vd, vs
        
        vgs = vg - vs_eval
        vds = vd_eval - vs_eval
        vbs = vb - vs_eval
        
        # BSIM4評価
        if comp.name in self.dc_analyzer.bsim4_devices:
            device = self.dc_analyzer.bsim4_devices[comp.name]
            result = device.evaluate(vgs, vds, vbs)
            ids = result['ids']
            gm = max(result['gm'], 1e-12)
            gds = max(result['gds'], 1e-12)
            gmb = max(result['gmb'], 0.0)
        else:
            ids, gm, gds, gmb = self.dc_analyzer._simple_mos_model(vgs, vds, vbs, is_pmos, comp)
        
        # スワップ時は電流方向が逆（Idsは「評価用ドレイン→評価用ソース」の電流）
        # 物理ノードへスタンプする際、スワップ時は符号反転
        if swap:
            ids = -ids
        
        # gm, gds, gmbは常に正
        # 電流式: Id = gm*(Vg-Vs_eval) + gds*(Vd_eval-Vs_eval) + gmb*(Vb-Vs_eval)
        # ここでVd_eval, Vs_evalはスワップ後の電圧
        
        # 物理ノードへのスタンプ（常にnd, nsを使用）
        # スワップ時でも物理ドレインnd、物理ソースnsへスタンプ
        
        # 線形化: I = I0 + dI/dVg*(Vg-V0g) + dI/dVd*(Vd-V0d) + dI/dVs*(Vs-V0s) + dI/dVb*(Vb-V0b)
        # Newton-Raphson: G*V = I(V0) - J*V0 + J*V = Ieq + J*V
        
        if swap:
            # スワップ時: dId_phys/dVd_phys, dId_phys/dVs_phys の関係が逆転
            # 物理ドレイン電流 = -Ids_eval（評価用ドレインは物理ソース）
            # dId_phys/dVg = -gm
            # dId_phys/dVd = -gds (物理ドレインは評価用ソース)
            # dId_phys/dVs = gm + gds + gmb (物理ソースは評価用ドレイン)
            # dId_phys/dVb = -gmb
            gm_eff = -gm
            gds_d_phys = gm + gds + gmb  # 物理ドレインに対する偏微分
            gds_s_phys = -gds           # 物理ソースに対する偏微分
            gmb_eff = -gmb
        else:
            # 通常時
            gm_eff = gm
            gds_d_phys = gds
            gds_s_phys = -(gm + gds + gmb)
            gmb_eff = gmb
        
        # 線形化のための定数項
        V0_g = vg
        V0_d = vd
        V0_s = vs
        V0_b = vb
        J_dot_V0 = gm_eff*V0_g + gds_d_phys*V0_d + gds_s_phys*V0_s + gmb_eff*V0_b
        Ieq = ids - J_dot_V0
        
        # 物理ドレインノードスタンプ
        if nd >= 0:
            G[nd, nd] += gds_d_phys
            if ng >= 0:
                G[nd, ng] += gm_eff
            if ns >= 0:
                G[nd, ns] += gds_s_phys
            if nb >= 0:
                G[nd, nb] += gmb_eff
            I[nd] += Ieq
        
        # 物理ソースノードスタンプ（電流保存）
        if ns >= 0:
            G[ns, ns] += (-gds_s_phys)
            if ng >= 0:
                G[ns, ng] += (-gm_eff)
            if nd >= 0:
                G[ns, nd] += (-gds_d_phys)
            if nb >= 0:
                G[ns, nb] += (-gmb_eff)
            I[ns] -= Ieq
    
    def _update_cap_history(self, x: np.ndarray):
        """キャパシタ電圧履歴を更新"""
        for name, comp in self.circuit.components.items():
            if isinstance(comp, Capacitor):
                n1 = self.circuit.node_idx(comp.nodes[0]) - 1
                n2 = self.circuit.node_idx(comp.nodes[1]) - 1
                v1 = x[n1] if n1 >= 0 else 0.0
                v2 = x[n2] if n2 >= 0 else 0.0
                self.cap_v_prev[name] = v1 - v2
        
        # MOSFET容量の電圧履歴を更新（CGS/CGD/Cbd/Cbsモデル用）
        if self.use_mos_caps:
            for name, comp in self.circuit.components.items():
                if self._is_mosfet(comp) and name in self.dc_analyzer.bsim4_devices:
                    nd = self.circuit.node_idx(comp.nodes[0]) - 1
                    ng = self.circuit.node_idx(comp.nodes[1]) - 1
                    ns = self.circuit.node_idx(comp.nodes[2]) - 1
                    nb = self.circuit.node_idx(comp.nodes[3]) - 1
                    
                    vd = x[nd] if nd >= 0 else 0.0
                    vg = x[ng] if ng >= 0 else 0.0
                    vs = x[ns] if ns >= 0 else 0.0
                    vb = x[nb] if nb >= 0 else 0.0
                    
                    # CGS/CGD/Cbd/Cbsの電圧履歴を保存
                    self.mos_q_prev[name] = {
                        'vgd': vg - vd,
                        'vgs': vg - vs,
                        'vdb': vd - vb,
                        'vsb': vs - vb,
                    }

    # =============================================================================
    # Newton-Raphson形式のスタンプメソッド（TRAN解析用）
    # =============================================================================
    
    def _stamp_resistor_nr(self, J: np.ndarray, f: np.ndarray, comp: Resistor, x: np.ndarray):
        """抵抗のスタンプ（Newton-Raphson残差形式）"""
        n1 = self.circuit.node_idx(comp.nodes[0]) - 1
        n2 = self.circuit.node_idx(comp.nodes[1]) - 1
        g = 1.0 / comp.resistance
        
        v1 = x[n1] if n1 >= 0 else 0.0
        v2 = x[n2] if n2 >= 0 else 0.0
        current = g * (v1 - v2)
        
        if n1 >= 0:
            J[n1, n1] -= g
            if n2 >= 0:
                J[n1, n2] += g
            f[n1] -= current
        if n2 >= 0:
            J[n2, n2] -= g
            if n1 >= 0:
                J[n2, n1] += g
            f[n2] += current
    
    def _stamp_vsource_nr(self, J: np.ndarray, f: np.ndarray, comp: VSource,
                          x: np.ndarray, t: float, vsrc_row: int):
        """電圧源のスタンプ（Newton-Raphson残差形式）"""
        n_pos = self.circuit.node_idx(comp.nodes[0]) - 1
        n_neg = self.circuit.node_idx(comp.nodes[1]) - 1
        
        vdc = comp.get_voltage(t)
        
        v_pos = x[n_pos] if n_pos >= 0 else 0.0
        v_neg = x[n_neg] if n_neg >= 0 else 0.0
        i_src = x[vsrc_row]
        
        # 残差: v_pos - v_neg - vdc = 0
        f[vsrc_row] = v_pos - v_neg - vdc
        if n_pos >= 0:
            J[vsrc_row, n_pos] = 1.0
        if n_neg >= 0:
            J[vsrc_row, n_neg] = -1.0
        
        # KCL: 電流の寄与
        if n_pos >= 0:
            f[n_pos] -= i_src
            J[n_pos, vsrc_row] -= 1.0
        if n_neg >= 0:
            f[n_neg] += i_src
            J[n_neg, vsrc_row] += 1.0
    
    def _stamp_isource_nr(self, f: np.ndarray, comp: ISource):
        """電流源のスタンプ（Newton-Raphson残差形式）"""
        n1 = self.circuit.node_idx(comp.nodes[0]) - 1
        n2 = self.circuit.node_idx(comp.nodes[1]) - 1
        
        if n1 >= 0:
            f[n1] -= comp.dc_value
        if n2 >= 0:
            f[n2] += comp.dc_value
    
    def _stamp_capacitor_nr(self, J: np.ndarray, f: np.ndarray, comp: Capacitor,
                            x: np.ndarray, dt: float):
        """キャパシタのスタンプ（Newton-Raphson残差形式、後退オイラー法）"""
        n1 = self.circuit.node_idx(comp.nodes[0]) - 1
        n2 = self.circuit.node_idx(comp.nodes[1]) - 1
        c = comp.capacitance
        
        v1 = x[n1] if n1 >= 0 else 0.0
        v2 = x[n2] if n2 >= 0 else 0.0
        v_prev = self.cap_v_prev.get(comp.name, 0.0)
        
        # BE: i = C/dt * (v - v_prev)
        geq = c / dt
        current = geq * ((v1 - v2) - v_prev)
        
        if n1 >= 0:
            J[n1, n1] -= geq
            if n2 >= 0:
                J[n1, n2] += geq
            f[n1] -= current
        if n2 >= 0:
            J[n2, n2] -= geq
            if n1 >= 0:
                J[n2, n1] += geq
            f[n2] += current
    
    def _stamp_mosfet_nr(self, J: np.ndarray, f: np.ndarray, comp: MOSFET, x: np.ndarray, dt: float = None, vsrc_nodes: Dict[int, str] = None):
        """MOSFETのスタンプ（Newton-Raphson残差形式）- TRAN解析用
        
        dtが指定された場合、MOSFET寄生容量も含める
        vsrc_nodesが指定された場合、電圧源ノードへの容量スタンプをスキップ
        内部ノード対応版: use_internal_nodesがTrueの場合、dp/spを使用
        """
        if vsrc_nodes is None:
            vsrc_nodes = {}
            
        nd = self.circuit.node_idx(comp.nodes[0]) - 1  # Drain（外部）
        ng = self.circuit.node_idx(comp.nodes[1]) - 1  # Gate
        ns = self.circuit.node_idx(comp.nodes[2]) - 1  # Source（外部）
        nb = self.circuit.node_idx(comp.nodes[3]) - 1  # Bulk
        
        # 内部ノード対応
        if self.use_internal_nodes:
            info = self.internal_node_map.get(comp.name, {})
            ndp = info.get('dp', -1)  # drain prime
            nsp = info.get('sp', -1)  # source prime
            gdpr = info.get('gdpr', 0.0)
            gspr = info.get('gspr', 0.0)
        else:
            ndp, nsp = nd, ns
            gdpr, gspr = 0.0, 0.0
        
        # 内部ノードがない場合は外部ノードを使用
        if ndp < 0:
            ndp = nd
        if nsp < 0:
            nsp = ns
        
        # 電圧取得
        vd = x[nd] if nd >= 0 else 0.0
        vg = x[ng] if ng >= 0 else 0.0
        vs = x[ns] if ns >= 0 else 0.0
        vb = x[nb] if nb >= 0 else 0.0
        vdp = x[ndp] if ndp >= 0 and ndp != nd else vd
        vsp = x[nsp] if nsp >= 0 and nsp != ns else vs
        
        is_pmos = comp.mos_type.lower() in ('pmos', 'p')
        
        # スワップ判定は内部ノード間電圧で
        vds_int = vdp - vsp
        swap = (vds_int > 0.05) if is_pmos else (vds_int < -0.05)
        
        # 評価用電圧（内部ノード基準）
        if swap:
            vdp_eval, vsp_eval = vsp, vdp
        else:
            vdp_eval, vsp_eval = vdp, vsp
        
        vgs = vg - vsp_eval
        vds = vdp_eval - vsp_eval
        vbs = vb - vsp_eval
        
        # BSIM4評価
        if comp.name in self.dc_analyzer.bsim4_devices:
            device = self.dc_analyzer.bsim4_devices[comp.name]
            result = device.evaluate(vgs, vds, vbs)
            ids = result['ids']
            gm = max(result['gm'], 1e-12)
            gds = max(result['gds'], 1e-9)  # 最小gdsを増加（収束改善）
            gmb = max(result['gmb'], 0.0)
            # Gdpr/Gsprは評価結果を使用（Julia同等）
            gdpr = result.get('Gdpr', gdpr)
            gspr = result.get('Gspr', gspr)
        else:
            ids, gm, gds, gmb = self.dc_analyzer._simple_mos_model(vgs, vds, vbs, is_pmos, comp)
            gds = max(gds, 1e-9)
        
        # スワップ時は電流方向が逆
        if swap:
            ids = -ids
        
        # ===== Gdpr抵抗スタンプ（D外部 - dp間） =====
        gdpr_eff = gdpr if gdpr > 1e-12 else 1e12
        if gdpr_eff > 0 and nd >= 0 and ndp >= 0 and nd != ndp:
            i_gdpr = gdpr_eff * (vd - vdp)
            f[nd] -= i_gdpr
            J[nd, nd] -= gdpr_eff
            J[nd, ndp] += gdpr_eff
            f[ndp] += i_gdpr
            J[ndp, nd] += gdpr_eff
            J[ndp, ndp] -= gdpr_eff
        
        # ===== Gspr抵抗スタンプ（S外部 - sp間） =====
        gspr_eff = gspr if gspr > 1e-12 else 1e12
        if gspr_eff > 0 and ns >= 0 and nsp >= 0 and ns != nsp:
            i_gspr = gspr_eff * (vs - vsp)
            f[ns] -= i_gspr
            J[ns, ns] -= gspr_eff
            J[ns, nsp] += gspr_eff
            f[nsp] += i_gspr
            J[nsp, ns] += gspr_eff
            J[nsp, nsp] -= gspr_eff
        
        # Jacobian計算（MOSFET本体：dp, sp間）
        if swap:
            gm_eff, gmb_eff = -gm, -gmb
            gds_dp = gm + gds + gmb
            gds_sp = -gds
        else:
            gm_eff, gmb_eff = gm, gmb
            gds_dp = gds
            gds_sp = -(gm + gds + gmb)
        
        # dp KCL: f[dp] -= Ids
        if ndp >= 0:
            f[ndp] -= ids
            J[ndp, ndp] -= gds_dp
            if ng >= 0: J[ndp, ng] -= gm_eff
            if nsp >= 0: J[ndp, nsp] -= gds_sp
            if nb >= 0: J[ndp, nb] -= gmb_eff
        
        # sp KCL: f[sp] += Ids
        if nsp >= 0:
            f[nsp] += ids
            J[nsp, nsp] += gds_sp
            if ng >= 0: J[nsp, ng] += gm_eff
            if ndp >= 0: J[nsp, ndp] += gds_dp
            if nb >= 0: J[nsp, nb] += gmb_eff
        
        # MOSFET寄生容量のスタンプ（TRAN解析時のみ）
        # 内部ノード使用時はdp/spを渡す
        if dt is not None and self.use_mos_caps and comp.name in self.dc_analyzer.bsim4_devices:
            self._stamp_mosfet_caps(J, f, comp, ndp, ng, nsp, nb, vdp, vg, vsp, vb, 
                                    vgs, vds, vbs, swap, dt, vsrc_nodes)
    
    def _stamp_mosfet_caps(self, J: np.ndarray, f: np.ndarray, comp: MOSFET,
                           nd: int, ng: int, ns: int, nb: int,
                           vd: float, vg: float, vs: float, vb: float,
                           vgs: float, vds: float, vbs: float,
                           swap: bool, dt: float, vsrc_nodes: Dict[int, str]):
        """MOSFET寄生容量のスタンプ（リファクタリング後）
        
        符号規約: KCL残差形式で流出電流を正とする
        - f[n1] -= i_cap (n1から流出)
        - f[n2] += i_cap (n2へ流入)
        - J[n1,n1] -= geq, J[n1,n2] += geq
        """
        device = self.dc_analyzer.bsim4_devices[comp.name]
        result = device.evaluate(vgs, vds, vbs)
        
        # BSIM4から容量値取得（内部ソース/ドレイン基準）
        cgdb_int = abs(result.get('cgdb', 0.0))
        cgsb_int = abs(result.get('cgsb', 0.0))
        capbd_int = abs(result.get('capbd', 0.0))
        capbs_int = abs(result.get('capbs', 0.0))
        
        # スワップ時は物理D/Sと内部D/Sを入れ替え
        if swap:
            cgd, cgs = cgsb_int, cgdb_int
            cbd, cbs = capbs_int, capbd_int
        else:
            cgd, cgs = cgdb_int, cgsb_int
            cbd, cbs = capbd_int, capbs_int
        
        # 最小値を確保
        min_cap = 1e-18
        cgd = max(cgd, min_cap)
        cgs = max(cgs, min_cap)
        cbd = max(cbd, min_cap) + self.min_drain_cap
        cbs = max(cbs, min_cap)
        
        # 各容量をスタンプ（共通関数使用）
        prev = self.mos_q_prev.get(comp.name, {})
        
        # CGD: ゲート-ドレイン間
        if ng != nd:
            self._stamp_two_node_cap(J, f, ng, nd, cgd, vg - vd, 
                                     prev.get('vgd', vg - vd), dt, vsrc_nodes)
        
        # CGS: ゲート-ソース間
        if ng != ns:
            self._stamp_two_node_cap(J, f, ng, ns, cgs, vg - vs,
                                     prev.get('vgs', vg - vs), dt, vsrc_nodes)
        
        # Cbd: ドレイン-バルク間
        if nd != nb:
            self._stamp_two_node_cap(J, f, nd, nb, cbd, vd - vb,
                                     prev.get('vdb', vd - vb), dt, vsrc_nodes)
        
        # Cbs: ソース-バルク間
        if ns != nb:
            self._stamp_two_node_cap(J, f, ns, nb, cbs, vs - vb,
                                     prev.get('vsb', vs - vb), dt, vsrc_nodes)
    
    def _stamp_two_node_cap(self, J: np.ndarray, f: np.ndarray,
                            n1: int, n2: int, cap: float,
                            v_diff: float, v_diff_prev: float,
                            dt: float, vsrc_nodes: Dict[int, str]):
        """2端子間容量のスタンプ（共通ヘルパー関数）
        
        符号規約（KCL残差形式、流出電流を正）:
        - i_cap = C/dt * (v12 - v12_prev)  (n1→n2方向の電流)
        - f[n1] -= i_cap (n1から流出)
        - f[n2] += i_cap (n2へ流入)
        - J[n1,n1] -= geq, J[n1,n2] += geq
        - J[n2,n2] -= geq, J[n2,n1] += geq
        
        Args:
            n1, n2: ノードインデックス（-1の場合はGND）
            cap: 容量値 [F]
            v_diff: 現在の電圧差 (v1 - v2)
            v_diff_prev: 前ステップの電圧差
            dt: タイムステップ
            vsrc_nodes: 電圧源ノード（スキップ用）
        """
        geq = cap / dt
        i_cap = geq * (v_diff - v_diff_prev)
        
        if n1 >= 0 and n1 not in vsrc_nodes:
            f[n1] -= i_cap
            J[n1, n1] -= geq
            if n2 >= 0 and n2 not in vsrc_nodes:
                J[n1, n2] += geq
        
        if n2 >= 0 and n2 not in vsrc_nodes:
            f[n2] += i_cap
            J[n2, n2] -= geq
            if n1 >= 0 and n1 not in vsrc_nodes:
                J[n2, n1] += geq


# =============================================================================
# ネットリストパーサー
# =============================================================================

def parse_netlist(netlist_str: str) -> Circuit:
    """Spectreライクなネットリストをパース"""
    circuit = Circuit()
    
    # 行を結合
    lines = []
    current_line = ""
    for line in netlist_str.split('\n'):
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('//'):
            continue
        if line.endswith('\\'):
            current_line += line[:-1] + " "
        else:
            current_line += line
            lines.append(current_line)
            current_line = ""
    
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        
        name = parts[0]
        
        if name.upper().startswith('M'):
            nodes, params, model_name = _parse_mosfet(line)
            mos_type = "pmos" if "p" in model_name.lower() else "nmos"
            
            # モデルパラメータにtypeを設定
            params['type'] = mos_type
            
            comp = MOSFET(
                name=name,
                nodes=nodes,
                comp_type=ComponentType.MOSFET,
                w=params.pop('w', 1e-6),
                l=params.pop('l', 180e-9),
                nf=int(params.pop('nf', 1)),
                mos_type=mos_type,
                model_params=params,
            )
            circuit.add_component(comp)
        
        elif name.upper().startswith('V'):
            nodes, params = _parse_two_terminal(line)
            dc_val = params.get('dc', 0.0)
            
            pulse_params = None
            if params.get('type', 'dc').lower() == 'pulse':
                pulse_params = {
                    'val0': params.get('val0', 0.0),
                    'val1': params.get('val1', 0.0),
                    'delay': params.get('delay', 0.0),
                    'rise': params.get('rise', 1e-12),
                    'fall': params.get('fall', 1e-12),
                    'width': params.get('width', 1e-9),
                    'period': params.get('period', 1e-6),
                }
            
            comp = VSource(
                name=name,
                nodes=nodes,
                comp_type=ComponentType.VSOURCE,
                dc_value=dc_val,
                pulse_params=pulse_params,
            )
            circuit.add_component(comp)
        
        elif name.upper().startswith('I'):
            nodes, params = _parse_two_terminal(line)
            comp = ISource(
                name=name,
                nodes=nodes,
                comp_type=ComponentType.ISOURCE,
                dc_value=params.get('dc', 0.0),
            )
            circuit.add_component(comp)
        
        elif name.upper().startswith('R'):
            nodes, params = _parse_two_terminal(line)
            comp = Resistor(
                name=name,
                nodes=nodes,
                comp_type=ComponentType.RESISTOR,
                resistance=params.get('r', 1e6),
            )
            circuit.add_component(comp)
        
        elif name.upper().startswith('C'):
            nodes, params = _parse_two_terminal(line)
            comp = Capacitor(
                name=name,
                nodes=nodes,
                comp_type=ComponentType.CAPACITOR,
                capacitance=params.get('c', 1e-15),
            )
            circuit.add_component(comp)
    
    return circuit


def _parse_mosfet(line: str) -> Tuple[List[str], Dict[str, Any], str]:
    """MOSFETをパース"""
    nodes = []
    params = {}
    model_name = ""
    
    if '(' in line:
        node_start = line.index('(') + 1
        node_end = line.index(')')
        node_str = line[node_start:node_end]
        nodes = node_str.split()
        rest = line[node_end+1:].strip().split()
    else:
        parts = line.split()
        nodes = parts[1:5]
        rest = parts[5:]
    
    for item in rest:
        if '=' in item:
            key, val = item.split('=', 1)
            params[key.lower()] = _parse_value(val)
        elif not model_name:
            model_name = item
    
    return nodes, params, model_name


def _parse_two_terminal(line: str) -> Tuple[List[str], Dict[str, Any]]:
    """2端子素子をパース"""
    nodes = []
    params = {}
    
    if '(' in line:
        node_start = line.index('(') + 1
        node_end = line.index(')')
        node_str = line[node_start:node_end]
        nodes = node_str.split()
        rest = line[node_end+1:].strip().split()
    else:
        parts = line.split()
        nodes = parts[1:3]
        rest = parts[3:]
    
    for item in rest:
        if '=' in item:
            key, val = item.split('=', 1)
            try:
                params[key.lower()] = _parse_value(val)
            except:
                params[key.lower()] = val
    
    return nodes, params


def _parse_value(val_str: str) -> float:
    """値をパース（単位接頭辞対応）"""
    val_str = val_str.strip().lower()
    
    suffixes = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'meg': 1e6, 'g': 1e9, 't': 1e12,
    }
    
    for suffix, mult in sorted(suffixes.items(), key=lambda x: -len(x[0])):
        if val_str.endswith(suffix):
            return float(val_str[:-len(suffix)]) * mult
    
    return float(val_str)


# =============================================================================
# ゴールデンデータとの比較用関数
# =============================================================================

def compare_with_golden(times: List[float], waves: Dict[str, List[float]],
                        golden_file: str, signal: str = "VOUT",
                        abs_tol: float = 0.05, rel_tol: float = 0.01) -> Tuple[bool, float, float]:
    """ゴールデンデータとの比較
    
    Returns:
        (passed, max_abs_error, max_rel_error)
    """
    import csv
    
    # ゴールデンデータ読み込み
    golden_t = []
    golden_v = []
    
    with open(golden_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # カラム名の検索
            t_key = None
            v_key = None
            for k in row.keys():
                if k.lower() in ('time', 't'):
                    t_key = k
                if signal.lower() in k.lower():
                    v_key = k
            
            if t_key and v_key:
                golden_t.append(float(row[t_key]))
                golden_v.append(float(row[v_key]))
    
    if not golden_t:
        print(f"Warning: No golden data found for signal '{signal}'")
        return False, float('inf'), float('inf')
    
    # シミュレーション波形
    sim_t = np.array(times)
    sim_v = np.array(waves.get(signal, []))
    
    if len(sim_v) == 0:
        print(f"Warning: Signal '{signal}' not found in simulation")
        return False, float('inf'), float('inf')
    
    # 補間してゴールデンタイムポイントでの値を取得
    golden_t = np.array(golden_t)
    golden_v = np.array(golden_v)
    
    # シミュレーション結果をゴールデンタイムに補間
    sim_v_interp = np.interp(golden_t, sim_t, sim_v)
    
    # 誤差計算
    abs_err = np.abs(sim_v_interp - golden_v)
    rel_err = abs_err / np.maximum(np.abs(golden_v), 1e-3)
    
    max_abs_err = np.max(abs_err)
    max_rel_err = np.max(rel_err)
    
    passed = max_abs_err <= abs_tol
    
    return passed, max_abs_err, max_rel_err


if __name__ == "__main__":
    # 簡単なテスト
    print("=" * 60)
    print("BSIM4 Circuit Analyzer Test")
    print("=" * 60)
    
    netlist = """
    M0 (VOUT VIN VSS VOUT) mp25od33_svt l=460n w=400n multi=1 nf=1
    V2 (VIN 0) vsource dc=1.0 type=dc
    V1 (VDD 0) vsource dc=3 type=dc
    V0 (VSS 0) vsource dc=0 type=dc
    I4 (VDD I4_MINUS) isource dc=10u type=dc
    R0 (I4_MINUS VOUT) resistor r=2.2K
    """
    
    circuit = parse_netlist(netlist)
    print(f"\nComponents: {len(circuit.components)}")
    
    # DC解析
    print("\n--- DC Analysis ---")
    dc_analyzer = BSIM4DCAnalyzer(circuit)
    node_voltages, mos_currents = dc_analyzer.solve(verbose=True)
    
    print("\nNode Voltages:")
    for node, v in sorted(node_voltages.items()):
        print(f"  {node}: {v:.6f} V")
    
    print("\nMOS Currents:")
    for name, i in mos_currents.items():
        print(f"  {name}: {i*1e6:.3f} uA")
