#!/usr/bin/env python3
"""
BSIM4 Python Wrapper - Juliaのbsim4_interface.jlと同等の機能を提供
"""
import ctypes
from ctypes import *
import numpy as np
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

# ===== ライブラリパスの検索 =====
def find_bsim4_library():
    """BSIM4ライブラリを検索"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    candidates = [
        os.path.join(project_root, "bsim450", "BSIM450", "src", "libbsim4_direct.so"),
        os.path.join(project_root, "bsim450", "lib", "libbsim4_direct.so"),
        os.path.join(script_dir, "libbsim4_direct.so"),
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find libbsim4_direct.so. Searched: {candidates}")


# ===== C構造体定義 =====
class BSIM4_InstParam(Structure):
    """インスタンスパラメータ（トランジスタの寸法など）"""
    _fields_ = [
        ("w", c_double),
        ("l", c_double),
        ("nf", c_double),
        ("multi", c_double),
        ("sa", c_double),
        ("sb", c_double),
        ("sd", c_double),
        ("sca", c_double),
        ("scb", c_double),
        ("scc", c_double),
        ("sc", c_double),
        ("min", c_double),
        ("ad", c_double),
        ("as_", c_double),  # 'as' is Python keyword
        ("pd", c_double),
        ("ps", c_double),
        ("nrd", c_double),
        ("nrs", c_double),
        ("off", c_double),
        ("rbdb", c_double),
        ("rbsb", c_double),
        ("rbpb", c_double),
        ("rbps", c_double),
        ("rbpd", c_double),
        ("delvto", c_double),
        ("xgw", c_double),
        ("ngcon", c_double),
    ]


class BSIM4_SimParam(Structure):
    """シミュレーションパラメータ"""
    _fields_ = [
        ("vgs", c_double),
        ("vds", c_double),
        ("vbs", c_double),
        ("temp", c_double),
    ]


# BSIM4_DeviceParamの定義 - C構造体と完全一致
# 整数パラメータ
BSIM4_INT_PARAMS = [
    'capmod', 'diomod', 'rdsmod', 'trnqsmod', 'acnqsmod', 'mobmod', 'rbodymod',
    'rgatemod', 'permod', 'geomod', 'fnoimod', 'tnoimod', 'igcmod', 'igbmod',
    'tempmod', 'paramchk', 'binunit'
]

# 浮動小数点パラメータ - C構造体と完全一致する順序
BSIM4_DOUBLE_PARAMS = [
    'toxe', 'toxp', 'toxm', 'toxref', 'dtox', 'epsrox', 'cdsc', 'cdscb', 'cdscd',
    'cit', 'nfactor', 'xj', 'vsat', 'at', 'a0', 'ags', 'a1', 'a2', 'keta',
    'nsub', 'ndep', 'nsd', 'phin', 'ngate', 'gamma1', 'gamma2', 'vbx', 'vbm',
    'xt', 'k1', 'kt1', 'kt1l', 'kt2', 'k2', 'k3', 'k3b', 'w0',
    'dvtp0', 'dvtp1', 'lpe0', 'lpeb', 'dvt0', 'dvt1', 'dvt2', 'dvt0w', 'dvt1w', 'dvt2w',
    'drout', 'dsub', 'vth0', 'vtho', 'ua', 'ua1', 'ub', 'ub1', 'uc', 'uc1',
    'ud', 'ud1', 'up', 'lp', 'u0', 'eu', 'ute', 'voff', 'minv', 'voffl',
    'tnom', 'cgso', 'cgdo', 'cgbo', 'xpart', 'delta', 'rsh', 'rdsw', 'rdswmin',
    'rsw', 'rdw', 'rdwmin', 'rswmin', 'prwg', 'prwb', 'prt', 'eta0', 'etab', 'pclm', 
    'pdiblc1', 'pdiblc2', 'pdiblcb', 'fprout', 'pdits',
    'pditsl', 'pditsd', 'pscbe1', 'pscbe2', 'pvag', 'jss', 'jsws', 'jswgs', 'pbs', 'njs',
    'xtis', 'mjs', 'pbsws', 'mjsws', 'pbswgs', 'mjswgs', 'cjs', 'cjsws', 'cjswgs',
    'jsd', 'jswd', 'jswgd', 'pbd', 'njd', 'xtid', 'mjd', 'pbswd', 'mjswd',
    'pbswgd', 'mjswgd', 'cjd', 'cjswd', 'cjswgd', 'vfbcv', 'vfb', 'tpb', 'tcj', 'tpbsw',
    'tcjsw', 'tpbswg', 'tcjswg', 'acde', 'moin', 'noff', 'voffcv', 'dmcg', 'dmci', 'dmdg',
    'dmcgt', 'xgw', 'xgl', 'rshg', 'ngcon', 'xrcrg1', 'xrcrg2', 'lambda', 'vtl', 'lc',
    'xn', 'vfbsdoff', 'tvfbsdoff', 'tvoff', 'lintnoi', 'lint', 'll', 'llc', 'lln', 'lw',
    'lwc', 'lwn', 'lwl', 'lwlc', 'lmin', 'lmax', 'wr', 'wint', 'dwg', 'dwb', 
    'wl', 'wlc', 'wln', 'ww', 'wwc',
    'wwn', 'wwl', 'wwlc', 'wmin', 'wmax', 'b0', 'b1', 'cgsl', 'cgdl', 'ckappas',
    'ckappad', 'cf', 'clc', 'cle', 'dwc', 'dlc', 'xw', 'xl', 'dlcig', 'dwj',
    'alpha0', 'alpha1', 'beta0', 'agidl', 'bgidl', 'cgidl', 'egidl', 'aigc', 'bigc', 'cigc',
    'aigsd', 'bigsd', 'cigsd', 'aigbacc', 'bigbacc', 'cigbacc', 'aigbinv', 'bigbinv', 'cigbinv',
    'nigc', 'nigbinv', 'nigbacc', 'ntox', 'eigbinv', 'pigcd', 'poxedge', 'ijthdfwd', 'ijthsfwd', 'ijthdrev',
    'ijthsrev', 'xjbvd', 'xjbvs', 'bvd', 'bvs', 'jtss', 'jtsd', 'jtssws', 'jtsswd', 'jtsswgs',
    'jtsswgd', 'njts', 'njtssw', 'njtsswg', 'xtss', 'xtsd', 'xtssws', 'xtsswd', 'xtsswgs', 'xtsswgd',
    'tnjts', 'tnjtssw', 'tnjtsswg', 'vtss', 'vtsd', 'vtssws', 'vtsswd', 'vtsswgs', 'vtsswgd',
    'gbmin', 'rbdb', 'rbpb', 'rbsb', 'rbps', 'rbpd', 'rbps0', 'rbpsl', 'rbpsw', 'rbpsnf',
    'rbpd0', 'rbpdl', 'rbpdw', 'rbpdnf', 'rbpbx0', 'rbpbxl', 'rbpbxw', 'rbpbxnf', 'rbpby0', 'rbpbyl',
    'rbpbyw', 'rbpbynf', 'rbsbx0', 'rbsby0', 'rbdbx0', 'rbdby0', 'rbsdbxl', 'rbsdbxw', 'rbsdbxnf', 'rbsdbyl',
    'rbsdbyw', 'rbsdbynf', 
    # L scaling
    'lcdsc', 'lcdscb', 'lcdscd', 'lcit', 'lnfactor', 'lxj', 'lvsat', 'lat', 'la0',
    'lags', 'la1', 'la2', 'lketa', 'lnsub', 'lndep', 'lnsd', 'lphin', 'lngate', 'lgamma1',
    'lgamma2', 'lvbx', 'lvbm', 'lxt', 'lk1', 'lkt1', 'lkt1l', 'lkt2', 'lk2', 'lk3',
    'lk3b', 'lw0', 'ldvtp0', 'ldvtp1', 'llpe0', 'llpeb', 'ldvt0', 'ldvt1', 'ldvt2', 'ldvt0w',
    'ldvt1w', 'ldvt2w', 'ldrout', 'ldsub', 'lvth0', 'lvtho', 'lua', 'lua1', 'lub', 'lub1',
    'luc', 'luc1', 'lud', 'lud1', 'lup', 'llp', 'lu0', 'lute', 'lvoff', 'lminv',
    'ldelta', 'lrdsw', 'lrsw', 'lrdw', 'lprwg', 'lprwb', 'lprt', 'leta0', 'letab', 'lpclm', 
    'lpdiblc1', 'lpdiblc2', 'lpdiblcb',
    'lfprout', 'lpdits', 'lpditsd', 'lpscbe1', 'lpscbe2', 'lpvag', 'lwr', 'ldwg', 'ldwb', 'lb0',
    'lb1', 'lcgsl', 'lcgdl', 'lckappas', 'lckappad', 'lcf', 'lclc', 'lcle', 'lalpha0', 'lalpha1',
    'lbeta0', 'lagidl', 'lbgidl', 'lcgidl', 'legidl', 'laigc', 'lbigc', 'lcigc', 'laigsd', 'lbigsd',
    'lcigsd', 'laigbacc', 'lbigbacc', 'lcigbacc', 'laigbinv', 'lbigbinv', 'lcigbinv', 'lnigc', 'lnigbinv', 'lnigbacc',
    'lntox', 'leigbinv', 'lpigcd', 'lpoxedge', 'lvfbcv', 'lvfb', 'lacde', 'lmoin', 'lnoff', 'lvoffcv',
    'lxrcrg1', 'lxrcrg2', 'llambda', 'lvtl', 'lxn', 'leu', 'lvfbsdoff', 'ltvfbsdoff', 'ltvoff',
    # W scaling
    'wcdsc', 'wcdscb', 'wcdscd', 'wcit', 'wnfactor', 'wxj', 'wvsat', 'wat', 'wa0', 'wags',
    'wa1', 'wa2', 'wketa', 'wnsub', 'wndep', 'wnsd', 'wphin', 'wngate', 'wgamma1', 'wgamma2',
    'wvbx', 'wvbm', 'wxt', 'wk1', 'wkt1', 'wkt1l', 'wkt2', 'wk2', 'wk3', 'wk3b',
    'ww0', 'wdvtp0', 'wdvtp1', 'wlpe0', 'wlpeb', 'wdvt0', 'wdvt1', 'wdvt2', 'wdvt0w', 'wdvt1w',
    'wdvt2w', 'wdrout', 'wdsub', 'wvth0', 'wvtho', 'wua', 'wua1', 'wub', 'wub1', 'wuc',
    'wuc1', 'wud', 'wud1', 'wup', 'wlp', 'wu0', 'wute', 'wvoff', 'wminv', 'wdelta',
    'wrdsw', 'wrsw', 'wrdw', 'wprwg', 'wprwb', 'wprt', 'weta0', 'wetab', 'wpclm', 'wpdiblc1',
    'wpdiblc2', 'wpdiblcb', 'wfprout', 'wpdits', 'wpditsd', 'wpscbe1', 'wpscbe2', 'wpvag', 'wwr', 'wdwg',
    'wdwb', 'wb0', 'wb1', 'wcgsl', 'wcgdl', 'wckappas', 'wckappad', 'wcf', 'wclc', 'wcle',
    'walpha0', 'walpha1', 'wbeta0', 'wagidl', 'wbgidl', 'wcgidl', 'wegidl', 'waigc', 'wbigc', 'wcigc',
    'waigsd', 'wbigsd', 'wcigsd', 'waigbacc', 'wbigbacc', 'wcigbacc', 'waigbinv', 'wbigbinv', 'wcigbinv', 'wnigc',
    'wnigbinv', 'wnigbacc', 'wntox', 'weigbinv', 'wpigcd', 'wpoxedge', 'wvfbcv', 'wvfb', 'wacde', 'wmoin',
    'wnoff', 'wvoffcv', 'wxrcrg1', 'wxrcrg2', 'wlambda', 'wvtl', 'wxn', 'weu', 'wvfbsdoff', 'wtvfbsdoff',
    'wtvoff', 
    # P scaling
    'pcdsc', 'pcdscb', 'pcdscd', 'pcit', 'pnfactor', 'pxj', 'pvsat', 'pat', 'pa0',
    'pags', 'pa1', 'pa2', 'pketa', 'pnsub', 'pndep', 'pnsd', 'pphin', 'pngate', 'pgamma1',
    'pgamma2', 'pvbx', 'pvbm', 'pxt', 'pk1', 'pkt1', 'pkt1l', 'pkt2', 'pk2', 'pk3',
    'pk3b', 'pw0', 'pdvtp0', 'pdvtp1', 'plpe0', 'plpeb', 'pdvt0', 'pdvt1', 'pdvt2', 'pdvt0w',
    'pdvt1w', 'pdvt2w', 'pdrout', 'pdsub', 'pvth0', 'pvtho', 'pua', 'pua1', 'pub', 'pub1',
    'puc', 'puc1', 'pud', 'pud1', 'pup', 'plp', 'pu0', 'pute', 'pvoff', 'pminv',
    'pdelta', 'prdsw', 'prsw', 'prdw', 'pprwg', 'pprwb', 'pprt', 'peta0', 'petab', 'ppclm',
    'ppdiblc1', 'ppdiblc2', 'ppdiblcb', 'pfprout', 'ppdits', 'ppditsd', 'ppscbe1', 'ppscbe2', 'ppvag', 'pwr',
    'pdwg', 'pdwb', 'pb0', 'pb1', 'pcgsl', 'pcgdl', 'pckappas', 'pckappad', 'pcf', 'pclc',
    'pcle', 'palpha0', 'palpha1', 'pbeta0', 'pagidl', 'pbgidl', 'pcgidl', 'pegidl', 'paigc', 'pbigc',
    'pcigc', 'paigsd', 'pbigsd', 'pcigsd', 'paigbacc', 'pbigbacc', 'pcigbacc', 'paigbinv', 'pbigbinv', 'pcigbinv',
    'pnigc', 'pnigbinv', 'pnigbacc', 'pntox', 'peigbinv', 'ppigcd', 'ppoxedge', 'pvfbcv', 'pvfb', 'pacde',
    'pmoin', 'pnoff', 'pvoffcv', 'pxrcrg1', 'pxrcrg2', 'plambda', 'pvtl', 'pxn', 'peu', 'pvfbsdoff',
    'ptvfbsdoff', 'ptvoff', 
    # Additional
    'saref', 'sbref', 'wlod', 'ku0', 'kvsat', 'kvth0', 'tku0', 'llodku0',
    'wlodku0', 'llodvth', 'wlodvth', 'lku0', 'wku0', 'pku0', 'lkvth0', 'wkvth0', 'pkvth0', 'stk2',
    'lodk2', 'steta0', 'lodeta0', 'web', 'wec', 'kvth0we', 'k2we', 'ku0we', 'scref', 'wpemod',
    'lkvth0we', 'lk2we', 'lku0we', 'wkvth0we', 'wk2we', 'wku0we', 'pkvth0we', 'pk2we', 'pku0we', 'noia',
    'noib', 'noic', 'tnoia', 'tnoib', 'rnoia', 'rnoib', 'ntnoi', 'em', 'ef', 'af',
    'kf', 'type'
]


def create_bsim4_device_param_struct():
    """BSIM4_DeviceParam構造体を動的に生成 - C構造体と完全一致"""
    fields = []
    # 整数パラメータ
    for name in BSIM4_INT_PARAMS:
        fields.append((name, c_int))
    # version pointer
    fields.append(('version', c_char_p))
    # 浮動小数点パラメータ
    for name in BSIM4_DOUBLE_PARAMS:
        fields.append((name, c_double))
    # nmos, pmos flags
    fields.append(('nmos', c_int))
    fields.append(('pmos', c_int))
    
    class BSIM4_DeviceParam(Structure):
        _fields_ = fields
    
    return BSIM4_DeviceParam


BSIM4_DeviceParam = create_bsim4_device_param_struct()


class BSIM4_Output(Structure):
    """出力構造体 - Cコードと完全一致する必要あり"""
    _fields_ = [
        # currents & small-signal
        ("ids", c_double), ("id", c_double), ("gm", c_double), ("gds", c_double), ("gmb", c_double),
        # Gate row: Qg w.r.t Vgb/Vdb/Vsb/Vbb
        ("cggb", c_double), ("cgdb", c_double), ("cgsb", c_double), ("cgbb", c_double),
        # Drain row: Qd w.r.t Vgb/Vdb/Vsb/Vbb
        ("cdgb", c_double), ("cddb", c_double), ("cdsb", c_double), ("cdbb", c_double),
        # Source row: Qs w.r.t Vgb/Vdb/Vsb/Vbb
        ("csgb", c_double), ("csdb", c_double), ("cssb", c_double), ("csbb", c_double),
        # Bulk row: Qb w.r.t Vgb/Vdb/Vsb/Vbb
        ("cbgb", c_double), ("cbdb", c_double), ("cbsb", c_double), ("cbbb", c_double),
        # Junction caps (diode caps)
        ("capbd", c_double), ("capbs", c_double),
        # NQS charge derivatives (optional)
        ("cqgb", c_double), ("cqdb", c_double), ("cqsb", c_double), ("cqbb", c_double),
        # raw charges (optional)
        ("qgate", c_double), ("qbulk", c_double), ("qdrn", c_double), ("qsrc", c_double),
        # legacy aliases (for backward compatibility)
        ("cgg", c_double), ("cgd", c_double), ("cgs", c_double), ("cgb", c_double),
        ("cdd", c_double), ("cds", c_double), ("cdb", c_double), ("css", c_double), ("csb", c_double),
        # DC diode conductances
        ("gbd", c_double), ("gbs", c_double),
        # ダイオードの DC 電流（右辺に必要）
        ("cbd", c_double), ("cbs", c_double),
        # 外端子直列コンダクタンス
        ("Gdpr", c_double), ("Gspr", c_double),
        # 外部ゲート ↔ GatePrime のブリッジ導通
        ("Ggpr", c_double),
        # Gate leakage currents
        ("Igs", c_double), ("Igd", c_double), ("Igb", c_double), ("Igcs", c_double), ("Igcd", c_double),
        # Gate leakage small-signal partials (subset)
        ("gIgsg", c_double), ("gIgss", c_double),
        ("gIgdg", c_double), ("gIgdd", c_double),
        ("gIgbg", c_double), ("gIgbd", c_double), ("gIgbs", c_double), ("gIgbb", c_double),
        ("gIgcdg", c_double), ("gIgcdd", c_double), ("gIgcds", c_double), ("gIgcdb", c_double),
        ("gIgcsg", c_double), ("gIgcsd", c_double), ("gIgcss", c_double), ("gIgcsb", c_double),
        # State charges
        ("qbd", c_double), ("qbs", c_double),
        ("qg_state", c_double), ("qd_state", c_double), ("qs_state", c_double), ("qb_state", c_double),
    ]


# ===== BSIM4デバイスクラス =====
class BSIM4Device:
    """BSIM4デバイス（Juliaの BSIM4InstParam 相当）"""
    
    _lib = None
    _lib_path = None
    
    @classmethod
    def _load_library(cls):
        """ライブラリをロード"""
        if cls._lib is None:
            cls._lib_path = find_bsim4_library()
            cls._lib = CDLL(cls._lib_path)
            
            # 関数プロトタイプ設定
            cls._lib.bsim4_create_context.argtypes = [c_char_p]
            cls._lib.bsim4_create_context.restype = c_void_p
            
            cls._lib.bsim4_destroy_context.argtypes = [c_void_p]
            cls._lib.bsim4_destroy_context.restype = None
            
            cls._lib.bsim4_direct_init_ctx.argtypes = [
                c_void_p, POINTER(BSIM4_SimParam), 
                POINTER(BSIM4_InstParam), POINTER(BSIM4_DeviceParam)
            ]
            cls._lib.bsim4_direct_init_ctx.restype = c_int
            
            # 5引数: ctx, simParam, instParam, devParam, output
            cls._lib.bsim4_direct_calculate_ctx.argtypes = [
                c_void_p, POINTER(BSIM4_SimParam), 
                POINTER(BSIM4_InstParam), POINTER(BSIM4_DeviceParam),
                POINTER(BSIM4_Output)
            ]
            cls._lib.bsim4_direct_calculate_ctx.restype = c_int
            
            print(f"Loaded BSIM4 library: {cls._lib_path}")
        return cls._lib
    
    def __init__(self, name: str, device_params: Dict[str, Any],
                 w: float = 1e-6, l: float = 180e-9, nf: float = 1.0,
                 multi: float = 1.0, sa: float = 0.0, sb: float = 0.0,
                 sd: float = 0.0, ad: float = 0.0, as_: float = 0.0,
                 pd: float = 0.0, ps: float = 0.0, nrd: float = 0.0, nrs: float = 0.0):
        """
        BSIM4デバイスを初期化
        
        Args:
            name: デバイス名
            device_params: モデルパラメータ辞書
            w, l, nf, ...: インスタンスパラメータ
        """
        self._lib = self._load_library()
        self.name = name
        self.ctx = None
        
        # コンテキスト作成
        self.ctx = self._lib.bsim4_create_context(name.encode('utf-8'))
        if not self.ctx:
            raise RuntimeError(f"Failed to create BSIM4 context for '{name}'")
        
        # インスタンスパラメータ
        self._inst_param = BSIM4_InstParam()
        
        # 全フィールドをまずNaNで初期化（不明なフィールドはBSIM4側でデフォルト使用）
        for field_name, _ in BSIM4_InstParam._fields_:
            setattr(self._inst_param, field_name, float('nan'))
        
        # 既知のパラメータを設定
        self._inst_param.w = w
        self._inst_param.l = l
        self._inst_param.nf = nf
        self._inst_param.multi = multi
        self._inst_param.sa = sa
        self._inst_param.sb = sb
        self._inst_param.sd = sd
        self._inst_param.ad = ad
        self._inst_param.as_ = as_
        self._inst_param.pd = pd
        self._inst_param.ps = ps
        self._inst_param.nrd = nrd
        self._inst_param.nrs = nrs
        
        # デバイスパラメータ
        self._dev_param = BSIM4_DeviceParam()
        self._setup_device_params(device_params)
        
        # シミュレーションパラメータ（初期値）
        self._sim_param = BSIM4_SimParam()
        self._sim_param.vgs = 0.0
        self._sim_param.vds = 0.0
        self._sim_param.vbs = 0.0
        self._sim_param.temp = 27.0
        
        # 初期化
        ret = self._lib.bsim4_direct_init_ctx(
            self.ctx,
            byref(self._sim_param),
            byref(self._inst_param),
            byref(self._dev_param)
        )
        if ret != 0:
            self._lib.bsim4_destroy_context(self.ctx)
            self.ctx = None
            raise RuntimeError(f"BSIM4 initialization failed with code: {ret}")
        
        self._initialized = True
    
    @property
    def w(self) -> float:
        """デバイス幅 [m]"""
        return self._inst_param.w
    
    @property
    def l(self) -> float:
        """デバイス長 [m]"""
        return self._inst_param.l
    
    @property
    def nf(self) -> float:
        """フィンガー数"""
        return self._inst_param.nf
    
    def _setup_device_params(self, params: Dict[str, Any]):
        """デバイスパラメータを構造体に設定"""
        # デフォルト: すべてNaN
        for name in BSIM4_INT_PARAMS:
            setattr(self._dev_param, name, 0)
        for name in BSIM4_DOUBLE_PARAMS:
            setattr(self._dev_param, name, float('nan'))
        
        # NMOS/PMOS判定
        mos_type = params.get('type', 'nmos')
        if isinstance(mos_type, str):
            mos_type_str = mos_type.lower()
            if mos_type_str in ('p', 'pmos', '-1'):
                self._dev_param.nmos = 0
                self._dev_param.pmos = 1
                # 'type'フィールドに-1.0を設定
                setattr(self._dev_param, 'type', -1.0)
            else:
                self._dev_param.nmos = 1
                self._dev_param.pmos = 0
                setattr(self._dev_param, 'type', 1.0)
        elif isinstance(mos_type, (int, float)):
            if mos_type < 0:
                self._dev_param.nmos = 0
                self._dev_param.pmos = 1
                setattr(self._dev_param, 'type', -1.0)
            else:
                self._dev_param.nmos = 1
                self._dev_param.pmos = 0
                setattr(self._dev_param, 'type', 1.0)
        
        # パラメータ設定
        for key, val in params.items():
            key_lower = key.lower()
            if key_lower == 'type':
                continue  # 既に処理済み
            
            # 整数パラメータ
            if key_lower in BSIM4_INT_PARAMS:
                try:
                    setattr(self._dev_param, key_lower, int(val))
                except:
                    pass
                continue
            
            # 浮動小数点パラメータ
            if key_lower in BSIM4_DOUBLE_PARAMS:
                try:
                    setattr(self._dev_param, key_lower, float(val))
                except:
                    pass
    
    def evaluate(self, vgs: float, vds: float, vbs: float = 0.0, temp: float = 27.0) -> Dict[str, float]:
        """
        指定バイアスでの動作点を計算
        
        Returns:
            ids, gm, gds, gmb, 容量値などを含む辞書
        """
        if not self._initialized:
            raise RuntimeError("Device not initialized")
        
        self._sim_param.vgs = vgs
        self._sim_param.vds = vds
        self._sim_param.vbs = vbs
        self._sim_param.temp = temp
        
        output = BSIM4_Output()
        ret = self._lib.bsim4_direct_calculate_ctx(
            self.ctx,
            byref(self._sim_param),
            byref(self._inst_param),
            byref(self._dev_param),
            byref(output)
        )
        if ret != 0:
            raise RuntimeError(f"BSIM4 calculation failed with code: {ret}")
        
        return {
            'ids': output.ids,
            'id': output.id,  # DC解析用（Julia側と同じ）
            'gm': output.gm,
            'gds': output.gds,
            'gmb': output.gmb,
            'cggb': output.cggb,
            'cgdb': output.cgdb,
            'cgsb': output.cgsb,
            'cdgb': output.cdgb,
            'cddb': output.cddb,
            'cdsb': output.cdsb,
            'cbgb': output.cbgb,
            'cbdb': output.cbdb,
            'cbsb': output.cbsb,
            'capbd': output.capbd,
            'capbs': output.capbs,
            'qgate': output.qgate,
            'qdrn': output.qdrn,
            'qsrc': output.qsrc,
            'qbulk': output.qbulk,
            'gbd': output.gbd,
            'gbs': output.gbs,
            'cbd': output.cbd,  # DC diode currents
            'cbs': output.cbs,
            'Gdpr': output.Gdpr,
            'Gspr': output.Gspr,
            'Ggpr': output.Ggpr,
        }
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'ctx') and self.ctx:
            self._lib.bsim4_destroy_context(self.ctx)
            self.ctx = None


# ===== テスト =====
def test_bsim4_device():
    """BSIM4デバイスの基本テスト"""
    print("=" * 60)
    print("BSIM4 Python Wrapper Test")
    print("=" * 60)
    
    # 最小限のモデルパラメータ
    model_params = {
        'type': 'nmos',
        'toxe': 5.75e-9,
        'toxp': 4.65e-9,
        'toxm': 5.75e-9,
        'vth0': 0.576,
        'k1': 0.5,
        'k2': 0.0,
        'u0': 0.04477,
        'vsat': 1.5e5,
        'rdsw': 200.0,
        'ndep': 1.7e17,
        'nsub': 1e16,
        'ngate': 1e20,
    }
    
    # デバイス作成
    print("\n[Test 1] Creating NMOS device...")
    try:
        nmos = BSIM4Device(
            name="NMOS_test",
            device_params=model_params,
            w=400e-9,
            l=550e-9,
            nf=1.0
        )
        print("  ✓ Device created successfully")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # 動作点計算
    print("\n[Test 2] Calculating operating point...")
    try:
        result = nmos.evaluate(vgs=1.0, vds=1.0, vbs=0.0)
        print(f"  Vgs=1.0V, Vds=1.0V:")
        print(f"    Ids = {result['ids']*1e6:.3f} uA")
        print(f"    gm  = {result['gm']*1e6:.3f} uS")
        print(f"    gds = {result['gds']*1e6:.3f} uS")
        print(f"    Cgg = {result['cggb']*1e15:.3f} fF")
        
        # 電流が正の値であることを確認
        if result['ids'] > 0:
            print("  ✓ Positive drain current (expected for NMOS)")
        else:
            print("  ⚠ Warning: Non-positive drain current")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Vgsスイープ
    print("\n[Test 3] Vgs sweep...")
    try:
        vgs_values = np.linspace(0, 1.5, 16)
        ids_values = []
        for vgs in vgs_values:
            r = nmos.evaluate(vgs=vgs, vds=1.0)
            ids_values.append(r['ids'])
        
        print(f"    Max Ids = {max(ids_values)*1e6:.3f} uA at Vgs={vgs_values[np.argmax(ids_values)]:.2f}V")
        print("  ✓ Sweep completed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


def test_source_follower_circuit():
    """ソースフォロワ回路のテスト (Juliaのtest_full_dc_analysis.jlを参考)"""
    print("\n" + "=" * 60)
    print("Source Follower Circuit Test (BSIM4)")
    print("=" * 60)
    
    # ソースフォロワの構成:
    # M0: ゲート=VIN, ドレイン=VDD, ソース=VOUT, バルク=VOUT
    # R0: VOUT - I4_PLUS (2.2kΩ)
    # I4: 電流源 10uA (I4_PLUS -> GND)
    # 
    # 回路方程式:
    # - MOSFETのIds = 電流源の10uA + 抵抗を流れる電流
    # - VOUT/R0 = Ids - Ibias
    
    # mn25od33_svtに近いパラメータ (NMOS)
    model_params = {
        'type': 'nmos',
        'toxe': 5.75e-9,
        'toxp': 4.65e-9,
        'toxm': 5.75e-9,
        'vth0': 0.45,     # 閾値電圧
        'k1': 0.5,
        'k2': 0.0,
        'u0': 0.04,       # 移動度
        'vsat': 1.5e5,
        'rdsw': 200.0,
        'ndep': 1.7e17,
        'nsd': 1e20,
        'ngate': 1e20,
        'voff': -0.1,
    }
    
    # Juliaと同じインスタンスパラメータ
    w = 360e-9
    l = 495e-9
    nf = 1.0
    
    print(f"\n[Circuit Parameters]")
    print(f"  M0: l={l*1e9:.0f}nm, w={w*1e9:.0f}nm, nf={nf}")
    print(f"  Vth0 = {model_params['vth0']} V")
    print(f"  VDD = 3.0 V")
    print(f"  R0 = 2.2 kΩ")
    print(f"  Ibias = 10 uA")
    
    # デバイス作成
    try:
        nmos = BSIM4Device(
            name="M0_SF",
            device_params=model_params,
            w=w,
            l=l,
            nf=nf
        )
        print("  ✓ NMOS device created")
    except Exception as e:
        print(f"  ✗ Failed to create device: {e}")
        return False
    
    # 単純なI-V特性の確認
    print("\n[I-V Characteristics Check]")
    test_biases = [(0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0)]
    for vgs, vds in test_biases:
        result = nmos.evaluate(vgs=vgs, vds=vds, vbs=0.0)
        print(f"  Vgs={vgs:.1f}V, Vds={vds:.1f}V -> Ids={result['ids']*1e6:.3f}uA, gm={result['gm']*1e6:.3f}uS")
    
    # VIN スイープで動作点を確認
    print("\n[VIN Sweep - Source Follower Operation]")
    VDD = 3.0
    R_load = 2.2e3  # 2.2kΩ
    Ibias = 10e-6   # 10uA 電流源
    
    vin_values = np.linspace(1.0, 3.0, 9)
    vout_values = []
    ids_values = []
    
    for vin in vin_values:
        # 反復法でVOUTを求める
        # KCL: Ids = (VOUT - 0)/R_load + Ibias  (実際はI4_PLUSノードで)
        # 簡略化: Ids ≈ Ibias と仮定 (抵抗で落ちる電圧が小さい場合)
        
        # 初期推定: ソースフォロワなので VOUT ≈ VIN - Vth
        vout = max(0.1, vin - model_params['vth0'] - 0.1)
        
        for iteration in range(100):
            # バイアス電圧計算
            # Vgs = Vgate - Vsource = VIN - VOUT
            vgs = vin - vout
            # Vds = Vdrain - Vsource = VDD - VOUT  
            vds = VDD - vout
            # Vbs = Vbulk - Vsource = VOUT - VOUT = 0 (バルクはVOUTに接続)
            vbs = 0.0
            
            # バイアス条件チェック
            if vgs < 0:
                vgs = 0.01
            if vds < 0:
                vds = 0.01
            
            result = nmos.evaluate(vgs=vgs, vds=vds, vbs=vbs)
            ids = result['ids']
            gm = result['gm']
            
            # 回路方程式: Ids = Ibias (電流源で設定)
            # 実際の回路では抵抗も関与するが、簡略化
            # VOUTを調整してIds = Ibias となるようにする
            
            i_error = ids - Ibias
            
            if abs(i_error) < 1e-11:  # 収束判定
                break
            
            # Newton-Raphson的な補正
            # dIds/dVout ≈ -gm (Vgsが減るとIdsも減る)
            if gm > 1e-12:
                dvout = i_error / gm
                vout = vout + dvout * 0.5  # ダンピング
            else:
                # gmが小さい場合は小さく動かす
                vout = vout + 0.01 if i_error > 0 else vout - 0.01
            
            # 境界条件
            vout = max(0.01, min(VDD - 0.01, vout))
        
        vout_values.append(vout)
        ids_values.append(ids)
        print(f"  VIN={vin:.2f}V -> VOUT={vout:.3f}V, Ids={ids*1e6:.3f}uA (iter={iteration+1})")
    
    # 結果の確認
    print("\n[Analysis Results]")
    print(f"  VIN range: {vin_values[0]:.1f} - {vin_values[-1]:.1f} V")
    print(f"  VOUT range: {min(vout_values):.3f} - {max(vout_values):.3f} V")
    
    # ゲイン計算 (差分で近似)
    gains = []
    for i in range(1, len(vin_values)):
        dvin = vin_values[i] - vin_values[i-1]
        dvout = vout_values[i] - vout_values[i-1]
        if abs(dvin) > 1e-9:
            gains.append(dvout / dvin)
    
    avg_gain = np.mean(gains) if gains else 0
    print(f"  Average Gain (dVout/dVin): {avg_gain:.4f}")
    print(f"  Expected: ~0.8-0.95 (source follower)")
    
    if 0.5 < avg_gain < 1.1:
        print("  ✓ Gain is in expected range for source follower")
        success = True
    else:
        print(f"  ⚠ Gain={avg_gain:.3f} may be outside expected range")
        success = False
    
    print("\n" + "=" * 60)
    print("Source Follower Test Completed")
    print("=" * 60)
    return success


if __name__ == "__main__":
    test_bsim4_device()
    test_source_follower_circuit()
