import json
import csv
import re
import math
import sys
from typing import Dict, List, Tuple, Optional
import argparse
import os


# === グローバル変数：失敗情報を記録 ===
FAILED_PARAMS: List[Dict[str, any]] = []
CALCULATION_TRACE: Dict[str, List[Dict]] = {}  # 計算トレース記録

# ★★★ 追加 ★★★
# === グローバル変数：デフォルト補間の理由を記録 ===
DEFAULT_SUPPLEMENT_REASONS: Dict[str, Dict[str, any]] = {}

# === トランジスタパラメータ設定 ===
TRANSISTOR_PARAMS = {
    'l': 550e-9,           # チャネル長 (m)
    'w': 400e-9,           # チャネル幅 (m)
    'multi': 1,            # マルチプライヤ
    'nf': 1,               # フィンガー数
    'sd': 190e-9,          # ソース/ドレイン間隔 (m)
    'ad': 5.98395e-14,     # ドレイン面積 (m²)
    'as': 5.98395e-14,     # ソース面積 (m²)
    'pd': 1.113333e-6,     # ドレイン周囲長 (m)
    'ps': 1.113333e-6,     # ソース周囲長 (m)
    'nrd': 0.2567568,      # ドレイン抵抗の正規化
    'nrs': 0.2567568,      # ソース抵抗の正規化
    'sa': 131.0e-9,        # 活性領域からゲートまでの距離 (m)
    'sb': 131.0e-9,        # 活性領域からゲートまでの距離 (m)
    'temp':27,
}

PROCESS_CORNER = 'tt'      # プロセスコーナー: tt, ss, ff, sf, fs, ssg, ffg, sfg, fsg
BIN_ID = '32'              # ビンID
# ===============================

sys.tracebacklimit = 0  # トレースバックを非表示


import decimal
from decimal import Decimal, ROUND_HALF_UP

# 関数の先頭（importの後）に追加
def format_significant_digits(value, sig_digits=8):
    """
    有効数字を指定桁数で繰り上げ
    
    Args:
        value: 数値
        sig_digits: 有効数字の桁数（デフォルト8桁）
    
    Returns:
        フォーマット済みの文字列
    """
    if value == 0:
        return "0"
    
    # Decimalに変換（高精度計算）
    d = Decimal(str(value))
    
    # 有効数字で丸める（繰り上げ）
    context = decimal.Context(prec=sig_digits, rounding=ROUND_HALF_UP)
    rounded = context.create_decimal(d)
    
    # 指数表記にするかどうかを判定
    abs_val = abs(float(rounded))
    
    if abs_val >= 1e4 or (abs_val < 1e-3 and abs_val != 0):
        # 指数表記
        formatted = f"{rounded:.{sig_digits-1}e}"
    else:
        # 通常表記（有効数字を維持）
        formatted = str(rounded)
        
        # 末尾のゼロを削除（ただし小数点以下は1桁残す）
        if '.' in formatted:
            formatted = formatted.rstrip('0')
            if formatted.endswith('.'):
                formatted += '0'
    
    return formatted

def parse_unit_value(value_str):
    """単位付き文字列を数値に変換"""
    if not isinstance(value_str, str):
        return value_str
    
    value_str = value_str.strip()
    
    # 単位の定義
    units = {
        'T': 1e12, 't': 1e12,
        'G': 1e9, 'g': 1e9, 'Gig': 1e9,
        'M': 1e6, 'Meg': 1e6,
        'K': 1e3, 'k': 1e3,
        'm': 1e-3,
        'u': 1e-6, 'μ': 1e-6,
        'n': 1e-9,
        'p': 1e-12,
        'f': 1e-15,
        'a': 1e-18,
    }
    
    # 数値+単位のパターンをマッチ
    pattern = r'^([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([a-zA-Zμ]+)?$'
    match = re.match(pattern, value_str)
    
    if match:
        number = float(match.group(1))
        unit = match.group(2)
        
        if unit and unit in units:
            return number * units[unit]
        else:
            return number
    
    return value_str

def get_process_corner_settings(corner='tt'):
    """プロセスコーナーの設定を返す"""
    # 全コーナーを0に初期化
    corners = {
        'od33svtn_tt_set': 0,
        'od33svtn_ss_set': 0,
        'od33svtn_ff_set': 0,
        'od33svtn_sf_set': 0,
        'od33svtn_fs_set': 0,
        'od33svtn_ssg_set': 0,
        'od33svtn_ffg_set': 0,
        'od33svtn_sfg_set': 0,
        'od33svtn_fsg_set': 0,
    }
    
    # 指定されたコーナーを1に設定
    corner_map = {
        'tt': 'od33svtn_tt_set',
        'ss': 'od33svtn_ss_set',
        'ff': 'od33svtn_ff_set',
        'sf': 'od33svtn_sf_set',
        'fs': 'od33svtn_fs_set',
        'ssg': 'od33svtn_ssg_set',
        'ffg': 'od33svtn_ffg_set',
        'sfg': 'od33svtn_sfg_set',
        'fsg': 'od33svtn_fsg_set',
    }
    
    corner_lower = corner.lower()
    if corner_lower in corner_map:
        corners[corner_map[corner_lower]] = 1
    else:
        print(f"警告: 未知のコーナー '{corner}'。TTコーナーを使用します。", file=sys.stderr)
        corners['od33svtn_tt_set'] = 1
    
    return corners

def get_section_parameters(result, section_name='core', process_corner='tt'):
    """セクションレベルのパラメータを取得"""
    section_params = {}
    
    # まずプロセスコーナー設定を追加（最優先）
    corner_settings = get_process_corner_settings(process_corner)
    section_params.update(corner_settings)
    
    if section_name in result:
        if 'parameters' in result[section_name]:
            # コーナー設定で上書きしないように更新
            for key, value in result[section_name]['parameters'].items():
                if key not in section_params:
                    # 単位付き値を変換
                    section_params[key] = parse_unit_value(value)
        
        # inline subcktのパラメータも取得
        if 'inline_subckts' in result[section_name]:
            for subckt_name, subckt_data in result[section_name]['inline_subckts'].items():
                if 'parameters' in subckt_data:
                    for key, value in subckt_data['parameters'].items():
                        if key not in section_params:
                            # 単位付き値を変換
                            section_params[key] = parse_unit_value(value)
    
    return section_params

def print_transistor_info(transistor_params):
    """トランジスタパラメータを表示"""
    print(f"\n{'='*70}", file=sys.stderr)
    print("トランジスタパラメータ", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"  L (チャネル長):      {transistor_params['l']*1e9:.1f} nm", file=sys.stderr)
    print(f"  W (チャネル幅):      {transistor_params['w']*1e9:.1f} nm", file=sys.stderr)
    print(f"  multi:               {transistor_params['multi']}", file=sys.stderr)
    print(f"  nf (フィンガー数):   {transistor_params['nf']}", file=sys.stderr)
    print(f"  sd:                  {transistor_params['sd']*1e9:.1f} nm", file=sys.stderr)
    print(f"  ad (ドレイン面積):   {transistor_params['ad']:.6e} m²", file=sys.stderr)
    print(f"  as (ソース面積):     {transistor_params['as']:.6e} m²", file=sys.stderr)
    print(f"  pd (ドレイン周長):   {transistor_params['pd']*1e6:.6f} um", file=sys.stderr)
    print(f"  ps (ソース周長):     {transistor_params['ps']*1e6:.6f} um", file=sys.stderr)
    print(f"  nrd:                 {transistor_params['nrd']:.6f}", file=sys.stderr)
    print(f"  nrs:                 {transistor_params['nrs']:.6f}", file=sys.stderr)
    print(f"  sa:                  {transistor_params['sa']*1e9:.1f} nm", file=sys.stderr)
    print(f"  sb:                  {transistor_params['sb']*1e9:.1f} nm", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

# Compile regex for unit conversion once
_UNIT_PATTERN = re.compile(r'(\d+\.?\d*)([TGMkmunpf])(?=[\s\*\+\-/\)]|$)')
_UNIT_MAP = {
    'T': 'e12', 'G': 'e9', 'M': 'e6', 'k': 'e3',
    'm': 'e-3', 'u': 'e-6', 'n': 'e-9', 'p': 'e-12', 'f': 'e-15',
}

def convert_units_in_formula(formula):
    """数式内の単位表記を数値に変換"""
    if not isinstance(formula, str):
        return formula
    
    return _UNIT_PATTERN.sub(lambda m: f"{m.group(1)}{_UNIT_MAP[m.group(2)]}", formula)

def safe_replace_variable(formula, var_name, value):
    """
    変数名を安全に置換（単語境界を考慮）
    
    Args:
        formula: 数式文字列
        var_name: 置換する変数名
        value: 置換後の値
    
    Returns:
        置換後の数式
    """
    # 変数名の前後に英数字・アンダースコアがないことを確認
    pattern = r'(?<![a-zA-Z0-9_])' + re.escape(var_name) + r'(?![a-zA-Z0-9_])'
    
    # 置換
    new_formula = re.sub(pattern, f'({value})', formula)
    
    return new_formula



def convert_ternary(formula):
    # Quick check
    if '?' not in formula:
        return formula

    # パターン: condition ? true_expr : false_expr
    # re.DOTALL: . matches newline
    pattern = r'(.+?)\?(.+?):(.+)'
    match = re.search(pattern, formula, re.DOTALL)
    if match:
        cond, true_expr, false_expr = match.groups()
        return f"({true_expr.strip()} if {cond.strip()} else {false_expr.strip()})"
    return formula


class LazyParamContext(dict):
    """
    eval()用の遅延評価コンテキスト
    変数が要求されたタイミングで再帰的に計算を行う
    """
    def __init__(self, bin_params, call_context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bin_params = bin_params
        self.call_context = call_context  # (depth, max_depth, update_params)
        
    def __missing__(self, key):
        # パラメータ辞書に存在するか確認
        if key in self.bin_params:
            val = self.bin_params[key]
            
            # すでに計算済みの数値ならそのまま返す
            if isinstance(val, (int, float)):
                return val
            
            # 文字列（数式）なら再帰的に計算
            if isinstance(val, str):
                depth, max_depth, update_params = self.call_context
                
                # 再帰呼び出し（新しい数式として depth+1 ）
                result = calculate_formula(
                    self.bin_params, val, depth + 1, max_depth, update_params,
                    param_name_for_tracking=key
                )
                
                if result is None:
                    # 計算失敗時は NameError として伝播（eval が停止する）
                    raise NameError(f"Dependency calculation failed for '{key}'")
                
                # update_params=True なら辞書を更新（メモ化）
                if update_params:
                    self.bin_params[key] = result
                
                return result
        
        # 存在しない変数は NameError
        raise NameError(f"name '{key}' is not defined")


# Math environment optimized
MATH_ENV = {
    "__builtins__": {},
    "sqrt": math.sqrt,
    "pow": math.pow,
    "abs": abs,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "int": int,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
}

def calculate_formula(bin_params, formula, depth=0, max_depth=1000, update_params=True, 
                     param_name_for_tracking=None):
    """
    再帰的に変数を置換して計算（失敗追跡機能付き）
    最適化: 文字列置換を行わず、LazyParamContextを用いて変数を解決時のみ計算する
    
    Args:
        bin_params: パラメータ辞書
        formula: 計算する数式
        depth: 再帰の深さ
        max_depth: 最大再帰深度
        update_params: Trueの場合、計算結果をbin_paramsに保存
        param_name_for_tracking: 追跡用のパラメータ名
    
    Returns:
        計算結果（float）、または失敗時はNone
    """
    if depth > max_depth:
        error_info = {
            'param': param_name_for_tracking or 'unknown',
            'reason': 'max_recursion_depth',
            'detail': f'再帰の深さが上限({max_depth})に達しました',
            'formula': str(formula)[:200]
        }
        FAILED_PARAMS.append(error_info)
        return None
    
    # 最初の呼び出し時のみ正規化・単位変換
    if depth == 0 or isinstance(formula, str): # formulaが文字列でない(数値)場合のガード
         if isinstance(formula, str):
            formula = convert_ternary(formula)
            formula = convert_units_in_formula(formula)
         elif isinstance(formula, (int, float)):
             return formula

    # 遅延評価コンテキストの作成
    # eval の locals として渡すことで、変数が参照されたときだけ計算が走る
    lazy_locals = LazyParamContext(bin_params, (depth, max_depth, update_params), MATH_ENV)
    
    try:
        # 計算実行
        # formula が空文字列などの場合のエラーハンドリング
        if not formula.strip():
            return 0.0

        result = eval(formula, {}, lazy_locals)
        return result
        
    except NameError as e:
        error_info = {
            'param': param_name_for_tracking or 'unknown',
            'reason': 'variable_not_found',
            'detail': str(e),
            'formula': str(formula)[:200]
        }
        FAILED_PARAMS.append(error_info)
        return None
        
    except SyntaxError as e:
        error_info = {
            'param': param_name_for_tracking or 'unknown',
            'reason': 'syntax_error',
            'detail': str(e),
            'formula': str(formula)[:200]
        }
        FAILED_PARAMS.append(error_info)
        return None
    
    except ZeroDivisionError:
        error_info = {
            'param': param_name_for_tracking or 'unknown',
            'reason': 'zero_division',
            'detail': 'ゼロ除算エラー',
            'formula': str(formula)[:200]
        }
        FAILED_PARAMS.append(error_info)
        return None
    
    except Exception as e:
        error_info = {
            'param': param_name_for_tracking or 'unknown',
            'reason': 'unexpected_error',
            'detail': f"{type(e).__name__}: {str(e)}",
            'formula': str(formula)[:200]
        }
        FAILED_PARAMS.append(error_info)
        return None


def export_failed_params_report(failed_list: List[Dict], output_file='failed_params_report.csv'):
    """
    失敗したパラメータのレポートをCSV出力
    
    Args:
        failed_list: 失敗情報のリスト
        output_file: 出力ファイル名
    """
    if not failed_list:
        print("\n✓ 計算失敗したパラメータはありません", file=sys.stderr)
        return
    
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"=== 計算失敗パラメータレポート ({len(failed_list)}件) ===", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)
    
    # 失敗理由別に集計
    reason_counts = {}
    for item in failed_list:
        reason = item.get('reason', 'unknown')
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    print("失敗理由の内訳:", file=sys.stderr)
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}: {count:4d}件", file=sys.stderr)
    
    # CSV出力
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Parameter', 'Reason', 'Detail', 'Formula_Preview', 
            'Missing_Variable', 'Failed_Dependency'
        ])
        
        for item in failed_list:
            writer.writerow([
                item.get('param', ''),
                item.get('reason', ''),
                item.get('detail', ''),
                item.get('formula', ''),
                item.get('missing_var', ''),
                item.get('failed_dependency', '')
            ])
    
    print(f"\n✓ 失敗レポートをCSVに出力しました: {output_file}", file=sys.stderr)
    
    # 最初の10件を表示
    print(f"\n{'='*70}", file=sys.stderr)
    print("失敗例（最初の10件）:", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    for i, item in enumerate(failed_list[:10], 1):
        print(f"\n[{i}] パラメータ: {item.get('param', 'unknown')}", file=sys.stderr)
        print(f"    理由: {item.get('reason', 'unknown')}", file=sys.stderr)
        print(f"    詳細: {item.get('detail', '')}", file=sys.stderr)
        if item.get('missing_var'):
            print(f"    未定義変数: {item['missing_var']}", file=sys.stderr)
        if item.get('failed_dependency'):
            print(f"    失敗した依存変数: {item['failed_dependency']}", file=sys.stderr)
        formula_preview = item.get('formula', '')
        if len(formula_preview) > 80:
            formula_preview = formula_preview[:80] + "..."
        print(f"    数式: {formula_preview}", file=sys.stderr)
    
    missing_vars = [item['missing_var'] for item in failed_list if 'missing_var' in item]
    failed_deps = [item['failed_dependency'] for item in failed_list if 'failed_dependency' in item]

    print("\n不足している変数一覧:", file=sys.stderr)
    for var in sorted(set(missing_vars + failed_deps)):
        print(f"  - {var}", file=sys.stderr)


def show_calculated_parameters(combined_params):
    """計算されたパラメータを表示"""
    print(f"\n{'='*70}", file=sys.stderr)
    print("=== 計算されたパラメータ ===", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    # 数値化されたパラメータを抽出
    calculated = {k: v for k, v in combined_params.items() 
                  if isinstance(v, (int, float)) and not k in TRANSISTOR_PARAMS}
    
    # 特定のパラメータを表示
    interesting_params = [
        'od33svtn_dvth_lay', 'f_vtst', 'g_vtin',
        'kstress_vth0_lod', 'l_si', 'w_si',
        'dvth0_lod_vtst', 'dvth0_osex_vtst', 'dvth0_osey_vtst',
    ]
    
    print("\n主要なパラメータ:", file=sys.stderr)
    for param in interesting_params:
        if param in calculated:
            val = calculated[param]
            print(f"  {param:25s} = {val:.6e}", file=sys.stderr)
    
    print(f"\n総計算パラメータ数: {len(calculated)}", file=sys.stderr)
    
    # 未計算（文字列のまま）のパラメータ
    uncalculated = {k: v for k, v in combined_params.items() if isinstance(v, str)}
    if uncalculated and len(uncalculated) <= 20:
        print(f"\n未計算パラメータ ({len(uncalculated)}個):", file=sys.stderr)
        for key in sorted(uncalculated.keys())[:20]:
            val_str = str(uncalculated[key])
            if len(val_str) > 60:
                print(f"  {key:25s} = {val_str[:60]}...", file=sys.stderr)
            else:
                print(f"  {key:25s} = {val_str}", file=sys.stderr)
    elif uncalculated:
        print(f"\n未計算パラメータが {len(uncalculated)} 個あります" , file=sys.stderr)


def export_bsim4_params_to_csv(combined_params, output_file='bsim4_params.csv'):
    """
    BSIM4パラメータをCSV形式で出力（シミュレータ形式に合わせる）
    
    Args:
        combined_params: 統合・計算済みパラメータ辞書
        output_file: 出力ファイル名
    """
    import csv

    # === 最終的なパラメータ置き換え ===
    # print(f"\n{'='*70}", file=sys.stderr)
    # print("=== 最終パラメータ置き換え ===", file=sys.stderr)
    # print(f"{'='*70}", file=sys.stderr)
    
    # w_si と l_si を取得（combined_paramsから）
    if 'w_si' in combined_params and isinstance(combined_params['w_si'], (int, float)):
        w_si = combined_params['w_si']
        # print(f"  w_si = {w_si*1e9:.1f} nm", file=sys.stderr)
    else:
        # print("  ✗ w_si が見つからないため、TRANSISTOR_PARAMSから取得", file=sys.stderr)
        w_si = TRANSISTOR_PARAMS.get('w', 400e-9)
    
    if 'l_si' in combined_params and isinstance(combined_params['l_si'], (int, float)):
        l_si = combined_params['l_si']
        # print(f"  l_si = {l_si*1e9:.1f} nm", file=sys.stderr)
    else:
        # print("  ✗ l_si が見つからないため、TRANSISTOR_PARAMSから取得", file=sys.stderr)
        l_si = TRANSISTOR_PARAMS.get('l', 550e-9)
    
    # lshift を定義（必要に応じて変更）
    if 'lshift' in combined_params and isinstance(combined_params['lshift'], (int, float)):
        lshift = combined_params['lshift']
        # print(f"  lshift (from params) = {lshift*1e9:.1f} nm", file=sys.stderr)
    else:
        lshift = 8e-9  # デフォルトは0
        # print(f"  lshift (hardcoded) = {lshift*1e9:.1f} nm", file=sys.stderr)
    
    # combined_params内の w と l を置き換え
    old_w = combined_params.get('w', 'N/A')
    old_l = combined_params.get('l', 'N/A')
    
    combined_params['w'] = w_si
    combined_params['l'] = l_si - lshift
    
    # print(f"\n  置き換え結果:", file=sys.stderr)
    # if isinstance(old_w, (int, float)):
    #     print(f"    w: {old_w*1e9:.1f} nm → {combined_params['w']*1e9:.1f} nm", file=sys.stderr)
    # else:
    #     print(f"    w: {old_w} → {combined_params['w']*1e9:.1f} nm", file=sys.stderr)
    
    # if isinstance(old_l, (int, float)):
    #     print(f"    l: {old_l*1e9:.1f} nm → {combined_params['l']*1e9:.1f} nm", file=sys.stderr)
    # else:
    #     print(f"    l: {old_l} → {combined_params['l']*1e9:.1f} nm", file=sys.stderr)
    
    # print(f"{'='*70}\n", file=sys.stderr)
    
    # 標準的なBSIM4パラメータのデフォルト値
    default_params = {}
    # L/W/P依存パラメータのベース名
    lwp_base_params = []
    
    final_params = {}
    
    # print("=== BSIM4パラメータ収集中 ===", file=sys.stderr)
    
    # combined_paramsから数値パラメータを全て収集
    base_param_count = 0
    lwp_param_count = 0
    other_param_count = 0
    
    for param_name, param_value in combined_params.items():
        # 数値は従来通り収集
        if isinstance(param_value, (int, float)):
            final_params[param_name] = param_value
            other_param_count += 1
            continue

        # 'type' で 'n'/'p' は文字列のまま収集
        if param_name.lower() == "type" and isinstance(param_value, str) and param_value in ("n", "p"):
            final_params[param_name] = param_value
            other_param_count += 1
            continue

        # デフォルトパラメータに存在するか確認
        if param_name in default_params:
            final_params[param_name] = param_value
            base_param_count += 1
            continue
        
        # L/W/P依存パラメータをチェック
        is_lwp = False
        for base in lwp_base_params:
            if param_name in [f'l{base}', f'w{base}', f'p{base}']:
                final_params[param_name] = param_value
                lwp_param_count += 1
                is_lwp = True
                break
        
        if not is_lwp:
            final_params[param_name] = param_value
            other_param_count += 1
    
    print(f"  combined_paramsから収集: {base_param_count}個（基本パラメータ）", file=sys.stderr)
    print(f"  combined_paramsから収集: {lwp_param_count}個（L/W/Pパラメータ）", file=sys.stderr)
    print(f"  combined_paramsから収集: {other_param_count}個（その他）", file=sys.stderr)
    print(f"  最終パラメータ数: {len(final_params)}個", file=sys.stderr)
    
    # CSVに出力
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'value'])
        
        for param_name in sorted(final_params.keys()):
            value = final_params[param_name]
            if isinstance(value, float):
                value_str = format_significant_digits(value, sig_digits=32)
            elif isinstance(value, int):
                value_str = str(value)
            else:
                value_str = str(value)
            
            writer.writerow([param_name, value_str])
    
    print(f"\n✓ BSIM4パラメータをCSVに出力しました: {output_file}", file=sys.stderr)
    print(f"  総パラメータ数: {len(final_params)}", file=sys.stderr)
    return final_params


def calculate_all_parameters(combined_params, max_iterations=2000):
    """
    全てのパラメータを計算
    
    Args:
        combined_params: 統合パラメータ辞書
        max_iterations: 最大反復回数
    
    Returns:
        dict: 計算済みパラメータ辞書
    """
    # 計算が必要なパラメータのリスト
    params_to_calculate = []
    
    for key, value in combined_params.items():
        if isinstance(value, str):
            params_to_calculate.append(key)
    
    if len(params_to_calculate) == 0:
        print("全てのパラメータは既に計算済みです。", file=sys.stderr)
        return combined_params
 
    calculated_count = 0
    failed_count = 0
    
    iteration = 0
    remaining_params = params_to_calculate
    
    for param_name in remaining_params:
        if iteration >= max_iterations:
            print(f"\n最大反復回数 {max_iterations} に達しました。", file=sys.stderr)
            break
        
        # 既に計算済みならスキップ
        if param_name in combined_params and isinstance(combined_params[param_name], (int, float)):
            continue
        
        # type は 'n'/'p' は式評価をせず、そのまま保持
        if param_name.lower() == "type":
            val = combined_params.get(param_name)
            if isinstance(val, str) and val in ("n", "p"):
                continue

        iteration += 1
        
        if param_name not in combined_params:
            continue
        
        value = combined_params[param_name]
        
        if not isinstance(value, str):
            continue
        
        # 進捗表示（50個ごと）
        # if iteration % 50 == 0:
        #     print(f"  進捗: {iteration}/{min(len(remaining_params), max_iterations)} ({iteration*100//min(len(remaining_params), max_iterations)}%)", file=sys.stderr)
        
        result = calculate_formula(combined_params, value, depth=0, max_depth=1000, update_params=True, param_name_for_tracking=param_name)
        
        if result is not None:
            combined_params[param_name] = result
            calculated_count += 1
        else:
            failed_count += 1

    print(f"\n{'='*70}", file=sys.stderr)
    print(f"計算完了:", file=sys.stderr)
    print(f"  成功: {calculated_count}/{len(params_to_calculate)}", file=sys.stderr)
    print(f"  失敗: {failed_count}/{len(params_to_calculate)}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    # 計算済みパラメータの統計
    numeric_params = {k: v for k, v in combined_params.items() if isinstance(v, (int, float))}
    string_params = {k: v for k, v in combined_params.items() if isinstance(v, str)}
    
    print(f"\n最終統計:", file=sys.stderr)
    print(f"  数値パラメータ: {len(numeric_params)}", file=sys.stderr)
    print(f"  未計算パラメータ: {len(string_params)} {string_params}", file=sys.stderr)
    print(f"  総パラメータ数: {len(combined_params)}", file=sys.stderr)
    
    return combined_params


def calculate_formula_with_trace(bin_params, formula, depth=0, max_depth=1000, 
                                  update_params=True, param_name_for_tracking=None,
                                  trace_enabled=False):
    """
    計算過程を追跡しながら数式を計算
    
    Args:
        trace_enabled: Trueの場合、計算過程を詳細に記録
    """
    if param_name_for_tracking and trace_enabled:
        if param_name_for_tracking not in CALCULATION_TRACE:
            CALCULATION_TRACE[param_name_for_tracking] = []
        
        # 現在のステップを記録
        step_info = {
            'depth': depth,
            'formula': str(formula)[:500],
            'substitutions': []
        }
    
    if depth > max_depth:
        error_info = {
            'param': param_name_for_tracking or 'unknown',
            'reason': 'max_recursion_depth',
            'detail': f'再帰の深さが上限({max_depth})に達しました',
            'formula': str(formula)[:200]
        }
        FAILED_PARAMS.append(error_info)
        return None
    
    if depth == 0:
        formula = convert_units_in_formula(formula)
    
    try:
        # 計算を試みる
        result = eval(formula, {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "min": min,
            "max": max,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "int": int,
            "floor": math.floor,
            "ceil": math.ceil,
            "round": round,
        })
        
        # トレース記録（成功時）
        if param_name_for_tracking and trace_enabled:
            step_info['result'] = result
            step_info['success'] = True
            CALCULATION_TRACE[param_name_for_tracking].append(step_info)
        
        return result
        
    except NameError as e:
        error_msg = str(e)
        match = re.search(r"'([^']*)'", error_msg)
        
        if not match:
            error_info = {
                'param': param_name_for_tracking or 'unknown',
                'reason': 'name_error_parse_failed',
                'detail': f'変数名を抽出できませんでした: {error_msg}',
                'formula': str(formula)[:200]
            }
            FAILED_PARAMS.append(error_info)
            return None
        
        var_name = match.group(1)
        
        if var_name not in bin_params:
            error_info = {
                'param': param_name_for_tracking or var_name,
                'reason': 'undefined_variable',
                'detail': f'変数 "{var_name}" が辞書に存在しません',
                'formula': str(formula)[:200],
                'missing_var': var_name
            }
            FAILED_PARAMS.append(error_info)
            return None
        
        val = bin_params[var_name]
        
        # トレースに変数置換を記録
        if param_name_for_tracking and trace_enabled:
            sub_info = {
                'variable': var_name,
                'value_before': str(val)[:100] if isinstance(val, str) else val,
                'type': type(val).__name__
            }
        
        if isinstance(val, str):
            val_converted = convert_units_in_formula(val)
            
            # 依存変数を再帰的に計算
            calculated_value = calculate_formula_with_trace(
                bin_params, val_converted, depth + 1, max_depth, update_params,
                param_name_for_tracking=var_name,
                trace_enabled=trace_enabled
            )
            
            if calculated_value is not None:
                if update_params:
                    bin_params[var_name] = calculated_value
                
                # トレースに計算結果を記録
                if param_name_for_tracking and trace_enabled:
                    sub_info['value_after'] = calculated_value
                    step_info['substitutions'].append(sub_info)
                
                new_formula = safe_replace_variable(formula, var_name, str(calculated_value))
            else:
                error_info = {
                    'param': param_name_for_tracking or 'unknown',
                    'reason': 'dependency_calculation_failed',
                    'detail': f'依存変数 "{var_name}" の計算に失敗',
                    'formula': str(formula)[:200],
                    'failed_dependency': var_name
                }
                FAILED_PARAMS.append(error_info)
                return None
            
            if new_formula == formula:
                error_info = {
                    'param': param_name_for_tracking or 'unknown',
                    'reason': 'substitution_failed',
                    'detail': f'変数 "{var_name}" の置換が行われませんでした',
                    'formula': str(formula)[:200],
                    'var_name': var_name
                }
                FAILED_PARAMS.append(error_info)
                return None
            
        elif isinstance(val, (int, float)):
            new_formula = safe_replace_variable(formula, var_name, str(val))
            
            # トレースに置換を記録
            if param_name_for_tracking and trace_enabled:
                sub_info['value_after'] = val
                step_info['substitutions'].append(sub_info)
        else:
            error_info = {
                'param': param_name_for_tracking or var_name,
                'reason': 'unsupported_type',
                'detail': f'未対応の型: {type(val).__name__}',
                'formula': str(formula)[:200],
                'var_name': var_name
            }
            FAILED_PARAMS.append(error_info)
            return None
        
        # 現在のステップを記録してから再帰
        if param_name_for_tracking and trace_enabled:
            CALCULATION_TRACE[param_name_for_tracking].append(step_info)
        
        return calculate_formula_with_trace(
            bin_params, new_formula, depth, max_depth, update_params,
            param_name_for_tracking=param_name_for_tracking,
            trace_enabled=trace_enabled
        )
    
    except (SyntaxError, ZeroDivisionError, Exception) as e:
        # エラー処理
        error_info = {
            'param': param_name_for_tracking or 'unknown',
            'reason': f'{type(e).__name__}',
            'detail': str(e),
            'formula': str(formula)[:200]
        }
        FAILED_PARAMS.append(error_info)
        return None


def export_calculation_trace(param_name, output_file=None):
    """
    特定のパラメータの計算トレースを出力
    
    Args:
        param_name: トレースするパラメータ名
        output_file: 出力ファイル名（Noneの場合は標準出力）
    """
    if param_name not in CALCULATION_TRACE:
        print(f"\n✗ パラメータ '{param_name}' のトレース情報がありません", file=sys.stderr)
        return
    
    trace = CALCULATION_TRACE[param_name]
    
    if output_file:
        f = open(output_file, 'w', encoding='utf-8')
    else:
        f = sys.stderr
    
    try:
        f.write(f"\n{'='*70}\n")
        f.write(f"パラメータ '{param_name}' の計算トレース\n")
        f.write(f"{'='*70}\n\n")
        
        f.write(f"総ステップ数: {len(trace)}\n\n")
        
        for i, step in enumerate(trace, 1):
            f.write(f"{'─'*70}\n")
            f.write(f"ステップ {i} (深さ: {step['depth']})\n")
            f.write(f"{'─'*70}\n")
            
            f.write(f"\n【数式】\n")
            formula = step['formula']
            if len(formula) > 80:
                for j in range(0, len(formula), 80):
                    f.write(f"  {formula[j:j+80]}\n")
            else:
                f.write(f"  {formula}\n")
            
            if step.get('substitutions'):
                f.write(f"\n【変数置換】\n")
                for sub in step['substitutions']:
                    f.write(f"  変数名: {sub['variable']}\n")
                    f.write(f"  型: {sub['type']}\n")
                    
                    if isinstance(sub.get('value_before'), str):
                        val_before = sub['value_before']
                        if len(val_before) > 60:
                            f.write(f"  置換前: {val_before[:60]}...\n")
                        else:
                            f.write(f"  置換前: {val_before}\n")
                    else:
                        f.write(f"  置換前: {sub.get('value_before')}\n")
                    
                    val_after = sub.get('value_after')
                    if isinstance(val_after, float):
                        f.write(f"  置換後: {val_after:.12e}\n")
                    else:
                        f.write(f"  置換後: {val_after}\n")
                    f.write("\n")
            
            if step.get('success') and 'result' in step:
                result = step['result']
                f.write(f"\n【計算結果】\n")
                if isinstance(result, float):
                    f.write(f"  {result:.12e}\n")
                    f.write(f"  {result:.12f}\n")
                else:
                    f.write(f"  {result}\n")
            
            f.write("\n")
        
        f.write(f"{'='*70}\n")
        
        if output_file:
            print(f"\n✓ トレース情報を出力しました: {output_file}", file=sys.stderr)
    
    finally:
        if output_file:
            f.close()


def extract_bin_bounds(bin_entry: dict) -> Optional[Tuple[float, float, float, float]]:
    """
    BINエントリから (lmin, lmax, wmin, wmax) を SI[m] で取得。
    """
    if not isinstance(bin_entry, dict):
        return None

    # まず parameters 部を取る（無ければ bin_entry 自身）
    params = bin_entry.get("parameters", bin_entry)

    # キーのバリアントに対応
    key_aliases = {
        "lmin": ["lmin", "l_min"],
        "lmax": ["lmax", "l_max"],
        "wmin": ["wmin", "w_min"],
        "wmax": ["wmax", "w_max"],
    }

    def get_param(klist):
        for k in klist:
            if k in params:
                val = params[k]
                return parse_unit_value(val) if isinstance(val, str) else float(val)
        return None

    lmin = get_param(key_aliases["lmin"])
    lmax = get_param(key_aliases["lmax"])
    wmin = get_param(key_aliases["wmin"])
    wmax = get_param(key_aliases["wmax"])

    # 必須が欠ける場合は None
    if lmin is None or lmax is None or wmin is None or wmax is None:
        return None

    # 境界の整合性（逆転時の補正）
    if lmin > lmax:
        lmin, lmax = lmax, lmin
    if wmin > wmax:
        wmin, wmax = wmax, wmin

    return (float(lmin), float(lmax), float(wmin), float(wmax))


def rect_distance(l: float, w: float, bounds: Tuple[float, float, float, float]) -> float:
    """
    点 (l, w) と矩形 bounds=(lmin,lmax,wmin,wmax) の距離。
    """
    lmin, lmax, wmin, wmax = bounds
    dl = 0.0 if lmin <= l <= lmax else min(abs(l - lmin), abs(l - lmax))
    dw = 0.0 if wmin <= w <= wmax else min(abs(w - wmin), abs(w - wmax))
    return (dl**2 + dw**2) ** 0.5


def choose_bin_id_for_dimensions(bins: dict, l_si: float, w_si: float) -> Optional[str]:
    """
    bins と寸法 (l_si, w_si) から最適 BIN_ID を選ぶ。
    """
    if not isinstance(bins, dict) or not bins:
        return None

    candidates_in = []   # (bin_id, bounds, area)
    candidates_all = []  # (bin_id, bounds, distance)

    for bin_id, bin_entry in bins.items():
        bounds = extract_bin_bounds(bin_entry)
        if bounds is None:
            continue
        lmin, lmax, wmin, wmax = bounds
        inside = (lmin <= l_si <= lmax) and (wmin <= w_si <= wmax)
        if inside:
            area = max(lmax - lmin, 0.0) * max(wmax - wmin, 0.0)
            candidates_in.append((bin_id, bounds, area))
        else:
            dist = rect_distance(l_si, w_si, bounds)
            candidates_all.append((bin_id, bounds, dist))

    # 1) 包含優先：最小面積
    if candidates_in:
        candidates_in.sort(key=lambda t: (t[2], t[0]))
        return candidates_in[0][0]

    # 2) 近傍優先：最小距離
    if candidates_all:
        candidates_all.sort(key=lambda t: (t[2], t[0]))
        return candidates_all[0][0]

    return None


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--lib", type=str, required=True, help="model lib name")
#     parser.add_argument("--corner", type=str, default="tt", help="process corner")
#     parser.add_argument("--jsondir", type=str, default="data/model/SPECTRE", help="JSON directory")
#     parser.add_argument("--w", type=float, default=400e-9)
#     parser.add_argument("--l", type=float, default=550e-9)
#     parser.add_argument("--nf", type=float, default=1)
#     parser.add_argument("--m", type=float, default=1)
#     parser.add_argument("--sd", type=float, default=190e-9)
#     parser.add_argument("--ad", type=float, default=5.98395e-14)
#     parser.add_argument("--as", type=float, default=5.98395e-14)
#     parser.add_argument("--pd", type=float, default=1.113333e-6)
#     parser.add_argument("--ps", type=float, default=1.113333e-6)
#     parser.add_argument("--nrd", type=float, default=0.2567568)
#     parser.add_argument("--nrs", type=float, default=0.2567568)
#     parser.add_argument("--sa", type=float, default=131.0e-9)
#     parser.add_argument("--sb", type=float, default=131.0e-9)
#     parser.add_argument("--outcsv", type=str, default="bsim4_params.csv")
#     parser.add_argument("--outfmt", type=str, default="json", choices=["csv", "json"])
#     args = parser.parse_args()

#     # グローバル上書き
#     global PROCESS_CORNER, TRANSISTOR_PARAMS, BIN_ID
#     PROCESS_CORNER = args.corner

#     # トランジスタ寸法の上書き
#     overrides = {k: v for k, v in vars(args).items()
#                  if k in TRANSISTOR_PARAMS and v is not None}
#     TRANSISTOR_PARAMS.update(overrides)

#     # JSONロード
#     json_path = os.path.join(args.jsondir, f"{args.lib}.json")
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             result = json.load(f)
#     except FileNotFoundError:
#         print(f"✗ JSON not found: {json_path}", file=sys.stderr)
#         sys.exit(1)
#     except json.JSONDecodeError as e:
#         print(f"✗ JSON decode error: {e}", file=sys.stderr)
#         sys.exit(1)

#     # セクション取得
#     sections = result.get("section", {})
#     if not sections:
#         print("✗ JSON構造に 'section' がありません", file=sys.stderr)
#         sys.exit(1)

#     # corner セクション
#     corner_key = args.corner.lower()
#     if corner_key not in sections:
#         print(f"✗ プロセスコーナー '{args.corner}' が JSON に見つかりません", file=sys.stderr)
#         print(f"  利用可能: {', '.join(sections.keys())}", file=sys.stderr)
#         sys.exit(1)
#     corner_data = sections[corner_key]
#     process_param = corner_data.get("parameters", {})

#     # メインセクション
#     main_key = None
#     for k, v in sections.items():
#         if isinstance(v, dict) and len(v.get("includes", [])) == 0:
#             main_key = k
#             break
#     if main_key is None:
#         main_key = next(iter(sections.keys()))
#     core_data = sections.get(main_key, {})
#     core_params = core_data.get("parameters", {})

#     # inline_subckt パラメータ
#     inline_params = {}
#     inline = core_data.get("inline_subckts", {})
#     lib_subckt = inline.get(args.lib, {})
#     inline_params = lib_subckt.get("parameters", {})

#     # bin パラメータを探索（L/WでBIN選択）
#     selected_bin_id = None
#     bin_params = {}

#     models = lib_subckt.get("models", {})
#     if models:
#         first_model_data = next(iter(models.values()))
#         bins = first_model_data.get("bins", {})
#         if bins:
#             l_si = float(TRANSISTOR_PARAMS.get('l', 550e-9))
#             w_si = float(TRANSISTOR_PARAMS.get('w', 400e-9))

#             # L/WからBINを選択
#             selected_bin_id = choose_bin_id_for_dimensions(bins, l_si, w_si)

#             if selected_bin_id is None:
#                 # フォールバック
#                 if BIN_ID in bins:
#                     selected_bin_id = BIN_ID
#                     print(f"⚠ BIN auto-select failed. Fallback to BIN_ID='{BIN_ID}'.", file=sys.stderr)
#                 else:
#                     selected_bin_id = next(iter(bins.keys()))
#                     print(f"⚠ BIN auto-select failed. Fallback to first bin '{selected_bin_id}'.", file=sys.stderr)

#             # 選択したBINのパラメータ
#             bin_entry = bins[selected_bin_id]
#             bin_params = bin_entry.get("parameters", {})

#             # ログ表示
#             bounds = extract_bin_bounds(bin_entry)
#             if bounds is not None:
#                 lmin, lmax, wmin, wmax = bounds
#                 print(f"✓ Selected BIN_ID: {selected_bin_id}  (by L/W)", file=sys.stderr)
#                 print(f"   Bounds: L=[{lmin:.3e},{lmax:.3e}] m, W=[{wmin:.3e},{wmax:.3e}] m", file=sys.stderr)
#             else:
#                 print(f"✓ Selected BIN_ID: {selected_bin_id} (bounds unavailable)", file=sys.stderr)

#     # === パラメータ統合 ===
#     combined_params = {}

#     # 1) コーナー設定フラグ
#     combined_params.update(get_process_corner_settings(PROCESS_CORNER))

#     # 2) corner の parameters
#     for k, v in process_param.items():
#         if k not in combined_params:
#             combined_params[k] = parse_unit_value(v)

#     # 3) core parameters
#     for k, v in core_params.items():
#         if k not in combined_params:
#             combined_params[k] = parse_unit_value(v)

#     # 4) inline_subckt parameters
#     for k, v in inline_params.items():
#         if k not in combined_params:
#             combined_params[k] = parse_unit_value(v)

#     # 5) bin parameters
#     for k, v in bin_params.items():
#         if k not in combined_params:
#             combined_params[k] = parse_unit_value(v)

#     # 6) トランジスタ寸法
#     combined_params.update(TRANSISTOR_PARAMS)

#     # === 依存式の数値化 ===
#     combined_params = calculate_all_parameters(combined_params)

#     # === 最終出力 ===
#     final_params_dict = export_bsim4_params_to_csv(combined_params, args.outcsv)

#     if args.outfmt == "json":
#         payload = {}
#         for k, v in final_params_dict.items():
#             if isinstance(v, (int, float)):
#                 payload[k] = float(v)
#             else:
#                 payload[k] = v

#         # ★ デバッグ出力を追加
#         if "type" in payload:
#             print(f"DEBUG4: type = {payload['type']}", file=sys.stderr)
#         else:
#             print("DEBUG4: 'type' key missing in payload", file=sys.stderr)
        
#         print(json.dumps(payload))

#     # 失敗レポート
#     if FAILED_PARAMS:
#         export_failed_params_report(FAILED_PARAMS, 'failed_params_report.csv')
#     else:
#         print("\n✓ 計算失敗はありません", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=str, required=True, help="model lib name")
    parser.add_argument("--corner", type=str, default="tt", help="process corner")
    parser.add_argument("--jsondir", type=str, default="data/model/SPECTRE", help="JSON directory")
    parser.add_argument("--w", type=float, default=400e-9)
    parser.add_argument("--l", type=float, default=550e-9)
    parser.add_argument("--nf", type=float, default=1)
    parser.add_argument("--multi", type=float, default=1)
    parser.add_argument("--sd", type=float, default=190e-9)
    parser.add_argument("--ad", type=float, default=5.98395e-14)
    parser.add_argument("--as", type=float, default=5.98395e-14)
    parser.add_argument("--pd", type=float, default=1.113333e-6)
    parser.add_argument("--ps", type=float, default=1.113333e-6)
    parser.add_argument("--nrd", type=float, default=0.2567568)
    parser.add_argument("--nrs", type=float, default=0.2567568)
    parser.add_argument("--sa", type=float, default=131.0e-9)
    parser.add_argument("--sb", type=float, default=131.0e-9)
    parser.add_argument("--outcsv", type=str, default="bsim4_params.csv")
    parser.add_argument("--outfmt", type=str, default="json", choices=["csv", "json"])
    args = parser.parse_args()


    # グローバル上書き
    global PROCESS_CORNER, TRANSISTOR_PARAMS, BIN_ID
    PROCESS_CORNER = args.corner

    # トランジスタ寸法の上書き
    overrides = {k: v for k, v in vars(args).items()
                 if k in TRANSISTOR_PARAMS and v is not None}
    TRANSISTOR_PARAMS.update(overrides)


    # JSONロード
    json_path = os.path.join(args.jsondir, f"{args.lib}.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    except FileNotFoundError:
        print(f"✗ JSON not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ JSON decode error: {e}", file=sys.stderr)
        sys.exit(1)

    # セクション取得
    sections = result.get("section", {})
    if not sections:
        print("✗ JSON構造に 'section' がありません", file=sys.stderr)
        sys.exit(1)

    # corner セクション
    corner_key = args.corner.lower()
    if corner_key not in sections:
        print(f"✗ プロセスコーナー '{args.corner}' が JSON に見つかりません", file=sys.stderr)
        print(f"  利用可能: {', '.join(sections.keys())}", file=sys.stderr)
        sys.exit(1)
    
    corner_data = sections[corner_key]
    corner_params = corner_data.get("parameters", {})
    
    # ★★★ ここを変更：JSONから直接コーナーパラメータを取得 ★★★
    # print(f"\n{'='*70}", file=sys.stderr)
    # print(f"=== コーナーパラメータ取得 (section->'{corner_key}'->parameters) ===", file=sys.stderr)
    # print(f"{'='*70}", file=sys.stderr)
    # print(f"  取得したパラメータ数: {len(corner_params)}", file=sys.stderr)
    
    # デバッグ：コーナーパラメータの内容を表示
    # for key, value in corner_params.items():
    #     print(f"    {key}: {value}", file=sys.stderr)
    # print(f"{'='*70}\n", file=sys.stderr)

    # メインセクション（coreなど）を特定
    main_key = None
    for k, v in sections.items():
        if isinstance(v, dict) and len(v.get("includes", [])) == 0:
            main_key = k
            break
    if main_key is None:
        main_key = next(iter(sections.keys()))
    
    core_data = sections.get(main_key, {})
    core_params = core_data.get("parameters", {})

    # inline_subckt パラメータ
    inline_params = {}
    inline = core_data.get("inline_subckts", {})
    lib_subckt = inline.get(args.lib, {})
    inline_params = lib_subckt.get("parameters", {})

    # bin パラメータを探索（L/WでBIN選択）
    selected_bin_id = None
    bin_params = {}

    models = lib_subckt.get("models", {})
    if models:
        first_model_data = next(iter(models.values()))
        bins = first_model_data.get("bins", {})
        if bins:
            l_si = float(TRANSISTOR_PARAMS.get('l', 550e-9))
            w_si = float(TRANSISTOR_PARAMS.get('w', 400e-9))

            # L/WからBINを選択
            selected_bin_id = choose_bin_id_for_dimensions(bins, l_si, w_si)

            if selected_bin_id is None:
                # フォールバック
                if BIN_ID in bins:
                    selected_bin_id = BIN_ID
                    print(f"⚠ BIN auto-select failed. Fallback to BIN_ID='{BIN_ID}'.", file=sys.stderr)
                else:
                    selected_bin_id = next(iter(bins.keys()))
                    print(f"⚠ BIN auto-select failed. Fallback to first bin '{selected_bin_id}'.", file=sys.stderr)

            # 選択したBINのパラメータ
            bin_entry = bins[selected_bin_id]
            bin_params = bin_entry.get("parameters", {})
            if os.environ.get("BSIM4SIM_DEBUG", "0") == "1":
                print(
                    f"DEBUG_BIN_CHECK: Selected BIN={selected_bin_id}. type={bin_params.get('type','MISSING')}",
                    file=sys.stderr,
                )

            # ログ表示
            # bounds = extract_bin_bounds(bin_entry)
            # if bounds is not None:
            #     lmin, lmax, wmin, wmax = bounds
            #     print(f"✓ Selected BIN_ID: {selected_bin_id}  (by L/W)", file=sys.stderr)
            #     print(f"   Bounds: L=[{lmin:.3e},{lmax:.3e}] m, W=[{wmin:.3e},{wmax:.3e}] m", file=sys.stderr)
            # else:
            #     print(f"✓ Selected BIN_ID: {selected_bin_id} (bounds unavailable)", file=sys.stderr)

    # === パラメータ統合 ===
    # print(f"\n{'='*70}", file=sys.stderr)
    # print("=== パラメータ統合順序 (Modified: Flags -> Bin -> Inline -> Core -> Corner) ===", file=sys.stderr)
    # print(f"{'='*70}", file=sys.stderr)
    
    combined_params = {}

    # 0) コーナー設定フラグ (Always needed for conditional parameters)
    combined_params.update(get_process_corner_settings(PROCESS_CORNER))

    # 1) Bin parameters (Most Specific - Highest Priority)
    for k, v in bin_params.items():
        combined_params[k] = parse_unit_value(v)
    # print(f"  1. binパラメータ: {len(bin_params)}個", file=sys.stderr)

    # 2) Inline Subckt parameters
    added_count = 0
    for k, v in inline_params.items():
        if k not in combined_params:
            combined_params[k] = parse_unit_value(v)
            added_count += 1
    # print(f"  2. inline_subcktパラメータ: {added_count}個追加（総{len(inline_params)}個）", file=sys.stderr)

    # 3) Core parameters
    added_count = 0
    for k, v in core_params.items():
        if k not in combined_params:
            combined_params[k] = parse_unit_value(v)
            added_count += 1
    # print(f"  3. coreパラメータ: {added_count}個追加（総{len(core_params)}個）", file=sys.stderr)

    # 4) Corner parameters (Globals/Variables)
    added_count = 0
    for k, v in corner_params.items():
        if k not in combined_params:
            combined_params[k] = parse_unit_value(v)
            added_count += 1
    print(f"  4. コーナーパラメータ: {added_count}個追加（総{len(corner_params)}個）", file=sys.stderr)

    # 5) トランジスタ寸法（最優先・強制上書き）
    combined_params.update(TRANSISTOR_PARAMS)
    print(f"  5. トランジスタ寸法: {len(TRANSISTOR_PARAMS)}個（強制上書き）", file=sys.stderr)
    
    print(f"\n  統合後の総パラメータ数: {len(combined_params)}個", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)

    # === 依存式の数値化 ===
    combined_params = calculate_all_parameters(combined_params)

    # === 最終出力 ===
    final_params_dict = export_bsim4_params_to_csv(combined_params, args.outcsv)
    if os.environ.get("BSIM4SIM_DEBUG", "0") == "1":
        print(f"DEBUG4: type = {final_params_dict.get('type', 'NOT_FOUND')}", file=sys.stderr)
    if args.outfmt == "json":
        payload = {}
        for k, v in final_params_dict.items():
            if isinstance(v, (int, float)):
                payload[k] = float(v)
            else:
                payload[k] = v

  
        
        print(json.dumps(payload))

    # 失敗レポート
    if FAILED_PARAMS:
        export_failed_params_report(FAILED_PARAMS, 'failed_params_report.csv')
    else:
        print("\n✓ 計算失敗はありません", file=sys.stderr)


if __name__ == "__main__":
    main()
# ==========================================
# Optimized Class for PyCall / Persistent Use
# ==========================================
class BSIM4Calculator:
    def __init__(self, json_dir, lib_name):
        self.json_dir = json_dir
        self.lib_name = lib_name
        self.json_path = os.path.join(json_dir, f"{lib_name}.json")
        self.data = self._load_json()

    def _load_json(self):
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON {self.json_path}: {e}", file=sys.stderr)
            return None

    def calculate(self, corner, transistor_params_override):
        """
        corner: str ('tt', 'ss', etc)
        transistor_params_override: dict (e.g. {'w': 4e-7, 'l': 5.5e-7 ...})
        """
        if self.data is None:
             raise RuntimeError(f"JSON data not loaded for {self.lib_name}")

        result = self.data
        sections = result.get("section", {})
        if not sections:
            raise ValueError("No 'section' in JSON")

        corner_key = corner.lower()
        if corner_key not in sections:
            raise ValueError(f"Corner '{corner}' not found. Available: {list(sections.keys())}")

        corner_data = sections[corner_key]
        corner_params = corner_data.get("parameters", {})

        # Main Key (Core)
        main_key = None
        for k, v in sections.items():
            if isinstance(v, dict) and len(v.get("includes", [])) == 0:
                main_key = k
                break
        if main_key is None:
            main_key = next(iter(sections.keys()))
        
        core_data = sections.get(main_key, {})
        core_params = core_data.get("parameters", {})

        # Inline
        inline = core_data.get("inline_subckts", {})
        lib_subckt = inline.get(self.lib_name, {})
        inline_params = lib_subckt.get("parameters", {})

        # Bin
        bin_params = {}
        bins = {}
        models = lib_subckt.get("models", {})
        if models:
            first_model_data = next(iter(models.values()))
            bins = first_model_data.get("bins", {})
            if bins:
                l_si = float(transistor_params_override.get('l', 550e-9))
                w_si = float(transistor_params_override.get('w', 400e-9))
                selected_bin_id = choose_bin_id_for_dimensions(bins, l_si, w_si)
                
                if selected_bin_id is None:
                     if '32' in bins:
                         selected_bin_id = '32'
                     else:
                         selected_bin_id = next(iter(bins.keys()))
                
                bin_entry = bins[selected_bin_id]
                bin_params = bin_entry.get("parameters", {})

        # Combine
        combined = {}
        
        # 0) Corner settings
        combined.update(get_process_corner_settings(corner))

        # 1) Bin
        for k, v in bin_params.items():
            combined[k] = parse_unit_value(v)

        # 2) Inline
        for k, v in inline_params.items():
            if k not in combined:
                combined[k] = parse_unit_value(v)

        # 3) Core
        for k, v in core_params.items():
            if k not in combined:
                combined[k] = parse_unit_value(v)

        # 4) Corner
        for k, v in corner_params.items():
            if k not in combined:
                combined[k] = parse_unit_value(v)

        # 5) Transistor overrides
        current_t_params = TRANSISTOR_PARAMS.copy()
        current_t_params.update(transistor_params_override)
        combined.update(current_t_params)

        # Calculate
        final = calculate_all_parameters(combined)
        return final

