#!/usr/bin/env python3
"""gcn5_woe 配置加载与路径解析工具"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import shutil


def load_config(config_path: Optional[str] = None) -> tuple[dict, Path]:
    """加载YAML配置，返回(config, config_path)"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("缺少PyYAML，请先安装: pip install pyyaml") from exc

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config, config_path


def _latest_timestamp(base_dir: Path) -> str:
    candidates = [p.name for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"数据目录为空: {base_dir}")
    return sorted(candidates)[-1]


def resolve_data_root(config: dict) -> Tuple[Path, str, str]:
    """
    根据config解析数据路径

    Returns:
        (data_root, timestamp, device)
        data_root = <data_root>/<timestamp>/<device>/<gnn_data_subdir>
    """
    data_cfg = config.get("data", {})
    base = Path(config.get("paths", {}).get("data_root", "/home/hu/Diana-Sim/gdata"))

    timestamp = data_cfg.get("timestamp", "latest")
    if timestamp == "latest":
        timestamp = _latest_timestamp(base)

    device = data_cfg.get("device", "nmos")
    subdir = data_cfg.get("gnn_data_subdir", "gnn_data")

    data_root = base / timestamp / device / subdir
    if not data_root.exists():
        raise FileNotFoundError(f"数据路径不存在: {data_root}")

    return data_root, timestamp, device


def resolve_output_dir(config: dict) -> Path:
    return Path(config.get("paths", {}).get("output_dir", "/home/hu/Diana-Sim/gcn5_woe"))


def copy_config_snapshot(config_path: Path, output_dir: Path, filename: str) -> Path:
    """拷贝配置文件快照到输出目录"""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename
    shutil.copy(config_path, dest)
    return dest
