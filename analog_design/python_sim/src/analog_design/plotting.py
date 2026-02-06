"""Plotting utilities for regression and waveform comparison."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass
class Columns:
    t: str
    vin: Optional[str]
    vout: str


def _read_csv(path: Path) -> tuple[Dict[str, np.ndarray], list[str]]:
    import csv

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    cols = reader.fieldnames or []
    data = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    return data, cols


def _pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        lc = cand.lower()
        if lc in lower:
            return lower[lc]
    return None


def _find_col(cols, candidates):
    for cand in candidates:
        if cand in cols:
            return cand
        for c in cols:
            if c.lower() == cand.lower():
                return c
    return None


def _lininterp(x, y, xq):
    return np.interp(xq, x, y)


def detect_columns(cols) -> Columns:
    tcol = _pick_col(cols, ["t", "time", "Time"])
    if not tcol:
        raise ValueError(f"time column not found. cols={cols}")

    vout = _pick_col(cols, ["VOUT", "_VOUT", "/VOUT", "vout"])
    if not vout:
        raise ValueError(f"VOUT column not found. cols={cols}")

    vin = _pick_col(cols, ["VIN", "_VIN", "/VIN", "vin"])
    return Columns(t=tcol, vin=vin, vout=vout)


def plot_tran_compare(sim_csv: Path, golden_csv: Path, out_png: Path, title: str) -> float:
    sim_data, sim_cols = _read_csv(sim_csv)
    gold_data, gold_cols = _read_csv(golden_csv)

    sim_c = detect_columns(sim_cols)
    gold_c = detect_columns(gold_cols)

    t_sim = sim_data[sim_c.t]
    vout_sim = sim_data[sim_c.vout]

    t_gold = gold_data[gold_c.t]
    vout_gold = gold_data[gold_c.vout]

    vout_gold_on_sim = _lininterp(t_gold, vout_gold, t_sim)
    err = vout_sim - vout_gold_on_sim
    abs_err = np.abs(err)

    plt = _import_matplotlib()

    fig, axes = plt.subplots(3 if (sim_c.vin and gold_c.vin) else 2, 1, figsize=(10, 8), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax0 = axes[0]
    ax0.plot(t_sim * 1e9, vout_sim, label="sim VOUT")
    ax0.plot(t_sim * 1e9, vout_gold_on_sim, label="gold VOUT (interp)", linestyle="--")
    ax0.set_ylabel("VOUT [V]")
    ax0.grid(True)
    ax0.legend(loc="best")

    idx = 1
    if sim_c.vin and gold_c.vin:
        vin_sim = sim_data[sim_c.vin]
        vin_gold = gold_data[gold_c.vin]
        vin_gold_on_sim = _lininterp(t_gold, vin_gold, t_sim)

        ax1 = axes[1]
        ax1.plot(t_sim * 1e9, vin_sim, label="sim VIN")
        ax1.plot(t_sim * 1e9, vin_gold_on_sim, label="gold VIN (interp)", linestyle="--")
        ax1.set_ylabel("VIN [V]")
        ax1.grid(True)
        ax1.legend(loc="best")
        idx = 2

    axe = axes[idx]
    axe.plot(t_sim * 1e9, err, label="VOUT error (sim-gold)")
    axe.plot(t_sim * 1e9, abs_err, label="|error|", linestyle=":")
    axe.set_xlabel("time [ns]")
    axe.set_ylabel("error [V]")
    axe.grid(True)
    axe.legend(loc="best")

    max_abs = float(np.max(abs_err))
    t_at = float(t_sim[np.argmax(abs_err)])
    fig.suptitle(f"{title} | max|err|={max_abs:.6g} V at t={t_at*1e9:.6g} ns")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    return max_abs


def plot_comparator_waveforms(sim_csv: Path, gold_csv: Path, out_png: Path, title: str = "Comparator Waveforms") -> None:
    plt = _import_matplotlib()

    sim_data, sim_cols = _read_csv(sim_csv)
    gold_data, gold_cols = _read_csv(gold_csv)

    t_sim = sim_data[_find_col(sim_cols, ["t", "time"])]
    t_gold = gold_data[_find_col(gold_cols, ["t", "time"])]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 1. VIN
    ax = axes[0]
    vin_sim = sim_data.get(_find_col(sim_cols, ["VIN"]))
    vin_gold = gold_data.get(_find_col(gold_cols, ["/VIN", "VIN"]))
    if vin_sim is not None:
        ax.plot(t_sim * 1e9, vin_sim, label="Python VIN", color="blue")
    if vin_gold is not None:
        ax.plot(t_gold * 1e9, vin_gold, label="Spectre VIN", linestyle="--", color="red")
    ax.set_ylabel("VIN [V]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # 2. VOUT
    ax = axes[1]
    vout_sim = sim_data.get(_find_col(sim_cols, ["VOUT"]))
    vout_gold = gold_data.get(_find_col(gold_cols, ["/VOUT", "VOUT"]))
    if vout_sim is not None:
        ax.plot(t_sim * 1e9, vout_sim, label="Python VOUT", color="blue")
    if vout_gold is not None:
        ax.plot(t_gold * 1e9, vout_gold, label="Spectre VOUT", linestyle="--", color="red")
    ax.set_ylabel("VOUT [V]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 3. VOUT2
    ax = axes[2]
    vout2_sim = sim_data.get(_find_col(sim_cols, ["VOUT2"]))
    vout2_gold = gold_data.get(_find_col(gold_cols, ["/VOUT2", "VOUT2"]))
    if vout2_sim is not None:
        ax.plot(t_sim * 1e9, vout2_sim, label="Python VOUT2", color="blue")
    if vout2_gold is not None:
        ax.plot(t_gold * 1e9, vout2_gold, label="Spectre VOUT2", linestyle="--", color="red")
    ax.set_ylabel("VOUT2 [V]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 4. Internal nodes
    ax = axes[3]
    m1d_sim = sim_data.get(_find_col(sim_cols, ["M1_D"]))
    m2d_sim = sim_data.get(_find_col(sim_cols, ["M2_D"]))
    m1d_gold = gold_data.get(_find_col(gold_cols, ["/M1_D", "M1_D"]))
    m2d_gold = gold_data.get(_find_col(gold_cols, ["/M2_D", "M2_D"]))

    if m1d_sim is not None:
        ax.plot(t_sim * 1e9, m1d_sim, label="Python M1_D", color="blue")
    if m2d_sim is not None:
        ax.plot(t_sim * 1e9, m2d_sim, label="Python M2_D", color="green")
    if m1d_gold is not None:
        ax.plot(t_gold * 1e9, m1d_gold, label="Spectre M1_D", linestyle="--", color="red")
    if m2d_gold is not None:
        ax.plot(t_gold * 1e9, m2d_gold, label="Spectre M2_D", linestyle="--", color="orange")

    ax.set_ylabel("Internal [V]")
    ax.set_xlabel("Time [ns]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if vout_sim is not None and vout_gold is not None:
        vout_gold_interp = _lininterp(t_gold, vout_gold, t_sim)
        max_err = np.max(np.abs(vout_sim - vout_gold_interp))
        fig.text(0.5, 0.01, f"Max VOUT Error: {max_err:.4f} V", ha="center", fontsize=10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[OK] Saved: {out_png}")
