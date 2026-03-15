"""
Pseudo-Steady Convergence Study
==================================
1. Tessellation (u and w panels): increase until |xi-xi-1|/|xi| < 0.01
2. Wake iterations: run at high count, see when it stops changing
3. Trailing wake nodes: different counts, plot differences

Metrics tracked: CT (CFx), CQ (CMx), FOM
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openvsp as vsp
from openvsp import VSPAERO_PROP_PSEUDO_STEADY

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from vspaero_config import (
    VSP_FILE, R, RHO, OMEGA, VREF, MREF, SREF, BREF, CREF, RE_CREF, NCPU,
    NUM_REVS, RPM,
)
from geom_utils import find_prop_geom, set_rpm
from prep_vspaero import run_vspaero
from read_result import parse_results

# ── study output dir ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(_HERE, "pseudo_steady_convergence")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── reference (fixed) parameters for each sub-study ──────────────────────────
REF_TESS_U    = 27   # spanwise panels (odd)
REF_TESS_W    = 17    # chordwis panels  (odd) numW is twice the amount of pannels panels->lattice
REF_WAKE_ITER = 20
REF_WAKE_NODES = 24

MODE       = VSPAERO_PROP_PSEUDO_STEADY
AVG_LAST_N = None   # pseudo-steady: use last iteration only

CONV_TOL = 0.01     # 1 % relative convergence criterion

# ── parameter ranges ──────────────────────────────────────────────────────────
# Tessellation: (Tess_U, Tess_W) pairs – both odd, increasing linearly
TESS_PAIRS = [
    (5,  3),
    (10,  6),
    (19,  9),
    (27, 17),
    (38, 25),
]

WAKE_ITER_VALUES  = [2, 3, 5, 8, 12, 18, 27, 41]
WAKE_NODE_VALUES  = [4, 8, 16, 32, 64, 128]

METRICS       = ["CT", "CQ", "FOM"]
METRIC_LABELS = {"CT": r"$C_T$  (CFx)", "CQ": r"$C_Q$  (CMx)", "FOM": "FOM"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_base_geometry():
    """Clear and load the base VSP geometry from the repo root."""
    vsp_path = os.path.join(_ROOT, VSP_FILE)
    if not os.path.isfile(vsp_path):
        sys.exit(f"ERROR: '{vsp_path}' not found. Run from Design_code/.")
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(vsp_path)
    vsp.Update()
    return find_prop_geom()


def _set_tessellation(geom_id, tess_u, tess_w):
    """Set chordwise (U) and spanwise (W) panel counts on the propeller blade."""
    for parm_name, value in [("Tess_U", tess_u), ("Tess_W", tess_w)]:
        pid = vsp.GetParm(geom_id, parm_name, "Shape")
        if pid:
            vsp.SetParmVal(pid, float(value))
        else:
            print(f"  WARNING: parm '{parm_name}' not found on geom {geom_id}")
    vsp.Update()


def _run_case(case_label, sub_dir, tess_u, tess_w, wake_iter, wake_nodes):
    """Load geometry, set parameters, run VSPAERO, return parsed result dict."""
    case_dir = os.path.join(OUTPUT_DIR, sub_dir, case_label)
    os.makedirs(case_dir, exist_ok=True)
    case_vsp = os.path.join(case_dir, f"{case_label}.vsp3")

    geom_id = _load_base_geometry()
    set_rpm(RPM)
    _set_tessellation(geom_id, tess_u, tess_w)

    vsp.SetVSP3FileName(case_vsp)
    vsp.WriteVSPFile(case_vsp, vsp.SET_ALL)

    run_vspaero(
        omega=OMEGA, R=R, mode=MODE,
        rho=RHO, vref=VREF, mref=MREF, sref=SREF, bref=BREF,
        cref=CREF, Reref=RE_CREF,
        nwakenodes=wake_nodes, ncpu=NCPU,
        wakeiter=wake_iter, revs=NUM_REVS,
    )

    return parse_results(case_dir, case_label, avg_last_n=AVG_LAST_N)


def _relative_change(values):
    """Compute |xi - xi-1| / |xi| for each consecutive pair (returns n-1 values)."""
    arr = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = np.abs(np.diff(arr)) / np.abs(arr[1:])
    rc = np.where(np.isfinite(rc), rc, np.nan)
    return rc


# ── plot helpers ──────────────────────────────────────────────────────────────

def _save_convergence_plot(results_df, x_label, x_key,
                           title, filename, rel_change=True):
    """
    results_df: DataFrame with columns [x_key, CT, CQ, FOM]
    rel_change : also plot the relative change panel when True
    """
    n_rows = 2 if rel_change else 1
    fig, axes = plt.subplots(n_rows, len(METRICS),
                             figsize=(5 * len(METRICS), 4 * n_rows),
                             squeeze=False)
    fig.suptitle(title, fontsize=13)

    for j, m in enumerate(METRICS):
        ax = axes[0, j]
        ax.plot(results_df[x_key], results_df[m], "o-", color=f"C{j}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(METRIC_LABELS[m])
        ax.grid(True, alpha=0.4)

    if rel_change:
        for j, m in enumerate(METRICS):
            ax = axes[1, j]
            rc = _relative_change(results_df[m].values)
            x_rc = results_df[x_key].values[1:]
            ax.semilogy(x_rc, rc, "s--", color=f"C{j}")
            ax.axhline(CONV_TOL, color="red", ls=":", lw=1.2,
                       label=f"{CONV_TOL*100:.0f}% tolerance")
            ax.set_xlabel(x_label)
            ax.set_ylabel(r"$|x_i - x_{i-1}| / |x_i|$")
            ax.set_title(f"Relative change  —  {METRIC_LABELS[m]}")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ── Study 1: Tessellation ─────────────────────────────────────────────────────

def study_tessellation():
    print("\n" + "=" * 60)
    print("STUDY 1 — Tessellation convergence")
    print(f"  Wake iter  : {REF_WAKE_ITER}   Wake nodes: {REF_WAKE_NODES}")
    print("=" * 60)

    rows = []
    for tess_u, tess_w in TESS_PAIRS:
        label = f"tess_u{tess_u:02d}_w{tess_w:02d}"
        print(f"  [{label}]  Tess_U={tess_u}  Tess_W={tess_w}")
        res = _run_case(label, "tessellation",
                        tess_u, tess_w,
                        REF_WAKE_ITER, REF_WAKE_NODES)
        row = {"Tess_U": tess_u, "Tess_W": tess_w, "n_panels": tess_u * tess_w}
        for m in METRICS:
            row[m] = res.get(m)
        rows.append(row)
        print(f"    CT={row['CT']:.5f}  CQ={row['CQ']:.5f}  FOM={row['FOM']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "tess_convergence.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Data saved → {csv_path}")

    _save_convergence_plot(
        results_df=df,
        x_label="Number of panels  (Tess_U × Tess_W)",
        x_key="n_panels",
        title="Tessellation Convergence  (pseudo-steady)",
        filename="tess_convergence.png",
        rel_change=True,
    )
    return df


# ── Study 2: Wake iterations ──────────────────────────────────────────────────

def study_wake_iterations():
    print("\n" + "=" * 60)
    print("STUDY 2 — Wake iteration convergence")
    print(f"  Tess_U={REF_TESS_U}  Tess_W={REF_TESS_W}  Wake nodes: {REF_WAKE_NODES}")
    print("=" * 60)

    rows = []
    for n_iter in WAKE_ITER_VALUES:
        label = f"wakeiter_{n_iter:03d}"
        print(f"  [{label}]  WakeIter={n_iter}")
        res = _run_case(label, "wake_iterations",
                        REF_TESS_U, REF_TESS_W,
                        n_iter, REF_WAKE_NODES)
        row = {"wake_iter": n_iter}
        for m in METRICS:
            row[m] = res.get(m)
        rows.append(row)
        print(f"    CT={row['CT']:.5f}  CQ={row['CQ']:.5f}  FOM={row['FOM']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "wake_iter_convergence.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Data saved → {csv_path}")

    _save_convergence_plot(
        results_df=df,
        x_label="Number of wake iterations",
        x_key="wake_iter",
        title="Wake Iteration Convergence  (pseudo-steady)",
        filename="wake_iter_convergence.png",
        rel_change=True,
    )
    return df


# ── Study 3: Wake nodes ───────────────────────────────────────────────────────

def study_wake_nodes():
    print("\n" + "=" * 60)
    print("STUDY 3 — Trailing wake node convergence")
    print(f"  Tess_U={REF_TESS_U}  Tess_W={REF_TESS_W}  WakeIter: {REF_WAKE_ITER}")
    print("=" * 60)

    rows = []
    for n_nodes in WAKE_NODE_VALUES:
        label = f"wakenodes_{n_nodes:03d}"
        print(f"  [{label}]  WakeNodes={n_nodes}")
        res = _run_case(label, "wake_nodes",
                        REF_TESS_U, REF_TESS_W,
                        REF_WAKE_ITER, n_nodes)
        row = {"wake_nodes": n_nodes}
        for m in METRICS:
            row[m] = res.get(m)
        rows.append(row)
        print(f"    CT={row['CT']:.5f}  CQ={row['CQ']:.5f}  FOM={row['FOM']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "wake_nodes_convergence.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Data saved → {csv_path}")

    _save_convergence_plot(
        results_df=df,
        x_label="Number of trailing wake nodes",
        x_key="wake_nodes",
        title="Wake Node Convergence  (pseudo-steady)",
        filename="wake_nodes_convergence.png",
        rel_change=True,
    )
    return df


# ── summary plot ─────────────────────────────────────────────────────────────

def plot_summary(df_tess, df_wake_iter, df_wake_nodes):
    """One combined figure showing all three studies side-by-side for each metric."""
    fig, axes = plt.subplots(len(METRICS), 3,
                             figsize=(15, 4 * len(METRICS)),
                             squeeze=False)
    fig.suptitle("Pseudo-Steady Convergence Summary", fontsize=14)

    studies = [
        (df_tess,       "n_panels",    "Panels (U×W)"),
        (df_wake_iter,  "wake_iter",   "Wake iterations"),
        (df_wake_nodes, "wake_nodes",  "Wake nodes"),
    ]

    for i, m in enumerate(METRICS):
        for j, (df, x_key, xlabel) in enumerate(studies):
            ax = axes[i, j]
            ax.plot(df[x_key], df[m], "o-", color=f"C{j}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(METRIC_LABELS[m] if j == 0 else "")
            ax.grid(True, alpha=0.4)
            if i == 0:
                ax.set_title(xlabel)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "convergence_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSummary plot → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Pseudo-Steady Convergence Study")
    print(f"  RPM         : {RPM:.0f}")
    print(f"  Output dir  : {OUTPUT_DIR}")

    df_tess       = study_tessellation()
    df_wake_iter  = study_wake_iterations()
    df_wake_nodes = study_wake_nodes()
    plot_summary(df_tess, df_wake_iter, df_wake_nodes)

    print("\nAll done.")


if __name__ == "__main__":
    main()
