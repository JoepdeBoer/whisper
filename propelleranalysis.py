"""
Propeller Tangential Curve Sweep — Geometry + VSPAERO Unsteady VLM
==================================================================
For each amplitude step:
  1. Load base geometry (nacapropeller-mod.vsp3)
  2. Set tangential PCurve (index 8) to a half-sine shape
  3. Save modified .vsp3
  4. Set RPM on unsteady group 0
  5. Run VSPAERO unsteady VLM
  6. Parse CT, CQ, Thrust, Torque, FOM, radial thrust distribution
  7. Save results + plots

Run from your Design_code folder:
  python3.13 prop_tangential_sweep.py
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openvsp as vsp

# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
VSP_FILE       = "nacapropeller-mod.vsp3"
OUTPUT_DIR     = "tangential_sweep_results"

# Sweep
N_STEPS        = 2          # number of amplitude steps (includes A=0)
AMPLITUDE_FRAC = 0.6       # max amplitude = AMPLITUDE_FRAC * R  (= R/4)
N_CURVE_PTS    = 11         # control points for the half-sine PCurve

# Propeller geometry
DIAMETER       = 0.508      # m
R              = DIAMETER / 2.0
R_ROOT_FRAC    = 0.2        # r/R at blade root
R_TIP_FRAC     = 1.0        # r/R at blade tip

# VSPAERO unsteady settings
RPM            = 5000.0
VINF           = 0.0    # m/s

# Atmos
RHO            = 1.225      # kg/m³
MU             = 1.4146e-5  # m^2/s

NUM_WAKE_NODES = 24
WAKE_NUM_ITER  = 10      # Only settable for Pseudo-steady, Why??
NCPU           = 8
NUM_REVS       = 3       # revolutions for auto timestep
AUTO_TIMESTEP  = True
THIN_GEOM_SET  =  2   # prop-only thin set as configured in GUI # shown, not shown set0
ANALYSIS_MODE  = vsp.VSPAERO_PROP_UNSTEADY # 0=steady, 1=unsteady, 2=pseudo-steady
# ANALYSIS_MODE  = vsp.VSPAERO_PROP_PSEUDO_STEADY #TODO debug seems that this does not effect choice steady/pseudo only from gui
ANALYSIS_BASE_NAME = "Hover_analysis"

# Reference values (derived)
OMEGA          = RPM * 2.0 * math.pi / 60.0
VREF           = OMEGA * R                  # blade tip speed  [m/s]
MREF           = VREF / 340.0              # tip Mach number
CREF           = 1                         # mean chord estimate [m]  TODO: improve
BREF           = R                          # reference span = R
SREF           = math.pi * R**2            # rotor disk area [m²]
RE_CREF        =  RHO / MU * OMEGA * DIAMETER                          # def from Ohad Gur tip RE pa/mu * Omega * R * D

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def find_prop_geom():
    for gid in vsp.FindGeoms():
        if vsp.GetGeomTypeName(gid) == "Propeller":
            return gid
    raise RuntimeError("No Propeller geom found in file.")

def find_set_by_name(name):
    """Return the set index matching the given name, or raise if not found."""
    for i in range(vsp.GetNumSets()):
        if vsp.GetSetName(i) == name:
            return i
    raise RuntimeError(f"No VSP set named '{name}' found.")


def set_tangential_curve(geom_id, amplitude):
    """
    Set PCurve 8 (tangential) to a half-sine with given amplitude (m).
      tan(r) = A * sin( pi * (r/R - r_root) / (r_tip - r_root) )
    Zero at root and tip, peak at mid-span.
    """
    tvec   = np.linspace(R_ROOT_FRAC, R_TIP_FRAC, N_CURVE_PTS)
    valvec = amplitude * np.sin(
                np.pi * (tvec - R_ROOT_FRAC) / (R_TIP_FRAC - R_ROOT_FRAC))
    vsp.SetPCurve(geom_id, vsp.PROP_TANGENTIAL,
                  tvec.tolist(), valvec.tolist(),
                  vsp.PCHIP)
    vsp.Update()


def set_rpm(rpm):
    """Set RPM on unsteady group 0."""
    group_id = vsp.FindUnsteadyGroup(0)
    if not group_id:
        print("  WARNING: could not find unsteady group 0")
        return
    pid = vsp.FindParm(group_id, "RPM", "UnsteadyGroup")
    if pid:
        vsp.SetParmVal(pid, rpm)
    else:
        print("  WARNING: RPM parm not found in unsteady group")


# ─────────────────────────────────────────────────────────────────────────────
# VSPAERO
# ─────────────────────────────────────────────────────────────────────────────

def run_vspaero():
    """Run VSPAERO unsteady VLM. Returns results_id or None."""

    # BUG UnsteadyType does not change analysis mode correctly:
    def set_prop_blades_mode(mode):
        cid = vsp.FindContainer("VSPAEROSettings", 0)
        pid = vsp.FindParm(cid, "m_PropBladesMode", "VSPAERO")
        if pid:
            vsp.SetParmVal(pid, float(mode))
            vsp.Update()
            print(f"  Set m_PropBladesMode={mode}")
        else:
            print("  WARNING: m_PropBladesMode not found")
    set_prop_blades_mode(ANALYSIS_MODE)

    cg = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(cg)
    vsp.SetIntAnalysisInput(cg, "GeomSet", [vsp.SET_NONE]) #TODO test
    vsp.SetIntAnalysisInput(cg, "ThinGeomSet", [vsp.SET_ALL])
    if not vsp.ExecAnalysis(cg):
        print("    ERROR: VSPAEROComputeGeometry failed")
        return None

    sw = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(sw)

    mach =  OMEGA * R * 0.7 / 340 #TODO is this right

    vsp.SetIntAnalysisInput   (sw, "UnsteadyType",      [ANALYSIS_MODE])
    vsp.SetIntAnalysisInput   (sw, "GeomSet",           [vsp.SET_NONE])
    vsp.SetIntAnalysisInput   (sw, "ThinGeomSet",       [vsp.SET_ALL])

    # Flight condition
    vsp.SetDoubleAnalysisInput(sw, "Vinf",              [0.])
    vsp.SetDoubleAnalysisInput(sw, "Rho",               [RHO])
    vsp.SetDoubleAnalysisInput(sw, "AlphaStart",        [0.])
    vsp.SetDoubleAnalysisInput(sw, "AlphaEnd",          [0.])
    vsp.SetIntAnalysisInput   (sw, "AlphaNpts",         [1])
    vsp.SetDoubleAnalysisInput(sw, "MachStart",         [mach])
    vsp.SetDoubleAnalysisInput(sw, "MachEnd",           [mach])
    vsp.SetIntAnalysisInput   (sw, "MachNpts",          [1])
    vsp.SetDoubleAnalysisInput(sw, "ReCref",            [RE_CREF])

    # Reference values
    vsp.SetIntAnalysisInput   (sw, "ManualVrefFlag",    [True])
    vsp.SetDoubleAnalysisInput(sw, "Vref",              [VREF])
    vsp.SetDoubleAnalysisInput(sw, "Machref",           [MREF])
    vsp.SetDoubleAnalysisInput(sw, "Sref",              [SREF])
    vsp.SetDoubleAnalysisInput(sw, "bref",              [BREF])
    vsp.SetDoubleAnalysisInput(sw, "cref",              [CREF])

    # Wake
    vsp.SetIntAnalysisInput   (sw, "NumWakeNodes",      [NUM_WAKE_NODES])
    vsp.SetIntAnalysisInput   (sw, "WakeNumIter",       [WAKE_NUM_ITER]) # only used in pseudo steady
    vsp.SetIntAnalysisInput   (sw, "FixedWakeFlag",     [False]) # TODO checkl if bool instead of int is okay

    # Time stepping
    vsp.SetIntAnalysisInput   (sw, "AutoTimeStepFlag",  [int(AUTO_TIMESTEP)])
    vsp.SetIntAnalysisInput   (sw, "AutoTimeNumRevs",   [int(NUM_REVS)])

    # CPU
    vsp.SetIntAnalysisInput   (sw, "NCPU",              [NCPU])

    vsp.Update()
    rid = vsp.ExecAnalysis(sw)

    if not rid:
        print("    ERROR: VSPAEROSweep failed")
    return rid


def parse_rotor(rotor_file):
    result = dict(r_norm=[], r_m=[], dCT_dR=[])
    if not os.path.isfile(rotor_file):
        return result
    try:
        df = pd.read_csv(rotor_file, sep=r"\s+", skiprows=1, engine="python")
        df.columns = df.columns.str.strip()
        row = df.iloc[-1]
        result["CT"]       = float(row["CT"])
        result["CQ"]       = float(row["CQ"])
        result["FOM"]      = float(row["FOM"])
        result["Thrust_N"]  = float(row["Thrust"])
        result["Torque_Nm"] = float(row["Moment"])
    except Exception as e:
        print(f"    WARNING: could not parse rotor file: {e}")
    return result


def parse_results(case_dir, case_name):
    """Parse output files from a completed VSPAERO case."""
    rotor_file = os.path.join(case_dir, f"{case_name}.rotor.1")
    rot = parse_rotor(rotor_file)
    return rot


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(VSP_FILE):
        sys.exit(f"ERROR: '{VSP_FILE}' not found.\n"
                 f"Run from your Design_code folder.")

    amplitudes = np.linspace(0.0, AMPLITUDE_FRAC * R, N_STEPS)

    print("=" * 60)
    print("Propeller Tangential Curve Sweep")
    print("=" * 60)
    print(f"  File         : {VSP_FILE}")
    print(f"  R            : {R*1000:.1f} mm")
    print(f"  Max amplitude: {AMPLITUDE_FRAC*R*1000:.1f} mm  (R/{int(1/AMPLITUDE_FRAC)})")
    print(f"  Steps        : {N_STEPS}")
    print(f"  RPM          : {RPM:.0f}")
    print(f"  Vinf         : {VINF} m/s")
    print(f"  Rho          : {RHO} kg/m³")
    print(f"  ReCref       : {RE_CREF:.0f}")
    print(f"  Wake nodes   : {NUM_WAKE_NODES}")
    print(f"  Wake iters   : {WAKE_NUM_ITER}")
    print(f"  Num revs     : {NUM_REVS}")
    print(f"  NCPU         : {NCPU}")
    print(f"  Output       : {OUTPUT_DIR}/")
    print("=" * 60 + "\n")

    summary = []

    for idx, A in enumerate(amplitudes):
        A_frac   = A / R
        label    = f"A{idx:02d}_amp{round(A*1000)}mm"
        case_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(case_dir, exist_ok=True)
        case_vsp = os.path.join(case_dir, f"{label}.vsp3")

        print(f"[{idx+1}/{N_STEPS}]  A = {A*1000:.2f} mm  (A/R = {A_frac:.3f})")

        # 1. fresh load every iteration — no state carried over
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(VSP_FILE)
        vsp.Update()
        geom_id = find_prop_geom()

        # 2. set tangential PCurve
        set_tangential_curve(geom_id, A)

        # 3. set RPM on unsteady group
        set_rpm(RPM)

        # 4. save geometry immediately after Update()
        vsp.SetVSP3FileName(case_vsp)
        vsp.WriteVSPFile(case_vsp, vsp.SET_ALL)
        print(f"    Geometry saved → {case_vsp}")

        # 5. run VSPAERO — set filename so output files go to case_dir
        rid = run_vspaero()
        row = dict(
            step            = idx,
            amplitude_m     = round(float(A), 6),
            amplitude_frac_R= round(float(A_frac), 6),
            CT=None, CQ=None, Thrust_N=None, Torque_Nm=None, FOM=None
        )

        # parse from output files (more reliable than GetDoubleResults)
        res = parse_results(case_dir, label)
        for k in ("CT", "CQ", "Thrust_N", "Torque_Nm", "FOM"):
            row[k] = res[k]

        if not rid:
            print("    WARNING: ExecAnalysis returned no rid")

        def fmt(v, spec): return format(v, spec) if v is not None else "N/A"
        print(f"CT={fmt(res['CT'],'.5f')}  "
              f"CQ={fmt(res['CQ'],'.5f')}  "
              f"T={fmt(res['Thrust_N'],'.3f')} N  "
              f"FOM={fmt(res['FOM'],'.4f')}")

        if res.get("r_norm"):
            pd.DataFrame({
                "r_norm": res["r_norm"],
                "r_m"   : res["r_m"],
                "dCT_dR": res["dCT_dR"],
            }).to_csv(os.path.join(case_dir, "thrust_distribution.csv"),
                      index=False)

        summary.append(row)
        print()

    # ── summary CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(summary)
    csv_path = os.path.join(OUTPUT_DIR, "sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary → {csv_path}")
    print(df.to_string(index=False))

    _make_plots(df, amplitudes)
    print(f"\nAll done. Results in: {OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _make_plots(df, amplitudes):
    valid = df["CT"].notna()
    af    = df["amplitude_frac_R"].values

    # 1. 2×2 performance summary
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Propeller Tangential Sweep  (half-sine, 0 → R/4)\n"
        f"D={DIAMETER*1000:.0f} mm  |  {RPM:.0f} RPM  |  "
        f"Vinf={VINF} m/s  |  Unsteady VLM",
        fontsize=11)
    for ax, col, title, color, fmt in [
        (axes[0,0], "CT",       "Thrust Coefficient CT", "steelblue",  "o-"),
        (axes[0,1], "CQ",       "Torque Coefficient CQ", "darkorange", "s-"),
        (axes[1,0], "FOM",      "Figure of Merit",       "seagreen",   "^-"),
        (axes[1,1], "Thrust_N", "Thrust  [N]",           "crimson",    "D-"),
    ]:
        if valid.any():
            ax.plot(af[valid], df[col][valid], fmt, color=color, ms=7, lw=1.8)
        ax.set_xlabel("Amplitude / R")
        ax.set_ylabel(col.replace("_N", " [N]"))
        ax.set_title(title)
        ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "sweep_summary.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"Plot → {p}")

    # 2. radial thrust distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis
    for _, row in df[valid].iterrows():
        f = os.path.join(OUTPUT_DIR,
                f"A{int(row['step']):02d}_amp{row['amplitude_m']*1000:.1f}mm",
                "thrust_distribution.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            c = cmap(row["amplitude_frac_R"] / (AMPLITUDE_FRAC + 1e-9))
            ax.plot(d["r_norm"], d["dCT_dR"], color=c,
                    label=f"A/R={row['amplitude_frac_R']:.2f}")
    ax.set_xlabel("r / R")
    ax.set_ylabel("dCT / dR")
    ax.set_title("Radial Thrust Distribution — tangential amplitude sweep")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "thrust_distribution_sweep.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"Plot → {p}")

    # 3. tangential curve shapes
    r_fracs = np.linspace(R_ROOT_FRAC, R_TIP_FRAC, 200)
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap2 = plt.cm.plasma
    for i, A in enumerate(amplitudes):
        c    = cmap2(i / max(N_STEPS - 1, 1))
        offs = A * np.sin(np.pi * (r_fracs - R_ROOT_FRAC) /
                          (R_TIP_FRAC - R_ROOT_FRAC))
        ax.plot(r_fracs, offs * 1000, color=c,
                label=f"A={A*1000:.1f}mm  (A/R={A/R:.2f})")
    ax.set_xlabel("r / R")
    ax.set_ylabel("Tangential offset  [mm]")
    ax.set_title("Tangential Curve Shapes  (half-sine, root & tip = 0)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "tangential_shapes.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"Plot → {p}")


if __name__ == "__main__":
    main()
