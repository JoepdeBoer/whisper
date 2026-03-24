"""
Propeller Tangential Curve Sweep — Geometry + VSPAERO  VLM
==================================================================
For each amplitude step:
  1. Load base geometry (nacapropeller-mod.vsp3)
  2. Set tangential PCurve (index 8) to a half-sine shape
  3. Save modified .vsp3
  4. Set RPM on unsteady group 0
  5. Run VSPAERO un/pseudo-steady VLM
  6. Parse CT, CQ, Thrust, Torque, FOM, radial thrust distribution
  7. Save results + plots
"""

import os
import sys
import numpy as np
import pandas as pd
import openvsp as vsp
from vspaero_config import *
from plots import make_plots
from geom_utils import find_prop_geom, set_rpm, set_tangential_curve
from read_result import parse_results
from prep_vspaero import run_vspaero


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(VSP_FILE):
        sys.exit(f"ERROR: '{VSP_FILE}' not found.\n"
                 f"Run from your Design_code folder.")

    amplitudes = np.linspace(-AMPLITUDE_FRAC*R, AMPLITUDE_FRAC * R, N_STEPS)

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
        rid = run_vspaero(omega=OMEGA, R=R, mode=ANALYSIS_MODE, rho=RHO,
                          vref=VREF, mref=MREF, sref=SREF, bref=BREF,
                          cref=CREF, Reref=RE_CREF, nwakenodes=NUM_WAKE_NODES,
                          ncpu=NCPU, wakeiter=WAKE_NUM_ITER, revs=NUM_REVS)
        row = dict(
            step            = idx,
            amplitude_m     = round(float(A), 6),
            amplitude_frac_R= round(float(A_frac), 6),
            # CT_=None, CQ=None, Thrust_N=None, Torque_Nm=None, FOM=None
        )

        # parse from output files (more reliable than GetDoubleResults)
        res = parse_results(case_dir, label, avg_last_n=AVG_LAST_N)
        for k in ("CT_H", "CQ_H", "Thrust_total", "Moment_total", "FOM_total", "CT_h", "CQ_h", "Thrust", "Moment", "FOM"):
            row[k] = res[k]

        if not rid:
            print("    WARNING: ExecAnalysis returned no rid")

        def fmt(v, spec): return format(v, spec) if v is not None else "N/A"
        print(f"CT={fmt(res['CT_H'],'.5f')}  "
              f"CQ={fmt(res['CQ_H'],'.5f')}  "
              f"T={fmt(res['Thrust_total'],'.3f')} N  "
              f"FOM={fmt(res['FOM_total'],'.4f')}")

        # if res.get("r_norm"):
        #     pd.DataFrame({
        #         "r_norm": res["r_norm"],
        #         "CT_h": res["CT_h"],
        #         "CQ_h": res["CQ_h"],
        #         "Thrust" : res["Thrust"],
        #
        #     }).to_csv(os.path.join(case_dir, "radial_distribution.csv"),
        #               index=False)

        summary.append(row)

    # ── summary CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(summary)
    csv_path = os.path.join(OUTPUT_DIR, "sweep_summary.csv")
    df.to_csv(csv_path, index=False)

    make_plots(df, amplitudes,
               output_dir=OUTPUT_DIR, diameter=DIAMETER, rpm=RPM, vinf=VINF,
               amplitude_frac=AMPLITUDE_FRAC, r_root_frac=R_ROOT_FRAC,
               r_tip_frac=R_TIP_FRAC, n_steps=N_STEPS, radius=R)
    print(f"\nAll done. Results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
