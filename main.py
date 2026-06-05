"""
Propeller Tangential Curve Sweep — Geometry + VSPAERO  VLM
==================================================================
For each amplitude step:
  1. Load base geometry
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
from plots_2d import make_plots_2d
from vspaero_config import *
from prop_utils import set_rpm, set_tangential_curve, find_prop_geom
from read_result import parse_results
from prep_vspaero import run_vspaero


def main():
    if not os.path.isfile(VSP_FILE):
        sys.exit(f"ERROR: '{VSP_FILE}' not found.\n"
                 f"Run from your Design_code folder.")

    vsp.ClearVSPModel()
    vsp.ReadVSPFile(VSP_FILE)
    geom_id = find_prop_geom()
    # handle_mesh(geom_id, params=mesh_params)
    vsp.Update()
    rootstart = vsp.GetParmVal(vsp.FindParm(geom_id, "RadiusFrac", "XSec_0"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    amplitudes = np.linspace(-AMPLITUDE_FRAC, 0, N_STEPS, endpoint=False)  # Remove values very close to zero
    locations = np.linspace(rootstart+0.2, .8, N_STEPS)
    case_counter = 0
    total_cases = N_STEPS * N_STEPS

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

    for a_idx, A in enumerate(amplitudes):
        for p_idx, pos in enumerate(locations):
            case_counter += 1
            case_id = f"A{a_idx:02d}P{p_idx:02d}"
            A_frac  = A
            case_dir = os.path.join(OUTPUT_DIR, case_id)
            os.makedirs(case_dir, exist_ok=True)
            case_vsp = os.path.join(case_dir, f"{case_id}.vsp3")

            print(f"[{case_counter}/{total_cases}] "
                  f"A={A * 1000:+.2f}mm (A/R={A_frac:+.4f})  "
                  f"pos={pos:.4f} (r_norm)")


            # 1. fresh load every iteration — no state carried over
            vsp.ClearVSPModel()
            vsp.ReadVSPFile(VSP_FILE)
            geom_id = find_prop_geom()
            # handle_mesh(geom_id, params=mesh_params)


            # 2. set tangential PCurve
            set_tangential_curve(geom_id, A_frac, pos)

            # 3. set RPM on unsteady group
            set_rpm(RPM)

            # 4. save geometry immediately after Update()
            vsp.SetVSP3FileName(case_vsp)
            vsp.Update()
            vsp.WriteVSPFile(case_vsp, vsp.SET_ALL)
            print(f"    Geometry saved → {case_vsp}")

            # 5. run VSPAERO — set filename so output files go to case_dir
            rid = run_vspaero(omega=OMEGA, R=R, mode=ANALYSIS_MODE, rho=RHO,
                              vref=VREF, mref=MREF, sref=SREF, bref=BREF,
                              cref=CREF, Reref=RE_CREF, nwakenodes=NUM_WAKE_NODES,
                              ncpu=NCPU, wakeiter=WAKE_NUM_ITER, revs=NUM_REVS)

            # 6. Build row with metadata
            row = {
                'case_id': case_id,
                'amplitude_idx': a_idx,
                'position_idx': p_idx,
                'amplitude_m': round(float(A), 6),
                'amplitude_frac_R': round(float(A_frac), 6),
                'position_frac_R': round(float(pos), 6),
            }

            # parse from output files (more reliable than GetDoubleResults)
            res = parse_results(case_dir, case_id, avg_last_n=AVG_LAST_N)
            for k in ("CT_H", "CQ_H", "Thrust_total", "Moment_total", "FOM_total", "CT_h",
                      "CQ_h", "Thrust", "Moment", "FOM"):
                row[k] = res[k]

            if not rid:
                print("   ⚠ WARNING: ExecAnalysis returned no rid")

            def fmt(v, spec):
                return format(v, spec) if v is not None else "N/A"

            print(f"CT={fmt(res['CT_H'],'.5f')}  "
                  f"CQ={fmt(res['CQ_H'],'.5f')}  "
                  f"T={fmt(res['Thrust_total'],'.3f')} N  "
                  f"FOM={fmt(res['FOM_total'],'.4f')}")

            summary.append(row)

    # ── summary CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(summary)
    csv_path = os.path.join(OUTPUT_DIR, "sweep_2d_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary saved → {csv_path}")


    # ── Generate 2D plots ──────────────────────────────────────────────────
    make_plots_2d(
        df, amplitudes, locations,
        output_dir=OUTPUT_DIR,
        diameter=DIAMETER, rpm=RPM, vinf=VINF,
        amplitude_frac=AMPLITUDE_FRAC,
        r_root_frac=R_ROOT_FRAC,
        r_tip_frac=R_TIP_FRAC,
        n_steps=N_STEPS,
        radius=R
    )

    print(f"\n✓ All done. Results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
