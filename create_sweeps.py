"""
Propeller Tangential Curve Sweep — Geometry + VSPAERO  VLM
==================================================================
For each amplitude step:
  1. Load base geometry
  2. Set tangential PCurve (index 8)
  3. Save modified .vsp3
  4. Set RPM on unsteady group 0
"""

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline

import openvsp as vsp
from bspline_util import cubicbez_to_bspline, sum_bsplines
from vspaero_config import *
from prop_utils import set_rpm, set_tangential_curve, find_prop_geom



def main(baseline_file, output_dir):
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


    amplitudes = np.linspace(-AMPLITUDE_FRAC, 0, N_STEPS, endpoint= False)  # Remove values very close to zero
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
                  f"pos={pos:.4f} (r_norm)", flush=True)


            # 1. fresh load every iteration — no state carried over
            vsp.ClearVSPModel()
            vsp.ReadVSPFile(VSP_FILE)
            geom_id = find_prop_geom()
            set_tangential_curve(geom_id, A_frac, pos)
            # 3. set RPM on unsteady group
            set_rpm(RPM)
            # 4. save geometry immediately after Update()
            vsp.SetVSP3FileName(case_vsp)
            vsp.Update()
            vsp.WriteVSPFile(case_vsp, vsp.SET_ALL)
            print(f"    Geometry saved → {case_vsp}", flush=True)


if __name__ == "__main__":
    main()
