"""
Propeller Tangential Curve Sweep — Geometry + VSPAERO  VLM
==================================================================
For each amplitude step:
  1. Load base geometry
  2. Set tangential PCurve (index 8) to a half-sine shape
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
            print("    Clearing...", flush=True)
            vsp.ClearVSPModel()
            print("    Reading...", flush=True)
            vsp.ReadVSPFile(VSP_FILE)
            print("    Finding geom...", flush=True)
            geom_id = find_prop_geom()
            # handle_mesh(geom_id, params=mesh_params)

            # 2. set tangential PCurve
            # print("    Setting curve...", flush=True)
            # valvec = vsp.PCurveGetValVec(geom_id, vsp.PROP_TANGENTIAL)
            # tvec = vsp.PCurveGetTVec(geom_id, vsp.PROP_TANGENTIAL)
            # bspline = cubicbez_to_bspline(tvec, valvec)
            # slope = A/((1-pos)*2/3)
            # cpsweep = np.array([
            #     [tvec[0], 0],
            #     [(pos-tvec[0])/3+tvec[0], 0],
            #     [(pos-tvec[0])*2/3+tvec[0], A],
            #     [pos, A],
            #     [(1-pos)/3 + pos, A],
            #     [(1-pos)*2/3 + pos, slope*((1-pos)/3)],
            #     [1, 0]
            # ])
            # knots_sweep = np.array([
            #     0, 0, 0, 0,
            #     pos,pos,pos,
            #     1, 1, 1, 1
            # ])
            # sweepspline = BSpline(knots_sweep, cpsweep, k=3)
            # combined = sum_bsplines(sweepspline, bspline)
            #
            # t = np.linspace(0, 1, 400)
            # eval_bspline = bspline(t)
            # eval_sweep = sweepspline(t)
            # eval_combined = combined(t)
            # plt.scatter(tvec, valvec, label="initial-cps")
            # plt.scatter(sweepspline.c[:,0], sweepspline.c[:,1], label="sweep")
            # plt.plot(eval_bspline[:,0], eval_bspline[:,1], label="bspline")
            # plt.plot(eval_sweep[:,0], eval_sweep[:,1], label="sweep")
            # plt.plot(eval_combined[:,0], eval_combined[:,1], label="combined")
            # plt.legend()
            # plt.show()

            set_tangential_curve(geom_id, A_frac, pos)
            print("    Tangential curve set", flush=True)

            # 3. set RPM on unsteady group
            set_rpm(RPM)
            print("    RPM set", flush=True)

            # 4. save geometry immediately after Update()
            vsp.SetVSP3FileName(case_vsp)
            vsp.Update()
            vsp.WriteVSPFile(case_vsp, vsp.SET_ALL)
            print(f"    Geometry saved → {case_vsp}", flush=True)


if __name__ == "__main__":
    main()
