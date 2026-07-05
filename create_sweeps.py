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
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline

import openvsp as vsp
from bspline_util import cubicbez_to_bspline, sum_bsplines
from vspaero_config import *
from prop_utils import set_rpm, set_tangential_curve, find_prop_geom, set_pcurve_pchip

# from phase_opt import phi #sweep function in polar cooridnates



# read original
# create output directory
# sequentially loop through the designs and apply sweep


def main(baseline_file, output_dir, max_sweeps:list[float], exponants:list[float]):

    def phi(r, k, max_sweep):
        return -(r**k)*max_sweep + max_sweep

    if not os.path.isfile(baseline_file):
        sys.exit(f"ERROR: '{baseline_file}' not found.\n"
                 f"Run from your Design_code folder.")

    vsp.ClearVSPModel()
    vsp.ReadVSPFile(baseline_file)
    geom_id = find_prop_geom()
    vsp.Update()
    # rootstart = vsp.GetParmVal(vsp.FindParm(geom_id, "RadiusFrac", "XSec_0"))

    # os.makedirs(output_dir, exist_ok=True)
    from pathlib import Path



    output_dir.mkdir(parents=True, exist_ok=True)

    case_counter = 0

    print("=" * 60)
    print("Propeller Tangential Curve Sweep")
    print("=" * 60)
    print(f"  File         : {baseline_file}")
    print(f"  R            : {R*1000:.1f} mm")
    print(f"  RPM          : {RPM:.0f}")
    print(f"  Vinf         : {VINF} m/s")
    print(f"  Rho          : {RHO} kg/m³")
    print(f"  ReCref       : {RE_CREF:.0f}")
    print(f"  Wake nodes   : {NUM_WAKE_NODES}")
    print(f"  Wake iters   : {WAKE_NUM_ITER}")
    print(f"  Num revs     : {NUM_REVS}")
    print(f"  NCPU         : {NCPU}")
    print(f"  Output       : {output_dir}/")
    print("=" * 60 + "\n")

    for sweep in max_sweeps:
        for k in exponants:
            case_counter += 1
            case_id = f"max_sweep{sweep * 180/np.pi:.4g}k{k:.4g}"
            case_dir = os.path.join(output_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)
            case_vsp = os.path.join(case_dir, f"{case_id}.vsp3")



            # 1. fresh load every iteration — no state carried over
            vsp.ClearVSPModel()
            vsp.ReadVSPFile(baseline_file)
            geom_id = find_prop_geom()

            # 2 create cos spaced samples
            # Cosine spacing between 0 and 1 (dense at both ends)
            N = 20
            i = np.linspace(0, np.pi, N)
            r_vec = 0.5 * (1 - np.cos(i))  # maps to [0, 1]
            angle = phi(r_vec, k, max_sweeps)
            param_vec = -r_vec*np.sin(angle) # - is backwards swept
            #3 set pchip interpolation points
            vsp.SetPCurve(geom_id, vsp.PROP_TANGENTIAL, r_vec, param_vec, vsp.PCHIP)
            vsp.Update()

            # 4. set RPM on unsteady group
            set_rpm(RPM)
            # 5. save geometry immediately after Update()
            vsp.SetVSP3FileName(case_vsp)
            vsp.Update()
            vsp.WriteVSPFile(case_vsp, vsp.SET_ALL)
            print(f"    Geometry saved → {case_vsp}", flush=True)

if __name__ == "__main__":
    # k = np.linspace(1.5, 2.7, 1, endpoint=True)
    k = [1.58]
    Max_sweeps = [.75]
    script_dir = Path(__file__).parent
    output_dir = script_dir / "75radvsp3files"
    base_file = str(Path(__file__).parent / "baseline" / "reverse_eng_TM_full_blade.vsp3")
    main(base_file,output_dir,Max_sweeps, k)
