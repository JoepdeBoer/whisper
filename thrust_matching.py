"""
thrust_matching.py
==================
Iteratively adjust the twist distribution of a propeller to match the
radial thrust distribution of a baseline run.

The source .vsp3 must already have the desired tangential and CLi curves
applied.  Each iteration the twist PCHIP interpolation knots are updated,
VSPAERO is rerun, and the resulting radial thrust is compared to the
baseline .lod distribution.

Typical call from main.py:

    from thrust_matching import match_radial_thrust_twist
    from baseline import mesh_params
    from vspaero_config import *

    result = match_radial_thrust_twist(
        baseline_lod_path="baseline/baseline.lod",
        vsp_source_file=case_vsp,          # already has tangential curve
        case_dir=case_dir,
        case_id=case_id,
        initial_twist_pts=initial_twist_pts,  # (N,2) array (r, deg)
        rpm=RPM,
        avg_last_n=AVG_LAST_N,
        max_iter=10,
        tol=0.03,
        gain=0.5,
        omega=OMEGA, R=R, mode=ANALYSIS_MODE, rho=RHO,
        vref=VREF, mref=MREF, sref=SREF, bref=BREF,
        cref=CREF, Reref=RE_CREF, nwakenodes=NUM_WAKE_NODES,
        ncpu=NCPU, wakeiter=WAKE_NUM_ITER, revs=NUM_REVS,
    )
    final_twist_pts = result["twist_pts"]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx.algorithms.bipartite.basic import color
from scipy.interpolate import interp1d, CubicSpline

import openvsp as vsp

from prop_utils import find_prop_geom, set_rpm, set_pcurve_bezier
from read_result import parse_lod
from prep_vspaero import run_vspaero as _run_vspaero

_VSPAERO_KEYS = frozenset({
    "omega", "R", "mode", "rho", "vref", "mref", "sref", "bref",
    "cref", "Reref", "nwakenodes", "ncpu", "wakeiter", "revs",
})


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_thrust_profile(lod_path: str, avg_last_n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_norm, thrust) arrays from a .lod file, sorted by r."""
    res = parse_lod(lod_path, avg_last_n=avg_last_n)
    r = np.asarray(res["r_norm"], dtype=float)
    t = np.asarray(res["Thrust"], dtype=float)
    return r, t


def _to_grid(r_src: np.ndarray, vals: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate vals(r_src) onto r_grid; zero outside domain."""
    f = interp1d(r_src, vals, kind="linear", bounds_error=False, fill_value=0.0)
    return f(r_grid)





def _update_twist_bezier(
        twist_pts_cur: np.ndarray[float],
        # (N,2) array r/R , twist[deg]
        r_grid: np.ndarray,
        # fine radial grid on which thrust is evaluated
        t_ref: np.ndarray,
        # target (reference) thrust distribution on r_grid
        t_cur: np.ndarray,
        # current iteration's thrust distribution on r_grid
        gain: float,
        # relaxation factor: 1.0 = full step, <1 = damped
        twist_min: float,
        # hard lower bound on twist [deg]
        twist_max: float,
        # hard upper bound on twist [deg]
        thr_frac: float = 0.01,
        # stations below this fraction of peak thrust are frozen
        clamping: tuple[float] = (.5, 1.5),
        # bounds on relative twist update
        n_eval: int = 100,
        # sample points per segment when evaluating the Bezier
) -> np.ndarray[tuple[float, float]]:
    """
    Return (r_pts, y_new): same r positions, updated twist values.

    The VSP cubic Bezier format is:
        knot, cp_right, cp_left, knot, cp_right, cp_left, knot, ...
    with total length 3N+1 for N segments.

    The first and last knots are never modified.  Interior knots are scaled by
    the same thrust-ratio logic as _update_twist.  Control points are then
    recomputed so that dy/dr is C1-continuous at every interior knot
    (one shared slope from both adjacent segments), which satisfies G1.
    """
    # ------------------------------------------------------------------ #
    # 0.  Interpolate thrust over full domain   #
    # ------------------------------------------------------------------ #
    r0 = twist_pts_cur[0,0]
    extended_ref = [0., *t_ref, 0.]
    extended_curr = [0., *t_cur, 0.]
    r_extended = [r0, *r_grid, 1.]
    # TODO check if extended end with near 0 thrust as well

    t_ref_spline = CubicSpline(r_extended, extended_ref, bc_type='not-a-knot', extrapolate= False)
    t_curr_spline = CubicSpline(r_extended, extended_curr, bc_type="not-a-knot", extrapolate=False)
    

    # ------------------------------------------------------------------ #
    # 1.  Parse the flat VSP arrays into knot and control-point indices   #
    # ------------------------------------------------------------------ #
    r_pts = twist_pts_cur[:, 0]
    y_pts = twist_pts_cur[:, 1]

    # (3N+1,) r/R positions: knot, cp, cp, knot, cp, cp, ...
    n_pts = len(r_pts)
    print(f"optimisation can alter {n_pts} points")
    assert (n_pts - 1) % 3 == 0, "r_pts length must be 3N+1 for N Bezier segments"
    n_seg = (n_pts - 1) // 3  # number of cubic Bezier segments

    # Knots sit at every third position: 0, 3, 6, ..., 3N
    knot_idx = np.arange(0, n_pts,
                         3)  # shape (N+1,)
    r_knots = r_pts[knot_idx]
    y_knots = y_pts[knot_idx]

    # ------------------------------------------------------------------ #
    # 2.  Evaluate the current Bezier curve on the fine r_grid            #
    #                                                                     #
    # Strategy: sample each segment at n_eval parameter values t∈[0,1]   #
    # to get dense (r, y) pairs, then interpolate onto r_grid.           #
    # This avoids inverting the cubic B(t)=r, which has no closed form.  #
    # ------------------------------------------------------------------ #
    r_dense_list, y_dense_list = [], []

    for i in range(n_seg):
        base = 3 * i
        # The four control points of segment i in (r, y) space
        P0 = np.array([r_pts[base], y_pts[
            base]])  # knot i
        P1 = np.array([r_pts[base + 1], y_pts[
            base + 1]])  # right cp of knot i
        P2 = np.array([r_pts[base + 2], y_pts[
            base + 2]])  # left  cp of knot i+1
        P3 = np.array([r_pts[base + 3], y_pts[
            base + 3]])  # knot i+1

        t = np.linspace(0.0, 1.0,
                        n_eval, endpoint= False)  # parameter values
        # Standard cubic Bezier formula B(t) = Σ Bernstein_k(t) * P_k
        pts = (np.outer((1 - t) ** 3, P0)
               + np.outer(3 * (1 - t) ** 2 * t, P1)
               + np.outer(3 * (1 - t) * t ** 2, P2)
               + np.outer(t ** 3,
                          P3))  # (n_eval, 2)

        r_dense_list.append(pts[:, 0])
        y_dense_list.append(pts[:, 1])

    r_dense = np.concatenate(r_dense_list)
    y_dense = np.concatenate(y_dense_list)


    # Sort by r (should already be monotone, but guards against floating-point
    # order issues at segment boundaries where adjacent segments share a knot)
    #TODO remove?
    # order = np.argsort(r_dense)
    # r_dense = r_dense[order]
    # y_dense = y_dense[order]


    # ------------------------------------------------------------------ #
    # 3.  Compute thrust scale factor   #
    # ------------------------------------------------------------------ #
    # # At active stations: scale = t_cur/t_ref - 1
    # #   < 0  →  blade produces too little thrust → twist should increase
    # #   > 0  →  blade produces too much  thrust → twist should decrease
    ref_vals = t_ref_spline(r_dense)
    curr_vals = t_curr_spline(r_dense)
    # At hub (r=r0) and tip (r=1) the reference thrust is 0; skip correction there.
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(ref_vals > thr_frac*ref_vals.max(), curr_vals / ref_vals - 1, 0.0)

    # ------------------------------------------------------------------ #
    # 4.  Apply the proportional correction on the fine grid              #
    # ------------------------------------------------------------------ #
    # gain < 1 under-relaxes the step, trading convergence speed for
    # stability (analogous to a damping factor in Newton iterations)

    twist_new = np.clip(y_dense * (1 -(scale * gain)), twist_min, twist_max)





    # ------------------------------------------------------------------ #
    # 5.  Project corrected twist back to the Bezier knot r-positions     #
    # ------------------------------------------------------------------ #
    f_corrected = CubicSpline(r_dense, twist_new, bc_type="natural", extrapolate=True)

    y_knots_new = f_corrected(
        r_knots)  # new twist value at each knot


    # ------------------------------------------------------------------ #
    # 6.  Estimate the twist slope dy/dr at every knot                   #
    # ------------------------------------------------------------------ #
    slopes = f_corrected.derivative()(r_knots)

    # ------------------------------------------------------------------ #
    # 7.  Reconstruct all control-point y-values enforcing G1 continuity  #
    # TODO refactor duplicate code
    #                                                                     #
    # G1 at knot k: the tangent direction must be continuous.             #
    # Since the r positions of control points are fixed by VSP, G1 is    #
    # equivalent to C1 in the r→y functional sense:                      #
    #                                                                     #
    #   slope_k = (y_k - y_cp_left)  / Δr_left                          #
    #           = (y_cp_right - y_k) / Δr_right                          #
    #                                                                     #
    # Rearranging:                                                        #
    #   y_cp_right = y_k + slope_k * Δr_right                            #
    #   y_cp_left  = y_k - slope_k * Δr_left                             #
    # ------------------------------------------------------------------ #
    y_new = y_pts.copy()  # start from the original array; only y changes

    for k in range(
            n_seg + 1):  # iterate over every knot
        ki = knot_idx[
            k]  # position of knot k in the flat array
        slope_k = slopes[k]
        y_k = y_knots_new[k]

        # Update the knot value itself (boundary knots already fixed above)
        if 0 < k < n_seg:
            y_new[ki] = y_k

            # Right control point of knot k  (exists for all but the last knot)
        if k < n_seg:
            cp_r = ki + 1  # index in flat array
            dr_r = r_pts[cp_r] - r_pts[
                ki]  # Δr to the right cp
            y_new[cp_r] = y_k + slope_k * dr_r

            # Left control point of knot k  (exists for all but the first knot)
        if k > 0:
            cp_l = ki - 1  # index in flat array
            dr_l = r_pts[ki] - r_pts[cp_l]  # Δr to the left cp
            y_new[cp_l] = y_k - slope_k * dr_l

            # r positions never change — only return a copy to make the API explicit
    res = np.column_stack((r_pts, y_new))

    # Debug plotting
    # fig, ax, = plt.subplots(ncols=2)
    # ax1 = ax[0]
    # ax3 = ax[1]
    #
    # # Left y-axis: Thrust
    # ax1.plot(r_extended, extended_ref, label="Thrust Old", linestyle="--", color="blue")
    # ax1.plot(r_extended, extended_curr, label="Thrust New", linestyle="-", color="blue")
    # ax1.set_xlabel("r")
    # ax1.set_ylabel("Thrust")
    # ax1.tick_params(axis="y")
    #
    # # Right y-axis: Twist
    # ax2 = ax1.twinx()
    # ax2.plot(r_dense, y_dense , label="Twist Old", linestyle="--", color = 'r')
    # ax2.plot(r_dense, twist_new, label="Twist New", linestyle="-", color = 'r')
    # ax2.set_ylabel("Twist [deg]")
    # ax2.tick_params(axis="y")
    #
    # # Combine legends from both axes
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    #
    #
    #
    # # Knot positions
    # ax3.scatter(r_knots, y_knots_new, color="tab:orange", zorder=5, label="knots")
    # ax3.scatter(r_pts, y_new, color="tab:blue", label="res")
    # ax3.plot(r_dense, f_corrected(r_dense), color="tab:pink", label="cubic")
    # ax3.plot(r_dense, f_corrected2(r_dense), color="black", label="lin")
    #
    # # Tangent lines at each knot
    # tangent_half_width = (r_knots[-1] - r_knots[0]) / (len(r_knots) * 3)
    # for r_k, y_k, s in zip(r_knots, y_knots_new, slopes):
    #     r_tan = np.array([r_k - tangent_half_width, r_k + tangent_half_width])
    #     y_tan = y_k + s * (r_tan - r_k)
    #     ax3.plot(r_tan, y_tan, color="tab:red", linewidth=1.5)
    #
    # ax3.set_xlabel("r/R")
    # ax3.set_ylabel("Twist [deg]")
    # ax3.set_title("Corrected twist spline with knot slopes")
    # ax3.legend()
    #
    # plt.title("Thrust and Twist Comparison")
    # plt.grid(True)
    # plt.show()

    return res


def _plot_thrust_convergence(
    r_grid: np.ndarray,
    t_ref_grid: np.ndarray,
    thrust_history: list[np.ndarray],
    rms_history: list[float],
    save_path: str | None = None,
) -> None:
    """
    Two-panel convergence figure.

    Top panel   – radial thrust distribution at every iteration plus the
                  baseline target.  Iterations are coloured from light (early)
                  to dark (late) so you can see which direction the curve
                  moves and whether it overshoots.

    Bottom panel – RMS relative thrust error vs iteration number, useful for
                   judging whether gain / clamping cause oscillations or slow
                   monotone convergence.
    """
    n_iter = len(thrust_history)
    # Colour map: early iterations = light blue, late = dark blue
    colours = cm.Blues(np.linspace(0.3, 0.95, max(n_iter, 1)))

    fig, (ax_thrust, ax_rms) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}
    )

    # ── Top: thrust profiles ──────────────────────────────────────────────
    for i, t_iter in enumerate(thrust_history):
        ax_thrust.plot(r_grid, t_iter, color=colours[i],
                       linewidth=1.2, label=f"iter {i}")

    # Reference on top so it is always visible
    ax_thrust.plot(r_grid, t_ref_grid, color="crimson", linewidth=2.0,
                   linestyle="--", label="baseline target")

    ax_thrust.set_xlabel("r/R  [–]")
    ax_thrust.set_ylabel("Thrust per unit span  [N/m]")
    ax_thrust.set_title("Thrust distribution convergence")
    ax_thrust.legend(fontsize=7, ncol=2, loc="upper left")
    ax_thrust.grid(True, alpha=0.3)

    # ── Bottom: RMS error ─────────────────────────────────────────────────
    ax_rms.plot(range(1, n_iter + 1), rms_history,
                marker="o", color="steelblue", linewidth=1.5)
    ax_rms.set_xlabel("Iteration  [–]")
    ax_rms.set_ylabel("RMS relative error  [–]")
    ax_rms.set_title("Convergence history")
    ax_rms.set_xticks(range(1, n_iter + 1))
    ax_rms.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  [twist match] convergence plot saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _rms_error(t_ref: np.ndarray, t_cur: np.ndarray, thr_frac: float = 0.05) -> float:
    """RMS relative thrust error over stations with significant baseline thrust."""
    t_max = np.max(np.abs(t_ref))
    mask = np.abs(t_ref) > thr_frac * t_max
    if not np.any(mask):
        return float("inf")
    rel = (t_cur[mask] - t_ref[mask]) / (np.abs(t_ref[mask]) + 1e-12)
    return float(np.sqrt(np.mean(rel ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def match_radial_thrust_twist(
        baseline_lod_path: str,
        vsp_source_file: str,
        case_dir: str,
        case_id: str,
        initial_twist_pts: np.ndarray,
        rpm: float,
        avg_last_n: int = 1,
        max_iter: int = 20,
        tol: float = 0.03,
        gain: float = 0.5,
        twist_min: float = 5,
        twist_max: float = 40.,
        **vspaero_kwargs: object,
) -> dict:
    """

    Iteratively match the radial thrust distribution of a modified propeller
    to the baseline by adjusting the twist PCHIP interpolation knots.

    Parameters
    ----------
    baseline_lod_path : str
        Path to the baseline .lod file (e.g. "baseline/baseline.lod").
    vsp_source_file : str
        .vsp3 to reload each iteration; must already contain the desired
        tangential and CLi curves (and any other geometry changes).
    case_dir : str
        Output directory; one sub-file per iteration is written here.
    case_id : str
        Filename prefix used for iteration outputs (e.g. "A01P02").
    initial_twist_pts : ndarray, shape (N, 2)
        Starting PCHIP interpolation knots as (r/R, degrees) rows.
    rpm : float
        Rotor RPM.
    avg_last_n : int
        Number of final timesteps to average when reading .lod files.
    max_iter : int
        Maximum twist update iterations.
    tol : float
        Convergence threshold on RMS relative thrust error.
    gain : float
        Update step size (0 < gain <= 1); lower = more stable, slower.
    twist_min, twist_max : float
        Hard bounds on twist values (degrees) after each update.
    **vspaero_kwargs
        Forwarded verbatim to run_vspaero() (omega, R, mode, rho, …).

    Returns
    -------
    dict with keys:
        "twist_pts"      – final (N, 2) bezier spline knots
        "converged"      – bool
        "n_iter"         – iterations performed
        "rms_history"    – list[float], one entry per iteration
        "twist_history"  – list[ndarray], one entry per iteration
    """
    os.makedirs(case_dir, exist_ok=True)

    vspaero_args = {k: v for k, v in vspaero_kwargs.items() if k in _VSPAERO_KEYS}

    r_base, t_base = _load_thrust_profile(baseline_lod_path, avg_last_n=avg_last_n)
    r_grid = np.linspace(r_base[0], r_base[-1], 300)
    t_base_grid = _to_grid(r_base, t_base, r_grid)

    twist_pts = initial_twist_pts.copy().astype(float)
    rms_history: list[float] = []
    twist_history: list[np.ndarray] = []
    thrust_history: list[np.ndarray] = []   # t_cur_grid per iteration, for plotting
    converged = False

    print(f"  [twist match] baseline : {baseline_lod_path}")
    print(f"  [twist match] source   : {vsp_source_file}")
    print(f"  [twist match] tol={tol}  gain={gain}  max_iter={max_iter}")

    for iteration in range(max_iter):
        iter_id  = f"{case_id}_twist{iteration:02d}"
        iter_vsp = os.path.join(case_dir, f"{iter_id}.vsp3")
        iter_lod = os.path.join(case_dir, f"{iter_id}.lod")

        vsp.ClearVSPModel()
        vsp.ReadVSPFile(vsp_source_file)
        geom_id = find_prop_geom()
        # set_pcurve_pchip(geom_id, vsp.PROP_TWIST, twist_pts)
        # vsp.SetPCurve(geom_id, vsp.PROP_TWIST, twist_pts[0], twist_pts[1], vsp.CEDIT)
        set_pcurve_bezier(geom_id, vsp.PROP_TWIST, twist_pts[:,0], twist_pts[:,1])

        set_rpm(rpm)
        vsp.SetVSP3FileName(iter_vsp)
        vsp.Update()
        vsp.WriteVSPFile(iter_vsp, vsp.SET_ALL)

        _run_vspaero(**vspaero_args)

        if not os.path.isfile(iter_lod):
            print(f"  [twist match iter {iteration:02d}] WARNING: .lod not found — stopping.")
            break

        r_cur, t_cur = _load_thrust_profile(iter_lod, avg_last_n=avg_last_n)
        t_cur_grid = _to_grid(r_cur, t_cur, r_grid)

        rms = _rms_error(t_base_grid, t_cur_grid)
        rms_history.append(rms)
        twist_history.append(twist_pts.copy())
        thrust_history.append(t_cur_grid.copy())

        print(f"  [twist match iter {iteration:02d}]  RMS thrust error = {rms:.4f}")

        if rms <= tol:
            converged = True
            print(f"  [twist match] Converged after {iteration + 1} iteration(s).")
            break


        twist_pts = _update_twist_bezier(
            twist_pts_cur=twist_pts, r_grid=r_grid, t_ref=t_base_grid, t_cur=t_cur_grid,
            gain= gain, twist_min=twist_min, twist_max=twist_max,
        )

    if not converged:
        final_rms = rms_history[-1] if rms_history else float("nan")
        print(f"  [twist match] Did not converge after {max_iter} iteration(s) "
              f"(final RMS = {final_rms:.4f}).")

    # Save convergence plot next to the case outputs so it is easy to find
    plot_path = os.path.join(case_dir, f"{case_id}_thrust_convergence.png")
    _plot_thrust_convergence(
        r_grid, t_base_grid, thrust_history, rms_history, save_path=plot_path
    )

    return {
        "twist_pts":      twist_pts,
        "converged":      converged,
        "n_iter":         len(rms_history),
        "rms_history":    rms_history,
        "twist_history":  twist_history,
        "thrust_history": thrust_history,
    }


if __name__ == "__main__":
    from vspaero_config import*
    baseline_path = "baseline/reverse_eng_TM.lod"
    # matching_prop = "tangential_sweep_Tm/A00P04/A00P04.vsp3"
    #matching_prop = "/home/joep/Downloads/test123_twist09_manual_tweak.vsp3"
    matching_prop = "match_iter/test123_twist03.vsp3"
    case_dir = "matching_test_jun_42150"

    vsp.ClearVSPModel()
    vsp.ReadVSPFile(matching_prop)
    geom_id = find_prop_geom()
    initial_twist_x = vsp.PCurveGetTVec(geom_id, vsp.PROP_TWIST)
    initial_twist_y = vsp.PCurveGetValVec(geom_id, vsp.PROP_TWIST)
    initial_twist_pts = np.column_stack([initial_twist_x, initial_twist_y])


    vsp_aero_kwargs = dict(
        omega=OMEGA,
        R=R,
        mode=ANALYSIS_MODE,
        rho=RHO,
        vref=VREF,
        mref=MREF,
        sref=SREF,
        bref=BREF,
        cref=CREF,
        Reref=RE_CREF,
        nwakenodes=NUM_WAKE_NODES,
        ncpu=NCPU,
        wakeiter= 20,#WAKE_NUM_ITER,
        revs= NUM_REVS,
    )

    match_radial_thrust_twist(
        baseline_lod_path=baseline_path,
        vsp_source_file=matching_prop,
        case_dir=case_dir,
        case_id="test123",
        initial_twist_pts=initial_twist_pts,
        rpm=RPM,
        avg_last_n=AVG_LAST_N,
        gain = .5,
        max_iter = 10 ,
        **vsp_aero_kwargs,
    )