from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from scipy.interpolate import CubicHermiteSpline, BSpline

from dataclasses import dataclass
from enum import Enum

from numpy.typing import NDArray
import openvsp as vsp
from bspline_util import cubicbez_to_bspline, sum_bsplines


class CapType(Enum):
    NO_END_CAP = vsp.NO_END_CAP
    FLAT_END_CAP = vsp.FLAT_END_CAP
    ROUND_END_CAP = vsp.ROUND_END_CAP
    EDGE_END_CAP = vsp.EDGE_END_CAP
    SHARP_END_CAP = vsp.SHARP_END_CAP
    POINT_END_CAP = vsp.POINT_END_CAP
    ROUND_EXT_END_CAP_NONE = vsp.ROUND_EXT_END_CAP_NONE
    ROUND_EXT_END_CAP_LE = vsp.ROUND_EXT_END_CAP_LE
    ROUND_EXT_END_CAP_TE = vsp.ROUND_EXT_END_CAP_TE
    ROUND_EXT_END_CAP_BOTH = vsp.ROUND_EXT_END_CAP_BOTH

class SectionType(Enum):
    XS_CIRCLE = vsp.XS_CIRCLE
    XS_ELLIPSE = vsp.XS_ELLIPSE
    XS_SUPER_ELLIPSE = vsp.XS_SUPER_ELLIPSE
    XS_ROUNDED_RECTANGLE = vsp.XS_ROUNDED_RECTANGLE
    XS_GENERAL_FUSE = vsp.XS_GENERAL_FUSE
    XS_FILE_FUSE = vsp.XS_FILE_FUSE
    XS_FOUR_SERIES = vsp.XS_FOUR_SERIES
    XS_SIX_SERIES = vsp.XS_SIX_SERIES
    XS_BICONVEX = vsp.XS_BICONVEX
    XS_WEDGE = vsp.XS_WEDGE
    XS_EDIT_CURVE = vsp.XS_EDIT_CURVE
    XS_FILE_AIRFOIL = vsp.XS_FILE_AIRFOIL
    XS_CST_AIRFOIL = vsp.XS_CST_AIRFOIL
    XS_VKT_AIRFOIL = vsp.XS_VKT_AIRFOIL
    XS_FOUR_DIGIT_MOD = vsp.XS_FOUR_DIGIT_MOD
    XS_FIVE_DIGIT = vsp.XS_FIVE_DIGIT
    XS_FIVE_DIGIT_MOD = vsp.XS_FIVE_DIGIT_MOD
    XS_ONE_SIX_SERIES = vsp.XS_ONE_SIX_SERIES
    XS_AC25_773 = vsp.XS_AC25_773


@dataclass
class PropGeom:
    R: float
    rhub_R: float
    nB: int  # num Blades

    constructXc: float  # chord position used to place crossections
    chord: NDArray[tuple[float, float, float]] # r/R, c/R, dc/dr
    twist: NDArray[tuple[float, float, float]] # r/R, deg, ddeg/dr
    axial: NDArray[tuple[float, float, float]] # r/R, axial/R, daxial/dr
    tangential: NDArray[tuple[float, float, float]] # r/R, tangential/R, tangential/dr
    thickness: NDArray[tuple[float, float, float]]  # r/R , t/c , dt/c/dr
    CLi: NDArray[tuple[float, float, float]] # r/R, cli, dcli/dr/R `
    airfoil_type: Optional[SectionType] = SectionType.XS_FOUR_SERIES
    root_cap_type: Optional[CapType] = CapType.FLAT_END_CAP
    tip_cap_type: Optional[CapType] = CapType.FLAT_END_CAP
    root_cap_length: Optional[float|None] = None
    root_cap_offset: Optional[float|None] = None
    root_cap_strength: Optional[float|None] = None
    tip_cap_length: Optional[float | None] = None
    tip_cap_offset: Optional[float | None] = None
    tip_cap_strength: Optional[float | None] = None

@dataclass
class MeshParams:
    Num_U: int
    Num_W: int
    CapTess: int
    le_clustering: float # clustering is [0,1]
    te_clustering: float
    root_clustering: float
    tip_clustering: float



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


def set_tangential_curve(geom_id, max: float, rel_pos: float):
    """
    Cresting 3 knot cubic bezier with middle-knot maximum.
        max is amplitude/R can be negative
        pos is y/R
        rhub is relative hub pos
    """

    valvec = vsp.PCurveGetValVec(geom_id, vsp.PROP_TANGENTIAL)
    tvec = vsp.PCurveGetTVec(geom_id, vsp.PROP_TANGENTIAL)
    bspline = cubicbez_to_bspline(tvec, valvec)

    end_slope = max / ((1 - rel_pos) * 2 / 3)
    cpsweep = np.array([
        [tvec[0], 0],
        [(rel_pos - tvec[0]) / 3 + tvec[0], 0],
        [(rel_pos- tvec[0]) * 2 / 3 + tvec[0], max],
        [rel_pos, max],
        [(1 - rel_pos) / 3 + rel_pos, max],
        [(1 - rel_pos) * 2 / 3 + rel_pos, end_slope * ((1 - rel_pos) / 3)],
        [1, 0]
    ])

    param_loc = (rel_pos-tvec[0])/(1-tvec[0])
    knots_sweep = np.array([
        0, 0, 0, 0,
        param_loc, param_loc, param_loc,
        1, 1, 1, 1
    ])
    sweepspline = BSpline(knots_sweep, cpsweep, k=3)
    combined = sum_bsplines(sweepspline, bspline)
    bspline = cubicbez_to_bspline(tvec, valvec)

    t = np.linspace(0, 1, 400)
    eval_bspline = bspline(t)
    eval_sweep = sweepspline(t)
    eval_combined = combined(t)
    plt.scatter(tvec, valvec, label="initial-cps")
    plt.scatter(sweepspline.c[:,0], sweepspline.c[:,1], label="sweep")
    plt.scatter(combined.c[:,0], combined.c[:,1], label="combined")
    plt.plot(eval_bspline[:,0], eval_bspline[:,1], label="bspline")
    plt.plot(eval_sweep[:,0], eval_sweep[:,1], label="sweep")
    plt.plot(eval_combined[:,0], eval_combined[:,1], label="combined")
    plt.legend()
    plt.show()

    set_pcurve_bezier(geom_id, vsp.PROP_TANGENTIAL,
                      combined.c[:,0], combined.c[:,1])


    # _, slope_tip = find_monotone_slopes(rel_pos, max, 0, 0) # for function domain [0,1]
    # medium_slope = slope_tip[1]/2 / (1-r_hub) # half the maximum allowed, scale due to domain
    # pts = np.array([(r_hub, 0, 0), (rel_pos, max, 0), (1., 0, medium_slope)])
    # g1_pcurve_bezier(geom_id, vsp.PROP_TANGENTIAL, pts)



# def set_tangential_curve(geom_id, max, rel_pos, r_hub):
#     """
#     Add a 3-knot cubic bezier perturbation on top of the existing tangential curve.
#
#     Result is exact: the sum of two piecewise cubics is itself piecewise cubic with
#     breakpoints at the union of both knot sets.  Both curves are converted to
#     CubicHermiteSpline, evaluated at every union knot, summed, and written back.
#
#         max is amplitude/R
#         pos is y/R
#         rhub is relative hub pos
#     """
#     _, slope_tip = find_monotone_slopes(rel_pos, max, 0, 0)
#     medium_slope = slope_tip[1] / 2 / (1 - r_hub)
#     perturb_knots = np.array([(r_hub, 0.0, 0.0), (rel_pos, float(max), 0.0), (1.0, 0.0, medium_slope)])
#     perturb = CubicHermiteSpline(perturb_knots[:, 0], perturb_knots[:, 1], perturb_knots[:, 2])
#
#     r_cp = np.array(vsp.PCurveGetTVec(geom_id, vsp.PROP_TANGENTIAL))
#     val_cp = np.array(vsp.PCurveGetValVec(geom_id, vsp.PROP_TANGENTIAL))
#     base_knot_data = bezier_cps_to_knots(np.column_stack([r_cp, val_cp]))
#     base = CubicHermiteSpline(base_knot_data[:, 0], base_knot_data[:, 1], base_knot_data[:, 2])
#
#     r_union = np.union1d(base_knot_data[:, 0], perturb_knots[:, 0])
#
#     val_perturb = np.zeros(len(r_union))
#     slope_perturb = np.zeros(len(r_union))
#     in_domain = (r_union >= r_hub) & (r_union <= 1.0)
#     if np.any(in_domain):
#         val_perturb[in_domain] = perturb(r_union[in_domain])
#         slope_perturb[in_domain] = perturb(r_union[in_domain], 1)
#
#     new_knots = np.column_stack([r_union, base(r_union) + val_perturb, base(r_union, 1) + slope_perturb])
#     g1_pcurve_bezier(geom_id, vsp.PROP_TANGENTIAL, new_knots)





def set_pcurve_bezier(geom_id: str, pcurve_id: int, r_vec: list, param_vec: list, continuity:list|None = None)-> None :
    """
    Set blade p-curve to cubic-bezier.
    """
    # get vectors:
    # r_old = vsp.PCurveGetTVec(geom_id, pcurve_id)
    # param_old = vsp.PCurveGetValVec(geom_id, pcurve_id)
    # print(len(r_old))
    # print(len(param_old))
    if not continuity:

        print(f"      [dbg] SetPCurve pcurve={pcurve_id} n={len(r_vec)}", flush=True)
        print(f"      [dbg] valvec  = {[float(x) for x in param_vec]}", flush=True)
        print(f"      [dbg] tvec = {[float(x) for x in r_vec]}", flush=True)
        vsp.SetPCurve(geom_id, pcurve_id, r_vec, param_vec, vsp.CEDIT)
        print("      [dbg] SetPCurve OK, calling Update...", flush=True)
        vsp.Update()
        print("      [dbg] Update OK", flush=True)
        return
    else:
        raise NotImplementedError("Continuity setting is not yet implemented")


def set_pcurve_pchip(geom_id: str, pcurveid: int, pts: npt.NDArray) -> None:
    """
    Set blade p-curve to PCHIP interpolation through (r, value) knots.

    Parameters
    ----------
    pts : ndarray, shape (N, 2)
        Interpolation knots as (r/R, value) rows; r must be strictly increasing.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be shape (N, 2)")
    r_vec = [float(x) for x in pts[:, 0]]
    param_vec = [float(x) for x in pts[:, 1]]
    vsp.SetPCurve(geom_id, pcurveid, r_vec, param_vec, vsp.PCHIP)
    vsp.Update()


def g1_pcurve_bezier(geom_id: str, pcurveid: int, pts: npt.NDArray[tuple[float, float, float]]) -> None:
    """
    Set G1 cubic bezier p-curve by pts (r, value, slope).

    Args:
        geom_id: Geometry ID
        pcurveid: P-curve ID
        points: List of tuples (r, value, slope) defining G1-continuous control points
    """

    if len(pts) < 2:
        raise ValueError("At least 2 points required")

    r, val, slope = pts[:, 0], pts[:, 1], pts[:, 2]

    if not np.all(np.diff(r) > 0):
        raise ValueError("r values must be strictly increasing")

    # Segment lengths
    dr = np.diff(r)

    # Build control points: knot, cp1 (1/3), cp2 (2/3), knot...
    r_vec = []
    param_vec = []

    for i in range(len(r) - 1):
        r_vec.extend([r[i], r[i] + dr[i] / 3, r[i] + 2 * dr[i] / 3])
        param_vec.extend([val[i], val[i] + slope[i] * dr[i] / 3, val[i + 1] - slope[i + 1] * dr[i] / 3])

    r_vec.append(r[-1])
    param_vec.append(val[-1])

    set_pcurve_bezier(geom_id, pcurveid, r_vec, param_vec, continuity=None)


def bezier_cps_to_knots(cps: npt.NDArray) -> npt.NDArray:
    """
    Convert a G1 cubic Bezier control-point array to (x, y, slope) knots.

    Inverse of the knot→control-point expansion used in g1_pcurve_bezier.

    Parameters
    ----------
    cps : ndarray, shape (3*m + 1, 2)
        Full control-point sequence for m cubic segments, as built by
        g1_pcurve_bezier: [knot0, h0_out, h1_in, knot1, h1_out, h2_in, knot2, …].
        x-values must be strictly increasing.

    Returns
    -------
    knots : ndarray, shape (m + 1, 3)
        Rows are (x, y, dy/dx) at each knot.  For interior knots the slope is
        read from the outgoing handle; for the final knot from the incoming
        handle.  Both handles give the same dy/dx under G1 continuity.
    """
    cps = np.asarray(cps, dtype=float)
    n = len(cps)
    if (n - 1) % 3 != 0:
        raise ValueError(f"Expected 3*m+1 control points, got {n}")

    n_knots = (n - 1) // 3 + 1
    knot_idx = np.arange(n_knots) * 3   # indices 0, 3, 6, …

    x_knots = cps[knot_idx, 0]
    y_knots = cps[knot_idx, 1]
    slopes = np.empty(n_knots)

    # All knots except the last: use outgoing handle (index 3i+1)
    for i in range(n_knots - 1):
        dx = cps[3 * i + 1, 0] - cps[3 * i, 0]
        dy = cps[3 * i + 1, 1] - cps[3 * i, 1]
        slopes[i] = dy / dx if abs(dx) > 1e-14 else 0.0

    # Last knot: use incoming handle (index -2)
    dx = cps[-1, 0] - cps[-2, 0]
    dy = cps[-1, 1] - cps[-2, 1]
    slopes[-1] = dy / dx if abs(dx) > 1e-14 else 0.0

    return np.column_stack([x_knots, y_knots, slopes])


def set_rpm(rpm):
    """Set RPM on unsteady group 0."""
    container_id = vsp.FindContainer("VSPAEROSettings", 0)
    if container_id:
        pid = vsp.FindParm(container_id, "GeomSet", "VSPAERO")
        if pid:
            vsp.SetParmVal(pid, vsp.SET_ALL)
    num_groups = vsp.GetNumUnsteadyGroups()
    if num_groups < 1:
        prop_id = find_prop_geom()
        vsp.SetParmVal( prop_id, "PropMode", "Design", vsp.PROP_BLADES )


    group_id = vsp.FindUnsteadyGroup(0)
    if not group_id:
        print("  WARNING: could not find prop")
        return
    pid = vsp.FindParm(group_id, "RPM", "UnsteadyGroup")
    if pid:
        vsp.SetParmVal(pid, rpm)
    else:
        print("  WARNING: RPM parm not found in unsteady group")


def handle_propgeom(prop_id: str, params: PropGeom) -> None:
    # Diameter and blade count
    find_set(prop_id, "Diameter", "Design", params.R * 2)
    find_set(prop_id, "NumBlade", "Design", params.nB)       # used by VSPAERO
    find_set(prop_id, "Sym_Rot_N", "Sym", params.nB)         # rotational symmetry (visual)

    # Hub extent and activity factor limit
    find_set(prop_id, "RadiusFrac", "XSec_0", params.rhub_R)
    find_set(prop_id, "AFLimit", "Design", params.rhub_R)

    # Chord construction position (fraction of chord used to place XSecs)
    find_set(prop_id, "ConstructXoC", "Design", params.constructXc)

    # Airfoil section type on all blade XSecs, with invert cleared
    xsec_surf = vsp.GetXSecSurf(prop_id, 0)
    for i in range(vsp.GetNumXSec(xsec_surf)):
        if params.airfoil_type is not None:
            vsp.ChangeXSecShape(xsec_surf, i, params.airfoil_type.value)
        xsec_id = vsp.GetXSec(xsec_surf, i)
        invert_id = vsp.GetXSecParm(xsec_id, "Invert")

        if invert_id:
            vsp.SetParmVal(invert_id, False)  # ensure not inverted

    # Blade distribution curves — each array is (r/R, value, slope)
    g1_pcurve_bezier(prop_id, vsp.PROP_CHORD,      params.chord)
    g1_pcurve_bezier(prop_id, vsp.PROP_TWIST,      params.twist)
    g1_pcurve_bezier(prop_id, vsp.PROP_RAKE,       params.axial)
    g1_pcurve_bezier(prop_id, vsp.PROP_TANGENTIAL, params.tangential)
    g1_pcurve_bezier(prop_id, vsp.PROP_THICK,      params.thickness)
    g1_pcurve_bezier(prop_id, vsp.PROP_CLI,        params.CLi)

    # Root cap (CapUMin = inboard end)
    find_set(prop_id, "CapUMinOption", "EndCap", params.root_cap_type.value)
    if params.root_cap_length is not None:
        find_set(prop_id, "CapUMinLength",   "EndCap", params.root_cap_length)
    if params.root_cap_offset is not None:
        find_set(prop_id, "CapUMinOffset",   "EndCap", params.root_cap_offset)
    if params.root_cap_strength is not None:
        find_set(prop_id, "CapUMinStrength", "EndCap", params.root_cap_strength)

    # Tip cap (CapUMax = outboard end)
    find_set(prop_id, "CapUMaxOption", "EndCap", params.tip_cap_type.value)
    if params.tip_cap_length is not None:
        find_set(prop_id, "CapUMaxLength",   "EndCap", params.tip_cap_length)
    if params.tip_cap_offset is not None:
        find_set(prop_id, "CapUMaxOffset",   "EndCap", params.tip_cap_offset)
    if params.tip_cap_strength is not None:
        find_set(prop_id, "CapUMaxStrength", "EndCap", params.tip_cap_strength)

    vsp.Update()


def handle_mesh(prop_id: str, params: MeshParams) -> None:
    find_set(prop_id, "Tess_U",      "Shape",  params.Num_U)
    find_set(prop_id, "Tess_W",      "Shape",  params.Num_W)
    find_set(prop_id, "CapUMinTess", "EndCap", params.CapTess)
    find_set(prop_id, "LECluster",   "Design", params.le_clustering)
    find_set(prop_id, "TECluster",   "Design", params.te_clustering)
    find_set(prop_id, "InCluster",   "Design", params.root_clustering)
    find_set(prop_id, "OutCluster",  "Design", params.tip_clustering)
    vsp.Update()


def find_set(parm_container_id: str, parm_name: str, group_name: str, value: Any) -> None:
    parm_id = vsp.FindParm(parm_container_id, parm_name, group_name)
    vsp.SetParmVal(parm_id, value)


def beta_distribution(x, maximum=30, peak_location=0.8, sharpness=6):
    """
    distribution using skewed Beta function

    Parameters:
    - max_sweep: peak value
    - peak_location: where max occurs (0–1)
    - sharpness: overall curvature (higher = sharper peak)

    Returns:
    - y: distribution
    """

    # Solve for a and b from peak location
    a = peak_location * sharpness
    b = (1 - peak_location) * sharpness

    y = (x**a) * ((1 - x)**b)

    # Normalize to maximum
    y = y / np.max(y) * maximum

    return y

def find_monotone_slopes(x_peak, M, f0=0, f1=0):
    """
    Find slopes at endpoints that satisfy monotonicity constraints.
    for monotone piecewise cubic spline x in [0,1]

    Returns:
    --------
    slope0_range : tuple
        (min_slope, max_slope) at t=0
    slope1_range : tuple
        (min_slope, max_slope) at t=1
    """
    h1 = x_peak
    delta1 = (M - f0) / h1

    h2 = 1 - x_peak
    delta2 = (f1 - M) / h2

    # For segment 1: slope0 must satisfy 0 <= slope0/delta1 <= 3
    slope0_min = 0
    slope0_max = 3 * delta1

    # For segment 2: slope1 must satisfy 0 <= slope1/delta2 <= 3
    # Note: delta2 is negative, so inequality flips
    if delta2 < 0:
        slope1_min = 3 * delta2  # more negative
        slope1_max = 0
    else:
        slope1_min = 0
        slope1_max = 3 * delta2

    return (slope0_min, slope0_max), (slope1_min, slope1_max)