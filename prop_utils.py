from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from vspaero_config import R_ROOT_FRAC, R_TIP_FRAC, N_CURVE_PTS, DIAMETER

from dataclasses import dataclass
from enum import Enum

from numpy.typing import NDArray
import openvsp as vsp

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


def set_tangential_curve(geom_id, max, rel_pos, r_hub):
    """
    Cresting 3 knot cubic bezier with middle-knot maximum.
        max is amplitude/R
        pos is y/R
        rhub is relative hub pos
    """
    slope_root = max/(rel_pos - r_hub)
    slope_tip = max/(rel_pos-1)
    pts = np.array([(r_hub, 0, slope_root), (rel_pos, max, 0), (1., 0, slope_tip)])
    g1_pcurve_bezier(geom_id, vsp.PROP_TANGENTIAL, pts)



def set_pcurve_bezier(geom_id: str, pcurveid: int, r_vec: list, param_vec: list, continuity:list|None = None)-> None :
    """
    Set blade p-curve to cubic-bezier.
    """

    if not continuity:
        vsp.SetPCurve(geom_id, pcurveid,
                      param_vec, r_vec, vsp.CEDIT)
        vsp.Update()
        return
    else:
        raise NotImplementedError("Continuity setting is not yet implemented")


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

    set_pcurve_bezier(geom_id, pcurveid, param_vec, r_vec, continuity=None)


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

    # Airfoil section type on all blade XSecs
    if params.airfoil_type is not None:
        xsec_surf = vsp.GetXSecSurf(prop_id, 0)
        for i in range(vsp.GetNumXSec(xsec_surf)):
            vsp.ChangeXSecShape(xsec_surf, i, params.airfoil_type.value)

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