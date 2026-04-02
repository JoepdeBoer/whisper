import numpy as np
import numpy.typing as npt
import openvsp as vsp
from vspaero_config import R_ROOT_FRAC, R_TIP_FRAC, N_CURVE_PTS


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


def set_pcurve_bezier(geom_id: str, pcurveid: int, r_vec: list, param_vec: list, continuity:list|None = None)-> None :
    # TODO check input type geom_id
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

