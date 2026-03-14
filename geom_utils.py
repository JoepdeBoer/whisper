import numpy as np
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

