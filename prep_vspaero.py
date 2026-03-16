import openvsp as vsp

_SWEEP = "VSPAEROSweep"

# BUG UnsteadyType does not change analysis mode correctly:
def set_prop_blades_mode(mode: int) -> None:
    cid = vsp.FindContainer("VSPAEROSettings", 0)
    pid = vsp.FindParm(cid, "m_PropBladesMode", "VSPAERO")
    if pid:
        vsp.SetParmVal(pid, mode)
        vsp.Update()
        print(f"  Set m_PropBladesMode={mode}")
    else:
        print("  WARNING: m_PropBladesMode not found")

def vspaero_compute_geometry() -> None:
    cg = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(cg)
    vsp.SetIntAnalysisInput(cg, "GeomSet", [vsp.SET_NONE])  # TODO test
    vsp.SetIntAnalysisInput(cg, "ThinGeomSet", [vsp.SET_ALL])
    if not vsp.ExecAnalysis(cg):
        print("    ERROR: VSPAEROComputeGeometry failed")
        return None

def prep_sweep_analysis(omega: float,
                        R: float,
                        mode: int,
                        rho: float,
                        vref: float,
                        mref: float,
                        sref: float,
                        bref: float,
                        cref: float,
                        Reref: float,
                        nwakenodes: int,
                        ncpu: int,
                        wakeiter: int | None = None,
                        revs: int | None = None,
                        time_step: int | None = None,
                        ) -> None:
    """Applies the given settings to a sweep analysis and updates analysis manager."""

    sw = _SWEEP
    vsp.SetAnalysisInputDefaults(sw)

    mach = omega * R * 0.7 / 340  # TODO is this right

    # Set to VLM no panels
    vsp.SetIntAnalysisInput(sw, "UnsteadyType", [mode])
    vsp.SetIntAnalysisInput(sw, "GeomSet", [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(sw, "ThinGeomSet", [vsp.SET_ALL])

    # Flight condition
    vsp.SetDoubleAnalysisInput(sw, "Vinf", [0.])
    vsp.SetDoubleAnalysisInput(sw, "Rho", [rho])
    vsp.SetDoubleAnalysisInput(sw, "AlphaStart", [0.])
    vsp.SetDoubleAnalysisInput(sw, "AlphaEnd", [0.])
    vsp.SetIntAnalysisInput(sw, "AlphaNpts", [1])
    vsp.SetDoubleAnalysisInput(sw, "MachStart", [mach])
    vsp.SetDoubleAnalysisInput(sw, "MachEnd", [mach])
    vsp.SetIntAnalysisInput(sw, "MachNpts", [1])


    # Reference values
    vsp.SetIntAnalysisInput(sw, "ManualVrefFlag", [True])
    vsp.SetDoubleAnalysisInput(sw, "Vref", [vref])
    vsp.SetDoubleAnalysisInput(sw, "ReCref", [Reref]) # TODO test effect
    vsp.SetDoubleAnalysisInput(sw, "Machref", [mref])
    vsp.SetDoubleAnalysisInput(sw, "Sref", [sref])
    vsp.SetDoubleAnalysisInput(sw, "bref", [bref])
    vsp.SetDoubleAnalysisInput(sw, "cref", [cref])

    # Wake
    vsp.SetIntAnalysisInput(sw, "FixedWakeFlag", [False])
    vsp.SetIntAnalysisInput(sw, "NumWakeNodes", [nwakenodes])
    if mode != 1:
        vsp.SetIntAnalysisInput(sw, "WakeNumIter", [wakeiter])  # not used in unsteady analysis

    # Time stepping
    if not time_step:
        vsp.SetIntAnalysisInput(sw, "AutoTimeStepFlag", [True])
    else:
        raise NotImplementedError('No manual time step ')
    vsp.SetIntAnalysisInput(sw, "AutoTimeNumRevs", [int(revs)])

    # CPU
    vsp.SetIntAnalysisInput(sw, "NCPU", [ncpu])

    vsp.Update()


def run_vspaero(omega: float,
                        R: float,
                        mode: int,
                        rho: float,
                        vref: float,
                        mref: float,
                        sref: float,
                        bref: float,
                        cref: float,
                        Reref: float,
                        nwakenodes: int,
                        ncpu: int,
                        wakeiter: int | None = None,
                        revs: int | None = None,
                        time_step: int | None = None,) -> str:
    """Run VSPAERO unsteady/Pseudo-steady VLM. Returns results_id or None."""
    set_prop_blades_mode(mode = mode)
    vspaero_compute_geometry()
    prep_sweep_analysis(omega, R, mode, rho, vref, mref, sref, bref,cref, Reref, nwakenodes,
                        ncpu, wakeiter=wakeiter, revs=revs, time_step=time_step)

    rid = vsp.ExecAnalysis(_SWEEP)
    if not rid:
        print("    ERROR: VSPAEROSweep failed")
    return rid

