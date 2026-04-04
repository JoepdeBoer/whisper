import numpy as np
from prop_utils import MeshParams, SectionType, CapType, PropGeom


mesh_params = MeshParams()

# Fixed geometric parameters
R = 0.254
rhub_R = 0.2
baseline_prop = PropGeom(R = R, rhub_R = rhub_R, nB = 2, constructXc = .25,
                         chord= np.array([(rhub_R, 0.076, 0.08), (0.67, 0.17, 0), (1., 0.075, -0.16)]))

baseline_prop.twist = np.array([(baseline_prop.rhub_R, )])
baseline_prop.axial = np.array([(0, 0), (0, 0), (0, 0)])
baseline_prop.tangential = np.array([(0, 0), (0, 0), (0, 0)])
baseline_prop.thickness = np.array([
    (baseline_prop.rhub_R*baseline_prop.R, .1, -2),
    (0.5, .05, -.2),
    (1., .04, 0)])
baseline_prop.CLine = np.array([(0, 0), (0, 0), (0, 0)])

baseline_prop.root_cap_type = CapType.FLAT_END_CAP
baseline_prop.tip_cap_type = CapType.FLAT_END_CAP
baseline_prop.SectionType = SectionType.XS_FOUR_SERIES

# Mesh variables
mesh_params.Num_U = 30
mesh_params.Num_W = 17
mesh_params.CapTess = 3
mesh_params.le_clustering = .3
mesh_params.te_clustering = .3
mesh_params.root_clustering = 1.5
mesh_params.tip_clustering = .3


if __name__ == "__main__":
    import openvsp as vsp
    import matplotlib.pyplot as plt
    from pathlib import Path
    from prop_utils import handle_propgeom

    vsp.VSPCheckSetup() # checks and otherwise initilises
    vsp.ClearVSPModel()
    vsp.SetVSP3FileName("baseline.vsp")
    propgeom_id = vsp.AddGeom("PROP")

    handle_propgeom(propgeom_id, params=baseline_prop)







