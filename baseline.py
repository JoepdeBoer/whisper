import numpy as np
from prop_utils import MeshParams, SectionType, CapType, PropGeom, handle_mesh

# Fixed geometric parameters
R = 0.254
rhub_R = 0.2
baseline_prop = PropGeom(R = R, rhub_R = rhub_R, nB = 2, constructXc = .25,
                         chord= np.array([(rhub_R, 0.076, 0.08), (0.67, 0.17, 0), (1., 0.075, -0.16)]),
                         twist= np.array([(rhub_R, 20, -40), (.5, 11.9, -6.667), (1., 9.9, -2)]),
                         axial= np.array([(rhub_R, 0.0, 0.0), (0.5, 0., 0.), (1., 0.0, 0.)]),
                         tangential= np.array([(rhub_R, 0.0, 0.0), (0.5, 0., 0.), (1., 0.0, 0.)]),
                         thickness= np.array([(rhub_R, .1, -2), (0.5, .05, -.2), (1., .04, 0)]),
                         CLi= np.array([(rhub_R, 0.61, 0.0), (0.65, 0.55, -0.5), (1., 0.0, -2.5)]),
                         root_cap_type= CapType.FLAT_END_CAP,
                         tip_cap_type= CapType.FLAT_END_CAP,
                         airfoil_type = SectionType.XS_FOUR_SERIES
                         )


# Mesh variables
mesh_params = MeshParams(Num_U=30, Num_W = 17, CapTess= 3, le_clustering=.3, te_clustering=.3,
                         root_clustering=1.5, tip_clustering=.3)



if __name__ == "__main__":
    import openvsp as vsp
    import matplotlib.pyplot as plt
    from pathlib import Path
    from prop_utils import handle_propgeom

    file_name = "baseline.vsp3"

    vsp.VSPCheckSetup() # checks and otherwise initilises
    vsp.ClearVSPModel()
    vsp.SetVSP3FileName(file_name)
    propgeom_id = vsp.AddGeom("PROP")
    handle_propgeom(propgeom_id, params=baseline_prop)
    handle_mesh(propgeom_id, params=mesh_params)
    vsp.WriteVSPFile(file_name, vsp.SET_ALL)







