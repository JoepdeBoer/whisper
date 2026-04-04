from pathlib import Path

import openvsp as vsp
from vspaero.optimizer import pyVSPOptimizer
from vspaero import functions
from prep_vspaero import vspaero_compute_geometry


# Settings
prefix = "nacapropeller-mod"
maindir = Path(__file__).parent.parent.absolute()
VSP_PREFIX = maindir/prefix
vsp.ReadVSPFile(str(VSP_PREFIX))




opt = pyVSPOptimizer()
opt.set_unsteady_analysis(0) #unsteady?? what about pseudo steady
opt.set_optimization_functions([functions.ROTOR_EFFICIENCY])
vspaero_compute_geometry()
opt.setup(str(VSP_PREFIX))
opt.solve(calculate_gradients=True)
print('solved???')




