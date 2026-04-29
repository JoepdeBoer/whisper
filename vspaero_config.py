from openvsp import VSPAERO_PROP_UNSTEADY, VSPAERO_PROP_PSEUDO_STEADY
from math import pi
# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
VSP_FILE       = "baseline/reverse_eng_TM.vsp3"
OUTPUT_DIR     = "tangential_sweep_results"

# Design exploration settings
N_STEPS        = 2          # number of amplitude steps and location steps
AMPLITUDE_FRAC = 0.2      # max amplitude = AMPLITUDE_FRAC * R
# N_CURVE_PTS    = 11         # control points for the half-sine PCurve No longer used

# Propeller geometry
DIAMETER       = 0.508      # m
R              = DIAMETER / 2.0
R_ROOT_FRAC    = 0.2        # r/R at blade root
R_TIP_FRAC     = 1.0        # r/R at blade tip

# VSPAERO operational settings
RPM            = 5684
VINF           = 0.0    # m/s

# Atmos
RHO            = 1.225      # kg/m³
MU             = 1.4146e-5  # m^2/s


# VSPAero Solver Settings

WAKE_NUM_ITER  = 22      # Only settable for pseudo-steady #TODO in gui also effects unsteady
NCPU           = 8
NUM_REVS       = 3       # revolutions for auto timestep
TIME_STEP_SIZE = 1/(RPM/60)/360 * 5 # 5 deg per step
TIME_STEPS = int(NUM_REVS/(RPM/60)/TIME_STEP_SIZE) # Enough time steps to do the NUM_REVS
NUM_WAKE_NODES = TIME_STEPS # ensures none of the wake is discarded


THIN_GEOM_SET  =  2   # prop-only thin set as configured in GUI # shown, not shown set0
ANALYSIS_MODE  = VSPAERO_PROP_UNSTEADY
#ANALYSIS_MODE  = VSPAERO_PROP_PSEUDO_STEADY #TODO debug seems that this does not effect choice steady/pseudo only from gui
ANALYSIS_BASE_NAME = "Hover_analysis"
AVG_LAST_N         = int(TIME_STEPS/NUM_REVS/2)     #  None or 1 for pseudo-steady (last value only) last half rotation for unsteady 2-bladed

# Reference values (derived)
OMEGA          = RPM * 2.0 * pi / 60.0
VREF           = OMEGA * R                 # blade tip speed  [m/s]
MREF           = VREF / 340.0              # tip Mach number
CREF           = 1                         # mean chord estimate [m]  TODO: improve
BREF           = R                         # reference span = R
SREF           = pi * R**2                 # rotor disk area [m²]
RE_CREF        =  RHO / MU * OMEGA * DIAMETER                          # def from Ohad Gur tip RE pa/mu * Omega * R * D
