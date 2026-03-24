from openvsp import VSPAERO_PROP_UNSTEADY, VSPAERO_PROP_PSEUDO_STEADY
from math import pi
# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
VSP_FILE       = "nacapropeller-mod.vsp3"
OUTPUT_DIR     = "tangential_sweep_results"

# Sweep
N_STEPS        = 6          # number of amplitude steps (includes A=0)
AMPLITUDE_FRAC = 0.5       # max amplitude = AMPLITUDE_FRAC * R  (= R/4)
N_CURVE_PTS    = 11         # control points for the half-sine PCurve

# Propeller geometry
DIAMETER       = 0.508      # m
R              = DIAMETER / 2.0
R_ROOT_FRAC    = 0.2        # r/R at blade root
R_TIP_FRAC     = 1.0        # r/R at blade tip

# VSPAERO unsteady settings
RPM            = 5000.0
VINF           = 0.0    # m/s

# Atmos
RHO            = 1.225      # kg/m³
MU             = 1.4146e-5  # m^2/s

NUM_WAKE_NODES = 60
WAKE_NUM_ITER  = 22      # Only settable for pseudo-steady
NCPU           = 8
NUM_REVS       = 3       # revolutions for auto timestep
AUTO_TIMESTEP  = True
THIN_GEOM_SET  =  2   # prop-only thin set as configured in GUI # shown, not shown set0
#ANALYSIS_MODE  = VSPAERO_PROP_UNSTEADY
ANALYSIS_MODE  = VSPAERO_PROP_PSEUDO_STEADY #TODO debug seems that this does not effect choice steady/pseudo only from gui
ANALYSIS_BASE_NAME = "Hover_analysis"
AVG_LAST_N         = 1     # timesteps to average for unsteady; None or 1 for pseudo-steady (last value only)

# Reference values (derived)
OMEGA          = RPM * 2.0 * pi / 60.0
VREF           = OMEGA * R                  # blade tip speed  [m/s]
MREF           = VREF / 340.0              # tip Mach number
CREF           = 1                         # mean chord estimate [m]  TODO: improve
BREF           = R                          # reference span = R
SREF           = pi * R**2            # rotor disk area [m²]
RE_CREF        =  RHO / MU * OMEGA * DIAMETER                          # def from Ohad Gur tip RE pa/mu * Omega * R * D
