import math
import openmdao.api as om
import juliacall

jl = juliacall.newmodule("PropellerVLM")

jl.include("vlm_propeller.jl")

from omjlcomps import JuliaExplicitComp

prob = om.Problem()

prob.model.add_subsystem(
    "vlm",
    JuliaExplicitComp(jlcomp=jl.PropellerVLMComp())
)

prob.model.add_design_var("vlm.chord_root", lower=0.03, upper=0.20)
prob.model.add_design_var("vlm.chord_tip",  lower=0.02, upper=0.15)
prob.model.add_design_var("vlm.twist_root", lower=0.05, upper=0.60)
prob.model.add_design_var("vlm.twist_tip",  lower=0.05, upper=0.40)

# Thrust fixed at exactly 50 N — equality constraint
prob.model.add_constraint("vlm.thrust", lower=50.0)

# Minimise torque = minimise power at fixed RPM
prob.model.add_objective("vlm.torque")

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["tol"] = 1e-6

prob.setup()

# Initial values
prob.set_val("vlm.chord_root", 0.15)
prob.set_val("vlm.chord_tip",  0.08)
prob.set_val("vlm.twist_root", 0.4)
prob.set_val("vlm.twist_tip",  0.1)

# Step 1: forward solve
prob.run_model()
print(f"Initial thrust: {prob.get_val('vlm.thrust')[0]:.2f} N")
print(f"Initial torque: {prob.get_val('vlm.torque')[0]:.2f} Nm")

# Step 2: verify derivatives
data = prob.check_totals(
    of=["vlm.thrust", "vlm.torque"],
    wrt=["vlm.chord_root", "vlm.chord_tip", "vlm.twist_root", "vlm.twist_tip"],
    compact_print=True
)

# Step 3: optimise
prob.run_driver()

# Extract values
T     = prob.get_val('vlm.thrust')[0]
Q     = abs(prob.get_val('vlm.torque')[0])
RPM   = 5000.0
R     = 0.508/2
rho   = 1.225

omega = 2 * math.pi * RPM / 60.0
A     = math.pi * R**2      # rotor disk area
P_actual = Q * omega                          # actual power [W]
P_ideal  = abs(T)**1.5 / math.sqrt(2 * rho * A)   # ideal (actuator disk) power [W]
FOM = P_ideal / P_actual

print(f"\n--- Hover Performance ---")
print(f"Thrust:        {T:.2f} N")
print(f"Torque:        {Q:.2f} Nm")
print(f"Power actual:  {P_actual:.2f} W")
print(f"Power ideal:   {P_ideal:.2f} W")
print(f"Figure of Merit: {FOM:.4f}")