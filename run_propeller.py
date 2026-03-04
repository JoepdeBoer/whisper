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
prob.set_val("vlm.chord_root", 0.10)
prob.set_val("vlm.chord_tip",  0.07)
prob.set_val("vlm.twist_root", 0.1)
prob.set_val("vlm.twist_tip",  0.05)

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
# prob.run_driver()
print(f"\nOptimised thrust: {prob.get_val('vlm.thrust')[0]:.2f} N")
print(f"Optimised torque: {prob.get_val('vlm.torque')[0]:.2f} Nm")
print(f"chord_root = {prob.get_val('vlm.chord_root')[0]:.4f} m")
print(f"chord_tip  = {prob.get_val('vlm.chord_tip')[0]:.4f} m")
print(f"twist_root = {prob.get_val('vlm.twist_root')[0]:.4f} rad")
print(f"twist_tip  = {prob.get_val('vlm.twist_tip')[0]:.4f} rad")
print(f"\nFigure of Merit proxy — T/Q = {50.0 / abs(prob.get_val('vlm.torque')[0]):.3f} N/Nm")