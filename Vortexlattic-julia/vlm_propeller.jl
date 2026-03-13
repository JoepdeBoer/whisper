# vlm_propeller.jl
# Requires: VortexLattice, OpenMDAOCore, ForwardDiff

using VortexLattice
using OpenMDAOCore
using ForwardDiff


# ──────────────────────────────────────────────────────────────
# Helper: build the propeller VLM system from design variables
# ──────────────────────────────────────────────────────────────
function run_vlm(chord_root, chord_tip, twist_root, twist_tip, n_blades)

    # Fixed parameters
    RPM      = 5000.0
    R        = 0.508 / 2        # 20 inches diameter
    n_blades = 2
    rho      = 1.225

    ns = 12
    nc = 4

    omega = 2π * RPM / 60.0
    Vtip  = omega * R
    r_hub = 0.1 * R

    # Use a type-consistent reference velocity
    # eltype() ensures Vinf_eff is Dual when inputs are Dual
    TF       = eltype(promote(chord_root, chord_tip, twist_root, twist_tip))
#     Vinf_eff = TF(0.05 * Vtip)   # 5% of tip speed — small but well away from zero

    xle   = [TF(0.0),   TF(0.0)]
    yle   = [TF(r_hub), TF(R)  ]
    zle   = [TF(0.0),   TF(0.0)]
    chord = [chord_root, chord_tip]
    theta = [twist_root, twist_tip]
    phi   = [TF(0.0),   TF(0.0)]
    fc    = fill((xc) -> zero(TF), 2)

    Sref = TF(π * R^2)
    cref = (chord_root + chord_tip) / 2
    bref = TF(R - r_hub)
    rref = [TF(0.0), TF(0.0), TF(0.0)]
    ref  = Reference(Sref, cref, bref, rref, 1)

    # Freestream must also be typed with TF
    alpha = TF(0.0)
    beta  = TF(0.0)
    Omega = [TF(omega), TF(0.0), TF(0.0)]
    fs    = Freestream(0, alpha, beta, Omega)

    grid, surface = wing_to_surface_panels(xle, yle, zle, chord, theta, phi, ns, nc;
                        fc=fc,
                        spacing_s=Sine(),
                        spacing_c=Uniform())

    surfaces = [surface]
    system   = steady_analysis(surfaces, ref, fs; symmetric=false)

    CF, CM = body_forces(system; frame=Wind())

    q = TF(0.5 * rho) * 1 # ref velocity of 1
    T = CF[3] * q * Sref
    Q = CM[1] * q * Sref * cref

    return T, Q
end

# ──────────────────────────────────────────────────────────────
# OpenMDAO component
# ──────────────────────────────────────────────────────────────
struct PropellerVLMComp <: OpenMDAOCore.AbstractExplicitComp end

function OpenMDAOCore.setup(self::PropellerVLMComp)
    # Design variables (inputs)
    inputs = [
        VarData("chord_root", shape=1, val=0.1,  units="m"),
        VarData("chord_tip",  shape=1, val=0.07, units="m"),
        VarData("twist_root", shape=1, val=0.35, units="rad"),  # ~20 deg
        VarData("twist_tip",  shape=1, val=0.15, units="rad"),  # ~8 deg
        VarData("RPM",        shape=1, val=2000.0),
        VarData("Vinf",       shape=1, val=20.0, units="m/s"),
    ]

    # Outputs of interest
    outputs = [
        VarData("thrust", shape=1, val=0.0, units="N"),
        VarData("torque", shape=1, val=0.0, units="N*m"),
    ]

    # Partial derivatives we will provide
    partials = [
        PartialsData("thrust", "chord_root"),
        PartialsData("thrust", "chord_tip"),
        PartialsData("thrust", "twist_root"),
        PartialsData("thrust", "twist_tip"),
        PartialsData("thrust", "RPM"),
        PartialsData("thrust", "Vinf"),
        PartialsData("torque", "chord_root"),
        PartialsData("torque", "chord_tip"),
        PartialsData("torque", "twist_root"),
        PartialsData("torque", "twist_tip"),
        PartialsData("torque", "RPM"),
        PartialsData("torque", "Vinf"),
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::PropellerVLMComp, inputs, outputs)
    cr  = inputs["chord_root"][1]
    ct  = inputs["chord_tip"][1]
    tr  = inputs["twist_root"][1]
    tt  = inputs["twist_tip"][1]
    rpm = inputs["RPM"][1]
    V   = inputs["Vinf"][1]

    T, Q = run_vlm(cr, ct, tr, tt, 2)  # R=0.5m, 3 blades

    outputs["thrust"][1] = T
    outputs["torque"][1] = Q
end

function OpenMDAOCore.compute_partials!(self::PropellerVLMComp, inputs, partials)
    x = [
        inputs["chord_root"][1],
        inputs["chord_tip"][1],
        inputs["twist_root"][1],
        inputs["twist_tip"][1],
    ]

    function f(x)
        T, Q = run_vlm(x[1], x[2], x[3], x[4], 2)
        return [T, Q]
    end

    J = ForwardDiff.jacobian(f, x)

    vars = ["chord_root", "chord_tip", "twist_root", "twist_tip"]
    for (j, var) in enumerate(vars)
        partials["thrust", var][1] = J[1, j]
        partials["torque", var][1] = J[2, j]
    end
end