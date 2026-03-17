"""
noise_analysis.py
-----------------
Post-processing script: loads VSPAERO aerodynamic results produced by main.py
and computes propeller noise using the rotating-dipole model in noise/.

For each sweep case found in OUTPUT_DIR (tangential_sweep_results/):
  1. Load radial force distribution from radial_distribution.csv
  2. Convert dCT/dR, dCQ/dR → dimensional forces FtR, FdR [N/station] per blade
  3. Run compute_noise_from_distributed_dipole_sources() for each blade
  4. Compute SPL, SPLA spectra; OSWL, OSWLA overall sound-power levels
  5. Save polar-directivity, spectrum, and radial-OSPL plots per case
  6. Save an OSWL comparison plot across all cases

Force conversion
----------------
dCT_dR in radial_distribution.csv is summed over all blades.
Per-blade: FtR[i] = dCT_dR[i] * dr[i] * rho * A * (Omega*R)^2 / N_BLADES
           FdR[i] = dCQ_dR[i] * dr[i] * rho * A * (Omega*R)^2 / (rR[i] * N_BLADES)
  (drag force = dQ/dr  / r,  where CQ = Q / (rho A (Omega R)^2 R))

Sweep-angle reconstruction
--------------------------
Matches set_tangential_curve() in geom_utils.py:
  tan_offset(rR) = A * sin( pi * (rR - R_ROOT) / (R_TIP - R_ROOT) )
  phi_deg(rR)    = arctan2( tan_offset, rR * R )  [deg]
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from noise import (compute_noise_from_distributed_dipole_sources,
                         compute_a_weighting_factor)

# ── propeller / aerodynamic config ────────────────────────────────────────────
from vspaero_config import OUTPUT_DIR, R, OMEGA, RHO, R_ROOT_FRAC, R_TIP_FRAC
from read_result import parse_lod

# ═════════════════════════════════════════════════════════════════════════════
# USER-CONFIGURABLE NOISE PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
N_BLADES    = 2      # number of blades (blades are evenly spaced in azimuth)
CO          = 340.0  # speed of sound [m/s]
NM          = 10     # number of harmonics to compute
RMIC        = 10.0   # microphone distance [m]
ZETA_DEG    = 0.0    # elevation angle of observers above rotor plane [deg]
NTHETA      = 91     # number of azimuthal observer angles  (0–360°)
PMAXINT     = 30     # Dirac-delta series truncation order
PCT_IMPULSE = 0.05   # impulse load = PCT_IMPULSE * steady load
PHI_I_DEG1  = 90.0   # impulse-event azimuth for blade 1 [deg]
THETA_SPEC  = 45.0   # observer azimuth used for the spectrum subplot [deg]

PREF        = 20e-6  # acoustic reference pressure [Pa]
PREF_W      = 1e-12  # acoustic reference power    [W]

NOISE_DIR   = "noise_results"  # output directory for all noise plots
# ═════════════════════════════════════════════════════════════════════════════


# ── helpers ───────────────────────────────────────────────────────────────────

def _sweep_angle_deg(rR, amplitude_m):
    """Reconstruct sweep (phi0) angle [deg] at each r/R for a given amplitude."""
    tan_offset = amplitude_m * np.sin(
        np.pi * (rR - R_ROOT_FRAC) / (R_TIP_FRAC - R_ROOT_FRAC))
    r_phys = np.where(rR < 1e-9, 1e-9, rR) * R
    return np.degrees(np.arctan2(tan_offset, r_phys))


def _forces_per_blade(rR, dCT_dR, dCQ_dR):
    """Convert dimensionless radial distributions to per-blade forces [N/station].

    Returns
    -------
    FtR : steady thrust force  at each radial station [N]
    FdR : steady drag  force   at each radial station [N]
    """
    dr    = np.gradient(rR)                            # non-uniform spacing ok
    q_ref = RHO * np.pi * R**2 * (OMEGA * R)**2       # rho * A * (Omega R)^2

    FtR = dCT_dR * dr * q_ref / N_BLADES
    # Q_section = dCQ * dr * q_ref * R  →  F_drag = Q_section / (rR * R)
    safe_rR = np.where(rR < 1e-9, 1e-9, rR)
    FdR = dCQ_dR * dr * q_ref / (safe_rR * N_BLADES)
    return FtR, FdR


def _nan_zeros(arr):
    out = arr.copy()
    out[np.abs(out) <= 0] = np.nan
    return out


def _oswl(P2sum, area_fac):
    if P2sum == 0:
        return np.nan
    return 10 * np.log10(P2sum * area_fac) - 10 * np.log10(PREF_W)


def _oswl_a(SPL_A, area_fac):
    total = np.nansum(10 ** (SPL_A / 10))
    if total == 0:
        return np.nan
    return (10 * np.log10(total)
            + 10 * np.log10(area_fac)
            + 10 * np.log10(PREF**2 / PREF_W))


def _total_spl(P):
    P2sum = np.nansum(np.abs(P)**2, axis=1)
    P2sum[P2sum == 0] = np.nan
    spl = 10 * np.log10(P2sum) - 20 * np.log10(PREF)
    spl[spl <= 30] = np.nan
    return spl


def _radial_ospl(PR):
    P2R = np.nansum(np.abs(PR)**2, axis=(0, 1))
    P2R[P2R == 0] = np.nan
    return 10 * np.log10(P2R) - 20 * np.log10(PREF)


# ── core noise routine ────────────────────────────────────────────────────────

def run_noise_for_case(rR, dCT_dR, dCQ_dR, amplitude_m, label, out_dir):
    """Run full noise analysis for one sweep case and save plots.

    Returns a dict of OSWL/OSWLA scalars for summary plotting.
    """
    rR     = np.asarray(rR,     dtype=float)
    dCT_dR = np.asarray(dCT_dR, dtype=float)
    dCQ_dR = np.asarray(dCQ_dR, dtype=float)
    nR     = len(rR)

    phi0_base        = _sweep_angle_deg(rR, amplitude_m)
    FtR, FdR         = _forces_per_blade(rR, dCT_dR, dCQ_dR)
    FtiR, FdiR       = PCT_IMPULSE * FtR, PCT_IMPULSE * FdR

    vm       = np.arange(1, NM + 1)
    thetaDeg = np.linspace(0, 360, NTHETA)
    # Blade azimuthal positions (evenly distributed)
    vphiB    = np.linspace(0, 360, N_BLADES, endpoint=False)

    # Accumulators
    shape_2d = (NTHETA, NM)
    shape_3d = (NTHETA, NM, nR)
    Pthat,  Pdhat,  Ptihat,  Pdihat  = (np.zeros(shape_2d, dtype=complex) for _ in range(4))
    PthatR, PdhatR, PtihatR, PdihatR = (np.zeros(shape_3d, dtype=complex) for _ in range(4))

    for iB, phi_B in enumerate(vphiB):
        phi0DegR = phi_B + phi0_base
        phiI1    = PHI_I_DEG1 + phi_B
        phiI2    = PHI_I_DEG1 + 180.0 + phi_B

        # Two impulse events per blade (180° apart within one revolution)
        res1 = compute_noise_from_distributed_dipole_sources(
            R, 1, OMEGA, rR, phi0DegR, FtR, FdR, FtiR, FdiR,
            CO, vm, ZETA_DEG, thetaDeg, phiI1, PMAXINT, RMIC)
        _, _, Ptihat1, Pdihat1, _, _, PtihatR1, PdihatR1 = res1

        res2 = compute_noise_from_distributed_dipole_sources(
            R, 1, OMEGA, rR, phi0DegR, FtR, FdR, FtiR, FdiR,
            CO, vm, ZETA_DEG, thetaDeg, phiI2, PMAXINT, RMIC)
        PtB, PdB, Ptihat2, Pdihat2, PtRB, PdRB, PtihatR2, PdihatR2 = res2

        Pthat   += PtB
        Pdhat   += PdB
        PthatR  += PtRB
        PdhatR  += PdRB
        Ptihat  += Ptihat1 + Ptihat2
        Pdihat  += Pdihat1 + Pdihat2
        PtihatR += PtihatR1 + PtihatR2
        PdihatR += PdihatR1 + PdihatR2

    # ── Filter zero entries → NaN ──────────────────────────────────────────
    Pthat   = _nan_zeros(Pthat)
    Pdhat   = _nan_zeros(Pdhat)
    Ptihat  = _nan_zeros(Ptihat)
    Pdihat  = _nan_zeros(Pdihat)
    PthatR  = _nan_zeros(PthatR)
    PdhatR  = _nan_zeros(PdhatR)
    PtihatR = _nan_zeros(PtihatR)
    PdihatR = _nan_zeros(PdihatR)

    # ── SPL ───────────────────────────────────────────────────────────────
    SPLt  = 20 * np.log10(np.abs(Pthat)  / PREF)
    SPLd  = 20 * np.log10(np.abs(Pdhat)  / PREF)
    SPLti = 20 * np.log10(np.abs(Ptihat) / PREF)
    SPLdi = 20 * np.log10(np.abs(Pdihat) / PREF)
    for arr in (SPLt, SPLd, SPLti, SPLdi):
        arr[arr <= 0] = np.nan

    # ── A-weighting ───────────────────────────────────────────────────────
    vFreq  = vm * OMEGA / (2 * np.pi)           # harmonics of rotation freq
    dSPL_A = compute_a_weighting_factor(vFreq)
    aw_tile = np.tile(dSPL_A, (NTHETA, 1))
    SPLtA  = SPLt  + aw_tile
    SPLdA  = SPLd  + aw_tile
    SPLtiA = SPLti + aw_tile
    SPLdiA = SPLdi + aw_tile

    # ── OSWL ──────────────────────────────────────────────────────────────
    dtheta_rad = (thetaDeg[1] - thetaDeg[0]) * np.pi / 180
    area_fac   = 2 * RMIC**2 * dtheta_rad / RHO / CO

    OSWLt  = _oswl(np.nansum(np.abs(Pthat) **2), area_fac)
    OSWLd  = _oswl(np.nansum(np.abs(Pdhat) **2), area_fac)
    OSWLti = _oswl(np.nansum(np.abs(Ptihat)**2), area_fac)
    OSWLdi = _oswl(np.nansum(np.abs(Pdihat)**2), area_fac)

    OSWLtA  = _oswl_a(SPLtA,  area_fac)
    OSWLdA  = _oswl_a(SPLdA,  area_fac)
    OSWLtiA = _oswl_a(SPLtiA, area_fac)
    OSWLdiA = _oswl_a(SPLdiA, area_fac)

    def _fmt(v): return f"{v:.2f}" if v is not None and np.isfinite(v) else "N/A"
    print(f"  OSWLt  = {_fmt(OSWLt)} dB    OSWLtA  = {_fmt(OSWLtA)} dBA")
    print(f"  OSWLd  = {_fmt(OSWLd)} dB    OSWLdA  = {_fmt(OSWLdA)} dBA")
    print(f"  OSWLti = {_fmt(OSWLti)} dB   OSWLtiA = {_fmt(OSWLtiA)} dBA")
    print(f"  OSWLdi = {_fmt(OSWLdi)} dB   OSWLdiA = {_fmt(OSWLdiA)} dBA")

    titles   = ['Steady Thrust', 'Steady Drag', 'Impulse Thrust', 'Impulse Drag']
    theta_rad = thetaDeg * np.pi / 180
    iTheta    = int(np.argmin(np.abs(thetaDeg - THETA_SPEC)))

    # ── Figure A: Polar directivity ────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, subplot_kw={'projection': 'polar'}, figsize=(16, 4))
    for ax, data, title in zip(axes,
                               [_total_spl(Pthat), _total_spl(Pdhat),
                                _total_spl(Ptihat), _total_spl(Pdihat)],
                               titles):
        ax.plot(theta_rad, data)
        ax.set_rlim([30, 110])
        ax.set_title(title)
    fig.suptitle(f'Total SPL Polar Directivity — {label}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_polar.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Figure B: SPL / SPLA spectrum at THETA_SPEC ───────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, (title, spl, splA) in enumerate(
            zip(titles, [SPLt, SPLd, SPLti, SPLdi],
                        [SPLtA, SPLdA, SPLtiA, SPLdiA])):
        axes[0, col].stem(vFreq, spl[iTheta],  markerfmt='C0o', linefmt='C0-', basefmt='k-')
        axes[0, col].set(xlabel='Frequency [Hz]', ylabel='SPL [dB]', title=title,
                         xlim=[0, max(vFreq)*1.1], ylim=[30, 120])
        axes[0, col].grid(True)
        axes[1, col].stem(vFreq, splA[iTheta], markerfmt='C1o', linefmt='C1-', basefmt='k-')
        axes[1, col].set(xlabel='Frequency [Hz]', ylabel='SPLA [dBA]', title=title,
                         xlim=[0, max(vFreq)*1.1], ylim=[30, 100])
        axes[1, col].grid(True)
    fig.suptitle(f'SPL Spectrum at θ={THETA_SPEC}° — {label}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_spectrum.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Figure C: Radial OSPL and phase ───────────────────────────────────
    OSPLtR  = _radial_ospl(PthatR)
    OSPLdR  = _radial_ospl(PdhatR)
    OSPLtiR = _radial_ospl(PtihatR)
    OSPLdiR = _radial_ospl(PdihatR)
    phitR  = np.unwrap(np.angle(PthatR[iTheta, 0, :]))  * 180 / np.pi
    phidR  = np.unwrap(np.angle(PdhatR[iTheta, 0, :]))  * 180 / np.pi
    phitiR = np.unwrap(np.angle(PtihatR[iTheta, 0, :])) * 180 / np.pi
    phidiR = np.unwrap(np.angle(PdihatR[iTheta, 0, :])) * 180 / np.pi

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, (title, ospl, phase) in enumerate(zip(
            titles,
            [OSPLtR, OSPLdR, OSPLtiR, OSPLdiR],
            [phitR,  phidR,  phitiR,  phidiR])):
        axes[0, col].plot(rR, ospl, '-o', ms=4)
        axes[0, col].set(xlabel='r/R', ylabel='OSPL [dB]', title=title,
                         xlim=[0, 1], ylim=[30, 120])
        axes[0, col].grid(True)
        axes[1, col].plot(rR, phase, '-o', ms=4)
        axes[1, col].set(xlabel='r/R', ylabel='Phase [deg]', title=title,
                         xlim=[0, 1], ylim=[-180, 180])
        axes[1, col].grid(True)
    fig.suptitle(f'Radial OSPL and Phase — {label}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_radial.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    return dict(
        label=label, amplitude_m=amplitude_m,
        OSWLt=OSWLt,  OSWLd=OSWLd,  OSWLti=OSWLti,  OSWLdi=OSWLdi,
        OSWLtA=OSWLtA, OSWLdA=OSWLdA, OSWLtiA=OSWLtiA, OSWLdiA=OSWLdiA,
    )


# ── case discovery ────────────────────────────────────────────────────────────

def _find_cases(results_dir):
    """Yield (label, amplitude_m, lod_path) for every case with a .lod file."""
    if not os.path.isdir(results_dir):
        return
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        lod_path = os.path.join(entry.path, f"{entry.name}.lod")
        if not os.path.isfile(lod_path):
            continue
        # Parse amplitude from directory name: "A00_amp0mm" → 0.0 m
        parts = entry.name.split("_amp")
        try:
            amp_m = float(parts[1].rstrip("mm")) / 1000.0 if len(parts) == 2 else 0.0
        except ValueError:
            amp_m = 0.0
        yield entry.name, amp_m, lod_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(NOISE_DIR, exist_ok=True)
    cases = list(_find_cases(OUTPUT_DIR))

    if not cases:
        print(f"No cases with radial_distribution.csv found in '{OUTPUT_DIR}/'.")
        return

    print(f"Found {len(cases)} case(s).  Saving noise plots to '{NOISE_DIR}/'.\n")
    summary = []

    for label, amp_m, lod_path in cases:
        print(f"[{label}]  amplitude = {amp_m*1000:.1f} mm")
        # Read last iteration only (pseudo-steady converged result)
        res = parse_lod(lod_path, avg_last_n=1)
        if not res["r_norm"]:
            print(f"  WARNING: no radial data in {lod_path}, skipping.")
            continue
        rR     = np.array(res["r_norm"])
        dCT_dR = np.array(res["dCT_dR"])
        dCQ_dR = np.array(res["dCQ_dR"])
        # Keep only stations within the actual blade span; hub panels below
        # R_ROOT_FRAC have non-zero VSPAERO loads but no physical blade.
        mask   = (rR >= R_ROOT_FRAC) & (rR <= R_TIP_FRAC)
        row = run_noise_for_case(
            rR          = rR[mask],
            dCT_dR      = dCT_dR[mask],
            dCQ_dR      = dCQ_dR[mask],
            amplitude_m = amp_m,
            label       = label,
            out_dir     = NOISE_DIR,
        )
        summary.append(row)
        print()

    # ── Comparison plot (only meaningful for ≥2 cases) ────────────────────
    if len(summary) >= 2:
        labels = [r["label"] for r in summary]
        x      = np.arange(len(labels))

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(6, 2*len(labels)+2), 8))
        for metrics, ax, ylabel in [
            (["OSWLt", "OSWLd", "OSWLti", "OSWLdi"],   ax_top, "OSWL [dB]"),
            (["OSWLtA", "OSWLdA", "OSWLtiA", "OSWLdiA"], ax_bot, "OSWLA [dBA]"),
        ]:
            for m in metrics:
                vals = [r[m] for r in summary]
                ax.plot(x, vals, '-o', label=m)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)
        fig.suptitle("OSWL comparison across sweep cases")
        fig.tight_layout()
        path = os.path.join(NOISE_DIR, "oswl_comparison.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison plot → {path}")

    print(f"\nDone. All noise results in '{NOISE_DIR}/'.")


if __name__ == "__main__":
    main()