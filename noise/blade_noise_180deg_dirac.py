"""
test_blade_noise_180deg_dirac.py
Equivalent to testBladeNoise180DegDirac.m

Two-bladed propeller with two 180-degree impulses.
Computes polar directivity, SPL/SPLA spectra, radial OSPL/phase,
and overall sound power levels (OSWL / OSWLA).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from noise_utils import compute_noise_from_distributed_dipole_sources, compute_a_weighting_factor

# ── Parameters ───────────────────────────────────────────────────────────────
co  = 340.0
rho = 1.225

nB          = 2                      # number of blades
vphiB       = np.array([0.0, 180.0]) # blade angles [deg]
Rtip        = 0.25                   # m
n           = 4000                   # RPM
phiIDeg1    = 90.0                   # impulse angle blade 1 [deg]
phiIDeg2    = 90.0 + 180.0           # impulse angle blade 2 [deg]
FM          = 0.6                    # figure of merit
TotalThrust = 7 * 9.81 / nB          # N per blade

Omega   = n * 2 * np.pi / 60
A       = np.pi * Rtip**2
CT      = (nB * TotalThrust) / (rho * A * (Omega * Rtip)**2)
CP      = CT**1.5 / (np.sqrt(2) * FM)
CQ      = CP
Torque  = CQ * (rho * A * (Omega * Rtip)**2 * Rtip) / nB
TotalDrag = Torque / (0.6 * Rtip)

pctImpulse         = 0.05
impulseDeltaThrust = TotalThrust * pctImpulse
impulseDeltaDrag   = TotalDrag   * pctImpulse

# ── Radial discretisation ────────────────────────────────────────────────────
nR   = 32
rR   = np.linspace(0.2, 1.0, nR)
dr   = rR[1] - rR[0]

# Half-sine sweep
phi0DegR_base = -np.sin(np.linspace(0, np.pi, nR)) * 10.0

# Triangular load distribution (peak at rRmax = 0.8)
def triangular_distribution(rR, peak_r=0.8):
    peakV      = 2.0 / (rR[-1] - rR[0])
    slopePlus  = peakV / (peak_r - rR[0])
    slopeMinus = -peakV / (rR[-1] - peak_r)
    dist = np.where(rR <= peak_r,
                    (rR - rR[0]) * slopePlus,
                    (rR - rR[-1]) * slopeMinus)
    return dist

dist      = triangular_distribution(rR)
FtR       = dist * TotalThrust      * dr
FdR       = dist * TotalDrag        * dr
FtiR      = dist * impulseDeltaThrust * dr
FdiR      = dist * impulseDeltaDrag   * dr

chord = 0.05 * np.ones(nR)

# ── Figure 1: Blade geometry ─────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(6, 6), num=1)
ang_circle = np.linspace(0, 2 * np.pi, 501)
ax1.plot(np.cos(ang_circle), np.sin(ang_circle), 'k-.')

for iB in range(nB):
    phi_sweep = phi0DegR_base * np.pi / 180
    xle = rR * np.cos(phi_sweep)
    yle = rR * np.sin(phi_sweep)

    # Rotate for blade angle
    rotAng = vphiB[iB] * np.pi / 180
    Q      = np.array([[np.cos(rotAng), -np.sin(rotAng)],
                       [np.sin(rotAng),  np.cos(rotAng)]])
    pts    = Q @ np.vstack([xle, yle])
    ax1.plot(pts[0], pts[1], '-', label=f'Blade {iB+1}')

ax1.set_xlabel('r/R')
ax1.set_ylabel('r/R')
ax1.set_xlim([-1.05, 1.05])
ax1.set_ylim([-1.05, 1.05])
ax1.set_aspect('equal')
ax1.grid(True)
ax1.legend()
ax1.set_title('Blade geometry')
fig1.tight_layout()

# ── Noise computation parameters ─────────────────────────────────────────────
nm       = 10
vm       = np.arange(1, nm + 1)
Rmic     = 10.0
zetaDeg  = 0.0
ntheta   = 91
thetaDeg = np.linspace(0, 360, ntheta)
pmaxint  = 30

# ── Compute noise for each blade ─────────────────────────────────────────────
Pthat   = np.zeros((ntheta, nm), dtype=complex)
Pdhat   = np.zeros((ntheta, nm), dtype=complex)
Ptihat  = np.zeros((ntheta, nm), dtype=complex)
Pdihat  = np.zeros((ntheta, nm), dtype=complex)
PthatR  = np.zeros((ntheta, nm, nR), dtype=complex)
PdhatR  = np.zeros((ntheta, nm, nR), dtype=complex)
PtihatR = np.zeros((ntheta, nm, nR), dtype=complex)
PdihatR = np.zeros((ntheta, nm, nR), dtype=complex)

B_single = 1   # treat each blade independently

for iB in range(nB):
    phi0DegR = vphiB[iB] + phi0DegR_base

    res1 = compute_noise_from_distributed_dipole_sources(
        Rtip, B_single, Omega, rR, phi0DegR, FtR, FdR, FtiR, FdiR,
        co, vm, zetaDeg, thetaDeg, phiIDeg1, pmaxint, Rmic)
    _, _, Ptihat1, Pdihat1, _, _, PtihatR1, PdihatR1 = res1

    res2 = compute_noise_from_distributed_dipole_sources(
        Rtip, B_single, Omega, rR, phi0DegR, FtR, FdR, FtiR, FdiR,
        co, vm, zetaDeg, thetaDeg, phiIDeg2, pmaxint, Rmic)
    PthatB, PdhatB, Ptihat2, Pdihat2, PthatRB, PdhatRB, PtihatR2, PdihatR2 = res2

    Pthat   += PthatB
    Pdhat   += PdhatB
    PthatR  += PthatRB
    PdhatR  += PdhatRB
    Ptihat  += Ptihat1 + Ptihat2
    Pdihat  += Pdihat1 + Pdihat2
    PtihatR += PtihatR1 + PtihatR2
    PdihatR += PdihatR1 + PdihatR2

# ── Filter zeros ─────────────────────────────────────────────────────────────
def nan_zeros(arr):
    out = arr.copy()
    out[np.abs(out) <= 0] = np.nan
    return out

Pthat   = nan_zeros(Pthat)
Pdhat   = nan_zeros(Pdhat)
Ptihat  = nan_zeros(Ptihat)
Pdihat  = nan_zeros(Pdihat)
PthatR  = nan_zeros(PthatR)
PdhatR  = nan_zeros(PdhatR)
PtihatR = nan_zeros(PtihatR)
PdihatR = nan_zeros(PdihatR)

# ── SPL ───────────────────────────────────────────────────────────────────────
pref = 20e-6   # Pa
Pref = 1e-12   # W

SPLt  = 20 * np.log10(np.abs(Pthat)  / pref)
SPLd  = 20 * np.log10(np.abs(Pdhat)  / pref)
SPLti = 20 * np.log10(np.abs(Ptihat) / pref)
SPLdi = 20 * np.log10(np.abs(Pdihat) / pref)

for arr in [SPLt, SPLd, SPLti, SPLdi]: # Filter
    arr[arr <= 0] = np.nan

# ── A-weighting ───────────────────────────────────────────────────────────────
B_aw            = B_single
vFreq           = vm * B_aw * Omega / (2 * np.pi)
dSPL_Aweighting = compute_a_weighting_factor(vFreq)

SPLtA  = SPLt  + np.tile(dSPL_Aweighting, (ntheta, 1))
SPLdA  = SPLd  + np.tile(dSPL_Aweighting, (ntheta, 1))
SPLtiA = SPLti + np.tile(dSPL_Aweighting, (ntheta, 1))
SPLdiA = SPLdi + np.tile(dSPL_Aweighting, (ntheta, 1))

# ── OSWL ─────────────────────────────────────────────────────────────────────
dtheta_rad = (thetaDeg[1] - thetaDeg[0]) * np.pi / 180
area_fac   = 2 * Rmic**2 * dtheta_rad / rho / co

def oswl(P2sum):
    return 10 * np.log10(P2sum * area_fac) - 10 * np.log10(Pref)

OSWLt  = oswl(np.nansum(np.abs(Pthat)**2))
OSWLd  = oswl(np.nansum(np.abs(Pdhat)**2))
OSWLti = oswl(np.nansum(np.abs(Ptihat)**2))
OSWLdi = oswl(np.nansum(np.abs(Pdihat)**2))

def oswl_a(SPL_A):
    total = np.nansum(10**(SPL_A / 10))
    if total == 0:
        return np.nan
    return (10 * np.log10(total)
            + 10 * np.log10(area_fac)
            + 10 * np.log10(pref**2 / Pref))

OSWLtA  = oswl_a(SPLtA)
OSWLdA  = oswl_a(SPLdA)
OSWLtiA = oswl_a(SPLtiA)
OSWLdiA = oswl_a(SPLdiA)

print(f"OSWLt  = {OSWLt:.2f} dB    OSWLtA  = {OSWLtA:.2f} dBA")
print(f"OSWLd  = {OSWLd:.2f} dB    OSWLdA  = {OSWLdA:.2f} dBA")
print(f"OSWLti = {OSWLti:.2f} dB   OSWLtiA = {OSWLtiA:.2f} dBA")
print(f"OSWLdi = {OSWLdi:.2f} dB   OSWLdiA = {OSWLdiA:.2f} dBA")

# ── Figure 11: Polar plots (total SPL) ───────────────────────────────────────
theta_rad = thetaDeg * np.pi / 180

def total_spl(P):
    P2sum = np.nansum(np.abs(P)**2, axis=1)
    P2sum[P2sum == 0] = np.nan
    spl   = 10 * np.log10(P2sum) - 20 * np.log10(pref)
    spl[spl <= 30] = np.nan
    return spl

SPLtT  = total_spl(Pthat)
SPLdT  = total_spl(Pdhat)
SPLtiT = total_spl(Ptihat)
SPLdiT = total_spl(Pdihat)

fig11, axes11 = plt.subplots(1, 4, subplot_kw={'projection': 'polar'}, figsize=(16, 4), num=11)
for ax, data, title in zip(axes11,
                           [SPLtT, SPLdT, SPLtiT, SPLdiT],
                           ['Steady Thrust', 'Steady Drag', 'Impulse Thrust', 'Impulse Drag']):
    ax.plot(theta_rad, data)
    ax.set_rlim([30, 110])
    ax.set_title(title)
fig11.suptitle('Total SPL Polar Directivity')
fig11.tight_layout()

# ── Figure 12: Spectrum subplots ─────────────────────────────────────────────
thetaPlot = 45.0
iTheta    = np.argmin(np.abs(thetaDeg - thetaPlot))

titles   = ['Steady Thrust', 'Steady Drag', 'Impulse Thrust', 'Impulse Drag']
spl_list = [SPLt, SPLd, SPLti, SPLdi]
splA_list= [SPLtA, SPLdA, SPLtiA, SPLdiA]

fig12, axes12 = plt.subplots(2, 4, figsize=(16, 7), num=12)
for col, (title, spl, splA) in enumerate(zip(titles, spl_list, splA_list)):
    ax_top = axes12[0, col]
    ax_bot = axes12[1, col]
    ax_top.stem(vFreq, spl[iTheta, :],  markerfmt='C0o', linefmt='C0-', basefmt='k-')
    ax_top.set(xlabel='Frequency [Hz]', ylabel='SPL [dB]', title=title,
               xlim=[50, 2000], ylim=[30, 120])
    ax_top.grid(True)
    ax_bot.stem(vFreq, splA[iTheta, :], markerfmt='C1o', linefmt='C1-', basefmt='k-')
    ax_bot.set(xlabel='Frequency [Hz]', ylabel='SPLA [dBA]', title=title,
               xlim=[50, 2000], ylim=[30, 100])
    ax_bot.grid(True)
fig12.tight_layout()

# ── Figure 101: Radial OSPL and phase ────────────────────────────────────────
def radial_ospl(PR):
    P2R = np.nansum(np.abs(PR)**2, axis=(0, 1))
    P2R[P2R == 0] = np.nan
    return 10 * np.log10(P2R) - 20 * np.log10(pref)

OSPLtR  = radial_ospl(PthatR)
OSPLdR  = radial_ospl(PdhatR)
OSPLtiR = radial_ospl(PtihatR)
OSPLdiR = radial_ospl(PdihatR)

phitR  = np.unwrap(np.angle(PthatR[iTheta, 0, :]))  * 180 / np.pi
phidR  = np.unwrap(np.angle(PdhatR[iTheta, 0, :]))  * 180 / np.pi
phitiR = np.unwrap(np.angle(PtihatR[iTheta, 0, :])) * 180 / np.pi
phidiR = np.unwrap(np.angle(PdihatR[iTheta, 0, :])) * 180 / np.pi

fig101, axes101 = plt.subplots(2, 4, figsize=(16, 7), num=101)
ospl_data  = [OSPLtR, OSPLdR, OSPLtiR, OSPLdiR]
phase_data = [phitR, phidR, phitiR, phidiR]

for col, (title, ospl, phase) in enumerate(zip(titles, ospl_data, phase_data)):
    ax_top = axes101[0, col]
    ax_bot = axes101[1, col]
    ax_top.plot(rR, ospl, '-o')
    ax_top.set(xlabel='Radial percentage', ylabel='OSPL [dB]',
               title=title, xlim=[0, 1], ylim=[30, 120])
    ax_top.grid(True)
    ax_bot.plot(rR, phase, '-o')
    ax_bot.set(xlabel='Radial percentage', ylabel='Phase angle [deg]',
               title=title, xlim=[0, 1], ylim=[-180, 180])
    ax_bot.grid(True)
fig101.tight_layout()

# ── Figure 201: OSWL bar chart ───────────────────────────────────────────────
labels = ['Steady\nThrust', 'Steady\nDrag', 'Impulse\nThrust', 'Impulse\nDrag']
x      = np.arange(1, 5)
y      = [OSWLt, OSWLd, OSWLti, OSWLdi]
yA     = [OSWLtA, OSWLdA, OSWLtiA, OSWLdiA]

fig201, (ax_top201, ax_bot201) = plt.subplots(2, 1, figsize=(6, 8), num=201)
ax_top201.plot(x, y, '-o')
ax_top201.set(ylabel='OSWL [dB]', ylim=[30, 140], xticks=x, xticklabels=labels)
ax_top201.grid(True)

ax_bot201.plot(x, yA, '-o')
ax_bot201.set(ylabel='OSWLA [dBA]', ylim=[30, 140], xticks=x, xticklabels=labels)
ax_bot201.grid(True)

fig201.tight_layout()
plt.figure(1).savefig("dirac180_fig1.png", dpi=120, bbox_inches="tight")
plt.figure(11).savefig("dirac180_fig11.png", dpi=120, bbox_inches="tight")
plt.figure(12).savefig("dirac180_fig12.png", dpi=120, bbox_inches="tight")
plt.figure(101).savefig("dirac180_fig101.png", dpi=120, bbox_inches="tight")
plt.figure(201).savefig("dirac180_fig201.png", dpi=120, bbox_inches="tight")
plt.close("all")
print("test_blade_noise_180deg_dirac: done.")
