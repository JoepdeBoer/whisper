"""
Utility functions for distributed rotating dipole noise computation.
"""

import numpy as np
from rotating_dipole_model import (
    steady_thrust_noise_rotating_dipole_fd,
    steady_drag_noise_rotating_dipole_fd,
    dirac_delta_thrust_noise_rotating_dipole_fd,
    dirac_delta_drag_noise_rotating_dipole_fd,
)


def compute_noise_from_distributed_dipole_sources(
    Rtip, B, Omega, rR, phi0DegR, FtR, FdR, FtiR, FdiR,
    co, vm, zetaDeg, thetaDeg, phiIDeg, pmaxint, Rmic
):
    """
    Estimate noise from a series of rotating dipoles distributed along the
    blade radius Rtip at normalised positions rR (0–1) with sweep angle phi0DegR.

    Parameters
    ----------
    Rtip      : float       - Tip radius [m]
    B         : int         - Number of blades
    Omega     : float       - Rotational speed [rad/s]
    rR        : array(nR,)  - Normalised radial stations [0–1]
    phi0DegR  : array(nR,)  - Sweep angle at each radial station [deg]
    FtR       : array(nR,)  - Steady thrust per station [N]
    FdR       : array(nR,)  - Steady drag per station [N]
    FtiR      : array(nR,)  - Impulse thrust per station [N]
    FdiR      : array(nR,)  - Impulse drag per station [N]
    co        : float       - Speed of sound [m/s]
    vm        : array(nm,)  - Harmonic number vector
    zetaDeg   : float       - Elevation angle [deg]
    thetaDeg  : array(ntheta,) - Observer azimuthal angles [deg]
    phiIDeg   : float       - Impulse propeller angle [deg]
    pmaxint   : int         - Dirac delta integration truncation
    Rmic      : float       - Microphone distance [m]

    Returns
    -------
    Pthat, Pdhat, Ptihat, Pdihat : arrays (ntheta, nm)
        Total summed pressure amplitudes
    PthatR, PdhatR, PtihatR, PdihatR : arrays (ntheta, nm, nR)
        Per-radial-station pressure amplitudes
    """
    nR     = len(rR)
    nm     = len(vm)
    ntheta = len(thetaDeg)

    Pthat  = np.zeros((ntheta, nm), dtype=complex)
    Pdhat  = np.zeros((ntheta, nm), dtype=complex)
    Ptihat = np.zeros((ntheta, nm), dtype=complex)
    Pdihat = np.zeros((ntheta, nm), dtype=complex)

    PthatR  = np.zeros((ntheta, nm, nR), dtype=complex)
    PdhatR  = np.zeros((ntheta, nm, nR), dtype=complex)
    PtihatR = np.zeros((ntheta, nm, nR), dtype=complex)
    PdihatR = np.zeros((ntheta, nm, nR), dtype=complex)

    zeta = zetaDeg * np.pi / 180
    phii = phiIDeg * np.pi / 180
    theta_arr = thetaDeg * np.pi / 180

    for ir in range(nR):
        r    = rR[ir] * Rtip
        Ma   = Omega * r / co
        Ft   = FtR[ir]
        Fd   = FdR[ir]
        Fti  = FtiR[ir]
        Fdi  = FdiR[ir]
        phio = phi0DegR[ir] * np.pi / 180

        for itheta, theta in enumerate(theta_arr):
            for im, m in enumerate(vm):
                pt  = steady_thrust_noise_rotating_dipole_fd(Ft,  m, Omega, B, Rmic, phio, theta, zeta, Ma, co)
                pd  = steady_drag_noise_rotating_dipole_fd(Fd,   m, Omega, B, Rmic, phio, theta, zeta, Ma, co)
                pti = 0
                pdi = 0
                for phi in phii: # TODO check with literature/derrivation
                    for i in range(B):
                        pti += dirac_delta_thrust_noise_rotating_dipole_fd(Fti, m, Omega, B, Rmic, phio, theta, zeta, phi, Ma, co, pmaxint)
                        pdi += dirac_delta_drag_noise_rotating_dipole_fd(Fdi,   m, Omega, B, Rmic, phio, theta, zeta, phi, Ma, co, pmaxint)

                PthatR[itheta, im, ir]  = pt
                PdhatR[itheta, im, ir]  = pd
                PtihatR[itheta, im, ir] = pti
                PdihatR[itheta, im, ir] = pdi

                Pthat[itheta, im]  += pt
                Pdhat[itheta, im]  += pd
                Ptihat[itheta, im] += pti
                Pdihat[itheta, im] += pdi

    return Pthat, Pdhat, Ptihat, Pdihat, PthatR, PdhatR, PtihatR, PdihatR


def compute_a_weighting_factor(freq_hz):
    """
    Compute A-weighting correction in dB for an array of frequencies [Hz].

    Uses the standard IEC 61672 formula.

    Parameters
    ----------
    freq_hz : array-like - Frequencies [Hz]

    Returns
    -------
    dSPL : ndarray - A-weighting correction [dB]
    """
    f = np.asarray(freq_hz, dtype=float)
    f2 = f ** 2
    numerator   = 12194.0**2 * f2**2
    denominator = (
        (f2 + 20.6**2)
        * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12194.0**2)
    )
    Ra = numerator / denominator
    dSPL = 20 * np.log10(Ra) + 2.0  # normalise so 1 kHz → 0 dB
    return dSPL
