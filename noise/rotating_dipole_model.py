"""
Rotating Dipole Acoustic Noise Models
Frequency-domain acoustic pressure amplitude functions for rotating dipoles.
"""

import numpy as np
from scipy.special import jv  # Bessel function of the first kind


def steady_thrust_noise_rotating_dipole_fd(Ft, m, Omega, B, Rmic, phio, theta, zeta, Ma, co):
    """
    Frequency-domain acoustic pressure amplitude based on steady thrust (Ft) of a rotating dipole.

    Parameters
    ----------
    Ft    : float - Steady thrust force [N]
    m     : int   - Harmonic number
    Omega : float - Rotational speed [rad/s]
    B     : int   - Number of blades
    Rmic  : float - Microphone distance [m]
    phio  : float - Initial propeller angle [rad]
    theta : float - Observer azimuthal angle [rad]
    zeta  : float - Elevation angle [rad]
    Ma    : float - Mach number at blade section
    co    : float - Speed of sound [m/s]

    Returns
    -------
    Pthat : complex - Acoustic pressure amplitude
    """
    R1 = -1j * Omega * B**2 * np.exp(-1j * m * B * Omega * Rmic / co) / (4 * np.pi * Rmic * co)
    R2 = np.exp(-1j * m * B * phio)
    R3 = np.exp(-1j * m * B * (zeta - np.pi / 2))
    R4 = jv(m * B, m * B * Ma * np.sin(theta))
    R5 = m * np.cos(theta)

    Pthat = R1 * R2 * R3 * R4 * R5 * Ft
    return Pthat


def steady_drag_noise_rotating_dipole_fd(Fd, m, Omega, B, Rmic, phio, theta, zeta, Ma, co):
    """
    Frequency-domain acoustic pressure amplitude based on steady drag (Fd) of a rotating dipole.

    Parameters
    ----------
    Fd    : float - Steady drag force [N]
    m     : int   - Harmonic number
    Omega : float - Rotational speed [rad/s]
    B     : int   - Number of blades
    Rmic  : float - Microphone distance [m]
    phio  : float - Initial propeller angle [rad]
    theta : float - Observer azimuthal angle [rad]
    zeta  : float - Elevation angle [rad]
    Ma    : float - Mach number at blade section
    co    : float - Speed of sound [m/s]

    Returns
    -------
    Pdhat : complex - Acoustic pressure amplitude
    """
    R1 = -1j * Omega * B**2 * np.exp(-1j * m * B * Omega * Rmic / co) / (4 * np.pi * Rmic * co)
    R2 = np.exp(-1j * m * B * phio)
    R3 = np.exp(-1j * m * B * (zeta - np.pi / 2))
    R4 = jv(m * B, m * B * Ma * np.sin(theta))
    R5 = -m / Ma

    Pdhat = R1 * R2 * R3 * R4 * R5 * Fd
    return Pdhat

# def steady_side_noise_rotating_dipole_fd(Fy, m, Omega, B, Rmic, phio, theta, zeta, Ma, co):
#     #TODO check by Derriving from Hanson 1980 eq 28
#     """
#     Frequency-domain acoustic pressure amplitude based on steady side force (Fy) of a rotating dipole.
#     Fy is the in-plane force perpendicular to drag (Fz), coupling via the radial direction.
#
#     Parameters
#     ----------
#     Fy    : float - Steady side force [N]
#     m     : int   - Harmonic number
#     Omega : float - Rotational speed [rad/s]
#     B     : int   - Number of blades
#     Rmic  : float - Microphone distance [m]
#     phio  : float - Initial propeller angle [rad]
#     theta : float - Observer azimuthal angle [rad]
#     zeta  : float - Elevation angle [rad]
#     Ma    : float - Mach number at blade section
#     co    : float - Speed of sound [m/s]
#
#     Returns
#     -------
#     Pyhat : complex - Acoustic pressure amplitude
#     """
#     R1 = -1j * Omega * B**2 * np.exp(-1j * m * B * Omega * Rmic / co) / (4 * np.pi * Rmic * co)
#     R2 = np.exp(-1j * m * B * phio)
#     R3 = np.exp(-1j * m * B * (zeta - np.pi / 2))
#
#     # Bessel function derivative via recurrence: J'_n(x) = 0.5*(J_{n-1}(x) - J_{n+1}(x))
#     mB   = m * B
#     arg  = mB * Ma * np.sin(theta)
#     dJv  = 0.5 * (jv(mB - 1, arg) - jv(mB + 1, arg))
#
#     R5 = np.sin(theta) * dJv   # replaces the plain Jv(mB, arg) * directivity factor
#
#     Pyhat = R1 * R2 * R3 * R5 * Fy
#     return Pyhat

def dirac_delta_thrust_noise_rotating_dipole_fd(Fti, m, Omega, B, Rmic, phio, theta, zeta, phii, Ma, co, pmaxint):
    """
    Frequency-domain acoustic pressure amplitude based on Dirac delta thrust (Fti) of a rotating dipole.

    Parameters
    ----------
    Fti     : float - Impulse thrust force [N]
    m       : int   - Harmonic number
    Omega   : float - Rotational speed [rad/s]
    B       : int   - Number of blades
    Rmic    : float - Microphone distance [m]
    phio    : float - Initial propeller angle [rad]
    theta   : float - Observer azimuthal angle [rad]
    zeta    : float - Elevation angle [rad]
    phii    : float - Impulse propeller angle [rad]
    Ma      : float - Mach number at blade section
    co      : float - Speed of sound [m/s]
    pmaxint : int   - Integration truncation limit for Dirac delta sum

    Returns
    -------
    Ptihat : complex - Acoustic pressure amplitude
    """
    R1 = -1j * Omega * B**2 * np.exp(-1j * m * B * Omega * Rmic / co) / (4 * np.pi * Rmic * co)
    R2 = np.exp(-1j * m * B * phio)

    p_range = np.arange(-pmaxint, pmaxint + 1)
    R3 = np.exp(-1j * (m * B - p_range) * (zeta - np.pi / 2))
    R4 = jv(m * B - p_range, m * B * Ma * np.sin(theta))
    R5 = m * np.cos(theta) * np.exp(-1j * p_range * phii)

    Ptihat = np.sum(R3 * R4 * R5) * R1 * R2 * Fti
    return Ptihat


def dirac_delta_drag_noise_rotating_dipole_fd(Fdi, m, Omega, B, Rmic, phio, theta, zeta, phii, Ma, co, pmaxint):
    """
    Frequency-domain acoustic pressure amplitude based on Dirac delta drag (Fdi) of a rotating dipole.

    Parameters
    ----------
    Fdi     : float - Impulse drag force [N]
    m       : int   - Harmonic number
    Omega   : float - Rotational speed [rad/s]
    B       : int   - Number of blades
    Rmic    : float - Microphone distance [m]
    phio    : float - Initial propeller angle [rad]
    theta   : float - Observer azimuthal angle [rad]
    zeta    : float - Elevation angle [rad]
    phii    : float - Impulse propeller angle [rad]
    Ma      : float - Mach number at blade section
    co      : float - Speed of sound [m/s]
    pmaxint : int   - Integration truncation limit for Dirac delta sum

    Returns
    -------
    Pdihat : complex - Acoustic pressure amplitude
    """
    R1 = -1j * Omega * B**2 * np.exp(-1j * m * B * Omega * Rmic / co) / (4 * np.pi * Rmic * co)
    R2 = np.exp(-1j * m * B * phio)

    p_range = np.arange(-pmaxint, pmaxint + 1)
    R3 = np.exp(-1j * (m * B - p_range) * (zeta - np.pi / 2))
    R4 = jv(m * B - p_range, m * B * Ma * np.sin(theta))
    R5 = -(m * B - p_range) * np.exp(-1j * p_range * phii) / (B * Ma)

    Pdihat = np.sum(R3 * R4 * R5) * R1 * R2 * Fdi
    return Pdihat
