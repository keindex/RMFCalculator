# ============================================================================
# TOV (Tolman-Oppenheimer-Volkoff) 方程求解器
# ============================================================================
#
# TOV 方程描述了中子星内部的流体静力学平衡:
#
#   (1) 径向压力梯度 (Tolman-Oppenheimer-Volkoff 方程):
#       $$\frac{dP}{dr} = -\frac{(\varepsilon + P)(m + 4\pi r^3 P)}{r(r - 2m)}$$
#
#   (2) 质量累积方程:
#       $$\frac{dm}{dr} = 4\pi r^2 \varepsilon$$
#
#   (3) 致密度 (Compactness):
#       $$C = \frac{M}{R}$$
#
#   (4) 声速:
#       $$C_s^2 = \frac{dP}{d\varepsilon}$$
#
#   (5) 潮汐形变参数 (Tidal Deformability):
#       $$\Lambda = \frac{2}{15\pi E}\kappa^2(1-2C)^2(2+C(y-1)-y)$$
#
# ============================================================================

# Python packages
import numpy as np
from scipy.interpolate import interp1d

# Import files
from RMFCalculator.tov import solver_code as TOV_solver
from RMFCalculator.tov import EoS_import
from RMFCalculator.tov import speed_of_sound
from RMFCalculator.tov.unit import g_cm_3, G, c, km, Msun


def _has_physical_mr_points(mass, radius):
    """Return True when an MR sequence contains enough plausible NS points."""
    if len(mass) < 3 or len(radius) < 3:
        return False

    mass = np.asarray(mass, dtype=float)
    radius = np.asarray(radius, dtype=float)
    radius_km = radius / km
    mass_msun = mass / Msun

    physical = (
        np.isfinite(radius_km)
        & np.isfinite(mass_msun)
        & (radius_km > 4.0)
        & (radius_km < 40.0)
        & (mass_msun > 0.02)
        & (mass_msun < 4.0)
    )
    return np.count_nonzero(physical) >= 3


def OutputMR(input_file="", density=[], pressure=[],
             central_density_range=np.logspace(14.3, 15.6, 50) * g_cm_3):
    """Outputs the mass, radius, and tidal deformability

    Args:
        file_name (string, optional): string. CSV file to be opened.
        density (array, optional): numpy 1Darray. Passed into a check function and returned if valid.
        pressure (array, optional): numpy 1Darray. Passed into a check function and returned if valid.

    Returns:
        MR (tuple): tuple with mass, radius.
    """
    energy_density, pressure = EoS_import.EOS_import(input_file, density, pressure)
    original_energy_density = energy_density.copy()
    original_pressure = pressure.copy()

    energy_density = energy_density * G / c**2
    pressure = pressure * G / c**4

    unique_pressure_indices = np.unique(pressure, return_index=True)[1]
    unique_pressure = pressure[np.sort(unique_pressure_indices)]

    eos = interp1d(energy_density, pressure, kind="cubic", fill_value="extrapolate")

    inveos = interp1d(
        unique_pressure,
        energy_density[unique_pressure_indices],
        kind="cubic",
        fill_value="extrapolate",
    )

    Pmin = pressure[20]

    Radius = []
    Mass = []

    for i in range(len(central_density_range)):
        try:
            center_rho = central_density_range[i] * G / c**2
            M, R = TOV_solver.solveTOV(center_rho, Pmin, eos, inveos)
            Mass.append(M)
            Radius.append(R)
        except OverflowError as e:
            break

    if not _has_physical_mr_points(Mass, Radius):
        Radius = []
        Mass = []
        for center_density in central_density_range:
            try:
                M, R, _ = TOV_solver.solveTOV_tidal(
                    center_density,
                    original_energy_density,
                    original_pressure,
                )
                Mass.append(M)
                Radius.append(R)
            except OverflowError:
                break

    MR = np.vstack((Mass, Radius)).T

    return MR


def OutputMRT(input_file="", density=[], pressure=[]):
    """Outputs the mass, radius, and tidal deformability

    Args:
        file_name (string, optional): string. CSV file to be opened.
        density (array, optional): numpy 1Darray. Passed into a check function and returned if valid.
        pressure (array, optional): numpy 1Darray. Passed into a check function and returned if valid.

    Returns:
        MRT (tuple): tuple with mass, radius, and tidal deformability.
    """
    energy_density, pressure = EoS_import.EOS_import(input_file, density, pressure)

    Radius = []
    Mass = []
    tidal = []
    density = np.logspace(14.3, 15.6, 50) * g_cm_3

    for i in range(len(density)):
        try:
            M, R, T = TOV_solver.solveTOV_tidal(density[i], energy_density, pressure)
            Radius.append(R)
            Mass.append(M)
            tidal.append(T)
        except OverflowError:
            break

    MRT = np.vstack((Radius, Mass, tidal)).T
    return MRT


def OutputC_s(input_file="", density=[], pressure=[]):
    """Calls function to open csv (if needed) and check equation of state validity.
    Then calls function to calculate speed of sound.

    Args:
        file_name (string, optional): string. CSV file to be opened.
        density (array, optional): numpy 1Darray. Passed into a check function and returned if valid.
        pressure (array, optional): numpy 1Darray. Passed into a check function and returned if valid.

    Returns:
        C_s (array): numpy 1D array. List of speeds of sound.
    """
    energy_density, pressure = EoS_import.EOS_import(input_file, density, pressure)
    C_s = speed_of_sound.speed_of_sound_calc(energy_density, pressure)
    return C_s


def OutputMRTpoint(central_density, energy_density, pressure):
    """Outputs the mass, radius, and tidal deformability (single point)

    Args:
        central_density (float): central density that we want to compute
        density (array, optional): numpy 1Darray. Density of EoS
        pressure (array, optional): numpy 1Darray. pressure of EoS

    Returns:
        MRT (tuple): tuple with mass, radius and tidal.
    """
    Radius = []
    Mass = []
    tidal = []
    try:
        M, R, T = TOV_solver.solveTOV_tidal(
            central_density * g_cm_3, energy_density, pressure
        )
        Mass.append(M)
        Radius.append(R)
        tidal.append(T)
    except OverflowError as e:
        print(
            "This EOS is ill-defined to reach an infinity result, that is not phyiscal, No Mass radius will be generated."
        )
    MRT = np.vstack((Radius, Mass, tidal)).T

    return MRT
