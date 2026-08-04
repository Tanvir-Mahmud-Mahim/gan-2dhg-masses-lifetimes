"""Fundamental physical constants.

CODATA 2018 values, as distributed with scipy.constants.  Kept in one place so
that every module in the package agrees to the last digit and so that the
provenance of each number is explicit.
"""

from scipy import constants as _c

# Fundamental constants (CODATA 2018, via scipy.constants)
Q = _c.elementary_charge          # C
KB = _c.Boltzmann                 # J/K
KB_EV = _c.value("Boltzmann constant in eV/K")   # eV/K
HBAR = _c.hbar                    # J s
H = _c.h                          # J s
M0 = _c.electron_mass             # kg
EPS0 = _c.epsilon_0               # F/m
PI = _c.pi

# Unit helpers -----------------------------------------------------------
CM = 1.0e-2                       # m per cm
CM2 = 1.0e-4                      # m^2 per cm^2
CM3 = 1.0e-6                      # m^3 per cm^3
NM = 1.0e-9                       # m per nm
UM = 1.0e-6                       # m per um


def kT_eV(T: float) -> float:
    """Thermal energy in eV at temperature T in kelvin."""
    return KB_EV * T


def kT_J(T: float) -> float:
    """Thermal energy in joules at temperature T in kelvin."""
    return KB * T


def celsius_to_kelvin(TC):
    """Convert degrees Celsius to kelvin."""
    return TC + 273.15


def kelvin_to_celsius(TK):
    """Convert kelvin to degrees Celsius."""
    return TK - 273.15
