"""Physical constants, emission-line definitions, and default cosmology."""

from dataclasses import dataclass

import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM

#: Speed of light in km/s.
C_KMS = const.c.to("km/s").value

#: Speed of light in Angstrom/s (used for AB magnitude conversions).
C_AA_S = 2.99792458e18

#: Default cosmology (Planck-like), the one used in Chaves-Montero et al. (2022).
DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=67.26, Om0=0.314, Ob0=0.049)


@dataclass(frozen=True)
class EmissionLine:
    """Definition of a broad emission line used for virial mass estimates.

    Attributes
    ----------
    name : str
        Short identifier (``"Hbeta"``, ``"MgII"``, ``"CIV"``).
    label : str
        LaTeX label for plots.
    rest_wavelength : float
        Rest-frame central wavelength in Angstrom.
    fit_window : tuple of float
        Rest-frame wavelength range (Angstrom) of the bands used to fit the line.
        It is wide enough (~4e4 km/s) to resolve lines up to ~1e4 km/s.
    continuum_wavelength : float
        Rest-frame wavelength (Angstrom) at which the continuum luminosity
        entering the virial estimator is measured.
    """

    name: str
    label: str
    rest_wavelength: float
    fit_window: tuple
    continuum_wavelength: float


#: Emission lines analysed by the pipeline, in the order used by all outputs.
LINES = (
    EmissionLine("Hbeta", r"H$\beta$", 4862.68, (4700.0, 5100.0), 5100.0),
    EmissionLine("MgII", "MgII", 2799.12, (2600.0, 3000.0), 3000.0),
    EmissionLine("CIV", "CIV", 1549.48, (1450.0, 1630.0), 1350.0),
)

LINE_NAMES = tuple(line.name for line in LINES)


def get_line(name):
    """Return the :class:`EmissionLine` called ``name``."""
    for line in LINES:
        if line.name == name:
            return line
    raise KeyError(f"Unknown line {name!r}; available: {LINE_NAMES}")
