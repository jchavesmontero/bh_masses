"""Emission-line profiles.

Line profiles are expressed in units of the continuum, i.e. they model
``F_line / F_continuum`` as a function of observed wavelength.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


def gaussian(wav, amplitude, sigma, center):
    """Gaussian line profile.

    Parameters
    ----------
    wav : array_like
        Observed wavelength [Angstrom].
    amplitude : float
        Peak height in units of the continuum.
    sigma : float
        Standard deviation [Angstrom].
    center : float
        Central wavelength [Angstrom].
    """
    xx = wav - center
    return amplitude * np.exp(-0.5 * xx**2 / sigma**2)


def gaussian_fwhm(amplitude, sigma, center):
    """FWHM [Angstrom] of :func:`gaussian`."""
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma


def _sigmoid(x, x0, k, ylo, yhi):
    return ylo + (yhi - ylo) / (1.0 + np.exp(-k * (x - x0)))


def nagao06(wav, amplitude, exp1, exp2, center):
    """Asymmetric power-law line profile from Nagao et al. (2006).

    The profile is ``amplitude * (wav / center) ** beta`` with ``beta``
    smoothly switching from ``exp1`` (blue wing) to ``exp2`` (red wing).
    """
    kk = 2e-2
    base = wav / center
    expo = _sigmoid(wav, center, kk, exp1, exp2)
    return amplitude * np.power(base, expo)


def nagao06_fwhm(amplitude, exp1, exp2, center):
    """FWHM [Angstrom] of :func:`nagao06`."""
    return center * (0.5 ** (1.0 / exp2) - 0.5 ** (1.0 / exp1))


@dataclass(frozen=True)
class LineProfile:
    """A line profile together with its parameters and default flat priors.

    The last parameter is always the line centre; its prior is set at fit time
    from the bands adjacent to the expected line position.
    """

    name: str
    function: Callable
    fwhm: Callable
    param_names: tuple
    priors: tuple

    @property
    def nparams(self):
        return len(self.param_names)


PROFILES = {
    "gaussian": LineProfile(
        "gaussian",
        gaussian,
        gaussian_fwhm,
        ("amplitude", "sigma", "center"),
        # negative amplitudes allowed to assess the significance of the line
        ((-1.0, 5.0), (0.0, 200.0), (-1.0, 1.0)),
    ),
    "nagao06": LineProfile(
        "nagao06",
        nagao06,
        nagao06_fwhm,
        ("amplitude", "exp1", "exp2", "center"),
        ((-1.0, 5.0), (20.0, 250.0), (-250.0, -20.0), (-1.0, 1.0)),
    ),
}


def get_profile(profile):
    """Return a :class:`LineProfile` from its name (or pass one through)."""
    if isinstance(profile, LineProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError:
        raise KeyError(
            f"Unknown profile {profile!r}; available: {tuple(PROFILES)}"
        ) from None
