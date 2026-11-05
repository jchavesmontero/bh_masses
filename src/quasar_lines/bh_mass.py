"""Virial black hole masses, bolometric luminosities, and Eddington ratios.

Single-epoch virial masses are computed as

.. math:: \\log(M_\\mathrm{BH}/M_\\odot) = A + B\\,[\\log(\\lambda L_\\lambda / \\mathrm{erg\\,s^{-1}}) - 44]
          + 2 \\log(\\mathrm{FWHM} / \\mathrm{km\\,s^{-1}})

using the continuum luminosity at 5100, 3000, and 1350 Angstrom for Hbeta,
MgII, and CIV, respectively.
"""

import numpy as np

from .constants import LINE_NAMES

#: Virial coefficients ``{calibration: {line: (A, B)}}``.
#:
#: ``"shen11"``
#:     Coefficients used by SDSS quasar catalogues (Shen et al. 2011; Rakshit et
#:     al. 2020): Vestergaard & Peterson (2006) for Hbeta and CIV, and Shen et
#:     al. (2011) / Trakhtenbrot & Netzer (2012) for MgII.
#: ``"chaves-montero22"``
#:     Recalibration for single-epoch *photometry* derived in Chaves-Montero et
#:     al. (2022) by matching miniJPAS photometric masses to SDSS spectroscopic
#:     ones. It absorbs systematic differences between the two methods.
CALIBRATIONS = {
    "shen11": {
        "Hbeta": (0.91, 0.50),
        "MgII": (0.74, 0.62),
        "CIV": (0.66, 0.53),
    },
    "chaves-montero22": {
        "Hbeta": (0.87577631, 0.2631655),
        "MgII": (0.2921838, 0.58968014),
        "CIV": (0.89396922, 0.26316507),
    },
}

DEFAULT_CALIBRATION = "chaves-montero22"

#: Bolometric corrections ``L_bol = BC * lambda L_lambda`` (Rakshit et al. 2020)
#: at 5100, 3000, and 1350 Angstrom.
BOLOMETRIC_CORRECTIONS = {"Hbeta": 9.26, "MgII": 5.15, "CIV": 3.81}


def _coefficients(line, calibration):
    if line not in LINE_NAMES:
        raise KeyError(f"Unknown line {line!r}; available: {LINE_NAMES}")
    try:
        return CALIBRATIONS[calibration][line]
    except KeyError:
        raise KeyError(
            f"Unknown calibration {calibration!r}; available: {tuple(CALIBRATIONS)}"
        ) from None


def virial_mass(log_lum, log_fwhm, line, calibration=DEFAULT_CALIBRATION):
    """Single-epoch virial black hole mass, ``log10(M_BH / M_sun)``.

    Parameters
    ----------
    log_lum : float or array_like
        ``log10(lambda L_lambda / erg s^-1)`` of the continuum.
    log_fwhm : float or array_like
        ``log10(FWHM / km s^-1)`` of the line.
    line : {"Hbeta", "MgII", "CIV"}
    calibration : str, optional
        Key of :data:`CALIBRATIONS`.
    """
    aa, bb = _coefficients(line, calibration)
    return aa + bb * (np.asarray(log_lum) - 44.0) + 2.0 * np.asarray(log_fwhm)


def virial_mass_error(log_lum_err, log_fwhm_err, line, calibration=DEFAULT_CALIBRATION):
    """Measurement uncertainty of :func:`virial_mass` [dex].

    Only propagates the errors on luminosity and FWHM; the intrinsic scatter of
    virial estimators (~0.3-0.4 dex) is not included.
    """
    _, bb = _coefficients(line, calibration)
    return np.sqrt((bb * np.asarray(log_lum_err)) ** 2 + (2.0 * np.asarray(log_fwhm_err)) ** 2)


def bolometric_luminosity(log_lum, line):
    """``log10(L_bol / erg s^-1)`` from the continuum luminosity used for ``line``."""
    return np.asarray(log_lum) + np.log10(BOLOMETRIC_CORRECTIONS[line])


def eddington_ratio(log_mbh, log_lbol):
    """``log10(L_bol / L_Edd)`` with ``L_Edd = 1.257e38 (M_BH / M_sun) erg/s``."""
    return np.asarray(log_lbol) - (np.log10(1.257e38) + np.asarray(log_mbh))


def eddington_ratio_error(log_lum_err, log_fwhm_err, line, calibration=DEFAULT_CALIBRATION):
    """Measurement uncertainty of the Eddington ratio [dex].

    The black hole mass and the bolometric luminosity both depend on the
    continuum luminosity, so their errors are correlated. Since
    ``log(L_bol/L_Edd) = (1 - B) log(lambda L_lambda) - 2 log(FWHM) + const``,
    the uncertainty is ``sqrt([(1 - B) err_L]^2 + [2 err_FWHM]^2)``.
    """
    _, bb = _coefficients(line, calibration)
    return np.sqrt(((1.0 - bb) * np.asarray(log_lum_err)) ** 2
                   + (2.0 * np.asarray(log_fwhm_err)) ** 2)
