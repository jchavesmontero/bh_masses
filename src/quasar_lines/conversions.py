"""Unit conversions: magnitudes, fluxes, luminosities, and line widths."""

import numpy as np

from .constants import C_AA_S, C_KMS, DEFAULT_COSMOLOGY


def mag_to_flux(mag, pivot_wav):
    """AB magnitude to flux density F_lambda [erg s^-1 cm^-2 A^-1]."""
    fnu = 10 ** (-(mag + 48.6) / 2.5)
    return fnu * C_AA_S / pivot_wav**2


def flux_to_mag(flux, pivot_wav):
    """Flux density F_lambda [erg s^-1 cm^-2 A^-1] to AB magnitude."""
    fnu = flux * pivot_wav**2 / C_AA_S
    return -2.5 * np.log10(fnu) - 48.6


def flux_to_loglum(obs_wav, flux, z, cosmology=DEFAULT_COSMOLOGY):
    """Monochromatic luminosity ``log10(lambda L_lambda / erg s^-1)``.

    Uses ``lambda_e L_lambda(lambda_e) = 4 pi D_L^2 lambda_o F_lambda(lambda_o)``
    (Hogg et al. 2002).

    Parameters
    ----------
    obs_wav : float or array_like
        Observed-frame wavelength [Angstrom].
    flux : float or array_like
        Flux density at ``obs_wav`` [erg s^-1 cm^-2 A^-1].
    z : float
        Redshift.
    cosmology : astropy.cosmology.Cosmology, optional
    """
    dl = cosmology.luminosity_distance(z).to("cm").value
    return np.log10(4.0 * np.pi * dl**2 * obs_wav * flux)


def mag_to_loglum(mag, z, pivot_wav=6245.36, rest_wav=3000.0, slope=-1.56,
                  cosmology=DEFAULT_COSMOLOGY):
    """Rough continuum luminosity at ``rest_wav`` from a broad-band magnitude.

    The flux is extrapolated from the band pivot wavelength assuming
    ``F_lambda ∝ lambda^slope``; the default slope is that of the Vanden Berk
    et al. (2001) composite and the default pivot is that of SDSS r.
    """
    obs_wav = rest_wav * (1 + z)
    flux = mag_to_flux(mag, pivot_wav) * (obs_wav / pivot_wav) ** slope
    return flux_to_loglum(obs_wav, flux, z, cosmology=cosmology)


def width_aa_to_kms(center, width_aa):
    """Convert a line width from Angstrom to km/s (``dv = c dlambda / lambda``)."""
    return width_aa / center * C_KMS


def width_kms_to_aa(center, width_kms):
    """Convert a line width from km/s to Angstrom."""
    return center * width_kms / C_KMS
