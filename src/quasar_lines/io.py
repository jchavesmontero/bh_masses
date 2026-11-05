"""Readers for miniJPAS/J-PAS catalogues and SDSS spectra."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PhotoSpectraCatalog:
    """Photospectra read from a catalogue.

    Attributes
    ----------
    flux, flux_err : ndarray, shape (nsources, nbands)
        Flux density and uncertainty [erg s^-1 cm^-2 A^-1].
    z : ndarray, shape (nsources,)
        Redshift (``NaN`` if not in the catalogue).
    band_names : list of str
        Band of each flux column, e.g. ``"J0378"``.
    table : numpy structured array
        Full catalogue.
    """

    flux: np.ndarray
    flux_err: np.ndarray
    z: np.ndarray
    band_names: list
    table: np.ndarray

    def __len__(self):
        return self.flux.shape[0]


def read_minijpas_catalog(path, band_names=None, error_prefix="e", z_column="zspec"):
    """Read a whitespace-separated miniJPAS flux catalogue.

    The file must have a header line (optionally starting with ``#``) with one
    column per band containing F_lambda [erg s^-1 cm^-2 A^-1], and the
    corresponding uncertainty in a column with the same name prefixed by
    ``error_prefix``.

    Parameters
    ----------
    path : str or path-like
    band_names : list of str, optional
        Bands to read. By default, all columns whose name starts with ``J``
        (the 54 J-PAS narrow bands plus ``J1007``).
    error_prefix : str, optional
    z_column : str, optional
        Column with the redshift.

    Returns
    -------
    PhotoSpectraCatalog
    """
    table = np.atleast_1d(np.genfromtxt(path, names=True))
    if band_names is None:
        band_names = [name for name in table.dtype.names if name.startswith("J")]
    flux = np.array([table[name] for name in band_names]).T
    flux_err = np.array([table[error_prefix + name] for name in band_names]).T
    if z_column in table.dtype.names:
        z = np.asarray(table[z_column], dtype=float)
    else:
        z = np.full(table.shape[0], np.nan)
    return PhotoSpectraCatalog(flux, flux_err, z, list(band_names), table)


def read_sdss_spectrum(path):
    """Read an SDSS ``spec-PLATE-MJD-FIBER.fits`` file.

    Returns
    -------
    wav : ndarray
        Vacuum wavelength [Angstrom].
    flux : ndarray
        Flux density [erg s^-1 cm^-2 A^-1].
    flux_err : ndarray
        Uncertainty (``inf`` where the inverse variance is zero).
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = hdul[1].data
        wav = 10 ** np.asarray(data["LOGLAM"], dtype=float)
        flux = np.asarray(data["FLUX"], dtype=float) * 1e-17
        ivar = np.asarray(data["IVAR"], dtype=float)
    with np.errstate(divide="ignore"):
        flux_err = np.where(ivar > 0, 1e-17 / np.sqrt(ivar), np.inf)
    return wav, flux, flux_err
