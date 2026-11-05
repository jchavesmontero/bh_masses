"""Photometric filters and synthetic photometry.

Synthetic photometry follows Tokunaga & Vacca (2005): the mean flux density
within a band is

.. math:: \\langle F_\\lambda \\rangle =
   \\frac{\\int \\lambda F_\\lambda T(\\lambda) d\\lambda}
         {\\int \\lambda T(\\lambda) d\\lambda}
"""

from dataclasses import dataclass
from importlib import resources

import numpy as np
from scipy.interpolate import interp1d

# np.trapz was renamed to np.trapezoid in numpy 2.0
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class FilterSet:
    """Transmission curves of a set of photometric bands.

    Attributes
    ----------
    wavelength : ndarray, shape (nbands, nsamples)
        Wavelengths [Angstrom] at which each transmission curve is tabulated.
        All bands must be sampled with the same number of points.
    transmission : ndarray, shape (nbands, nsamples)
        Throughput of each band (between 0 and 1).
    names : ndarray of str, shape (nbands,)
        Band names.
    """

    wavelength: np.ndarray
    transmission: np.ndarray
    names: np.ndarray

    def __len__(self):
        return self.wavelength.shape[0]

    @property
    def pivot(self):
        """Pivot wavelength of each band [Angstrom]."""
        return pivot_wavelength(self.wavelength, self.transmission)

    def select(self, names):
        """Return a new :class:`FilterSet` with the bands in ``names`` (in that order)."""
        index = {name: ii for ii, name in enumerate(self.names)}
        try:
            ind = np.array([index[name] for name in names])
        except KeyError as err:
            raise KeyError(f"Band {err} not in filter set") from None
        return FilterSet(self.wavelength[ind], self.transmission[ind], self.names[ind])


def load_jpas_filters(names=None):
    """Load the J-PAS filter system bundled with the package.

    Parameters
    ----------
    names : list of str, optional
        Bands to return, in that order. By default all 56 bands are returned:
        ``uJPAS``, the 54 narrow bands ``J0378`` ... ``J0910``, and ``J1007``.

    Returns
    -------
    FilterSet
    """
    path = resources.files("quasar_lines").joinpath("data/jpas_filters.npz")
    with resources.as_file(path) as fname, np.load(fname) as data:
        filters = FilterSet(
            data["wavelength"].copy(),
            data["transmission"].copy(),
            data["names"].astype(str),
        )
    if names is not None:
        filters = filters.select(names)
    return filters


def pivot_wavelength(filt_wav, filt_trans):
    """Pivot wavelength of each band (Tokunaga & Vacca 2005, Eq. A11)."""
    top = _trapezoid(filt_wav * filt_trans, filt_wav, axis=-1)
    bot = _trapezoid(filt_trans / filt_wav, filt_wav, axis=-1)
    return np.sqrt(top / bot)


def band_norm(filt_wav, filt_trans):
    """Denominator of the synthetic photometry integral, ``int lambda T dlambda``."""
    return _trapezoid(filt_wav * filt_trans, filt_wav, axis=-1)


def band_line_flux(filt_wav, filt_trans, params, profile):
    """Numerator of the synthetic photometry integral for a line profile.

    Parameters
    ----------
    filt_wav, filt_trans : ndarray, shape (nbands, nsamples)
        Filter curves.
    params : array_like
        Parameters of ``profile``.
    profile : callable
        Line profile ``profile(wav, *params)``.
    """
    return _trapezoid(filt_wav * profile(filt_wav, *params) * filt_trans, filt_wav, axis=-1)


def synthetic_photometry(filters, wav, flux):
    """Mean flux density of spectra within each band.

    Parameters
    ----------
    filters : FilterSet
    wav : ndarray, shape (nwav,)
        Wavelength of the spectra [Angstrom]; must cover all bands.
    flux : ndarray, shape (nwav,) or (nspec, nwav)
        Flux density of the spectra.

    Returns
    -------
    ndarray, shape (nspec, nbands)
    """
    flux = np.atleast_2d(flux)
    interp = interp1d(wav, flux, kind="quadratic", axis=1)
    norm = band_norm(filters.wavelength, filters.transmission)
    out = np.zeros((flux.shape[0], len(filters)))
    for ii in range(len(filters)):
        fw = filters.wavelength[ii]
        top = _trapezoid(fw * interp(fw) * filters.transmission[ii], fw, axis=1)
        out[:, ii] = top / norm[ii]
    return out
