"""quasar_lines: black hole masses from quasar photospectra.

Measure continuum luminosities and broad emission-line properties (Hbeta,
MgII, CIV) from narrow-band photometry such as J-PAS/miniJPAS, and turn them
into single-epoch virial black hole masses (Chaves-Montero et al. 2022).

Quick start::

    import quasar_lines as ql

    cat = ql.read_minijpas_catalog("catalog.cat")
    filters = ql.load_jpas_filters(cat.band_names)
    fit = ql.fit_photospectrum(cat.flux[0], cat.flux_err[0], cat.z[0], filters, rng=0)
    print(fit.summary())
    ql.plot_fit(fit)
"""

from .bh_mass import (
    CALIBRATIONS,
    bolometric_luminosity,
    eddington_ratio,
    eddington_ratio_error,
    virial_mass,
    virial_mass_error,
)
from .constants import DEFAULT_COSMOLOGY, LINE_NAMES, LINES
from .continuum import ContinuumFit, fit_continuum
from .filters import FilterSet, load_jpas_filters, pivot_wavelength, synthetic_photometry
from .fitting import PhotoSpectrumFit, fit_catalog, fit_photospectrum
from .io import PhotoSpectraCatalog, read_minijpas_catalog, read_sdss_spectrum
from .lines import LineFit, fit_line
from .mcmc import MCMCSettings
from .plotting import plot_fit, plot_line_posterior

__version__ = "1.0.0"

__all__ = [
    "CALIBRATIONS",
    "DEFAULT_COSMOLOGY",
    "LINES",
    "LINE_NAMES",
    "ContinuumFit",
    "FilterSet",
    "LineFit",
    "MCMCSettings",
    "PhotoSpectraCatalog",
    "PhotoSpectrumFit",
    "bolometric_luminosity",
    "eddington_ratio",
    "eddington_ratio_error",
    "fit_catalog",
    "fit_continuum",
    "fit_line",
    "fit_photospectrum",
    "load_jpas_filters",
    "pivot_wavelength",
    "plot_fit",
    "plot_line_posterior",
    "read_minijpas_catalog",
    "read_sdss_spectrum",
    "synthetic_photometry",
    "virial_mass",
    "virial_mass_error",
]
