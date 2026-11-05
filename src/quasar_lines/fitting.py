"""High-level interface: fit a photospectrum and derive black hole properties."""

from dataclasses import dataclass, field

import numpy as np

from . import bh_mass
from .constants import DEFAULT_COSMOLOGY, LINE_NAMES, get_line
from .continuum import ContinuumFit, fit_continuum
from .filters import FilterSet
from .lines import LineFit, fit_line
from .mcmc import CONTINUUM_MCMC, LINE_MCMC
from .profiles import get_profile


@dataclass
class PhotoSpectrumFit:
    """Result of :func:`fit_photospectrum`.

    Attributes
    ----------
    wav, flux, flux_err : ndarray
        Input photospectrum (pivot wavelength [Angstrom] and flux density).
    z : float
        Redshift.
    filters : FilterSet
        Filters of the photospectrum.
    continuum : ContinuumFit or None
        Power-law continuum fit (``None`` if it failed).
    cont, cont_err : ndarray
        Median and standard deviation of the continuum model in each band.
    lines : dict of {str: LineFit}
        Fits of the lines within the observed range.
    """

    wav: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    z: float
    filters: FilterSet
    continuum: ContinuumFit = None
    cont: np.ndarray = None
    cont_err: np.ndarray = None
    lines: dict = field(default_factory=dict)

    def black_hole_mass(self, calibration=bh_mass.DEFAULT_CALIBRATION):
        """Virial black hole mass from each fitted line.

        Returns
        -------
        dict of {str: (float, float)}
            ``log10(M_BH / M_sun)`` and its measurement uncertainty per line.
        """
        out = {}
        for name, lfit in self.lines.items():
            values = _log_values(lfit)
            if values is None:
                continue
            lum, lum_err, log_fwhm, fwhm_err = values
            out[name] = (
                float(bh_mass.virial_mass(lum, log_fwhm, name, calibration)),
                float(bh_mass.virial_mass_error(lum_err, fwhm_err, name, calibration)),
            )
        return out

    def eddington_ratio(self, calibration=bh_mass.DEFAULT_CALIBRATION):
        """``log10(L_bol / L_Edd)`` from each fitted line.

        The uncertainty accounts for the correlation between black hole mass
        and bolometric luminosity, which both depend on the continuum
        luminosity (see :func:`~quasar_lines.bh_mass.eddington_ratio_error`).
        """
        out = {}
        for name, (mbh, _) in self.black_hole_mass(calibration).items():
            lum, lum_err, _, fwhm_err = _log_values(self.lines[name])
            lbol = bh_mass.bolometric_luminosity(lum, name)
            err = bh_mass.eddington_ratio_error(lum_err, fwhm_err, name, calibration)
            out[name] = (float(bh_mass.eddington_ratio(mbh, lbol)), float(err))
        return out

    def summary(self, calibration=bh_mass.DEFAULT_CALIBRATION):
        """Human-readable summary of the fit."""
        rows = [f"Photospectrum fit at z = {self.z:.4f}"]
        if self.continuum is None:
            rows.append("  continuum fit failed")
            return "\n".join(rows)
        alpha = self.continuum.slope
        rows.append(
            f"  continuum slope alpha = {alpha[0]:.2f} (+{alpha[2] - alpha[0]:.2f} / -{alpha[0] - alpha[1]:.2f})"
        )
        if not self.lines:
            rows.append("  no line within the observed range")
        masses = self.black_hole_mass(calibration)
        for name, lfit in self.lines.items():
            line = get_line(name)
            rows.append(f"  {name}:")
            peak, lo, hi = lfit.center
            rows.append(
                f"    center      = {peak:.1f} A (rest frame; posterior above half max: {lo:.1f}-{hi:.1f})"
            )
            rows.append(f"    FWHM        = {_fmt(lfit.fwhm, 0)} km/s")
            rows.append(f"    EW          = {_fmt(lfit.ew, 1)} A (rest frame)")
            rows.append(
                f"    log L{int(line.continuum_wavelength)} = {_fmt(lfit.log_lum_cont, 2)} erg/s"
            )
            if name in masses:
                mbh, err = masses[name]
                rows.append(f"    log M_BH    = {mbh:.2f} +/- {err:.2f} M_sun ({calibration})")
        return "\n".join(rows)


def _log_values(lfit):
    """Log continuum luminosity and FWHM of a line fit and their 1-sigma errors."""
    lum, lum_lo, lum_hi = lfit.log_lum_cont
    fwhm, fwhm_lo, fwhm_hi = lfit.fwhm
    if fwhm <= 0:
        return None
    lum_err = 0.5 * (lum_hi - lum_lo)
    fwhm_err = 0.5 * (np.log10(fwhm_hi) - np.log10(max(fwhm_lo, 1e-10)))
    return lum, lum_err, np.log10(fwhm), fwhm_err


def _fmt(values, ndec):
    med, lo, hi = values
    return f"{med:.{ndec}f} (+{hi - med:.{ndec}f} / -{med - lo:.{ndec}f})"


def fit_photospectrum(flux, flux_err, z, filters, lines=LINE_NAMES, profile="gaussian",
                      correct_contamination=True, continuum_mcmc=CONTINUUM_MCMC,
                      line_mcmc=LINE_MCMC, cosmology=DEFAULT_COSMOLOGY, rng=None):
    """Measure continuum and broad-line properties from a photospectrum.

    The fit proceeds in two steps:

    1. A power law is fitted to rest-frame continuum anchors
       (:func:`~quasar_lines.continuum.fit_continuum`).
    2. Each of Hbeta, MgII, and CIV within the observed range is fitted in the
       continuum-normalised photospectrum, forward-modelling the line profile
       through the filter transmission curves
       (:func:`~quasar_lines.lines.fit_line`).

    Parameters
    ----------
    flux, flux_err : ndarray, shape (nbands,)
        Flux density and uncertainty [erg s^-1 cm^-2 A^-1] in each band.
    z : float
        Spectroscopic (or accurate photometric) redshift.
    filters : FilterSet
        Filters matching ``flux``, sorted by wavelength. The pivot wavelengths
        should be roughly evenly spaced (e.g. the J-PAS narrow bands).
    lines : sequence of str, optional
        Lines to fit (subset of ``("Hbeta", "MgII", "CIV")``).
    profile : {"gaussian", "nagao06"}, optional
        Line profile.
    correct_contamination : bool, optional
        Correct for contamination from FeII, [OIII], etc. Set to ``False`` for
        idealised mocks without those features.
    continuum_mcmc, line_mcmc : MCMCSettings, optional
        emcee settings of each step.
    cosmology : astropy.cosmology.Cosmology, optional
        Cosmology used to compute luminosities.
    rng : numpy.random.Generator or int, optional
        Random seed for reproducible results.

    Returns
    -------
    PhotoSpectrumFit
    """
    rng = np.random.default_rng(rng)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    wav = filters.pivot
    if flux.shape != wav.shape or flux_err.shape != wav.shape:
        raise ValueError(
            f"flux {flux.shape} and flux_err {flux_err.shape} must match the "
            f"number of filters {wav.shape}"
        )
    profile = get_profile(profile)

    result = PhotoSpectrumFit(wav=wav, flux=flux, flux_err=flux_err, z=z, filters=filters)

    cfit = fit_continuum(wav, flux, flux_err, z, correct_contamination=correct_contamination,
                         mcmc=continuum_mcmc, rng=rng)
    if cfit is None:
        return result
    cont = cfit.evaluate(wav)
    result.continuum = cfit
    result.cont = np.median(cont, axis=0)
    result.cont_err = np.std(cont, axis=0)

    # allow for stronger lines at high redshift
    amplitude_max = 5.0 if z > 3 else 3.0

    for name in lines:
        line = get_line(name)
        lfit = fit_line(line, wav, flux, flux_err, result.cont, result.cont_err, z, filters,
                        profile=profile, correct_contamination=correct_contamination,
                        amplitude_max=amplitude_max, mcmc=line_mcmc, rng=rng)
        if lfit is None:
            continue
        lfit.log_lum_cont = cfit.log_luminosity(line.continuum_wavelength, cosmology=cosmology)
        result.lines[name] = lfit

    return result


def fit_catalog(flux, flux_err, z, filters, verbose=True, **kwargs):
    """Fit many photospectra and collect the results in a structured array.

    Parameters
    ----------
    flux, flux_err : ndarray, shape (nsources, nbands)
    z : ndarray, shape (nsources,)
    filters : FilterSet
    verbose : bool, optional
        Print progress.
    **kwargs
        Passed to :func:`fit_photospectrum` (``calibration`` selects the
        virial calibration).

    Returns
    -------
    ndarray
        Structured array with one entry per source. Line quantities have shape
        ``(3, ...)`` following the order Hbeta, MgII, CIV; percentile-type
        quantities store ``[median, 16th, 84th]``. Entries for lines not
        fitted are zero.
    """
    calibration = kwargs.pop("calibration", bh_mass.DEFAULT_CALIBRATION)
    profile = get_profile(kwargs.get("profile", "gaussian"))
    nlines = len(LINE_NAMES)
    dtype = np.dtype([
        ("z", np.float64),
        ("cont_slope", np.float64, (3,)),
        ("center", np.float64, (nlines, 3)),
        ("fwhm", np.float64, (nlines, 3)),
        ("ew", np.float64, (nlines, 3)),
        ("log_lum_cont", np.float64, (nlines, 3)),
        ("best_params", np.float64, (nlines, profile.nparams)),
        ("params", np.float64, (nlines, profile.nparams, 3)),
        ("log_mbh", np.float64, (nlines, 2)),
    ])
    flux = np.atleast_2d(flux)
    flux_err = np.atleast_2d(flux_err)
    z = np.atleast_1d(z)
    out = np.zeros(flux.shape[0], dtype=dtype)
    out["z"] = z
    for ii in range(flux.shape[0]):
        if verbose:
            print(f"Fitting source {ii + 1}/{flux.shape[0]} (z = {z[ii]:.3f})", flush=True)
        res = fit_photospectrum(flux[ii], flux_err[ii], z[ii], filters, **kwargs)
        if res.continuum is None:
            continue
        out["cont_slope"][ii] = res.continuum.slope
        masses = res.black_hole_mass(calibration)
        for jj, name in enumerate(LINE_NAMES):
            if name not in res.lines:
                continue
            lfit = res.lines[name]
            out["center"][ii, jj] = lfit.center
            out["fwhm"][ii, jj] = lfit.fwhm
            out["ew"][ii, jj] = lfit.ew
            out["log_lum_cont"][ii, jj] = lfit.log_lum_cont
            out["best_params"][ii, jj] = lfit.best_params
            out["params"][ii, jj] = lfit.params
            if name in masses:
                out["log_mbh"][ii, jj] = masses[name]
    return out
