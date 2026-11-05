"""Fit of broad emission lines in continuum-normalised photospectra.

Each line is modelled as a profile (Gaussian by default) in units of the
continuum, ``F / F_cont - 1``, which is convolved with the transmission curves
of the bands around the line and compared with the data via MCMC.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.stats import gaussian_kde

from .conversions import width_aa_to_kms
from .filters import band_line_flux, band_norm
from .mcmc import LINE_MCMC, FlatPriorGaussianLikelihood, converged_walkers, run_mcmc
from .profiles import get_profile

#: Contaminating lines whose bands are down-weighted when fitting each line:
#: ``(rest wavelengths, correction for the closest and second-closest band)``.
#: [OIII] 4960, 5008 for Hbeta; FeII bumps for MgII; none for CIV.
CONTAMINANTS = {
    "Hbeta": (np.array([4960.30, 5008.54]), np.array([0.75, 0.5])),
    "MgII": (np.array([2700.0, 2950.0]), np.array([0.5, 0.5])),
    "CIV": (np.array([]), np.array([])),
}


@dataclass
class LineFit:
    """Result of :func:`fit_line`.

    ``fwhm``, ``ew``, and ``log_lum_cont`` are ``[median, 16th, 84th]``
    percentiles of the posterior. ``center`` is different: it is
    ``[peak, lower, upper]``, where ``peak`` is the mode of the posterior of the
    line centre and ``lower``/``upper`` are where the posterior drops to half
    its maximum (see :func:`line_center`).

    Attributes
    ----------
    name : str
        Line identifier.
    profile : LineProfile
        Line profile used in the fit.
    samples : ndarray, shape (nsamples, nparams)
        Posterior samples of converged walkers after burn-in.
    best_params : ndarray, shape (nparams,)
        Maximum-likelihood parameters.
    band_index : ndarray of int
        Bands whose data were fitted (excluding the two continuum pads).
    continuum_offset : float
        Apparent continuum level (``F / F_cont - 1``) adopted in the fit.
    center : ndarray
        Rest-frame line centre [Angstrom]: ``[peak, lower, upper]`` (half
        maximum of the posterior, *not* percentiles).
    fwhm : ndarray
        FWHM [km/s].
    ew : ndarray
        Rest-frame equivalent width [Angstrom].
    log_lum_cont : ndarray
        ``log10(lambda L_lambda / erg s^-1)`` of the continuum at the line's
        :attr:`~quasar_lines.constants.EmissionLine.continuum_wavelength`.
    """

    name: str
    profile: object
    samples: np.ndarray
    best_params: np.ndarray
    band_index: np.ndarray
    continuum_offset: float
    center: np.ndarray
    fwhm: np.ndarray
    ew: np.ndarray
    log_lum_cont: np.ndarray = None

    @property
    def params(self):
        """Median and 16th/84th percentiles of each parameter, shape (nparams, 3)."""
        return np.percentile(self.samples, [50, 16, 84], axis=0).T


def locate_lines(wav, z, lines, edge_bands=2):
    """Band closest to each line, or -1 if the line is outside the observed range.

    Lines must lie between the centres of band ``edge_bands - 1`` and band
    ``-edge_bands`` (with half a band spacing of margin) so that there are
    bands on both sides of the line.
    """
    half = 0.5 * (wav[2] - wav[1])
    out = np.full(len(lines), -1, dtype=int)
    for ii, line in enumerate(lines):
        owav = line.rest_wavelength * (1 + z)
        if wav[edge_bands - 1] + half <= owav <= wav[-edge_bands] - half:
            out[ii] = np.argmin(np.abs(owav - wav))
    return out


def fit_line(line, wav, flux, flux_err, cont, cont_err, z, filters, profile="gaussian",
             correct_contamination=True, amplitude_max=None, mcmc=LINE_MCMC, rng=None):
    """Fit one broad emission line.

    Parameters
    ----------
    line : EmissionLine
    wav : ndarray, shape (nbands,)
        Pivot wavelength of each band [Angstrom].
    flux, flux_err : ndarray, shape (nbands,)
        Photospectrum and uncertainty.
    cont, cont_err : ndarray, shape (nbands,)
        Continuum model and its uncertainty in each band.
    z : float
        Redshift.
    filters : FilterSet
        Filters matching ``wav``.
    profile : str or LineProfile, optional
        Line profile (``"gaussian"`` or ``"nagao06"``).
    correct_contamination : bool, optional
        Down-weight bands affected by nearby lines (see :data:`CONTAMINANTS`).
    amplitude_max : float, optional
        Upper prior on the line amplitude; defaults to that of ``profile``.
    mcmc : MCMCSettings, optional
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    LineFit or None
        ``None`` if the line is outside the observed range.
    """
    rng = np.random.default_rng(rng)
    profile = get_profile(profile)
    nbands = wav.shape[0]
    half = 0.5 * (wav[2] - wav[1])

    peak = locate_lines(wav, z, [line])[0]
    if peak == -1:
        return None

    owin = np.array(line.fit_window) * (1 + z)
    ind_bands = np.flatnonzero((wav > owin[0] - half) & (wav < owin[1] + half))

    priors = np.array(profile.priors)
    if amplitude_max is not None:
        priors[0, 1] = amplitude_max
    priors[-1] = [wav[peak - 1], wav[peak + 1]]

    xx = wav[ind_bands]
    # continuum-normalised line flux and its uncertainty
    yy = flux[ind_bands] / cont[ind_bands] - 1
    eyy = np.sqrt(flux_err[ind_bands] ** 2 + (yy + 1) ** 2 * cont_err[ind_bands] ** 2)
    eyy /= cont[ind_bands]

    # apparent continuum level; clipped at zero to reduce the impact of absorptions
    level = max(yy.min(), 0.0)

    if correct_contamination:
        cwav, fcorr = CONTAMINANTS[line.name]
        for oth in cwav:
            dist = np.abs(xx - oth * (1 + z))
            closest = np.argsort(dist)
            for jj in range(fcorr.shape[0]):
                kk = closest[jj]
                if dist[kk] >= 1.5 * half:
                    continue
                if jj == 0 and oth != 4960.30:
                    yy[kk] = level
                else:
                    yy[kk] = max(yy[kk] * fcorr[jj], level)

    # pad with one band on each side at the continuum level; this anchors the
    # continuum and improves the fit
    pad_err = 0.25 * np.min(eyy)
    use = np.concatenate([[ind_bands[0] - 1], ind_bands, [ind_bands[-1] + 1]])
    if use[0] >= 0:
        yy = np.concatenate([[level], yy])
        eyy = np.concatenate([[pad_err], eyy])
    if use[-1] < nbands:
        yy = np.concatenate([yy, [level]])
        eyy = np.concatenate([eyy, [pad_err]])
    use = use[(use >= 0) & (use < nbands)]

    fwav = filters.wavelength[use]
    ftrans = filters.transmission[use]
    fnorm = band_norm(fwav, ftrans)

    def model(params):
        return band_line_flux(fwav, ftrans, params, profile.function) / fnorm + level

    log_prob = FlatPriorGaussianLikelihood(model, yy, eyy, priors)
    sampler = run_mcmc(log_prob, priors, mcmc, rng)

    # emcee returns arrays of shape (nsteps, nwalkers, ...)
    chain = sampler.get_chain().swapaxes(0, 1)
    log_prob = sampler.get_log_prob().T
    keep = converged_walkers(log_prob, mcmc.nburn)
    samples = chain[keep, mcmc.nburn :, :].reshape(-1, profile.nparams)
    best = chain[log_prob == np.max(log_prob)][0]

    ocen = line_center(samples[:, -1])
    fwhm_aa = np.percentile(profile.fwhm(*samples.T), [50, 16, 84])

    return LineFit(
        name=line.name,
        profile=profile,
        samples=samples,
        best_params=best,
        band_index=ind_bands,
        continuum_offset=level,
        center=ocen / (1 + z),
        fwhm=width_aa_to_kms(ocen[0], fwhm_aa),
        ew=line_ew(samples, profile.function, ocen[0], fwhm_aa[0], z),
    )


def line_center(center_samples, npoints=1000):
    """Peak of the posterior of the line centre and where it drops to half.

    The posterior of the centre is often multimodal or truncated by its prior
    (the centre is restricted to lie between the bands adjacent to the line),
    so its mode is a more robust estimate than the median. Its width is
    described by the full width at half maximum of the posterior, estimated
    with a Gaussian kernel density estimate; ``upper - lower`` is therefore an
    FWHM, roughly 2.35 sigma for a Gaussian posterior, not a 68% interval.

    Returns
    -------
    ndarray
        ``[peak, lower half-maximum, upper half-maximum]``.
    """
    xx = np.linspace(center_samples.min() * 0.98, center_samples.max() * 1.02, npoints)
    pdf = gaussian_kde(center_samples)(xx)
    pdf /= pdf.max()
    peak = xx[np.argmax(pdf)]
    upper = np.min(xx[(pdf < 0.5) & (xx > peak)])
    lower = np.max(xx[(pdf < 0.5) & (xx < peak)])
    return np.array([peak, lower, upper])


def line_ew(samples, profile, center, fwhm, z, npoints=100, nwidths=5):
    """Rest-frame equivalent width [Angstrom], ``[median, 16th, 84th]``.

    The continuum-normalised profile is integrated over ``center ± nwidths * fwhm``.
    """
    owav = np.linspace(center - nwidths * fwhm, center + nwidths * fwhm, npoints)
    params = [p[:, None] for p in samples.T]
    ew = simpson(profile(owav[None, :], *params), x=owav / (1 + z), axis=1)
    return np.percentile(ew, [50, 16, 84])
