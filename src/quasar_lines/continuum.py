"""Power-law fit to the quasar continuum.

The AGN continuum is close to a power law, ``F_lambda ∝ lambda^alpha``, and
there are rest-frame wavelengths almost free of emission-line contamination.
We measure the photospectrum at those anchor wavelengths and fit a power law
to them with MCMC.
"""

from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_COSMOLOGY
from .conversions import flux_to_loglum
from .mcmc import CONTINUUM_MCMC, FlatPriorGaussianLikelihood, run_mcmc

#: Rest-frame wavelengths [Angstrom] used to anchor the continuum.
ANCHOR_WAVELENGTHS = np.array([1350.0, 1700.0, 1800.0, 2200.0, 3100.0, 4000.0, 4200.0, 5200.0])

#: Multiplicative factors that correct each anchor for residual contamination
#: from emission lines and the FeII pseudo-continuum. Only applied when
#: ``correct_contamination=True`` (i.e. for real data, not for simple mocks).
ANCHOR_CORRECTIONS = np.array([0.9, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9])

#: Flat priors on (intercept, slope) of ``ln(F / norm) = a + alpha ln(lambda_rest)``.
CONTINUUM_PRIORS = np.array([[-25.0, 25.0], [-3.5, 3.0]])

#: Placeholder uncertainty used in some miniJPAS catalogues for missing errors.
_MISSING_ERROR = 1e-16


@dataclass
class ContinuumFit:
    """Result of :func:`fit_continuum`.

    Attributes
    ----------
    samples : ndarray, shape (nsamples, 2)
        Posterior samples of the intercept and spectral index ``alpha`` of
        ``ln(F_lambda / norm) = intercept + alpha ln(lambda_rest)``.
    norm : float
        Flux normalisation [erg s^-1 cm^-2 A^-1].
    z : float
        Redshift.
    anchor_wav : ndarray
        Observed wavelength of the anchors used in the fit [Angstrom].
    anchor_flux : ndarray
        Flux of the anchors used in the fit.
    """

    samples: np.ndarray
    norm: float
    z: float
    anchor_wav: np.ndarray
    anchor_flux: np.ndarray

    @property
    def slope(self):
        """Median and 16th/84th percentiles of the spectral index alpha."""
        return np.percentile(self.samples[:, 1], [50, 16, 84])

    def evaluate(self, wav):
        """Continuum flux for every posterior sample.

        Parameters
        ----------
        wav : array_like, shape (nwav,)
            Observed wavelength [Angstrom].

        Returns
        -------
        ndarray, shape (nsamples, nwav)
        """
        lnwav = np.log(np.asarray(wav) / (1 + self.z))
        return np.exp(self.samples[:, :1] + self.samples[:, 1:] * lnwav) * self.norm

    def log_luminosity(self, rest_wav, cosmology=DEFAULT_COSMOLOGY):
        """Median and 16th/84th percentiles of ``log10(lambda L_lambda)`` at ``rest_wav``."""
        flux = self.evaluate(rest_wav * (1 + self.z))[:, 0]
        loglum = flux_to_loglum(rest_wav * (1 + self.z), flux, self.z, cosmology=cosmology)
        return np.percentile(loglum, [50, 16, 84])


def fit_continuum(wav, flux, flux_err, z, correct_contamination=True,
                  edge_margin=75.0, mcmc=CONTINUUM_MCMC, rng=None):
    """Fit a power law to the continuum of a photospectrum.

    For each rest-frame anchor wavelength within the observed range, the
    continuum level is the median of the (positive) fluxes of the closest band
    and its two neighbours.

    Parameters
    ----------
    wav : ndarray, shape (nbands,)
        Pivot wavelength of each band [Angstrom].
    flux, flux_err : ndarray, shape (nbands,)
        Flux density and uncertainty [erg s^-1 cm^-2 A^-1].
    z : float
        Redshift.
    correct_contamination : bool, optional
        Apply :data:`ANCHOR_CORRECTIONS` to the anchors.
    edge_margin : float, optional
        Anchors closer than this [Angstrom] to the edges of the observed range
        are not used.
    mcmc : MCMCSettings, optional
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    ContinuumFit or None
        ``None`` if fewer than two anchors are available.
    """
    rng = np.random.default_rng(rng)
    rwav = wav / (1 + z)
    if correct_contamination:
        corrections = ANCHOR_CORRECTIONS
    else:
        corrections = np.ones_like(ANCHOR_WAVELENGTHS)

    anchor = np.zeros(ANCHOR_WAVELENGTHS.shape[0])
    anchor_err = np.zeros(ANCHOR_WAVELENGTHS.shape[0])
    for ii, awav in enumerate(ANCHOR_WAVELENGTHS):
        owav = awav * (1 + z)
        if not (wav.min() + edge_margin < owav < wav.max() - edge_margin):
            continue
        ind = np.argmin(np.abs(rwav - awav))
        neighbours = flux[max(ind - 1, 0) : ind + 2]
        neighbours = neighbours[neighbours > 0]
        if neighbours.size == 0:
            continue
        anchor[ii] = np.median(neighbours) * corrections[ii]
        if flux_err[ind] == _MISSING_ERROR:
            # use the error of a neighbouring band (the next one for the first band)
            anchor_err[ii] = flux_err[ind - 1] if ind > 0 else flux_err[1]
        else:
            anchor_err[ii] = flux_err[ind]

    use = anchor > 0
    if use.sum() < 2:
        return None

    norm = np.max(anchor[use])
    xx = np.log(ANCHOR_WAVELENGTHS[use])
    yy = np.log(anchor[use] / norm)
    eyy = anchor_err[use] / anchor[use]

    def model(params):
        return params[0] + params[1] * xx

    log_prob = FlatPriorGaussianLikelihood(model, yy, eyy, CONTINUUM_PRIORS)
    sampler = run_mcmc(log_prob, CONTINUUM_PRIORS, mcmc, rng)
    samples = sampler.get_chain(discard=mcmc.nburn, flat=True)

    return ContinuumFit(
        samples=samples,
        norm=norm,
        z=z,
        anchor_wav=ANCHOR_WAVELENGTHS[use] * (1 + z),
        anchor_flux=anchor[use],
    )
