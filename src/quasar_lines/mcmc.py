"""Helpers to run emcee with flat priors."""

from dataclasses import dataclass

import emcee
import numpy as np
from scipy.stats import qmc


@dataclass
class MCMCSettings:
    """Configuration of an emcee run.

    Attributes
    ----------
    nwalkers : int
        Number of walkers.
    nsteps : int
        Number of steps per walker.
    nburn : int
        Number of initial steps discarded as burn-in.
    """

    nwalkers: int
    nsteps: int
    nburn: int


#: Defaults used for the power-law continuum fit.
CONTINUUM_MCMC = MCMCSettings(nwalkers=100, nsteps=150, nburn=75)

#: Defaults used for each emission-line fit.
LINE_MCMC = MCMCSettings(nwalkers=75, nsteps=500, nburn=150)


class FlatPriorGaussianLikelihood:
    """Gaussian log-likelihood with flat (top-hat) priors.

    Parameters
    ----------
    model : callable
        ``model(params) -> prediction`` with the same shape as ``data``.
    data, error : ndarray
        Measurements and their 1-sigma uncertainties.
    priors : ndarray, shape (nparams, 2)
        Lower and upper limit of each parameter.
    """

    def __init__(self, model, data, error, priors):
        self.model = model
        self.data = np.asarray(data)
        self.var = np.square(error)
        self.priors = np.asarray(priors)

    def __call__(self, params):
        if not np.all((params > self.priors[:, 0]) & (params < self.priors[:, 1])):
            return -np.inf
        return -0.5 * np.sum(np.square(self.model(params) - self.data) / self.var)


def initial_walkers(priors, nwalkers, rng, margin=0.1):
    """Latin-hypercube initial positions within the priors.

    Walkers are placed in the inner part of each prior range, leaving a
    fraction ``margin`` of the range free on both sides.
    """
    priors = np.asarray(priors)
    lo, hi = priors[:, 0], priors[:, 1]
    unit = qmc.LatinHypercube(d=priors.shape[0], seed=rng).random(nwalkers)
    return lo + (hi - lo) * (margin + (1.0 - 2.0 * margin) * unit)


def run_mcmc(log_prob, priors, settings, rng):
    """Run emcee and return the sampler.

    Parameters
    ----------
    log_prob : callable
    priors : ndarray, shape (nparams, 2)
    settings : MCMCSettings
    rng : numpy.random.Generator
    """
    priors = np.asarray(priors)
    p0 = initial_walkers(priors, settings.nwalkers, rng)
    sampler = emcee.EnsembleSampler(settings.nwalkers, priors.shape[0], log_prob)
    sampler.random_state = np.random.RandomState(rng.integers(2**32 - 1)).get_state()
    sampler.run_mcmc(p0, settings.nsteps)
    return sampler


def converged_walkers(log_prob, nburn, nsplit=5, max_diff=3.0):
    """Flag walkers that have converged.

    The post-burn-in chain of each walker is split into ``nsplit`` pieces; a
    walker is kept if the mean log-probability of every piece is within
    ``max_diff`` of the best mean log-probability among all walkers.

    Parameters
    ----------
    log_prob : ndarray, shape (nwalkers, nsteps)
        Log-probability of each walker and step.

    Returns
    -------
    ndarray of bool, shape (nwalkers,)
    """
    pieces = np.array_split(log_prob[:, nburn:], nsplit, axis=1)
    means = np.array([piece.mean(axis=1) for piece in pieces]).T
    best = np.max(means.mean(axis=1))
    return (np.abs(means - best) < max_diff).all(axis=1)
