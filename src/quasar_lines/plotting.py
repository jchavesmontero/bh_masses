"""Plots of photospectrum fits."""

import matplotlib.pyplot as plt
import numpy as np

from .constants import get_line
from .filters import band_line_flux, band_norm

FLUX_LABEL = r"$F_\lambda\ [\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}]$"
WAV_LABEL = r"Observed wavelength $[\mathrm{\AA}]$"


def plot_fit(fit, ax=None, spectrum=None, spectrum_label="Spectrum", nwidths=6, npoints=200):
    """Plot a photospectrum together with the best-fitting model.

    Parameters
    ----------
    fit : PhotoSpectrumFit
        Output of :func:`~quasar_lines.fitting.fit_photospectrum`.
    ax : matplotlib.axes.Axes, optional
    spectrum : tuple of ndarray, optional
        ``(wavelength, flux)`` of a spectrum of the same source (e.g. from
        SDSS) to show for comparison.
    spectrum_label : str, optional
    nwidths : float, optional
        Half-width, in units of the FWHM, of the high-resolution line model.
    npoints : int, optional
        Number of points of the high-resolution line model.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if spectrum is not None:
        swav, sflux = spectrum
        ax.plot(swav, sflux, color="0.6", lw=0.5, alpha=0.8, label=spectrum_label)

    ax.errorbar(fit.wav, fit.flux, fit.flux_err, fmt="o", ms=3, color="k", lw=1,
                label="Photospectrum")

    if fit.continuum is not None:
        ax.fill_between(fit.wav, fit.cont - fit.cont_err, fit.cont + fit.cont_err,
                        color="C0", alpha=0.3, lw=0)
        ax.plot(fit.wav, fit.cont, color="C0", label="Continuum (power law)")
        ax.plot(fit.continuum.anchor_wav, fit.continuum.anchor_flux, "s", mfc="none",
                color="C0", ms=8, label="Continuum anchors")

    for jj, (name, lfit) in enumerate(fit.lines.items()):
        color = f"C{jj + 1}"
        line = get_line(name)
        best = lfit.best_params
        func = lfit.profile.function

        # high-resolution model
        ocen = lfit.center[0] * (1 + fit.z)
        owidth = lfit.profile.fwhm(*best)
        wlo = max(ocen - nwidths * owidth, fit.wav[0])
        whi = min(ocen + nwidths * owidth, fit.wav[-1])
        hwav = np.linspace(wlo, whi, npoints)
        hcont = np.interp(hwav, fit.wav, fit.cont)
        ax.plot(hwav, (1 + lfit.continuum_offset + func(hwav, *best)) * hcont, "-", color=color,
                label=f"{line.label} model")

        # model convolved with the filters, as compared with the data
        ind = lfit.band_index
        fwav = fit.filters.wavelength[ind]
        ftrans = fit.filters.transmission[ind]
        conv = band_line_flux(fwav, ftrans, best, func) / band_norm(fwav, ftrans)
        ax.plot(fit.wav[ind], (1 + conv + lfit.continuum_offset) * fit.cont[ind], "D",
                color=color, ms=5, label=f"{line.label} model (in bands)")

    good = fit.flux > 0
    if np.any(good):
        ax.set_ylim(0, 1.25 * np.max(fit.flux[good]))
    ax.set_xlabel(WAV_LABEL)
    ax.set_ylabel(FLUX_LABEL)
    ax.set_title(f"$z = {fit.z:.3f}$")
    ax.legend(fontsize="small", ncol=2)
    return ax


def plot_line_posterior(line_fit, **kwargs):
    """Corner plot of the posterior of a line fit (requires ``corner``)."""
    import corner

    return corner.corner(line_fit.samples, labels=list(line_fit.profile.param_names), **kwargs)
