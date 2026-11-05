from pathlib import Path

import numpy as np
import pytest

import quasar_lines as ql

DATA = Path(__file__).parents[1] / "examples" / "data"

FAST_LINE = ql.MCMCSettings(nwalkers=40, nsteps=300, nburn=100)


@pytest.fixture(scope="module")
def catalog():
    return ql.read_minijpas_catalog(DATA / "minijpas_qso_examples.cat")


def test_read_catalog(catalog):
    assert len(catalog) == 3
    assert catalog.flux.shape == (3, 55)
    assert np.all(catalog.flux_err > 0)


def test_fit_high_z_quasar(catalog):
    # miniJPAS 2243-14873, z = 3.22; Chaves-Montero et al. (2022) find
    # log L1350 = 46.08, log FWHM(CIV) = 3.90 and log M_BH = 9.24
    ii = 2
    filters = ql.load_jpas_filters(catalog.band_names)
    fit = ql.fit_photospectrum(catalog.flux[ii], catalog.flux_err[ii], catalog.z[ii],
                               filters, line_mcmc=FAST_LINE, rng=1)
    assert list(fit.lines) == ["CIV"]
    civ = fit.lines["CIV"]
    assert civ.log_lum_cont[0] == pytest.approx(46.08, abs=0.05)
    assert np.log10(civ.fwhm[0]) == pytest.approx(3.90, abs=0.1)
    assert fit.black_hole_mass()["CIV"][0] == pytest.approx(9.24, abs=0.2)
