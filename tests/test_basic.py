import numpy as np
import pytest

import quasar_lines as ql
from quasar_lines import conversions, profiles
from quasar_lines.lines import locate_lines


def test_gaussian_fwhm():
    wav = np.linspace(4000, 6000, 20001)
    line = profiles.gaussian(wav, 1.0, 30.0, 5000.0)
    above = wav[line >= 0.5]
    assert above[-1] - above[0] == pytest.approx(profiles.gaussian_fwhm(1.0, 30.0, 5000.0), abs=0.2)


def test_mag_flux_roundtrip():
    mag = np.array([18.0, 21.5])
    assert conversions.flux_to_mag(conversions.mag_to_flux(mag, 6000.0), 6000.0) == pytest.approx(mag)


def test_width_roundtrip():
    assert conversions.width_kms_to_aa(5000.0, conversions.width_aa_to_kms(5000.0, 50.0)) == pytest.approx(50.0)


def test_jpas_filters():
    filters = ql.load_jpas_filters()
    assert len(filters) == 56
    assert filters.names[0] == "uJPAS" and filters.names[-1] == "J1007"
    # narrow-band names encode the central wavelength in nm
    nb = filters.select(["J0500", "J0700"])
    assert nb.pivot == pytest.approx([5000.0, 7000.0], rel=0.01)


def test_virial_mass():
    # Vestergaard & Peterson (2006): log M = 6.91 for L5100 = 1e44 erg/s and FWHM = 1000 km/s
    assert ql.virial_mass(44.0, 3.0, "Hbeta", calibration="shen11") == pytest.approx(6.91)
    with pytest.raises(KeyError):
        ql.virial_mass(44.0, 3.0, "Halpha")


def test_locate_lines():
    wav = np.arange(3800.0, 9200.0, 100.0)
    pos = locate_lines(wav, 0.676, ql.LINES)
    assert pos[0] != -1 and pos[1] != -1 and pos[2] == -1


def test_eddington_ratio_error_is_correlated():
    # log(L/L_Edd) = (1 - B) log L - 2 log FWHM + const
    _, bb = ql.CALIBRATIONS["shen11"]["Hbeta"]
    err = ql.eddington_ratio_error(0.1, 0.0, "Hbeta", calibration="shen11")
    assert err == pytest.approx((1 - bb) * 0.1)


def test_missing_error_in_first_band():
    from quasar_lines.continuum import _MISSING_ERROR, fit_continuum

    filters = ql.load_jpas_filters()
    wav = filters.pivot[1:]
    z = wav[0] / 1350.0 - 1  # first anchor falls on the first band
    flux = 1e-16 * (wav / 5000.0) ** -1.5
    flux_err = 0.05 * flux
    flux_err[0] = _MISSING_ERROR
    flux_err[-1] = np.nan  # wrapping around to the last band would poison the fit
    fit = fit_continuum(wav, flux, flux_err, z, edge_margin=-75.0, rng=0)
    assert fit is not None and np.isfinite(fit.samples).all()
