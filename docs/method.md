# Method

This document describes what `quasar_lines.fit_photospectrum` does. All
wavelengths are in Angstrom and fluxes in erg s⁻¹ cm⁻² Å⁻¹. Each band is
represented by its pivot wavelength (Tokunaga & Vacca 2005).

## 1. Continuum

`quasar_lines.continuum.fit_continuum`

1. For each rest-frame anchor wavelength
   λ_a ∈ {1350, 1700, 1800, 2200, 3100, 4000, 4200, 5200} whose observed
   wavelength lies more than 75 Å inside the observed range, the continuum
   level is the median of the positive fluxes of the closest band and its two
   neighbours. The uncertainty is that of the closest band.
2. With `correct_contamination=True` (default), the anchors at 1350, 3100, and
   5200 Å are multiplied by 0.9 to account for residual contamination from
   emission lines and the FeII pseudo-continuum.
3. A straight line, `ln(F/norm) = a + α ln λ_rest`, is fitted to the anchors
   with `emcee` (100 walkers × 150 steps, 75 burn-in) using flat priors
   `a ∈ [−25, 25]`, `α ∈ [−3.5, 3]`. At least two anchors are required.
   Walkers start from a Latin hypercube covering the central 80% of each prior
   range (the same initialisation is used for line fits).

The posterior gives the continuum in every band (median and standard
deviation) and the monochromatic luminosity λL_λ at 5100, 3000, and 1350 Å.

## 2. Emission lines

`quasar_lines.lines.fit_line`

A line is fitted only if its observed wavelength lies between the second band
and the second-to-last band, so that there are data on both sides.

1. **Data.** The bands whose pivot wavelength falls within the line's
   rest-frame fitting window (Hβ: 4700–5100 Å, MgII: 2600–3000 Å, CIV:
   1450–1630 Å; ~4×10⁴ km/s wide), extended by half a band spacing, are
   normalised by the continuum: `y = F / F_cont − 1`, with uncertainties that
   include those of the continuum.
2. **Apparent continuum.** The minimum of `y` within the window (clipped at 0)
   defines the apparent continuum level, which reduces the impact of
   absorptions and continuum mis-estimation.
3. **Contamination** (`correct_contamination=True`):
   * Hβ: [OIII] λ5008 — the closest band is set to the apparent continuum and
     the second closest multiplied by 0.5; [OIII] λ4960 — the closest band is
     multiplied by 0.75 and the second closest by 0.5.
   * MgII: FeII bumps at 2700 and 2950 Å — the closest band is set to the
     apparent continuum and the second closest multiplied by 0.5.
   * CIV: no correction.

   Only bands within 1.5 half-band-spacings of the contaminant are modified,
   and never below the apparent continuum.
4. **Padding.** One band on each side of the window is added with the value of
   the apparent continuum and a small uncertainty (¼ of the smallest error),
   which anchors the line wings.
5. **Model.** The line profile (Gaussian by default; the asymmetric
   Nagao et al. 2006 profile is also available) plus the apparent continuum is
   integrated through each filter transmission curve.
6. **Fit.** `emcee` with 75 walkers × 500 steps (150 burn-in) and flat priors:
   amplitude ∈ [−1, 3] (or [−1, 5] at z > 3), σ ∈ [0, 200] Å (observed frame),
   centre between the pivot wavelengths of the bands adjacent to the expected
   line position. Walkers whose post-burn-in log-probability differs by more
   than 3 from the best are discarded.

### Derived properties

* **Centre:** `[peak, lower, upper]`, where `peak` is the mode of the kernel
  density estimate of the centre posterior and `lower`/`upper` are the points
  where it drops to half its maximum (rest frame). Unlike the other
  quantities these are **not** percentiles: `upper − lower` is the FWHM of the
  posterior (≈2.35σ for a Gaussian). The mode is used because the posterior
  of the centre is often truncated by its prior.
* **FWHM:** percentiles of the posterior FWHM, converted to km/s at the
  observed line centre.
* **EW:** integral of the continuum-normalised profile over ±5 FWHM in the
  rest frame.

All summaries are `[median, 16th, 84th]` percentiles.

## 3. Black hole mass

`quasar_lines.bh_mass`

`log(M_BH/M☉) = A + B [log λL_λ − 44] + 2 log FWHM[km/s]`, with the continuum
luminosity at 5100, 3000, and 1350 Å for Hβ, MgII, and CIV. Uncertainties
propagate half the 16–84 percentile range of `log λL_λ` and `log FWHM`.

Bolometric luminosities use the corrections 9.26, 5.15, and 3.81 (Rakshit et
al. 2020), and Eddington ratios assume `L_Edd = 1.257×10³⁸ M_BH/M☉ erg/s`.
Because both `M_BH` and `L_bol` depend on `λL_λ`,
`log(L_bol/L_Edd) = (1 − B) log λL_λ − 2 log FWHM + const`, and its
uncertainty is `sqrt([(1 − B) σ_L]² + [2 σ_FWHM]²)`.

## Limitations

* The redshift must be known (spectroscopic or high-quality photometric).
* A single component is used per line; FWHMs are therefore systematically
  smaller than multi-component spectroscopic fits (absorbed by the
  `chaves-montero22` calibration).
* The method was validated on miniJPAS quasars with r < 21 (Chaves-Montero et
  al. 2022); results for fainter sources are noisier.
