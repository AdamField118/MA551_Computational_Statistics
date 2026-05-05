#!/usr/bin/env python3
"""
metacal_pipeline.py
===================
Multiplicative shear bias estimation pipeline for MA 551 final project.

Mirrors shear_bias/m/main.py and shear_bias/m/helpers.py from ShearNet
exactly.  The only additions are:

  * reads the raw CSV catalog instead of the pre-processed FITS version
  * runs the full paired +/-shear metacal loop and jackknife inline (no
    superbit_lensing dependency)
  * writes results/metacal_bias.json for consumption by run_all.R

Simulation parameters match the ShearNet unit-test defaults:
  nse_sd   = 12.719674   (SuperBIT sky noise in ADU)
  scale    = 0.141       arcsec/pixel
  npix     = 53          stamp size
  hlr      = 0.5 arcsec  (constant, as in shear_bias/m/config.yaml)
  flux     = 12258.97    (constant, as in shear_bias/m/config.yaml)
  shear    = 0.01        injected |shear|

Usage (laptop / login node, quick test):
  python metacal_pipeline.py \\
      --catalog cosmos15_superbit2023_phot_shapes_with_sigma.csv \\
      --n-obs 1000 --n-workers 8 --psf-fwhm 0.5

Usage (Turing HPC with real SuperBIT PSFEx model):
  python metacal_pipeline.py \\
      --catalog cosmos15_superbit2023_phot_shapes_with_sigma.csv \\
      --n-obs 50000 --n-workers 32 \\
      --psfex /home/adfield/SHEARNET_DATA/psfex-output/Abell3411_1_300_1684688714_clean_starcat.psf

The PSFEx path is optional. If omitted, a Gaussian PSF with --psf-fwhm
(default 0.5 arcsec) is used as fallback, matching ShearNet's ideal-PSF
experiments.
"""

import argparse
import json
import logging
import math
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from astropy.table import Table

import galsim
import galsim.des
import ngmix
from ngmix.shape import e1e2_to_g1g2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('metacal')

# ---------------------------------------------------------------------------
# Fixed simulation constants (mirror shear_bias/m/config.yaml)
# ---------------------------------------------------------------------------
NSE_SD      = 12.719674   # SuperBIT sky noise, ADU
PSF_NOISE   = 1.0e-6      # tiny PSF image noise for ngmix stability
SCALE       = 0.141       # arcsec / pixel
NPIX        = 53          # postage stamp size
PSF_NPIX    = 53
GAL_HLR     = 0.5         # constant half-light radius [arcsec]
GAL_FLUX    = 12258.97    # constant galaxy flux [ADU]
SHEAR_TRUE  = 0.01        # injected shear (matching ShearNet benchmarks)
MCAL_STEP   = 0.01        # metacal shear perturbation step
MCAL_TYPES  = ['noshear', '1p', '1m']
NJACK       = 20          # jackknife groups

# WCS parameters needed when querying PSFEx at a focal-plane position
_WCS_PARAMS = {
    'image_xsize': 9600,
    'image_ysize': 6400,
    'pixel_scale': SCALE,
    'center_ra':   13.3,   # degrees (Abell 3411 approx)
    'center_dec':  33.1,
    'theta':       0.0,
}


# ---------------------------------------------------------------------------
# Helper: adaptive-moment PSF fit
# Replaces the superbit_lensing.utils.get_admoms_ngmix_fit dependency.
# Source: shearnet/utils/metrics.py
# ---------------------------------------------------------------------------
def get_admoms_ngmix_fit(obs, reduced=True):
    """Measure PSF moments; returns dict with keys e1, e2, T, flags."""
    jac   = obs._jacobian
    scale = jac.get_scale()
    image = obs.image
    norm  = np.sum(image[image > 0])
    if norm <= 0:
        return {'e1': np.nan, 'e2': np.nan, 'T': np.nan, 'flags': 1}
    obs_norm = ngmix.Observation(image=image / norm, jacobian=jac)
    am  = ngmix.admom.AdmomFitter()
    res = am.go(obs_norm, guess=0.5)
    try:
        gal_image = galsim.Image(image / norm, scale=scale)
        admoms    = galsim.hsm.FindAdaptiveMom(gal_image)
        sigma     = admoms.moments_sigma * scale
        T_val     = 2.0 * sigma ** 2
        flag      = 0 if (admoms.moments_status == 0 and res['flags'] == 0) else 1
    except Exception:
        T_val = np.nan
        flag  = 1
    e1 = res.get('e1', np.nan)
    e2 = res.get('e2', np.nan)
    if reduced and flag == 0:
        try:
            e1, e2 = e1e2_to_g1g2(e1, e2)
        except Exception:
            pass
    return {'e1': e1, 'e2': e2, 'T': T_val, 'flags': flag}


# ---------------------------------------------------------------------------
# WCS helper for PSFEx (mirrors shearnet/utils/simutils.py)
# ---------------------------------------------------------------------------
def _make_galsim_wcs(params=_WCS_PARAMS):
    xsize  = params['image_xsize']
    ysize  = params['image_ysize']
    pscale = params['pixel_scale']
    theta  = params.get('theta', 0.0) * galsim.degrees
    ra     = params['center_ra']  * galsim.hours
    dec    = params['center_dec'] * galsim.degrees

    fiducial = galsim.ImageF(xsize, ysize)
    dudx =  np.cos(theta) * pscale
    dudy = -np.sin(theta) * pscale
    dvdx =  np.sin(theta) * pscale
    dvdy =  np.cos(theta) * pscale
    affine  = galsim.AffineTransform(dudx, dudy, dvdx, dvdy,
                                      origin=fiducial.true_center)
    sky_cen = galsim.CelestialCoord(ra=ra, dec=dec)
    return galsim.TanWCS(affine, sky_cen, units=galsim.arcsec)


# ---------------------------------------------------------------------------
# 1. Catalog loading
# ---------------------------------------------------------------------------
def load_catalog(csv_path, n_max=None, seed=42):
    """
    Read cosmos15_superbit2023_phot_shapes_with_sigma.csv and extract
    the morphological columns used in shear_bias/m/main.py.

    Column mapping (CSV -> ShearNet FITS equivalent):
      c10_sersic_fit_q   -> Q   (axis ratio)
      c10_sersic_fit_phi -> PHI (position angle, radians)
      c10_sersic_fit_hlr -> HLR (half-light radius, arcsec)
      c10_viable_sersic  -> quality flag (keep == 1 only)
    """
    log.info('Loading catalog: %s', csv_path)
    df = pd.read_csv(csv_path)
    log.info('Raw catalog: %d rows', len(df))

    viable = df['c10_viable_sersic'].fillna(0).astype(bool)
    q_ok   = df['c10_sersic_fit_q'] > 0.05
    hlr_ok = df['c10_sersic_fit_hlr'] > 0.0
    mask   = viable & q_ok & hlr_ok
    df     = df[mask].reset_index(drop=True)
    log.info('After quality cuts: %d rows', len(df))

    if n_max is not None and len(df) > n_max:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), n_max, replace=False)
        df  = df.iloc[idx].reset_index(drop=True)
        log.info('Subsampled to %d rows', len(df))

    return {
        'Q':   np.asarray(df['c10_sersic_fit_q'],   dtype=float),
        'PHI': np.asarray(df['c10_sersic_fit_phi'],  dtype=float),
        'HLR': np.asarray(df['c10_sersic_fit_hlr'],  dtype=float),
        'n':   len(df),
    }


# ---------------------------------------------------------------------------
# 2. PSF construction
# ---------------------------------------------------------------------------
_GALSIM_PSF_CACHE = {}

def _get_psfex_psf(psfex_file, ud, xsize=9600, ysize=6400, margin=500):
    """Draw a random PSFEx PSF at a random focal-plane position."""
    if psfex_file not in _GALSIM_PSF_CACHE:
        wcs = _make_galsim_wcs()
        _GALSIM_PSF_CACHE[psfex_file] = galsim.des.DES_PSFEx(
            psfex_file, wcs=wcs)
    psfex = _GALSIM_PSF_CACHE[psfex_file]
    x = margin + (xsize - 2 * margin) * ud()
    y = margin + (ysize - 2 * margin) * ud()
    return psfex.getPSF(galsim.PositionD(x=x, y=y))


# ---------------------------------------------------------------------------
# 3. Single-galaxy data generation (mirrors make_data in main.py)
# ---------------------------------------------------------------------------
def make_data(rng, q, phi_rad, shear_true, psf_fwhm,
              psfex_file=None, ud=None,
              nse_sd=NSE_SD, gal_hlr=GAL_HLR, gal_flux=GAL_FLUX):
    """
    Simulate one galaxy triplet (null, +shear, -shear) exactly as in
    shear_bias/m/main.py.

    Returns:
      (obs_null, obs_plus, obs_minus, g_th_plus, g_th_minus, gal_hlr, gal_flux)
    """
    gsp = galsim.GSParams(maximum_fft_size=32768)

    dy, dx = rng.uniform(low=-SCALE / 2, high=SCALE / 2, size=2)

    # PSF: PSFEx if available, else Gaussian
    if psfex_file is not None and ud is not None:
        psf = _get_psfex_psf(psfex_file, ud)
    else:
        psf = galsim.Gaussian(fwhm=psf_fwhm)

    # Base galaxy (Exponential, axis ratio + PA from catalog)
    obj0 = (galsim.Exponential(half_light_radius=gal_hlr, flux=gal_flux)
            .shear(q=float(q), beta=float(phi_rad) * galsim.radians))

    # +/- shear versions (mirrors shear_bias/m/main.py exactly)
    objp = obj0.shear(g1= shear_true, g2=0.0)
    objm = obj0.shear(g1=-shear_true, g2=0.0)

    # Theoretical g1/g2 from GalSim moments on the unconvolved sheared galaxy
    def _g_th(obj):
        im = obj.drawImage(nx=NPIX, ny=NPIX, scale=SCALE)
        res = galsim.hsm.FindAdaptiveMom(im, strict=False)
        s = res.observed_shape
        return np.array([s.g1, s.g2])

    try:
        g_th_p = _g_th(objp)
        g_th_m = _g_th(objm)
    except Exception:
        g_th_p = np.array([ shear_true, 0.0])
        g_th_m = np.array([-shear_true, 0.0])

    # Convolve with PSF and shift
    objp_psf  = galsim.Convolve(psf, objp.shift(dx=dx, dy=dy),  gsparams=gsp)
    objm_psf  = galsim.Convolve(psf, objm.shift(dx=dx, dy=dy),  gsparams=gsp)
    obj0_psf  = galsim.Convolve(psf, obj0.shift(dx=dx, dy=dy),  gsparams=gsp)

    psf_im = psf.drawImage(nx=PSF_NPIX, ny=PSF_NPIX, scale=SCALE).array
    im_0   = obj0_psf.drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array
    im_p   = objp_psf.drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array
    im_m   = objm_psf.drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array

    # Single shared noise realisation (same for +/- to cancel shape noise)
    im_noise = rng.normal(scale=nse_sd, size=im_0.shape)
    im_0 += im_noise
    im_p += im_noise
    im_m += im_noise

    cen     = (np.array(im_0.shape) - 1.0) / 2.0
    psf_cen = (np.array(psf_im.shape) - 1.0) / 2.0

    jac     = ngmix.DiagonalJacobian(
        row=cen[0] + dy / SCALE, col=cen[1] + dx / SCALE, scale=SCALE)
    psf_jac = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=SCALE)

    wt     = np.full_like(im_0, 1.0 / nse_sd ** 2)
    psf_wt = np.full_like(psf_im, 1.0 / PSF_NOISE ** 2)

    psf_obs = ngmix.Observation(psf_im, weight=psf_wt, jacobian=psf_jac)

    # Cache PSF admom in metadata so make_struct can read it without refitting
    admom = get_admoms_ngmix_fit(psf_obs, reduced=True)
    psf_obs.update_meta_data(
        {'e1': admom['e1'], 'e2': admom['e2'],
         'T': admom['T'], 'flags': admom['flags']})

    def _obs(im):
        return ngmix.Observation(im, weight=wt, jacobian=jac, psf=psf_obs)

    return (_obs(im_0), _obs(im_p), _obs(im_m),
            g_th_p, g_th_m, gal_hlr, gal_flux)


# ---------------------------------------------------------------------------
# 4. Priors (exact copy from shear_bias/m/helpers.py)
# ---------------------------------------------------------------------------
def _get_priors(seed):
    rng = np.random.RandomState(seed)
    g_prior   = ngmix.priors.GPriorBA(0.3, rng=rng)
    cen_prior = ngmix.priors.CenPrior(0.0, 0.0, 0.2, 0.2, rng=rng)
    T_prior   = ngmix.priors.FlatPrior(-1.0, 1000.0, rng=rng)
    F_prior   = ngmix.priors.FlatPrior(-10.0, 1.0e5, rng=rng)
    return ngmix.joint_prior.PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)


# ---------------------------------------------------------------------------
# 5. make_struct (exact port from shear_bias/m/helpers.py)
# ---------------------------------------------------------------------------
def make_struct(res, obs, shear_type):
    dt = [
        ('flags', 'i4'), ('shear_type', 'U7'),
        ('s2n', 'f8'), ('g', 'f8', 2), ('T', 'f8'),
        ('flux', 'f8'), ('Tpsf', 'f8'), ('gpsf', 'f8', 2),
    ]
    data = np.zeros(1, dtype=dt)
    data['shear_type'] = shear_type
    data['flags']      = res['flags']
    if res['flags'] == 0:
        data['s2n']  = res['s2n']
        try:
            data['g'] = res['e']
        except KeyError:
            data['g'] = res['g']
        data['T']    = res['T']
        data['flux'] = res['flux']
    else:
        data['s2n']  = np.nan
        data['g']    = np.nan
        data['T']    = np.nan
        data['flux'] = np.nan

    # PSF admoms from cached metadata (avoids redundant re-fitting)
    meta = getattr(obs.psf, 'meta', {})
    data['Tpsf'] = meta.get('T', np.nan)
    data['gpsf'] = np.array([meta.get('e1', np.nan), meta.get('e2', np.nan)])
    return data


# ---------------------------------------------------------------------------
# 6. process_obs (exact port from shear_bias/m/helpers.py)
# ---------------------------------------------------------------------------
def process_obs(obs, boot):
    resdict, obsdict = boot.go(obs)
    dlist = [
        make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
        for stype, sres in resdict.items()
    ]
    return np.hstack(dlist)


# ---------------------------------------------------------------------------
# 7. Per-galaxy worker
# ---------------------------------------------------------------------------
def _process_one(args):
    """
    Simulate one galaxy at +/- shear_true and run metacalibration on both.
    Returns (data_plus, data_minus, g_th_plus, g_th_minus) or None on failure.
    """
    (i, q, phi_rad, seed, psf_fwhm, psfex_file) = args
    rng = np.random.RandomState(seed)
    ud  = galsim.BaseDeviate(seed) if psfex_file else None

    try:
        (obs0, obsp, obsm,
         g_th_p, g_th_m,
         gal_hlr, gal_flux) = make_data(
            rng, q, phi_rad,
            shear_true=SHEAR_TRUE,
            psf_fwhm=psf_fwhm,
            psfex_file=psfex_file,
            ud=ud)
    except Exception as exc:
        log.debug('Galaxy %d simulation failed: %s', i, exc)
        return None

    # Build MetacalBootstrapper (mirrors shear_bias/m/main.py runner setup)
    prior   = _get_priors(seed)
    Tguess  = 4.0 * SCALE ** 2
    lm_pars = {'maxfev': 2000, 'xtol': 5e-5, 'ftol': 5e-5}

    fitter  = ngmix.fitting.Fitter(model='gauss', prior=prior,
                                    fit_pars=lm_pars)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng, T=Tguess, prior=prior)
    runner  = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=20)

    psf_fitter  = ngmix.fitting.Fitter(
        model='gauss', fit_pars={'maxfev': 4000, 'xtol': 5e-5, 'ftol': 5e-5})
    psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
    psf_runner  = ngmix.runners.PSFRunner(
        fitter=psf_fitter, guesser=psf_guesser, ntry=20)

    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng, psf='dilate', step=MCAL_STEP, types=MCAL_TYPES)

    try:
        data_p = process_obs(obsp, boot)
        data_m = process_obs(obsm, boot)
    except Exception as exc:
        log.debug('Galaxy %d metacal failed: %s', i, exc)
        return None

    return data_p, data_m, g_th_p, g_th_m


# ---------------------------------------------------------------------------
# 8. shear_data_to_table (exact port from shear_bias/m/helpers.py)
# ---------------------------------------------------------------------------
def shear_data_to_table(data_list, mcal_shear=MCAL_STEP):
    rows = []
    for arr in data_list:
        row = {}
        g_store = {}
        row['flag'] = 0 if np.all(arr['flags'] == 0) else 1
        for rec in arr:
            stype = str(rec['shear_type'])
            if rec['flags'] == 0:
                g_store[stype] = np.array(rec['g'], dtype=float)
                row[f'g_{stype}']   = g_store[stype]
                row[f'T_{stype}']   = float(rec['T'])
                row[f's2n_{stype}'] = float(rec['s2n'])
            else:
                g_store[stype]     = np.array([np.nan, np.nan])
                row[f'g_{stype}']  = g_store[stype]

        dg = 2.0 * mcal_shear
        if '1p' in g_store and '1m' in g_store:
            row['r11'] = (g_store['1p'][0] - g_store['1m'][0]) / dg
            row['r22'] = (g_store['1p'][1] - g_store['1m'][1]) / dg
        rows.append(row)
    return Table(rows)


# ---------------------------------------------------------------------------
# 9. jackknife_mc_v2 (exact port from shear_bias/m/helpers.py)
# ---------------------------------------------------------------------------
def jackknife_mc_v2(tab_p, tab_m, shear_true=SHEAR_TRUE, njac=NJACK,
                    g_col='g_noshear', r11_col='r11'):
    """
    Paired +/- shear jackknife estimator.
    Mirrors jackknife_mc_v2 in shear_bias/m/helpers.py line-for-line.
    """
    N    = len(tab_p)
    njac = min(njac, N // 2)

    g_arr_p = np.asarray(tab_p[g_col])
    R_arr_p = np.asarray(tab_p[r11_col])
    g_arr_m = np.asarray(tab_m[g_col])
    R_arr_m = np.asarray(tab_m[r11_col])

    gamma1_per = (g_arr_p[:, 0] - g_arr_m[:, 0]) / 2.0
    c_per      = (g_arr_p[:, 1] + g_arr_m[:, 1]) / 2.0
    R1_pair    = 0.5 * (R_arr_p + R_arr_m)

    shear_est = np.nanmean(gamma1_per) / np.nanmean(R1_pair)
    m_full    = float(shear_est / shear_true - 1.0)
    c_full    = float(np.nanmean(c_per))

    indices = np.arange(N)
    chunks  = np.array_split(indices, njac)

    m_jk, c_jk, r11_jk = [], [], []
    for chunk in chunks:
        mask = np.ones(N, dtype=bool)
        mask[chunk] = False
        g_mean  = np.nanmean(gamma1_per[mask])
        R_mean  = np.nanmean(R1_pair[mask])
        m_jk.append(g_mean / R_mean / shear_true - 1.0)
        c_jk.append(float(np.nanmean(c_per[mask])))
        r11_jk.append(float(R_mean))

    m_jk   = np.array(m_jk)
    c_jk   = np.array(c_jk)
    r11_jk = np.array(r11_jk)

    m_err   = float(np.sqrt((njac - 1) / njac *
                             np.sum((m_jk  - m_jk.mean())  ** 2)))
    c_err   = float(np.sqrt((njac - 1) / njac *
                             np.sum((c_jk  - c_jk.mean())  ** 2)))
    r11_err = float(np.sqrt((njac - 1) / njac *
                             np.sum((r11_jk - r11_jk.mean()) ** 2)))

    return dict(
        m_full   = m_full,     c_full   = c_full,
        m_mean   = float(m_jk.mean()),  m_err  = m_err,
        c_mean   = float(c_jk.mean()),  c_err  = c_err,
        r11_mean = float(r11_jk.mean()),r11_err= r11_err,
        m_jk     = m_jk.tolist(),
        c_jk     = c_jk.tolist(),
        n_gal    = N,
        njac     = njac,
    )


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------
def _sci_exp(v):
    if v == 0: return 0
    return int(math.floor(math.log10(abs(v))))


def main():
    parser = argparse.ArgumentParser(
        description='Metacalibration bias pipeline (mirrors ShearNet)')
    parser.add_argument('--catalog', required=True,
                        help='Path to cosmos15_superbit2023_phot_shapes_with_sigma.csv')
    parser.add_argument('--n-obs', type=int, default=10000,
                        help='Number of galaxy pairs to process (default: 10000)')
    parser.add_argument('--n-workers', type=int,
                        default=max(1, cpu_count() // 2),
                        help='Parallel worker processes')
    parser.add_argument('--psf-fwhm', type=float, default=0.5,
                        help='Gaussian PSF FWHM arcsec (used when --psfex absent)')
    parser.add_argument('--psfex', type=str, default=None,
                        help='Path to PSFEx .psf model (SuperBIT); optional')
    parser.add_argument('--n-jack', type=int, default=NJACK,
                        help=f'Jackknife groups (default: {NJACK})')
    parser.add_argument('--seed', type=int, default=150,
                        help='Base RNG seed (default: 150, matching ShearNet configs)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Output directory (default: results/)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Validate PSFEx path if given
    # ------------------------------------------------------------------
    if args.psfex is not None and not os.path.exists(args.psfex):
        log.error('PSFEx file not found: %s', args.psfex)
        log.error('Falling back to Gaussian PSF (fwhm=%.2f")', args.psf_fwhm)
        args.psfex = None

    psf_desc = (f'PSFEx ({os.path.basename(args.psfex)})'
                if args.psfex else f'Gaussian fwhm={args.psf_fwhm}"')

    # ------------------------------------------------------------------
    # Load catalog
    # ------------------------------------------------------------------
    cat = load_catalog(args.catalog, n_max=args.n_obs, seed=args.seed)
    n   = cat['n']

    log.info('Processing %d galaxy pairs on %d workers', n, args.n_workers)
    log.info('PSF: %s', psf_desc)
    log.info('nse_sd=%.3f  scale=%.3f  hlr=%.2f  flux=%.1f  shear_true=%.3f',
             NSE_SD, SCALE, GAL_HLR, GAL_FLUX, SHEAR_TRUE)

    # ------------------------------------------------------------------
    # Build worker arguments
    # ------------------------------------------------------------------
    args_list = [
        (i, cat['Q'][i], cat['PHI'][i], args.seed + i,
         args.psf_fwhm, args.psfex)
        for i in range(n)
    ]

    # ------------------------------------------------------------------
    # Parallel processing (mirrors Pool usage in shear_bias/m/main.py)
    # ------------------------------------------------------------------
    data_list_p, data_list_m = [], []
    gth_list_p,  gth_list_m  = [], []
    n_fail = 0

    log.info('Starting metacalibration ...')
    with Pool(processes=args.n_workers) as pool:
        for result in pool.imap(_process_one, args_list, chunksize=50):
            if result is None:
                n_fail += 1
                continue
            struct_p, struct_m, g_th_p, g_th_m = result
            data_list_p.append(struct_p)
            data_list_m.append(struct_m)
            gth_list_p.append(g_th_p)
            gth_list_m.append(g_th_m)

    n_ok = len(data_list_p)
    log.info('Succeeded: %d / %d  (%.1f%% failure rate)',
             n_ok, n, 100.0 * n_fail / max(n, 1))

    if n_ok < 20:
        log.error('Too few successful measurements (%d). Exiting.', n_ok)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build tables (mirrors shear_data_to_table in helpers.py)
    # ------------------------------------------------------------------
    tab_p = shear_data_to_table(data_list_p)
    tab_m = shear_data_to_table(data_list_m)
    tab_p['g_th'] = np.asarray(gth_list_p)
    tab_m['g_th'] = np.asarray(gth_list_m)

    # ------------------------------------------------------------------
    # Bias estimation (mirrors jackknife_mc_v2 in helpers.py)
    # ------------------------------------------------------------------
    jk = jackknife_mc_v2(tab_p, tab_m,
                          shear_true=SHEAR_TRUE,
                          njac=args.n_jack)

    m, merr = jk['m_full'], jk['m_err']
    c, cerr = jk['c_full'], jk['c_err']

    # ------------------------------------------------------------------
    # Print summary (mirrors print block at end of shear_bias/m/main.py)
    # ------------------------------------------------------------------
    exp_m = _sci_exp(m)
    exp_c = _sci_exp(c) if c != 0 else -5

    print('\n' + '=' * 60)
    print('METACALIBRATION BIAS ESTIMATES')
    print('=' * 60)
    print(f'  Galaxies processed : {n_ok:,} / {n:,}')
    print(f'  m  = ({m/10**exp_m:.3f} +/- {merr/10**exp_m:.3f}) x 10^{exp_m}')
    print(f'  c  = ({c:.5f} +/- {cerr:.5f})')
    print(f'  R11 (ensemble)     : {jk["r11_mean"]:.4f} +/- {jk["r11_err"]:.4f}')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Write JSON for run_all.R
    # ------------------------------------------------------------------
    output = {
        'pipeline':     'metacal_pipeline.py (ShearNet-matched)',
        'psf':          psf_desc,
        'n_input':      n,
        'n_success':    n_ok,
        'success_rate': n_ok / max(n, 1),
        'shear_true':   SHEAR_TRUE,
        'mcal_step':    MCAL_STEP,
        'nse_sd':       NSE_SD,
        'gal_hlr':      GAL_HLR,
        'gal_flux':     GAL_FLUX,
        'scale':        SCALE,
        'npix':         NPIX,
        'seed':         args.seed,
        'n_jack':       args.n_jack,
        **{k: v for k, v in jk.items()
           if not isinstance(v, list)},
        'm_jk': jk['m_jk'],
        'c_jk': jk['c_jk'],
        # Aliases expected by run_all.R
        'm_hat':    m,
        'c_hat':    c,
        'm_hat_se': merr,
        'c_hat_se': cerr,
    }

    out_path = os.path.join(args.outdir, 'metacal_bias.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    log.info('Results written to %s', out_path)
    print(f'\nOutput : {out_path}')
    print('Next   : Rscript R/run_all.R  (reads metacal_bias.json automatically)')


if __name__ == '__main__':
    main()
