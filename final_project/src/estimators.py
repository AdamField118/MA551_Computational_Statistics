import numpy as np
import ngmix
import galsim
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import sys

# ===========================================================================================
# METACAL CODE
# ===========================================================================================

def measure_moments(im):
    """
    Measures moments for a GalSim image object or a NumPy array.
    
    Args:
        - im: Either a GalSim image object or a 2D NumPy array.
        
    Returns:
        - M_xx, M_yy, M_xy: Second-order moments.
    """
    if hasattr(im, 'bounds') and hasattr(im, 'array'):
        # Handle GalSim image object
        xmin, xmax, ymin, ymax = im.bounds.xmin, im.bounds.xmax, im.bounds.ymin, im.bounds.ymax
        x, y = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        I = im.array
    elif isinstance(im, np.ndarray):
        # Handle NumPy array
        ny, nx = im.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        I = im
    else:
        raise TypeError("Input 'im' must be a GalSim image object or a 2D NumPy array.")

    I_sum = np.sum(I)
    M_x = np.sum(I * x) / I_sum
    M_y = np.sum(I * y) / I_sum
    M_xx = np.sum(I * (x - M_x) ** 2) / I_sum
    M_yy = np.sum(I * (y - M_y) ** 2) / I_sum
    M_xy = np.sum(I * (x - M_x) * (y - M_y)) / I_sum
    return M_xx, M_yy, M_xy

def measure_e1e2(g1=None, g2=None, im=None, sigma=1, scale=0.2, npix=53):
    """
    Calculate ellipticity components e1 and e2.
    
    Parameters:
    - g1, g2: Arrays for shear components. If provided, moments are calculated using plot_shear_image.
    - im: 2D array for image. If provided, moments are calculated using measure_moments.
    - sigma: Standard deviation for Gaussian smoothing (used with g1, g2).
    - scale: Pixel scale (used with g1, g2).
    - npix: Number of pixels along one dimension (used with g1, g2).
    
    Returns:
    - e1, e2: Ellipticity components.
    """
    if im is not None:
        M_xx, M_yy, M_xy = measure_moments(im)
    elif g1 is not None and g2 is not None:
        M_xx, M_yy, M_xy = plot_shear_image(g1, g2, sigma=sigma, nx=npix, ny=npix, scale=scale, verbose=False, return_moments=True, plot=False)
    else:
        raise ValueError("Either (g1 and g2) or im must be provided.")
    
    Mr = M_xx + M_yy
    Mplus = M_xx - M_yy
    Mcross = 2 * M_xy
    e1 = Mplus / Mr
    e2 = Mcross / Mr
    return e1, e2

def calculate_responsivity(psf_sigma, seed, h=0.01):
    from ..core.dataset import sim_func
    obj_im_p = sim_func(h, 0, seed=seed, psf_sigma=psf_sigma)
    obj_im_m = sim_func(-h, 0, seed=seed, psf_sigma=psf_sigma)
    e1p, _ = measure_e1e2(im=obj_im_p.image)
    e1m, _ = measure_e1e2(im=obj_im_m.image)
    R1 = (e1p - e1m) / (2 * h)
    obj_im_p = sim_func(0, h, seed=seed, psf_sigma=psf_sigma)
    obj_im_m = sim_func(0, -h, seed=seed, psf_sigma=psf_sigma)
    _, e2p = measure_e1e2(im=obj_im_p.image)
    _, e2m  = measure_e1e2(im=obj_im_m.image)
    R2 = (e2p - e2m) / (2 * h)
    return R1, R2

def obs_g1g2(im, psf_sigma):
    """
    Calculates the observed g1 and g2 values for a given image and PSF FWHM.

    Args:
        im (numpy.ndarray): The input image.
        psf_sigma (float): The Sigma of the PSF.

    Returns:
        tuple: A tuple containing the observed g1 and g2 values.
    """
    R1, R2 = calculate_responsivity(psf_sigma, 1234)
    e1, e2 = measure_e1e2(im=im)
    obs_g1 = e1 / R1
    obs_g2 = e2 / R2
    
    return obs_g1, obs_g2

def mcal_preds(images, psf_sigma):
    """
    Calculates the observed g1 and g2 values for a list of images and a given PSF FWHM.

    Args:
        images (list): A list of input images.
        psf_sigma (float): The Sigma of the PSF.

    Returns:
        tuple: A tuple containing two lists, one for the observed g1 values and one for the observed g2 values.
    """
    
    preds = []
    for image in images:
        g1, g2 = obs_g1g2(image, psf_sigma)
        g1g2 = np.array([g1, g2])
        preds.append(g1g2)
    
    return np.array(preds)

# ===========================================================================================
# NGMIX CODE
# ===========================================================================================


# Function to remove NaN values and count them
def clean_and_report_nans(data_list, name):
    clean_list = np.array(data_list)
    nan_count = np.isnan(clean_list).sum()
    
    if nan_count > 0:
        print(f"Removed {nan_count} NaN values from {name}.")
    
    return clean_list[~np.isnan(clean_list)]  # Return array without NaNs

def fourier_transform(psf):
    # Compute the Fourier Transform
    ft_psf = np.fft.fft2(psf)
    ft_psf_shifted = np.fft.fftshift(ft_psf)  # Shift zero frequency to center
    return ft_psf_shifted

def inverse_fourier_transform(ft_psf_shifted):
    # Shift back and compute Inverse Fourier Transform
    ft_psf = np.fft.ifftshift(ft_psf_shifted)
    psf_reconstructed = np.fft.ifft2(ft_psf)
    return np.abs(psf_reconstructed)

def fft_ifft(psf):
    # Compute the Fourier Transform
    ft_psf = np.fft.fft2(psf)
    ft_psf_shifted = np.fft.fftshift(ft_psf)  # Shift zero frequency to center
    # Shift back and compute Inverse Fourier Transform
    ft_psf = np.fft.ifftshift(ft_psf_shifted)
    psf_reconstructed = np.fft.ifft2(ft_psf)
    return np.abs(psf_reconstructed)

def convolve2d(image, psf, mode='same', boundary='wrap'):
    """
    Convolve an image with a PSF using 2D FFT-based convolution.
    Always returns an image of the same dimensions as the input.

    Parameters:
    image (np.ndarray): The input image.
    psf (np.ndarray): The PSF.
    mode (str): Not used in FFT implementation, kept for API consistency.
    boundary (str): Not used in FFT implementation, kept for API consistency.

    Returns:
    np.ndarray: The convolved image with same dimensions as input.
    """
    # Ensure PSF is centered in an array of the same size as the image
    psf_padded = np.zeros(image.shape, dtype=psf.dtype)
    psf_center = np.array(psf.shape) // 2
    image_center = np.array(image.shape) // 2
    
    # Calculate the corner positions for placing the PSF
    top = image_center[0] - psf_center[0]
    left = image_center[1] - psf_center[1]
    
    # Handle PSFs larger than the image
    psf_top = max(0, -top)
    psf_left = max(0, -left)
    psf_bottom = min(psf.shape[0], image.shape[0] - top)
    psf_right = min(psf.shape[1], image.shape[1] - left)
    
    # Handle image boundaries
    img_top = max(0, top)
    img_left = max(0, left)
    img_bottom = min(image.shape[0], top + psf.shape[0])
    img_right = min(image.shape[1], left + psf.shape[1])
    
    # Place the PSF in the padded array
    psf_padded[img_top:img_bottom, img_left:img_right] = psf[psf_top:psf_bottom, psf_left:psf_right]
    
    # Perform FFT convolution
    ft_image = np.fft.fft2(image)
    ft_psf = np.fft.fft2(psf_padded)
    ft_result = ft_image * ft_psf
    result = np.abs(np.fft.ifft2(ft_result))
    
    return np.fft.fftshift(result)

def sample_half_gaussian(size=1, sigma=0.5, seed=None):
    """
    Generate samples from a half-Gaussian distribution with given sigma.
    Ensures that all values are strictly greater than zero.

    Parameters:
    size (int): Number of samples to generate.
    sigma (float): Standard deviation of the full Gaussian.
    seed (int, optional): Seed for reproducibility.

    Returns:
    np.ndarray: Array of sampled values.
    """
    rng = np.random.default_rng(seed)  # Use numpy's random generator with seed
    samples = np.abs(rng.normal(loc=0, scale=sigma, size=size))
    samples = samples[samples > 0.14]  # Remove zeros
    while len(samples) < size:
        extra_samples = np.abs(rng.normal(loc=0, scale=sigma, size=size - len(samples)))
        extra_samples = extra_samples[extra_samples > 0.14]
        samples = np.concatenate((samples, extra_samples))
    return samples

def g1_g2_sigma_sample(num_samples=10000, g_std=0.26, sigma_std=0.5, seed=None):
    """
    Generate samples for g1, g2, and sigma. g1 and g2 are sampled from a Gaussian 
    distribution with specified standard deviation, and sigma from a half-Gaussian distribution.

    Parameters:
    num_samples (int): Number of samples to generate.
    g_std (float): Standard deviation for the Gaussian distribution of g1 and g2. Default is 0.26.
    sigma_std (float): Standard deviation for the half-Gaussian distribution of sigma. Default is 0.5.
    seed (int, optional): Seed for reproducibility.

    Returns:
    tuple: Arrays of sampled g1, g2, and sigma values.
    """
    rng = np.random.default_rng(seed)  # Use numpy's random generator with seed
    sigma = sample_half_gaussian(size=num_samples, sigma=sigma_std, seed=seed)
    
    # Generate g1 and g2 from Gaussian distribution with specified std
    g1_selected = np.zeros(num_samples)
    g2_selected = np.zeros(num_samples)

    for i in range(num_samples):
        while True:
            # Sample from Gaussian with specified std (mean=0)
            g1, g2 = rng.normal(0, g_std, 2)
            if np.abs(g1 + 1j * g2) < 1.0:  # Check the constraint
                g1_selected[i] = g1
                g2_selected[i] = g2
                break  # Accept only valid values

    return g1_selected, g2_selected, sigma

def make_struct(res, obs, shear_type):
    """
    make the data structure

    Parameters
    ----------
    res: dict
        With keys 's2n', 'e', 'T', and 'g_cov'
    obs: ngmix.Observation
        The observation for this shear type
    shear_type: str
        The shear type

    Returns
    -------
    1-element array with fields
    """
    dt = [
        ('flags', 'i4'),
        ('shear_type', 'U7'),
        ('s2n', 'f8'),
        ('g', 'f8', 2),
        ('T', 'f8'),
        ('flux', 'f8'),
        ('Tpsf', 'f8'),
        ('g_cov', 'f8', (2, 2)),
    ]
    data = np.zeros(1, dtype=dt)
    data['shear_type'] = shear_type
    data['flags'] = res['flags']

    if res['flags'] == 0:
        data['s2n'] = res['s2n']
        # for Gaussian moments we are actually measuring e, the ellipticity
        try:
            data['g'] = res['e']
        except KeyError:
            data['g'] = res['g']
        data['T'] = res['T']
        data['flux'] = res['flux']
        data['g_cov'] = res.get('g_cov', np.nan * np.ones((2, 2)))
    else:
        data['s2n'] = np.nan
        data['g'] = np.nan
        data['T'] = np.nan
        data['flux'] = np.nan
        data['Tpsf'] = np.nan
        data['g_cov'] = np.nan * np.ones((2, 2))

    # Get the psf T by averaging over epochs (and eventually bands)
    tpsf_list = []
    
    try:
        tpsf_list.append(obs.psf.meta['result']['T'])
    except:
        print(f"No PSF T found for observation")
            
    data['Tpsf'] = np.mean(tpsf_list)

    return data

def _get_priors(seed):

    # This bit is needed for ngmix v2.x.x
    # won't work for v1.x.x
    rng = np.random.RandomState(seed)

    # prior on ellipticity.  The details don't matter, as long
    # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014

    g_sigma = 0.3
    g_prior = ngmix.priors.GPriorBA(g_sigma, rng=rng)

    # 2-d gaussian prior on the center
    # row and column center (relative to the center of the jacobian, which would be zero)
    # and the sigma of the gaussians

    # units same as jacobian, probably arcsec
    row, col = 0.0, 0.0
    row_sigma, col_sigma = 0.2, 0.2 # a bit smaller than pix size of SuperBIT
    cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma, rng=rng)

    # T prior.  This one is flat, but another uninformative you might
    # try is the two-sided error function (TwoSidedErf)

    Tminval = -1.0 # arcsec squared
    Tmaxval = 1000
    T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

    # similar for flux.  Make sure the bounds make sense for
    # your images

    Fminval = -1.e1
    Fmaxval = 1.e5
    F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval, rng=rng)

    # now make a joint prior.  This one takes priors
    # for each parameter separately
    priors = ngmix.joint_prior.PriorSimpleSep(
    cen_prior,
    g_prior,
    T_prior,
    F_prior)

    return priors

def get_em_ngauss(name):
    ngauss=int( name[2:] )
    return ngauss

def get_coellip_ngauss(name):
    ngauss=int( name[7:] )
    return ngauss

def process_obs(obs, boot):
    resdict, obsdict = boot.go(obs)
    dlist = [make_struct(res=sres, obs=obsdict[stype], shear_type=stype) for stype, sres in resdict.items()]
    return np.hstack(dlist)

def mp_fit_one(obslist, prior, rng, psf_model='gauss', gal_model='gauss', mcal_pars= {'psf': 'dilate', 'mcal_shear': 0.01}):
    """
    Multiprocessing version of original _fit_one()

    Method to perfom metacalibration on an object. Returns the unsheared ellipticities
    of each galaxy, as well as entries for each shear step

    inputs:
    - obslist: Observation list for MEDS object of given ID
    - prior: ngmix mcal priors
    - mcal_pars: mcal running parameters

    TO DO: add a label indicating whether the galaxy passed the selection
    cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).
    """

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # get image pixel scale (assumes constant across list)
    jacobian = obslist[0]._jacobian
    Tguess = 4*jacobian.get_scale()**2
    ntry = 20
    lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
    psf_lm_pars={'maxfev': 4000, 'xtol':5.0e-5,'ftol':5.0e-5}

    fitter = ngmix.fitting.Fitter(model=gal_model, prior=prior, fit_pars=lm_pars)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=Tguess, prior=prior)

    # psf fitting
    if 'em' in psf_model:
        em_pars={'tol': 1.0e-6, 'maxiter': 50000}
        psf_ngauss = get_em_ngauss(psf_model)
        psf_fitter = ngmix.em.EMFitter(maxiter=em_pars['maxiter'], tol=em_pars['tol'])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif 'coellip' in psf_model:
        psf_ngauss = get_coellip_ngauss(psf_model)
        psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss, fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif psf_model == 'gauss':
        psf_fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
    else:
        raise ValueError('psf_model must be one of emn, coellipn, or gauss')

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=ntry)

    runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=ntry)

    #types = ['noshear', '1p', '1m', '2p', '2m']
    psf = mcal_pars['psf']
    mcal_shear = mcal_pars['mcal_shear']
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng,
        psf=psf,
        step = mcal_shear,
        #types=types,
    )



    num_cores = cpu_count()
    print(f"Starting NGmix ML fitting: num_gal: {len(obslist)} | psf_model: {psf_model} | gal_model: {gal_model} | num_cores: {num_cores}")

    with Pool(num_cores) as pool:
        data_list = list(pool.starmap(process_obs, [(obs, boot) for obs in obslist]))

    return data_list

def ngmix_pred(data_list):
    g1 = np.array([d[0][3][0] for d in data_list])
    g2 = np.array([d[0][3][1] for d in data_list])
    T = np.array([d[0][4] for d in data_list])
    # T = 2 * sigma**2
    sigma = np.sqrt(T / 2)
    flux = np.array([d[0][5] for d in data_list])
    preds = np.vstack((g1, g2, sigma, flux)).T
    return preds

def mp_fit_one_single(obslist, prior, rng, psf_model='gauss', gal_model='gauss', mcal_pars= {'psf': 'dilate', 'mcal_shear': 0.01}):
    """
    Multiprocessing version of original _fit_one()

    Method to perfom metacalibration on an object. Returns the unsheared ellipticities
    of each galaxy, as well as entries for each shear step

    inputs:
    - obslist: Observation list for MEDS object of given ID
    - prior: ngmix mcal priors
    - mcal_pars: mcal running parameters

    TO DO: add a label indicating whether the galaxy passed the selection
    cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).
    """
    # get image pixel scale (assumes constant across list)
    jacobian = obslist[0]._jacobian
    Tguess = 4*jacobian.get_scale()**2
    ntry = 20
    lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
    psf_lm_pars={'maxfev': 4000, 'xtol':5.0e-5,'ftol':5.0e-5}

    fitter = ngmix.fitting.Fitter(model=gal_model, prior=prior, fit_pars=lm_pars)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=Tguess, prior=prior)

    # psf fitting
    if 'em' in psf_model:
        em_pars={'tol': 1.0e-6, 'maxiter': 50000}
        psf_ngauss = get_em_ngauss(psf_model)
        psf_fitter = ngmix.em.EMFitter(maxiter=em_pars['maxiter'], tol=em_pars['tol'])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif 'coellip' in psf_model:
        psf_ngauss = get_coellip_ngauss(psf_model)
        psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss, fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif psf_model == 'gauss':
        psf_fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
    else:
        raise ValueError('psf_model must be one of emn, coellipn, or gauss')

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=ntry)

    runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=ntry)

    #types = ['noshear', '1p', '1m', '2p', '2m']
    psf = mcal_pars['psf']
    mcal_shear = mcal_pars['mcal_shear']
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng,
        psf=psf,
        step = mcal_shear,
        #types=types,
    )



    num_cores = cpu_count()
    print(f"Using {num_cores} cores out of {cpu_count()} available.")

    data_list  = []
    resdict_list = []
    obsdict_list = []
    for i in tqdm(range(len(obslist))):
        resdict, obsdict = boot.go(obslist[i])
        dlist = []
        for stype, sres in resdict.items():
            st = make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
            dlist.append(st)

        data = np.hstack(dlist)
        data_list.append(data)
        resdict_list.append(resdict) 

    return data_list, resdict_list

def get_memory_usage(obj):
    from pympler import asizeof
    """Prints the memory usage of each attribute in an object in MB."""
    memory_usage = {attr: asizeof.asizeof(getattr(obj, attr)) / (1024 * 1024) for attr in obj.__dict__}
    
    for attr, size in memory_usage.items():
        print(f"{attr}: {size:.6f} MB")
    
    total_memory = sum(memory_usage.values())
    print(f"\nTotal memory used by instance: {total_memory:.6f} MB")

def response_calculation(data_list, mcal_shear):
    r11_list, r22_list, r12_list, r21_list, c1_list, c2_list, c1_psf_list, c2_psf_list = [], [], [], [], [], [], [], []

    for i in tqdm(range(len(data_list))):
        g_noshear = data_list[i][0][3]
        g_1p = data_list[i][1][3]
        g_1m = data_list[i][2][3]
        g_2p = data_list[i][3][3]
        g_2m = data_list[i][4][3]
        g_1p_psf =  data_list[i][5][3]
        g_1m_psf =  data_list[i][6][3]
        g_2p_psf =  data_list[i][7][3]
        g_2m_psf =  data_list[i][8][3]
        r11 = (g_1p[0] - g_1m[0])/(2. * mcal_shear)
        r22 = (g_2p[1] - g_2m[1])/(2. * mcal_shear)
        r12 = (g_2p[0] - g_2m[0])/(2. * mcal_shear)
        r21 = (g_1p[1] - g_1m[1])/(2. * mcal_shear)
        c1 = (g_1p[0] + g_1m[0])/2 - g_noshear[0]
        c2 = (g_2p[1] + g_2m[1])/2 - g_noshear[1]
        c1_psf = (g_1p_psf[0] + g_1m_psf[0])/2 - g_noshear[0]
        c2_psf = (g_2p_psf[1] + g_2m_psf[1])/2 - g_noshear[1]
        r11_list.append(r11)
        r22_list.append(r22)
        r12_list.append(r12)
        r21_list.append(r21)
        c1_list.append(c1)
        c2_list.append(c2)
        c1_psf_list.append(c1_psf)
        c2_psf_list.append(c2_psf)

    return r11_list, r22_list, r12_list, r21_list, c1_list, c2_list, c1_psf_list, c2_psf_list