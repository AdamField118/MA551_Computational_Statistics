Here's a comprehensive project plan that covers both:

---

## **Project: Understanding Shear Measurement Systematics Through Custom Simulation**

### **Phase 1: Build Core Simulation Engine (Weeks 1-5)**

**Week 1-2: Galaxy Profile Implementation**
- Implement 2D Gaussian galaxy profile
- Implement exponential profile (Sérsic n=1)
- Test: verify profiles integrate to correct total flux
- Test: create visualizations showing profile shapes

**Week 3: Shear Transformation**
- Implement reduced shear transformation matrix
- Apply shear to coordinate grid before profile evaluation
- Test: verify ellipticity changes match theoretical predictions
- Deliverable: function `apply_shear(profile, g1, g2)`

**Week 4: PSF Convolution**
- Implement Gaussian PSF
- FFT-based convolution using R's `fft()` or `fftwtools`
- Test: verify convolution preserves flux
- Test: measure PSF-convolved sizes match expectations
- Deliverable: function `convolve_psf(galaxy, psf)`

**Week 5: Noise and Image Generation**
- Add Poisson photon noise
- Add Gaussian read noise
- Add constant sky background
- Deliverable: complete `simulate_galaxy_image(params)` function

**Phase 1 Checkpoint:** Can generate realistic sheared galaxy images with known input parameters

---

### **Phase 2: Implement Shape Measurement Methods (Weeks 6-8)**

**Week 6: Moments-based Measurement**
- Implement weighted second moments (adaptive or fixed weight)
- Calculate ellipticity from moments: e = (Q11 - Q22)/(Q11 + Q22)
- Account for weight function bias
- Deliverable: `measure_moments(image, method="KSB-like")`

**Week 7: Alternative Estimator (choose one)**
Option A: Simple model fitting (fit elliptical Gaussian)
Option B: Different moment scheme (unweighted, Gaussian-weighted)
- Implement using optimization (`optim()` or `nlminb()`)
- Deliverable: `measure_shapes_modelfit(image)` or alternative moments

**Week 8: Measurement Pipeline**
- Centroid finding
- Size estimation
- Quality cuts (SNR, resolution)
- Deliverable: robust end-to-end measurement function

**Phase 2 Checkpoint:** Can measure shapes from simulated images

---

### **Phase 3: Systematics Study (Weeks 9-12)**

**Week 9: Monte Carlo Framework**
- Design parameter grid: shear (g1, g2), galaxy size, flux, ellipticity
- Generate N realizations per parameter combination (bootstrap concept)
- Store true vs measured values
- Use parallel processing if needed (`parallel` package)

**Week 10: Bias Quantification**
- Calculate multiplicative bias: m = (g_meas - g_true)/g_true
- Calculate additive bias: c = g_meas - g_true (when g_true = 0)
- Fit linear models to bias vs parameters
- Implement jackknife/bootstrap for uncertainty on bias estimates

**Week 11: Parameter Dependencies**
Study how biases depend on:
- SNR (signal-to-noise ratio)
- Galaxy size relative to PSF
- Intrinsic galaxy ellipticity
- PSF size
- Create diagnostic plots showing bias(SNR), bias(size), etc.

**Week 12: Method Comparison**
- Compare biases between your two measurement methods
- Identify which galaxy properties cause largest biases
- Compute requirements: what accuracy do you need for cosmic shear?
- Connect to ShearNet: which biases would ML need to overcome?

**Phase 3 Checkpoint:** Complete characterization of measurement biases

---

### **Phase 4: Deliverables (Weeks 13-14)**

**Week 13: Analysis & Visualization**
- Publication-quality plots (use `ggplot2`)
- Bias as function of galaxy parameters
- Residual distributions
- Method comparison tables

**Week 14: Presentation & Report**
- **Presentation (15%):** 
  - Motivation from weak lensing
  - Your simulation approach
  - Key systematic findings
  - Connection to why ML methods like ShearNet are needed
  
- **Report (20%):**
  - Introduction: weak lensing and shear measurement challenge
  - Methods: simulation implementation, measurement algorithms
  - Results: bias characterization, parameter dependencies
  - Discussion: implications for ShearNet training data
  - Code appendix or GitHub repo

---

### **Key Computational Statistics Concepts Demonstrated**

✓ **Monte Carlo simulation:** generating galaxy populations
✓ **Stochastic sampling:** noise realizations  
✓ **Numerical optimization:** model fitting for shape measurement
✓ **Resampling methods:** bootstrap/jackknife for bias uncertainties
✓ **Statistical inference:** quantifying and modeling biases
✓ **Randomization tests:** could add permutation tests for bias significance

---

### **Suggested Scope Adjustments**

**If ahead of schedule:**
- Add second PSF model (Moffat profile)
- Implement metacalibration-style approach
- Study blending effects (overlapping galaxies)

**If behind schedule:**
- Use only Gaussian galaxies and PSF (still pedagogically valid)
- Focus on one measurement method
- Reduce parameter grid size

---

### **Deliverable Structure**

```
shear-systematics-r/
├── R/
│   ├── profiles.R          # Galaxy profile functions
│   ├── shear.R             # Shear transformations
│   ├── convolution.R       # PSF convolution
│   ├── noise.R             # Noise models
│   ├── simulate.R          # Main simulation function
│   ├── measure.R           # Shape measurement
│   └── bias_analysis.R     # Systematics quantification
├── scripts/
│   ├── run_simulations.R   # Generate dataset
│   └── analyze_biases.R    # Analyze results
├── tests/
│   └── test_*.R            # Unit tests
├── report/
│   └── final_report.Rmd    # R Markdown report
└── README.md
```

---

**Does this timeline feel realistic given your other commitments? Would you want to adjust the scope of Phase 2 or 3?**
