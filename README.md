# Sindy_MLDM_Project

# PDE Discovery from Unregistered 2D Images using SINDy

This project builds a full pipeline for recovering PDE coefficients directly from **noisy, misaligned 2D spatiotemporal images**. The goal is to register real images, compute accurate derivatives, and apply **Sparse Identification of Nonlinear Dynamics (SINDy)** to extract the governing PDE (e.g., the Kuramoto–Sivashinsky equation).

**Core Steps:**
- **Image Registration:** Compare No Registration, Cross-Correlation, and FFT-based Phase Correlation to realign frames before differentiation.
- **Noise & Denoising:** Test synthetic Gaussian noise, spatial shifts, and varying initial conditions. Apply denoising methods (spectral filtering, Gaussian smoothing, Savitzky–Golay) to improve coefficient recovery.
- **Derivative Computation:** Evaluate Finite-Difference vs. FFT Spectral derivatives for robustness under noise and misalignment.
- **SINDy Regression:** Recover coefficients of  
  \[
  u_t = a\,u_{xx} + b\,u_{xxxx} + c\,\partial_x(u^2)
  \]
  using sparse regression on the registered data.
**Outcome:**  
The pipeline demonstrates that proper registration + spectral derivatives + denoising significantly improves SINDy’s ability to recover PDE coefficients from noisy 2D images, providing a foundation for applying SINDy to real experimental datasets.

**Team:** MLDM Project Group, Université Jean Monnet (2025)
