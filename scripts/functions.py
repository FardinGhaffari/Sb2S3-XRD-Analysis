import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from pybaselines.whittaker import asls
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


def moving_average(data, window_size):
    """Applies a moving average filter to the data."""
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')  # 'same' keeps output the same size

def subtract_background(intensity, lam=1e5, p=0.0001):
    """
    Subtracts background using Asymmetric Least Squares (AsLS).
    Returns baseline and background-subtracted intensity.
    """
    baseline = asls(intensity, lam=lam, p=p)[0]
    corrected = intensity - baseline
    return baseline, corrected

def pseudo_voigt(x, amp, cent, fwhm, eta):
    """
    amp: Amplitude (height)
    cent: Center of peak
    fwhm: Full Width at Half Maximum
    eta: Mixing parameter (0 = Pure Gaussian, 1 = Pure Lorentzian)
    """
    # Sigma for Gaussian
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # Gamma for Lorentzian
    gamma = fwhm / 2.0
    
    gauss = np.exp(-(x - cent)**2 / (2 * sigma**2))
    lorentz = 1 / (1 + ((x - cent) / gamma)**2)
    
    return amp * (eta * lorentz + (1 - eta) * gauss)

def shift_spectrum_to_peak(two_theta, intensity, target_peak=37.9, window=0.6, plot=True):
    """
    Shifts the spectrum using a Pseudo-Voigt fit to align the 
    specified reference peak (e.g., FTO) with its theoretical position.
    """
    # 1. Mask data around the reference peak
    mask = (two_theta >= target_peak - window) & (two_theta <= target_peak + window)
    x_window = two_theta[mask]
    y_window = intensity[mask]
    
    if len(x_window) < 5:
        print("Warning: Insufficient data points for alignment.")
        return two_theta, 0.0

    # 2. Initial guesses for Pseudo-Voigt: [amp, cent, fwhm, eta]
    initial_guess = [np.max(y_window), target_peak, 0.15, 0.5]
    
    # 3. Fit using the previously defined pseudo_voigt function
    try:
        popt, _ = curve_fit(pseudo_voigt, x_window, y_window, p0=initial_guess)
        fitted_peak_center = popt[1]  # 'cent' is the second parameter
    except Exception as e:
        print(f"Alignment fit failed: {e}. Falling back to max intensity.")
        fitted_peak_center = x_window[np.argmax(y_window)]

    # 4. Calculate and apply the shift
    shift = target_peak - fitted_peak_center
    shifted_two_theta = two_theta + shift
    
    # --- Add this plotting block inside shift_spectrum_to_peak ---
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        
        # Plot the raw data points used for fitting
        plt.scatter(x_window, y_window, s=15, color='black', label='Data (FTO Region)')
        
        # Plot the smooth Pseudo-Voigt fit curve
        x_fine = np.linspace(x_window.min(), x_window.max(), 500)
        plt.plot(x_fine, pseudo_voigt(x_fine, *popt), 'r-', 
                 label=f'Pseudo-Voigt Fit (η={popt[3]:.2f})')
        
        # Visual lines for the shift
        plt.axvline(fitted_peak_center, color='blue', linestyle='--', label=f'Fitted: {fitted_peak_center:.3f}°')
        plt.axvline(target_peak, color='green', linestyle='-', alpha=0.6, label=f'Target: {target_peak}°')
        
        plt.title(f"Alignment Check | Shift: {shift:.4f}°")
        plt.xlabel("2θ (degrees)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

    return shifted_two_theta, shift


def calculate_fitted_peak_area(two_theta, intensity, target_peak, search_window=0.5, plot=True):
    # 1. Broad search for raw peak, then narrow symmetric mask
    mask_init = (two_theta >= target_peak - search_window) & (two_theta <= target_peak + search_window)
    p_guess = two_theta[mask_init][np.argmax(intensity[mask_init])]
    mask = (two_theta >= p_guess - search_window) & (two_theta <= p_guess + search_window)
    x_data, y_data = two_theta[mask], intensity[mask]

    if len(x_data) < 5: return 0, 0

    # 2. Initial Guesses [Amplitude, Center, FWHM, Eta]
    # We guess eta=0.5 (equal mix) and FWHM ~ 0.2 degrees
    initial_guess = [np.max(y_data), x_data[np.argmax(y_data)], 0.2, 0.5]
    
    # 3. Perform the Fit
    try:
        # Bounds: amp > 0, center must stay in window, fwhm > 0, eta between 0 and 1
        bounds = ([0, x_data.min(), 0.01, 0], 
                  [np.inf, x_data.max(), 1.0, 1])
        
        popt, _ = curve_fit(pseudo_voigt, x_data, y_data, p0=initial_guess, bounds=bounds)
        amp_fit, cent_fit, fwhm_fit, eta_fit = popt
    except Exception as e:
        print(f"Fit failed for peak {target_peak}: {e}")
        return 0, 0

    # 4. Define Area (4 * FWHM around the FITTED center)
    int_min = cent_fit - (2 * fwhm_fit)
    int_max = cent_fit + (2 * fwhm_fit)
    
    int_mask = (two_theta >= int_min) & (two_theta <= int_max)
    area = trapezoid(intensity[int_mask], x=two_theta[int_mask])

    # 5. Visualizing the fit quality
    if plot:
        plt.figure(figsize=(7, 4))
        plt.scatter(x_data, y_data, s=10, color='black', label='Data')
        x_fine = np.linspace(x_data.min(), x_data.max(), 500)
        plt.plot(x_fine, pseudo_voigt(x_fine, *popt), 'r-', label=f'Pseudo-Voigt Fit (η={eta_fit:.2f})')
        plt.fill_between(two_theta[int_mask], intensity[int_mask], color='green', alpha=0.2, label='Integrated Area')
        plt.axvline(cent_fit, color='blue', linestyle='--', alpha=0.5)
        plt.title(f"Peak: {cent_fit:.3f}° | FWHM: {fwhm_fit:.3f}°")
        plt.legend()
        plt.show()

    return area, fwhm_fit