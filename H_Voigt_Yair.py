import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz
import tifffile as tiff


# --- Constants ---
nm_per_px = 0.0039 # Conversion of pixels to nm
lambda_Ha = 656.28 # H-alpha wavelength in nm
lambda_C1 = 657.736 # C line wavelength in nm (approximate, adjust as needed)
lambda_C2 = 658.1978  # C2 line wavelength in nm (approximate, adjust as needed)
inst_fwhm_Ha = 0.05 # Instrumental FWHM for H-alpha in nm
inst_fwhm_Hb = 0.1 # Not used in this script
inst_sigma_Ha = inst_fwhm_Ha / (2.35482 * nm_per_px) 
inst_sigma_C1 = 0 # for now until i ask about it, dont want to bound the fit
inst_sigma_C2 = 0 # for now until i ask about it, dont want to bound the fit
# 2.35482 Convert instrumental FWHM to sigma in pixels for H-alpha assuming gaussian instrumental profile
inst_sigma_Hb = inst_fwhm_Hb / (2.35482 * nm_per_px) #Not used in this script
k_ev = 11600 # 11600 Kelvin = 1 eV
mH = 1 # Hydrogen mass in amu
mC = 12 # Carbon mass in amu

# --- Load and preprocess ---
def load_profile(tiff_path, x_range, y_range):
    data = np.array(tiff.imread(tiff_path))
    roi = data[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    profile = np.mean(roi, axis=0)
    x = np.arange(x_range[0], x_range[1])
    return x, profile

#Load full x range, just for plotting purposes
def load_full_profile(tiff_path, y_range, x_max=1300):
    """Load the full horizontal profile from pixel 0 to x_max (default 1300)."""
    data = np.array(tiff.imread(tiff_path))
    xmax = min(x_max, data.shape[1])
    roi = data[y_range[0]:y_range[1], 0:xmax]
    profile = np.mean(roi, axis=0)
    x = np.arange(0, xmax)
    return x, profile

def voigt(x, amplitude, center, sigma, gamma, offset):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi)) + offset

def gaussian(x, amplitude, center, sigma, offset):
    norm_factor = amplitude / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - center)**2) / (2 * sigma**2)
    return norm_factor * np.exp(exponent) + offset


# --- File paths and ranges ---
tiff_Ha = r"C:/Users/User/Documents/Technion/סמסטר 7/פרויקט ת/HVoigtYair/4Quik_LS_C77_656nm_Delay_20.25us_Exp_500ns_P_1.03e-2.tif"

yrange_Ha = (370, 560) #range of pixels in the y-axis of the Halpha line

### H-alpha Voigt fit ###

start_px_Ha = int(640)
end_px_Ha =  int(780)
# Full profiles from pixel 0 to 1300 (or image width if smaller)
x_full, profile_full = load_full_profile(tiff_Ha, yrange_Ha, x_max=1300)
print(type(start_px_Ha),type(end_px_Ha))
x_Ha = x_full[start_px_Ha:end_px_Ha]
profile_Ha = profile_full[start_px_Ha:end_px_Ha]

# --- Fit Hα ---
p0_Ha = [np.max(profile_Ha), x_Ha[np.argmax(profile_Ha)], 10, 10, np.min(profile_Ha)]
bounds_Ha = ([0, x_Ha[0], inst_sigma_Ha, 0, 0], [np.inf, x_Ha[-1], 100, 100, np.max(profile_Ha)])
popt_Ha, pcov_Ha = curve_fit(voigt, x_Ha, profile_Ha, p0=p0_Ha, bounds=bounds_Ha)
amplitude_Ha, center_Ha, sigma_Ha, gamma_Ha, offset_Ha = popt_Ha
fit_Ha = voigt(x_Ha, amplitude_Ha, center_Ha, sigma_Ha, gamma_Ha, offset_Ha)
print('sigmaHa:',sigma_Ha,'inst_sigma_Ha:',inst_sigma_Ha)
print("\n--- Fit Parameters ---")
sigma_Ha_nm = sigma_Ha * nm_per_px
gamma_Ha_nm = gamma_Ha * nm_per_px
print(f"Hα: sigma = {sigma_Ha:.4f} px = {sigma_Ha_nm:.4f} nm, gamma = {gamma_Ha:.4f} px = {gamma_Ha_nm:.4f} nm")

# --- Wavelength axis for Hα (use nominal nm_per_px) ---
x_nm_Ha = lambda_Ha + (x_Ha - center_Ha) * nm_per_px
x_nm_full = lambda_Ha + (x_full - center_Ha) * nm_per_px


### C1 line gaussian fit ###

start_px_C1 , end_px_C1 = 1065, 1100
# Full profiles from pixel 0 to 1300 (or image width if smaller)

x_C1 = x_full[start_px_C1:end_px_C1]
profile_C1 = profile_full[start_px_C1:end_px_C1]

# --- Fit C1 ---
p0_C1 = [np.max(profile_C1), x_C1[np.argmax(profile_C1)], 10, np.min(profile_C1)]
bounds_C1 = ([0, x_C1[0], inst_sigma_C1, 0], [np.inf, x_C1[-1], 100, np.max(profile_C1)])
popt_C1, pcov_C1 = curve_fit(gaussian, x_C1, profile_C1, p0=p0_C1, bounds=bounds_C1)
amplitude_C1, center_C1, sigma_C1, offset_C1 = popt_C1
fit_C1 = gaussian(x_C1, amplitude_C1, center_C1, sigma_C1, offset_C1)
print('sigmaC1:',sigma_C1,'inst_sigma_C1:',inst_sigma_C1)
print("\n--- Fit Parameters ---")
sigma_C1_nm = sigma_C1 * nm_per_px
print(f"C1: sigma = {sigma_C1:.4f} px = {sigma_C1_nm:.4f} nm")

# --- Wavelength axis for C1 (use nominal nm_per_px) ---
x_nm_C1 = lambda_C1 + (x_C1 - center_C1) * nm_per_px
x_nm_full_C1 = lambda_C1 + (x_full - center_C1) * nm_per_px

### C2 line gaussian fit ###

start_px_C1 , end_px_C1 = 1186, 1216
# Full profiles from pixel 0 to 1300 (or image width if smaller)
x_C2 = x_full[start_px_C1:end_px_C1]
profile_C2 = profile_full[start_px_C1:end_px_C1]

# --- Fit C2 ---
p0_C2 = [np.max(profile_C2), x_C2[np.argmax(profile_C2)], 10, np.min(profile_C2)]
bounds_C2 = ([0, x_C2[0], inst_sigma_C2, 0], [np.inf, x_C2[-1], 100, np.max(profile_C2)])
popt_C2, pcov_C2 = curve_fit(gaussian, x_C2, profile_C2, p0=p0_C2, bounds=bounds_C2)
amplitude_C2, center_C2, sigma_C2, offset_C2 = popt_C2
fit_C2 = gaussian(x_C2, amplitude_C2, center_C2, sigma_C2, offset_C2)
print('sigmaC2:',sigma_C2,'inst_sigma_C2:',inst_sigma_C2)
print("\n--- Fit Parameters ---")
sigma_C2_nm = sigma_C2 * nm_per_px
print(f"C2: sigma = {sigma_C2:.4f} px = {sigma_C2_nm:.4f} nm")

# --- Wavelength axis for C2 (use nominal nm_per_px) ---
x_nm_C2 = lambda_C2 + (x_C2 - center_C2) * nm_per_px
x_nm_full_C2 = lambda_C2 + (x_full - center_C2) * nm_per_px


# --- Plot fit ---
plt.figure(figsize=(8, 5))
plt.plot(x_nm_full, profile_full, label="Hα Data (full)")
plt.plot(x_nm_Ha, profile_Ha, color='gray', alpha=0.5, label="Hα ROI")
plt.plot(x_nm_Ha, fit_Ha, 'r--', label="Voigt Fit")
plt.plot(x_nm_C1, profile_C1, color='gray', alpha=0.5,label="C1 ROI")
plt.plot(x_nm_C1, fit_C1, 'g--', label="Gaussian Fit")
plt.plot(x_nm_C2, profile_C2, color='gray', alpha=0.5,label="C2 ROI")
plt.plot(x_nm_C2, fit_C2, 'b--', label="Gaussian Fit")
plt.title("Hα Voigt Fit")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity [a.u.]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# --- Plasma Diagnostics (T and n_e) --- (Yair's code) ---
import math

print("\n--- Plasma Diagnostics ---")

# 1. Electron Density (n_e) Calculation
# gamma_Ha is the Lorentzian HWHM. Stark FWHM is 2 * gamma.
stark_fwhm_nm = 2 * gamma_Ha_nm

# Using Formula 5 for H-alpha
if stark_fwhm_nm > 0:
    n_e_cm3 = 10**17 * (stark_fwhm_nm / 1.098)**1.471
    print(f"Stark FWHM (Lorentzian): {stark_fwhm_nm:.4f} nm")
    print(f"Electron Density (n_e):  {n_e_cm3:.2e} cm^-3")
else:
    print("Electron Density (n_e):  Unable to calculate (Stark FWHM is 0)")

# 2. Temperature (T) Calculation
# Convert instrumental sigma back to nm (or use inst_fwhm_Ha directly)
inst_sigma_Ha_nm = inst_sigma_Ha * nm_per_px

# Subtract instrumental broadening from total Gaussian broadening
if sigma_Ha_nm > inst_sigma_Ha_nm:
    sigma_th_Ha_nm = math.sqrt(sigma_Ha_nm**2 - inst_sigma_Ha_nm**2)
    
    # Using Formula 11: sigma_th = (lambda_0 / c) * sqrt(k_B T / m)
    # T_eV = m_H * c^2 * (sigma_th / lambda_0)^2
    # Mass of Hydrogen (mH = 1 amu) in eV/c^2 is ~ 931.49 * 10^6 eV
    mc2_H_eV = 931.49e6 * mH 
    
    T_H_eV = mc2_H_eV * (sigma_th_Ha_nm / lambda_Ha)**2
    T_H_K = T_H_eV * k_ev  # Convert to Kelvin using k_ev constant
    
    print(f"Thermal Gaussian Sigma:  {sigma_th_Ha_nm:.4f} nm")
    print(f"Temperature (T):         {T_H_eV:.2f} eV ({T_H_K:.0f} K)")
else:
    print(f"Thermal Gaussian Sigma:  0.0000 nm")
    print("Temperature (T):         Warning: Total Gaussian sigma is smaller than instrumental sigma. Cannot resolve temperature.")

if sigma_C1_nm > inst_sigma_C1:
    sigma_th_C1_nm = math.sqrt(sigma_C1_nm**2 - inst_sigma_C1**2)
    mc2_C_eV = 931.49e6 * mC
    T_C1_eV = mc2_C_eV * (sigma_th_C1_nm / lambda_C1)**2
    T_C1_K = T_C1_eV * k_ev
    print(f"\nC line Thermal Gaussian Sigma:  {sigma_th_C1_nm:.4f} nm")
    print(f"C line Temperature (T):         {T_C1_eV:.2f} eV ({T_C1_K:.0f} K)")

if sigma_C2_nm > inst_sigma_C2:
    sigma_th_C2_nm = math.sqrt(sigma_C2_nm**2 - inst_sigma_C2**2)
    mc2_C_eV = 931.49e6 * mC
    T_C2_eV = mc2_C_eV * (sigma_th_C2_nm / lambda_C2)**2
    T_C2_K = T_C2_eV * k_ev
    print(f"\nC line Thermal Gaussian Sigma:  {sigma_th_C2_nm:.4f} nm")
    print(f"C line Temperature (T):         {T_C2_eV:.2f} eV ({T_C2_K:.0f} K)")