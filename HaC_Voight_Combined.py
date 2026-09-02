import os
import glob
import sys
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy.special import wofz

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# --- Calibration constants (pixel <-> wavelength) ---
PIXEL_CENTER_REF = 800.0
LAMBDA_CENTER_REF_NM = 658.0
R0_NM_PER_PX = 1.304735e-02
R1_NM_PER_PX_PER_NM = -1.285110e-05

# --- Physics & Fitting Constants ---
lambda_Ha = 656.28    # H-alpha wavelength in nm
lambda_C1 = 657.80482  # C II 3p 2P*(3/2), NIST (was 657.736)
lambda_C2 = 658.28761  # C II 3p 2P*(1/2), NIST (was 658.1978)
# inst_sigma_Ha / _C1 / _C2 are derived from inst_fwhm_Ha further down,
# once width_nm_to_px exists (it needs nm_to_pixel, defined below).
inst_fwhm_Ha = 0.05   # Instrumental FWHM for H-alpha in nm
k_ev = 11600          # 11600 Kelvin = 1 eV
mH = 1                # Hydrogen mass in amu
mC = 12               # Carbon mass in amu

# --- Wavelength window of interest ---
MIN_WAVELENGTH_NM = 654.0
MAX_WAVELENGTH_NM = 659.0

# --- Detector / acquisition settings ---
DETECTOR_X_MAX = 1600
Y_RANGE = (1, 4)  # rows of the ProEM frame averaged into the 1-D profile

# --- Baseline correction settings ---
DO_BASELINE_CORRECT = True   
BASELINE_POLY_DEGREE = 1
BASELINE_ENFORCE_NONNEGATIVE = True

# --- C3 data root ---
C3_BASE_DIR = r""
C3_BASE_DIR = C3_BASE_DIR if C3_BASE_DIR else os.getcwd()


# --- Mathematical Fitting Functions ---
def voigt(x, amplitude, center, sigma, gamma, offset):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi)) + offset

def gaussian(x, amplitude, center, sigma, offset):
    norm_factor = amplitude / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - center)**2) / (2 * sigma**2)
    return norm_factor * np.exp(exponent) + offset

def combined_model(x,
                   amp_Ha, cen_Ha, sig_Ha, gam_Ha,
                   amp_C1, cen_C1, sig_C1, gam_C1,
                   amp_C2, cen_C2, sig_C2, gam_C2,
                   offset):
    """
    Sum of 3 Voigt profiles (Ha, C1, C2) + global offset.

    C2 is a SINGLE Voigt, matching C1. It used to be split into C2a/C2b and
    the temperature was then taken from whichever of the two came out larger,
    which made T_C1 and T_C2 measurements of different things.
    """
    y_Ha = voigt(x, amp_Ha, cen_Ha, sig_Ha, gam_Ha, 0)
    y_C1 = voigt(x, amp_C1, cen_C1, sig_C1, gam_C1, 0)
    y_C2 = voigt(x, amp_C2, cen_C2, sig_C2, gam_C2, 0)
    return y_Ha + y_C1 + y_C2 + offset


# --- Utility Functions ---
def pixel_to_nm(x_px):
    x = np.asarray(x_px, dtype=float)
    a = float(R0_NM_PER_PX)
    b = float(R1_NM_PER_PX_PER_NM)
    x0 = float(PIXEL_CENTER_REF)
    lam0 = float(LAMBDA_CENTER_REF_NM)
    if abs(b) < 1e-15:
        return lam0 + a * (x - x0)
    return (-a / b) + (lam0 + a / b) * np.exp(b * (x - x0))

def nm_to_pixel(lam_nm):
    a = float(R0_NM_PER_PX)
    b = float(R1_NM_PER_PX_PER_NM)
    x0 = float(PIXEL_CENTER_REF)
    lam0 = float(LAMBDA_CENTER_REF_NM)
    lam_nm = float(lam_nm)
    if abs(b) < 1e-15:
        return x0 + (lam_nm - lam0) / a
    const_term = -a / b
    amp = lam0 + a / b
    return float(x0 + np.log((lam_nm - const_term) / amp) / b)

# --- Width conversion -------------------------------------------------------
# A pixel WIDTH is not a pixel POSITION, so it cannot be passed to pixel_to_nm
# directly. Convert widths by DIFFERENCING pixel_to_nm around the feature's own
# fitted centre. There used to be a dispersion_nm_per_px() here holding a
# closed-form derivative, and widths were converted by multiplying by it; that
# was a second, independent implementation of the wavelength scale and it
# silently disagreed with pixel_to_nm by 2.83x. These helpers hold no formula
# of their own, so they cannot drift out of sync with the instrument model.

def width_px_to_nm(width_px, center_px):
    """WIDTH in px -> nm, centred on the line so neither side is privileged."""
    c = np.asarray(center_px, dtype=float)
    w = np.asarray(width_px, dtype=float)
    return float((pixel_to_nm(c + w) - pixel_to_nm(c - w)) / 2.0)


def width_nm_to_px(width_nm, center_nm):
    """WIDTH in nm -> px, anchored at the line centre (no pixel centre yet)."""
    lam = float(center_nm)
    return nm_to_pixel(lam + float(width_nm)) - nm_to_pixel(lam)


# Derived instrumental widths. Defined here rather than beside inst_fwhm_Ha
# because converting an assumed nm width into pixels now goes through
# width_nm_to_px, which needs nm_to_pixel.
# Under the corrected conversion this is 4.60 px, not the 1.63 px the old
# R0-based multiplication gave: the assumption was ~1.48x too small, not 4.2x.
inst_sigma_Ha = width_nm_to_px(inst_fwhm_Ha / 2.35482, lambda_Ha)
inst_sigma_C1 = inst_sigma_Ha    # for now until i ask about it, dont want to bound the fit
inst_sigma_C2 = inst_sigma_Ha   # for now until i ask about it, dont want to bound the fit


def load_full_profile(tiff_path, y_range, x_min=0, x_max=1601):
    data = np.array(tiff.imread(tiff_path))
    xmin = max(0, int(x_min))
    xmax = min(int(x_max), data.shape[1])
    roi = data[y_range[0]:y_range[1], xmin:xmax]
    profile = np.mean(roi, axis=0)
    x = np.arange(xmin, xmax)
    return x, profile

def estimate_polynomial_baseline_from_edge_windows(x_px, y, left_window, right_window, degree=1):
    x_px = np.asarray(x_px, dtype=float)
    y = np.asarray(y, dtype=float)

    def _window_mask(win):
        lo, hi = float(win[0]), float(win[1])
        if lo > hi:
            lo, hi = hi, lo
        return np.isfinite(x_px) & np.isfinite(y) & (x_px >= lo) & (x_px <= hi)

    m_edge = _window_mask(left_window) | _window_mask(right_window)
    if np.count_nonzero(m_edge) < 5:
        mf = np.isfinite(x_px) & np.isfinite(y)
        if np.count_nonzero(mf) < 2:
            return np.array([0.0], dtype=float)
        x_fit, y_fit = x_px[mf], y[mf]
    else:
        x_fit, y_fit = x_px[m_edge], y[m_edge]

    deg = int(max(0, min(int(degree), max(0, x_fit.size - 1))))
    if x_fit.size < 2:
        return np.array([0.0], dtype=float)
    try:
        coeff = np.polyfit(x_fit, y_fit, deg=deg)
    except Exception:
        coeff = np.polyfit(x_fit, y_fit, deg=1)
    return np.asarray(coeff, dtype=float)

def subtract_polynomial_baseline(x_px, y, coeff):
    x_px = np.asarray(x_px, dtype=float)
    y = np.asarray(y, dtype=float)
    coeff = np.asarray(coeff, dtype=float)
    if coeff.size == 0:
        return y
    return y - np.polyval(coeff, x_px)

def apply_baseline_correction(x_px, profile):
    coeff = np.array([0.0])
    lift = 0.0

    if not DO_BASELINE_CORRECT:
        return profile, coeff, lift

    _left_pix = int(np.floor(nm_to_pixel(MIN_WAVELENGTH_NM)))
    _right_pix = int(np.ceil(nm_to_pixel(MAX_WAVELENGTH_NM)))
    left_window = (max(0, _left_pix), max(0, _left_pix) + 150)
    right_window = (max(750, min(1450, _right_pix - 150)), min(DETECTOR_X_MAX, _right_pix))

    coeff = estimate_polynomial_baseline_from_edge_windows(
        x_px, profile, left_window=left_window, right_window=right_window, degree=BASELINE_POLY_DEGREE
    )
    corrected_profile = subtract_polynomial_baseline(x_px, profile, coeff)

    if BASELINE_ENFORCE_NONNEGATIVE:
        min_after = float(np.nanmin(corrected_profile))
        if np.isfinite(min_after) and min_after < 0.0:
            lift = -min_after
            corrected_profile = corrected_profile + lift

    return corrected_profile, coeff, lift

def prompt_c3_tiff_path(exp_i=None, frame_i=None):
    proem_dir = os.path.join(C3_BASE_DIR, f"C{exp_i}", "ProEm")
    frame_token = f"{frame_i:02d}"
    candidates = []
    candidates += sorted(glob.glob(os.path.join(proem_dir, f"*-Frame-{frame_token}.tif")))
    candidates += sorted(glob.glob(os.path.join(proem_dir, f"*-Frame-{frame_token}.tiff")))
    candidates += sorted(glob.glob(os.path.join(proem_dir, f"*-Frame-{frame_i}.tif")))
    candidates += sorted(glob.glob(os.path.join(proem_dir, f"*-Frame-{frame_i}.tiff")))
    candidates = list(dict.fromkeys(candidates))

    for full_path in candidates:
        if os.path.isfile(full_path):
            return full_path, exp_i, frame_i

    print(f"File not found in {proem_dir} for frame {frame_i}. Please re-enter.")
    raise FileNotFoundError(f"CRITICAL: File not found in {proem_dir} for frame {frame_i}.")


#########################################################################
########################## --- Main flow --- ############################
#########################################################################

def extract_nt_from_frame(exp_i, frame_i, do_plot=False):
    tiff_path, exp_number, frame_number = prompt_c3_tiff_path(exp_i,frame_i)
    print(f"\n=====================================")
    print(f"Loading C{exp_number} Frame {frame_number}")

    # Pixel range covering [MIN_WAVELENGTH_NM, MAX_WAVELENGTH_NM]
    _min_pix = max(0, int(np.floor(nm_to_pixel(MIN_WAVELENGTH_NM))))
    _max_pix = min(DETECTOR_X_MAX, int(np.ceil(nm_to_pixel(MAX_WAVELENGTH_NM))) + 1)

    x_px, profile = load_full_profile(tiff_path, Y_RANGE, x_min=_min_pix, x_max=_max_pix)
    profile = np.asarray(profile, dtype=float)

    # Pixel -> wavelength
    x_nm = pixel_to_nm(x_px)

    # Baseline correction
    profile, coeff, lift = apply_baseline_correction(x_px, profile)

    # We will fit over the entire MIN to MAX window
    mask = (x_nm >= MIN_WAVELENGTH_NM) & (x_nm <= MAX_WAVELENGTH_NM)
    x_px_fit = x_px[mask]
    x_nm_fit = x_nm[mask]
    prof_fit = profile[mask]

    # --- Generate Smart Initial Guesses ---
    px_Ha = nm_to_pixel(lambda_Ha)
    px_C1 = nm_to_pixel(lambda_C1)
    px_C2 = nm_to_pixel(lambda_C2)

    def get_local_max(px_center, window_px=15):
        # Finds the peak height near the expected center
        idx = np.argmin(np.abs(x_px_fit - px_center))
        return np.max(prof_fit[max(0, idx-window_px) : min(len(prof_fit), idx+window_px)])

    # Amplitudes in our functions represent Area. Area roughly = Height * 10
    A_Ha = max(1.0, get_local_max(px_Ha)) * 10
    A_C1 = max(1.0, get_local_max(px_C1)) * 10
    A_C2 = max(1.0, get_local_max(px_C2)) * 10

    # Initial guesses: [amp, center, sigma, gamma]
    p0 = [
        A_Ha, px_Ha, 3.0, 3.0,          # Ha (Voigt)
        A_C1, px_C1, 3.0, 3.0,          # C1 (Voigt)
        A_C2, px_C2, 3.0, 3.0,          # C2 (Voigt) - single, same as C1
        0.0                             # global offset
    ]

    # Strict bounds to prevent lines from swapping places. C2 now gets the
    # same +/-15 px window as C1 (it was +/-20 to let the old C2a/C2b pair
    # separate; a single component does not need that freedom).
    bounds_lower = [
        0, px_Ha - 30, 0, 0,
        0, px_C1 - 15, 0, 0,
        0, px_C2 - 15, 0, 0,
        -np.max(prof_fit)
    ]

    bounds_upper = [
        np.inf, px_Ha + 30, 100, 100,
        np.inf, px_C1 + 15, 100, 100,
        np.inf, px_C2 + 15, 100, 100,
        np.max(prof_fit)
    ]

    # Initialize return variables safely
    n_e_cm3, T_C1_eV, T_C2_eV = 0, 0, 0

    try:
        # --- Perform the Global Fit ---
        popt, _ = curve_fit(combined_model, x_px_fit, prof_fit, p0=p0, bounds=(bounds_lower, bounds_upper))
        
        # Unpack the 13 parameters
        (amp_Ha, cen_Ha, sig_Ha, gam_Ha,
         amp_C1, cen_C1, sig_C1, gam_C1,
         amp_C2, cen_C2, sig_C2, gam_C2,
         offset) = popt

        # --- Plasma Diagnostics ---
        
        # H-alpha Density Calculation
        gam_Ha_nm = width_px_to_nm(gam_Ha, cen_Ha)
        stark_fwhm_nm = 2 * gam_Ha_nm
        if stark_fwhm_nm > 0:
            n_e_cm3 = 10**17 * (stark_fwhm_nm / 1.098)**1.471
            print(f"Hα Electron Density (n_e):  {n_e_cm3:.2e} cm^-3")

        # C1 Temperature Calculation
        # For a Voigt profile, the Gaussian width (sigma) represents the thermal broadening
        sig_C1_nm = width_px_to_nm(sig_C1, cen_C1)
        inst_sigma_C1_nm = width_px_to_nm(inst_sigma_C1, cen_C1)
        if sig_C1_nm > inst_sigma_C1_nm:
            sig_th_C1_nm = math.sqrt(sig_C1_nm**2 - inst_sigma_C1_nm**2)
            T_C1_eV = (931.49e6 * mC) * (sig_th_C1_nm / lambda_C1)**2
            print(f"C1 Temperature (T):         {T_C1_eV:.2f} eV")

        # C2 Temperature Calculation
        # Single Voigt now, so this is the same measurement as C1 above - no
        # "pick the dominant peak" step, which previously made T_C1 and T_C2
        # derived from differently-constructed quantities.
        sig_C2_nm = width_px_to_nm(sig_C2, cen_C2)
        inst_sigma_C2_nm = width_px_to_nm(inst_sigma_C2, cen_C2)
        if sig_C2_nm > inst_sigma_C2_nm:
            sig_th_C2_nm = math.sqrt(sig_C2_nm**2 - inst_sigma_C2_nm**2)
            lambda_C2_fitted = pixel_to_nm(cen_C2) # Use precise fitted center
            T_C2_eV = (931.49e6 * mC) * (sig_th_C2_nm / lambda_C2_fitted)**2
            print(f"C2 Temperature (T):         {T_C2_eV:.2f} eV")

    except RuntimeError as e:
        print(f"WARNING: Global curve fitting failed for frame {frame_number}. Returning zeros.")
        return 0, 0, 0

    # --- Plotting the Fit (Only if do_plot is True) ---
    if do_plot:
        # Generate the lines for plotting
        fit_total = combined_model(x_px_fit, *popt)
        fit_Ha  = voigt(x_px_fit, amp_Ha, cen_Ha, sig_Ha, gam_Ha, offset)
        fit_C1  = voigt(x_px_fit, amp_C1, cen_C1, sig_C1, gam_C1, offset)
        fit_C2  = voigt(x_px_fit, amp_C2, cen_C2, sig_C2, gam_C2, offset)

        plt.figure(figsize=(10, 6))
        plt.plot(x_nm_fit, prof_fit, label="Raw Data", color='lightgray', linewidth=2)
        plt.plot(x_nm_fit, fit_total, 'k--', label="Global Fit (Sum)", linewidth=2)
        
        # Plot individual components
        plt.plot(x_nm_fit, fit_Ha, 'r-', alpha=0.6, label="Hα (Voigt)")
        plt.plot(x_nm_fit, fit_C1, 'g-', alpha=0.6, label="C1 (Voigt)")
        plt.plot(x_nm_fit, fit_C2, 'b-', alpha=0.6, label="C2 (Voigt)")
        
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity [a.u.]")
        plt.title(f"C{exp_number} Frame {frame_number} Global Multi-Peak Fit (Voigts)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return n_e_cm3, T_C1_eV, T_C2_eV


if __name__ == "__main__":
    # @CHANGE_ME: Set the experiment and frame numbers you want to analyze here
    exp_i = 559
    first_frame = 5
    last_frame = 22
    first_frame_t = 8
    last_frame_t = 22

    frame_amount = last_frame - first_frame + 1
    n = [0] * frame_amount
    t1 = [0] * frame_amount
    t2 = [0] * frame_amount # Array for the C2 temperatures
    
    for frame_i in range(first_frame, last_frame + 1):
        if(frame_i == 13):  # plot only for frame 28
            extract_nt_from_frame(exp_i, frame_i, do_plot=True)
            
        n[frame_i - first_frame], t1[frame_i - first_frame], t2[frame_i - first_frame] = extract_nt_from_frame(exp_i, frame_i, do_plot=False)

    # Plot n_e, T_C1, and T_C2 vs frame_i
    plt.figure(figsize=(15, 5)) 
    
    plt.subplot(1, 3, 1)
    plt.plot(range(first_frame, last_frame + 1), n, marker='o', color='blue')
    plt.xlabel("Frame Number")
    plt.ylabel("Electron Density (n_e) [cm^-3]")
    plt.title(f"C{exp_i} n_e vs Frame")
    plt.grid(True)
    
    t_start_idx = first_frame_t - first_frame
    t_end_idx = last_frame_t - first_frame + 1
    t_frames = range(first_frame_t, last_frame_t + 1)

    plt.subplot(1, 3, 2)
    plt.plot(t_frames, t1[t_start_idx:t_end_idx], marker='o', color='green')
    plt.xlabel("Frame Number")
    plt.ylabel("Temperature (T_C1) [eV]")
    plt.title(f"C{exp_i} T(C1) vs Frame")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(t_frames, t2[t_start_idx:t_end_idx], marker='o', color='orange')
    plt.xlabel("Frame Number")
    plt.ylabel("Temperature (T_C2) [eV]")
    plt.title(f"C{exp_i} T(C2) vs Frame")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()