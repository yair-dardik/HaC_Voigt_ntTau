import os
import glob
import sys
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit, least_squares
from scipy.sparse import lil_matrix
from scipy.special import wofz

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
#----------------------------
#USER-CONFIGURABLE PARAMETERS
#----------------------------

# ============================================================================
# ZEEMAN JOINT-FIT ANALYSIS: run settings (edit these to change a run)
# ============================================================================
# Frame range to fit. The scan should stop before the C II lines get too weak
# to constrain a width - past that point the fitted sigmas rail against their
# bounds in the OLD (non-Zeeman) pipeline too, those frames carry no line
# information but do carry residual, and letting them in only lets noise vote
# on the shared B. Override the upper end at the command line with --last N.
ZEEMAN_FIRST_FRAME = 5
ZEEMAN_LAST_FRAME = 25

# B search range for the Zeeman field fit, in tesla.
B_MIN_TESLA = 0.0
B_MAX_TESLA = 2.5

# The Zeeman pattern is symmetric, so to first order in B the blend is
# unchanged: d(model)/dB is exactly 0 at B = 0 and the response is quadratic.
# A gradient-based fit seeded near zero therefore cannot climb out of B = 0
# even when a much better minimum exists elsewhere. Every B fit below is
# started from this ladder and the best chi2 kept.
B_SEEDS = (1.2, 1.6, 2.0, 2.2, 2.4)

# Which width model(s) to run: "both" fits independent sigma_C1/sigma_C2 AND
# one shared sigma, then prints the head-to-head comparison; "independent" or
# "shared" runs only that one model.
ZEEMAN_WIDTH_MODEL = "both"

# Frame to draw a detailed single-frame fit picture for (data + all three
# models + Zeeman components). None skips that plot. Override at the command
# line with --frame N or --frame none.
ZEEMAN_PLOT_FRAME = 13

# Same ROI half-width the existing single-Gaussian C1/C2 fits use.
#everything outside that window is excluded before the model is fit.
ZEEMAN_ROI_HALFWIDTH_NM = 0.2

# Name of the experiment folder under C3_BASE_DIR, which must itself contain a
# ProEM/ subfolder holding the .tif frames (the same layout as before - only
# the folder's NAME is configurable, not its internal structure). Leave empty
# to fall back to the old convention of "C<exp_i>" (e.g. exp_i=559 -> "C559").
EXPERIMENT_DIR_NAME = ""

# --- C3 data root ---
# Set C3_BASE_DIR explicitly to pin the data location. Left empty, it is
# whichever of {cwd, this file's own folder} actually contains a
# <EXPERIMENT_DIR_NAME>/ProEM tree (or a "C*"/ProEM tree, if
# EXPERIMENT_DIR_NAME is left on its own default), checked in that order.
# This makes no assumption about this file's own location relative to the
# data - it only requires the experiment folder (e.g. C559/, or whatever
# EXPERIMENT_DIR_NAME names) to sit next to this script, wherever that is.
C3_BASE_DIR = r""
if not C3_BASE_DIR:
    _here = os.path.dirname(os.path.abspath(__file__))
    _dir_glob = EXPERIMENT_DIR_NAME or "C*"
    for _cand in (os.getcwd(), _here):
        if glob.glob(os.path.join(_cand, _dir_glob, "ProEM")):
            C3_BASE_DIR = _cand
            break
    else:
        C3_BASE_DIR = os.getcwd()
#----------------------------
#END OF USER-CONFIGURABLE PARAMETERS
#----------------------------

# --- Calibration constants (pixel <-> wavelength) ---
PIXEL_CENTER_REF = 800.0
LAMBDA_CENTER_REF_NM = 658.0
R0_NM_PER_PX = 1.304735e-02
R1_NM_PER_PX_PER_NM = -1.285110e-05

# --- Physics & Fitting Constants ---
lambda_Ha = 656.28    # H-alpha wavelength in nm
# NIST critically-evaluated air wavelengths (Kramida & Haris 2022). The old
# values 657.736 / 658.1978 were low by 0.069 / 0.090 nm, which pushed the true
# lines toward the red edge of their own get_roi windows and truncated the red
# wing into the free offset parameter.
lambda_C1 = 657.80482  # C II 3s 2S(1/2) - 3p 2P*(3/2), upper level g = 4
lambda_C2 = 658.28761  # C II 3s 2S(1/2) - 3p 2P*(1/2), upper level g = 2
inst_fwhm_Ha = 0.05   # Instrumental FWHM for H-alpha in nm (ASSUMED, not measured)
# inst_sigma_Ha / _C1 / _C2 are derived from inst_fwhm_Ha further down, once
# width_nm_to_px exists (it needs nm_to_pixel, defined below).
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
DO_BASELINE_CORRECT = True   # set False to skip baseline subtraction entirely
BASELINE_POLY_DEGREE = 1
BASELINE_ENFORCE_NONNEGATIVE = True



# --- Mathematical Fitting Functions ---
def voigt(x, amplitude, center, sigma, gamma, offset):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi)) + offset

def gaussian(x, amplitude, center, sigma, offset):
    norm_factor = amplitude / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - center)**2) / (2 * sigma**2)
    return norm_factor * np.exp(exponent) + offset

# --- Utility Functions ---

def pixel_to_nm(x_px):
    """Convert pixel coordinate(s) to wavelength in nm using the instrument model."""
    x = np.asarray(x_px, dtype=float)
    a = float(R0_NM_PER_PX)
    b = float(R1_NM_PER_PX_PER_NM)
    x0 = float(PIXEL_CENTER_REF)
    lam0 = float(LAMBDA_CENTER_REF_NM)
    if abs(b) < 1e-15:
        return lam0 + a * (x - x0)
    return (-a / b) + (lam0 + a / b) * np.exp(b * (x - x0))


def nm_to_pixel(lam_nm):
    """Inverse of pixel_to_nm."""
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
    """Average rows in y_range and return (x_px, profile) for x in [x_min, x_max)."""
    data = np.array(tiff.imread(tiff_path))
    xmin = max(0, int(x_min))
    xmax = min(int(x_max), data.shape[1])
    roi = data[y_range[0]:y_range[1], xmin:xmax]
    profile = np.mean(roi, axis=0)
    x = np.arange(xmin, xmax)
    return x, profile


def estimate_polynomial_baseline_from_edge_windows(
    x_px,
    y,
    left_window,
    right_window,
    degree=1,
):
    """Fit a polynomial baseline using only the two edge (line-free) windows."""
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
    """
    Applies edge-window polynomial baseline correction to a 1D spectral profile.
    Returns the corrected profile, the fitted coefficients, and the applied lift.
    """
    coeff = np.array([0.0])
    lift = 0.0

    # If disabled globally, just return the raw data safely
    if not DO_BASELINE_CORRECT:
        return profile, coeff, lift

    # 1. Define Baseline edge windows
    _left_pix = int(np.floor(nm_to_pixel(MIN_WAVELENGTH_NM)))
    _right_pix = int(np.ceil(nm_to_pixel(MAX_WAVELENGTH_NM)))
    left_window = (max(0, _left_pix), max(0, _left_pix) + 150)
    right_window = (max(750, min(1450, _right_pix - 150)), min(DETECTOR_X_MAX, _right_pix))

    # 2. Estimate and subtract the polynomial baseline
    coeff = estimate_polynomial_baseline_from_edge_windows(
        x_px,
        profile,
        left_window=left_window,
        right_window=right_window,
        degree=BASELINE_POLY_DEGREE,
    )
    corrected_profile = subtract_polynomial_baseline(x_px, profile, coeff)

    # 3. Enforce non-negative values if required
    if BASELINE_ENFORCE_NONNEGATIVE:
        min_after = float(np.nanmin(corrected_profile))
        if np.isfinite(min_after) and min_after < 0.0:
            lift = -min_after
            corrected_profile = corrected_profile + lift

    return corrected_profile, coeff, lift

def prompt_c3_tiff_path(exp_i=None, frame_i=None, exp_dir_name=None):
    """
    Ask for an experiment number and frame number, return the matching C3
    ProEM TIFF path. exp_dir_name names the experiment folder directly
    (falls back to the module-level EXPERIMENT_DIR_NAME, then to "C<exp_i>");
    that folder must still contain a ProEM/ subfolder with the .tif frames.
    """
    dir_name = exp_dir_name or EXPERIMENT_DIR_NAME or f"C{exp_i}"
    proem_dir = os.path.join(C3_BASE_DIR, dir_name, "ProEM")
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





#########################################################################
########################## --- Main flow --- ############################
#########################################################################


def extract_nt_from_frame(exp_i, frame_i, do_plot=False, exp_dir_name=None):
    tiff_path, exp_number, frame_number = prompt_c3_tiff_path(exp_i, frame_i,
                                                               exp_dir_name)
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

    # Helper function to dynamically slice arrays based on physical wavelength
    #window_nm is the half-width of the window around the line center to consider for fitting
    def get_roi(lam_center, window_nm):
        mask = (x_nm >= lam_center - window_nm) & (x_nm <= lam_center + window_nm)
        return x_px[mask], x_nm[mask], profile[mask]

    # Initialize return variables safely in case of failure
    n_e_cm3 = 0
    T_C1_eV = 0
    T_C2_eV = 0
    try:
        # --- 1. Fit Hα (Voigt) ---
        x_px_Ha, x_nm_Ha, prof_Ha = get_roi(lambda_Ha, 0.6)
        p0_Ha = [np.max(prof_Ha), x_px_Ha[np.argmax(prof_Ha)], max(3.0, 1.5 * inst_sigma_Ha), 3, np.min(prof_Ha)]
        bounds_Ha = ([0, x_px_Ha[0], inst_sigma_Ha, 0, 0], [np.inf, x_px_Ha[-1], 100, 100, np.max(prof_Ha)])
        popt_Ha, _ = curve_fit(voigt, x_px_Ha, prof_Ha, p0=p0_Ha, bounds=bounds_Ha)
        amp_Ha, cen_Ha, sig_Ha, gam_Ha, off_Ha = popt_Ha
        fit_Ha = voigt(x_px_Ha, amp_Ha, cen_Ha, sig_Ha, gam_Ha, off_Ha)

        # --- 2. Fit C1 (Gaussian) ---
        x_px_C1, x_nm_C1, prof_C1 = get_roi(lambda_C1, 0.2)
        p0_C1 = [np.max(prof_C1), x_px_C1[np.argmax(prof_C1)], max(3.0, 1.5 * inst_sigma_C1), np.min(prof_C1)]
        bounds_C1 = ([0, x_px_C1[0], inst_sigma_C1, 0], [np.inf, x_px_C1[-1], 100, np.max(prof_C1)])
        popt_C1, _ = curve_fit(gaussian, x_px_C1, prof_C1, p0=p0_C1, bounds=bounds_C1)
        amp_C1, cen_C1, sig_C1, off_C1 = popt_C1
        fit_C1 = gaussian(x_px_C1, amp_C1, cen_C1, sig_C1, off_C1)

        # --- 3. Fit C2 (Gaussian) ---
        x_px_C2, x_nm_C2, prof_C2 = get_roi(lambda_C2, 0.2)
        p0_C2 = [np.max(prof_C2), x_px_C2[np.argmax(prof_C2)], max(3.0, 1.5 * inst_sigma_C2), np.min(prof_C2)]
        bounds_C2 = ([0, x_px_C2[0], inst_sigma_C2, 0], [np.inf, x_px_C2[-1], 100, np.max(prof_C2)])
        popt_C2, _ = curve_fit(gaussian, x_px_C2, prof_C2, p0=p0_C2, bounds=bounds_C2)
        amp_C2, cen_C2, sig_C2, off_C2 = popt_C2
        fit_C2 = gaussian(x_px_C2, amp_C2, cen_C2, sig_C2, off_C2)

        # --- 4. Plasma Diagnostics ---
        
        # H-alpha Density Calculation
        gam_Ha_nm = width_px_to_nm(gam_Ha, cen_Ha)
        stark_fwhm_nm = 2 * gam_Ha_nm
        if stark_fwhm_nm > 0:
            n_e_cm3 = 10**17 * (stark_fwhm_nm / 1.098)**1.471
            print(f"Hα Electron Density (n_e):  {n_e_cm3:.2e} cm^-3")

        # C1 Temperature Calculation
        sig_C1_nm = width_px_to_nm(sig_C1, cen_C1)
        inst_sigma_C1_nm = width_px_to_nm(inst_sigma_C1, cen_C1)
        if sig_C1_nm > inst_sigma_C1_nm:
            sig_th_C1_nm = math.sqrt(sig_C1_nm**2 - inst_sigma_C1_nm**2)
            mc2_C_eV = 931.49e6 * mC
            T_C1_eV = mc2_C_eV * (sig_th_C1_nm / lambda_C1)**2
            print(f"C1 Temperature (T):         {T_C1_eV:.2f} eV")

        # C2 Temperature Calculation
        sig_C2_nm = width_px_to_nm(sig_C2, cen_C2)
        inst_sigma_C2_nm = width_px_to_nm(inst_sigma_C2, cen_C2)
        if sig_C2_nm > inst_sigma_C2_nm:
            sig_th_C2_nm = math.sqrt(sig_C2_nm**2 - inst_sigma_C2_nm**2)
            mc2_C_eV = 931.49e6 * mC
            # Fixed lambda_C1 to lambda_C2 here
            T_C2_eV = mc2_C_eV * (sig_th_C2_nm / lambda_C2)**2 
            print(f"C2 Temperature (T):         {T_C2_eV:.2f} eV")

    except RuntimeError as e:
        print(f"WARNING: Curve fitting failed for frame {frame_number}. Returning zeros.")
        return 0, 0, 0

    # --- Plotting the Fit (Only if do_plot is True) ---
    if do_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(x_nm, profile, label="Raw Profile", color='lightgray')
        
        plt.plot(x_nm_Ha, fit_Ha, 'r--', label="Hα Voigt Fit")
        plt.plot(x_nm_C1, fit_C1, 'g--', label="C1 Gaussian Fit")
        plt.plot(x_nm_C2, fit_C2, 'b--', label="C2 Gaussian Fit")
        
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity [a.u.]")
        plt.title(f"C{exp_number} Frame {frame_number} Fits")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Return n_e from Ha, T from C1, and T from C2
    return n_e_cm3, T_C1_eV, T_C2_eV


#########################################################################
###### --- Explicit-component Zeeman fit of the C II doublet --- ########
#########################################################################
#
# This replaces the Gaussian-quadrature ("RMS shift added in quadrature to
# the width") treatment used in validate_zeeman.py. Here the Zeeman pattern
# is built STRUCTURALLY: each line is the sum of its individual m-components,
# each displaced by its own shift, and the field B is a fitted parameter of
# that sum. Nothing is subtracted in quadrature except the instrumental
# width, which the existing pipeline already treats that way.
#
# GEOMETRY ASSUMPTION: the component tables below are the TRANSVERSE pattern
# (line of sight perpendicular to B), which shows pi AND sigma components.
# The viewing geometry of this shot is not recorded. Longitudinal viewing
# (along B) would show sigma only, a different set of shifts and strengths,
# and would return a different B from the same data. This is an ASSUMED
# geometry, not a measured one.

# Wavelength shift per unit of reduced shift per tesla, in ANGSTROM.
ZEEMAN_ANGSTROM_PER_UNIT_PER_TESLA = 0.2020

ZEEMAN_PATTERN = {
    # C1: 657.80482 nm, 2P3/2 - 2S1/2, six components
    "C1": {
        "lam_nm": lambda_C1,
        "shifts": np.array([-5.0 / 3, -1.0, -1.0 / 3, 1.0 / 3, 1.0, 5.0 / 3]),
        "strengths": np.array([1.0, 3.0, 2.0, 2.0, 3.0, 1.0]),
    },
    # C2: 658.28761 nm, 2P1/2 - 2S1/2, four components
    "C2": {
        "lam_nm": lambda_C2,
        "shifts": np.array([-4.0 / 3, -2.0 / 3, 2.0 / 3, 4.0 / 3]),
        "strengths": np.array([1.0, 1.0, 1.0, 1.0]),
    },
}
# Strengths normalised to sum 1 so that amp_line keeps exactly the meaning it
# has in the existing pipeline: the TOTAL area of the line. This is a pure
# reparametrisation of amp (amp_here = amp_raw * sum(strengths)); the
# branching-ratio information still lives entirely in amp_C1 vs amp_C2.
for _d in ZEEMAN_PATTERN.values():
    _d["weights"] = _d["strengths"] / _d["strengths"].sum()
# ZEEMAN_FIRST_FRAME, ZEEMAN_LAST_FRAME, B_MAX_TESLA, B_MIN_TESLA and
# ZEEMAN_ROI_HALFWIDTH_NM are set at the top of the file, above the
# calibration constants.


def zeeman_shifts_px(tag, B_tesla, center_px):
    """
    Component shifts in PIXELS for one line at field B.

    The shift is computed in nm from the reduced-shift table and then carried
    into pixels with width_nm_to_px anchored at THIS line's fitted centre -
    the same centred-differencing helper the width conversions use. It is not
    a multiplication by a dispersion constant, so it cannot drift away from
    pixel_to_nm the way the old dispersion_nm_per_px did.

    At B = 0 every shift_nm is exactly 0.0 and width_nm_to_px returns exactly
    0.0, so the pattern collapses onto the single centre - that is what makes
    the B = 0 reduction exact rather than approximate.
    """
    d = ZEEMAN_PATTERN[tag]
    lam_center_nm = float(pixel_to_nm(center_px))
    out = np.empty(d["shifts"].size, dtype=float)
    for i, s in enumerate(d["shifts"]):
        shift_nm = float(s) * ZEEMAN_ANGSTROM_PER_UNIT_PER_TESLA * float(B_tesla) / 10.0
        out[i] = width_nm_to_px(shift_nm, lam_center_nm)
    return out


def zeeman_line_model(x_px, tag, amp, center_px, sigma_px, continuum, B_tesla):
    """continuum + amp * sum_i weight_i * gaussian(x, center + shift_i(B), sigma)."""
    d = ZEEMAN_PATTERN[tag]
    shifts = zeeman_shifts_px(tag, B_tesla, center_px)
    out = np.zeros_like(np.asarray(x_px, dtype=float))
    for w, s in zip(d["weights"], shifts):
        out += w * gaussian(x_px, amp, center_px + s, sigma_px, 0.0)
    return out + continuum


def c2_center_from_c1(center_C1_px):
    """
    C2's centre is NOT a free parameter. It is pinned to C1 by the NIST doublet
    separation, converted to pixels with width_nm_to_px anchored at C1's own
    fitted centre. Letting both centres float independently reopens the
    centre/width degeneracy this analysis has been fighting all along.
    """
    lam_center_nm = float(pixel_to_nm(center_C1_px))
    return center_C1_px + width_nm_to_px(lambda_C2 - lambda_C1, lam_center_nm)


def _robust_sigma(v):
    """1.4826 * MAD - a noise scale that a few outliers cannot inflate."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 1.0
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) or 1.0


def prepare_frame(exp_i, frame_i, exp_dir_name=None):
    """
    Load one frame, baseline-correct it, cut the two C II ROIs, and run the
    EXISTING single-Gaussian fits on each. Those existing fits serve three
    purposes: they seed the joint fit, they provide the "old" temperatures for
    the side-by-side comparison, and they are the reference the B = 0 reduction
    check is verified against.

    Returns None if the frame cannot be loaded or the existing fits fail, so a
    bad frame drops out of the analysis instead of poisoning the global fit.
    """
    try:
        tiff_path, _, _ = prompt_c3_tiff_path(exp_i, frame_i, exp_dir_name)
    except TypeError:
        return None

    _min_pix = max(0, int(np.floor(nm_to_pixel(MIN_WAVELENGTH_NM))))
    _max_pix = min(DETECTOR_X_MAX, int(np.ceil(nm_to_pixel(MAX_WAVELENGTH_NM))) + 1)
    x_px, profile = load_full_profile(tiff_path, Y_RANGE, x_min=_min_pix, x_max=_max_pix)
    profile = np.asarray(profile, dtype=float)
    x_nm = pixel_to_nm(x_px)
    profile, _, _ = apply_baseline_correction(x_px, profile)

    def get_roi(lam_center, window_nm):
        mask = (x_nm >= lam_center - window_nm) & (x_nm <= lam_center + window_nm)
        return x_px[mask], profile[mask]

    x1, y1 = get_roi(lambda_C1, ZEEMAN_ROI_HALFWIDTH_NM)
    x2, y2 = get_roi(lambda_C2, ZEEMAN_ROI_HALFWIDTH_NM)
    if x1.size < 8 or x2.size < 8:
        return None

    # Noise scale from the line-free edge windows of the same baseline-corrected
    # profile. An absolute noise estimate is what makes chi2r and delta-chi2
    # interpretable; without one, chi2r is 1 by construction and says nothing.
    edge = (x_px <= x_px[0] + 150) | (x_px >= x_px[-1] - 150)
    noise = _robust_sigma(profile[edge])

    # --- existing pipeline fits, verbatim in form ---
    # pcov is kept here only so the OLD temperatures can carry error bars too.
    # curve_fit is called unweighted, exactly as the pipeline does it, so its
    # pcov is already scaled by the fit's own residual variance; that is the
    # same absolute_sigma=False convention used for the joint fit below.
    try:
        p0_C1 = [np.max(y1), x1[np.argmax(y1)], max(3.0, 1.5 * inst_sigma_C1), np.min(y1)]
        b_C1 = ([0, x1[0], inst_sigma_C1, 0], [np.inf, x1[-1], 100, np.max(y1)])
        popt_C1, pcov_C1 = curve_fit(gaussian, x1, y1, p0=p0_C1, bounds=b_C1)
        p0_C2 = [np.max(y2), x2[np.argmax(y2)], max(3.0, 1.5 * inst_sigma_C2), np.min(y2)]
        b_C2 = ([0, x2[0], inst_sigma_C2, 0], [np.inf, x2[-1], 100, np.max(y2)])
        popt_C2, pcov_C2 = curve_fit(gaussian, x2, y2, p0=p0_C2, bounds=b_C2)
    except (RuntimeError, ValueError):
        return None

    def _perr(pcov):
        d = np.diag(np.asarray(pcov, dtype=float))
        return np.sqrt(np.clip(np.nan_to_num(d, nan=0.0, posinf=0.0), 0.0, np.inf))

    return {
        "frame": frame_i,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "noise": noise,
        "old_C1": popt_C1, "old_C2": popt_C2,
        "old_C1_err": _perr(pcov_C1), "old_C2_err": _perr(pcov_C2),
    }


def temperature_eV(sigma_px, center_px, lam_nm, inst_sigma_px):
    """T from a fitted Gaussian sigma, exactly as the existing pipeline does it."""
    sig_nm = width_px_to_nm(sigma_px, center_px)
    inst_nm = width_px_to_nm(inst_sigma_px, center_px)
    if sig_nm <= inst_nm:
        return 0.0
    sig_th_nm = math.sqrt(sig_nm ** 2 - inst_nm ** 2)
    return (931.49e6 * mC) * (sig_th_nm / lam_nm) ** 2


def temperature_eV_bounds(sigma_px, sigma_err_px, center_px, lam_nm,
                          inst_sigma_px):
    """
    (T, minus, plus): T and its ASYMMETRIC 1-sigma error bars, from the fitted
    width's own stderr pushed through the same T formula.

    Propagated by re-evaluating T at sigma -/+ its stderr rather than by a
    derivative. T is quadratic in sigma and the instrumental width is removed
    in quadrature underneath, so the mapping is distinctly non-linear near the
    instrumental floor: a width one sigma below the fit can sit AT the floor,
    where T pins to 0 and the lower bar is short while the upper one is not.
    A symmetric +/- bar would misstate exactly the frames where it matters.
    """
    T = temperature_eV(sigma_px, center_px, lam_nm, inst_sigma_px)
    e = float(sigma_err_px)
    if not np.isfinite(e) or e <= 0.0:
        return T, 0.0, 0.0
    T_lo = temperature_eV(max(sigma_px - e, 0.0), center_px, lam_nm, inst_sigma_px)
    T_hi = temperature_eV(sigma_px + e, center_px, lam_nm, inst_sigma_px)
    return T, max(T - T_lo, 0.0), max(T_hi - T, 0.0)


# --- parameter packing ------------------------------------------------------
# Per-frame block, in order: cen_C1, sigma_C1, sigma_C2, amp_C1, amp_C2,
# continuum_C1, continuum_C2.  B is separate: per-frame when fitted per frame,
# a single leading entry when shared globally.
#
# share_sigma=True drops sigma_C2 and gives both lines ONE width, 6 parameters
# per frame instead of 7. The widths are shared in PIXELS. Strictly the shared
# quantity should be the physical width: thermal broadening is a fixed
# Delta_lambda / lambda, and one pixel is not quite the same number of nm at
# 657.8 as at 658.3. Both corrections are below 0.2% over the 105 px between
# the lines - the dispersion changes 0.14% and lambda changes 0.07% - which is
# two orders of magnitude below the width uncertainties here, and sharing the
# pixel width keeps the same convention as inst_sigma_C1 == inst_sigma_C2 in
# the existing pipeline.
FRAME_PNAMES = ["center_C1", "sigma_C1", "sigma_C2", "amp_C1", "amp_C2",
                "cont_C1", "cont_C2"]
FRAME_PNAMES_SHARED = ["center_C1", "sigma", "amp_C1", "amp_C2",
                       "cont_C1", "cont_C2"]


def frame_pnames(share_sigma):
    return FRAME_PNAMES_SHARED if share_sigma else FRAME_PNAMES


def n_frame_params(share_sigma):
    return len(frame_pnames(share_sigma))


def _frame_seed_and_bounds(fd, share_sigma=False):
    a1, c1, s1, k1 = fd["old_C1"]
    a2, c2, s2, k2 = fd["old_C2"]
    if share_sigma:
        # seed the shared width at the mean of the two independent seeds
        p0 = np.array([c1, 0.5 * (s1 + s2), a1, a2, k1, k2], dtype=float)
        lo = np.array([fd["x1"][0], min(inst_sigma_C1, inst_sigma_C2),
                       0.0, 0.0, -np.inf, -np.inf])
        hi = np.array([fd["x1"][-1], 100.0, np.inf, np.inf, np.inf, np.inf])
    else:
        p0 = np.array([c1, s1, s2, a1, a2, k1, k2], dtype=float)
        lo = np.array([fd["x1"][0], inst_sigma_C1, inst_sigma_C2, 0.0, 0.0,
                       -np.inf, -np.inf])
        hi = np.array([fd["x1"][-1], 100.0, 100.0, np.inf, np.inf, np.inf, np.inf])
    return np.clip(p0, lo + 1e-9, hi - 1e-9), lo, hi


def _frame_residual(fp, fd, B, share_sigma=False):
    if share_sigma:
        cen1, sig, a1, a2, k1, k2 = fp
        sig1 = sig2 = sig
    else:
        cen1, sig1, sig2, a1, a2, k1, k2 = fp
    cen2 = c2_center_from_c1(cen1)
    m1 = zeeman_line_model(fd["x1"], "C1", a1, cen1, sig1, k1, B)
    m2 = zeeman_line_model(fd["x2"], "C2", a2, cen2, sig2, k2, B)
    return np.concatenate([m1 - fd["y1"], m2 - fd["y2"]]) / fd["noise"]


def _covariance(res, chi2r):
    """Parameter covariance from the least_squares Jacobian, scaled by chi2r."""
    J = res.jac
    try:
        J = J.toarray()
    except AttributeError:
        J = np.asarray(J)
    # pinv, not inv: on the late frames several widths rail against their
    # bounds, which leaves exactly-zero Jacobian columns and makes J^T J
    # singular. inv returns nan for the WHOLE matrix then, including the B
    # entry, which is otherwise perfectly well determined.
    try:
        cov = np.linalg.pinv(J.T @ J, rcond=1e-12) * max(chi2r, 0.0)
    except np.linalg.LinAlgError:
        cov = np.full((J.shape[1], J.shape[1]), np.nan)
    return cov





def fit_frame(fd, fit_B=True, B_fixed=0.0, share_sigma=False):
    """
    Joint C1+C2 fit of one frame.

    fit_B=False holds B at B_fixed; B_fixed = 0 is the nested no-Zeeman model,
    and other values give the profile chi2(B) used for the global B interval.
    share_sigma=True gives the two lines a single common width.
    """
    p0f, lof, hif = _frame_seed_and_bounds(fd, share_sigma)
    n_data = fd["x1"].size + fd["x2"].size

    if fit_B:
        lo = np.concatenate([[B_MIN_TESLA], lof])
        hi = np.concatenate([[B_MAX_TESLA], hif])
        fun = lambda p: _frame_residual(p[1:], fd, p[0], share_sigma)
        res = None
        for b0 in B_SEEDS:
            try:
                r = least_squares(fun, np.concatenate([[b0], p0f]),
                                  bounds=(lo, hi), method="trf",
                                  x_scale="jac", max_nfev=20000)
            except (ValueError, RuntimeError):
                continue
            if res is None or r.cost < res.cost:
                res = r
        if res is None:
            raise RuntimeError("all B seeds failed")
    else:
        p0, lo, hi = p0f, lof, hif
        fun = lambda p: _frame_residual(p, fd, B_fixed, share_sigma)
        res = least_squares(fun, p0, bounds=(lo, hi), method="trf",
                            x_scale="jac", max_nfev=20000)
    chi2 = float(2.0 * res.cost)
    dof = max(n_data - res.x.size, 1)
    chi2r = chi2 / dof
    cov = _covariance(res, chi2r)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    # per-line chi2r, to see whether a shared B fits one line at the other's cost
    r = fun(res.x)
    n1 = fd["x1"].size
    out = {
        "ok": bool(res.success), "chi2": chi2, "chi2r": chi2r, "dof": dof,
        "chi2r_C1": float(np.sum(r[:n1] ** 2) / max(n1 - res.x.size / 2.0, 1.0)),
        "chi2r_C2": float(np.sum(r[n1:] ** 2) / max(r.size - n1 - res.x.size / 2.0, 1.0)),
        "n_data": n_data,
    }
    if fit_B:
        out["B"] = float(res.x[0])
        out["B_err"] = float(err[0])
        out["B_at_bound"] = bool(res.x[0] <= lo[0] + 1e-9 or res.x[0] >= hi[0] - 1e-9)
        fp, fe = res.x[1:], err[1:]
    else:
        out["B"] = float(B_fixed)
        out["B_err"] = 0.0
        out["B_at_bound"] = False
        fp, fe = res.x, err
    for name, v, e in zip(frame_pnames(share_sigma), fp, fe):
        out[name] = float(v)
        out[name + "_err"] = float(e)
    if share_sigma:
        # Expose the shared width under both names so every consumer downstream
        # - the T calculation, the tables, the plots - keeps working unchanged.
        out["sigma_C1"] = out["sigma_C2"] = out["sigma"]
        out["sigma_C1_err"] = out["sigma_C2_err"] = out["sigma_err"]
    out["center_C2"] = float(c2_center_from_c1(out["center_C1"]))
    out["share_sigma"] = bool(share_sigma)
    return out


def fit_global_B(frames, share_sigma=False):
    """
    All frames at once with ONE shared B, everything else free per frame.
    Parameter vector: [B, (6 or 7 params) x n_frames].
    """
    k = n_frame_params(share_sigma)
    seeds, los, his = zip(*[_frame_seed_and_bounds(fd, share_sigma)
                            for fd in frames])
    p0 = np.concatenate([[0.05]] + list(seeds))
    lo = np.concatenate([[B_MIN_TESLA]] + list(los))
    hi = np.concatenate([[B_MAX_TESLA]] + list(his))

    def fun(p):
        B = p[0]
        return np.concatenate([_frame_residual(p[1 + k * i: 1 + k * (i + 1)],
                                               fd, B, share_sigma)
                               for i, fd in enumerate(frames)])

    # Only B couples the frames, so the Jacobian is arrow-shaped. Declaring that
    # keeps a 300+ parameter fit tractable instead of dense-differencing it.
    rows = [fd["x1"].size + fd["x2"].size for fd in frames]
    spars = lil_matrix((int(np.sum(rows)), p0.size), dtype=int)
    spars[:, 0] = 1
    r0 = 0
    for i, m in enumerate(rows):
        spars[r0:r0 + m, 1 + k * i: 1 + k * (i + 1)] = 1
        r0 += m

    res = None
    for b0 in B_SEEDS:
        p0[0] = b0
        r = least_squares(fun, p0, bounds=(lo, hi), method="trf", x_scale="jac",
                          jac_sparsity=spars, max_nfev=40000)
        print(f"    seed B0 = {b0:.2f} T  ->  B = {r.x[0]:.4f} T, "
              f"chi2 = {2.0 * r.cost:.1f}, status {r.status}")
        if res is None or r.cost < res.cost:
            res = r
    chi2 = float(2.0 * res.cost)
    dof = max(int(np.sum(rows)) - res.x.size, 1)
    chi2r = chi2 / dof
    cov = _covariance(res, chi2r)
    err_all = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    B = float(res.x[0])
    B_err = float(np.sqrt(max(cov[0, 0], 0.0)))

    per = []
    for i, fd in enumerate(frames):
        fp = res.x[1 + k * i: 1 + k * (i + 1)]
        fe = err_all[1 + k * i: 1 + k * (i + 1)]
        d = {"frame": fd["frame"]}
        for name, v, e in zip(frame_pnames(share_sigma), fp, fe):
            d[name] = float(v)
            d[name + "_err"] = float(e)
        if share_sigma:
            d["sigma_C1"] = d["sigma_C2"] = d["sigma"]
            d["sigma_C1_err"] = d["sigma_C2_err"] = d["sigma_err"]
        d["center_C2"] = float(c2_center_from_c1(d["center_C1"]))
        rr = _frame_residual(fp, fd, B, share_sigma)
        d["chi2"] = float(np.sum(rr ** 2))
        d["chi2r"] = d["chi2"] / max(rr.size - k, 1)
        n1 = fd["x1"].size
        d["chi2r_C1"] = float(np.sum(rr[:n1] ** 2) / max(n1 - k / 2.0, 1.0))
        d["chi2r_C2"] = float(np.sum(rr[n1:] ** 2) / max(rr.size - n1 - k / 2.0, 1.0))
        per.append(d)

    return {"ok": bool(res.success), "status": int(res.status), "nfev": int(res.nfev),
            "B": B, "B_err": B_err, "chi2": chi2, "chi2r": chi2r, "dof": dof,
            "at_bound": bool(B <= B_MIN_TESLA + 1e-9 or B >= B_MAX_TESLA - 1e-9),
            "per_frame": per, "n_par": int(res.x.size),
            "share_sigma": bool(share_sigma)}


def plot_frame_fit(fd, models, exp_i, out_dir=C3_BASE_DIR, suffix=""):
    """
    One frame, both ROIs, on the wavelength axis: the data, each fitted model,
    and - for the model asked to show them - the individual Zeeman components
    that sum to it. This is the picture that shows what B is actually doing to
    the profile, which no chi2 table can.

    models: list of dicts with keys label, params (a fit_frame result), B,
            color, ls, and optionally components=True.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, tag, xk, yk, cenk, sigk, ampk, contk, lam in (
            (axes[0], "C1", "x1", "y1", "center_C1", "sigma_C1", "amp_C1",
             "cont_C1", lambda_C1),
            (axes[1], "C2", "x2", "y2", "center_C2", "sigma_C2", "amp_C2",
             "cont_C2", lambda_C2)):
        x, y = fd[xk], fd[yk]
        xf = np.linspace(x[0], x[-1], 1200)
        ax.plot(pixel_to_nm(x), y, "o", ms=3.5, color="0.45",
                label="data (baseline-corrected)")
        for m in models:
            p, B = m["params"], m["B"]
            ax.plot(pixel_to_nm(xf),
                    zeeman_line_model(xf, tag, p[ampk], p[cenk], p[sigk],
                                      p[contk], B),
                    m.get("ls", "-"), color=m["color"], lw=m.get("lw", 1.8),
                    label=f"{m['label']}  (B = {B:.3f} T)")
            if not m.get("components"):
                continue
            shifts = zeeman_shifts_px(tag, B, p[cenk])
            for j, (w, s) in enumerate(zip(ZEEMAN_PATTERN[tag]["weights"], shifts)):
                ax.plot(pixel_to_nm(xf),
                        p[contk] + w * gaussian(xf, p[ampk], p[cenk] + s,
                                                p[sigk], 0.0),
                        "-", color=m["color"], lw=0.9, alpha=0.45,
                        label="Zeeman components" if j == 0 else None)
            ax.axvline(pixel_to_nm(p[cenk]), color=m["color"], lw=0.8,
                       ls=":", alpha=0.7)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity [a.u.]")
        n_comp = ZEEMAN_PATTERN[tag]["shifts"].size
        ax.set_title(f"{tag}  {lam} nm  ({n_comp} components)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"C{exp_i} frame {fd['frame']}: explicit Zeeman component fit "
                 f"(TRANSVERSE pattern assumed)", fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(out_dir,
                           f"zeeman_frame{fd['frame']}_C{exp_i}{suffix}.png")
    plt.savefig(out_png, dpi=130)
    print(f"  frame {fd['frame']} fit figure written to {out_png}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def check_B0_reduction(frames):
    """
    SANITY CHECK, run before any real fitting: at B = 0 the component model must
    be the existing single-Gaussian-per-line model, bit for bit (up to the
    normalisation of the weight sum). Both the model values and the residual
    vector are compared against the existing fit's.
    """
    print("\n" + "=" * 78)
    print("SANITY CHECK: B = 0 REDUCTION TO THE EXISTING NON-ZEEMAN MODEL")
    print("=" * 78)
    print("  At B = 0 every component shift is identically 0.0 px, so the")
    print("  component sum must collapse onto one Gaussian per line. Comparing")
    print("  the Zeeman model at B = 0, evaluated at the EXISTING fit's own")
    print("  parameters, against the existing gaussian() call:")
    worst_model = 0.0
    worst_resid = 0.0
    for fd in frames:
        for tag, xk, yk, pk in (("C1", "x1", "y1", "old_C1"),
                                ("C2", "x2", "y2", "old_C2")):
            amp, cen, sig, off = fd[pk]
            sh = zeeman_shifts_px(tag, 0.0, cen)
            if np.any(sh != 0.0):
                print(f"  FAIL: non-zero shifts at B=0 on {tag}: {sh}")
            m_new = zeeman_line_model(fd[xk], tag, amp, cen, sig, off, 0.0)
            m_old = gaussian(fd[xk], amp, cen, sig, off)
            scale = float(np.max(np.abs(m_old)))
            worst_model = max(worst_model, float(np.max(np.abs(m_new - m_old))) / scale)
            worst_resid = max(worst_resid,
                              float(np.max(np.abs((m_new - fd[yk]) - (m_old - fd[yk])))) / scale)
    print(f"\n  max |model_Zeeman(B=0) - model_existing| / peak = {worst_model:.3e}")
    print(f"  max |resid_Zeeman(B=0) - resid_existing| / peak = {worst_resid:.3e}")
    print(f"  machine epsilon (float64)                       = {np.finfo(float).eps:.3e}")
    ok = worst_model < 1e-12 and worst_resid < 1e-12
    print("  -> " + ("PASS: identical to floating-point round-off."
                     if ok else "FAIL: the B=0 model is NOT the existing model."))
    print("  (The residual difference is the difference of the same two models")
    print("   against the same data, so it equals the model difference exactly.")
    print("   The floor is the weight sum 1/10+3/10+2/10+2/10+3/10+1/10, which")
    print("   is 1 only to round-off in binary floating point.)")
    return ok


def run_zeeman_analysis(exp_i=559, first_frame=ZEEMAN_FIRST_FRAME,
                        last_frame=ZEEMAN_LAST_FRAME, do_plot=True,
                        plot_frame=13, share_sigma=False, frames=None,
                        exp_dir_name=None):
    mode = ("SHARED sigma (one width for both lines)" if share_sigma
            else "INDEPENDENT sigma per line")
    print("=" * 78)
    print("C II DOUBLET: EXPLICIT ZEEMAN COMPONENT FIT (JOINT C1 + C2)")
    print(f"WIDTH MODEL: {mode}")
    print("=" * 78)
    print("  GEOMETRY: TRANSVERSE pattern assumed (pi + sigma, Delta_m = 0,+/-1).")
    print("  The viewing geometry relative to B is NOT measured in this")
    print("  experiment. Longitudinal viewing would show sigma components only,")
    print("  with different shifts and strengths, and the same data would then")
    print("  return a different B. Every field below is conditional on the")
    print("  transverse assumption.")
    print(f"\n  scale: shift_nm = shift_reduced * "
          f"{ZEEMAN_ANGSTROM_PER_UNIT_PER_TESLA} A/T * B, then to px via")
    print("         width_nm_to_px anchored at each line's fitted centre.")
    for tag in ("C1", "C2"):
        d = ZEEMAN_PATTERN[tag]
        px1T = zeeman_shifts_px(tag, 1.0, nm_to_pixel(d["lam_nm"]))
        print(f"\n  {tag} ({d['lam_nm']} nm), {d['shifts'].size} components")
        print("    " + "  ".join(f"{s:+.4f}" for s in d["shifts"]) + "   (reduced shift)")
        print("    " + "  ".join(f"{s:+.4f}" for s in d["strengths"]) + "   (strength)")
        print("    " + "  ".join(f"{s:+.4f}" for s in px1T) + "   (px at B = 1 T)")
    print(f"\n  doublet separation {lambda_C2 - lambda_C1:.5f} nm = "
          f"{width_nm_to_px(lambda_C2 - lambda_C1, lambda_C1):.3f} px "
          f"(C2 centre is pinned to this, not free)")
    print(f"  instrumental sigma {inst_sigma_C1:.3f} px "
          f"({inst_fwhm_Ha} nm FWHM assumed, unchanged from the pipeline)")

    # ---- load frames ------------------------------------------------------
    # frames can be passed in so the two width models are compared on byte-
    # identical data and byte-identical seed fits, not on two separate loads.
    if frames is None:
        print(f"\nLoading C{exp_i} frames {first_frame}-{last_frame} ...")
        frames = []
        for f in range(first_frame, last_frame + 1):
            fd = prepare_frame(exp_i, f, exp_dir_name)
            if fd is None:
                print(f"  frame {f}: skipped (load or seed fit failed)")
            else:
                frames.append(fd)
    print(f"  {len(frames)} frames usable")
    if not frames:
        print("  nothing to fit.")
        return None

    # ---- sanity check -----------------------------------------------------
    check_B0_reduction(frames)

    # ---- (1) per-frame fits ----------------------------------------------
    print("\n" + "=" * 78)
    print("1. PER-FRAME JOINT FIT   (B free, one B per frame)")
    print("=" * 78)
    print("  B, center_C1, sigma_C1, sigma_C2, amp_C1, amp_C2 and one constant")
    print("  continuum per line, fitted to both ROIs in a single residual")
    print("  vector. center_C2 is pinned to center_C1 + the NIST separation.")
    print("  Residuals are normalised by a per-frame noise estimate taken from")
    print("  the line-free edge windows (robust MAD), so chi2r is an absolute")
    print("  number and delta-chi2 is interpretable. stderr is the Jacobian")
    print("  covariance scaled by chi2r.")
    print(f"\n  {'frame':>5} | {'B [T]':>7} {'+/-':>7} {'bnd':>4} | "
          f"{'chi2r':>7} {'chi2r0':>7} {'dchi2':>9} | "
          f"{'sigC1':>6} {'sigC2':>6} | {'T1_new':>7} {'T1_old':>7} "
          f"{'T2_new':>7} {'T2_old':>7}")
    rows = []
    for fd in frames:
        rb = fit_frame(fd, fit_B=True, share_sigma=share_sigma)
        r0 = fit_frame(fd, fit_B=False, share_sigma=share_sigma)
        a1o, c1o, s1o, _ = fd["old_C1"]
        a2o, c2o, s2o, _ = fd["old_C2"]
        s1o_e, s2o_e = fd["old_C1_err"][2], fd["old_C2_err"][2]
        T1o, T1o_m, T1o_p = temperature_eV_bounds(s1o, s1o_e, c1o, lambda_C1,
                                                  inst_sigma_C1)
        T2o, T2o_m, T2o_p = temperature_eV_bounds(s2o, s2o_e, c2o, lambda_C2,
                                                  inst_sigma_C2)
        rec = {
            "frame": fd["frame"], "fit": rb, "fit0": r0,
            "dchi2": r0["chi2"] - rb["chi2"],
            "T1_new": temperature_eV(rb["sigma_C1"], rb["center_C1"], lambda_C1, inst_sigma_C1),
            "T2_new": temperature_eV(rb["sigma_C2"], rb["center_C2"], lambda_C2, inst_sigma_C2),
            "T1_old": T1o, "T2_old": T2o,
            "T1_old_m": T1o_m, "T1_old_p": T1o_p,
            "T2_old_m": T2o_m, "T2_old_p": T2o_p,
            "sig1_old": s1o, "sig2_old": s2o,
            "sig1_old_err": s1o_e, "sig2_old_err": s2o_e,
        }
        rows.append(rec)
        print(f"  {rec['frame']:5d} | {rb['B']:7.3f} {rb['B_err']:7.3f} "
              f"{('*' if rb['B_at_bound'] else ''):>4} | "
              f"{rb['chi2r']:7.2f} {r0['chi2r']:7.2f} {rec['dchi2']:9.2f} | "
              f"{rb['sigma_C1']:6.2f} {rb['sigma_C2']:6.2f} | "
              f"{rec['T1_new']:7.1f} {rec['T1_old']:7.1f} "
              f"{rec['T2_new']:7.1f} {rec['T2_old']:7.1f}")
    print("  ('*' = B railed against a bound; its stderr is not a meaningful")
    print("   1-sigma interval there. dchi2 = chi2(B=0) - chi2(B free), 1 dof.)")
    print("  Enormous stderr values on the frames that settle near B = 0 are")
    print("  real and structural, not a bug: the pattern is symmetric, so the")
    print("  model responds to B only at SECOND order and d(model)/dB is")
    print("  exactly 0 at B = 0. The linear covariance therefore diverges as")
    print("  B -> 0. Those frames mean 'B unconstrained', not 'B = 0 +/- 3000 T'.")

    Ba = np.array([r["fit"]["B"] for r in rows])
    Be = np.array([r["fit"]["B_err"] for r in rows])
    dch = np.array([r["dchi2"] for r in rows])
    n_bound = sum(r["fit"]["B_at_bound"] for r in rows)
    print(f"\n  B per frame: median {np.median(Ba):.3f} T, mean {np.mean(Ba):.3f} T, "
          f"range {np.min(Ba):.3f}-{np.max(Ba):.3f} T")
    print(f"  scatter frame-to-frame (std) {np.std(Ba):.3f} T; "
          f"median stderr {np.nanmedian(Be):.3f} T")
    print(f"  {n_bound}/{len(rows)} frames rail B against a bound")
    print(f"  delta-chi2 for adding B: median {np.median(dch):.2f}, "
          f"range {np.min(dch):.2f}-{np.max(dch):.2f}   (1 dof: 3.84 = 95%)")
    print(f"  frames with dchi2 > 3.84: "
          f"{int(np.sum(dch > 3.84))}/{len(rows)}")

    # ---- (2) global shared-B fit -----------------------------------------
    print("\n" + "=" * 78)
    print("2. GLOBAL FIT   (ONE B shared across every frame)")
    print("=" * 78)
    print("  Same model, all frames in one residual vector, a single B for the")
    print("  whole discharge; center_C1, both sigmas, both amps and both")
    print("  continua stay free per frame.")
    g = fit_global_B(frames, share_sigma=share_sigma)
    chi2_0_sum = float(np.sum([r["fit0"]["chi2"] for r in rows]))
    dof_0 = int(np.sum([r["fit0"]["dof"] for r in rows]))
    print(f"\n  converged: {g['ok']}  (least_squares status {g['status']}, "
          f"{g['nfev']} function evaluations, {g['n_par']} free parameters)")
    print(f"  B_global = {g['B']:.4f} +/- {g['B_err']:.4f} T"
          + ("   [AT A BOUND - stderr not a real interval]" if g["at_bound"] else ""))
    print(f"  chi2 = {g['chi2']:.1f} / {g['dof']} dof  ->  chi2r = {g['chi2r']:.3f}")
    print(f"  B = 0 nested model (B fixed 0, frames decouple so this is the sum")
    print(f"  of the per-frame B=0 fits): chi2 = {chi2_0_sum:.1f} / {dof_0} dof"
          f"  ->  chi2r = {chi2_0_sum / max(dof_0, 1):.3f}")
    print(f"  delta-chi2 for one shared B = {chi2_0_sum - g['chi2']:.2f}  (1 dof)")

    # Profile chi2(B): B held fixed on a grid, everything else refitted. With B
    # fixed the frames decouple, so this is exact and cheap. It is the honest
    # interval for the global B - the Jacobian stderr is not, because
    # d(model)/dB -> 0 as B -> 0 and the linear covariance diverges there.
    print("\n  profile chi2 with B held fixed (everything else refitted):")

    def _profile(bs):
        return np.array([float(np.sum([fit_frame(fd, fit_B=False, B_fixed=b,
                                                 share_sigma=share_sigma)["chi2"]
                                       for fd in frames])) for b in bs])

    coarse = np.concatenate([[B_MIN_TESLA],
                             np.linspace(B_MIN_TESLA + 0.1, B_MAX_TESLA, 20)])
    p_coarse = _profile(coarse)
    b_star = coarse[int(np.argmin(p_coarse))]
    # Refine around the coarse minimum - the well turns out to be far narrower
    # than one coarse step, so the coarse grid alone cannot resolve the interval.
    fine = np.round(np.arange(max(B_MIN_TESLA, b_star - 0.12),
                              min(B_MAX_TESLA, b_star + 0.12) + 1e-9, 0.01), 4)
    p_fine = _profile(fine)
    grid = np.concatenate([coarse, fine])
    prof = np.concatenate([p_coarse, p_fine])
    order = np.argsort(grid)
    grid, prof = grid[order], prof[order]
    i_min = int(np.argmin(prof))
    print(f"    {'B [T]':>7} {'chi2':>12} {'delta-chi2':>12}")
    for Bv, cv in zip(coarse, p_coarse):
        print(f"    {Bv:7.3f} {cv:12.1f} {cv - prof[i_min]:12.1f}")
    print(f"    --- refined around the minimum ---")
    for Bv, cv in zip(fine, p_fine):
        mark = "  <- minimum" if cv == prof[i_min] else ""
        print(f"    {Bv:7.3f} {cv:12.1f} {cv - prof[i_min]:12.1f}{mark}")
    dchi_prof = prof - prof[i_min]
    thresh = max(g["chi2r"], 0.0)   # delta-chi2 = 1, scaled by chi2r
    inside = grid[dchi_prof <= thresh]
    print(f"\n    profile minimum at B = {grid[i_min]:.3f} T")
    if inside.size:
        print(f"    delta-chi2 <= 1 (scaled by chi2r = {g['chi2r']:.3f}) spans "
              f"B = {inside.min():.3f} to {inside.max():.3f} T")
    # The well is narrower than the fine grid step, so read the width off the
    # local curvature instead of off the grid.
    if 0 < i_min < grid.size - 1:
        h1, h2 = grid[i_min] - grid[i_min - 1], grid[i_min + 1] - grid[i_min]
        if abs(h1 - h2) < 1e-9 and h1 > 0:
            curv = (prof[i_min - 1] + prof[i_min + 1] - 2 * prof[i_min]) / h1 ** 2
            if curv > 0:
                print(f"    local curvature gives 1-sigma = "
                      f"{np.sqrt(2.0 * max(g['chi2r'], 0.0) / curv):.4f} T")
    print("    That interval is the STATISTICAL error only. It is far smaller")
    print("    than the systematic spread between the per-frame values, which")
    print("    is the honest measure of how well this data pins a field.")

    # Beyond about frame 26 the C II lines are weak enough that the widths rail
    # against their bounds in the OLD pipeline too. Those frames carry no line
    # information but do carry residual, so repeat the shared-B fit on the
    # window the pipeline itself trusts for T and see whether B moves.
    sub = [fd for fd in frames if 8 <= fd["frame"] <= 22]
    if len(sub) >= 2:
        print("\n  same global fit restricted to frames 8-22 (where the pipeline")
        print("  itself trusts the lines enough to quote a temperature):")
        g2 = fit_global_B(sub, share_sigma=share_sigma)
        chi2_0_sub = float(np.sum([r["fit0"]["chi2"] for r in rows
                                   if 8 <= r["frame"] <= 22]))
        print(f"    B_global(8-22) = {g2['B']:.4f} +/- {g2['B_err']:.4f} T, "
              f"chi2r = {g2['chi2r']:.3f}, converged {g2['ok']}")
        print(f"    delta-chi2 vs B = 0 on the same frames: "
              f"{chi2_0_sub - g2['chi2']:.1f}")

    print(f"\n  per-line chi2r under the shared B (is one line paying for the other?)")
    gc1 = np.array([d["chi2r_C1"] for d in g["per_frame"]])
    gc2 = np.array([d["chi2r_C2"] for d in g["per_frame"]])
    pc1 = np.array([r["fit"]["chi2r_C1"] for r in rows])
    pc2 = np.array([r["fit"]["chi2r_C2"] for r in rows])
    zc1 = np.array([r["fit0"]["chi2r_C1"] for r in rows])
    zc2 = np.array([r["fit0"]["chi2r_C2"] for r in rows])
    print(f"    {'':<22} {'C1':>10} {'C2':>10}")
    print(f"    {'global shared B':<22} {np.median(gc1):10.2f} {np.median(gc2):10.2f}")
    print(f"    {'per-frame free B':<22} {np.median(pc1):10.2f} {np.median(pc2):10.2f}")
    print(f"    {'B = 0':<22} {np.median(zc1):10.2f} {np.median(zc2):10.2f}")
    print("    (median over frames of the per-line reduced chi2)")

    # ---- (3) temperatures under the global B ------------------------------
    print("\n" + "=" * 78)
    print("3. WIDTHS AND TEMPERATURES UNDER THE GLOBAL SHARED B")
    print("=" * 78)
    print("  sigma is the fitted physical width with the Zeeman splitting")
    print("  carried structurally by the component sum, NOT removed in")
    print("  quadrature afterwards. T still has the instrumental sigma removed")
    print("  in quadrature, exactly as the existing pipeline does.")
    print(f"\n  {'frame':>5} | {'sigC1':>6} {'sigC1_old':>9} {'sigC2':>6} "
          f"{'sigC2_old':>9} | {'T1_new':>7} {'T1_old':>7} {'dT1%':>7} | "
          f"{'T2_new':>7} {'T2_old':>7} {'dT2%':>7}")
    gT = []
    for d, rec in zip(g["per_frame"], rows):
        T1, T1_m, T1_p = temperature_eV_bounds(
            d["sigma_C1"], d.get("sigma_C1_err", 0.0), d["center_C1"],
            lambda_C1, inst_sigma_C1)
        T2, T2_m, T2_p = temperature_eV_bounds(
            d["sigma_C2"], d.get("sigma_C2_err", 0.0), d["center_C2"],
            lambda_C2, inst_sigma_C2)
        # A percentage against an old T that railed at 0 eV is meaningless, so
        # print a dash rather than a 13-digit number.
        def _pct(new, old):
            return f"{100.0 * (new - old) / old:7.1f}" if old > 1.0 else f"{'-':>7}"
        gT.append({"frame": d["frame"], "T1": T1, "T2": T2,
                   "T1_m": T1_m, "T1_p": T1_p, "T2_m": T2_m, "T2_p": T2_p,
                   "T1_old": rec["T1_old"], "T2_old": rec["T2_old"],
                   "T1_old_m": rec["T1_old_m"], "T1_old_p": rec["T1_old_p"],
                   "T2_old_m": rec["T2_old_m"], "T2_old_p": rec["T2_old_p"]})
        print(f"  {d['frame']:5d} | {d['sigma_C1']:6.2f} {rec['sig1_old']:9.2f} "
              f"{d['sigma_C2']:6.2f} {rec['sig2_old']:9.2f} | "
              f"{T1:7.1f} {rec['T1_old']:7.1f} {_pct(T1, rec['T1_old'])} | "
              f"{T2:7.1f} {rec['T2_old']:7.1f} {_pct(T2, rec['T2_old'])}")

    if share_sigma:
        # Both temperatures come from ONE fitted width, so they differ only by
        # the deterministic lambda^2 and dispersion factors, not by anything
        # measured. Quantify that before the plot collapses them into one curve.
        spread = [200.0 * abs(t["T1"] - t["T2"]) / (t["T1"] + t["T2"])
                  for t in gT if (t["T1"] + t["T2"]) > 0]
        if spread:
            print(f"\n  shared sigma: T(C1) and T(C2) come from the same fitted")
            print(f"  width and differ by at most {max(spread):.2f}% "
                  f"(median {np.median(spread):.2f}%), which is the lambda^2 and")
            print("  dispersion conversion alone. They are NOT two independent")
            print("  measurements here, and the T plot draws them as one curve.")

    # The existing pipeline only reports T over frames 8-22; outside that window
    # the C lines are weak enough that the widths rail against their bounds and
    # neither the old nor the new T means anything. Summarise both windows.
    for lo_f, hi_f, label in ((first_frame, last_frame, "all frames"),
                              (8, 22, "frames 8-22 (the pipeline's T window)")):
        sel = [(t, r) for t, r in zip(gT, rows) if lo_f <= t["frame"] <= hi_f]
        if not sel:
            continue
        a = np.array([[t["T1"], t["T1_old"], t["T2"], t["T2_old"]] for t, _ in sel])
        print(f"\n  median T over {label}:")
        print(f"    T(C1) {np.median(a[:, 0]):8.1f} eV  (old {np.median(a[:, 1]):8.1f} eV)")
        print(f"    T(C2) {np.median(a[:, 2]):8.1f} eV  (old {np.median(a[:, 3]):8.1f} eV)")

    # ---- single-frame fit picture ----------------------------------------
    if do_plot and plot_frame is not None:
        fd_p = next((f for f in frames if f["frame"] == plot_frame), None)
        if fd_p is None:
            print(f"\n  frame {plot_frame} not among the usable frames - no fit plot")
        else:
            rec_p = next(r for r in rows if r["frame"] == plot_frame)
            gd_p = next(d for d in g["per_frame"] if d["frame"] == plot_frame)
            print(f"\n  --- frame {plot_frame} fit detail ---")
            print(f"    B = 0 refit         chi2r {rec_p['fit0']['chi2r']:.3f}  "
                  f"sigma_C1 {rec_p['fit0']['sigma_C1']:6.2f}  "
                  f"sigma_C2 {rec_p['fit0']['sigma_C2']:6.2f} px")
            print(f"    per-frame free B    chi2r {rec_p['fit']['chi2r']:.3f}  "
                  f"sigma_C1 {rec_p['fit']['sigma_C1']:6.2f}  "
                  f"sigma_C2 {rec_p['fit']['sigma_C2']:6.2f} px  "
                  f"B = {rec_p['fit']['B']:.3f} T")
            print(f"    global shared B     chi2r {gd_p['chi2r']:.3f}  "
                  f"sigma_C1 {gd_p['sigma_C1']:6.2f}  "
                  f"sigma_C2 {gd_p['sigma_C2']:6.2f} px  B = {g['B']:.3f} T")
            # B = 0 is drawn widest and dashed so it stays visible on the frames
            # where the free-B fit lands on top of it - which is the whole point
            # on a frame like 13, where B buys nothing.
            plot_frame_fit(fd_p, [
                {"label": "B = 0 (no Zeeman)", "params": rec_p["fit0"],
                 "B": 0.0, "color": "0.15", "ls": "--", "lw": 3.2},
                {"label": "per-frame free B", "params": rec_p["fit"],
                 "B": rec_p["fit"]["B"], "color": "tab:blue", "ls": "-",
                 "lw": 1.4},
                {"label": "global shared B", "params": gd_p, "B": g["B"],
                 "color": "tab:red", "ls": "-", "components": True},
            ], exp_i, suffix="_sharedsigma" if share_sigma else "")

    # ---- plots ------------------------------------------------------------
    if do_plot:
        fr = [r["frame"] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ax = axes[0]
        # Frames that settle near B = 0 have a stderr that diverges (d/dB -> 0
        # there), and plotting those bars puts the y-axis in the tens of
        # thousands of tesla and flattens everything real. Draw them as open
        # markers with no bar, and say so, rather than clipping a bar that
        # would otherwise read as a genuine 1-sigma interval.
        fr_a = np.asarray(fr)
        det = Be <= 1.0
        ax.errorbar(fr_a[det], Ba[det], yerr=Be[det], fmt="o", ms=4, lw=1,
                    capsize=2, color="tab:blue", label="per-frame B")
        if np.any(~det):
            ax.plot(fr_a[~det], Ba[~det], "o", ms=6, mfc="none",
                    color="tab:blue", label="B unconstrained (stderr diverges)")
        ax.axhline(g["B"], color="tab:red", lw=2,
                   label=f"global B = {g['B']:.3f} T")
        ax.fill_between([min(fr), max(fr)], g["B"] - g["B_err"], g["B"] + g["B_err"],
                        color="tab:red", alpha=0.2, label="global +/- 1 sigma")
        ax.set_ylim(B_MIN_TESLA - 0.08, B_MAX_TESLA + 0.08)
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Fitted B [T]")
        ax.set_title("Zeeman field from the joint C1+C2 fit\n(transverse pattern assumed)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(fr, [r["fit0"]["chi2r"] for r in rows], "s-", ms=4,
                color="tab:gray", label="B = 0")
        ax.plot(fr, [r["fit"]["chi2r"] for r in rows], "o-", ms=4,
                color="tab:blue", label="B free (per frame)")
        ax.plot(fr, [d["chi2r"] for d in g["per_frame"]], "^-", ms=4,
                color="tab:red", label="global shared B")
        ax.set_xlabel("Frame number")
        ax.set_ylabel("reduced chi2")
        ax.set_title("Did adding B actually buy any fit quality?")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[2]
        # Bars are the fitted width's own stderr pushed through the T formula,
        # so they are ASYMMETRIC: T goes as (sigma^2 - sigma_inst^2), which
        # compresses the low side and stretches the high side, and pins the
        # low side at 0 once sigma - 1 sigma reaches the instrumental floor.
        # They are the STATISTICAL width error only - they do not carry the
        # assumed instrumental sigma, the geometry assumption, or the
        # B-vs-width degeneracy, all of which are larger.
        def _tbars(key_m, key_p):
            return np.array([[t[key_m] for t in gT], [t[key_p] for t in gT]])

        # The OLD curves stay separate in both width models: the existing
        # pipeline fits C1 and C2 with independent widths, so those two
        # temperatures carry genuinely different information.
        ax.errorbar(fr, [t["T1_old"] for t in gT],
                    yerr=_tbars("T1_old_m", "T1_old_p"), fmt="s--", ms=4, lw=1,
                    elinewidth=0.8, capsize=2, color="tab:green", alpha=0.55,
                    label="T(C1) old")
        ax.errorbar(fr, [t["T2_old"] for t in gT],
                    yerr=_tbars("T2_old_m", "T2_old_p"), fmt="s--", ms=4, lw=1,
                    elinewidth=0.8, capsize=2, color="tab:orange", alpha=0.55,
                    label="T(C2) old")

        if share_sigma:
            # One width feeds both lines, so T(C1) and T(C2) are the same
            # number up to the lambda^2 and dispersion factors - a sub-percent
            # difference that would draw as one curve with two overlapping sets
            # of bars. Plot the mean once instead of pretending to two
            # independent measurements, which under a shared sigma it is not.
            ax.errorbar(fr, [0.5 * (t["T1"] + t["T2"]) for t in gT],
                        yerr=np.array(
                            [[0.5 * (t["T1_m"] + t["T2_m"]) for t in gT],
                             [0.5 * (t["T1_p"] + t["T2_p"]) for t in gT]]),
                        fmt="o-", ms=4, lw=1.4, elinewidth=1.0, capsize=2,
                        color="tab:purple",
                        label="T(C1 & C2) global-B, shared sigma")
        else:
            ax.errorbar(fr, [t["T1"] for t in gT], yerr=_tbars("T1_m", "T1_p"),
                        fmt="o-", ms=4, lw=1.4, elinewidth=1.0, capsize=2,
                        color="tab:green", label="T(C1) global-B")
            ax.errorbar(fr, [t["T2"] for t in gT], yerr=_tbars("T2_m", "T2_p"),
                        fmt="o-", ms=4, lw=1.4, elinewidth=1.0, capsize=2,
                        color="tab:orange", label="T(C2) global-B")
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Frame number")
        ax.set_ylabel("T [eV]")
        ax.set_title("Temperature before and after\nthe Zeeman component model")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out_png = os.path.join(
            C3_BASE_DIR,
            f"zeeman_joint_fit_C{exp_i}"
            f"{'_sharedsigma' if share_sigma else ''}.png")
        plt.savefig(out_png, dpi=130)
        print(f"\n  figure written to {out_png}")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)

    print("\n" + "=" * 78)
    print("CAVEATS THAT TRAVEL WITH EVERY NUMBER ABOVE")
    print("=" * 78)
    print("  * TRANSVERSE (pi + sigma) geometry is ASSUMED, not measured. A")
    print("    longitudinal (sigma-only) view has different shifts and")
    print("    strengths and would return a different B from this same data.")
    print("  * B enters only through component displacement. Anything else that")
    print("    widens or reshapes the blend - Stark, flow, unresolved structure,")
    print("    an imperfect continuum - can be absorbed into B, so a non-zero")
    print("    fitted B is an upper bound on the field, not a detection.")
    print("  * The instrumental sigma is still the assumed 0.05 nm FWHM value;")
    print("    T scales directly off it and it has never been measured.")
    return {"rows": rows, "global": g, "gT": gT, "frames": frames,
            "share_sigma": share_sigma, "chi2_B0_sum": chi2_0_sum,
            "dof_B0_sum": dof_0}


def run_both_width_models(exp_i=559, first_frame=ZEEMAN_FIRST_FRAME,
                          last_frame=ZEEMAN_LAST_FRAME, plot_frame=13,
                          exp_dir_name=None):
    """
    Run the whole analysis twice on the SAME loaded frames - independent sigma
    per line, then one sigma shared between the lines - and compare them.

    The shared-sigma model is nested inside the independent one (it is that
    model with sigma_C1 = sigma_C2 imposed), so their chi2 difference is a
    proper likelihood-ratio statistic with one degree of freedom per frame.
    """
    print(f"Loading C{exp_i} frames {first_frame}-{last_frame} ...")
    frames = []
    for f in range(first_frame, last_frame + 1):
        fd = prepare_frame(exp_i, f, exp_dir_name)
        if fd is None:
            print(f"  frame {f}: skipped (load or seed fit failed)")
        else:
            frames.append(fd)
    if not frames:
        print("  nothing to fit.")
        return None

    indep = run_zeeman_analysis(exp_i, first_frame, last_frame, do_plot=True,
                                plot_frame=plot_frame, share_sigma=False,
                                frames=frames, exp_dir_name=exp_dir_name)
    print("\n\n")
    shared = run_zeeman_analysis(exp_i, first_frame, last_frame, do_plot=True,
                                 plot_frame=plot_frame, share_sigma=True,
                                 frames=frames, exp_dir_name=exp_dir_name)

    # ---- head-to-head -----------------------------------------------------
    print("\n" + "=" * 78)
    print("4. INDEPENDENT sigma  vs  SHARED sigma")
    print("=" * 78)
    gi, gs = indep["global"], shared["global"]
    print("  The shared-sigma model is the independent one with sigma_C1 =")
    print("  sigma_C2 imposed, so it is nested: it can only fit worse, and the")
    print("  question is by how much for the parameters it gives back.")
    print(f"\n  {'':<26} {'independent':>14} {'shared':>14}")
    print(f"  {'global B [T]':<26} {gi['B']:14.4f} {gs['B']:14.4f}")
    print(f"  {'  stderr [T]':<26} {gi['B_err']:14.4f} {gs['B_err']:14.4f}")
    print(f"  {'global chi2':<26} {gi['chi2']:14.1f} {gs['chi2']:14.1f}")
    print(f"  {'  dof':<26} {gi['dof']:14d} {gs['dof']:14d}")
    print(f"  {'  chi2r':<26} {gi['chi2r']:14.3f} {gs['chi2r']:14.3f}")
    print(f"  {'free parameters':<26} {gi['n_par']:14d} {gs['n_par']:14d}")
    print(f"  {'B = 0 chi2':<26} {indep['chi2_B0_sum']:14.1f} "
          f"{shared['chi2_B0_sum']:14.1f}")
    print(f"  {'delta-chi2 for B':<26} "
          f"{indep['chi2_B0_sum'] - gi['chi2']:14.1f} "
          f"{shared['chi2_B0_sum'] - gs['chi2']:14.1f}")

    n_shared_constraints = gi["n_par"] - gs["n_par"]
    d_chi2 = gs["chi2"] - gi["chi2"]
    print(f"\n  cost of tying the widths: delta-chi2 = {d_chi2:.1f} for "
          f"{n_shared_constraints} constraints")
    print(f"  (one sigma_C1 = sigma_C2 per frame). Per constraint that is "
          f"{d_chi2 / max(n_shared_constraints, 1):.2f};")
    print("  a constraint the data did not mind costs about 1.")

    print(f"\n  per-frame width agreement under the INDEPENDENT model")
    print("  (how far apart the two widths wanted to be in the first place):")
    print(f"    {'frame':>5} {'sigC1':>7} {'sigC2':>7} {'ratio':>7} | "
          f"{'sigma_shared':>12} | {'chi2r ind':>10} {'chi2r shr':>10}")
    for di, ds in zip(gi["per_frame"], gs["per_frame"]):
        ratio = di["sigma_C2"] / di["sigma_C1"] if di["sigma_C1"] > 0 else np.nan
        flag = "  <- T window" if 8 <= di["frame"] <= 22 else ""
        print(f"    {di['frame']:5d} {di['sigma_C1']:7.2f} {di['sigma_C2']:7.2f} "
              f"{ratio:7.3f} | {ds['sigma']:12.2f} | "
              f"{di['chi2r']:10.3f} {ds['chi2r']:10.3f}{flag}")

    print(f"\n  temperature from the shared width, frames 8-22:")
    sel_s = [t for t in shared["gT"] if 8 <= t["frame"] <= 22]
    sel_i = [t for t in indep["gT"] if 8 <= t["frame"] <= 22]
    print(f"    {'T(C1) shared':<18} median {np.median([t['T1'] for t in sel_s]):8.1f} eV")
    print(f"    {'T(C2) shared':<18} median {np.median([t['T2'] for t in sel_s]):8.1f} eV")
    print(f"    {'T(C1) independent':<18} median {np.median([t['T1'] for t in sel_i]):8.1f} eV")
    print(f"    {'T(C2) independent':<18} median {np.median([t['T2'] for t in sel_i]):8.1f} eV")
    print(f"    {'T(C1) old pipeline':<18} median {np.median([t['T1_old'] for t in sel_i]):8.1f} eV")
    print(f"    {'T(C2) old pipeline':<18} median {np.median([t['T2_old'] for t in sel_i]):8.1f} eV")
    print("  With one shared width the two lines necessarily give the same T up")
    print("  to the lambda^2 factor, so T(C1) and T(C2) stop being an")
    print("  independent consistency check - that is the price of the constraint.")
    return {"independent": indep, "shared": shared}


SHOW_PLOTS = True


def run_legacy_frame_scan(exp_i=559, first_frame=5, last_frame=50,
                          first_frame_t=8, last_frame_t=22, exp_dir_name=None):
    """The original per-frame n_e / T scan, unchanged."""
    frame_amount = last_frame - first_frame + 1
    n = [0] * frame_amount
    t1 = [0] * frame_amount
    t2 = [0] * frame_amount # Array for the C2 temperatures

    for frame_i in range(first_frame, last_frame + 1):
        # You can test a single frame plot by changing this temporarily, e.g., if frame_i == 5: extract_nt_from_frame(exp_i, frame_i, do_plot=True)
        if(frame_i == 13):  # plot only for frame
            extract_nt_from_frame(exp_i, frame_i, do_plot=True, exp_dir_name=exp_dir_name)
        n[frame_i - first_frame], t1[frame_i - first_frame], t2[frame_i - first_frame] = extract_nt_from_frame(exp_i, frame_i, do_plot=False, exp_dir_name=exp_dir_name)

    # Plot n_e, T_C1, and T_C2 vs frame_i
    plt.figure(figsize=(15, 5)) # Increased width to fit 3 plots nicely
    
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
    plt.title(f"C{exp_i} T(C1) vs Frame (not all frames)")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(t_frames, t2[t_start_idx:t_end_idx], marker='o', color='orange')
    plt.xlabel("Frame Number")
    plt.ylabel("Temperature (T_C2) [eV]")
    plt.title(f"C{exp_i} T(C2) vs Frame (not all frames)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # @CHANGE_ME: experiment number for the report
    EXP_I = 559

    # Command-line overrides of the config block at the top of the file.
    PLOT_FRAME = ZEEMAN_PLOT_FRAME
    if "--frame" in sys.argv:
        v = sys.argv[sys.argv.index("--frame") + 1]
        PLOT_FRAME = None if v.lower() == "none" else int(v)

    FIRST = ZEEMAN_FIRST_FRAME
    if "--first" in sys.argv:
        FIRST = int(sys.argv[sys.argv.index("--first") + 1])

    LAST = ZEEMAN_LAST_FRAME
    if "--last" in sys.argv:
        LAST = int(sys.argv[sys.argv.index("--last") + 1])

    WIDTH_MODEL = ZEEMAN_WIDTH_MODEL
    if "--width-model" in sys.argv:
        WIDTH_MODEL = sys.argv[sys.argv.index("--width-model") + 1]

    EXP_DIR_NAME = EXPERIMENT_DIR_NAME or None
    if "--exp-dir" in sys.argv:
        EXP_DIR_NAME = sys.argv[sys.argv.index("--exp-dir") + 1]

    if "--legacy" in sys.argv:
        run_legacy_frame_scan(EXP_I, exp_dir_name=EXP_DIR_NAME)
    else:
        SHOW_PLOTS = "--no-plot" not in sys.argv
        if WIDTH_MODEL == "both":
            run_both_width_models(EXP_I, first_frame=FIRST, last_frame=LAST,
                                  plot_frame=PLOT_FRAME,
                                  exp_dir_name=EXP_DIR_NAME)
        else:
            run_zeeman_analysis(EXP_I, first_frame=FIRST, last_frame=LAST,
                                do_plot=True, plot_frame=PLOT_FRAME,
                                share_sigma=(WIDTH_MODEL == "shared"),
                                exp_dir_name=EXP_DIR_NAME)