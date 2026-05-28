import os
import glob
import sys
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


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

# --- C3 data root ---
C3_BASE_DIR = r""
C3_BASE_DIR = C3_BASE_DIR if C3_BASE_DIR else os.getcwd()

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

def prompt_c3_tiff_path(exp_i=None, frame_i=None):
    """Ask for an experiment number and frame number, return the matching C3 ProEM TIFF path."""
        

    proem_dir = os.path.join(C3_BASE_DIR, f"C{exp_i}", "ProEM")
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




tiff_path, exp_number, frame_number = prompt_c3_tiff_path(559,6)
print(f"Loading: {tiff_path}")

# Pixel range covering [MIN_WAVELENGTH_NM, MAX_WAVELENGTH_NM]
_min_pix = max(0, int(np.floor(nm_to_pixel(MIN_WAVELENGTH_NM))))
_max_pix = min(DETECTOR_X_MAX, int(np.ceil(nm_to_pixel(MAX_WAVELENGTH_NM))) + 1)

x_px, profile = load_full_profile(tiff_path, Y_RANGE, x_min=_min_pix, x_max=_max_pix)
profile = np.asarray(profile, dtype=float)

# Pixel -> wavelength
x_nm = pixel_to_nm(x_px)

# Baseline correction
profile, coeff, lift = apply_baseline_correction(x_px, profile)

print(
    f"Loaded C{exp_number} frame {frame_number}: {profile.size} samples, "
    f"wavelength range [{x_nm[0]:.3f}, {x_nm[-1]:.3f}] nm"
)
if DO_BASELINE_CORRECT:
    print(
        f"Baseline correction (degree {BASELINE_POLY_DEGREE}): "
        f"coeff={np.array2string(coeff, precision=4)}, lift={lift:.6g}"
    )
else:
    print("Baseline correction: disabled")

plt.figure(figsize=(8, 5))
plt.plot(x_nm, profile)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity [a.u.]")
plt.title(f"C{exp_number} frame {frame_number}")
plt.grid(True)
plt.tight_layout()
plt.show()
