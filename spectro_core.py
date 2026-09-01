"""
Shared core for the C559 ProEM Ha + C II analysis.

Calibration, frame loading, noise model, line profiles and the C II line table.
Everything downstream (calibrate_lines, ha_density, stark_consistency,
global_T_fit, run_n_T_vs_frame) imports from here.

The pixel<->wavelength calibration and the TIFF loader are taken unchanged from
HaC_ProEM_Yair.py - they are correct and well tested.
"""

import os
import sys
import glob
import json

import numpy as np
import tifffile as tiff
from scipy.special import wofz

# The project lives under a path with Hebrew characters, and the Windows
# console defaults to cp1252. Without this, printing any path raises
# UnicodeEncodeError. Same guard the original scripts used.
for _stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# --- Calibration constants (pixel <-> wavelength) ---
PIXEL_CENTER_REF = 800.0
LAMBDA_CENTER_REF_NM = 658.0
R0_NM_PER_PX = 1.304735e-02
R1_NM_PER_PX_PER_NM = -1.285110e-05

# --- Physics constants ---
LAMBDA_HA_NM = 656.28      # H-alpha rest wavelength
MC2_H_EV = 931.49e6 * 1     # hydrogen rest mass energy
MC2_C_EV = 931.49e6 * 12    # carbon rest mass energy
C_KM_S = 299792.458

# --- Detector / acquisition settings ---
DETECTOR_X_MAX = 1600
# The ProEM frames are 4 rows tall, so (1, 4) silently discarded row 0 - a
# quarter of the signal. validate_calibration.py check 2 measured the centroid
# drift across all four rows at 0.37-0.41 px (<1 km/s) and the extra smearing
# from averaging at +0.03 px (+0.2%), i.e. negligible, so row 0 is now included
# for the ~15% SNR it is worth.
Y_RANGE = (0, 4)           # rows of the ProEM frame averaged into the 1-D profile

# --- Usable frame range ---
# Outside this range the C II lines are not visible: C1 peak SNR collapses from
# ~50 at frame 22 to ~1 from frame 26 onward. Everything in this project is
# plotted against FRAME NUMBER - there is deliberately no time axis.
FIRST_FRAME = 5
LAST_FRAME = 23

# --- Fit windows (pixels) ---
HA_WINDOW = (120, 700)     # H-alpha only, no C lines fall in here
C_WINDOW = (700, 915)      # the whole C II group
C1_WINDOW = (700, 800)     # C1 pair alone
C2_WINDOW = (810, 915)     # C2 pair alone

# --- Noise model -------------------------------------------------------------
# Measured from Savitzky-Golay residuals over frames 5-23, binned by signal
# level, with high-curvature points excluded so the steep H-alpha wing does not
# masquerade as noise. The variance is linear in signal with a large negative
# intercept:
#
#     signal  560-650  -> var  22.8   (var/y 0.037)
#     signal  900-1100 -> var 173.2   (var/y 0.179)
#     signal 1900-2600 -> var 604.1   (var/y 0.281)
#
# fit:  var = 0.387 * (y - 532)
#
# i.e. the camera carries a ~532 count bias pedestal (nothing in the old code
# subtracted it) and ~0.387 counts of variance per count of real signal above
# it. Plain sqrt(y) weighting overstates the noise by 4.8x at 600 counts but
# only 1.9x at 2000 counts, so it under-weights the line wings by ~2.5x
# relative to the core. Since the Lorentzian gamma (and therefore n_e) is set
# almost entirely by the wings, that biases the density.
# Rerun calibrate_noise() if the camera settings change.
CAMERA_BIAS = 531.8         # counts
CAMERA_VAR_GAIN = 0.3874    # variance counts per signal count above bias
MIN_SIGNAL_FOR_NOISE = 20.0  # floor so sigma never reaches zero

# --- Instrumental width ------------------------------------------------------
# NOT a lamp measurement. Derived from the data as the floor of the shared
# Gaussian width of the C II group over frames 20-23, where the plasma has
# decayed and Stark broadening is smallest:
#
#     frame 20  6.838 px      frame 22  6.817 px
#     frame 21  6.794 px      frame 23  6.809 px
#
# This is an UPPER BOUND on the true instrumental width, so temperatures
# derived from it are lower bounds. The old assumption (inst_fwhm_Ha = 0.05 nm)
# converts to 4.60 px under the CORRECTED dispersion, not the 1.63 px that the
# old buggy dispersion implied. So it was about 1.48x too small, not 4.2x, and
# the original T ~ 470 eV becomes ~59 eV once the dispersion factor (8.02x on
# T) is applied. Both the old 4.2x claim and the 470 eV figure were themselves
# products of the dispersion bug; see dispersion_nm_per_px.
#
# Every T in this project scales with the difference between the measured
# width and this number, so it is the dominant systematic. Replace it the
# moment a calibration lamp spectrum (same slit and grating) is available.
# This is only the fallback: calibrate_lines.py writes the authoritative value
# into line_table.json, and load_line_table() is what everything downstream uses.
SIGMA_INST_PX = 6.794
SIGMA_INST_SOURCE = "data-floor (frames 20-23); NOT lamp-calibrated"

# --- C II line table ---------------------------------------------------------
# NIST critically-evaluated air wavelengths (Kramida & Haris 2022), consistent
# with lambda_Ha = 656.28:
#
#     C II 3s 2S(1/2) - 3p 2P*(3/2)   657.80482 nm   upper level g = 4
#     C II 3s 2S(1/2) - 3p 2P*(1/2)   658.28761 nm   upper level g = 2
#
# The old constants lambda_C1 = 657.736 and lambda_C2 = 658.1978 were low by
# 0.069 and 0.090 nm respectively.
#
# NIST places only these TWO transitions in the window, but two components do
# not fit the observed group (chi2r 14.0 against 3.4 for four; see
# validate_calibration.py check 6), and the measured doublet branching ratio is
# ~3.09 against an optically thin limit of 2.0 - so the group carries flux that
# C II alone cannot supply, from an emitter not yet identified. The four
# components below are therefore an EMPIRICAL description, not four claimed
# transitions: each pair straddles one NIST line and together they absorb the
# real profile shape. Treat the individual component wavelengths as fit
# coordinates, not as line identifications.
NIST_C1_NM = 657.80482
NIST_C2_NM = 658.28761
NIST_SEP_NM = NIST_C2_NM - NIST_C1_NM      # 0.48279 nm, absolute standard

# ABSOLUTE OFFSET, kept deliberately visible: nm_to_pixel(NIST_C1_NM) = 757.50,
# but the doublet is measured centred on 752.30 px - a 5.2 px = 0.024 nm offset
# in the absolute zero point. The SEPARATION is right to -0.36%, so the
# dispersion is sound and only the offset is off; widths (which depend on the
# scale, not the zero) are unaffected. Seeds below use the OBSERVED pixels so
# the fits start on the data, not 5 px away from it. Do not read the absolute
# wavelengths here as a calibration claim.
DEFAULT_LINE_TABLE = [
    {"label": "C1a", "pixel": 745.60, "lambda_nm": 657.7509, "group": "C1"},
    {"label": "C1b", "pixel": 759.00, "lambda_nm": 657.8124, "group": "C1"},
    {"label": "C2a", "pixel": 847.80, "lambda_nm": 658.2202, "group": "C2"},
    {"label": "C2b", "pixel": 866.40, "lambda_nm": 658.3057, "group": "C2"},
]

LINE_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "line_table.json")

# --- Data root ---
C3_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Wavelength calibration --------------------------------------------------

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
# THE RULE: a pixel WIDTH is not a pixel POSITION, so it cannot be handed to
# pixel_to_nm directly. Convert a width by DIFFERENCING the trusted position
# function around the feature's own centre.
#
# There used to be a dispersion_nm_per_px(x) here returning a closed-form
# derivative, and widths were converted by multiplying by it. That was a second,
# independent implementation of the wavelength scale, and it silently disagreed
# with pixel_to_nm by a factor of 2.83 for the whole life of the project -
# inflating every velocity by 2.83x, every T by 8.02x and every n_e by 4.62x.
# Nothing caught it because nothing compared the two.
#
# The functions below cannot drift that way: they contain no formula of their
# own, only calls to pixel_to_nm / nm_to_pixel. If the instrument model changes,
# they follow automatically. That is the point of the refactor - the numeric bug
# was fixed first, separately; this removes the shape of code that allowed it.
#
# Differencing is also strictly more correct than multiplying: it captures the
# curvature of the wavelength axis across the line, which a single dispersion
# value cannot. See test_width_conversion() for the size of that term.

def width_px_to_nm(width_px, center_px):
    """
    Convert a WIDTH (sigma, gamma, FWHM...) in pixels to nm.

    Centred/symmetric difference about the feature's fitted centre, so neither
    side of the line is privileged:

        width_nm = (pixel_to_nm(c + w) - pixel_to_nm(c - w)) / 2

    Pass the fitted centre of the line the width belongs to, not a generic
    reference pixel.
    """
    c = np.asarray(center_px, dtype=float)
    w = np.asarray(width_px, dtype=float)
    return (pixel_to_nm(c + w) - pixel_to_nm(c - w)) / 2.0


def width_nm_to_px(width_nm, center_nm):
    """
    Convert a WIDTH in nm to pixels, anchored at the line centre center_nm.

        width_px = (nm_to_pixel(lam + w) - nm_to_pixel(lam - w)) / 2

    CENTRED, deliberately, so that this is the exact inverse of width_px_to_nm.

    The one-sided form nm_to_pixel(lam + w) - nm_to_pixel(lam) also only needs
    the line centre, but it is not the inverse of the centred forward
    conversion: the two disagree at O(|b|*w/2) ~ 5e-5 for a 7 px width, and
    since T ~ sigma^2 that shows up as a systematic ~1e-4 shift in every fitted
    temperature. Centring costs nothing (it needs no pixel centre either) and
    removes that shift, leaving only the O((b*w)^2) curvature term ~1e-9.

    Like width_px_to_nm this holds no formula of its own - only nm_to_pixel.
    """
    lam = float(center_nm)
    w = float(width_nm)
    return (nm_to_pixel(lam + w) - nm_to_pixel(lam - w)) / 2.0


def test_width_conversion(tol=1e-5, verbose=False):
    """
    Guard the width-conversion mechanism. Run:  python spectro_core.py

    (a) round trip - nm -> px -> nm returns the original width.
    (b) small-width limit agrees with a NUMERICAL derivative of pixel_to_nm,
        which is the check that never existed and would have caught the 2.83x
        bug immediately.
    (c) agrees with the historical closed-form dispersion this refactor
        replaced - the CORRECTED one, (lam0*b + a)*exp(b dx) - proving the
        refactor moves no number.

    (c) is deliberately NOT asserted at machine epsilon, because the two are
    not the same quantity. Differencing captures the curvature of the
    wavelength axis across the line; multiplying by one dispersion value does
    not. Their ratio is exactly sinh(b*w)/(b*w) = 1 + (b*w)^2/6 + ..., i.e.
    ~1e-9 for a 7 px sigma and ~3e-7 for a 100 px gamma. The differenced form
    is the more correct of the two; the agreement is what shows nothing moved.
    """
    xs = np.array([50.0, 200.0, 426.0, 700.0, 800.0, 900.0, 1200.0, 1550.0])
    b = float(R1_NM_PER_PX_PER_NM)
    slope0 = LAMBDA_CENTER_REF_NM * b + R0_NM_PER_PX

    def _historical_dispersion(x):
        """The corrected closed form this refactor removed. TEST FIXTURE ONLY."""
        return slope0 * np.exp(b * (np.asarray(x, dtype=float)
                                    - PIXEL_CENTER_REF))

    worst_deriv = 0.0
    worst_hist = 0.0
    if verbose:
        print(f"  {'x':>7} {'width':>7} {'differenced [nm]':>18} "
              f"{'vs numerical':>13} {'vs old closed form':>19}")
    for x in xs:
        # (b) small-width limit against a numerical derivative
        h = 1e-3
        num = (float(pixel_to_nm(x + h)) - float(pixel_to_nm(x - h))) / (2 * h)
        implied = float(width_px_to_nm(h, x)) / h
        rel_d = abs(implied - num) / abs(num)
        worst_deriv = max(worst_deriv, rel_d)
        assert rel_d < 1e-9, (
            f"width_px_to_nm at x={x} implies dispersion {implied:.9e}, but the "
            f"numerical derivative of pixel_to_nm is {num:.9e}")

        for w in (7.0, 100.0):
            diff_nm = float(width_px_to_nm(w, x))
            hist_nm = w * float(_historical_dispersion(x))
            rel_h = abs(diff_nm - hist_nm) / abs(hist_nm)
            worst_hist = max(worst_hist, rel_h)
            if verbose:
                print(f"  {x:7.1f} {w:7.1f} {diff_nm:18.9f} "
                      f"{rel_d:13.2e} {rel_h:19.2e}")
            assert rel_h < tol, (
                f"width_px_to_nm({w}, {x}) = {diff_nm:.9e} disagrees with the "
                f"historical closed form {hist_nm:.9e} by {rel_h:.2e} > {tol:.0e}")

        # (a) round trip. Inverse to O((b*w)^2) ~ 4e-9, not to machine epsilon:
        # width_px_to_nm is symmetric in PIXELS while width_nm_to_px is
        # symmetric in WAVELENGTH, and a nonlinear map does not carry one
        # symmetric interval onto the other exactly. (The earlier one-sided
        # convention closed only to |b|*w/2 ~ 4.5e-5, four orders worse, and
        # showed up as a systematic 1e-4 shift in every fitted temperature.)
        lam = float(pixel_to_nm(x))
        w_nm = float(width_px_to_nm(7.0, x))
        back = width_nm_to_px(w_nm, lam)
        rel_rt = abs(back - 7.0) / 7.0
        assert rel_rt < 1e-7, (
            f"round trip at x={x}: 7.0 px -> {w_nm:.9e} nm -> {back:.9f} px, "
            f"relative {rel_rt:.2e}, worse than the O((b*w)^2) curvature term "
            f"can explain")

    return worst_deriv, worst_hist


# --- Line profiles -----------------------------------------------------------
# amplitude is the integrated AREA, not the peak height, in both profiles.
# sigma and gamma are in PIXELS. gamma is the Lorentzian half width at half max,
# so the Lorentzian FWHM is 2*gamma.

def voigt(x, amplitude, center, sigma, gamma):
    """Area-normalised Voigt profile (no offset - continuum is fitted separately)."""
    sigma = abs(sigma)
    gamma = abs(gamma)
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def gaussian(x, amplitude, center, sigma):
    """Area-normalised Gaussian profile (no offset)."""
    sigma = abs(sigma)
    return (amplitude / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)))


def polynomial_continuum(x, coeffs, x_ref=800.0, x_scale=100.0):
    """
    Smooth continuum under a fit window, in a scaled coordinate so the
    coefficients stay order-1 and the fit is well conditioned.

    Used instead of the old edge-window baseline subtraction. Both of those
    windows (654.31-655.0 and 658.32-659.0 nm) sit inside the H-alpha wing -
    at 5% of peak H-alpha spans px 1-787 - and the right one starts 19 px from
    a real C line, so subtracting a line through them removes real signal.
    A cubic over a 200 px window is well separated in spatial frequency from
    the ~7 px wide C lines, so it absorbs the H-alpha wing without eating them.
    """
    t = (np.asarray(x, dtype=float) - x_ref) / x_scale
    out = np.zeros_like(t)
    for i, c in enumerate(coeffs):
        out = out + c * t ** i
    return out


# --- Broadening helpers ------------------------------------------------------

def thermal_sigma_px(T_eV, lambda_nm, mc2_eV, x_px):
    """
    Doppler sigma in pixels for a species of rest mass energy mc2_eV at T_eV.

        sigma_lambda / lambda = sqrt(kT / mc^2)
    """
    T_eV = max(float(T_eV), 0.0)
    sigma_nm = lambda_nm * np.sqrt(T_eV / mc2_eV)
    # Anchor at the actual pixel position; callers pass x_px = nm_to_pixel of
    # this same line, so pixel_to_nm(x_px) is its centre wavelength.
    return width_nm_to_px(sigma_nm, float(pixel_to_nm(x_px)))


def temperature_from_sigma(sigma_px, lambda_nm, mc2_eV, x_px,
                           sigma_inst_px=None):
    """
    Invert thermal_sigma_px: deconvolve the instrumental width and return T in eV.
    Returns 0.0 when the line is at or below the instrumental width (unresolved).
    """
    if sigma_inst_px is None:
        sigma_inst_px = SIGMA_INST_PX
    sigma_px = abs(float(sigma_px))
    if sigma_px <= sigma_inst_px:
        return 0.0
    sigma_th_px = np.sqrt(sigma_px ** 2 - sigma_inst_px ** 2)
    sigma_th_nm = float(width_px_to_nm(sigma_th_px, x_px))
    return mc2_eV * (sigma_th_nm / lambda_nm) ** 2


def n_e_from_ha_stark(gamma_px, center_px):
    """
    Electron density from the H-alpha Stark width, Konjevic et al. (2012) Eq. 5:

        n_e = 1e17 * (FWHM_Stark[nm] / 1.098)^1.471   [cm^-3]

    The Lorentzian FWHM is 2*gamma. Uses the local dispersion at the fitted
    centre rather than the constant R0.
    """
    gamma_px = abs(float(gamma_px))
    fwhm_nm = 2.0 * float(width_px_to_nm(gamma_px, center_px))
    if fwhm_nm <= 0:
        return np.nan
    return 1e17 * (fwhm_nm / 1.098) ** 1.471


# --- Data loading ------------------------------------------------------------

def frame_tiff_path(exp_i, frame_i):
    """Return the ProEM TIFF path for one frame, or raise if it is missing."""
    proem_dir = os.path.join(C3_BASE_DIR, f"C{exp_i}", "ProEM")
    frame_token = f"{frame_i:02d}"
    patterns = [
        f"*-Frame-{frame_token}.tif", f"*-Frame-{frame_token}.tiff",
        f"*-Frame-{frame_i}.tif", f"*-Frame-{frame_i}.tiff",
    ]
    candidates = []
    for pat in patterns:
        candidates += sorted(glob.glob(os.path.join(proem_dir, pat)))
    for full_path in dict.fromkeys(candidates):
        if os.path.isfile(full_path):
            return full_path
    raise FileNotFoundError(
        f"No TIFF for experiment C{exp_i} frame {frame_i} in {proem_dir}")


def load_profile(exp_i, frame_i, y_range=Y_RANGE):
    """Average the spatial rows of one frame into a 1-D spectral profile."""
    data = np.array(tiff.imread(frame_tiff_path(exp_i, frame_i)), dtype=float)
    profile = np.mean(data[y_range[0]:y_range[1], :], axis=0)
    return np.arange(profile.size, dtype=float), profile


def load_stack(exp_i, frames, y_range=Y_RANGE):
    """Mean profile over several frames, for high-SNR line-position work."""
    acc = None
    n = 0
    for frame_i in frames:
        _, profile = load_profile(exp_i, frame_i, y_range)
        acc = profile if acc is None else acc + profile
        n += 1
    if n == 0:
        raise ValueError("load_stack got no frames")
    return np.arange(acc.size, dtype=float), acc / n


def noise_sigma(y, n_frames=1):
    """
    Per-point 1-sigma uncertainty from the measured camera noise model.
    Pass this to lmfit as weights=1/noise_sigma(y).

    n_frames is the number of frames averaged into y - averaging N frames
    divides the variance by N. Getting this wrong shows up immediately as a
    reduced chi-square of ~1/N instead of ~1.
    """
    y = np.asarray(y, dtype=float)
    signal = np.maximum(y - CAMERA_BIAS, MIN_SIGNAL_FOR_NOISE)
    return np.sqrt(CAMERA_VAR_GAIN * signal / float(n_frames))


def weights(y, n_frames=1):
    """lmfit weights (1/sigma) for a measured profile."""
    return 1.0 / noise_sigma(y, n_frames)


# --- Line table persistence --------------------------------------------------

def load_line_table(path=LINE_TABLE_PATH):
    """Line table written by calibrate_lines.py, falling back to the defaults."""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload["lines"], payload
    return list(DEFAULT_LINE_TABLE), None


def save_line_table(lines, extra=None, path=LINE_TABLE_PATH):
    payload = {"lines": lines}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def group_lines(lines, group):
    return [ln for ln in lines if ln["group"] == group]


# --- Diagnostics -------------------------------------------------------------

def c_group_snr(profile):
    """
    Peak SNR of the C1 group, used to decide whether a frame is worth fitting.
    Noise is taken from the line-free region redward of the C lines.
    """
    baseline = np.median(profile[900:960])
    noise = np.std(profile[950:1100])
    if noise <= 0:
        return 0.0
    return float((np.max(profile[740:770]) - baseline) / noise)


def calibrate_noise(exp_i, frames, verbose=True):
    """
    Re-derive CAMERA_BIAS and CAMERA_VAR_GAIN from the data.

    Residuals from a Savitzky-Golay smooth are binned by signal level and a
    straight line is fitted to variance against signal. Run this if the camera
    gain or binning changes.
    """
    from scipy.signal import savgol_filter

    smooth_all, resid_all = [], []
    for frame_i in frames:
        _, profile = load_profile(exp_i, frame_i)
        smooth = savgol_filter(profile, 15, 3)
        smooth_all.append(smooth)
        resid_all.append(profile - smooth)
    smooth_all = np.concatenate(smooth_all)
    resid_all = np.concatenate(resid_all)

    edges = [600, 700, 900, 1200, 1700, 2500, 4000, 7000]
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (smooth_all >= lo) & (smooth_all < hi)
        if m.sum() < 200:
            continue
        # savgol over 15 points with order 3 removes 4 degrees of freedom
        var = resid_all[m].var() * 15.0 / 14.0
        xs.append(smooth_all[m].mean())
        ys.append(var)
        if verbose:
            print(f"  signal {lo:5d}-{hi:<5d} n={m.sum():6d} "
                  f"rms={np.sqrt(var):6.2f} var={var:7.1f} var/y={var/smooth_all[m].mean():6.3f}")
    gain, intercept = np.polyfit(xs, ys, 1)
    bias = -intercept / gain
    if verbose:
        print(f"  => var = {gain:.4f} * (y - {bias:.1f})")
    return bias, gain


if __name__ == "__main__":
    # Regression guard. Widths are now converted by differencing pixel_to_nm,
    # so there is no second implementation of the wavelength scale left to
    # drift out of sync - which is how the 2.83x dispersion bug survived.
    print("Checking width conversion against pixel_to_nm ...")
    worst_deriv, worst_hist = test_width_conversion(verbose=True)
    print(f"\nOK")
    print(f"  vs numerical derivative of pixel_to_nm : worst {worst_deriv:.2e}")
    print(f"  vs the closed form this replaced       : worst {worst_hist:.2e}")
    print("  (the second is not zero and should not be: differencing captures")
    print("   the axis curvature across the line, sinh(b*w)/(b*w), which")
    print("   multiplying by a single dispersion value drops.)")
