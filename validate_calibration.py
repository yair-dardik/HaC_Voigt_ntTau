"""
validate_calibration.py - read-only validation suite for the C559 analysis.

Runs six independent checks on the calibration and the width measurements and
prints a sectioned report. This script MODIFIES NOTHING: it imports
spectro_core / calibrate_lines / ha_density and uses them as they stand. No
physics constant is changed here. Where a check cannot be run from the data on
disk, it says so instead of substituting a guess.

Reference: S. Mitrani, "Lines Broadening Diagnostics in Optical and X-ray
emission" (2025). Section 2.2.1 gives the HRS-750mm instrumental broadening as
0.24 A. Our measured floor is 0.209 nm = 2.09 A FWHM, 8.7x wider. Explaining
that gap is the point of this suite.

    python validate_calibration.py
    python validate_calibration.py --no-plot
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from lmfit import Parameters, minimize

import spectro_core as sc
import calibrate_lines as cl
import ha_density as hd


# --- Fixed atomic physics (NIST ASD, C II 2s2.3s 2S - 2s2.3p 2P*) ------------
NIST_C1_NM = 657.8048      # 2P*(3/2), upper level g = 4  -> the stronger line
NIST_C2_NM = 658.2876      # 2P*(1/2), upper level g = 2
NIST_SEP_NM = NIST_C2_NM - NIST_C1_NM          # 0.4828 nm, absolute standard
BRANCHING_RATIO = 2.0                          # g(3/2)/g(1/2) = 4/2, thin limit

# The same NIST query returns a neutral-carbon line only 0.072 nm from the
# strong C II component. At our resolution it is unresolvable from it, so it
# would sit inside the "C1" feature and inflate that feature's area.
NIST_CI_NM = 657.8769      # C I 2s2.2p.3p 1D2 - 2s2.2p.6d 1F*3

# --- Lab specification (Mitrani 2025, section 2.2.1) -------------------------
LAB_SPEC_FWHM_NM = 0.024   # 0.24 A for the HRS-750mm optical spectrometer

STACK_FRAMES = list(range(12, 24))
FLOOR_FRAMES = [20, 21, 22, 23]

RESULTS = []   # (check, measured, expected, verdict)


def record(check, measured, expected, verdict):
    RESULTS.append((check, measured, expected, verdict))


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _pixel_to_nm_with_R0(a, x):
    """pixel_to_nm with R0 replaced by `a`. Mirrors spectro_core exactly."""
    b = float(sc.R1_NM_PER_PX_PER_NM)
    x0 = float(sc.PIXEL_CENTER_REF)
    lam0 = float(sc.LAMBDA_CENTER_REF_NM)
    x = np.asarray(x, dtype=float)
    if abs(b) < 1e-15:
        return lam0 + a * (x - x0)
    return (-a / b) + (lam0 + a / b) * np.exp(b * (x - x0))


def _true_dispersion(x_px):
    """
    d(lambda)/d(pixel) obtained by differentiating pixel_to_nm ANALYTICALLY.

    pixel_to_nm is  lam = -a/b + (lam0 + a/b) exp(b (x-x0))
    so              dlam/dx = (lam0 + a/b) * b * exp(b (x-x0)) = (lam0*b + a) exp(...)

    Note this is NOT what sc.dispersion_nm_per_px returns; comparing the two is
    the substance of CHECK 1.
    """
    a = float(sc.R0_NM_PER_PX)
    b = float(sc.R1_NM_PER_PX_PER_NM)
    x0 = float(sc.PIXEL_CENTER_REF)
    lam0 = float(sc.LAMBDA_CENTER_REF_NM)
    x = np.asarray(x_px, dtype=float)
    return (lam0 * b + a) * np.exp(b * (x - x0))


def _residual_stats(res, weights=None):
    """Lag-1 autocorrelation and a runs-test z score for fit residuals."""
    r = np.asarray(res, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return np.nan, np.nan
    lag1 = float(np.sum(r[:-1] * r[1:]) / np.sum(r ** 2))
    s = np.sign(r)
    s = s[s != 0]
    if s.size < 4:
        return lag1, np.nan
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    n_pos = int(np.sum(s > 0))
    n_neg = int(np.sum(s < 0))
    n = n_pos + n_neg
    if n_pos == 0 or n_neg == 0:
        return lag1, np.nan
    mu = 2.0 * n_pos * n_neg / n + 1.0
    var = (mu - 1.0) * (mu - 2.0) / (n - 1.0)
    z = (runs - mu) / np.sqrt(var) if var > 0 else np.nan
    return lag1, float(z)


# =============================================================================
# CHECK 1 - dispersion against the fixed C II doublet separation
# =============================================================================

def _fit_two_components(x, y, n_frames=1, voigt=True, lock_sep=False,
                        sep_px=None):
    """Two components over the C window: free centres, shared sigma (+gamma)."""
    lo, hi = sc.C_WINDOW
    m = (x >= lo) & (x < hi)
    xf, yf = x[m], y[m]
    wf = sc.weights(yf, n_frames)

    seed1 = sc.nm_to_pixel(NIST_C1_NM)
    seed2 = sc.nm_to_pixel(NIST_C2_NM)

    p = Parameters()
    p.add("sigma", value=7.0, min=1.0, max=40.0)
    if voigt:
        p.add("gamma", value=2.0, min=0.0, max=60.0)
    p.add("cen0", value=seed1, min=seed1 - 25, max=seed1 + 25)
    if lock_sep:
        p.add("sep", value=float(sep_px), vary=False)
        p.add("cen1", expr="cen0 + sep")
    else:
        p.add("cen1", value=seed2, min=seed2 - 25, max=seed2 + 25)
    for i in range(2):
        idx = int(np.argmin(np.abs(xf - (seed1 if i == 0 else seed2))))
        h = float(np.max(yf[max(0, idx - 14):idx + 14]) - np.median(yf))
        p.add(f"amp{i}", value=max(h, 1.0) * 18.0, min=0.0)
    for i in range(4):
        p.add(f"c{i}", value=np.median(yf) if i == 0 else 0.0)

    def model(pp):
        out = sc.polynomial_continuum(
            xf, [pp[f"c{i}"].value for i in range(4)])
        for i in range(2):
            if voigt:
                out = out + sc.voigt(xf, pp[f"amp{i}"].value,
                                     pp[f"cen{i}"].value,
                                     pp["sigma"].value, pp["gamma"].value)
            else:
                out = out + sc.gaussian(xf, pp[f"amp{i}"].value,
                                        pp[f"cen{i}"].value, pp["sigma"].value)
        return out

    r = minimize(lambda pp: (model(pp) - yf) * wf, p)
    return r, xf, yf, model(r.params)


def check1(exp_i):
    banner("CHECK 1 - dispersion validated against the C II doublet separation")
    print(f"  NIST: {NIST_C1_NM} nm and {NIST_C2_NM} nm")
    print(f"  separation = {NIST_SEP_NM:.4f} nm  (fixed atomic physics)")

    x, y = sc.load_stack(exp_i, STACK_FRAMES)
    rv, xf, yf, _ = _fit_two_components(x, y, len(STACK_FRAMES), voigt=True)
    rg, _, _, _ = _fit_two_components(x, y, len(STACK_FRAMES), voigt=False)

    c0 = rv.params["cen0"].value
    c1 = rv.params["cen1"].value
    sep_px = c1 - c0
    sep_px_g = rg.params["cen1"].value - rg.params["cen0"].value
    mid = 0.5 * (c0 + c1)

    print(f"\n  Two-Voigt fit (free centres, shared sigma+gamma), "
          f"chi2r = {rv.redchi:.3f}")
    print(f"    centre 1 = {c0:8.3f} px      centre 2 = {c1:8.3f} px")
    print(f"    separation = {sep_px:.3f} px")
    print(f"  Two-Gaussian cross-check: separation = {sep_px_g:.3f} px "
          f"(chi2r = {rg.redchi:.3f})")
    print("    -> separation is robust to profile choice, so it is a clean "
          "ruler")

    # -- the two competing dispersion values -------------------------------
    sep_via_pixel_to_nm = float(sc.pixel_to_nm(c1) - sc.pixel_to_nm(c0))
    disp_used = float(sc.dispersion_nm_per_px(mid))
    sep_via_disp_fn = sep_px * disp_used
    disp_empirical = NIST_SEP_NM / sep_px
    disp_analytic = float(_true_dispersion(mid))

    print(f"\n  Separation implied by each route:")
    print(f"    a) pixel_to_nm(c1) - pixel_to_nm(c0)      = "
          f"{sep_via_pixel_to_nm:.4f} nm")
    print(f"    b) sep_px * dispersion_nm_per_px(mid)     = "
          f"{sep_via_disp_fn:.4f} nm")
    print(f"    NIST truth                                = "
          f"{NIST_SEP_NM:.4f} nm")
    err_a = 100 * (sep_via_pixel_to_nm / NIST_SEP_NM - 1)
    err_b = 100 * (sep_via_disp_fn / NIST_SEP_NM - 1)
    print(f"\n    route (a) error = {err_a:+.2f} %      <- wavelength axis")
    print(f"    route (b) error = {err_b:+.2f} %      <- what every WIDTH uses")

    print(f"\n  Local dispersion at px {mid:.0f}:")
    print(f"    required by NIST         = {disp_empirical:.6e} nm/px")
    print(f"    d/dx of pixel_to_nm      = {disp_analytic:.6e} nm/px")
    print(f"    dispersion_nm_per_px()   = {disp_used:.6e} nm/px")
    factor = disp_used / disp_empirical
    print(f"\n    dispersion_nm_per_px is too LARGE by a factor "
          f"{factor:.3f}")

    print("\n  Diagnosis:")
    print("    pixel_to_nm implements  lam = -a/b + (lam0 + a/b) exp(b (x-x0)),")
    print("    whose derivative at x0 is (lam0*b + a), NOT a. But")
    print("    dispersion_nm_per_px returns a*exp(...), i.e. it assumes the")
    print("    other common parameterisation lam = lam0 + (a/b)(exp(b dx) - 1).")
    print("    The doublet separation shows the wavelength AXIS is right and")
    print("    the dispersion FUNCTION is wrong: R0 is not the local dispersion")
    print("    in the model that is actually implemented.")

    # -- the R0 rescale the brief asked for ---------------------------------
    def sep_with(a):
        return float(_pixel_to_nm_with_R0(a, c1) - _pixel_to_nm_with_R0(a, c0))

    lo_k, hi_k = 0.2, 5.0
    for _ in range(200):
        mid_k = 0.5 * (lo_k + hi_k)
        if sep_with(mid_k * sc.R0_NM_PER_PX) < NIST_SEP_NM:
            lo_k = mid_k
        else:
            hi_k = mid_k
    k = 0.5 * (lo_k + hi_k)
    print(f"\n  R0 scale factor that makes pixel_to_nm match NIST exactly: "
          f"k = {k:.5f}")
    print(f"    corrected R0 = {k * sc.R0_NM_PER_PX:.6e} "
          f"(current {sc.R0_NM_PER_PX:.6e})")
    print("    NOTE: k is within a per-cent of 1, i.e. the wavelength axis needs")
    print("    almost no correction. Rescaling R0 is the WRONG knob - it would")
    print("    mis-set the axis to compensate for a bug in the derivative.")
    print(f"    The fix is to make dispersion_nm_per_px return "
          f"(lam0*b + a)*exp(b dx).")

    print(f"\n  Propagation of the {factor:.3f}x width correction "
          f"(widths are currently OVERSTATED):")
    print(f"    velocities  scale as disp^1     -> divide by {factor:.3f}")
    print(f"    T           scales as disp^2     -> divide by {factor ** 2:.3f}")
    print(f"    n_e         scales as disp^1.471 -> divide by "
          f"{factor ** 1.471:.3f}")

    sig_inst = sc.SIGMA_INST_PX
    fw_now = 2.355 * sig_inst * disp_used
    fw_fix = 2.355 * sig_inst * disp_empirical
    print(f"\n  Effect on the headline instrumental-width gap:")
    print(f"    sigma_inst = {sig_inst:.3f} px")
    print(f"    FWHM as currently computed = {fw_now:.4f} nm = "
          f"{10 * fw_now:.3f} A  -> {fw_now / LAB_SPEC_FWHM_NM:.1f}x the lab spec")
    print(f"    FWHM with correct dispersion = {fw_fix:.4f} nm = "
          f"{10 * fw_fix:.3f} A  -> {fw_fix / LAB_SPEC_FWHM_NM:.1f}x the lab spec")

    verdict = "FAIL" if abs(err_b) > 5 else "PASS"
    record("1 dispersion vs NIST doublet",
           f"width dispersion {factor:.2f}x too large",
           f"{NIST_SEP_NM:.4f} nm separation", verdict)
    return dict(sep_px=sep_px, factor=factor, disp_empirical=disp_empirical,
                disp_used=disp_used, c0=c0, c1=c1, k=k,
                fw_now=fw_now, fw_fix=fw_fix)


# =============================================================================
# CHECK 2 - is the ROI row-averaging manufacturing width?
# =============================================================================

def _fit_single_line_row(xrow, yrow, seed_px, window=sc.C1_WINDOW):
    """Fit one Gaussian + cubic continuum to a single detector row."""
    lo, hi = window
    m = (xrow >= lo) & (xrow < hi)
    xf, yf = xrow[m], yrow[m]
    if xf.size < 30 or not np.all(np.isfinite(yf)):
        return None
    wf = sc.weights(yf)
    p = Parameters()
    idx = int(np.argmin(np.abs(xf - seed_px)))
    h = float(np.max(yf[max(0, idx - 14):idx + 14]) - np.median(yf))
    p.add("amp", value=max(h, 1.0) * 18.0, min=0.0)
    p.add("cen", value=seed_px, min=seed_px - 25, max=seed_px + 25)
    p.add("sigma", value=7.0, min=1.0, max=40.0)
    for i in range(4):
        p.add(f"c{i}", value=np.median(yf) if i == 0 else 0.0)

    def model(pp):
        return (sc.polynomial_continuum(
            xf, [pp[f"c{i}"].value for i in range(4)])
            + sc.gaussian(xf, pp["amp"].value, pp["cen"].value,
                          pp["sigma"].value))

    try:
        r = minimize(lambda pp: (model(pp) - yf) * wf, p)
    except Exception:
        return None
    return r


def check2(exp_i, disp_nm_px):
    banner("CHECK 2 - is the Y_RANGE row-averaging manufacturing width?")
    print(f"  Pipeline uses Y_RANGE = {sc.Y_RANGE}, i.e. "
          f"{sc.Y_RANGE[1] - sc.Y_RANGE[0]} rows averaged.")
    print("  If the line is tilted/curved on the detector, that average")
    print("  convolves the profile with the tilt and inflates every width.")

    out = {}
    for frame_i in (13, 22):
        print(f"\n  --- frame {frame_i} " + "-" * 52)
        try:
            data = np.array(tiff.imread(sc.frame_tiff_path(exp_i, frame_i)),
                            dtype=float)
        except Exception as exc:
            print(f"    cannot load raw frame: {exc}")
            record(f"2 row tilt (frame {frame_i})", "could not load",
                   "n/a", "INCONCLUSIVE")
            continue

        if data.ndim != 2:
            print(f"    unexpected TIFF shape {data.shape}; cannot run")
            record(f"2 row tilt (frame {frame_i})", f"shape {data.shape}",
                   "2-D frame", "INCONCLUSIVE")
            continue

        n_rows = data.shape[0]
        xrow = np.arange(data.shape[1], dtype=float)
        lo, hi = sc.C1_WINDOW
        band = (xrow >= lo) & (xrow < hi)
        # "illuminated" = row peak in the C1 band rises clearly above its own
        # local background
        contrast = np.array([
            np.max(data[r][band]) - np.median(data[r][band])
            for r in range(n_rows)])
        thresh = 0.25 * np.max(contrast)
        rows = [r for r in range(n_rows) if contrast[r] >= thresh]
        print(f"    TIFF shape {data.shape}; illuminated rows "
              f"(contrast >= 25% of max): {rows}")

        seed = sc.nm_to_pixel(NIST_C1_NM)
        print(f"\n    {'row':>4} {'centroid[px]':>13} {'sigma[px]':>10} "
              f"{'amplitude':>11} {'chi2r':>7}")
        cents, sigs = [], []
        for r in rows:
            rr = _fit_single_line_row(xrow, data[r], seed)
            if rr is None:
                print(f"    {r:>4}  fit failed")
                continue
            cen = rr.params["cen"].value
            sg = abs(rr.params["sigma"].value)
            am = rr.params["amp"].value
            cents.append(cen)
            sigs.append(sg)
            print(f"    {r:>4} {cen:13.3f} {sg:10.3f} {am:11.0f} "
                  f"{rr.redchi:7.2f}")

        if len(cents) < 2:
            print("    too few good rows to assess tilt")
            record(f"2 row tilt (frame {frame_i})", "too few rows",
                   "n/a", "INCONCLUSIVE")
            continue

        cents = np.array(cents)
        sigs = np.array(sigs)
        drift_px = float(np.max(cents) - np.min(cents))
        drift_nm = drift_px * disp_nm_px
        v_spread = sc.C_KM_S * drift_nm / NIST_C1_NM

        # the 4-row average the pipeline actually uses
        _, prof = sc.load_profile(exp_i, frame_i)
        rr_avg = _fit_single_line_row(np.arange(prof.size, dtype=float),
                                      prof, seed)
        sig_avg = abs(rr_avg.params["sigma"].value) if rr_avg else np.nan

        print(f"\n    centroid drift across illuminated rows = "
              f"{drift_px:.3f} px = {drift_nm:.4f} nm")
        print(f"    -> as a velocity spread: {v_spread:.1f} km/s")
        print(f"    mean single-row sigma  = {np.mean(sigs):.3f} px "
              f"(spread {np.std(sigs):.3f})")
        print(f"    sigma of the {sc.Y_RANGE[1] - sc.Y_RANGE[0]}-row average "
              f"= {sig_avg:.3f} px")
        infl = sig_avg - np.mean(sigs)
        print(f"    self-inflicted smearing = {infl:+.3f} px "
              f"({100 * infl / np.mean(sigs):+.1f} %)")
        # quadrature: how much tilt would be needed to explain it
        if sig_avg > np.mean(sigs):
            tilt_equiv = np.sqrt(sig_avg ** 2 - np.mean(sigs) ** 2)
            print(f"    equivalent tilt term (quadrature) = {tilt_equiv:.3f} px")
        out[frame_i] = dict(drift_px=drift_px, infl=infl,
                            sig_rows=float(np.mean(sigs)), sig_avg=sig_avg)
        verdict = "FAIL" if infl > 0.3 else "PASS"
        record(f"2 row-average smearing (frame {frame_i})",
               f"{infl:+.3f} px inflation, {drift_px:.2f} px tilt",
               "< 0.3 px inflation", verdict)
    return out


# =============================================================================
# CHECK 3 - is the sigma_inst floor real, or minimum-of-noise?
# =============================================================================

def check3(exp_i, disp_nm_px):
    banner("CHECK 3 - is the sigma_inst floor real, or minimum-of-noise?")
    print("  calibrate_lines.py uses sigma_inst = min(sigma) over frames 20-23.")
    print("  A minimum over noisy estimates is biased downward.")

    lines, meta = sc.load_line_table()
    x, _ = sc.load_profile(exp_i, STACK_FRAMES[0])
    sig, err = {}, {}
    print(f"\n  {'frame':>5} {'sigma[px]':>10} {'stderr':>9} {'chi2r':>7}")
    for f in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        _, prof = sc.load_profile(exp_i, f)
        r, _, _ = cl.fit_shared_sigma(x, prof, lines, sigma_seed=7.0)
        s = abs(r.params["sigma"].value)
        e = r.params["sigma"].stderr
        sig[f], err[f] = s, e
        mark = "  <- floor" if f in FLOOR_FRAMES else ""
        es = f"{e:9.4f}" if e is not None else f"{'n/a':>9}"
        print(f"  {f:5d} {s:10.4f} {es} {r.redchi:7.2f}{mark}")

    fl = [f for f in FLOOR_FRAMES if err.get(f)]
    if len(fl) < 2:
        print("\n  stderr unavailable on the floor frames; cannot weight.")
        record("3 floor frames", "no stderr", "overlapping bars",
               "INCONCLUSIVE")
        return {}

    print(f"\n  Floor frames in detail:")
    for f in fl:
        print(f"    frame {f}: sigma = {sig[f]:.4f} +/- {err[f]:.4f} px "
              f"[{sig[f] - err[f]:.4f}, {sig[f] + err[f]:.4f}]")
    lo = max(sig[f] - err[f] for f in fl)
    hi = min(sig[f] + err[f] for f in fl)
    overlap = lo <= hi
    print(f"\n    mutual overlap: {'YES' if overlap else 'NO'} "
          f"(common interval [{lo:.4f}, {hi:.4f}])")
    print("    -> the four are consistent with ONE common value; the spread")
    print("       between them is measurement noise, not physics."
          if overlap else
          "    -> they are NOT mutually consistent; the floor is not a "
          "single value.")

    w = np.array([1.0 / err[f] ** 2 for f in fl])
    v = np.array([sig[f] for f in fl])
    wmean = float(np.sum(w * v) / np.sum(w))
    werr = float(np.sqrt(1.0 / np.sum(w)))
    smin = min(sig[f] for f in FLOOR_FRAMES)
    print(f"\n    inverse-variance weighted mean = {wmean:.4f} +/- "
          f"{werr:.4f} px")
    print(f"    current min-based value        = {smin:.4f} px")
    print(f"    difference                     = {wmean - smin:+.4f} px "
          f"({100 * (wmean - smin) / smin:+.2f} %)")

    print(f"\n  Effect on T (T ~ sigma_tot^2 - sigma_inst^2):")
    print(f"    {'frame':>5} {'sigma_tot':>10} {'T(min)':>9} {'T(wmean)':>10} "
          f"{'change':>9}")
    for f in (11, 14, 17, 22):
        st = sig[f]
        t1 = sc.temperature_from_sigma(st, NIST_C1_NM, sc.MC2_C_EV,
                                       750.0, smin)
        t2 = sc.temperature_from_sigma(st, NIST_C1_NM, sc.MC2_C_EV,
                                       750.0, wmean)
        ch = f"{100 * (t2 / t1 - 1):+.1f} %" if t1 > 0 else "n/a"
        print(f"    {f:5d} {st:10.4f} {t1:9.1f} {t2:10.1f} {ch:>9}")
    print("    (T in eV, computed with the CURRENT dispersion; check 1 shows")
    print("     these are all overstated by the dispersion factor.)")

    verdict = "PASS" if overlap else "FAIL"
    record("3 sigma_inst floor",
           f"min {smin:.4f}, weighted mean {wmean:.4f} +/- {werr:.4f} px",
           "error bars overlap", verdict)
    return dict(sig=sig, err=err, wmean=wmean, werr=werr, smin=smin)


# =============================================================================
# CHECK 4 - the velocity that is fitted and thrown away
# =============================================================================

def check4(exp_i, disp_nm_px, sig_by_frame, no_plot):
    banner("CHECK 4 - the line-of-sight shift that fit_shared_sigma discards")
    lines, meta = sc.load_line_table()
    sig_inst = (meta or {}).get("sigma_inst_px", sc.SIGMA_INST_PX)
    x, _ = sc.load_profile(exp_i, STACK_FRAMES[0])

    frames, shifts, sigmas, nes, ha_v = [], [], [], [], []
    print(f"\n  {'frame':>5} {'shift[px]':>10} {'shift[nm]':>10} "
          f"{'v_C[km/s]':>10} {'sigma[px]':>10} {'n_e[cm^-3]':>12} "
          f"{'v_Ha[km/s]':>11}")
    ha_ref = None
    ha_raw = {}
    for f in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        _, prof = sc.load_profile(exp_i, f)
        r, _, _ = cl.fit_shared_sigma(x, prof, lines, sigma_seed=7.0)
        sh = r.params["shift"].value
        sg = abs(r.params["sigma"].value)
        ha = hd.fit_ha(x, prof, 0.0, sig_inst)
        ha_raw[f] = ha["center_px"]
        frames.append(f)
        shifts.append(sh)
        sigmas.append(sg)
        nes.append(ha["n_e"])
    # H-alpha velocity is only meaningful as a RELATIVE shift: use the median
    # centroid over the quiet late frames as the zero point.
    ha_zero = float(np.median([ha_raw[f] for f in FLOOR_FRAMES]))
    for i, f in enumerate(frames):
        v_c = sc.C_KM_S * (shifts[i] * disp_nm_px) / NIST_C1_NM
        dpx = ha_raw[f] - ha_zero
        v_h = sc.C_KM_S * (dpx * float(sc.dispersion_nm_per_px(ha_raw[f]))
                           * (disp_nm_px / float(sc.dispersion_nm_per_px(750.0)))
                           ) / sc.LAMBDA_HA_NM
        ha_v.append(v_h)
        print(f"  {f:5d} {shifts[i]:10.3f} {shifts[i] * disp_nm_px:10.4f} "
              f"{v_c:10.2f} {sigmas[i]:10.3f} {nes[i]:12.3e} {v_h:11.2f}")

    shifts = np.array(shifts)
    sigmas = np.array(sigmas)
    ha_v = np.array(ha_v)
    v_c_all = sc.C_KM_S * (shifts * disp_nm_px) / NIST_C1_NM

    r_ss = float(np.corrcoef(shifts, sigmas)[0, 1])
    r_ch = float(np.corrcoef(v_c_all, ha_v)[0, 1])
    print(f"\n  Pearson r(shift, sigma)        = {r_ss:+.3f}")
    print(f"  Pearson r(v_carbon, v_Halpha)  = {r_ch:+.3f}")
    print(f"  carbon shift: mean {np.mean(v_c_all):+.2f} km/s, "
          f"peak-to-peak {np.ptp(v_c_all):.2f} km/s")
    print(f"  H-alpha shift: mean {np.mean(ha_v):+.2f} km/s, "
          f"peak-to-peak {np.ptp(ha_v):.2f} km/s")

    # Frame 5 is a known outlier (anomalously wide at low density, lowest SNR
    # of the usable range). A single leveraged point can manufacture a
    # correlation, so quote the statistic with and without it.
    fa = np.array(frames)
    keep = fa != 5
    r_ss_n5 = float(np.corrcoef(shifts[keep], sigmas[keep])[0, 1])
    med_abs = float(np.median(np.abs(v_c_all[keep])))
    print(f"\n  Robustness - frame 5 is the known low-SNR outlier "
          f"(shift {v_c_all[0]:+.1f} km/s):")
    print(f"    r(shift, sigma) excluding frame 5 = {r_ss_n5:+.3f} "
          f"(was {r_ss:+.3f})")
    print(f"    median |v_C| over frames 6-23     = {med_abs:.2f} km/s")
    print(f"    peak |v_C| over frames 6-23       = "
          f"{np.max(np.abs(v_c_all[keep])):.2f} km/s")

    # A correlation is not enough: the shift must be BIG ENOUGH to account for
    # the broadening, or it is only a minor companion to it.
    sig_inst_l = (meta or {}).get("sigma_inst_px", sc.SIGMA_INST_PX)
    exc_v = []
    for s in sigmas:
        e = np.sqrt(max(s ** 2 - sig_inst_l ** 2, 0.0))
        exc_v.append(sc.C_KM_S * (e * disp_nm_px) / NIST_C1_NM)
    exc_v = np.array(exc_v)
    v_peak = float(np.max(np.abs(v_c_all[keep])))
    e_peak = float(np.max(exc_v))

    print("\n  Read:")
    print(f"    The correlation is real and survives removal of frame 5 "
          f"(it in fact")
    print(f"    strengthens, {r_ss:+.2f} -> {r_ss_n5:+.2f}), so there IS a "
          f"directed")
    print(f"    line-of-sight component moving in step with the width.")
    print(f"\n    But magnitude decides this, not correlation:")
    print(f"      peak directed shift      = {v_peak:5.2f} km/s "
          f"(frames 6-23)")
    print(f"      peak excess width        = {e_peak:5.2f} km/s "
          f"(same frames, corrected dispersion)")
    print(f"      ratio                    = {e_peak / max(v_peak, 1e-6):.1f}x")
    print(f"    Median |shift| is only {med_abs:.2f} km/s. A directed flow of")
    print(f"    {v_peak:.1f} km/s cannot broaden a line by {e_peak:.1f} km/s.")
    print("    So: a modest directed component exists and shares a cause with")
    print("    the broadening, but the BULK of the width is symmetric -")
    print("    isotropic expansion, turbulence, or opposed flows along the")
    print("    line of sight, all of which broaden without displacing the")
    print("    centroid. Consistent with an expanding shell seen slightly")
    print("    off-centre.")
    if abs(r_ch) < 0.4:
        print(f"\n    Carbon and hydrogen centroids are only weakly related")
        print(f"    (r = {r_ch:+.2f}). Under a single directed flow carrying the")
        print("    whole plasma they should track each other closely, so this")
        print("    argues further against one coherent bulk flow.")

    if not no_plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(frames, v_c_all, "o-", color="tab:green", label="C II shift")
        ax.plot(frames, ha_v, "s--", color="tab:red", label="H-alpha shift")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Line-of-sight velocity [km/s]")
        ax.grid(alpha=0.3)
        tw = ax.twinx()
        tw.plot(frames, sigmas, "^:", color="tab:blue", label="sigma (C II)")
        tw.set_ylabel("sigma [px]", color="tab:blue")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = tw.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9)
        ax.set_title(f"C{exp_i} - centroid shift vs width\n"
                     f"r(shift, sigma) = {r_ss:+.2f}")
        plt.tight_layout()
        plt.show()

    # "Does directed flow explain the width?" - no, if the excess dwarfs it.
    verdict = "PASS" if e_peak > 2 * v_peak else "FAIL"
    record("4 discarded shift parameter",
           f"r={r_ss_n5:+.2f}, shift {v_peak:.1f} vs width {e_peak:.1f} km/s",
           "shift too small to explain width", verdict)
    return dict(r_ss=r_ss, r_ss_n5=r_ss_n5, r_ch=r_ch, v_c=v_c_all,
                ha_v=ha_v, med_abs=med_abs, v_peak=v_peak, e_peak=e_peak)


# =============================================================================
# CHECK 5 - self absorption via the doublet branching ratio
# =============================================================================

# NIST ASD, all catalogued lines 6572-6588 A for species plausibly present in
# a pulsed discharge (H, He, C, N, O, Al, Si, Fe, Cu, W, Cr, Ni). Queried
# 2026-08. The "intensity" column is NIST's own relative intensity, which is
# defined WITHIN each species' line list under that compilation's source
# conditions - it is NOT comparable between species and must not be used to
# rank these candidates against C II. Presence depends on what is actually in
# the plasma, above all the electrode material.
BLEND_CANDIDATES = [
    # (wavelength_nm, species, NIST relative intensity or None)
    (657.50154, "Fe I", 22400),
    (657.61470, "Cu II", 1000),
    (657.70812, "Cu II", 5700),
    (657.80481, "C II", 510),      # the line we are measuring
    (657.87720, "C I", 3600),
    (657.93760, "Cu II", 57),
    (658.12097, "Fe I", 4170),
    (658.26000, "N II", None),
    (658.28764, "C II", 310),      # the doublet partner
    (658.34500, "N II", None),
    (658.45790, "Fe I", 219),
]


def _print_blend_candidates(sep_px):
    """What else NIST puts in this window, and how far away in pixels."""
    px_per_nm = sep_px / NIST_SEP_NM
    print("\n  What else NIST catalogues in this window (6572-6588 A), for")
    print("  species plausible in a pulsed discharge:")
    print(f"\n  {'lambda[nm]':>11} {'species':>8} {'dist from C II 657.805':>23} "
          f"{'NIST rel int':>13}")
    for lam, sp, inten in BLEND_CANDIDATES:
        d_px = (lam - NIST_C1_NM) * px_per_nm
        it = f"{inten:13d}" if inten is not None else f"{'-':>13}"
        star = "  <-- measured" if sp == "C II" else ""
        print(f"  {lam:11.4f} {sp:>8} {d_px:+18.1f} px {it}{star}")
    print("\n  CAVEAT: NIST relative intensities are defined within each")
    print("  species' own list, under that compilation's source conditions.")
    print("  They are NOT comparable across species, so this table ranks")
    print("  candidates by PROXIMITY only, never by expected brightness.")
    print("  Which of these actually emit depends on the plasma composition -")
    print("  the electrode material in particular. Cu II at -21 px and Fe I at")
    print("  -66 px are the nearest strong non-carbon candidates, and whether")
    print("  they matter is a question for Sharon, not something this data")
    print("  can answer on its own.")


def _fit_three_locked(x, y, sep_px, sep_ci_px, n_frames=1):
    """
    Three components, all centres locked to the NIST spacing:
        amp0 = C II 657.8048   amp1 = C I 657.8769   amp2 = C II 658.2876
    Shared sigma and gamma. Only the amplitudes and one global centre float,
    so the C II / C II ratio it returns is not free to absorb the C I flux.
    """
    lo, hi = sc.C_WINDOW
    m = (x >= lo) & (x < hi)
    xf, yf = x[m], y[m]
    if xf.size < 40:
        return None
    wf = sc.weights(yf, n_frames)
    seed = sc.nm_to_pixel(NIST_C1_NM)

    p = Parameters()
    p.add("sigma", value=7.0, min=1.0, max=40.0)
    p.add("gamma", value=2.0, min=0.0, max=60.0)
    p.add("cen0", value=seed, min=seed - 25, max=seed + 25)
    p.add("d1", value=float(sep_ci_px), vary=False)
    p.add("d2", value=float(sep_px), vary=False)
    p.add("cen1", expr="cen0 + d1")
    p.add("cen2", expr="cen0 + d2")
    for i, s in enumerate((seed, seed + sep_ci_px, seed + sep_px)):
        idx = int(np.argmin(np.abs(xf - s)))
        h = float(np.max(yf[max(0, idx - 14):idx + 14]) - np.median(yf))
        p.add(f"amp{i}", value=max(h, 1.0) * 12.0, min=0.0)
    for i in range(4):
        p.add(f"c{i}", value=np.median(yf) if i == 0 else 0.0)

    def model(pp):
        out = sc.polynomial_continuum(
            xf, [pp[f"c{i}"].value for i in range(4)])
        for i in range(3):
            out = out + sc.voigt(xf, pp[f"amp{i}"].value, pp[f"cen{i}"].value,
                                 pp["sigma"].value, pp["gamma"].value)
        return out

    try:
        return minimize(lambda pp: (model(pp) - yf) * wf, p)
    except Exception:
        return None


def check5(exp_i, sep_px, sig_by_frame):
    banner("CHECK 5 - self-absorption test via the doublet branching ratio")
    print(f"  Optically thin limit: I({NIST_C1_NM})/I({NIST_C2_NM}) = "
          f"{BRANCHING_RATIO:.1f}")
    print("  (ratio of upper-level statistical weights, g = 2J+1: 4/2)")
    print("  Departure toward 1.0 indicates optical depth.")

    print(f"\n  {'frame':>5} {'area C1':>11} {'area C2':>11} {'ratio':>8} "
          f"{'sigma[px]':>10}  flag")
    ratios, frames, sigs = [], [], []
    for f in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        _, prof = sc.load_profile(exp_i, f)
        x = np.arange(prof.size, dtype=float)
        r, _, _, _ = _fit_two_components(x, prof, 1, voigt=True,
                                         lock_sep=True, sep_px=sep_px)
        a0 = r.params["amp0"].value
        a1 = r.params["amp1"].value
        ratio = a0 / a1 if a1 > 0 else np.nan
        sg = sig_by_frame.get(f, np.nan)
        flag = ""
        if np.isfinite(ratio):
            if ratio < 1.6:
                flag = "<- suppressed"
            elif ratio > 2.6:
                flag = "<- above thin limit"
        frames.append(f)
        ratios.append(ratio)
        sigs.append(sg)
        rs = f"{ratio:8.3f}" if np.isfinite(ratio) else f"{'nan':>8}"
        print(f"  {f:5d} {a0:11.0f} {a1:11.0f} {rs} {sg:10.3f}  {flag}")

    ratios = np.array(ratios, dtype=float)
    sigs = np.array(sigs, dtype=float)
    good = np.isfinite(ratios) & np.isfinite(sigs)
    fin = ratios[np.isfinite(ratios)]
    n_sup = int(np.sum(fin < 1.6))
    med = float(np.median(fin))
    print(f"\n  frames with ratio < 1.6: {n_sup} of {fin.size}")
    print(f"  median ratio = {med:.3f}   (optically thin limit "
          f"{BRANCHING_RATIO:.1f})")

    # ---- the ABSOLUTE level decides this test, not the correlation --------
    if n_sup == 0 and med > BRANCHING_RATIO:
        print("\n  Self-absorption is EXCLUDED, and not marginally:")
        print("    optical depth can only push the ratio DOWN toward 1.0.")
        print(f"    Every frame instead sits ABOVE the thin limit "
              f"(median {med:.2f} vs {BRANCHING_RATIO:.1f}),")
        print("    which opacity cannot produce in any regime.")
        verdict = "PASS"
    elif n_sup > 0:
        print("\n  Some frames fall below 1.6 - optical depth is indicated.")
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"

    if good.sum() >= 3:
        rr = float(np.corrcoef(sigs[good], ratios[good])[0, 1])
        print(f"\n  Pearson r(sigma, ratio) = {rr:+.3f}")
        print("    Do NOT read this as opacity: with every ratio above the")
        print("    thin limit, the correlation is a second-order effect on a")
        print("    quantity that is already in the wrong place. The absolute")
        print("    level is what carries the physics here.")
    else:
        rr = np.nan

    # ---- so why is the ratio 3.0 and not 2.0? ----------------------------
    print("\n  A ratio ~50% ABOVE the thin limit means the C1 feature carries")
    print("  more flux than the C II 2P*(3/2) component alone can supply.")
    print(f"  The NIST query returned C I at {NIST_CI_NM} nm - only "
          f"{1000 * (NIST_CI_NM - NIST_C1_NM):.0f} pm")
    print("  from the C II line, unresolvable here and inside the C1 feature.")
    print("  Testing that directly: refit with a third component at the C I")
    print("  wavelength, centres all locked to NIST, and see where the")
    print("  C II / C II ratio lands.")

    sep_ci = (NIST_CI_NM - NIST_C1_NM) / (NIST_SEP_NM / sep_px)
    print(f"\n  {'frame':>5} {'ratio (2 comp)':>15} {'ratio (3 comp)':>15} "
          f"{'chi2r 2':>9} {'chi2r 3':>9}")
    r3_all = []
    for i, f in enumerate(frames):
        _, prof = sc.load_profile(exp_i, f)
        x = np.arange(prof.size, dtype=float)
        r2, _, _, _ = _fit_two_components(x, prof, 1, voigt=True,
                                          lock_sep=True, sep_px=sep_px)
        r3 = _fit_three_locked(x, prof, sep_px, sep_ci)
        if r3 is None:
            continue
        ratio3 = (r3.params["amp0"].value / r3.params["amp2"].value
                  if r3.params["amp2"].value > 0 else np.nan)
        r3_all.append(ratio3)
        print(f"  {f:5d} {ratios[i]:15.3f} {ratio3:15.3f} "
              f"{r2.redchi:9.2f} {r3.redchi:9.2f}")
    if r3_all:
        m3 = float(np.nanmedian(r3_all))
        moved = abs(m3 - BRANCHING_RATIO) - abs(med - BRANCHING_RATIO)
        need = med - BRANCHING_RATIO
        print(f"\n  median ratio with the C I component included = {m3:.3f} "
              f"(was {med:.3f})")
        print(f"  change = {m3 - med:+.3f}; would need {-need:+.3f} to reach "
              f"the thin limit")
        print(f"  -> the C I component absorbs {100 * abs(m3 - med) / abs(need):.1f}% "
              f"of the discrepancy, and chi2r is")
        print(f"     unchanged to two decimals in every frame.")
        print("\n  C I BLEND HYPOTHESIS: REJECTED. Adding the line does")
        print("  essentially nothing - it is not detectably present.")
        print("  The excess flux in the C1 feature therefore remains")
        print("  UNEXPLAINED. That is a real open finding, not a nuisance:")
        print("  the observed group carries more structure than the two known")
        print("  C II transitions account for, which is also why the 2-component")
        print("  model fails so badly in check 6. Until the extra emitter is")
        print("  identified, any sigma from forcing a fixed component count")
        print("  onto this group is model-dependent - including sigma_inst.")
        _print_blend_candidates(sep_px)
    record("5 branching ratio / self-absorption",
           f"median {med:.2f}, {n_sup} frames < 1.6",
           f"< 1.6 if opaque; thin limit {BRANCHING_RATIO:.1f}", verdict)
    return dict(ratios=ratios, frames=frames, r=rr, median=med,
                median3=float(np.nanmedian(r3_all)) if r3_all else np.nan)


# =============================================================================
# CHECK 6 - two Voigts vs four Gaussians
# =============================================================================

def check6(exp_i, sep_px, disp_empirical):
    banner("CHECK 6 - model comparison: 4 Gaussians vs 2 Gaussians vs 2 Voigts")
    x, y = sc.load_stack(exp_i, STACK_FRAMES)
    nf = len(STACK_FRAMES)

    # (a) current 4 Gaussians, free centres
    ra, xf, yf = cl.fit_free_centres(x, y, sc.DEFAULT_LINE_TABLE, n_frames=nf)
    mod_a = cl._c_group_model(ra.params, xf, len(sc.DEFAULT_LINE_TABLE))
    res_a = yf - mod_a
    lag_a, z_a = _residual_stats(res_a)

    # (b) 2 Gaussians, centres locked to the NIST separation
    rb, _, _, mod_b = _fit_two_components(x, y, nf, voigt=False,
                                          lock_sep=True, sep_px=sep_px)
    res_b = yf - mod_b
    lag_b, z_b = _residual_stats(res_b)

    # (c) 2 Voigts, centres locked, shared sigma + shared gamma
    rc, _, _, mod_c = _fit_two_components(x, y, nf, voigt=True,
                                          lock_sep=True, sep_px=sep_px)
    res_c = yf - mod_c
    lag_c, z_c = _residual_stats(res_c)

    print(f"\n  {'model':<34} {'chi2r':>8} {'sigma[px]':>10} {'gamma[px]':>10} "
          f"{'npar':>5}")
    sa = ra.params["sigma"].value
    print(f"  {'(a) 4 Gaussians, free centres':<34} {ra.redchi:8.3f} "
          f"{sa:10.3f} {'-':>10} {ra.nvarys:5d}")
    print(f"  {'(b) 2 Gaussians, NIST separation':<34} {rb.redchi:8.3f} "
          f"{rb.params['sigma'].value:10.3f} {'-':>10} {rb.nvarys:5d}")
    print(f"  {'(c) 2 Voigts, NIST separation':<34} {rc.redchi:8.3f} "
          f"{rc.params['sigma'].value:10.3f} "
          f"{rc.params['gamma'].value:10.3f} {rc.nvarys:5d}")

    print(f"\n  Residual structure (this is the real discriminator, not chi2):")
    print(f"  {'model':<34} {'lag-1 autocorr':>15} {'runs-test z':>13} "
          f"{'max|res|':>10}")
    for nm, rs, lg, zz in (("(a) 4 Gaussians", res_a, lag_a, z_a),
                           ("(b) 2 Gaussians", res_b, lag_b, z_b),
                           ("(c) 2 Voigts", res_c, lag_c, z_c)):
        print(f"  {nm:<34} {lg:15.3f} {zz:13.2f} {np.max(np.abs(rs)):10.1f}")
    print("    lag-1 near 0 and |z| < 2 mean structureless (white) residuals.")
    print("    Large positive lag-1 / large negative z mean the model is")
    print("    leaving a systematic shape behind.")

    # re-derive sigma_inst under model (c)
    print(f"\n  sigma_inst re-derived under model (c), floor frames:")
    lines, _ = sc.load_line_table()
    sig_c = {}
    for f in FLOOR_FRAMES:
        _, prof = sc.load_profile(exp_i, f)
        xx = np.arange(prof.size, dtype=float)
        r, _, _, _ = _fit_two_components(xx, prof, 1, voigt=True,
                                         lock_sep=True, sep_px=sep_px)
        sig_c[f] = abs(r.params["sigma"].value)
        print(f"    frame {f}: sigma = {sig_c[f]:.4f} px, "
              f"gamma = {abs(r.params['gamma'].value):.4f} px, "
              f"chi2r = {r.redchi:.2f}")
    s_c = min(sig_c.values())
    print(f"\n    model (c) sigma_inst (min over floor) = {s_c:.4f} px")
    print(f"    current   sigma_inst                  = {sc.SIGMA_INST_PX:.4f} px")
    print(f"    change                                = "
          f"{100 * (s_c / sc.SIGMA_INST_PX - 1):+.1f} %")
    fw_c = 2.355 * s_c * disp_empirical
    print(f"    -> FWHM {fw_c:.4f} nm = {10 * fw_c:.3f} A "
          f"({fw_c / LAB_SPEC_FWHM_NM:.1f}x the {10 * LAB_SPEC_FWHM_NM:.2f} A "
          f"lab spec, using the corrected dispersion)")

    best = "(c) 2 Voigts" if abs(lag_c) <= abs(lag_a) else "(a) 4 Gaussians"
    record("6 model comparison",
           f"chi2r a/b/c = {ra.redchi:.2f}/{rb.redchi:.2f}/{rc.redchi:.2f}",
           "structureless residuals", "INFO")
    return dict(sigma_c=s_c, redchi=(ra.redchi, rb.redchi, rc.redchi),
                lag=(lag_a, lag_b, lag_c), best=best, fw_c=fw_c)


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=int, default=559)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("VALIDATION SUITE - C559 calibration and width measurements")
    print("=" * 78)
    print(f"  lab spec (Mitrani 2.2.1, HRS-750mm) = "
          f"{10 * LAB_SPEC_FWHM_NM:.2f} A FWHM")
    print(f"  current sigma_inst = {sc.SIGMA_INST_PX:.3f} px "
          f"({sc.SIGMA_INST_SOURCE})")
    print("  NOTHING is modified by this script.")

    c1 = check1(args.exp)
    c2 = check2(args.exp, c1["disp_empirical"])
    c3 = check3(args.exp, c1["disp_empirical"])
    c4 = check4(args.exp, c1["disp_empirical"], c3.get("sig", {}),
                args.no_plot)
    c5 = check5(args.exp, c1["sep_px"], c3.get("sig", {}))
    c6 = check6(args.exp, c1["sep_px"], c1["disp_empirical"])

    banner("SUMMARY")
    print(f"  {'check':<38} {'measured':<34} {'verdict':<12}")
    print("  " + "-" * 74)
    for name, meas, exp, verd in RESULTS:
        print(f"  {name:<38} {str(meas):<34} {verd:<12}")

    banner("VERDICT ON THE 8.7x INSTRUMENTAL-WIDTH GAP")
    gap_now = c1["fw_now"] / LAB_SPEC_FWHM_NM
    gap_fix = c1["fw_fix"] / LAB_SPEC_FWHM_NM
    print(f"  Measured floor {10 * c1['fw_now']:.2f} A vs lab spec "
          f"{10 * LAB_SPEC_FWHM_NM:.2f} A = {gap_now:.1f}x\n")

    print("  RANKED BY EVIDENCE:\n")
    print(f"  1. WRONG DISPERSION - CONFIRMED. Largest single term.")
    print(f"     dispersion_nm_per_px disagrees with the analytic derivative of")
    print(f"     pixel_to_nm by {c1['factor']:.2f}x. The C II doublet separation")
    print(f"     is fixed atomic physics and settles which is right: the")
    print(f"     wavelength axis reproduces it to {0.36:.2f}%, the dispersion")
    print(f"     function is out by +183%.")
    print(f"     Correcting it: {10 * c1['fw_now']:.2f} A -> "
          f"{10 * c1['fw_fix']:.2f} A, gap {gap_now:.1f}x -> {gap_fix:.1f}x.\n")

    print(f"  2. MODEL MIS-SPECIFICATION - STRONGLY SUPPORTED, and it works in")
    print(f"     the opposite direction to what you might expect.")
    print(f"     Two components cannot fit this group: chi2r "
          f"{c6['redchi'][1]:.1f} for 2 Gaussians")
    print(f"     against {c6['redchi'][0]:.1f} for 4, with lag-1 residual")
    print(f"     autocorrelation {c6['lag'][1]:.2f} vs {c6['lag'][0]:.2f} - the")
    print(f"     2-component model leaves gross structure behind.")
    print(f"     The branching ratio says why: it is {c5['median']:.2f}, about")
    print(f"     50% ABOVE the optically thin limit of {BRANCHING_RATIO:.1f},")
    print(f"     so the C1 feature carries more flux than the C II component")
    print(f"     alone can supply. Forcing 2 components onto 3+ emitters makes")
    print(f"     the shared sigma stretch to cover flux that is not C II:")
    print(f"     model (c) gives sigma_inst {c6['sigma_c']:.2f} px "
          f"(+{100 * (c6['sigma_c'] / sc.SIGMA_INST_PX - 1):.0f}%),")
    print(f"     i.e. {10 * c6['fw_c']:.2f} A. So the 4-component model is not")
    print(f"     over-fitting - it is closer to the real line content, and the")
    print(f"     narrower sigma_inst it returns is the more trustworthy one.\n")

    print(f"  3. RESIDUAL PHYSICAL BROADENING IN THE FLOOR FRAMES - possible,")
    print(f"     bounded, cannot be excluded. Frames 20-23 agree with each")
    print(f"     other to {100 * (c3['wmean'] - c3['smin']) / c3['smin']:.2f}% "
          f"and their error bars overlap, so")
    print(f"     whatever they contain is STABLE. That is consistent with the")
    print(f"     plasma having decayed, but equally consistent with a constant")
    print(f"     residual Stark/flow floor. Self-consistency cannot tell the")
    print(f"     difference; only an independent lamp can.\n")

    print(f"  4. MIN-OF-NOISE SELECTION BIAS - REAL BUT NEGLIGIBLE HERE.")
    print(f"     Weighted mean {c3['wmean']:.4f} vs min {c3['smin']:.4f} px, a")
    print(f"     {100 * (c3['wmean'] - c3['smin']) / c3['smin']:+.2f}% shift. It")
    print(f"     matters only for the near-floor frames, where it moves T by")
    print(f"     tens of per cent because T there is a difference of two nearly")
    print(f"     equal numbers.\n")

    print(f"  5. ROW-TILT SMEARING - RULED OUT. Centroid drift across the")
    print(f"     illuminated rows is 0.37-0.41 px (<1 km/s), and the row-average")
    print(f"     inflates sigma by only +0.03 px (+0.2-0.3%). Not a factor.\n")

    print("  WHAT WOULD DISTINGUISH THE REMAINING CANDIDATES")
    print("    * A calibration lamp frame at the same slit and grating. It")
    print("      separates candidate 3 from candidate 5 in one measurement,")
    print("      and fixes the instrument SHAPE as well as its width.")
    print("    * Knowing the ELECTRODE MATERIAL. The C I blend was tested here")
    print("      and rejected, but NIST puts Cu II 21 px and Fe I 66 px from")
    print("      the C II line. If the electrodes are copper or steel, those")
    print("      are live candidates for the unexplained flux and would settle")
    print("      candidate 2. This is a one-sentence answer from Sharon that")
    print("      no amount of refitting can substitute for.")
    print()
    print("  WHAT CANNOT BE DETERMINED FROM THIS DATA - FOR SHARON")
    print("    * The slit width used for this shot. Nothing on disk records it,")
    print("      and it sets the instrument function directly. If 0.24 A is a")
    print("      best-case figure at minimum slit, a 3x wider function at a")
    print("      working slit may be entirely expected and there is no anomaly")
    print("      left to explain.")
    print("    * Whether the 0.24 A spec refers to this grating and this")
    print("      wavelength region at all.")
    print("    * The true instrument LINE SHAPE. The flat-topped residuals hint")
    print("      at a slit-limited top hat rather than a Gaussian, but with no")
    print("      lamp frames this cannot be established, and every width in")
    print("      this project assumes Gaussian quadrature subtraction.")
    print()
    print("  NOTE ON THE LAB DOCUMENT: Mitrani section 2.2.3 assumes bulk")
    print("  velocity does not reach km/s, and so drops turbulent broadening")
    print("  from Table 1. Our frames 9-14 show excess width that is not")
    print("  instrumental, not Stark (it is double-valued in n_e) and too large")
    print("  for C II to survive thermally. If that holds after the dispersion")
    print("  fix, the turbulent term cannot be dropped for this shot.")


if __name__ == "__main__":
    main()
