"""
Stage 2 - electron density from the H-alpha Stark width, frame by frame.

The H-alpha Voigt is fitted with its GAUSSIAN WIDTH HELD FIXED at
sqrt(sigma_inst^2 + sigma_thermal_H(T)^2) and only the Lorentzian gamma free.
Letting sigma and gamma both float is degenerate - it is the exact failure mode
Mitrani section 3.1 warns about, and it is what the old scripts did.

The temperature that sets the (small) thermal part comes from the C II lines,
so the two stages are iterated; in practice n_e moves by well under a percent
between iterations because at these densities H-alpha is overwhelmingly Stark
dominated.

    python ha_density.py            # table + plot for all frames
    python ha_density.py --frame 12 # single-frame diagnostic plot
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize

import spectro_core as sc


# --- continuum order --------------------------------------------------------
# A cubic, not a straight line. The H-alpha window is 580 px wide and the
# plasma continuum is visibly curved: with a linear continuum the fit leaves
# systematic +-30..50 count residuals and reduced chi-square reaches 13 at
# frame 10. Going quadratic drops that to 4.9, and moves n_e at frame 10 from
# 1.57e17 to 7.3e16 - which also removes an unphysical 5x jump between frames
# 9 and 10 and makes the density rise monotonic. Quadratic and cubic agree on
# n_e to better than 0.5%, so the choice is converged; cubic is kept because it
# gives slightly lower chi-square and matches the C-window continuum order.
HA_CONTINUUM_ORDER = 4   # number of coefficients: c0 + c1 t + c2 t^2 + c3 t^3


def _model(params, x):
    """H-alpha Voigt on a cubic continuum."""
    coeffs = [params[f"c{i}"].value for i in range(HA_CONTINUUM_ORDER)]
    return (sc.voigt(x, params["amp"].value, params["cen"].value,
                     params["sigma"].value, params["gamma"].value)
            + sc.polynomial_continuum(x, coeffs, x_ref=420.0))


def fit_ha(x, profile, T_eV=0.0, sigma_inst_px=None):
    """
    Fit H-alpha in one frame and return a result dict.

    T_eV only sets the fixed Gaussian width; pass the current best C II
    temperature, or 0 to use the instrumental width alone.
    """
    if sigma_inst_px is None:
        sigma_inst_px = sc.SIGMA_INST_PX

    lo, hi = sc.HA_WINDOW
    m = (x >= lo) & (x < hi)
    xf, yf = x[m], profile[m]
    wf = sc.weights(yf)

    px_ha = sc.nm_to_pixel(sc.LAMBDA_HA_NM)
    sigma_th = sc.thermal_sigma_px(T_eV, sc.LAMBDA_HA_NM, sc.MC2_H_EV, px_ha)
    sigma_fixed = float(np.sqrt(sigma_inst_px ** 2 + sigma_th ** 2))

    peak = float(np.max(yf))
    floor = float(np.median(yf[:40]))

    params = Parameters()
    # amplitude is an AREA; height ~ area / (sigma*sqrt(2pi)) so seed accordingly
    params.add("amp", value=max(peak - floor, 1.0) * 60.0, min=0.0)
    params.add("cen", value=px_ha, min=px_ha - 40, max=px_ha + 40)
    params.add("sigma", value=sigma_fixed, vary=False)   # <- the key constraint
    params.add("gamma", value=20.0, min=0.0, max=400.0)
    for i in range(HA_CONTINUUM_ORDER):
        params.add(f"c{i}", value=floor if i == 0 else 0.0)

    result = minimize(lambda p: (_model(p, xf) - yf) * wf, params)

    gamma = abs(result.params["gamma"].value)
    cen = result.params["cen"].value
    gamma_err = result.params["gamma"].stderr

    n_e = sc.n_e_from_ha_stark(gamma, cen)
    # n_e ~ gamma^1.471  =>  dn/n = 1.471 * dgamma/gamma
    n_e_err = (np.nan if gamma_err is None or gamma <= 0
               else n_e * 1.471 * gamma_err / gamma)

    return {
        "frame": None,
        "n_e": n_e,
        "n_e_err": n_e_err,
        "gamma_px": gamma,
        "gamma_nm": gamma * sc.dispersion_nm_per_px(cen),
        "stark_fwhm_nm": 2 * gamma * sc.dispersion_nm_per_px(cen),
        "center_px": cen,
        "center_nm": float(sc.pixel_to_nm(cen)),
        "sigma_fixed_px": sigma_fixed,
        "redchi": result.redchi,
        "params": result.params,
        "x": xf,
        "y": yf,
        "model": _model(result.params, xf),
    }


def run_all(exp_i, T_by_frame=None, sigma_inst_px=None, verbose=True):
    """Fit every usable frame. T_by_frame maps frame -> T_eV for the fixed sigma."""
    T_by_frame = T_by_frame or {}
    out = {}
    if verbose:
        print(f"{'frame':>5}  {'n_e [cm^-3]':>12} {'+/-':>11}  "
              f"{'Stark FWHM':>10}  {'centre [nm]':>11}  {'chi2r':>6}")
    for frame_i in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        x, profile = sc.load_profile(exp_i, frame_i)
        res = fit_ha(x, profile, T_by_frame.get(frame_i, 0.0), sigma_inst_px)
        res["frame"] = frame_i
        out[frame_i] = res
        if verbose:
            print(f"{frame_i:5d}  {res['n_e']:12.4e} {res['n_e_err']:11.2e}  "
                  f"{res['stark_fwhm_nm']:9.4f}  {res['center_nm']:11.4f}  "
                  f"{res['redchi']:6.2f}")
    return out


def plot_frame(res, exp_i):
    """Diagnostic plot for a single frame: data, fit, continuum, residual."""
    lam = sc.pixel_to_nm(res["x"])
    cont = sc.polynomial_continuum(
        res["x"], [res["params"][f"c{i}"].value
                   for i in range(HA_CONTINUUM_ORDER)], x_ref=420.0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(lam, res["y"], color="0.6", lw=2, label="data")
    axes[0].plot(lam, res["model"], "k--", lw=2, label="Voigt + continuum")
    axes[0].plot(lam, cont, ":", color="0.4", lw=1, label="continuum")
    axes[0].set_ylabel("Intensity [counts]")
    axes[0].set_title(
        f"C{exp_i} frame {res['frame']} - H$\\alpha$\n"
        f"$\\sigma$ fixed at {res['sigma_fixed_px']:.2f} px, "
        f"$\\gamma$ = {res['gamma_px']:.2f} px "
        f"(Stark FWHM {res['stark_fwhm_nm']:.3f} nm)  ->  "
        f"$n_e$ = {res['n_e']:.3e} cm$^{{-3}}$")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(lam, res["y"] - res["model"], color="crimson", lw=1)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("Wavelength [nm]")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=int, default=559)
    ap.add_argument("--frame", type=int, default=None,
                    help="plot this single frame instead of the whole run")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    lines, meta = sc.load_line_table()
    sigma_inst = (meta or {}).get("sigma_inst_px", sc.SIGMA_INST_PX)

    print("=" * 74)
    print("STAGE 2 - n_e from the H-alpha Stark width")
    print("=" * 74)
    print(f"instrumental sigma = {sigma_inst:.3f} px "
          f"({sc.SIGMA_INST_SOURCE})\n")

    if args.frame is not None:
        x, profile = sc.load_profile(args.exp, args.frame)
        res = fit_ha(x, profile, 0.0, sigma_inst)
        res["frame"] = args.frame
        print(f"frame {args.frame}:  n_e = {res['n_e']:.4e} "
              f"+/- {res['n_e_err']:.2e} cm^-3   "
              f"Stark FWHM = {res['stark_fwhm_nm']:.4f} nm   "
              f"chi2r = {res['redchi']:.2f}")
        if not args.no_plot:
            plot_frame(res, args.exp)
        return

    results = run_all(args.exp, sigma_inst_px=sigma_inst)

    if not args.no_plot:
        frames = sorted(results)
        n_e = np.array([results[f]["n_e"] for f in frames])
        err = np.array([results[f]["n_e_err"] for f in frames])
        plt.figure(figsize=(9, 5.5))
        plt.errorbar(frames, n_e, yerr=err, marker="o", capsize=3, color="tab:blue")
        plt.xlabel("Frame number")
        plt.ylabel("Electron density $n_e$ [cm$^{-3}$]")
        plt.title(f"C{args.exp} - $n_e$ from H$\\alpha$ Stark broadening")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
