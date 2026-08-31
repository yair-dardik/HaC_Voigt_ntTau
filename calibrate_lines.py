"""
Stage 1 - establish the C II line table and the instrumental width. Run once.

Fits a high-SNR stack of frames with four free-centre components sharing one
Gaussian width, then measures the shared width frame by frame to find its floor.

This answers questions.txt items 3 and 4 with numbers:
  3. the C centres are not the assumed lambdas - here are the measured ones
  4. C1 IS two peaks (and so is C2)

Writes line_table.json, which everything downstream reads.

    python calibrate_lines.py
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize

import spectro_core as sc


STACK_FRAMES = range(12, 24)      # brightest C II frames
FLOOR_FRAMES = range(20, 24)      # late frames, plasma decayed, Stark smallest


def _c_group_model(params, x, n_lines):
    """Four Gaussians sharing one sigma, on a cubic continuum."""
    sigma = params["sigma"].value
    model = sc.polynomial_continuum(
        x, [params[f"c{i}"].value for i in range(4)])
    for i in range(n_lines):
        model = model + sc.gaussian(x, params[f"amp{i}"].value,
                                    params[f"cen{i}"].value, sigma)
    return model


def _residual(params, x, y, w, n_lines):
    return (_c_group_model(params, x, n_lines) - y) * w


def fit_free_centres(x, y, seed_lines, n_frames=1):
    """Fit the C group with free centres and one shared width."""
    lo, hi = sc.C_WINDOW
    m = (x >= lo) & (x < hi)
    xf, yf = x[m], y[m]
    wf = sc.weights(yf, n_frames)

    params = Parameters()
    params.add("sigma", value=7.0, min=1.0, max=30.0)
    for i, line in enumerate(seed_lines):
        idx = int(np.argmin(np.abs(xf - line["pixel"])))
        height = float(np.max(yf[max(0, idx - 12):idx + 12]) - np.median(yf))
        params.add(f"amp{i}", value=max(height, 1.0) * 18.0, min=0.0)
        params.add(f"cen{i}", value=line["pixel"],
                   min=line["pixel"] - 12, max=line["pixel"] + 12)
    for i in range(4):
        params.add(f"c{i}", value=np.median(yf) if i == 0 else 0.0)

    result = minimize(_residual, params, args=(xf, yf, wf, len(seed_lines)))
    return result, xf, yf


def fit_shared_sigma(x, y, lines, sigma_seed=7.0):
    """Fit one frame with centres locked (one global shift free) -> shared sigma."""
    lo, hi = sc.C_WINDOW
    m = (x >= lo) & (x < hi)
    xf, yf = x[m], y[m]
    wf = sc.weights(yf)

    params = Parameters()
    params.add("sigma", value=sigma_seed, min=1.0, max=30.0)
    params.add("shift", value=0.0, min=-12.0, max=12.0)
    for i, line in enumerate(lines):
        idx = int(np.argmin(np.abs(xf - line["pixel"])))
        height = float(np.max(yf[max(0, idx - 12):idx + 12]) - np.median(yf))
        params.add(f"amp{i}", value=max(height, 1.0) * 18.0, min=0.0)
        params.add(f"cen{i}", value=line["pixel"], vary=False)
        params.add(f"cen{i}_shifted", expr=f"cen{i} + shift")
    for i in range(4):
        params.add(f"c{i}", value=np.median(yf) if i == 0 else 0.0)

    def residual(p):
        sigma = p["sigma"].value
        model = sc.polynomial_continuum(
            xf, [p[f"c{i}"].value for i in range(4)])
        for i in range(len(lines)):
            model = model + sc.gaussian(xf, p[f"amp{i}"].value,
                                        p[f"cen{i}_shifted"].value, sigma)
        return (model - yf) * wf

    return minimize(residual, params), xf, yf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=int, default=559)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("STAGE 1 - C II line table and instrumental width")
    print("=" * 74)

    # --- line positions from the stack ---
    frames = list(STACK_FRAMES)
    x, y = sc.load_stack(args.exp, frames)
    print(f"\nStacked frames {frames[0]}-{frames[-1]} ({len(frames)} frames)")

    result, xf, yf = fit_free_centres(x, y, sc.DEFAULT_LINE_TABLE,
                                      n_frames=len(frames))
    sigma_stack = result.params["sigma"].value

    lines = []
    print("\nFitted components (free centres, one shared sigma):")
    print("  label   pixel        lambda [nm]     area       stderr(px)")
    for i, seed in enumerate(sc.DEFAULT_LINE_TABLE):
        cen = result.params[f"cen{i}"].value
        err = result.params[f"cen{i}"].stderr
        lam = float(sc.pixel_to_nm(cen))
        lines.append({"label": seed["label"], "pixel": float(cen),
                      "lambda_nm": lam, "group": seed["group"]})
        print(f"  {seed['label']:6s} {cen:8.2f}   {lam:12.4f}  "
              f"{result.params[f'amp{i}'].value:9.0f}   "
              f"{'n/a' if err is None else f'{err:.3f}'}")

    disp = sc.dispersion_nm_per_px(800.0)
    print(f"\n  shared sigma = {sigma_stack:.3f} px "
          f"= {sigma_stack * disp:.4f} nm  (FWHM {2.355 * sigma_stack * disp:.4f} nm)")
    print(f"  reduced chi-square = {result.redchi:.3f}")

    # --- instrumental width floor from the late frames ---
    print(f"\nShared sigma per frame (floor search over frames "
          f"{FLOOR_FRAMES.start}-{FLOOR_FRAMES.stop - 1}):")
    sigmas = {}
    for frame_i in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        _, prof = sc.load_profile(args.exp, frame_i)
        res, _, _ = fit_shared_sigma(x, prof, lines, sigma_seed=sigma_stack)
        sigma = abs(res.params["sigma"].value)
        sigmas[frame_i] = sigma
        mark = "  <- floor window" if frame_i in FLOOR_FRAMES else ""
        print(f"  frame {frame_i:2d}  sigma = {sigma:7.3f} px  "
              f"FWHM = {2.355 * sigma * disp:.4f} nm   "
              f"chi2r = {res.redchi:5.2f}{mark}")

    floor_frame = min(FLOOR_FRAMES, key=lambda f: sigmas[f])
    sigma_inst = sigmas[floor_frame]
    print(f"\n  instrumental sigma (floor) = {sigma_inst:.3f} px "
          f"= {sigma_inst * disp:.4f} nm  (FWHM {2.355 * sigma_inst * disp:.4f} nm), "
          f"from frame {floor_frame}")
    # The old assumption inst_fwhm_Ha = 0.05 nm is an H-alpha number, so convert
    # it at the H-alpha pixel with the CORRECTED dispersion. Under the old buggy
    # dispersion this came out 1.63 px and the ratio looked like 4.2x; both were
    # artefacts of that bug.
    disp_ha = float(sc.dispersion_nm_per_px(sc.nm_to_pixel(sc.LAMBDA_HA_NM)))
    old_sigma_px = 0.05 / (2.35482 * disp_ha)
    print(f"  the old constant was inst_sigma = {old_sigma_px:.3f} px "
          f"-> too small by {sigma_inst / old_sigma_px:.2f}x")
    print(f"  (under the pre-fix dispersion this printed 1.627 px and 4.2x;")
    print(f"   both were products of the dispersion bug, not real findings)")
    print("\n  NOTE: this is an UPPER BOUND on the instrumental width, so every")
    print("  temperature derived from it is a LOWER bound. Replace it with a")
    print("  calibration-lamp measurement as soon as one exists.")

    path = sc.save_line_table(lines, extra={
        "sigma_inst_px": float(sigma_inst),
        "sigma_inst_source": f"data floor, frame {floor_frame} of {args.exp}",
        "sigma_stack_px": float(sigma_stack),
        "stack_frames": [int(f) for f in frames],
        "sigma_per_frame": {str(k): float(v) for k, v in sigmas.items()},
    })
    print(f"\nWrote {path}")

    if not args.no_plot:
        model = _c_group_model(result.params, xf, len(lines))
        cont = sc.polynomial_continuum(
            xf, [result.params[f"c{i}"].value for i in range(4)])
        lam = sc.pixel_to_nm(xf)

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        axes[0].plot(lam, yf, color="0.6", lw=2, label="stacked data")
        axes[0].plot(lam, model, "k--", lw=2, label="fit (4 components)")
        for i, line in enumerate(lines):
            comp = cont + sc.gaussian(xf, result.params[f"amp{i}"].value,
                                      result.params[f"cen{i}"].value, sigma_stack)
            axes[0].plot(lam, comp, lw=1.2, alpha=0.8,
                         label=f"{line['label']} @ {line['lambda_nm']:.4f} nm")
        axes[0].plot(lam, cont, ":", color="0.4", lw=1, label="continuum")
        axes[0].set_ylabel("Intensity [counts]")
        axes[0].set_title(f"C{args.exp} C II group - frames "
                          f"{frames[0]}-{frames[-1]} stacked, shared sigma = "
                          f"{sigma_stack:.2f} px")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].plot(lam, yf - model, color="crimson", lw=1)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xlabel("Wavelength [nm]")
        axes[1].set_ylabel("Residual")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
