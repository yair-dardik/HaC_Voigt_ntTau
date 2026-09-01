"""
Stage 3 - show that the C II lines are Lorentzian (Stark) broadened, not just
Gaussian (thermal + instrumental).

This is the evidence that the C-line width cannot be read as a temperature.

Method: fit the C1 pair (px 700-800) and the C2 pair (px 810-915) SEPARATELY,
each with the Gaussian sigma HELD FIXED at the instrumental floor and a single
shared Lorentzian gamma free per pair. Holding sigma is essential - letting
sigma and gamma both float per frame is unidentifiable: gamma then jumps
between 0 and 8 px and the two pairs only correlate at r = 0.13.

Two independent checks are reported:

  1. Correlation. If the lines carried only Gaussian broadening, a sigma-fixed
     fit would drive gamma to zero in every frame. Instead gamma rises with
     n_e, and the two pairs - fitted in completely separate windows - agree
     with each other.

  2. Shape. The same data is also fitted as a pure Gaussian with sigma free
     (what a thermal-only interpretation predicts). Comparing reduced
     chi-square frame by frame says whether the profile SHAPE actually prefers
     the Lorentzian, independently of the correlation argument.

    python stark_consistency.py
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize

import spectro_core as sc
import ha_density as hd


# A frame only contributes to the correlations if the pair's fitted area is
# significant. This is measured from the fit itself (total area / its error)
# rather than hard-coding a frame range: C2 turns out to have significance > 7
# in every frame except 5 and 6, so the gamma_C2 = 0 values in the late frames
# are genuinely low Stark broadening, not missing signal.
MIN_AREA_SIGNIFICANCE = 7.0


def _pair_fit(x, profile, pair_lines, window, ha_pedestal,
              sigma_inst_px, free_sigma=False):
    """
    Fit one pair of C II components over `window`.

    free_sigma=False -> sigma fixed at the instrumental width, gamma free (Stark)
    free_sigma=True  -> pure Gaussian, sigma free, gamma pinned to 0 (thermal)

    ha_pedestal is the H-alpha Voigt from Stage 2 evaluated on the full pixel
    axis and held fixed, so the broad H-alpha wing under the C lines is
    accounted for rather than absorbed into the C widths.
    """
    lo, hi = window
    m = (x >= lo) & (x < hi)
    xf, yf = x[m], profile[m]
    wf = sc.weights(yf)
    ped = ha_pedestal[m]

    params = Parameters()
    if free_sigma:
        params.add("sigma", value=sigma_inst_px, min=1.0, max=30.0)
        params.add("gamma", value=0.0, vary=False)
    else:
        params.add("sigma", value=sigma_inst_px, vary=False)
        params.add("gamma", value=1.0, min=0.0, max=25.0)
    params.add("shift", value=0.0, min=-10.0, max=10.0)
    for i, line in enumerate(pair_lines):
        idx = int(np.argmin(np.abs(xf - line["pixel"])))
        height = float(np.max(yf[max(0, idx - 10):idx + 10]) - np.median(yf))
        params.add(f"amp{i}", value=max(height, 1.0) * 18.0, min=0.0)
    for i in range(4):
        params.add(f"c{i}", value=np.median(yf) if i == 0 else 0.0)

    def model(p):
        out = ped + sc.polynomial_continuum(
            xf, [p[f"c{i}"].value for i in range(4)],
            x_ref=0.5 * (lo + hi))
        for i, line in enumerate(pair_lines):
            out = out + sc.voigt(xf, p[f"amp{i}"].value,
                                 line["pixel"] + p["shift"].value,
                                 p["sigma"].value, p["gamma"].value)
        return out

    result = minimize(lambda p: (model(p) - yf) * wf, params)
    centre_px = float(np.mean([ln["pixel"] for ln in pair_lines]))
    value = (result.params["sigma"].value if free_sigma
             else result.params["gamma"].value)
    err = (result.params["sigma"].stderr if free_sigma
           else result.params["gamma"].stderr)

    # total fitted area and its significance, used to decide whether this frame
    # is worth including in the correlations
    area = sum(result.params[f"amp{i}"].value for i in range(len(pair_lines)))
    area_var = sum((result.params[f"amp{i}"].stderr or 0.0) ** 2
                   for i in range(len(pair_lines)))
    area_sig = area / np.sqrt(area_var) if area_var > 0 else np.inf

    return {
        "area": area, "area_significance": float(area_sig),
        "value_px": abs(value),
        "err_px": np.nan if err is None else abs(err),
        "value_nm": float(sc.width_px_to_nm(abs(value), centre_px)),
        "err_nm": (np.nan if err is None
                   else float(sc.width_px_to_nm(abs(err), centre_px))),
        "redchi": result.redchi,
        "centre_px": centre_px,
        "x": xf, "y": yf, "model": model(result.params), "params": result.params,
    }


def run(exp_i, sigma_inst_px=None, lines=None, verbose=True):
    """Fit every frame and collect gamma_C1, gamma_C2 alongside n_e from H-alpha."""
    if lines is None:
        lines, meta = sc.load_line_table()
        if sigma_inst_px is None:
            sigma_inst_px = (meta or {}).get("sigma_inst_px", sc.SIGMA_INST_PX)
    if sigma_inst_px is None:
        sigma_inst_px = sc.SIGMA_INST_PX

    c1_lines = sc.group_lines(lines, "C1")
    c2_lines = sc.group_lines(lines, "C2")

    rows = []
    if verbose:
        print(f"{'frame':>5} {'n_e [cm^-3]':>12} | {'gam_C1 [nm]':>11} {'+/-':>8} "
              f"{'chi2r':>6} | {'gam_C2 [nm]':>11} {'+/-':>8} {'chi2r':>6} |"
              f" {'chi2r Gauss-only':>17}")
    for frame_i in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        x, profile = sc.load_profile(exp_i, frame_i)
        ha = hd.fit_ha(x, profile, 0.0, sigma_inst_px)
        ped = sc.voigt(x, ha["params"]["amp"].value, ha["params"]["cen"].value,
                       ha["params"]["sigma"].value, ha["params"]["gamma"].value)

        c1 = _pair_fit(x, profile, c1_lines, sc.C1_WINDOW, ped, sigma_inst_px)
        c2 = _pair_fit(x, profile, c2_lines, sc.C2_WINDOW, ped, sigma_inst_px)
        # shape test: same window, pure Gaussian with sigma free
        c1g = _pair_fit(x, profile, c1_lines, sc.C1_WINDOW, ped, sigma_inst_px,
                        free_sigma=True)

        rows.append({
            "frame": frame_i, "n_e": ha["n_e"], "n_e_err": ha["n_e_err"],
            "gam_C1_nm": c1["value_nm"], "gam_C1_err_nm": c1["err_nm"],
            "gam_C2_nm": c2["value_nm"], "gam_C2_err_nm": c2["err_nm"],
            "redchi_C1_voigt": c1["redchi"], "redchi_C2_voigt": c2["redchi"],
            "redchi_C1_gauss": c1g["redchi"],
            "sigma_gauss_only_px": c1g["value_px"],
            "sig_C1": c1["area_significance"],
            "sig_C2": c2["area_significance"],
        })
        if verbose:
            r = rows[-1]
            print(f"{frame_i:5d} {r['n_e']:12.4e} | {r['gam_C1_nm']:11.5f} "
                  f"{r['gam_C1_err_nm']:8.5f} {r['redchi_C1_voigt']:6.2f} | "
                  f"{r['gam_C2_nm']:11.5f} {r['gam_C2_err_nm']:8.5f} "
                  f"{r['redchi_C2_voigt']:6.2f} | {r['redchi_C1_gauss']:17.2f}")
    return rows


def _through_origin(n_e, gamma):
    """Least-squares slope of gamma = k * n_e, forced through the origin."""
    n_e = np.asarray(n_e, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    good = np.isfinite(n_e) & np.isfinite(gamma)
    if good.sum() < 2:
        return np.nan
    return float(np.sum(gamma[good] * n_e[good]) / np.sum(n_e[good] ** 2))


def summarise(rows):
    """Correlations and Stark coefficients."""
    frames = np.array([r["frame"] for r in rows])
    n_e = np.array([r["n_e"] for r in rows])
    g1 = np.array([r["gam_C1_nm"] for r in rows])
    g2 = np.array([r["gam_C2_nm"] for r in rows])
    c1_mask = np.array([r["sig_C1"] >= MIN_AREA_SIGNIFICANCE for r in rows])
    c2_mask = np.array([r["sig_C2"] >= MIN_AREA_SIGNIFICANCE for r in rows])
    both = c1_mask & c2_mask

    k1 = _through_origin(n_e[c1_mask], g1[c1_mask])
    k2 = _through_origin(n_e[c2_mask], g2[c2_mask])

    stats = {
        "k1": k1, "k2": k2,
        "r_g1_ne": float(np.corrcoef(n_e[c1_mask], g1[c1_mask])[0, 1]),
        "r_g2_ne": float(np.corrcoef(n_e[c2_mask], g2[c2_mask])[0, 1]),
        "r_g1_g2": float(np.corrcoef(g1[both], g2[both])[0, 1]),
        "frames": frames, "n_e": n_e, "gam_C1": g1, "gam_C2": g2,
        "c1_mask": c1_mask, "c2_mask": c2_mask, "both_mask": both,
    }
    return stats


def report(rows, stats):
    print("\n" + "-" * 74)
    print("Is the C II broadening Lorentzian?")
    print("-" * 74)
    n1 = int(stats["c1_mask"].sum())
    n2 = int(stats["c2_mask"].sum())
    nb = int(stats["both_mask"].sum())
    print("\n[1] Correlation test")
    print(f"  (frames included need fitted area / error >= "
          f"{MIN_AREA_SIGNIFICANCE:.0f})")
    print(f"  Pearson r(gamma_C1, n_e from H-alpha) = {stats['r_g1_ne']:+.3f}"
          f"   ({n1} frames)")
    print(f"  Pearson r(gamma_C2, n_e from H-alpha) = {stats['r_g2_ne']:+.3f}"
          f"   ({n2} frames)")
    print(f"  Pearson r(gamma_C1, gamma_C2)         = {stats['r_g1_g2']:+.3f}"
          f"   ({nb} frames, independent fit windows)")
    print(f"\n  Stark coefficient C1  k1 = {stats['k1']:.3e} nm per cm^-3")
    print(f"  Stark coefficient C2  k2 = {stats['k2']:.3e} nm per cm^-3")
    print("\n  If the C lines were purely Gaussian (thermal + instrumental), a")
    print("  sigma-fixed fit would drive gamma to zero in every frame.")

    print("\n[2] Shape test - reduced chi-square, same data and window")
    voigt_better = 0
    for r in rows:
        if r["redchi_C1_voigt"] < r["redchi_C1_gauss"]:
            voigt_better += 1
    print(f"  Voigt (sigma fixed, gamma free) beats pure Gaussian (sigma free)")
    print(f"  in {voigt_better} of {len(rows)} frames.")
    mv = np.mean([r["redchi_C1_voigt"] for r in rows])
    mg = np.mean([r["redchi_C1_gauss"] for r in rows])
    print(f"  mean chi2r: Voigt {mv:.3f}  vs  Gaussian {mg:.3f}")
    if mg < mv:
        print("  -> the profile SHAPE alone does not separate the two models at")
        print("     this resolution. The correlation test above is the argument;")
        print("     the shape test is honest about not adding to it.")
    else:
        print("  -> the profile shape independently prefers the Lorentzian.")

    print("\n[3] Caveat to state when presenting this")
    print("  k1 and k2 are FITTED to match the H-alpha density, so plot 3 shows")
    print("  SHAPE consistency, not an independent absolute density. The")
    print("  non-trivial content is that a single constant per line reproduces")
    print("  the whole n_e curve across its full dynamic range.")


def _nice_limits(values, pad=0.15):
    """Axis limits from the data itself, so one bad error bar cannot squash it."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(v)), float(np.max(v))
    span = max(hi - lo, abs(hi) * 0.1, 1e-12)
    return (min(lo - pad * span, 0.0), hi + pad * span)


def plot(stats, rows, exp_i):
    frames, n_e = stats["frames"], stats["n_e"]
    g1, g2 = stats["gam_C1"], stats["gam_C2"]
    c1m, c2m = stats["c1_mask"], stats["c2_mask"]
    g1e = np.array([r["gam_C1_err_nm"] for r in rows])
    g2e = np.array([r["gam_C2_err_nm"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # (1) gamma and n_e vs frame
    ax = axes[0]
    ax.errorbar(frames[c1m], g1[c1m], yerr=g1e[c1m], marker="o", ms=4, capsize=2,
                color="tab:green", label=r"$\gamma_{C1}$")
    ax.errorbar(frames[c2m], g2[c2m], yerr=g2e[c2m], marker="s", ms=4, capsize=2,
                color="tab:orange", label=r"$\gamma_{C2}$")
    ax.set_xlabel("Frame number")
    ax.set_ylabel(r"C II Lorentzian width $\gamma$ [nm]")
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(frames, n_e, "--", color="tab:blue", lw=2, label=r"$n_e$ (H$\alpha$)")
    twin.set_ylabel(r"$n_e$ [cm$^{-3}$]", color="tab:blue")
    twin.tick_params(axis="y", labelcolor="tab:blue")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax.set_title("C II Lorentzian width tracks the density")
    # a single frame with a badly determined gamma otherwise squashes the axis
    ax.set_ylim(_nice_limits(np.concatenate([g1[c1m], g2[c2m]])))

    # (2) gamma vs n_e, straight line through origin
    ax = axes[1]
    ax.errorbar(n_e[c1m], g1[c1m], yerr=g1e[c1m], fmt="o", ms=5, capsize=2,
                color="tab:green", label=f"C1  (r = {stats['r_g1_ne']:+.2f})")
    ax.errorbar(n_e[c2m], g2[c2m], yerr=g2e[c2m], fmt="s", ms=5, capsize=2,
                color="tab:orange", label=f"C2  (r = {stats['r_g2_ne']:+.2f})")
    grid = np.linspace(0, float(np.nanmax(n_e)) * 1.05, 50)
    ax.plot(grid, stats["k1"] * grid, "-", color="tab:green", lw=1.2,
            label=f"$k_1$ = {stats['k1']:.2e}")
    ax.plot(grid, stats["k2"] * grid, "-", color="tab:orange", lw=1.2,
            label=f"$k_2$ = {stats['k2']:.2e}")
    ax.set_xlabel(r"$n_e$ from H$\alpha$ [cm$^{-3}$]")
    ax.set_ylabel(r"C II $\gamma$ [nm]")
    ax.set_title("$\\gamma$ rises with $n_e$ (but see Stage 4:\n"
                 "it is not single-valued in $n_e$)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(_nice_limits(np.concatenate([g1[c1m], g2[c2m]])))

    # (3) n_e recovered from each C line, against H-alpha
    ax = axes[2]
    ax.plot(frames, n_e, "o--", color="tab:blue", lw=2, ms=5,
            label=r"$n_e$ from H$\alpha$ (reference)")
    ax.plot(frames[c1m], g1[c1m] / stats["k1"], "o-", color="tab:green", ms=4,
            label=r"$n_e$ from C1 $\gamma / k_1$")
    ax.plot(frames[c2m], g2[c2m] / stats["k2"], "s-", color="tab:orange", ms=4,
            label=r"$n_e$ from C2 $\gamma / k_2$")
    ax.set_xlabel("Frame number")
    ax.set_ylabel(r"$n_e$ [cm$^{-3}$]")
    ax.set_title("Densities from the three lines agree in shape\n"
                 "(note: $k_1,k_2$ are fitted to H$\\alpha$ - shape, not absolute)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, float(np.nanmax(n_e)) * 2.2)

    fig.suptitle(f"C{exp_i} - the C II lines carry Lorentzian width that rises "
                 f"with $n_e$ (Stage 3)", fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=int, default=559)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("STAGE 3 - are the C II lines Lorentzian (Stark) broadened?")
    print("=" * 74 + "\n")

    rows = run(args.exp)
    stats = summarise(rows)
    report(rows, stats)

    if not args.no_plot:
        plot(stats, rows, args.exp)


if __name__ == "__main__":
    main()
