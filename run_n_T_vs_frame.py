"""
Stage 5 - the driver. Produces the n_e and T plots against FRAME NUMBER and
writes everything to CSV so the figures can be redrawn without refitting.

    python run_n_T_vs_frame.py                # everything
    python run_n_T_vs_frame.py --frame 22     # single-frame diagnostic
    python run_n_T_vs_frame.py --from-csv     # redraw from a previous run

Figures:
  1  n_e against frame, with error bars
  2  T against frame, with the sigma_inst systematic band and the upper limit
  3  the Stark-consistency panel from Stage 3
  4  excess width against n_e, split into rise and decay
  5  n_e against T, coloured by frame - the original question
  6  fit quality: reduced chi-square and C-group significance per frame

Failed or insignificant frames are written as NaN, never as 0. The old scripts
returned 0,0,0 on a failed fit and plotted it as a real measurement.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import spectro_core as sc
import ha_density as hd
import stark_consistency as st
import global_T_fit as gt


CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_n_T_vs_frame.csv")

CSV_FIELDS = [
    "frame", "n_e_cm3", "n_e_err", "stark_fwhm_nm", "ha_redchi",
    "sigma_tot_px", "excess_px", "excess_kms", "T_all_thermal_eV",
    "gam_C1_nm", "gam_C1_err_nm", "gam_C2_nm", "gam_C2_err_nm",
    "sig_C1", "sig_C2", "c_redchi",
]


# --- computation -------------------------------------------------------------

def compute(exp_i, verbose=True):
    """Run every stage once and collect the per-frame results."""
    lines, meta = sc.load_line_table()
    sigma_inst = (meta or {}).get("sigma_inst_px", sc.SIGMA_INST_PX)

    if verbose:
        print(f"instrumental sigma = {sigma_inst:.4f} px "
              f"({sc.SIGMA_INST_SOURCE})")
        print("\nStage 2/3 - H-alpha density and C II Lorentzian widths...")
    rows_c = st.run(exp_i, sigma_inst_px=sigma_inst, lines=lines, verbose=False)
    stats = st.summarise(rows_c)

    if verbose:
        print("Stage 4 - global multi-frame temperature fit...")
    frames = gt.prepare_frames(exp_i, sigma_inst, verbose=False)
    widths, _ = gt.test_stark_scaling(frames, sigma_inst, lines, verbose=False)
    fits = {m: gt.fit_model(frames, lines, sigma_inst, m) for m in gt.MODELS}
    T_global = fits["gamma-free"]["result"].params["T_eV"]
    T_thermal = gt.temperatures(fits["all-thermal"], frames, lines)

    # H-alpha Stark width and reduced chi-square per frame, for the CSV and the
    # quality panel
    ha_redchi, ha_fwhm = {}, {}
    for frame_i in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        x, profile = sc.load_profile(exp_i, frame_i)
        ha = hd.fit_ha(x, profile, 0.0, sigma_inst)
        ha_redchi[frame_i] = ha["redchi"]
        ha_fwhm[frame_i] = ha["stark_fwhm_nm"]

    records = []
    for rc, w, ta in zip(rows_c, widths, T_thermal):
        assert rc["frame"] == w["frame"] == ta["frame"]
        f = rc["frame"]
        sig1, sig2 = rc["sig_C1"], rc["sig_C2"]
        ok = sig1 >= st.MIN_AREA_SIGNIFICANCE
        records.append({
            "frame": f,
            "n_e_cm3": rc["n_e"], "n_e_err": rc["n_e_err"],
            "stark_fwhm_nm": ha_fwhm[f], "ha_redchi": ha_redchi[f],
            "sigma_tot_px": w["sigma_px"] if ok else np.nan,
            "excess_px": w["excess_px"] if ok else np.nan,
            "excess_kms": w["excess_kms"] if ok else np.nan,
            "T_all_thermal_eV": ta["T_eV"] if ok else np.nan,
            "gam_C1_nm": rc["gam_C1_nm"] if ok else np.nan,
            "gam_C1_err_nm": rc["gam_C1_err_nm"],
            "gam_C2_nm": rc["gam_C2_nm"] if sig2 >= st.MIN_AREA_SIGNIFICANCE else np.nan,
            "gam_C2_err_nm": rc["gam_C2_err_nm"],
            "sig_C1": sig1, "sig_C2": sig2,
            "c_redchi": rc["redchi_C1_voigt"],
        })

    return {
        "records": records, "rows_c": rows_c, "stats": stats,
        "sigma_inst": sigma_inst, "lines": lines, "frames": frames,
        "fits": fits, "T_global": T_global,
        "T_band": gt.sensitivity(fits["gamma-free"], frames, lines),
        "exp": exp_i,
    }


def write_csv(records, path=CSV_PATH):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    return path


def read_csv(path=CSV_PATH):
    records = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec = {}
            for k, v in row.items():
                if k == "frame":
                    rec[k] = int(v)
                else:
                    rec[k] = float(v) if v not in ("", "nan") else np.nan
            records.append(rec)
    return records


# --- plotting ----------------------------------------------------------------

def _col(records, key):
    return np.array([r[key] for r in records], dtype=float)


def plot_all(out):
    records = out["records"]
    exp_i = out["exp"]
    f = _col(records, "frame")
    n_e = _col(records, "n_e_cm3")
    n_e_err = _col(records, "n_e_err")
    v_ex = _col(records, "excess_kms")
    T_th = _col(records, "T_all_thermal_eV")
    T = out["T_global"]
    band = out["T_band"]

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

    # (1) n_e vs frame
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(f, n_e, yerr=n_e_err, marker="o", capsize=3, color="tab:blue")
    ax.set_xlabel("Frame number")
    ax.set_ylabel(r"$n_e$ [cm$^{-3}$]")
    ax.set_title(r"1. $n_e$ from H$\alpha$ Stark broadening")
    ax.grid(alpha=0.3)

    # (2) T vs frame
    ax = fig.add_subplot(gs[0, 1])
    hi = float(np.nanmax(band["low"]))
    ax.axhspan(0, hi, color="tab:red", alpha=0.15,
               label=r"$\sigma_{inst}\pm5\%$ systematic")
    ax.axhline(T.value, color="tab:red", lw=2,
               label=f"global fit T = {T.value:.1f} eV")
    if T.stderr is not None:
        ax.axhline(T.value + T.stderr, color="tab:red", ls=":", lw=1.2,
                   label=f"upper limit {T.value + T.stderr:.1f} eV")
    ax.plot(f, T_th, "s--", color="0.5", ms=4,
            label="if ALL excess were thermal")
    ax.axhline(gt.T_SANITY_EV, color="k", ls="-.", lw=1,
               label=f"{gt.T_SANITY_EV:.0f} eV - C II cannot exist above")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("C II temperature [eV]")
    ax.set_title("2. Temperature - an upper limit")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)

    # (3) Stark consistency: n_e from each line
    ax = fig.add_subplot(gs[0, 2])
    stats = out["stats"]
    c1m = stats["c1_mask"]
    c2m = stats["c2_mask"]
    ax.plot(f, n_e, "o--", color="tab:blue", lw=2, ms=5,
            label=r"H$\alpha$ (reference)")
    ax.plot(f[c1m], stats["gam_C1"][c1m] / stats["k1"], "o-",
            color="tab:green", ms=4, label=r"C1 $\gamma/k_1$")
    ax.plot(f[c2m], stats["gam_C2"][c2m] / stats["k2"], "s-",
            color="tab:orange", ms=4, label=r"C2 $\gamma/k_2$")
    ax.set_xlabel("Frame number")
    ax.set_ylabel(r"$n_e$ [cm$^{-3}$]")
    ax.set_title(f"3. C II Lorentzian widths track $n_e$\n"
                 f"r(C1,C2) = {stats['r_g1_g2']:+.2f}  "
                 f"($k_i$ fitted: shape, not absolute)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, float(np.nanmax(n_e)) * 2.2)

    # (4) excess width vs n_e, rise against decay
    ax = fig.add_subplot(gs[1, 0])
    good = np.isfinite(v_ex)
    peak = f[good][int(np.nanargmax(n_e[good]))]
    rise = good & (f <= peak)
    dec = good & (f > peak)
    ax.plot(n_e[rise], v_ex[rise], "o-", color="tab:red", label="density rise")
    ax.plot(n_e[dec], v_ex[dec], "s-", color="tab:green", label="density decay")
    ax.set_xlabel(r"$n_e$ [cm$^{-3}$]")
    ax.set_ylabel("Excess width [km/s]")
    ax.set_title("4. Excess width is not single-valued in $n_e$\n"
                 "-> not Stark, and not a temperature", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (5) n_e against T - the original question
    ax = fig.add_subplot(gs[1, 1])
    sctr = ax.scatter(n_e[good], T_th[good], c=f[good], cmap="viridis", s=55,
                      edgecolor="k", linewidth=0.4)
    ax.errorbar(n_e[good], T_th[good], xerr=n_e_err[good], fmt="none",
                ecolor="0.6", elinewidth=0.8)
    ax.axhline(gt.T_SANITY_EV, color="k", ls="-.", lw=1)
    ax.text(0.02, 0.94, f"above {gt.T_SANITY_EV:.0f} eV C II cannot survive",
            transform=ax.transAxes, fontsize=7.5, va="top")
    plt.colorbar(sctr, ax=ax, label="Frame number")
    ax.set_xlabel(r"$n_e$ [cm$^{-3}$]")
    ax.set_ylabel("T if all excess were thermal [eV]")
    ax.set_title("5. The original question, answered", fontsize=10)
    ax.grid(alpha=0.3)

    # (6) fit quality
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(f, _col(records, "ha_redchi"), "o-", color="tab:blue",
            label=r"H$\alpha$ fit $\chi^2_\nu$")
    ax.plot(f, _col(records, "c_redchi"), "s-", color="tab:green",
            label=r"C group fit $\chi^2_\nu$")
    ax.axhline(1.0, color="k", ls="--", lw=1, label=r"$\chi^2_\nu = 1$")
    ax.set_xlabel("Frame number")
    ax.set_ylabel(r"reduced $\chi^2$")
    ax.grid(alpha=0.3)
    tw = ax.twinx()
    tw.plot(f, _col(records, "sig_C1"), ":", color="tab:orange",
            label="C1 area significance")
    tw.axhline(st.MIN_AREA_SIGNIFICANCE, color="tab:orange", ls=":", lw=0.8)
    tw.set_ylabel("C1 area / error", color="tab:orange")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5)
    ax.set_title("6. Fit quality", fontsize=10)

    fig.suptitle(f"C{exp_i} - electron density and temperature against frame "
                 f"(frames {sc.FIRST_FRAME}-{sc.LAST_FRAME})", fontsize=14)
    plt.show()


def plot_single_frame(exp_i, frame_i):
    """
    Diagnostic for one frame: the four C II components plus the fixed H-alpha
    pedestal, and their sum against the data. The components genuinely add up to
    the drawn total here - the old scripts added the global offset to each
    component separately, so they never could.
    """
    lines, meta = sc.load_line_table()
    sigma_inst = (meta or {}).get("sigma_inst_px", sc.SIGMA_INST_PX)

    x, profile = sc.load_profile(exp_i, frame_i)
    ha = hd.fit_ha(x, profile, 0.0, sigma_inst)
    ped_full = sc.voigt(x, ha["params"]["amp"].value, ha["params"]["cen"].value,
                        ha["params"]["sigma"].value, ha["params"]["gamma"].value)
    lo, hi = sc.C_WINDOW
    m = (x >= lo) & (x < hi)
    fr = {"x": x[m], "y": profile[m], "w": sc.weights(profile[m]),
          "pedestal": ped_full[m], "n_e": ha["n_e"]}

    frames = [dict(fr, frame=frame_i)]
    widths, _ = gt.test_stark_scaling(frames, sigma_inst, lines, verbose=False)
    sigma_px = widths[0]["sigma_px"]
    _, coeffs, model = gt._solve_linear(fr, lines, sigma_px, 0.0, 0.0)

    lam = sc.pixel_to_nm(fr["x"])
    t = (fr["x"] - 0.5 * (fr["x"][0] + fr["x"][-1])) / 100.0
    cont = fr["pedestal"] + sum(coeffs[len(lines) + i] * t ** i
                                for i in range(gt.N_CONTINUUM))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(lam, fr["y"], color="0.6", lw=2.5, label="data")
    axes[0].plot(lam, model, "k--", lw=2, label="total fit")
    axes[0].plot(lam, cont, ":", color="0.4", lw=1.5,
                 label=r"H$\alpha$ wing + continuum")
    for i, line in enumerate(lines):
        comp = cont + sc.gaussian(fr["x"], coeffs[i], line["pixel"], sigma_px)
        axes[0].plot(lam, comp, lw=1.2, alpha=0.85,
                     label=f"{line['label']} @ {line['lambda_nm']:.4f} nm")
    axes[0].set_ylabel("Intensity [counts]")
    axes[0].set_title(
        f"C{exp_i} frame {frame_i} - C II group\n"
        f"$n_e$ = {ha['n_e']:.3e} cm$^{{-3}}$   "
        f"$\\sigma_{{tot}}$ = {sigma_px:.2f} px   "
        f"($\\sigma_{{inst}}$ = {sigma_inst:.2f} px, "
        f"excess = {widths[0]['excess_kms']:.1f} km/s)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(lam, fr["y"] - model, color="crimson", lw=1)
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
                    help="single-frame diagnostic plot instead of the full run")
    ap.add_argument("--from-csv", action="store_true",
                    help="redraw figure 1 from a previous run's CSV")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.frame is not None:
        plot_single_frame(args.exp, args.frame)
        return

    if args.from_csv:
        records = read_csv()
        f = _col(records, "frame")
        plt.figure(figsize=(9, 5.5))
        plt.errorbar(f, _col(records, "n_e_cm3"),
                     yerr=_col(records, "n_e_err"), marker="o", capsize=3)
        plt.xlabel("Frame number")
        plt.ylabel(r"$n_e$ [cm$^{-3}$]")
        plt.title(f"C{args.exp} - $n_e$ (from {os.path.basename(CSV_PATH)})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    print("=" * 74)
    print(f"C{args.exp} - n_e and T against frame "
          f"(frames {sc.FIRST_FRAME}-{sc.LAST_FRAME})")
    print("=" * 74 + "\n")

    out = compute(args.exp)
    records = out["records"]

    print(f"\n{'frame':>5} {'n_e [cm^-3]':>12} {'+/-':>10} "
          f"{'excess [km/s]':>13} {'T all-thermal':>14} {'C1 sig':>7}")
    for r in records:
        print(f"{r['frame']:5d} {r['n_e_cm3']:12.4e} {r['n_e_err']:10.2e} "
              f"{r['excess_kms']:13.1f} {r['T_all_thermal_eV']:14.1f} "
              f"{r['sig_C1']:7.1f}")

    T = out["T_global"]
    stats = out["stats"]
    print("\n" + "-" * 74)
    print("RESULTS")
    print("-" * 74)
    print(f"  n_e  ranges {np.nanmin(_col(records, 'n_e_cm3')):.2e} to "
          f"{np.nanmax(_col(records, 'n_e_cm3')):.2e} cm^-3, "
          f"peaking at frame "
          f"{records[int(np.nanargmax(_col(records, 'n_e_cm3')))]['frame']}")
    upper = T.value + (T.stderr or 0.0)
    print(f"  T    <= {upper:.1f} eV (global multi-frame fit, upper limit)")
    print(f"  C II Lorentzian widths correlate with n_e: "
          f"r(C1,n_e) = {stats['r_g1_ne']:+.2f}, "
          f"r(C1,C2) = {stats['r_g1_g2']:+.2f}")
    # Computed from the data, never hard-coded: these numbers all scale with
    # the dispersion, and stale literals here survived a 2.83x correction to it.
    _exc = _col(records, "excess_kms")
    _fr = _col(records, "frame")
    _rise = np.isfinite(_exc) & (_fr >= 9) & (_fr <= 13)
    if _rise.any():
        _lo, _hi = np.nanmin(_exc[_rise]), np.nanmax(_exc[_rise])
        _T = [sc.MC2_C_EV * (v / sc.C_KM_S) ** 2 for v in (_lo, _hi)]
        print(f"\n  The C II width in frames 9-13 carries {_lo:.0f}-{_hi:.0f} "
              f"km/s that is")
        print("  neither instrumental nor Stark (it is double-valued in n_e).")
        print(f"  Read as carbon thermal motion that would be {_T[0]:.0f}-"
              f"{_T[1]:.0f} eV, against")
        print("  24.4 eV where C II ionises - so it is not a temperature.")
        _ceil = sc.C_KM_S * np.sqrt(24.38 / sc.MC2_C_EV)
        print(f"  For scale, the C II thermal ceiling at 24.38 eV is "
              f"{_ceil:.1f} km/s.")

    path = write_csv(records)
    print(f"\nWrote {path}")

    if not args.no_plot:
        plot_all(out)


if __name__ == "__main__":
    main()
