"""
Stage 4 - global multi-frame fit for the C II temperature.

Fitting each frame on its own cannot separate the Gaussian width (thermal, what
we want) from the Lorentzian width (Stark, a nuisance): the C II lines sit
within about 1.02x of the instrumental width, so the two are degenerate. Stage 3
shows this directly - with both free, gamma jumps between 0 and 8 px at random.

The way out is that some unknowns are the SAME in every frame, so all 19 frames
are fitted simultaneously. Three models are compared:

  ALL-THERMAL   gamma = 0, sigma free per frame.
                What the old scripts assumed. Reproduces T ~ 200-500 eV.

  STARK-TIED    gamma(f) = gamma_ref * n_e(f)/1e17 with one shared gamma_ref,
                T free per frame. n_e(f) comes from H-alpha (Stage 2) and is
                fixed input. This is the model that would let 19 frames
                constrain one Stark coefficient.

  GAMMA-FREE    gamma free per frame (a nuisance parameter of whatever physical
                origin), ONE temperature shared by all frames. Because T is
                shared while gamma absorbs the per-frame excess, T is
                identifiable here even though it is not frame by frame.

The data REJECTS the STARK-TIED model: the excess width is not a single-valued
function of n_e. At n_e ~ 7e16 the excess is 44 km/s on the rise (frame 10) but
8 km/s on the decay (frame 17); at n_e ~ 1.4e17 it is 48 km/s (frame 12) versus
26 km/s (frame 14). The excess decays with FRAME NUMBER, not with density - it
behaves like a transient bulk motion that dies away, not like Stark broadening.
test_stark_scaling() prints this evidence.

So GAMMA-FREE is the model to quote a temperature from.

Amplitudes and continuum coefficients enter the model LINEARLY, so they are
solved exactly by weighted linear least squares inside each residual evaluation
(variable projection) rather than handed to the optimiser. That drops the
nonlinear parameter count from ~190 to ~40 and makes the fit far more stable.

    python global_T_fit.py
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize

import spectro_core as sc
import ha_density as hd


N_CONTINUUM = 4        # cubic continuum under the C group
T_MAX_EV = 400.0       # ceiling; reaching it means the model is wrong, not hot
T_SANITY_EV = 50.0     # above this the width is being mis-attributed (see plan)

# The Stark term is parameterised as the Lorentzian width AT A REFERENCE DENSITY
# rather than as a coefficient. Written as a coefficient it is ~3e-17 px per
# cm^-3, hopeless for an optimiser taking order-1 steps; expressed as "gamma in
# pixels when n_e = 1e17" it is ~3, which is well scaled.
#
#     gamma_C(f) = gamma_ref * n_e(f) / N_E_REF
N_E_REF = 1e17         # cm^-3

MODELS = ("all-thermal", "stark-tied", "gamma-free")


# --- frame preparation -------------------------------------------------------

def prepare_frames(exp_i, sigma_inst, verbose=True):
    """
    Load every usable frame, fit H-alpha, and cache what the global fit needs:
    the C-window data, its weights, and the fixed H-alpha pedestal underneath.
    """
    frames = []
    lo, hi = sc.C_WINDOW
    if verbose:
        print("Preparing frames (H-alpha fit for n_e and the pedestal)...")
    for frame_i in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        x, profile = sc.load_profile(exp_i, frame_i)
        ha = hd.fit_ha(x, profile, 0.0, sigma_inst)
        ped_full = sc.voigt(x, ha["params"]["amp"].value,
                            ha["params"]["cen"].value,
                            ha["params"]["sigma"].value,
                            ha["params"]["gamma"].value)
        m = (x >= lo) & (x < hi)
        frames.append({
            "frame": frame_i,
            "x": x[m], "y": profile[m], "w": sc.weights(profile[m]),
            "pedestal": ped_full[m],
            "n_e": ha["n_e"], "n_e_err": ha["n_e_err"],
        })
    return frames


# --- model core --------------------------------------------------------------

def _solve_linear(fr, lines, sigma_px, gamma_px, shift):
    """
    Given the nonlinear widths and shift, solve the linear amplitudes and
    continuum coefficients exactly (variable projection) and return the
    weighted residual.
    """
    x, y, w, ped = fr["x"], fr["y"], fr["w"], fr["pedestal"]

    columns = [sc.voigt(x, 1.0, ln["pixel"] + shift, sigma_px, gamma_px)
               for ln in lines]
    t = (x - 0.5 * (x[0] + x[-1])) / 100.0
    columns += [t ** i for i in range(N_CONTINUUM)]
    design = np.column_stack(columns)

    target = y - ped
    coeffs, *_ = np.linalg.lstsq(design * w[:, None], target * w, rcond=None)
    model = ped + design @ coeffs
    return (model - y) * w, coeffs, model


def _line_centre(lines):
    return (float(np.mean([ln["pixel"] for ln in lines])),
            float(np.mean([ln["lambda_nm"] for ln in lines])))


def _sigma_for_T(T_eV, sigma_inst, lines):
    """Total Gaussian width in px: instrumental and carbon-thermal in quadrature."""
    px, lam = _line_centre(lines)
    sigma_th = sc.thermal_sigma_px(T_eV, lam, sc.MC2_C_EV, px)
    return float(np.sqrt(sigma_inst ** 2 + sigma_th ** 2))


def build_params(frames, sigma_inst, model):
    params = Parameters()
    params.add("sigma_inst", value=sigma_inst, vary=False)
    if model == "stark-tied":
        params.add("gamma_ref",
                   value=3.9e-19 * N_E_REF / sc.dispersion_nm_per_px(752.0),
                   min=0.0, max=40.0)
    if model == "gamma-free":
        params.add("T_eV", value=5.0, min=0.0, max=T_MAX_EV)
    for fr in frames:
        f = fr["frame"]
        params.add(f"shift_{f}", value=0.0, min=-8.0, max=8.0)
        if model in ("all-thermal", "stark-tied"):
            params.add(f"T_{f}", value=5.0, min=0.0, max=T_MAX_EV)
        if model == "gamma-free":
            params.add(f"gam_{f}", value=1.0, min=0.0, max=25.0)
    return params


def _widths(params, fr, lines, model):
    """(sigma_px, gamma_px) for one frame under the chosen model."""
    sigma_inst = params["sigma_inst"].value
    f = fr["frame"]
    if model == "all-thermal":
        return _sigma_for_T(params[f"T_{f}"].value, sigma_inst, lines), 0.0
    if model == "stark-tied":
        return (_sigma_for_T(params[f"T_{f}"].value, sigma_inst, lines),
                params["gamma_ref"].value * fr["n_e"] / N_E_REF)
    return (_sigma_for_T(params["T_eV"].value, sigma_inst, lines),
            params[f"gam_{f}"].value)


def make_residual(frames, lines, model):
    def residual(params):
        chunks = []
        for fr in frames:
            sigma_px, gamma_px = _widths(params, fr, lines, model)
            res, _, _ = _solve_linear(fr, lines, sigma_px, gamma_px,
                                      params[f"shift_{fr['frame']}"].value)
            chunks.append(res)
        return np.concatenate(chunks)
    return residual


def fit_model(frames, lines, sigma_inst, model):
    params = build_params(frames, sigma_inst, model)
    result = minimize(make_residual(frames, lines, model), params)
    return {"model": model, "result": result}


# --- the test that decides which model to believe ----------------------------

def test_stark_scaling(frames, sigma_inst, lines, verbose=True):
    """
    Is the excess C II width a single-valued function of n_e, as Stark requires?

    Compares frames on the density rise against frames on the decay at matched
    n_e. If the excess is Stark it must be the same at the same density.
    """
    px, lam = _line_centre(lines)
    disp = sc.dispersion_nm_per_px(px)

    rows = []
    for fr in frames:
        # total Gaussian-equivalent width of this frame, measured with gamma = 0
        p = Parameters()
        p.add("sigma", value=sigma_inst * 1.2, min=1.0, max=30.0)
        p.add("shift", value=0.0, min=-8.0, max=8.0)

        def res(pp):
            r, _, _ = _solve_linear(fr, lines, pp["sigma"].value, 0.0,
                                    pp["shift"].value)
            return r

        out = minimize(res, p)
        sigma = abs(out.params["sigma"].value)
        excess = np.sqrt(max(sigma ** 2 - sigma_inst ** 2, 0.0))
        rows.append({"frame": fr["frame"], "n_e": fr["n_e"], "sigma_px": sigma,
                     "excess_px": excess,
                     "excess_kms": sc.C_KM_S * excess * disp / lam})

    peak = max(rows, key=lambda r: r["n_e"])["frame"]
    rise = [r for r in rows if r["frame"] <= peak]
    decay = [r for r in rows if r["frame"] > peak]

    pairs = []
    for r in rise:
        if not decay:
            break
        d = min(decay, key=lambda q: abs(q["n_e"] - r["n_e"]))
        if abs(d["n_e"] - r["n_e"]) / r["n_e"] < 0.20:
            pairs.append((r, d))

    if verbose:
        print("\n" + "-" * 74)
        print("Does the excess C II width scale with n_e, as Stark requires?")
        print("-" * 74)
        print("Frames at matched density, one on the rise and one on the decay:")
        print(f"  {'rise':>16}   {'decay':>16}      ratio")
        for r, d in pairs:
            print(f"  fr{r['frame']:2d} n_e={r['n_e']:.2e} {r['excess_kms']:5.1f} km/s"
                  f"   fr{d['frame']:2d} n_e={d['n_e']:.2e} {d['excess_kms']:5.1f} km/s"
                  f"   {r['excess_kms'] / max(d['excess_kms'], 1e-9):6.1f}x")
        if pairs:
            ratios = [r["excess_kms"] / max(d["excess_kms"], 1e-9) for r, d in pairs]
            print(f"\n  median rise/decay ratio at matched n_e: "
                  f"{np.median(ratios):.1f}x")
            print("  A Stark width must be the same at the same density (ratio 1).")
            print("  It is not, so the excess is NOT Stark broadening scaling with")
            print("  n_e. It decays with frame number instead - the signature of a")
            print("  transient bulk motion, which is mass-independent and therefore")
            print("  cannot be read as a temperature either.")
    return rows, pairs


# --- reporting ---------------------------------------------------------------

def temperatures(fit, frames, lines):
    """Per-frame T (with uncertainty) from a fit, plus the sanity flag."""
    result, model = fit["result"], fit["model"]
    rows = []
    for fr in frames:
        f = fr["frame"]
        if model == "gamma-free":
            p = result.params["T_eV"]
        else:
            p = result.params[f"T_{f}"]
        T = p.value
        rows.append({
            "frame": f, "T_eV": T,
            "T_err": np.nan if p.stderr is None else p.stderr,
            "n_e": fr["n_e"], "n_e_err": fr["n_e_err"],
            "sane": T <= T_SANITY_EV,
        })
    return rows


def sensitivity(fit, frames, lines, rel=0.05):
    """
    Recompute T for sigma_inst +- rel, holding the fitted total width fixed.

    The C lines sit barely above the resolution limit, so this systematic
    dominates the error budget and has to be shown, not buried.
    """
    result, model = fit["result"], fit["model"]
    sigma_inst = result.params["sigma_inst"].value
    px, lam = _line_centre(lines)
    bands = {}
    for tag, s in (("low", sigma_inst * (1 - rel)), ("high", sigma_inst * (1 + rel))):
        vals = []
        for fr in frames:
            key = "T_eV" if model == "gamma-free" else f"T_{fr['frame']}"
            sigma_tot = _sigma_for_T(result.params[key].value, sigma_inst, lines)
            vals.append(sc.temperature_from_sigma(sigma_tot, lam, sc.MC2_C_EV,
                                                  px, sigma_inst_px=s))
        bands[tag] = np.array(vals)
    return bands


def run(exp_i, sigma_inst=None, verbose=True):
    lines, meta = sc.load_line_table()
    if sigma_inst is None:
        sigma_inst = (meta or {}).get("sigma_inst_px", sc.SIGMA_INST_PX)

    frames = prepare_frames(exp_i, sigma_inst, verbose)
    fits = {}
    for model in MODELS:
        fits[model] = fit_model(frames, lines, sigma_inst, model)
        if verbose:
            r = fits[model]["result"]
            print(f"\n  {model:12s} chi2r = {r.redchi:.4f}  "
                  f"({r.ndata} points, {r.nvarys} varied)")
    return {"frames": frames, "lines": lines, "fits": fits,
            "sigma_inst": sigma_inst}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=int, default=559)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("STAGE 4 - global multi-frame fit for the C II temperature")
    print("=" * 74 + "\n")

    out = run(args.exp)
    frames, lines, fits = out["frames"], out["lines"], out["fits"]
    sigma_inst = out["sigma_inst"]
    print(f"\n  sigma_inst held at {sigma_inst:.4f} px (Stage 1 floor)")

    widths, _ = test_stark_scaling(frames, sigma_inst, lines)

    # what each model says
    print("\n" + "-" * 74)
    print("What each model gives")
    print("-" * 74)
    at = temperatures(fits["all-thermal"], frames, lines)
    st = temperatures(fits["stark-tied"], frames, lines)
    gf = fits["gamma-free"]["result"].params["T_eV"]
    print(f"  ALL-THERMAL  T ranges {min(r['T_eV'] for r in at):.0f} to "
          f"{max(r['T_eV'] for r in at):.0f} eV  "
          f"({sum(1 for r in at if not r['sane'])}/{len(at)} frames above "
          f"{T_SANITY_EV:.0f} eV - unphysical for C II)")
    print(f"  STARK-TIED   T ranges {min(r['T_eV'] for r in st):.0f} to "
          f"{max(r['T_eV'] for r in st):.0f} eV  "
          f"({sum(1 for r in st if not r['sane'])}/{len(st)} frames above "
          f"{T_SANITY_EV:.0f} eV - and the model is rejected above)")
    print(f"  GAMMA-FREE   T = {gf.value:.2f} "
          f"+/- {'n/a' if gf.stderr is None else f'{gf.stderr:.2f}'} eV "
          f"(one value for the shot)")

    # per-frame table
    print("\n" + "-" * 74)
    print("Per-frame width decomposition")
    print("-" * 74)
    print(f"{'frame':>5} {'n_e [cm^-3]':>12} {'sigma_tot':>9} {'excess':>8} "
          f"{'v_excess':>9} {'T if all thermal':>17}")
    print(f"{'':5} {'':12} {'[px]':>9} {'[px]':>8} {'[km/s]':>9} {'[eV]':>17}")
    for w, a in zip(widths, at):
        flag = "" if a["sane"] else "  <- unphysical"
        print(f"{w['frame']:5d} {w['n_e']:12.4e} {w['sigma_px']:9.3f} "
              f"{w['excess_px']:8.3f} {w['excess_kms']:9.1f} "
              f"{a['T_eV']:17.1f}{flag}")

    band = sensitivity(fits["gamma-free"], frames, lines)
    print("\n" + "-" * 74)
    print("Recommended result")
    print("-" * 74)
    print(f"  T = {gf.value:.1f} eV "
          f"(+/- {'n/a' if gf.stderr is None else f'{gf.stderr:.1f}'} fit), "
          f"sigma_inst +-5% gives {band['high'][0]:.1f} to {band['low'][0]:.1f} eV")
    print("  Quote this as an UPPER LIMIT: sigma_inst is measured as the width")
    print("  floor of these same lines, so it is an upper bound on the true")
    print("  resolution, and any real temperature is at or below this value.")
    # Derived from the fit, not hard-coded - these scale with the dispersion.
    _ex = np.array([r["excess_kms"] for r in rows
                    if 9 <= r["frame"] <= 13 and np.isfinite(r["excess_kms"])])
    if _ex.size:
        _ceil = sc.C_KM_S * np.sqrt(24.38 / sc.MC2_C_EV)
        print(f"\n  The frames on the density rise (roughly 9-13) carry an "
              f"extra")
        print(f"  {_ex.min():.0f}-{_ex.max():.0f} km/s of width that is neither "
              f"instrumental nor Stark")
        print(f"  (it is double-valued in n_e). Read as carbon thermal motion")
        print(f"  that is {sc.MC2_C_EV * (_ex.min() / sc.C_KM_S) ** 2:.0f}-"
              f"{sc.MC2_C_EV * (_ex.max() / sc.C_KM_S) ** 2:.0f} eV, against a "
              f"C II thermal ceiling of {_ceil:.1f} km/s")
        print(f"  ({24.38:.2f} eV) where the ion stops existing. Most likely")
        print("  bulk plasma motion, which is mass-independent and so is not a")
        print("  temperature at all.")
    print("\n  A calibration-lamp instrumental profile is the only thing that")
    print("  will turn this upper limit into a measurement.")

    if not args.no_plot:
        f = np.array([w["frame"] for w in widths])
        v = np.array([w["excess_kms"] for w in widths])
        n_e = np.array([w["n_e"] for w in widths])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.plot(f, v, "o-", color="tab:purple", label="excess width [km/s]")
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Excess width as velocity [km/s]", color="tab:purple")
        ax.grid(alpha=0.3)
        tw = ax.twinx()
        tw.plot(f, n_e, "--", color="tab:blue", label=r"$n_e$")
        tw.set_ylabel(r"$n_e$ [cm$^{-3}$]", color="tab:blue")
        ax.set_title("Excess width decays with frame, not with $n_e$")

        ax = axes[1]
        peak = f[int(np.argmax(n_e))]
        rise, dec = f <= peak, f > peak
        ax.plot(n_e[rise], v[rise], "o-", color="tab:red", label="rise")
        ax.plot(n_e[dec], v[dec], "s-", color="tab:green", label="decay")
        ax.set_xlabel(r"$n_e$ [cm$^{-3}$]")
        ax.set_ylabel("Excess width [km/s]")
        ax.set_title("Not single-valued in $n_e$ -> not Stark")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
