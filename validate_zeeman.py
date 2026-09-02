"""
validate_zeeman.py - Zeeman broadening as a term in the C II width budget.

SENSITIVITY ANALYSIS, NOT A CORRECTION. The magnetic field in this experiment
is NOT measured. Everything below asks "what would B have to be", or "what
would the budget look like if B were X". No number here may propagate into
production until Sharon supplies a field estimate.

Diagnostics only: nothing in this file modifies spectro_core.py,
calibrate_lines.py or any production constant.

    python validate_zeeman.py
    python validate_zeeman.py --no-plot
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize

import spectro_core as sc
import ha_density as hd


# --- Zeeman pattern, C II 3s 2S_1/2 - 3p 2P* ---------------------------------
# Lande g factors: g(2S_1/2) = 2, g(2P*_3/2) = 4/3, g(2P*_1/2) = 2/3.
# Component shift in units of mu_B*B/h is (g_u*m_u - g_l*m_l), Delta_m = 0,+/-1.
# Verified by direct enumeration over m_u, m_l - see the header of this module's
# development notes; the shifts below are not transcribed, they are the closed
# set that enumeration produces.
#
# Relative strengths are the standard LS values (3:2:1 for J=1/2 -> J=3/2).
# TRANSVERSE viewing (line of sight perpendicular to B) sees both pi and sigma.
# LONGITUDINAL viewing (along B) sees only sigma. Which applies depends on the
# geometry of the shot, which is also not recorded here, so both are reported.
ZEEMAN = {
    "C1": {
        "lam_nm": 657.80482,
        # (shift in mu_B*B/h units, strength, polarisation)
        "components": [(-5/3, 1, "sigma"), (-1.0, 3, "sigma"), (-1/3, 2, "pi"),
                       (+1/3, 2, "pi"), (+1.0, 3, "sigma"), (+5/3, 1, "sigma")],
    },
    "C2": {
        "lam_nm": 658.28761,
        "components": [(-4/3, 1, "sigma"), (-2/3, 1, "pi"),
                       (+2/3, 1, "pi"), (+4/3, 1, "sigma")],
    },
}

MU_B_OVER_HC = 0.46686        # cm^-1 per tesla
C_II_IONISATION_EV = 24.38    # C II -> C III; the thermal ceiling
FIELDS_T = (0.1, 0.3, 0.5, 1.0)


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def angstrom_per_unit_per_tesla(lam_nm):
    """
    Wavelength shift per unit of (g_u m_u - g_l m_l) per tesla.

        d_lambda = lambda^2 * (mu_B / hc) * shift

    lambda^2 in cm^2 times cm^-1/T gives cm/T; x 1e8 for Angstrom/T.
    """
    lam_cm = lam_nm * 1e-7
    return lam_cm ** 2 * MU_B_OVER_HC * 1e8


def rms_shift(components, polarisation="both"):
    """Intensity-weighted RMS component shift, in mu_B*B/h units."""
    sel = [(x, s) for x, s, p in components
           if polarisation == "both" or p == polarisation]
    w = np.array([s for _, s in sel], dtype=float)
    x = np.array([xx for xx, _ in sel], dtype=float)
    return float(np.sqrt(np.sum(w * x ** 2) / np.sum(w)))


def zeeman_sigma_px_per_tesla(tag, polarisation="both"):
    """Gaussian-equivalent Zeeman sigma in PIXELS per tesla for one line."""
    d = ZEEMAN[tag]
    rms = rms_shift(d["components"], polarisation)
    nm_per_T = rms * angstrom_per_unit_per_tesla(d["lam_nm"]) / 10.0
    return sc.width_nm_to_px(nm_per_T, d["lam_nm"]), rms, nm_per_T


# --- per-frame total Gaussian width, per line --------------------------------

def fit_group_gaussian(x, prof, ped, window, group_lines):
    """
    The group's components as Gaussians SHARING one sigma, on a cubic continuum
    with the H-alpha Voigt held fixed underneath. gamma is deliberately absent:
    this measures the TOTAL Gaussian-equivalent width, which is what a
    quadrature budget needs (the same convention global_T_fit uses).

    Two components per group, not one. Each C "line" is an unresolved pair in
    this data, and fitting a single Gaussian to a pair inflates the width by
    the pair's own splitting - which would then be miscounted as excess and
    turned into a spurious magnetic field. C2's split (18.6 px) is wider than
    C1's (13.3 px), so a single-Gaussian treatment inflates C2 more than C1 and
    manufactures a disagreement between the two lines that is not physical.
    """
    lo, hi = window
    m = (x >= lo) & (x < hi)
    xf, yf, pf = x[m], prof[m], ped[m]
    wf = sc.weights(yf)
    p = Parameters()
    p.add("sigma", value=8.0, min=1.0, max=40.0)
    p.add("shift", value=0.0, min=-12.0, max=12.0)
    for i, ln in enumerate(group_lines):
        idx = int(np.argmin(np.abs(xf - ln["pixel"])))
        h = float(np.max(yf[max(0, idx - 10):idx + 10]) - np.median(yf))
        p.add(f"amp{i}", value=max(h, 1.0) * 18.0, min=0.0)
    for i in range(4):
        p.add(f"c{i}", value=float(np.median(yf)) if i == 0 else 0.0)

    def model(pp):
        out = pf + sc.polynomial_continuum(
            xf, [pp[f"c{i}"].value for i in range(4)], x_ref=0.5 * (lo + hi))
        for i, ln in enumerate(group_lines):
            out = out + sc.gaussian(xf, pp[f"amp{i}"].value,
                                    ln["pixel"] + pp["shift"].value,
                                    pp["sigma"].value)
        return out

    r = minimize(lambda pp: (model(pp) - yf) * wf, p)
    cen = float(np.mean([ln["pixel"] for ln in group_lines])
                + r.params["shift"].value)
    return abs(r.params["sigma"].value), cen, r.redchi


def measure_widths(exp_i, sigma_inst, table):
    """Total Gaussian sigma and excess-over-instrumental, per frame per line."""
    wins = {"C1": sc.C1_WINDOW, "C2": sc.C2_WINDOW}
    groups = {t: sc.group_lines(table, t) for t in ("C1", "C2")}
    rows = []
    for f in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        x, prof = sc.load_profile(exp_i, f)
        ha = hd.fit_ha(x, prof, 0.0, sigma_inst)
        p = ha["params"]
        ped = sc.voigt(x, p["amp"].value, p["cen"].value,
                       p["sigma"].value, p["gamma"].value)
        row = {"frame": f, "n_e": ha["n_e"]}
        for tag in ("C1", "C2"):
            sig, cen, redchi = fit_group_gaussian(x, prof, ped, wins[tag],
                                                  groups[tag])
            exc = np.sqrt(max(sig ** 2 - sigma_inst ** 2, 0.0))
            exc_nm = float(sc.width_px_to_nm(exc, cen)) if exc > 0 else 0.0
            lam = ZEEMAN[tag]["lam_nm"]
            row[tag] = {
                "sigma_px": sig, "cen": cen, "redchi": redchi,
                "excess_px": exc, "excess_nm": exc_nm,
                "v_kms": sc.C_KM_S * exc_nm / lam,
                "T_eV": sc.MC2_C_EV * (exc_nm / lam) ** 2,
            }
        rows.append(row)
    return rows


def T_from_excess_px(exc_px, cen_px, lam_nm):
    if exc_px <= 0:
        return 0.0
    nm = float(sc.width_px_to_nm(exc_px, cen_px))
    return sc.MC2_C_EV * (nm / lam_nm) ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=int, default=559)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("ZEEMAN BROADENING IN THE C II WIDTH BUDGET")
    print("=" * 78)
    print("  B IS NOT MEASURED IN THIS EXPERIMENT. This is a sensitivity")
    print("  analysis: what field would be needed, and what the budget would")
    print("  look like at an assumed field. Nothing here is a correction.")

    lines, meta = sc.load_line_table()
    sigma_inst = (meta or {}).get("sigma_inst_px", sc.SIGMA_INST_PX)
    v_ceiling = sc.C_KM_S * np.sqrt(C_II_IONISATION_EV / sc.MC2_C_EV)
    print(f"\n  sigma_inst = {sigma_inst:.4f} px ({sc.SIGMA_INST_SOURCE})")
    print(f"  C II thermal ceiling at {C_II_IONISATION_EV} eV = "
          f"{v_ceiling:.2f} km/s")

    # ---------------------------------------------------------------- (1)
    banner("1. ZEEMAN WIDTH PER TESLA")
    print("  Intensity-weighted RMS over the pattern components, treated as a")
    print("  Gaussian-equivalent width. That is an approximation: the real")
    print("  pattern is a set of discrete components, and once the splitting")
    print("  approaches the line width the blend is not Gaussian. It is the")
    print("  right leading-order term for a quadrature budget.")
    per_T = {}
    for tag in ("C1", "C2"):
        d = ZEEMAN[tag]
        a_per_unit = angstrom_per_unit_per_tesla(d["lam_nm"])
        print(f"\n  {tag}  ({d['lam_nm']} nm)   "
              f"{a_per_unit:.4f} A per unit per tesla")
        print(f"    {'shift':>8} {'strength':>9} {'pol':>6}")
        for x, s, p in d["components"]:
            print(f"    {x:+8.4f} {s:9d} {p:>6}")
        for pol, label in (("both", "transverse (pi + sigma)"),
                           ("sigma", "longitudinal (sigma only)")):
            px_T, rms, nm_T = zeeman_sigma_px_per_tesla(tag, pol)
            print(f"    {label:<28} RMS shift {rms:.4f} units  ->  "
                  f"{nm_T*10:.4f} A/T = {px_T:.3f} px/T")
            if pol == "both":
                per_T[tag] = px_T
    print(f"\n  Using TRANSVERSE viewing for everything below:")
    print(f"    C1 {per_T['C1']:.3f} px/T,   C2 {per_T['C2']:.3f} px/T")
    print("  Longitudinal viewing would give ~20-26% MORE width per tesla, so")
    print("  the fields inferred below are upper bounds in that sense.")

    # ---------------------------------------------------------------- (2)
    banner("2. FIELD NEEDED TO EXPLAIN THE ENTIRE EXCESS WIDTH")
    print("  Sets sigma_excess = sigma_Zeeman, i.e. attributes ALL of the")
    print("  non-instrumental width to Zeeman and nothing to thermal, Stark,")
    print("  or flow. These are therefore MAXIMUM fields, not estimates.")
    rows = measure_widths(args.exp, sigma_inst, lines)
    print(f"\n  {'frame':>5} {'n_e [cm^-3]':>12} | "
          f"{'sig_C1':>7} {'exc_C1':>7} {'v_C1':>7} {'B_C1[T]':>8} | "
          f"{'sig_C2':>7} {'exc_C2':>7} {'v_C2':>7} {'B_C2[T]':>8}")
    for r in rows:
        c1, c2 = r["C1"], r["C2"]
        b1 = c1["excess_px"] / per_T["C1"]
        b2 = c2["excess_px"] / per_T["C2"]
        r["B_C1"], r["B_C2"] = b1, b2
        print(f"  {r['frame']:5d} {r['n_e']:12.3e} | "
              f"{c1['sigma_px']:7.3f} {c1['excess_px']:7.3f} "
              f"{c1['v_kms']:7.2f} {b1:8.3f} | "
              f"{c2['sigma_px']:7.3f} {c2['excess_px']:7.3f} "
              f"{c2['v_kms']:7.2f} {b2:8.3f}")
    b1a = np.array([r["B_C1"] for r in rows])
    b2a = np.array([r["B_C2"] for r in rows])
    print(f"\n  C1: median {np.median(b1a):.3f} T, range "
          f"{np.min(b1a):.3f}-{np.max(b1a):.3f} T")
    print(f"  C2: median {np.median(b2a):.3f} T, range "
          f"{np.min(b2a):.3f}-{np.max(b2a):.3f} T")
    print("\n  CONSISTENCY TEST - the sharpest thing in this script.")
    print("  The two lines sit in the same plasma and so must see the SAME")
    print("  field. Their Zeeman widths per tesla differ by only 5%, so if the")
    print("  excess were Zeeman, B_C1 and B_C2 would agree to about that.")
    both = (b1a > 0) & (b2a > 0)
    print(f"\n  Restricted to the {int(both.sum())} frames where BOTH lines have a")
    print("  measurable excess (C2 rails at zero in the rest, and a ratio")
    print("  against zero says nothing):")
    if both.sum() >= 2:
        ratio = b2a[both] / b1a[both]
        fr_both = np.array([r["frame"] for r in rows])[both]
        print(f"    {'frame':>6} {'B_C1':>7} {'B_C2':>7} {'ratio':>7}")
        for fr_i, x1, x2, rr in zip(fr_both, b1a[both], b2a[both], ratio):
            print(f"    {fr_i:6d} {x1:7.3f} {x2:7.3f} {rr:7.2f}")
        print(f"\n    median ratio {np.median(ratio):.2f}, "
              f"range {np.min(ratio):.2f}-{np.max(ratio):.2f}  "
              f"(Zeeman predicts 1.00 +/- 0.05)")
        spread = np.max(ratio) / np.min(ratio)
        print(f"    spread is a factor of {spread:.1f}")
        if spread > 3:
            print("    -> the two lines do NOT agree on a field. Whatever")
            print("       broadens them is not a common Zeeman splitting.")
    else:
        print("    too few frames to run the test")

    # ---------------------------------------------------------------- (3)
    banner("3. FRAMES THAT STRADDLE THE C II THERMAL CEILING")
    print(f"  Ceiling = {v_ceiling:.2f} km/s ({C_II_IONISATION_EV} eV), above")
    print("  which the excess cannot be C II thermal motion because the ion")
    print("  no longer exists. How small a field removes the excursion?")
    print(f"\n  {'frame':>5} {'line':>4} {'v_exc':>7} {'over ceiling':>13} "
          f"{'B to erase all':>15} {'B to reach ceiling':>19}")
    any_over = False
    for r in rows:
        for tag in ("C1", "C2"):
            c = r[tag]
            if c["v_kms"] <= v_ceiling:
                continue
            any_over = True
            lam = ZEEMAN[tag]["lam_nm"]
            # sigma in px that corresponds to the ceiling velocity
            ceil_nm = v_ceiling * lam / sc.C_KM_S
            ceil_px = sc.width_nm_to_px(ceil_nm, lam)
            b_all = c["excess_px"] / per_T[tag]
            need = np.sqrt(max(c["excess_px"] ** 2 - ceil_px ** 2, 0.0))
            b_ceil = need / per_T[tag]
            print(f"  {r['frame']:5d} {tag:>4} {c['v_kms']:7.2f} "
                  f"{c['v_kms'] - v_ceiling:+13.2f} {b_all:15.3f} "
                  f"{b_ceil:19.3f}")
    if not any_over:
        print("  (no line-frame combination exceeds the ceiling)")
    else:
        print("\n  'B to reach ceiling' is the field that would pull the excess")
        print("  down to exactly the C II thermal limit - the smallest field")
        print("  that makes the observed width physically allowed without")
        print("  invoking flow or turbulence.")

    # ---------------------------------------------------------------- (4)
    banner("4. BUDGET WITH ZEEMAN SUBTRACTED AT ASSUMED FIELDS")
    print("  sigma_resid^2 = sigma_excess^2 - sigma_Zeeman(B)^2, then T from")
    print("  the residual. 'over' means Zeeman alone exceeds the measured")
    print("  excess at that field, i.e. that field is already too large.")
    for tag in ("C1", "C2"):
        lam = ZEEMAN[tag]["lam_nm"]
        print(f"\n  --- {tag} ({lam} nm), {per_T[tag]:.3f} px/T " + "-" * 30)
        head = f"  {'frame':>5} {'exc_px':>7} {'T(B=0)':>8}"
        for B in FIELDS_T:
            head += f" | {'B=' + format(B, '.1f'):>15}"
        print(head)
        sub = f"  {'':5} {'':7} {'[eV]':>8}"
        for _ in FIELDS_T:
            sub += f" | {'resid_px':>7} {'T[eV]':>7}"
        print(sub)
        for r in rows:
            c = r[tag]
            line = (f"  {r['frame']:5d} {c['excess_px']:7.3f} "
                    f"{c['T_eV']:8.1f}")
            for B in FIELDS_T:
                z = B * per_T[tag]
                if z >= c["excess_px"]:
                    line += f" | {'over':>7} {'over':>7}"
                else:
                    res = np.sqrt(c["excess_px"] ** 2 - z ** 2)
                    line += (f" | {res:7.3f} "
                             f"{T_from_excess_px(res, c['cen'], lam):7.1f}")
            print(line)

    # ---------------------------------------------------------------- plot
    if not args.no_plot:
        fr = [r["frame"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.plot(fr, b1a, "o-", color="tab:green", label="B from C1")
        ax.plot(fr, b2a, "s-", color="tab:orange", label="B from C2")
        ax.set_xlabel("Frame number")
        ax.set_ylabel("B required [T]")
        ax.set_title("Field that would explain the ENTIRE excess width\n"
                     "(upper bound - assumes no thermal/Stark/flow)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        ax = axes[1]
        for tag, col in (("C1", "tab:green"), ("C2", "tab:orange")):
            ax.plot(fr, [r[tag]["v_kms"] for r in rows], "o-", color=col,
                    label=f"{tag} excess")
            for B, ls in zip(FIELDS_T, (":", "-.", "--", "-")):
                z = B * per_T[tag]
                resid = []
                for r in rows:
                    e = r[tag]["excess_px"]
                    rp = np.sqrt(max(e ** 2 - z ** 2, 0.0))
                    nm = float(sc.width_px_to_nm(rp, r[tag]["cen"]))
                    resid.append(sc.C_KM_S * nm / ZEEMAN[tag]["lam_nm"])
                if tag == "C1":
                    ax.plot(fr, resid, ls, color=col, alpha=0.5, lw=1,
                            label=f"C1 after B={B} T")
        ax.axhline(v_ceiling, color="k", ls="--", lw=1,
                   label=f"C II ceiling {v_ceiling:.1f} km/s")
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Excess width [km/s]")
        ax.set_title("Excess before and after removing Zeeman")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.show()

    banner("READ THIS BEFORE USING ANY NUMBER ABOVE")
    print("  * B is UNMEASURED. Every field here is inferred by assuming")
    print("    Zeeman is the only non-instrumental term, which is false - the")
    print("    same width also has thermal, Stark and flow contributions.")
    print("    The fields in section 2 are therefore UPPER BOUNDS.")
    print("  * The Gaussian-equivalent RMS treatment is leading-order only.")
    print("    A real Zeeman blend is a discrete pattern, and once the")
    print("    splitting is comparable to the line width it is not Gaussian")
    print("    and does not add in quadrature cleanly.")
    print("  * The viewing geometry relative to B is not recorded. Transverse")
    print("    is assumed; longitudinal would change the per-tesla widths by")
    print("    20-26%.")
    print("  * Nothing here may enter spectro_core.py or any production width")
    print("    budget until Sharon supplies a measured or modelled field.")
    print("\n  QUESTION FOR SHARON: is there a field in this device at all, and")
    print("  if so what magnitude and what orientation to the line of sight?")
    print("  Section 2 says what would be needed; that is the number to check")
    print("  against the machine, and it is cheap to rule in or out.")


if __name__ == "__main__":
    main()
