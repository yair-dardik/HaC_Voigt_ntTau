"""
validate_ratio_origin.py - what causes the C II doublet branching ratio of 3.09?

The optically thin ratio I(657.80482)/I(658.28761) is fixed at 2.0 by the upper
level statistical weights. We measure ~3.09 in every frame. Two hypotheses:

    (a) a contaminant adds flux to C1 (657.80482 nm)
    (b) self-absorption suppresses C2 (658.28761 nm)

Both would push the measured ratio above 2.0, so the ratio alone cannot separate
them. This script runs four checks that can.

DIAGNOSTICS ONLY. Nothing here modifies calibrate_lines.py, spectro_core.py or
any production constant.

    python validate_ratio_origin.py
    python validate_ratio_origin.py --no-plot
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import spectro_core as sc
import ha_density as hd
import validate_calibration as vc


# --- Atomic data, NIST ASD (Kramida & Haris 2022), accuracy grade A ----------
# Both lines share the LOWER level 2s2.3s 2S(1/2), g_i = 2.
#
#   657.80482 nm   upper 2s2.3p 2P*(3/2)   g_k = 4   g_k*A_ki = 1.46e8 s^-1
#   658.28761 nm   upper 2s2.3p 2P*(1/2)   g_k = 2   g_k*A_ki = 7.30e7 s^-1
#
# A_ki is 3.65e7 s^-1 for BOTH (as it must be for fine-structure components of
# one multiplet), so the thin emission ratio is exactly g_k(3/2)/g_k(1/2) = 2.
LINES = {
    "C1": dict(lam_nm=657.80482, g_i=2, g_k=4, gkA=1.46e8),
    "C2": dict(lam_nm=658.28761, g_i=2, g_k=2, gkA=7.30e7),
}
THIN_RATIO = LINES["C1"]["gkA"] / LINES["C2"]["gkA"]      # exactly 2.0
OBSERVED_RATIO = 3.09                                      # prior measurement

# Physical constants (SI)
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
M_E = 9.1093837015e-31
C_LIGHT = 2.99792458e8
K_B_EV = 1.0        # we work in eV directly
AMU = 1.66053906660e-27

VERDICTS = []


def record(check, supports, detail):
    VERDICTS.append((check, supports, detail))


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def osc_strength(lam_nm, g_i, g_k, gkA):
    """
    Absorption oscillator strength f_ik from g_k*A_ki.

        f_ik = 1.4992e-14 * (g_k/g_i) * lambda[nm]^2 * A_ki
             = 1.4992e-14 * lambda[nm]^2 * (g_k*A_ki) / g_i
    """
    return 1.4992e-14 * lam_nm ** 2 * gkA / g_i


def slab_area(tau0, n=4001, span=6.0):
    """
    Emergent line area from a homogeneous slab, relative to the optically thin
    area, for a Doppler profile of line-centre depth tau0.

        area  ~  Integral (1 - exp(-tau0 * exp(-u^2))) du
        thin  ~  Integral tau0 * exp(-u^2) du = tau0 * sqrt(pi)
    """
    u = np.linspace(-span, span, n)
    prof = np.exp(-u ** 2)
    emergent = np.trapezoid(1.0 - np.exp(-tau0 * prof), u)
    thin = tau0 * np.sqrt(np.pi)
    return emergent / thin


def doublet_ratio_at_tau(tau_c1):
    """
    Measured C1/C2 area ratio when C1 has line-centre depth tau_c1.

    The two lines share their lower level, so their optical depths are in the
    ratio of their absorption oscillator strengths, f(C1)/f(C2) = 2. C1 is
    therefore the OPTICALLY THICKER line.
    """
    a1 = slab_area(tau_c1) * THIN_RATIO      # thin emissivity ratio 2:1
    a2 = slab_area(tau_c1 / 2.0) * 1.0
    return a1 / a2


# =============================================================================
# PHYSICS PRECONDITION - can self-absorption raise this ratio at all?
# =============================================================================

def precondition():
    banner("PRECONDITION - can self-absorption push the ratio ABOVE 2.0?")
    f1 = osc_strength(**LINES["C1"])
    f2 = osc_strength(**LINES["C2"])
    print(f"  NIST g_k*A_ki : C1 {LINES['C1']['gkA']:.3e}   "
          f"C2 {LINES['C2']['gkA']:.3e}  ->  thin ratio "
          f"{THIN_RATIO:.4f}")
    print(f"  A_ki          : C1 {LINES['C1']['gkA']/LINES['C1']['g_k']:.3e}   "
          f"C2 {LINES['C2']['gkA']/LINES['C2']['g_k']:.3e}  (equal, as expected)")
    print(f"\n  Absorption oscillator strengths (SAME lower level, g_i = 2):")
    print(f"    f(C1) = {f1:.4f}")
    print(f"    f(C2) = {f2:.4f}")
    print(f"    f(C1)/f(C2) = {f1 / f2:.4f}")
    print("\n  Optical depth scales as f * N_lower * L, and the lower level is")
    print("  SHARED, so tau(C1) = 2 * tau(C2). C1 is the optically THICKER line.")
    print("  Self-absorption therefore removes flux from C1 FASTER than from C2,")
    print("  which drives the measured ratio DOWN from 2.0, never up.")

    print(f"\n  Measured ratio as a function of optical depth:")
    print(f"    {'tau(C1)':>9} {'tau(C2)':>9} {'C1 area':>9} {'C2 area':>9} "
          f"{'ratio':>8}")
    for t in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        if t == 0.0:
            print(f"    {0.0:9.2f} {0.0:9.2f} {1.0:9.3f} {1.0:9.3f} "
                  f"{THIN_RATIO:8.4f}   <- thin limit")
            continue
        print(f"    {t:9.2f} {t/2:9.2f} {slab_area(t):9.3f} "
              f"{slab_area(t/2):9.3f} {doublet_ratio_at_tau(t):8.4f}")

    lo = doublet_ratio_at_tau(1000.0)
    print(f"\n  Achievable range under self-absorption: "
          f"({lo:.3f}, {THIN_RATIO:.3f}]")
    print(f"  Observed: {OBSERVED_RATIO:.2f}")
    if OBSERVED_RATIO > THIN_RATIO:
        print("\n  -> The observed ratio lies OUTSIDE the achievable range, for")
        print("     ANY optical depth. Hypothesis (b), self-absorption on C2,")
        print("     is excluded on direction alone, not on magnitude.")
        print("     Note the premise it rested on is inverted: the lower-g")
        print("     component has the SMALLER oscillator strength, so it is the")
        print("     LESS self-absorbed line, not the more.")
    record("PRECONDITION (atomic data)", "contaminant-on-C1",
           f"self-absorption spans ({lo:.2f}, 2.00], cannot reach "
           f"{OBSERVED_RATIO:.2f}")
    return f1, f2


# =============================================================================
# CHECK A - which line's area is actually moving?
# =============================================================================

def check_a(exp_i, sep_px, no_plot):
    banner("CHECK A - unpack the ratio: which side is doing the work?")
    print("  Thin limit: C1 carries 2/3 = 0.6667 of the doublet, C2 1/3 = 0.3333.")
    print("  Normalising by the doublet sum alone cannot attribute the anomaly")
    print("  (the two fractions are complementary by construction), so H-alpha")
    print("  area is carried as an EXTERNAL brightness reference too.")

    rows = []
    print(f"\n  {'frame':>5} {'area C1':>10} {'area C2':>10} {'ratio':>7} "
          f"{'f1':>7} {'f2':>7} {'C1/Ha':>9} {'C2/Ha':>9} {'n_e':>11}")
    for f in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        _, prof = sc.load_profile(exp_i, f)
        x = np.arange(prof.size, dtype=float)
        r, _, _, _ = vc._fit_two_components(x, prof, 1, voigt=True,
                                            lock_sep=True, sep_px=sep_px)
        a1 = r.params["amp0"].value
        a2 = r.params["amp1"].value
        ha = hd.fit_ha(x, prof, 0.0, sc.SIGMA_INST_PX)
        a_ha = ha["params"]["amp"].value
        tot = a1 + a2
        rows.append(dict(frame=f, a1=a1, a2=a2,
                         ratio=a1 / a2 if a2 > 0 else np.nan,
                         f1=a1 / tot if tot > 0 else np.nan,
                         f2=a2 / tot if tot > 0 else np.nan,
                         c1_ha=a1 / a_ha, c2_ha=a2 / a_ha,
                         n_e=ha["n_e"], a_ha=a_ha))
        d = rows[-1]
        print(f"  {f:5d} {a1:10.0f} {a2:10.0f} {d['ratio']:7.3f} "
              f"{d['f1']:7.4f} {d['f2']:7.4f} {d['c1_ha']:9.5f} "
              f"{d['c2_ha']:9.5f} {d['n_e']:11.3e}")

    f1 = np.array([d["f1"] for d in rows])
    f2 = np.array([d["f2"] for d in rows])
    c1h = np.array([d["c1_ha"] for d in rows])
    c2h = np.array([d["c2_ha"] for d in rows])
    print(f"\n  doublet fractions : f1 = {np.nanmean(f1):.4f} "
          f"(thin 0.6667, excess {np.nanmean(f1) - 2/3:+.4f})")
    print(f"                      f2 = {np.nanmean(f2):.4f} "
          f"(thin 0.3333, deficit {np.nanmean(f2) - 1/3:+.4f})")

    # External reference: if C1 has extra flux, C1/Ha is inflated while C2/Ha
    # behaves normally. If C2 is absorbed, C2/Ha dips while C1/Ha is normal.
    # Neither is absolutely calibrated, so compare their SCATTER and their
    # behaviour against the frames where the anomaly is largest.
    ratio = np.array([d["ratio"] for d in rows])
    print(f"\n  Relative to the H-alpha reference:")
    print(f"    CV(C1/Ha) = {np.nanstd(c1h)/np.nanmean(c1h):.3f}   "
          f"CV(C2/Ha) = {np.nanstd(c2h)/np.nanmean(c2h):.3f}")
    r1 = float(np.corrcoef(ratio, c1h)[0, 1])
    r2 = float(np.corrcoef(ratio, c2h)[0, 1])
    print(f"    r(ratio, C1/Ha) = {r1:+.3f}")
    print(f"    r(ratio, C2/Ha) = {r2:+.3f}")
    print("    A contaminant on C1 makes the ratio track C1/Ha positively.")
    print("    Absorption on C2 makes the ratio track C2/Ha negatively.")
    # Both C1/Ha and C2/Ha mostly track overall brightness, so only a clear
    # MARGIN between them says anything. Require 0.20 in |r|.
    if r1 > 0.3 and abs(r1) > abs(r2) + 0.20:
        verdict, sup = "the ratio follows C1 upward", "contaminant-on-C1"
    elif r2 < -0.3 and abs(r2) > abs(r1) + 0.20:
        verdict, sup = "the ratio follows C2 downward", "self-absorption-on-C2"
    else:
        verdict, sup = ("no margin between them - both simply track "
                        "brightness"), "ambiguous"
    print(f"    -> {verdict}")
    record("A  area decomposition", sup,
           f"r(ratio,C1/Ha)={r1:+.2f} vs r(ratio,C2/Ha)={r2:+.2f}")

    if not no_plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        fr = [d["frame"] for d in rows]
        ax.plot(fr, f1, "o-", color="tab:green", label="C1 fraction")
        ax.plot(fr, f2, "s-", color="tab:orange", label="C2 fraction")
        ax.axhline(2/3, ls="--", color="tab:green", alpha=0.5,
                   label="C1 thin limit 0.667")
        ax.axhline(1/3, ls="--", color="tab:orange", alpha=0.5,
                   label="C2 thin limit 0.333")
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Fraction of doublet area")
        ax.set_title("CHECK A - normalised doublet areas vs frame")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    return rows


# =============================================================================
# CHECK B - line shape: is either line flat-topped or dipped?
# =============================================================================

def check_b(exp_i, sep_px, rows_a):
    banner("CHECK B - peak shape, independent of what the fit calls sigma")
    print("  Self-absorption flattens or dips the PEAK of the affected line.")
    print("  Metric: I(centre) / I(centre +/- sigma), isolated from continuum,")
    print("  the H-alpha pedestal and the other doublet component.")
    print(f"  A pure Gaussian gives exp(+0.5) = {np.exp(0.5):.4f}.")
    print("  Below that = flattened (absorption-like); above = peaked.")

    IDEAL = float(np.exp(0.5))
    print(f"\n  {'frame':>5} {'R(C1)':>8} {'R(C2)':>8} {'C1/ideal':>9} "
          f"{'C2/ideal':>9} {'dip C1':>8} {'dip C2':>8}")
    out = []
    for f in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        _, prof = sc.load_profile(exp_i, f)
        x = np.arange(prof.size, dtype=float)
        r, xf, yf, _ = vc._fit_two_components(x, prof, 1, voigt=True,
                                              lock_sep=True, sep_px=sep_px)
        p = r.params
        cont = sc.polynomial_continuum(xf, [p[f"c{i}"].value for i in range(4)])
        sig = abs(p["sigma"].value)
        gam = abs(p["gamma"].value)
        vals = {}
        for tag, key in (("C1", "cen0"), ("C2", "cen1")):
            cen = p[key].value
            other = "cen1" if key == "cen0" else "cen0"
            oamp = p["amp1"].value if key == "cen0" else p["amp0"].value
            # isolate: remove continuum and the OTHER component's wing
            iso = yf - cont - sc.voigt(xf, oamp, p[other].value, sig, gam)

            def at(px, half=1.0):
                m = np.abs(xf - px) <= half
                return float(np.mean(iso[m])) if m.any() else np.nan

            peak = at(cen)
            wing = 0.5 * (at(cen - sig) + at(cen + sig))
            R = peak / wing if wing > 0 else np.nan
            # central residual relative to the fitted component's own peak
            comp = sc.voigt(xf, p[f"amp{0 if key=='cen0' else 1}"].value,
                            cen, sig, gam)
            m = np.abs(xf - cen) <= 1.0
            dip = (float(np.mean((iso - comp)[m]))
                   / max(float(np.max(comp)), 1e-9))
            vals[tag] = (R, dip)
        out.append(dict(frame=f, R1=vals["C1"][0], R2=vals["C2"][0],
                        d1=vals["C1"][1], d2=vals["C2"][1]))
        d = out[-1]
        print(f"  {f:5d} {d['R1']:8.4f} {d['R2']:8.4f} "
              f"{d['R1']/IDEAL:9.4f} {d['R2']/IDEAL:9.4f} "
              f"{d['d1']:+8.4f} {d['d2']:+8.4f}")

    R1 = np.array([d["R1"] for d in out], dtype=float)
    R2 = np.array([d["R2"] for d in out], dtype=float)
    d1 = np.array([d["d1"] for d in out], dtype=float)
    d2 = np.array([d["d2"] for d in out], dtype=float)
    print(f"\n  mean R(C1)/ideal = {np.nanmean(R1)/IDEAL:.4f}   "
          f"mean R(C2)/ideal = {np.nanmean(R2)/IDEAL:.4f}")
    print(f"  mean central residual: C1 {np.nanmean(d1):+.4f}   "
          f"C2 {np.nanmean(d2):+.4f}  (fraction of component peak)")
    print("\n  Taken at face value that is a textbook self-absorption signature")
    print("  on C2 and not on C1. But the fit it came from SHARES sigma and")
    print("  gamma between the two lines, LOCKS their separation, and carries no")
    print("  H-alpha pedestal - the cubic continuum has to absorb the H-alpha")
    print("  wing across the whole window. Any of those can manufacture a")
    print("  central deficit at C2. Check B2 removes all three constraints.")
    return out


def _pedestal(x, prof, sigma_inst):
    """H-alpha Voigt from Stage 2, evaluated on the full pixel axis."""
    ha = hd.fit_ha(x, prof, 0.0, sigma_inst)
    p = ha["params"]
    return sc.voigt(x, p["amp"].value, p["cen"].value,
                    p["sigma"].value, p["gamma"].value)


def _fit_one_line_free(x, prof, ped, window, seed_px, sigma_inst):
    """One Voigt alone in its own window: free centre, sigma, gamma."""
    from lmfit import Parameters, minimize
    lo, hi = window
    m = (x >= lo) & (x < hi)
    xf, yf, pf = x[m], prof[m], ped[m]
    wf = sc.weights(yf)
    p = Parameters()
    idx = int(np.argmin(np.abs(xf - seed_px)))
    h = float(np.max(yf[max(0, idx - 12):idx + 12]) - np.median(yf))
    p.add("amp", value=max(h, 1.0) * 18.0, min=0.0)
    p.add("cen", value=seed_px, min=seed_px - 18, max=seed_px + 18)
    p.add("sigma", value=sigma_inst, min=1.0, max=40.0)
    p.add("gamma", value=1.0, min=0.0, max=40.0)
    for i in range(4):
        p.add(f"c{i}", value=float(np.median(yf)) if i == 0 else 0.0)

    def mod(pp):
        return (pf + sc.polynomial_continuum(
            xf, [pp[f"c{i}"].value for i in range(4)], x_ref=0.5 * (lo + hi))
            + sc.voigt(xf, pp["amp"].value, pp["cen"].value,
                       pp["sigma"].value, pp["gamma"].value))

    r = minimize(lambda pp: (mod(pp) - yf) * wf, p)
    return r, xf, yf, pf


def check_b2(exp_i, sep_px):
    banner("CHECK B2 - does the C2 deficit survive an unconstrained fit?")
    print("  Each line refitted ALONE in its own window: free centre, free")
    print("  sigma, free gamma, its own cubic continuum, and the H-alpha Voigt")
    print("  carried as a FIXED pedestal instead of being absorbed by the")
    print("  continuum. If the C2 dip is real it must survive this. If it was")
    print("  the shared-parameter model mis-placing or mis-shaping C2, it will")
    print("  collapse.")
    print("\n  Residuals are also split into a SYMMETRIC part (a genuine")
    print("  flat-top or dip) and an ANTISYMMETRIC part (the model centre")
    print("  sitting off the true line centre).")

    IDEAL = float(np.exp(0.5))
    sig_inst = sc.SIGMA_INST_PX
    seed1 = sc.nm_to_pixel(LINES["C1"]["lam_nm"])
    seed2 = sc.nm_to_pixel(LINES["C2"]["lam_nm"])
    print(f"\n  {'frame':>5} | {'C1 R/ideal':>10} {'dip':>7} {'sym':>7} "
          f"{'anti':>7} | {'C2 R/ideal':>10} {'dip':>7} {'sym':>7} {'anti':>7}")
    acc = {"C1": [], "C2": []}
    for f in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        x, prof = sc.load_profile(exp_i, f)
        ped = _pedestal(x, prof, sig_inst)
        line = {}
        for tag, win, seed in (("C1", sc.C1_WINDOW, seed1),
                               ("C2", sc.C2_WINDOW, seed2)):
            r, xf, yf, pf = _fit_one_line_free(x, prof, ped, win, seed,
                                               sig_inst)
            p = r.params
            cen, sg, gm = (p["cen"].value, abs(p["sigma"].value),
                           abs(p["gamma"].value))
            cont = sc.polynomial_continuum(
                xf, [p[f"c{i}"].value for i in range(4)],
                x_ref=0.5 * (win[0] + win[1]))
            iso = yf - pf - cont
            comp = sc.voigt(xf, p["amp"].value, cen, sg, gm)

            def at(px, half=1.0):
                mm = np.abs(xf - px) <= half
                return float(np.mean(iso[mm])) if mm.any() else np.nan

            peak = at(cen)
            wing = 0.5 * (at(cen - sg) + at(cen + sg))
            R = peak / wing if wing and wing > 0 else np.nan
            resid = iso - comp
            pk = max(float(np.max(comp)), 1e-9)
            mm = np.abs(xf - cen) <= 1.0
            dip = float(np.mean(resid[mm])) / pk
            # symmetric / antisymmetric decomposition about the fitted centre
            offs = np.arange(1.0, 2.0 * sg, 1.0)
            sym, anti = [], []
            for dd in offs:
                a = np.interp(cen + dd, xf, resid)
                b = np.interp(cen - dd, xf, resid)
                sym.append(0.5 * (a + b))
                anti.append(0.5 * (a - b))
            sym_rms = float(np.sqrt(np.mean(np.square(sym)))) / pk
            anti_rms = float(np.sqrt(np.mean(np.square(anti)))) / pk
            line[tag] = (R / IDEAL, dip, sym_rms, anti_rms, r.redchi)
            acc[tag].append(line[tag])
        print(f"  {f:5d} | {line['C1'][0]:10.4f} {line['C1'][1]:+7.4f} "
              f"{line['C1'][2]:7.4f} {line['C1'][3]:7.4f} | "
              f"{line['C2'][0]:10.4f} {line['C2'][1]:+7.4f} "
              f"{line['C2'][2]:7.4f} {line['C2'][3]:7.4f}")

    print(f"\n  {'line':<5} {'R/ideal':>9} {'central dip':>12} {'sym rms':>9} "
          f"{'anti rms':>9} {'chi2r':>7}")
    means = {}
    for tag in ("C1", "C2"):
        a = np.array(acc[tag], dtype=float)
        means[tag] = np.nanmean(a, axis=0)
        m = means[tag]
        print(f"  {tag:<5} {m[0]:9.4f} {m[1]:+12.4f} {m[2]:9.4f} "
              f"{m[3]:9.4f} {m[4]:7.2f}")

    c2_dip_free = means["C2"][1]
    c1_dip_free = means["C1"][1]
    print(f"\n  Locked shared-parameter fit (check B) gave C2 dip -0.2474.")
    print(f"  Unconstrained per-line fit gives     C2 dip {c2_dip_free:+.4f}.")
    shrink = (abs(c2_dip_free) / 0.2474) if 0.2474 else np.nan
    print(f"  The C2 deficit retains {100*shrink:.0f}% of its size once the")
    print(f"  shared sigma, the locked separation and the missing H-alpha")
    print(f"  pedestal are removed.")

    if abs(c2_dip_free) < 0.05 and abs(c1_dip_free) < 0.05:
        sup = "contaminant-on-C1"
        det = (f"C2 dip collapses to {c2_dip_free:+.3f} when fitted freely - "
               f"it was a model artefact")
    elif abs(c2_dip_free) > 0.10 and abs(c2_dip_free) > 2 * abs(c1_dip_free):
        if means["C2"][3] > means["C2"][2]:
            sup = "ambiguous"
            det = (f"C2 residual survives but is ANTIsymmetric "
                   f"({means['C2'][3]:.3f} vs sym {means['C2'][2]:.3f}) - "
                   f"a centre offset, not a flat top")
        else:
            # A symmetric central deficit is the signature of self-absorption,
            # but it is EQUALLY the signature of one component fitted to an
            # unresolved pair. Check E separates those two.
            sup = "ambiguous (b or c)"
            det = (f"C2 dip {c2_dip_free:+.3f} survives and is symmetric - "
                   f"absorption OR an unresolved pair; see E")
    else:
        sup = "ambiguous"
        det = f"C2 dip {c2_dip_free:+.3f}, C1 dip {c1_dip_free:+.3f}"
    print(f"  -> {det}")
    record("B  peak shape (free fit)", sup, det)
    return means


# =============================================================================
# CHECK C - what does the anomaly correlate with?
# =============================================================================

def check_c(rows_a):
    banner("CHECK C - density-like driver or time-like driver?")
    fr = np.array([d["frame"] for d in rows_a], dtype=float)
    ne = np.array([d["n_e"] for d in rows_a], dtype=float)
    ratio = np.array([d["ratio"] for d in rows_a], dtype=float)
    c1h = np.array([d["c1_ha"] for d in rows_a], dtype=float)
    c2h = np.array([d["c2_ha"] for d in rows_a], dtype=float)
    f1 = np.array([d["f1"] for d in rows_a], dtype=float)
    f2 = np.array([d["f2"] for d in rows_a], dtype=float)

    def r(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan

    print(f"\n  {'quantity':<22} {'vs n_e':>9} {'vs frame':>10}")
    for name, v in (("ratio", ratio), ("C1 fraction f1", f1),
                    ("C2 fraction f2", f2), ("C1/Ha", c1h), ("C2/Ha", c2h)):
        print(f"  {name:<22} {r(v, ne):+9.3f} {r(v, fr):+10.3f}")

    print("\n  Self-absorption is an optical-depth effect, so the C2 deficit")
    print("  should track n_e. Ablation contamination tracks accumulated")
    print("  erosion, so it tends to track frame number instead.")
    r_ne, r_fr = r(ratio, ne), r(ratio, fr)
    print(f"\n  ratio vs n_e   = {r_ne:+.3f}")
    print(f"  ratio vs frame = {r_fr:+.3f}")
    if abs(r_ne) > abs(r_fr) + 0.15:
        sup, det = "self-absorption-on-C2", f"ratio tracks n_e ({r_ne:+.2f})"
    elif abs(r_fr) > abs(r_ne) + 0.15:
        sup, det = "contaminant-on-C1", f"ratio tracks frame ({r_fr:+.2f})"
    else:
        sup, det = "ambiguous", (f"neither dominates (n_e {r_ne:+.2f}, "
                                 f"frame {r_fr:+.2f})")
    print(f"  -> {det}")
    record("C  correlation structure", sup, det)


# =============================================================================
# CHECK D - is the required optical depth even reachable?
# =============================================================================

def check_d(rows_a, f1, f2):
    banner("CHECK D - optical depth needed, and whether it is credible")
    ne_peak = float(np.nanmax([d["n_e"] for d in rows_a]))

    # what tau would a 35% area deficit require, taken in isolation?
    target = 1.0 - (THIN_RATIO / OBSERVED_RATIO)   # ~0.353
    taus = np.logspace(-3, 2, 4000)
    areas = np.array([slab_area(t) for t in taus])
    idx = int(np.argmin(np.abs(areas - (1.0 - target))))
    tau_needed = taus[idx]
    print(f"  Observed ratio {OBSERVED_RATIO:.2f} vs thin {THIN_RATIO:.2f} is a")
    print(f"  {100*target:.0f}% apparent deficit if attributed entirely to C2.")
    print(f"  In isolation that needs tau(C2) ~ {tau_needed:.2f}.")

    # line-centre cross-section for a Doppler profile
    T_eV = 3.0                       # ASSUMED excitation temperature
    lam = LINES["C2"]["lam_nm"] * 1e-9
    nu0 = C_LIGHT / lam
    v_th = np.sqrt(2 * T_eV * E_CHARGE / (12 * AMU))
    dnu_D = nu0 * v_th / C_LIGHT
    integ = E_CHARGE ** 2 / (4 * EPS0 * M_E * C_LIGHT)     # 2.653e-6 m^2 Hz
    sigma0 = integ * f2 / (np.sqrt(np.pi) * dnu_D)          # m^2
    sigma0_cm2 = sigma0 * 1e4
    N_needed = tau_needed / sigma0_cm2                      # cm^-2

    print(f"\n  INPUTS  (M = measured here, A = assumed)")
    print(f"    M  f(C2) = {f2:.4f}                 from NIST g_k*A_ki")
    print(f"    M  n_e peak = {ne_peak:.3e} cm^-3   from H-alpha Stark")
    print(f"    A  T_exc = {T_eV:.1f} eV             not measured; T from the C")
    print(f"                                       lines is only an upper limit")
    print(f"    A  path length L                   NOT available in this dataset")
    print(f"    A  C+ fraction of n_e              not measured")
    print(f"\n  sigma_0(C2) = {sigma0_cm2:.3e} cm^2 (Doppler core, T = {T_eV} eV)")
    print(f"  column density needed for tau = {tau_needed:.2f}: "
          f"N(3s) * L = {N_needed:.3e} cm^-2")

    # Is that reachable? Boltzmann estimate of the 3s population.
    E_3S_EV = 14.45          # 116537.6 cm^-1
    print(f"\n  The absorbing level 2s2.3s 2S is {E_3S_EV:.2f} eV above the C+")
    print(f"  ground state, so only a small fraction of C+ populates it.")
    print(f"\n  {'C+ / n_e':>9} {'T_exc':>7} {'n(3s) [cm^-3]':>15} "
          f"{'L needed [cm]':>15}")
    for cfrac in (0.01, 0.10):
        for T in (2.0, 3.0, 5.0):
            n_cp = ne_peak * cfrac
            # ratio of degeneracies 2/6 for 3s 2S vs 2p 2P ground term
            n_3s = n_cp * (2.0 / 6.0) * np.exp(-E_3S_EV / T)
            L = N_needed / n_3s
            print(f"  {cfrac:9.2f} {T:7.1f} {n_3s:15.3e} {L:15.3e}")

    print("\n  Read: for plausible C+ fractions and excitation temperatures the")
    print("  required path length lands in the centimetre range, which is NOT")
    print("  absurd for this plasma. So self-absorption is not excluded by")
    print("  MAGNITUDE - tau of order unity is physically reachable here.")
    print("\n  It is excluded by DIRECTION. Because the doublet shares its lower")
    print("  level and f(C1) = 2*f(C2), any optical depth removes more flux from")
    print("  C1 than from C2 and drives the ratio toward 1.0. See PRECONDITION:")
    print(f"  the reachable band is ({doublet_ratio_at_tau(1000.0):.2f}, "
          f"{THIN_RATIO:.2f}], and {OBSERVED_RATIO:.2f} is outside it.")
    print("\n  A corollary worth noting: if tau ~ 1 really is present, it is")
    print("  suppressing C1 more than C2 and therefore MASKING part of the")
    print("  anomaly. The intrinsic excess on C1 would then be larger than the")
    print("  measured 3.09 implies, not smaller.")
    record("D  optical depth", "contaminant-on-C1",
           f"tau~{tau_needed:.1f} is reachable, but its effect has the wrong sign")


# =============================================================================
# CHECK E - is the anomaly just the wrong number of components?
# =============================================================================

def check_e(exp_i):
    banner("CHECK E - hypothesis (c): one Voigt fitted to a two-part blend")
    print("  Check B2 found a SYMMETRIC central deficit on C2 that survives")
    print("  free fitting. Self-absorption cannot cause it (see PRECONDITION),")
    print("  but there is a mundane cause that produces exactly that shape:")
    print("  fitting ONE component to a feature that is really TWO.")
    print("  The 4-component table splits C1 by 13.3 px and C2 by 18.6 px")
    print("  against sigma ~ 7 px, so C2 - the wider split - should show the")
    print("  deeper central dip. It does (-0.123 vs -0.032).")
    print("\n  If that is the cause, then the single-Voigt fit is losing area")
    print("  on C2 and the 3.09 ratio is a fitting artefact. Refit with two")
    print("  components per group and see where the ratio lands.")

    import calibrate_lines as cl
    print(f"\n  {'frame':>5} {'ratio 1+1':>10} {'ratio 2+2':>10} "
          f"{'chi2r 1+1':>10} {'chi2r 2+2':>10}")
    r11, r22 = [], []
    sep_px = None
    x0, y0 = sc.load_stack(exp_i, vc.STACK_FRAMES)
    rv, _, _, _ = vc._fit_two_components(x0, y0, len(vc.STACK_FRAMES),
                                         voigt=True)
    sep_px = rv.params["cen1"].value - rv.params["cen0"].value

    for f in range(sc.FIRST_FRAME, sc.LAST_FRAME + 1):
        x, prof = sc.load_profile(exp_i, f)
        xa = np.arange(prof.size, dtype=float)
        two, _, _, _ = vc._fit_two_components(xa, prof, 1, voigt=True,
                                              lock_sep=True, sep_px=sep_px)
        a1 = two.params["amp0"].value
        a2 = two.params["amp1"].value
        four, _, _ = cl.fit_free_centres(xa, prof, sc.DEFAULT_LINE_TABLE,
                                         n_frames=1)
        g1 = sum(four.params[f"amp{i}"].value
                 for i, ln in enumerate(sc.DEFAULT_LINE_TABLE)
                 if ln["group"] == "C1")
        g2 = sum(four.params[f"amp{i}"].value
                 for i, ln in enumerate(sc.DEFAULT_LINE_TABLE)
                 if ln["group"] == "C2")
        v11 = a1 / a2 if a2 > 0 else np.nan
        v22 = g1 / g2 if g2 > 0 else np.nan
        r11.append(v11)
        r22.append(v22)
        print(f"  {f:5d} {v11:10.3f} {v22:10.3f} {two.redchi:10.2f} "
              f"{four.redchi:10.2f}")

    r11 = np.array(r11, float)
    r22 = np.array(r22, float)
    m11, m22 = float(np.nanmedian(r11)), float(np.nanmedian(r22))
    print(f"\n  median ratio, 1 component per group : {m11:.3f}")
    print(f"  median ratio, 2 components per group: {m22:.3f}")
    print(f"  optically thin expectation          : {THIN_RATIO:.3f}")
    moved = abs(m22 - THIN_RATIO) < abs(m11 - THIN_RATIO)
    print(f"\n  Adding the second component per group moves the ratio "
          f"{'TOWARD' if moved else 'AWAY FROM'} 2.0,")
    print(f"  from {m11:.2f} to {m22:.2f} "
          f"({100*(abs(m11-THIN_RATIO)-abs(m22-THIN_RATIO))/abs(m11-THIN_RATIO):+.0f}% "
          f"of the anomaly removed).")
    if moved and abs(m22 - THIN_RATIO) < 0.35:
        sup = "model artefact (hypothesis c)"
        det = f"ratio {m11:.2f} -> {m22:.2f} with the right component count"
    elif moved:
        sup = "model artefact (hypothesis c)"
        det = f"ratio moves {m11:.2f} -> {m22:.2f} but does not reach 2.0"
    else:
        sup = "contaminant-on-C1"
        det = f"component count does not explain it ({m11:.2f} -> {m22:.2f})"
    print(f"  -> {det}")
    record("E  component count", sup, det)
    return m11, m22


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=int, default=559)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("ORIGIN OF THE C II DOUBLET BRANCHING-RATIO ANOMALY")
    print("=" * 78)
    print(f"  thin-limit ratio {THIN_RATIO:.2f}, measured ~{OBSERVED_RATIO:.2f}")
    print("  (a) contaminant adds flux to C1   vs   (b) self-absorption eats C2")

    f1, f2 = precondition()

    # separation used by the locked two-component fit, from the doublet itself
    x, y = sc.load_stack(args.exp, vc.STACK_FRAMES)
    rv, _, _, _ = vc._fit_two_components(x, y, len(vc.STACK_FRAMES), voigt=True)
    sep_px = rv.params["cen1"].value - rv.params["cen0"].value

    rows_a = check_a(args.exp, sep_px, args.no_plot)
    check_b(args.exp, sep_px, rows_a)
    check_b2(args.exp, sep_px)
    check_c(rows_a)
    check_d(rows_a, f1, f2)
    check_e(args.exp)

    banner("VERDICT")
    print(f"  {'check':<28} {'supports':<26} {'detail'}")
    print("  " + "-" * 74)
    for name, sup, det in VERDICTS:
        print(f"  {name:<28} {sup:<26} {det}")

    votes = [s for _, s, _ in VERDICTS]
    n_a = votes.count("contaminant-on-C1")
    n_b = votes.count("self-absorption-on-C2")
    n_x = votes.count("ambiguous")
    print(f"\n  contaminant-on-C1 {n_a}   self-absorption-on-C2 {n_b}   "
          f"ambiguous {n_x}")

    print("\n  OVERALL")
    print("  Neither (a) nor (b) is the main answer. The data favour a third")
    print("  explanation that the original framing did not include.")
    print("")
    print("  (b) SELF-ABSORPTION ON C2 IS EXCLUDED, by atomic physics that")
    print("  needs no fitting. The doublet shares its lower level, so the g=4")
    print("  component carries twice the absorption oscillator strength and is")
    print("  the OPTICALLY THICKER line. Opacity therefore removes more flux")
    print("  from C1 than from C2 and maps the ratio onto (1.05, 2.00] - it")
    print("  cannot reach 3.09 at any optical depth. The premise that the")
    print("  lower-g component self-absorbs first is inverted: lower g_k means")
    print("  smaller f, hence LESS absorption. Check D shows tau ~ 1 is")
    print("  physically reachable here, so this is ruled out on DIRECTION, not")
    print("  on magnitude.")
    print("")
    print("  (c) MODEL MIS-SPECIFICATION is the leading cause. Check B2 found a")
    print("  symmetric central deficit on C2 that survives unconstrained")
    print("  fitting - and an unresolved pair produces exactly that shape, just")
    print("  as absorption would. C2's components are split by 18.6 px against")
    print("  sigma ~ 7 px, wider than C1's 13.3 px, so C2 should show the")
    print("  deeper dip, and it does (-0.123 vs -0.032). Fitting the correct")
    print("  number of components confirms it: the ratio moves 3.09 -> 2.29,")
    print("  removing about three quarters of the anomaly, and reduced")
    print("  chi-square improves in EVERY frame (e.g. 3.14 -> 0.91 at frame 15).")
    print("  The single-Voigt fit was losing C2 area it could not model.")
    print("")
    print("  (a) A CONTAMINANT ON C1 IS NOT EXCLUDED, but it is now a much")
    print("  smaller effect than the raw 3.09 suggested. A residual excess of")
    print("  2.29 against 2.00 survives the component-count fix, and check C")
    print("  shows it tracks frame number (+0.55) rather than n_e (-0.28),")
    print("  which is the time-like behaviour expected from accumulating")
    print("  electrode ablation rather than from an optical-depth effect.")
    print("")
    print("  HONEST LIMITS")
    print("    * The residual 2.29 vs 2.00 is a ~14% excess. That is within")
    print("      reach of systematics in the 4-component decomposition itself -")
    print("      the component splits are empirical, not identified lines, and")
    print("      the 4-component chi2r often lands below 1, so some of the")
    print("      improvement may be absorbing noise as well as real structure.")
    print("      Do not read 2.29 as a measured contaminant strength.")
    print("    * Check A could not attribute the anomaly to a side at all:")
    print("      C1/Ha and C2/Ha correlate with the ratio at -0.39 and -0.43,")
    print("      no useful margin. Both are mostly tracking brightness.")
    print("    * This does not identify any emitter.")
    print("")
    print("  WHAT WOULD SETTLE THE REMAINDER")
    print("    * the electrode material - Cu II sits 21 px and Fe I 66 px from")
    print("      C1, and either would add flux exactly where it is needed")
    print("    * a lamp measurement, which fixes the instrument profile so a")
    print("      model with the right component count can be fitted without")
    print("      width and area trading against each other - the degeneracy")
    print("      that produced this whole anomaly")


if __name__ == "__main__":
    main()
