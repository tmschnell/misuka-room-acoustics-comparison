"""Derived-quantity helpers for the misuka-vs-pyroomacoustics comparison
suite: ISO 3382 descriptors (via pyrato), peak-window energy integration,
and small error metrics. Relocated verbatim from
pyroomacoustics_comparison.ipynb (no numeric changes -- see
pra_comparison_runner.py's module docstring for the structural-refactor
this is part of) so the runner can call them without a notebook kernel.
"""

import warnings

import numpy as np
import pyfar as pf
import pyrato

import test_pyroomacoustics_comparison as cmp


# =====================================================================
#          Minimum ray count for a given detection confidence
# =====================================================================

def necessary_rays(detector_radius, speed_of_sound, t_max, alpha=0.0001):
    r"""Minimum number of independent Monte-Carlo rays/paths so that, with
    confidence (1 - alpha), at least one of them has hit a detector sphere
    of `detector_radius` by propagation time `t_max`:

        N_alpha = log(alpha) / log(1 - (r_d / (2 * c * t))^2)

    Richter, Masterarbeit, Eq. (26)/(69) (Section 2.4, significance level
    of the ray-hit probability). `detector_radius` (r_d) is, by
    reciprocity, misuka's *source* radius -- this suite already treats the
    finite emitter sphere as the relevant detector-side quantity
    throughout (see hull_to_center_shift_seconds/align_hull_to_center in
    test_pyroomacoustics_comparison.py).

    `t_max` is a required argument, not read from any scene's own
    max_time, so a caller can plug in whichever propagation horizon is
    the one that actually needs guaranteed ray coverage: the full
    simulated duration for decay-curve-based metrics (T60/EDT/...), or
    the (typically much shorter) mixing time t_mix = 2*sqrt(V)*1e-3
    s/m^1.5 (Eq. 2) if only the mixed/diffuse-field portion matters. N
    grows roughly with 1/t^2 for t << r_d/(2c), so this choice matters a
    lot in practice (verified for the CR4 auditorium: t_mix gives ~6*10^5
    rays, the full max_time=1.2s gives ~2.5*10^7).

    alpha defaults to 0.0001 (0.01%), not the more obvious-looking 1%:
    Richter's Discussion ("Suggested significance levels") found 1% was
    not sufficient in practice and needed roughly double the corresponding
    ray count (~0.01%) to reach acceptable RMSPE against the reference.
    """
    return np.log(alpha) / np.log(1 - (detector_radius / (2 * speed_of_sound * t_max)) ** 2)


# =====================================================================
#                   ISO 3382 descriptors via pyrato
# =====================================================================
#
# schroeder_integration itself does *not* normalize its output to 0 dB at
# t=0 (unlike the higher-level energy_decay_curve_* functions, which do by
# default) -- it returns the raw backward-integrated energy, on whatever
# absolute scale the input happens to be on. reverberation_time_linear_
# regression looks for *absolute* dB targets (e.g. -5 dB, -25 dB for T20),
# so skipping this normalization silently picks the wrong, meaningless fit
# window instead of raising an error. pyrato_edc below normalizes
# explicitly, the same way the official pyrato docstring examples plot an
# EDC (edc/edc.time[..., 0]).

# ---------------------------------------------------------------------
# Time origin of C50/C80/D50/TS (and G's direct-sound window): the start of
# the impulse response, ISO 3382-1 Annex A -- not the emission time t=0,
# which silently counts the source-receiver propagation delay into the early
# window/centre time (for Gewandhaus R1: 33 ms, i.e. C50 off by ~3 dB, TS by
# ~33 ms). Detection is pyfar's ISO implementation (the one pyrato's own
# time_shift uses): last sample more than 20 dB below the maximum before
# it. If that lands more than IR_START_TOLERANCE_S away from the analytic
# expectation (typically: too few rays hit the direct sound, so a reflection
# was detected instead), the analytic value is used and the fallback is
# recorded -- storage.save_run writes IR_START_LOG into the run's metadata.
IR_START_TOLERANCE_S = 1e-3
IR_START_CONVENTION = "impulse-response start (ISO 3382-1, pyfar find_impulse_response_start, 20 dB)"
IR_START_LOG = []


def ir_start(energy, sampling_rate, expected_start_s=None, tolerance_s=IR_START_TOLERANCE_S):
    """Start sample of an energy signal (ETC, squared RIR or histogram).

    `expected_start_s` is where the direct sound analytically begins in
    *this* signal: misuka unaligned (d - source_radius)/c (hull), misuka
    after align_hull_to_center_lossless d/c, PRA frac-delay-stripped d/c
    (unstripped room.rir: plus frac_delay_length//2 samples), GRAP histogram
    its "Direct sound delay". Returns (start_sample, used_fallback).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pyfar's SNR heuristic assumes measured noise tails
        detected = int(np.ravel(pf.dsp.find_impulse_response_start(
            pf.Signal(np.sqrt(np.maximum(energy, 0.0)), sampling_rate)))[0])
    # pyfar returns the last sample *before* the onset -- compare the onset
    # sample itself, which matters for coarse (1 ms histogram) sampling.
    fallback = (expected_start_s is not None
                and abs((detected + 1) / sampling_rate - expected_start_s) > tolerance_s)
    # same "last sample before the onset" convention as pyfar's detection
    start = max(int(np.floor(expected_start_s * sampling_rate)) - 1, 0) if fallback else detected
    IR_START_LOG.append(dict(detected_s=detected / sampling_rate, expected_s=expected_start_s,
                             fallback=bool(fallback)))
    return start, fallback


def pyrato_edc(energy, sampling_rate=cmp.SAMPLING_RATE):
    edc = pyrato.edc.schroeder_integration(pf.Signal(energy, sampling_rate), is_energy=True)
    return edc / edc.time[..., 0]  # 0 dB at t=0, required by reverberation_time_linear_regression


def pyrato_metrics(energy, expected_start_s=None):
    '''T20, T30, C50, D50 for one energy array, via pyrato (ISO 3382), from
    the impulse-response start (see ir_start; no fallback without
    expected_start_s).'''
    energy = energy[ir_start(energy, cmp.SAMPLING_RATE, expected_start_s)[0]:]
    with warnings.catch_warnings():
        # Some scenarios (by design, see coincident_reflections/flutter_corridor
        # in the .py suite) don't have enough dynamic range for a clean T20/T30
        # fit; pyrato warns and returns a value from a noisy/degenerate fit
        # rather than raising. The report below surfaces those cases via the
        # EDC plots -- suppress the warning spam here, not the underlying issue.
        warnings.simplefilter("ignore")
        edc = pyrato_edc(energy)
        return {
            "T20": float(pyrato.parameters.reverberation_time_linear_regression(edc, "T20")[0]),
            "T30": float(pyrato.parameters.reverberation_time_linear_regression(edc, "T30")[0]),
            "C50": float(pyrato.parameters.clarity(edc, early_time_limit=50)[0]),
            "D50": float(pyrato.parameters.definition(edc, early_time_limit=50)[0]),
        }


def parameters_full(etc, distance, source_radius, sampling_rate=cmp.SAMPLING_RATE,
                    speed_of_sound=cmp.SPEED_OF_SOUND, g_radius_term=None, expected_start_s=None):
    """C50, C80, EDT, T60, T30 via pyrato -- the same pyrato_edc()-based mechanism
    pyrato_metrics() above already uses for T20/T30/C50/D50, extended with
    early_time_limit=80 (C80) and T="EDT"/"T60" (both natively supported by
    reverberation_time_linear_regression). G and TS have no pyrato equivalent
    in the installed version (pyrato.parameters has no centre-time function;
    sound_strength needs a genuine calibrated free-field reference IR at 10 m,
    which no scenario here renders) -- kept as direct energy-domain formulas
    from Richter's Masterarbeit (Eq. 61/63/64); see the "Hybrid (parameter
    level)" section markdown for why G's absolute cross-renderer value is not
    meaningful on its own.

    g_radius_term : float, optional
        How far past `distance/speed_of_sound` the direct-sound window
        (used only for G's free-field-energy proxy) extends, in metres of
        equivalent path length. Defaults to `source_radius` (the original,
        symmetric behaviour: `distance/c + source_radius/c`), which is only
        correct for a caller that has *not* time-aligned `etc` -- see
        run_appendix_hybrid_params, which now passes this explicitly for
        both sides: `2*source_radius` for misuka's ETC after
        align_hull_to_center_lossless (whose whole visible-cap smear has
        shifted from [d-r, d+r]/c to [d, d+2r]/c) and `0` for PRA's RIR
        (frac_delay-stripped and already point-source, no smear to widen
        for at all -- the original `+source_radius/c` term had no physical
        justification on PRA's side to begin with). Verified in the H1
        t0-alignment investigation that this combination leaves misuka's
        own G value unchanged from the unaligned case (the shift and the
        widened window cancel exactly for a lossless shift) -- G's
        persistent misuka-vs-PRA gap is unrelated to alignment, see the
        module-level "not meaningful in absolute terms" caveat instead.

    expected_start_s : float, optional
        Analytic direct-sound onset in `etc`, for ir_start's fallback check.
        Defaults to distance/speed_of_sound -- correct for every current
        caller (misuka after align_hull_to_center_lossless, PRA frac-delay-
        stripped); pass (distance - source_radius)/c for an unaligned misuka
        ETC. C50/C80/TS and G's direct-sound window start at the detected IR
        start (see ir_start), EDT/T30/T60 are unaffected by the origin.
    """
    if g_radius_term is None:
        g_radius_term = source_radius
    if expected_start_s is None:
        expected_start_s = distance / speed_of_sound

    start, _ = ir_start(etc, sampling_rate, expected_start_s)
    times = np.arange(len(etc)) / sampling_rate
    etc_ir = etc[start:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edc_curve = pyrato_edc(etc_ir, sampling_rate)
        c50 = float(pyrato.parameters.clarity(edc_curve, early_time_limit=50)[0])
        c80 = float(pyrato.parameters.clarity(edc_curve, early_time_limit=80)[0])
        the_edt = float(pyrato.parameters.reverberation_time_linear_regression(edc_curve, "EDT")[0])
        the_t60 = float(pyrato.parameters.reverberation_time_linear_regression(edc_curve, "T60")[0])
        the_t30 = float(pyrato.parameters.reverberation_time_linear_regression(edc_curve, "T30")[0])

    ts = float(np.sum(etc_ir * times[:len(etc_ir)]) / np.sum(etc_ir))

    direct_sound_end = distance / speed_of_sound + g_radius_term / speed_of_sound
    end_bin = int(np.argmin(np.abs(direct_sound_end - times)))
    free_field_proxy = (distance ** 2 / 10 ** 2) * etc_ir[:max(end_bin - start + 1, 0)]
    proxy_energy = float(np.sum(free_field_proxy))
    if proxy_energy == 0.0:
        # distance/c assumes a straight-line path exists at all -- false for
        # a non-convex room where source and mic have no line of sight
        # (e.g. l_room_1000m3: the analytic direct-sound window closes
        # before any real energy arrives, since the earliest path has to go
        # around the notch corner). G is genuinely undefined here, not
        # "very large" -- nan (not the 10*log10(x/0)=inf this would
        # otherwise silently produce) says so explicitly and is what
        # plot_hybrid_jnd's np.isfinite() filtering already expects.
        g = float("nan")
    else:
        g = float(10 * np.log10(np.sum(etc_ir) / proxy_energy))

    return dict(C50=c50, C80=c80, G=g, EDT=the_edt, T60=the_t60, T30=the_t30, TS=ts)


# parameters_full() above applies no cross-renderer time alignment at all
# (see its docstring) -- misuka's hull-vs-center geometric offset
# (source_radius/speed_of_sound, see test_pyroomacoustics_comparison.py's
# "Direct-sound time-axis alignment" section) and pyroomacoustics' own
# frac_delay_length//2-sample RIR lead-in are both left uncorrected. That was
# fine as long as every caller's source_radius stayed small (originally
# documented in pyroomacoustics_comparison.ipynb as "<=0.2m, a sub-to-few-bin
# shift") -- but that was a static comment, not a check, and silently stopped
# holding once scaled scenarios were added (shoebox_*_5000m3: source_radius
# scales with room size, see _scaled_shoebox, and reaches 0.585m -- ~75 bins
# at SAMPLING_RATE, no longer negligible next to EDT's 10ms early window).
UNALIGNED_OFFSET_WARN_BINS = 4  # matches this file's narrowest window sensitivity (EDT); see docstring above


def check_alignment_offset_negligible(source_radius, scenario_name=None,
                                      sampling_rate=cmp.SAMPLING_RATE,
                                      speed_of_sound=cmp.SPEED_OF_SOUND):
    """Warn if parameters_full()'s no-alignment assumption no longer holds
    for the given source_radius -- replaces the static "source radii used
    here are <=0.2m" comment with an actual runtime check, so a new or
    rescaled scenario can't silently violate it again unnoticed.

    Returns the offset in bins (for callers that want to log/store it).
    """
    offset_bins = source_radius / speed_of_sound * sampling_rate
    if offset_bins > UNALIGNED_OFFSET_WARN_BINS:
        name = f" ({scenario_name})" if scenario_name else ""
        warnings.warn(
            f"parameters_full{name}: geometric hull-vs-center offset is "
            f"{offset_bins:.1f} bins (source_radius={source_radius:.4f} m) -- "
            f"exceeds UNALIGNED_OFFSET_WARN_BINS={UNALIGNED_OFFSET_WARN_BINS}; "
            "G/EDT/TS/C50 are computed without hull-vs-center or frac_delay "
            "alignment and may be biased at this scale.", stacklevel=2)
    return offset_bins


# JND per ISO 3382-1 / Masterarbeit Table 1. C50 has no JND of its own: as in
# Richter's thesis it is derived from the D50 JND of 0.05 via
# C50 = 10 lg(D50 / (1 - D50)), see jnd_band(). The 1 dB in JND_ABS is only a
# fallback for callers that need a single symmetric number.
JND_ABS = dict(C50=1.0, C80=1.0, G=1.0, TS=0.010)   # absolute JND
JND_REL = dict(EDT=0.05, T60=0.05)                   # relative JND (5%)
D50_JND = 0.05


def c50_from_d50(d50):
    return 10 * np.log10(d50 / (1 - d50))


def jnd_band(param, value, relative_to=None):
    """(lower, upper) JND band around `value`. C50 follows Richter: shift the
    corresponding D50 by +-0.05 and convert back, which gives a slightly
    asymmetric band. Relative JNDs use `relative_to` (default `value`)."""
    if param == "C50":
        d50 = 1 / (1 + 10 ** (-value / 10))
        eps = 1e-9
        return (c50_from_d50(max(d50 - D50_JND, eps)), c50_from_d50(min(d50 + D50_JND, 1 - eps)))
    if param in JND_ABS:
        return value - JND_ABS[param], value + JND_ABS[param]
    width = JND_REL.get(param, 0.05) * abs(value if relative_to is None else relative_to)
    return value - width, value + width


# =====================================================================
#                        Small error metrics
# =====================================================================
#
# Not present in test_pyroomacoustics_comparison.py (its edc_correlation/
# edc_rms_error_db operate on whole decay curves, not per-peak-window
# energy vectors) -- ported from the external renderer_vergleich.ipynb
# scaffold unchanged.

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return np.nan_to_num(np.dot(a, b) / denom, nan=0.0)


def rmspe(a, b):
    return 100 * np.sqrt(np.mean(((a - b) / b) ** 2))


def mpe(a, b):
    return 100 * np.mean((a - b) / b)


def rel_pct(misuka_value, pra_value):
    """Bounded (+/-100%) relative difference of two scalars -- the same
    formula as cmp.relative_difference_percent, applied to one scalar
    descriptor value instead of a bin-by-bin energy curve.
    """
    denom = max(abs(misuka_value), abs(pra_value))
    return 100.0 * (misuka_value - pra_value) / denom if denom > 0 else float("nan")


# =====================================================================
#           ISM peak-window energy integration (Section 1)
# =====================================================================

def pra_ism_peak_times_and_energies(scenario, pra_order, speed_of_sound=cmp.SPEED_OF_SOUND):
    """Pure ISM peak arrival times + energies for one cmp scenario, reusing its
    own pra_room_factory so per-scenario materials/geometry (e.g.
    flutter_corridor's asymmetric end walls) are respected exactly, instead of
    rebuilding a fresh single-material pra.ShoeBox from raw dimensions. Calls
    only image_source_model(), never compute_rir()/ray_tracing(), so no
    stochastic ray tracing actually runs even though pra_room_factory's rooms
    have ray_tracing=True baked in.
    """
    pra_room = scenario["pra_room_factory"](pra_order)
    pra_room.image_source_model()

    images = pra_room.sources[0].images
    damping = pra_room.sources[0].damping[0]
    mic = pra_room.mic_array.R[:, 0]

    distances = np.linalg.norm(images - mic[:, None], axis=0)
    times = distances / speed_of_sound
    energies = (damping / distances) ** 2

    order = np.argsort(times)
    return times[order], energies[order]


def build_peak_windows(peak_times, source_radius, sampling_rate=cmp.SAMPLING_RATE,
                       speed_of_sound=cmp.SPEED_OF_SOUND):
    """Windows [t - dt/2, t + dt/2] around each ISM peak, merging overlaps
    (Richter Masterarbeit 3.5.3). Half-width floored at one misuka bin
    (1/sampling_rate), so a window can never collapse to under one bin wide
    and produce a spurious all-zero window from pure bin quantization at low
    spp, regardless of how source_radius/c/2 compares to the bin width at
    misuka's actual native cmp.SAMPLING_RATE.
    """
    half_width = max(0.5 * cmp.hull_to_center_shift_seconds(source_radius, speed_of_sound),
                     1.0 / sampling_rate)
    windows = sorted((tt - half_width, tt + half_width) for tt in peak_times)
    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def integrate_ism_energy_in_windows(peak_times, peak_energies, windows):
    out = np.zeros(len(windows))
    for i, (start, end) in enumerate(windows):
        mask = (peak_times >= start) & (peak_times < end)
        out[i] = np.sum(peak_energies[mask])
    return out


def integrate_etc_energy_in_windows(etc, sampling_rate, windows):
    times = np.arange(len(etc)) / sampling_rate
    out = np.zeros(len(windows))
    for i, (start, end) in enumerate(windows):
        mask = (times >= start) & (times < end)
        out[i] = np.sum(etc[mask])
    return out
