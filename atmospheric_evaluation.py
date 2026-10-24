"""Evaluation of misuka's atmospheric extensions, on the Gewandhaus scene.

Backs the thesis section of the same name. Three blocks:

1. Inline vs. post-processing attenuation -- how the post-processing
   approximation's error depends on frequency and on time-bin width, and
   what inline attenuation costs in runtime.
2. Speed of sound -- how far the `cramer` / `ideal_gas` methods can drift
   from the ISO 9613-1 (temperature-only) formula within a realistic indoor
   parameter range.
3. Inverse atmospheric rendering -- gradient-based retrieval of temperature
   and relative humidity from a rendered ETC.

Scope relative to what the suite already had: the post-vs-inline error was
already swept over sampling rate and frequency by
analyze_post_vs_inline_attenuation.py, and the three runtimes (no / inline /
post attenuation) are already measured per render by
test_pyroomacoustics_comparison.compare_atmosphere(). Both are reused here
rather than reimplemented -- what is new is (a) running them on the
Gewandhaus scene instead of a shoebox, (b) the distance-error metric
max/avg |t_real - dt_i| that block 1b actually asks for (the existing sweep
measures an ETC difference in percent, a different quantity), (c) the
significance test on the runtimes, and (d) blocks 2 and 3, which had no
equivalent anywhere in the suite.

Every render uses the validated Gewandhaus setup from
pra_comparison_gewandhaus (RICHTER_* : source_radius=0.1 m, c=343.99 m/s,
GRAP S1/R1 and all six GRAP materials), with the
GRAP material tables evaluated at whatever frequency a given experiment
renders -- see that module's materials_at().

Run directly to (re)generate every stored run; the figures live in
pra_comparison_plots.py (plot_attenuation_*, plot_speed_of_sound_extremes,
plot_inverse_rendering).
"""
import os
import time

import numpy as np
import drjit as dr
import mitsuba as mi

# The AD variant is required by block 3 (gradients through acoustic_prb) and
# harmless for blocks 1/2, so the whole module renders under it -- mirroring
# the thesis statement that all measurements used "the LLVM variant or its
# AD variant".
mi.set_variant("llvm_ad_acoustic")

from scipy import optimize, stats

import test_pyroomacoustics_comparison as cmp
import pra_comparison_gewandhaus as gw
import pra_comparison_storage as storage
from pra_comparison_runner import _repeat_and_aggregate, _seed_for_repeat

SCENARIO_NAME = "gewandhaus"

# Frequencies the attenuation blocks render at. The suite's existing
# highlight set (pra_comparison_runner.HIGHLIGHT_FREQUENCIES_HZ), reused
# unchanged so this section's frequency axis matches the rest of the suite.
FREQUENCIES_HZ = [100.0, 1000.0, 4000.0, 8000.0, 14000.0, 20000.0]

# Separate, smaller highlight set for the bin-width blocks (1b/1c): each of
# these renders once per candidate bin width, so the render cost scales with
# len(frequencies) x len(bin_widths) rather than being paid once as in 1a.
BIN_WIDTH_FREQUENCIES_HZ = [100.0, 2000.0, 12500.0, 20000.0]  # GRAP third-octave centres, no interpolation

MAX_TIME = 4.0                  # the Gewandhaus scene's own response length
SAMPLING_RATE = cmp.SAMPLING_RATE
SPP = 2 ** 18                   # cmp.MISUKA_SPP; the attenuation contrast is a
                                # per-path factor, not a convergence question
RUNTIME_N_REPEATS = 50          # thesis: 50 seeds per variant
N_REPEATS = 10


def _scene(frequencies=FREQUENCIES_HZ, sampling_rate=SAMPLING_RATE,
           max_time=MAX_TIME, **kwargs):
    return gw.build_misuka_scene_dict_richter(
        max_time, sampling_rate, frequencies=frequencies, **kwargs)


# misuka's standard medium (the values an empty `acoustic_medium` falls back
# to, thesis Section 3.1.1). Every block of this module renders in it.
STANDARD_MEDIUM = dict(temperature=20.0, relative_humidity=0.5,
                       atmospheric_pressure=101325.0,
                       saturation_vapor_pressure=2333.0, co2_ppm=428.73,
                       speed_of_sound_method="simple")


def _atmosphere_kwargs():
    """Temperature, humidity and pressure of the standard medium, in the form
    the post-processing attenuation takes them."""
    return {k: STANDARD_MEDIUM[k]
            for k in ("temperature", "relative_humidity", "atmospheric_pressure")}


def _standard_speed_of_sound():
    """Speed of sound misuka derives from the standard medium."""
    m = STANDARD_MEDIUM
    return cmp._scalar(mi.acoustic.speed_of_sound(
        temperature=m["temperature"], relative_humidity=m["relative_humidity"],
        atmospheric_pressure=m["atmospheric_pressure"],
        saturation_vapor_pressure=m["saturation_vapor_pressure"],
        co2_ppm=m["co2_ppm"], method=m["speed_of_sound_method"]))


# =====================================================================
#   1a: frequency dependence of the post-vs-inline attenuation error
# =====================================================================

def run_attenuation_frequency_dependence(n_repeats=N_REPEATS):
    """Render the Gewandhaus scene with inline and with post-processing
    attenuation at several frequencies and compare the resulting ETCs.

    The attenuation coefficient grows with f^2, so the same per-bin distance
    ambiguity maps to a larger energy error at higher frequencies -- that is
    the dependency this quantifies.
    """
    section = "atmospheric_attenuation_frequency"
    scene_dict = _scene()
    c_atmo = _standard_speed_of_sound()

    def render_one(seed):
        e_raw = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                  atmosphere=True, apply_attenuation=False,
                                  acoustic_medium=STANDARD_MEDIUM)
        e_inline = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                     atmosphere=True, apply_attenuation=True,
                                  acoustic_medium=STANDARD_MEDIUM)
        e_post = cmp.apply_pure_tone_attenuation(
            e_raw, sampling_rate=SAMPLING_RATE, speed_of_sound_ms=c_atmo,
            frequencies=FREQUENCIES_HZ, **_atmosphere_kwargs())
        diff_pct = np.abs(cmp.relative_difference_percent(e_post, e_inline))
        return dict(mean_abs_pct=diff_pct.mean(axis=0),
                    max_abs_pct=diff_pct.max(axis=0))

    per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
    arrays = dict(
        frequencies_hz=np.array(FREQUENCIES_HZ),
        mean_abs_pct=mean["mean_abs_pct"], mean_abs_pct_std=std["mean_abs_pct"],
        max_abs_pct=mean["max_abs_pct"], max_abs_pct_std=std["max_abs_pct"],
    )
    metrics = dict(
        n_repeats=n_repeats,
        mean_abs_pct_by_frequency=[float(v) for v in mean["mean_abs_pct"]],
        max_abs_pct_by_frequency=[float(v) for v in mean["max_abs_pct"]],
        # monotone growth with frequency is the claim under test
        increases_with_frequency=bool(
            np.all(np.diff(mean["mean_abs_pct"]) > 0)),
    )
    storage.save_run(
        section, SCENARIO_NAME, metrics=metrics, arrays=arrays,
        room=_room_metadata(), render_params=_render_params(n_repeats, speed_of_sound_atmo=c_atmo),
    )
    print(f"done: {section} -- mean|delta|% by frequency = "
          f"{np.round(mean['mean_abs_pct'], 4)}")
    return arrays


# =====================================================================
#   1b: bin-width dependence of the distance approximation itself
# =====================================================================

def run_attenuation_bin_width(n_repeats=N_REPEATS, bin_widths_hz=None,
                              frequencies=BIN_WIDTH_FREQUENCIES_HZ):
    """max/avg |t_real - dt_i| as a function of time-bin width, per frequency.

    Post-processing attenuates a bin by the distance its *bin time* implies
    (`apply_pure_tone_attenuation` uses the bin's left edge,
    `t_i = i/sampling_rate`), while the paths summed into that bin actually
    arrived anywhere within it. The resulting per-path error is measured
    here directly rather than assumed: one high-resolution reference render
    (8x the suite's own sampling rate) supplies the true arrival-time
    distribution, each candidate bin width is formed by re-binning that
    reference, and the error is the energy-weighted |t_real - t_bin| over
    all reference bins.

    The reference is rendered *with* inline attenuation (not the raw,
    unattenuated ETC): the weighting is meant to reflect where sound energy
    actually arrives, and attenuation shifts that energy earlier in time
    more aggressively at higher frequencies (it scales with f^2). Computing
    the error separately per frequency band therefore captures a real,
    frequency-dependent effect -- at high frequencies the tail carries
    little energy, and errors far out in that tail count for little in the
    energy-weighted average and drop out of the max entirely once they carry
    no energy.

    Reported in metres as well as seconds, since it is the distance that
    enters the attenuation exponent.
    """
    section = "atmospheric_attenuation_bin_width"
    # Reference resolution: 8x the suite's own sampling rate, so that even
    # the finest candidate bin width below still aggregates many reference
    # bins and the "true" time within a bin is properly resolved.
    reference_sampling_rate = 8 * SAMPLING_RATE
    if bin_widths_hz is None:
        bin_widths_hz = [2000.0, 4000.0, 8000.0, 16000.0, 22050.0, 44100.0]

    scene_dict = _scene(frequencies=frequencies, sampling_rate=reference_sampling_rate)
    c_atmo = _standard_speed_of_sound()

    def render_one(seed):
        etc = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                atmosphere=True, apply_attenuation=True,
                                  acoustic_medium=STANDARD_MEDIUM)
        etc = np.atleast_2d(np.asarray(etc, dtype=np.float64))  # (n_time_bins, n_freq)
        t_true = np.arange(etc.shape[0]) / reference_sampling_rate

        max_err_s = np.full((len(bin_widths_hz), len(frequencies)), np.nan)
        avg_err_s = np.full((len(bin_widths_hz), len(frequencies)), np.nan)
        for i, fs in enumerate(bin_widths_hz):
            # which coarse bin each reference bin falls into, and that bin's
            # left-edge time -- exactly the time post-processing would use
            bin_index = np.floor(t_true * fs).astype(np.int64)
            t_bin = bin_index / fs
            err = np.abs(t_true - t_bin)
            for j in range(len(frequencies)):
                energy = etc[:, j]
                occupied = energy > 0
                if not np.any(occupied):
                    continue
                max_err_s[i, j] = err[occupied].max()
                # energy-weighted: an error in a bin carrying no energy
                # cannot affect the attenuated ETC
                avg_err_s[i, j] = np.average(err[occupied], weights=energy[occupied])
        return dict(max_err_s=max_err_s, avg_err_s=avg_err_s)

    per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
    bin_widths_s = 1.0 / np.array(bin_widths_hz)
    arrays = dict(
        bin_widths_hz=np.array(bin_widths_hz),
        bin_widths_s=bin_widths_s,
        frequencies_hz=np.array(frequencies),
        max_err_s=mean["max_err_s"], max_err_s_std=std["max_err_s"],
        avg_err_s=mean["avg_err_s"], avg_err_s_std=std["avg_err_s"],
        max_err_m=mean["max_err_s"] * c_atmo, avg_err_m=mean["avg_err_s"] * c_atmo,
    )
    metrics = dict(
        n_repeats=n_repeats,
        reference_sampling_rate_hz=float(reference_sampling_rate),
        speed_of_sound_atmo=float(c_atmo),
        max_err_grows_with_bin_width=bool(np.all(np.diff(mean["max_err_s"], axis=0) < 0)),
        avg_err_grows_with_bin_width=bool(np.all(np.diff(mean["avg_err_s"], axis=0) < 0)),
    )
    storage.save_run(
        section, SCENARIO_NAME, metrics=metrics, arrays=arrays,
        room=_room_metadata(),
        render_params=_render_params(n_repeats, sampling_rate=reference_sampling_rate,
                                     bin_widths_hz=list(bin_widths_hz),
                                     frequencies_hz=list(frequencies)),
    )
    print(f"done: {section} -- max err (m) per frequency =\n{np.round(arrays['max_err_m'], 4)}")
    return arrays


# =====================================================================
#   1c: relative ETC error (post vs. inline) as a function of bin width
# =====================================================================

def run_attenuation_bin_width_etc_error(n_repeats=N_REPEATS, bin_widths_hz=None,
                                        frequencies=BIN_WIDTH_FREQUENCIES_HZ):
    """The same bin-width sweep as 1b, but measuring the actual quantity 1a
    reports (relative ETC difference between post-processing and inline
    attenuation, in %) instead of the underlying distance-approximation
    error.

    Unlike 1b, which re-bins one high-resolution reference render, this
    needs a real render *at* each candidate bin width -- the bin width is
    the scene's own sampling rate here, not something applied afterwards --
    so the render cost scales with len(bin_widths_hz), not just
    len(frequencies).
    """
    section = "atmospheric_attenuation_bin_width_etc_error"
    if bin_widths_hz is None:
        bin_widths_hz = [2000.0, 4000.0, 8000.0, 16000.0, 22050.0, 44100.0]

    c_atmo = _standard_speed_of_sound()
    scenes = [_scene(frequencies=frequencies, sampling_rate=fs) for fs in bin_widths_hz]

    def render_one(seed):
        shape = (len(bin_widths_hz), len(frequencies))
        out = {k: np.empty(shape) for k in ("mean_abs_pct", "max_abs_pct", "energy_pct", "eps_pct",
                                            "mean_abs_pct_centre", "max_abs_pct_centre",
                                            "energy_pct_centre", "eps_pct_centre", "eps_noise_pct")}
        for i, (fs, scene_dict) in enumerate(zip(bin_widths_hz, scenes)):
            e_raw = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                      atmosphere=True, apply_attenuation=False,
                                  acoustic_medium=STANDARD_MEDIUM)
            e_inline = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                         atmosphere=True, apply_attenuation=True,
                                  acoustic_medium=STANDARD_MEDIUM)
            # eps: energy-weighted mean |relative error| per bin, sum|a - b| / sum b
            eps = lambda a, b: np.abs(np.asarray(a) - b).sum(axis=0) / np.asarray(b).sum(axis=0) * 100
            # Monte Carlo noise on the same scale: inline vs. inline with another seed
            e_inline_other = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed + 10_000,
                                               atmosphere=True, apply_attenuation=True,
                                               acoustic_medium=STANDARD_MEDIUM)
            out["eps_noise_pct"][i] = eps(e_inline_other, e_inline)
            # distance of each bin from its start (the former misuka function) and
            # from its centre; energy_pct is the difference of the total energy
            for suffix, offset in (("", 0.0), ("_centre", 0.5)):
                e_post = cmp.apply_pure_tone_attenuation(
                    e_raw, sampling_rate=fs, speed_of_sound_ms=c_atmo,
                    frequencies=frequencies, bin_offset=offset, **_atmosphere_kwargs())
                diff_pct = np.abs(cmp.relative_difference_percent(e_post, e_inline))
                out["mean_abs_pct" + suffix][i] = diff_pct.mean(axis=0)
                out["max_abs_pct" + suffix][i] = diff_pct.max(axis=0)
                out["energy_pct" + suffix][i] = (np.asarray(e_post).sum(axis=0) / np.asarray(e_inline).sum(axis=0) - 1) * 100
                out["eps_pct" + suffix][i] = eps(e_post, e_inline)
        return out

    per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
    arrays = dict(
        bin_widths_hz=np.array(bin_widths_hz),
        bin_widths_s=1.0 / np.array(bin_widths_hz),
        frequencies_hz=np.array(frequencies),
        **{k: mean[k] for k in mean}, **{k + "_std": std[k] for k in std},
    )
    metrics = dict(n_repeats=n_repeats, speed_of_sound_atmo=c_atmo)
    storage.save_run(
        section, SCENARIO_NAME, metrics=metrics, arrays=arrays,
        room=_room_metadata(),
        render_params=_render_params(n_repeats, bin_widths_hz=list(bin_widths_hz),
                                     frequencies_hz=list(frequencies)),
    )
    print(f"done: {section} -- mean |delta|% per frequency =\n{np.round(arrays['mean_abs_pct'], 4)}")
    return arrays


# =====================================================================
#   1d: runtime of inline vs. no vs. post-processing attenuation
# =====================================================================

def run_attenuation_runtime(n_repeats=RUNTIME_N_REPEATS, variant="scalar_acoustic"):
    """Repeated-measures runtime comparison of the three attenuation modes.

    Runs under `scalar_acoustic` by default rather than this module's LLVM
    variant. Timing is what this block measures, and the scalar variant is
    single-threaded and JIT-free -- compare_atmosphere()'s own docstring
    already assumes it ("no JIT warm-up needed"). Under LLVM each sample is
    spread over all cores (measured: ~970% CPU, 11 threads), so run-to-run
    scheduling noise was larger than the effect under test. The cost is
    wall-clock time: ~8.7 s per render instead of ~0.87 s.

    Reuses compare_atmosphere(), which already times exactly these three
    (no attenuation, inline, and no-attenuation-plus-post-processing) per
    call, and adds the significance test the thesis reports. Renders are
    single-frequency (the suite's FREQUENCY_HZ) so that the timing reflects
    one band's cost rather than the six-band film the accuracy blocks use.

    The three samples are paired (same seed renders all three variants), so
    the paired tests are the appropriate ones; both a parametric
    (Wilcoxon-matched t-test) and a non-parametric (Wilcoxon signed-rank)
    result are stored, together with a Shapiro-Wilk normality check on the
    paired differences that says which of the two to quote.
    """
    section = "atmospheric_attenuation_runtime"
    previous_variant = mi.variant()
    if variant is not None and previous_variant != variant:
        mi.set_variant(variant)
    timing_variant = mi.variant()
    scene = dict(name=SCENARIO_NAME, max_time=MAX_TIME, spp=SPP,
                 misuka_scene=_scene(frequencies=[cmp.FREQUENCY_HZ]))

    def render_one(seed):
        metrics, _, _ = cmp.compare_atmosphere(scene, seed=seed, medium=STANDARD_MEDIUM)
        return dict(
            runtime_none_ms=metrics["runtime_no_atmo_s"] * 1e3,
            runtime_inline_ms=metrics["runtime_atmo_s"] * 1e3,
            runtime_post_ms=metrics["runtime_post_total_s"] * 1e3,
        )

    try:
        per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
    finally:
        # block 3 (acoustic_prb, gradients) needs the module's AD variant back
        mi.set_variant(previous_variant)
    samples = {key: per_repeat[key] for key in
               ("runtime_none_ms", "runtime_inline_ms", "runtime_post_ms")}

    metrics = dict(n_repeats=n_repeats)
    for key, values in samples.items():
        metrics[f"{key}_mean"] = float(np.mean(values))
        metrics[f"{key}_std"] = float(np.std(values))
        metrics[f"{key}_median"] = float(np.median(values))

    # Pairwise, on the paired samples. inline-vs-none is the comparison the
    # thesis claim ("inline is slower, but not significantly") rests on.
    for a, b in (("runtime_inline_ms", "runtime_none_ms"),
                 ("runtime_inline_ms", "runtime_post_ms"),
                 ("runtime_post_ms", "runtime_none_ms")):
        differences = samples[a] - samples[b]
        normal_p = float(stats.shapiro(differences).pvalue) if len(differences) >= 3 else float("nan")
        t_p = float(stats.ttest_rel(samples[a], samples[b]).pvalue)
        w_p = float(stats.wilcoxon(samples[a], samples[b]).pvalue)
        tag = f"{a.replace('runtime_', '').replace('_ms', '')}_vs_" \
              f"{b.replace('runtime_', '').replace('_ms', '')}"
        metrics[f"{tag}_mean_diff_ms"] = float(np.mean(differences))
        metrics[f"{tag}_shapiro_p"] = normal_p
        metrics[f"{tag}_ttest_rel_p"] = t_p
        metrics[f"{tag}_wilcoxon_p"] = w_p
        # which test to quote: paired t-test only if the differences are
        # plausibly normal, otherwise the signed-rank test
        metrics[f"{tag}_p"] = t_p if (normal_p == normal_p and normal_p > 0.05) else w_p
        metrics[f"{tag}_test"] = "ttest_rel" if (normal_p == normal_p and normal_p > 0.05) else "wilcoxon"

    arrays = {key: np.asarray(values) for key, values in samples.items()}
    storage.save_run(
        section, SCENARIO_NAME, metrics=metrics, arrays=arrays,
        room=_room_metadata(),
        render_params=_render_params(n_repeats, frequencies_hz=[cmp.FREQUENCY_HZ],
                                     timing_variant=timing_variant),
        misuka_variant=timing_variant,
    )
    print(f"done: {section} [{timing_variant}] -- inline={metrics['runtime_inline_ms_mean']:.1f}ms "
          f"none={metrics['runtime_none_ms_mean']:.1f}ms "
          f"post={metrics['runtime_post_ms_mean']:.1f}ms "
          f"(inline vs none: p={metrics['inline_vs_none_p']:.3f})")
    return metrics


# =====================================================================
#   2: speed-of-sound method spread against the ISO 9613-1 formula
# =====================================================================

# Realistic indoor range the search runs over (thesis section).
CO2_PPM_RANGE = (500.0, 2000.0)
RELATIVE_HUMIDITY_RANGE = (0.20, 0.70)
TEMPERATURE_RANGE = (5.0, 30.0)
FIXED_PRESSURE = 101325.0


def magnus_saturation_vapor_pressure(temperature_c):
    """Magnus formula, in Pa -- the same expression cmp.ATMO_SATURATION_
    VAPOR_PRESSURE uses for its single fixed temperature, here as a function
    so the optimizer can vary temperature without leaving saturation vapor
    pressure behind at a stale value."""
    return 610.78 * 10 ** (7.5 * temperature_c / (temperature_c + 237.3))


def speed_of_sound_iso(temperature_c):
    """ISO 9613-1's temperature-only speed of sound, misuka's "simple"
    method. (pyroomacoustics uses 331.4 + 0.6 T + 0.0124 RH instead.)"""
    return 343.2 * np.sqrt((temperature_c + 273.15) / 293.15)


def _speed_of_sound(method, temperature_c, relative_humidity, co2_ppm):
    # float() rather than the caller's numpy scalars: misuka's binding only
    # accepts Python floats (or drjit Floats), and the optimizer hands out
    # np.float64 from its parameter vectors.
    return cmp._scalar(mi.acoustic.speed_of_sound(
        temperature=float(temperature_c),
        relative_humidity=float(relative_humidity),
        atmospheric_pressure=float(FIXED_PRESSURE),
        saturation_vapor_pressure=float(magnus_saturation_vapor_pressure(temperature_c)),
        co2_ppm=float(co2_ppm),
        method=method))


def run_speed_of_sound_extremes():
    """Find, per method, the in-range parameter combination that maximises
    |method - ISO|.

    The methods themselves are O(1) and evaluated once per render (they set
    the integrator's speed of sound, they do not run per path or per
    sample), so only accuracy is of interest -- asserted here rather than
    benchmarked, see the assertion below.

    scipy's `differential_evolution` is used rather than a grid: the
    objective is cheap, smooth and only three-dimensional, and a global
    method avoids reporting a local corner of the box as "the" maximum.
    """
    section = "atmospheric_speed_of_sound_extremes"

    # The O(1)/once-per-render claim, as a check rather than a comment: the
    # scene dict carries exactly one speed-of-sound entry, independent of
    # ray count, path length or number of time bins.
    integrator_keys = ["speed_of_sound", "acoustic_medium"]
    scene_probe = _scene(frequencies=[cmp.FREQUENCY_HZ])
    assert "emitter" in scene_probe, "scene must still build for the probe below"
    assert all(k not in scene_probe for k in integrator_keys), (
        "speed of sound belongs to the integrator, applied once per render -- "
        "if it ever appears per shape/emitter here, the O(1) claim needs revisiting")

    results = {}
    for method in ("cramer", "ideal_gas"):
        def negative_difference(x, method=method):
            temperature_c, relative_humidity, co2_ppm = x
            return -abs(_speed_of_sound(method, temperature_c, relative_humidity, co2_ppm)
                        - speed_of_sound_iso(temperature_c))

        result = optimize.differential_evolution(
            negative_difference,
            bounds=[TEMPERATURE_RANGE, RELATIVE_HUMIDITY_RANGE, CO2_PPM_RANGE],
            seed=0, tol=1e-10, polish=True)
        temperature_c, relative_humidity, co2_ppm = result.x
        results[method] = dict(
            max_abs_difference_ms=float(-result.fun),
            temperature_c=float(temperature_c),
            relative_humidity=float(relative_humidity),
            co2_ppm=float(co2_ppm),
            saturation_vapor_pressure_pa=float(magnus_saturation_vapor_pressure(temperature_c)),
            speed_of_sound_method_ms=float(_speed_of_sound(method, temperature_c,
                                                           relative_humidity, co2_ppm)),
            speed_of_sound_iso_ms=float(speed_of_sound_iso(temperature_c)),
        )

    # A coarse map of the objective for the figure, at the worst-case CO2 of
    # each method (CO2's influence is far weaker than temperature's/RH's).
    temperatures = np.linspace(*TEMPERATURE_RANGE, 40)
    humidities = np.linspace(*RELATIVE_HUMIDITY_RANGE, 40)
    arrays = dict(temperatures_c=temperatures, relative_humidities=humidities)
    for method in results:
        co2 = results[method]["co2_ppm"]
        grid = np.array([[abs(_speed_of_sound(method, t, rh, co2) - speed_of_sound_iso(t))
                          for rh in humidities] for t in temperatures])
        arrays[f"difference_grid_{method}"] = grid

    metrics = dict(
        fixed_pressure_pa=FIXED_PRESSURE,
        co2_ppm_range=list(CO2_PPM_RANGE),
        relative_humidity_range=list(RELATIVE_HUMIDITY_RANGE),
        temperature_range_c=list(TEMPERATURE_RANGE),
        **{f"{method}_{key}": value
           for method, entry in results.items() for key, value in entry.items()},
    )
    storage.save_run(
        section, SCENARIO_NAME, metrics=metrics, arrays=arrays,
        extra=dict(saturation_vapor_pressure="Magnus formula, evaluated at the "
                                             "optimizer's temperature (not searched over)"),
    )
    for method, entry in results.items():
        print(f"done: {section} -- {method} vs ISO: "
              f"max |delta c| = {entry['max_abs_difference_ms']:.4f} m/s at "
              f"T={entry['temperature_c']:.2f} C, RH={entry['relative_humidity']:.3f}, "
              f"CO2={entry['co2_ppm']:.0f} ppm")
    return metrics


# =====================================================================
#   3: inverse atmospheric rendering
# =====================================================================

# Mirrors tutorials_acoustic/inverse_rendering/atmosphere_optimization.ipynb:
# acoustic_prb + mi.traverse + mi.ad.Adam, common random numbers between the
# reference and the current render. The scene is the Gewandhaus one, so
# unlike the tutorial's single free-field source the attenuation acts along
# every reflected path as well.
# No tolerance band around the goal. Convergence is judged by how much each
# run still moves over its last INVERSE_SETTLE_ITERATIONS iterations; accuracy
# by the mean and spread of the final values.
INVERSE_PARAMETERS = {
    "temperature": dict(
        scene_key="medium_temperature",
        goal=30.0, initial=10.0, learning_rate=0.5, clip=(-100.0, 100.0),
    ),
    "relative_humidity": dict(
        scene_key="medium_relative_humidity",
        goal=0.7, initial=0.2, learning_rate=0.03, clip=(0.0, 1.0),
    ),
}
INVERSE_FREQUENCY_HZ = 8000.0   # attenuation grows with f^2: the strongest
                                # gradient signal of the suite's bands
INVERSE_MAX_TIME = 0.5          # only the attenuated early part is needed
INVERSE_SAMPLING_RATE = 8000.0
INVERSE_SPP = 2 ** 16            # target: INVERSE_TARGET_SPP_FACTOR x this = 2^20
INVERSE_ITERATIONS = 500
INVERSE_SETTLE_ITERATIONS = 50  # window for the per-run final change
# The target stands in for a measurement: rendered once per optimisation with
# 16x the rays of the current estimate and its own seed, so it is smoother than
# the estimate and shares none of its random numbers.
INVERSE_TARGET_SPP_FACTOR = 16
# Against an independent noisy target a constant step keeps following the
# noise (measured: +/-2 C around the goal). The learning rate therefore decays
# exponentially to INVERSE_LR_FINAL_FRACTION of its start over the run.
INVERSE_LR_FINAL_FRACTION = 0.02
INVERSE_TARGET_SEED_OFFSET = 1_000_000
# The loss compares energies in windows of this length (10 ms, the resolution of
# GRAP's histograms) instead of single time bins, which lowers the Monte Carlo
# noise per compared value.
INVERSE_LOSS_WINDOW_S = 0.01
# The loss starts after the direct sound: INVERSE_LOSS_AFTER_DIRECT_S after it,
# but not before INVERSE_LOSS_START_S (50 ms; S1-R1 direct sound at 33 ms).
# With 2^14 rays the direct sound and the first sparse reflections are hit by
# single rays or not at all, while the 16x target contains them; these spikes
# would dominate the loss (measured at 22.8 m with a fixed 50 ms start, whose
# direct sound arrives at 66 ms: 20.7 +/- 7.0 C instead of 30 C). The dense
# later part carries the attenuation information.
INVERSE_LOSS_START_S = 0.05
INVERSE_LOSS_AFTER_DIRECT_S = 0.015
INVERSE_N_SEEDS = 10           # independent optimisations per configuration


def _inverse_integrator(parameter_name, value, max_time):
    """acoustic_prb with only the optimised medium parameter free.

    speed_of_sound is pinned to the scene's own value so that the optimised
    parameter can influence the render *only* through air attenuation --
    the same isolation the tutorial makes, and the reason a temperature
    retrieval here is a statement about attenuation, not about path timing.
    """
    spec = INVERSE_PARAMETERS[parameter_name]
    # The fixed parameters keep their standard-medium values; the speed of
    # sound is set explicitly below, so the method entry is dropped.
    medium = {k: v for k, v in STANDARD_MEDIUM.items() if k != "speed_of_sound_method"}
    medium.update(
                  # Without this the medium is carried but never applied, and
                  # the optimised parameter has no path to the render at all:
                  # measured |ETC(10 C) - ETC(30 C)| = 4e-7 (numerical noise)
                  # with it off versus 1.3 with it on. The gradient is then
                  # exactly zero and the optimizer never moves.
                  apply_attenuation=True)
    medium[parameter_name] = value
    return mi.load_dict({
        "type": "acoustic_prb",
        "max_depth": -1,
        "max_time": max_time,
        "speed_of_sound": gw.RICHTER_SPEED_OF_SOUND,
        "acoustic_medium": medium,
    }), spec


def _windowed(etc, sampling_rate, start_s):
    """Sum of the ETC over consecutive INVERSE_LOSS_WINDOW_S windows from start_s."""
    per_window = int(round(INVERSE_LOSS_WINDOW_S * sampling_rate))
    start = int(round(start_s * sampling_rate))
    n_windows = (etc.shape[0] - start) // per_window
    rest = tuple(etc.shape[1:])
    x = dr.reshape(mi.TensorXf, etc[start:start + n_windows * per_window], (n_windows, per_window) + rest)
    return dr.sum(x, axis=1)


def _run_inverse(parameter_name, goal=None, initial=None, frequency_hz=INVERSE_FREQUENCY_HZ,
                 distance_scale=1.0, mic_pos=None, iterations=INVERSE_ITERATIONS, spp=INVERSE_SPP,
                 max_time=INVERSE_MAX_TIME, sampling_rate=INVERSE_SAMPLING_RATE, seed=0):
    """One retrieval run. Returns the per-iteration history plus timings."""
    spec = INVERSE_PARAMETERS[parameter_name]
    goal = spec["goal"] if goal is None else goal
    initial = spec["initial"] if initial is None else initial

    # Distance variants move the microphone along the source->mic direction,
    # keeping the room, the source and the emitter radius untouched -- or
    # place it explicitly (mic_pos), where that line leaves the hall.
    source = np.array(gw.RICHTER_SOURCE_POS, dtype=float)
    mic = np.array(gw.RICHTER_MIC_POS, dtype=float)
    mic_scaled = (source + (mic - source) * distance_scale if mic_pos is None
                  else np.array(mic_pos, dtype=float))
    gw.check_position(mic_scaled, source)
    distance = float(np.linalg.norm(mic_scaled - source))

    # The scene's own "mic" entry is dropped: the optimisation passes an
    # explicit sensor to mi.render (as the tutorial does), and leaving a
    # second sensor in the scene would only be dead weight.
    scene_dict = _scene(frequencies=[frequency_hz], sampling_rate=sampling_rate,
                        max_time=max_time, mic_pos=tuple(mic_scaled))
    scene_dict.pop("mic")
    scene = mi.load_dict(scene_dict)
    microphone = mi.load_dict({
        "type": "microphone", "origin": list(mic_scaled), "direction": list(source - mic_scaled),
        "film": {"type": "tape", "frequencies": str(frequency_hz),
                 "time_bins": int(round(max_time * sampling_rate)),
                 "rfilter": {"type": "box"}},
    })

    integrator_ref, _ = _inverse_integrator(parameter_name, goal, max_time)
    integrator, _ = _inverse_integrator(parameter_name, initial, max_time)

    opt = mi.ad.Adam(lr=spec["learning_rate"])
    opt[parameter_name] = mi.Float(initial)
    params = mi.traverse(integrator)

    def update():
        opt[parameter_name] = dr.clip(opt[parameter_name], *spec["clip"])
        params[spec["scene_key"]] = opt[parameter_name]
        params.update(opt)

    update()

    values = np.empty(iterations)
    losses = np.empty(iterations)
    iteration_times = np.empty(iterations)

    loss_start_s = max(INVERSE_LOSS_START_S,
                       distance / gw.RICHTER_SPEED_OF_SOUND + INVERSE_LOSS_AFTER_DIRECT_S)
    target_spp = INVERSE_TARGET_SPP_FACTOR * spp
    etc_ref = mi.render(scene, sensor=microphone, integrator=integrator_ref,
                        spp=target_spp, seed=INVERSE_TARGET_SEED_OFFSET + seed) / target_spp
    etc_ref = dr.detach(etc_ref)

    for i in range(iterations):
        t0 = time.perf_counter()
        it_seed = seed + i
        etc = mi.render(scene, params, integrator=integrator, sensor=microphone,
                        spp=spp, seed=it_seed) / spp
        # Relative squared error of the energies in 10 ms windows after the
        # direct sound. Relative, not absolute: this room's spp-normalised ETC
        # is ~1e-5, so an absolute loss lands at ~1e-14 and its gradient below
        # Adam's epsilon (measured: the parameter never moves). Tested: a
        # product of two independent residuals (unbiased by construction)
        # ended at the same values as this single-render loss.
        w_ref, w_1 = _windowed(etc_ref, sampling_rate, loss_start_s), _windowed(etc, sampling_rate, loss_start_s)
        loss = dr.mean(dr.square(w_1 - w_ref)) / dr.mean(dr.square(w_ref))

        values[i] = opt[parameter_name].array[0]
        losses[i] = loss.array[0]

        dr.backward(loss)
        opt.set_learning_rate({parameter_name: float(spec["learning_rate"] * INVERSE_LR_FINAL_FRACTION ** (i / iterations))})
        opt.step()
        update()
        iteration_times[i] = time.perf_counter() - t0

    etc_final = np.asarray(etc.numpy()).squeeze()
    etc_reference = np.asarray(etc_ref.numpy()).squeeze()

    return dict(
        values=values, losses=losses, iteration_times_s=iteration_times,
        etc_final=etc_final, etc_reference=etc_reference,
        goal=float(goal), initial=float(initial),
        distance_m=distance, frequency_hz=float(frequency_hz),
        final_value=float(values[-1]), total_runtime_s=float(iteration_times.sum()),
        mean_iteration_time_s=float(iteration_times.mean()),
    )


def _run_inverse_seeds(parameter_name, n_seeds=INVERSE_N_SEEDS,
                       iterations=INVERSE_ITERATIONS, **kwargs):
    """`n_seeds` independent optimisations of the same configuration.

    Each gets a disjoint seed range (`run * iterations`), so no two runs
    share a random-number sequence -- within a run the reference and the
    current render still use the same seed (common random numbers), which is
    what keeps the loss gradient clean.
    """
    runs = [_run_inverse(parameter_name, seed=run * iterations,
                         iterations=iterations, **kwargs)
            for run in range(n_seeds)]
    stacked = {key: np.stack([r[key] for r in runs])
               for key in ("values", "losses", "iteration_times_s")}
    first = runs[0]
    return dict(
        **stacked,
        runtime_per_seed_s=np.array([r["total_runtime_s"] for r in runs]),
        final_value_per_seed=np.array([r["final_value"] for r in runs]),
        # The ETC panel shows one representative pair rather than an average:
        # averaging two noisy echograms would flatten exactly the fine
        # structure the zoom is there to show.
        etc_final=first["etc_final"], etc_reference=first["etc_reference"],
        goal=first["goal"], initial=first["initial"], distance_m=first["distance_m"],
        frequency_hz=first["frequency_hz"], n_seeds=n_seeds,
    )


def _save_inverse(section, variant, result, parameter_name=None, extra=None):
    values = result["values"]              # (n_seeds, iterations)
    losses = result["losses"]
    finals = result["final_value_per_seed"]
    # how far each run still moves at the end: peak-to-peak of its last
    # INVERSE_SETTLE_ITERATIONS values
    final_change = np.ptp(values[:, -INVERSE_SETTLE_ITERATIONS:], axis=1)

    arrays = dict(
        values=values, losses=losses,
        iteration_times_s=result["iteration_times_s"],
        runtime_per_seed_s=result["runtime_per_seed_s"],
        final_change_per_seed=final_change, final_value_per_seed=finals,
        # mean/min/max across seeds -- what the figure's average line and
        # range band are drawn from
        values_mean=values.mean(axis=0), values_min=values.min(axis=0),
        values_max=values.max(axis=0),
        losses_mean=losses.mean(axis=0), losses_min=losses.min(axis=0),
        losses_max=losses.max(axis=0),
        etc_final=result["etc_final"], etc_reference=result["etc_reference"],
        times_s=np.arange(len(result["etc_final"])) / INVERSE_SAMPLING_RATE,
    )
    metrics = {key: float(result[key]) for key in
               ("goal", "initial", "distance_m", "frequency_hz")}
    metrics["n_seeds"] = int(result["n_seeds"])
    metrics["iterations"] = int(values.shape[1])
    metrics["final_value"] = float(finals.mean())
    metrics["final_value_std"] = float(finals.std())
    metrics["absolute_error"] = float(abs(finals.mean() - result["goal"]))
    metrics["final_change_max"] = float(final_change.max())
    metrics["total_runtime_s"] = float(result["runtime_per_seed_s"].sum())
    metrics["mean_runtime_per_seed_s"] = float(result["runtime_per_seed_s"].mean())
    # Which medium parameter this run retrieved. Every variant except
    # "relative_humidity" is a temperature retrieval (the distance/frequency
    # studies vary the scene, not the parameter), and the figure needs to
    # know which axis label to use -- storing it beats
    # re-deriving it from the variant name at plot time.
    metrics["parameter"] = parameter_name or variant
    storage.save_run(section, SCENARIO_NAME, metrics=metrics, arrays=arrays,
                     variant=variant, room=_room_metadata(), extra=extra)
    return metrics


def run_inverse_retrieval():
    """3a: retrieve temperature and relative humidity, each from a far-off
    initial guess, and record iterations-to-convergence and runtime."""
    section = "atmospheric_inverse_retrieval"
    out = {}
    for parameter_name in INVERSE_PARAMETERS:
        result = _run_inverse_seeds(parameter_name)
        out[parameter_name] = _save_inverse(section, parameter_name, result,
                                            parameter_name=parameter_name)
        m = out[parameter_name]
        print(f"done: {section}/{parameter_name} -- recovered "
              f"{m['final_value']:.3f} +/- {m['final_value_std']:.3f} "
              f"(goal {m['goal']}), last {INVERSE_SETTLE_ITERATIONS} iterations "
              f"move <= {m['final_change_max']:.3f}, "
              f"{m['total_runtime_s']:.0f}s total")
    return out


def run_inverse_distance_frequency(only=None):
    """3b: the same retrieval at half/double distance and half/double
    frequency, to compare convergence speed. `only`: subset of variant names."""
    section = "atmospheric_inverse_distance_frequency"
    variants = {
        "distance_half": dict(distance_scale=0.5),
        # 2x along S1->R1 is under the raked stalls; see DISTANCE_DOUBLE_MIC_POS
        "distance_double": dict(mic_pos=gw.DISTANCE_DOUBLE_MIC_POS),
        "frequency_half": dict(frequency_hz=INVERSE_FREQUENCY_HZ / 2),
        "frequency_double": dict(frequency_hz=INVERSE_FREQUENCY_HZ * 2),
    }
    out = {}
    for variant, kwargs in variants.items():
        if only is not None and variant not in only:
            continue
        result = _run_inverse_seeds("temperature", **kwargs)
        out[variant] = _save_inverse(section, variant, result,
                                     parameter_name="temperature")
        m = out[variant]
        print(f"done: {section}/{variant} -- {m['final_value']:.2f} +/- "
              f"{m['final_value_std']:.2f}, final change <= {m['final_change_max']:.3f} "
              f"(d={m['distance_m']:.1f}m, f={m['frequency_hz']:.0f}Hz)")
    return out


def _attenuation_curves(temperatures):
    """m(T) of the standard medium at the inverse frequency and at half and
    double of it, for the diagnostic curves of 3c."""
    frequencies = np.array([INVERSE_FREQUENCY_HZ / 2, INVERSE_FREQUENCY_HZ, INVERSE_FREQUENCY_HZ * 2])
    curves = np.array([[
        cmp._scalar(mi.acoustic.energy_attenuation_coefficient(
            temperature=float(t), frequency=float(f),
            relative_humidity=STANDARD_MEDIUM["relative_humidity"],
            atmospheric_pressure=STANDARD_MEDIUM["atmospheric_pressure"]))
        for t in temperatures] for f in frequencies])
    return dict(alpha_temperatures_c=temperatures, alpha_1_per_m=curves[1],
                alpha_frequencies_hz=frequencies, alpha_1_per_m_by_frequency=curves)


def run_inverse_out_of_range():
    """3c: does the result depend on the initial value? Goal -10 C, initial
    value 10 C, on the other side of the maximum of m(T) at 8 kHz (about 5 C).
    Above the maximum m never falls to its value at -10 C; the runs settle at
    the local minimum near 40 C, which the curves (-40 to 50 C) include."""
    section = "atmospheric_inverse_out_of_range"
    result = _run_inverse_seeds("temperature", goal=-10.0, initial=10.0)
    metrics = _save_inverse(
        section, "temperature_minus10", result, parameter_name="temperature",
        extra=dict(failure_mechanism=(
            "The start (10 C) and the goal (-10 C) lie on different sides of "
            "the maximum of the attenuation coefficient; the runs converge to "
            "the temperature above the maximum whose attenuation comes closest "
            "to the goal's.")))
    # attach the diagnostic curves to the same run
    run_dir = storage.run_dir(section, SCENARIO_NAME, "temperature_minus10")
    existing = dict(np.load(f"{run_dir}/arrays.npz"))
    existing.update(_attenuation_curves(np.linspace(-40.0, 50.0, 181)))
    np.savez(f"{run_dir}/arrays.npz", **existing)
    print(f"done: {section} -- recovered {metrics['final_value']:.2f} C "
          f"(goal -10 C), final change <= {metrics['final_change_max']:.3f}")
    return metrics


# =====================================================================
#   3d: joint retrieval of temperature AND relative humidity (known goal)
#   3e: retrieval from GRAP's RAVEN reference (unknown, external target)
# =====================================================================
#
# Both are optimised in normalised coordinates u in [0, 1] (one Adam learning
# rate for two parameters of very different scale) and mapped onto
# JOINT_RANGES -- inside ISO 9613-1's +/-10 % validity range for temperature.
# Speed of sound stays pinned, so the parameters act through attenuation only.
#
# The loss landscapes are not rendered per grid point: attenuation is a
# per-path factor exp(-m d), so one attenuation-free render, attenuated per
# fine time bin (post-processing, error well below 0.1 % at this resolution,
# see block 1), gives the loss for any (T, RH) at the cost of evaluating m.

JOINT_RANGES = dict(temperature=(-10.0, 45.0), relative_humidity=(0.05, 1.0))
JOINT_GOAL = dict(temperature=30.0, relative_humidity=0.7)
JOINT_STARTS = [(10.0, 0.2), (0.0, 0.9), (40.0, 0.15), (20.0, 0.5), (-5.0, 0.4), (35.0, 0.95)]
JOINT_LEARNING_RATE = 0.01          # in normalised units per step
JOINT_ITERATIONS = 400
JOINT_SINGLE_FREQUENCY_HZ = [8000.0]
JOINT_MULTI_FREQUENCY_HZ = [2000.0, 8000.0, 16000.0]
LANDSCAPE_T = np.linspace(*JOINT_RANGES["temperature"], 56)
LANDSCAPE_RH = np.linspace(*JOINT_RANGES["relative_humidity"], 39)

RAVEN_PRESSURE = 101325.0           # RAVEN standard conditions (Richter)
RAVEN_EXPECTED = dict(temperature=20.0, relative_humidity=0.5)
RAVEN_FREQUENCIES_HZ = [4000.0, 8000.0, 16000.0]
RAVEN_MAX_TIME = 1.0
RAVEN_BIN_S = 0.010                 # 10 ms bins: GRAP's 1 ms histogram summed
RAVEN_WINDOW_START_S = 0.050        # after direct sound and the sparse early part
RAVEN_DYNAMIC_RANGE_DB = 60.0       # bins more than this below GRAP's peak are dropped
RAVEN_SPP = 2 ** 16


def _unit(name, value):
    lo, hi = JOINT_RANGES[name]
    return (value - lo) / (hi - lo)


def _from_unit(name, u):
    lo, hi = JOINT_RANGES[name]
    return lo + u * (hi - lo)


def _alpha(frequency_hz, temperature, relative_humidity, pressure):
    return cmp._scalar(mi.acoustic.energy_attenuation_coefficient(
        temperature=float(temperature), frequency=float(frequency_hz),
        relative_humidity=float(relative_humidity), atmospheric_pressure=float(pressure)))


def _medium_integrator(temperature, relative_humidity, pressure, max_time):
    return mi.load_dict({
        "type": "acoustic_prb", "max_depth": -1, "max_time": max_time,
        "speed_of_sound": gw.RICHTER_SPEED_OF_SOUND,
        "acoustic_medium": dict(temperature=temperature, relative_humidity=relative_humidity,
                                atmospheric_pressure=pressure, apply_attenuation=True)})


def _medium_scene(frequencies, sampling_rate, max_time):
    scene_dict = _scene(frequencies=frequencies, sampling_rate=sampling_rate, max_time=max_time)
    scene_dict.pop("mic")
    source = np.array(gw.RICHTER_SOURCE_POS, dtype=float)
    mic = np.array(gw.RICHTER_MIC_POS, dtype=float)
    sensor = mi.load_dict({
        "type": "microphone", "origin": list(mic), "direction": list(source - mic),
        "film": {"type": "tape", "frequencies": ", ".join(str(f) for f in frequencies),
                 "time_bins": int(round(max_time * sampling_rate)), "rfilter": {"type": "box"}}})
    return mi.load_dict(scene_dict), sensor


def _channel(etc, j, n_freq):
    """Time series of frequency channel j from a (time, freq[, 1]) tensor, differentiably."""
    n_t = dr.width(etc.array) // n_freq
    return dr.gather(mi.Float, etc.array, dr.arange(mi.UInt32, n_t) * n_freq + j)


def _optimise_medium(free, start, fixed, loss_fn, scene, sensor, pressure, max_time, spp,
                     iterations=None, seed=0):
    """Adam on the normalised free parameters ('temperature' and/or
    'relative_humidity'); loss_fn(etc, it_seed) returns a scalar mi.Float."""
    iterations = iterations or JOINT_ITERATIONS
    values = dict(fixed, **start)
    integrator = _medium_integrator(values["temperature"], values["relative_humidity"],
                                    pressure, max_time)
    params = mi.traverse(integrator)
    opt = mi.ad.Adam(lr=JOINT_LEARNING_RATE)
    for name in free:
        opt[name] = mi.Float(_unit(name, start[name]))

    def update():
        for name in free:
            opt[name] = dr.clip(opt[name], 0.0, 1.0)
            params[f"medium_{name}"] = _from_unit(name, opt[name])
        params.update()

    update()
    trace = np.empty((iterations, len(free)))
    losses = np.empty(iterations)
    t0 = time.perf_counter()
    for i in range(iterations):
        it_seed = seed + i
        etc = mi.render(scene, params, integrator=integrator, sensor=sensor, spp=spp, seed=it_seed) / spp
        loss = loss_fn(etc, it_seed)
        trace[i] = [_from_unit(n, opt[n].array[0]) for n in free]
        losses[i] = loss.array[0]
        dr.backward(loss)
        opt.step()
        update()
    return dict(trace=trace, losses=losses, runtime_s=time.perf_counter() - t0,
                etc_final=np.asarray(etc.numpy(), dtype=float).squeeze())


def _attenuation_free_etc(frequencies, sampling_rate, max_time, spp, seed):
    scene, sensor = _medium_scene(frequencies, sampling_rate, max_time)
    integrator = mi.load_dict({"type": "acoustic_prb", "max_depth": -1, "max_time": max_time,
                               "speed_of_sound": gw.RICHTER_SPEED_OF_SOUND,
                               "acoustic_medium": {"apply_attenuation": False}})
    etc = mi.render(scene, integrator=integrator, sensor=sensor, spp=spp, seed=seed) / spp
    return np.asarray(etc.numpy(), dtype=float).reshape(-1, len(frequencies))


def _attenuate(raw, frequencies, sampling_rate, temperature, relative_humidity, pressure):
    t_mid = (np.arange(raw.shape[0]) + 0.5) / sampling_rate
    d = t_mid * gw.RICHTER_SPEED_OF_SOUND
    m = np.array([_alpha(f, temperature, relative_humidity, pressure) for f in frequencies])
    return raw * np.exp(-d[:, None] * m[None, :])


# ----------------------------- 3d -----------------------------------

def run_inverse_joint(variant="single_frequency", starts=JOINT_STARTS):
    """Joint retrieval of (T, RH) towards a known goal, from several starts.
    'single_frequency' (8 kHz): one attenuation coefficient, one equation for
    two unknowns -- expected to be ambiguous along the iso-attenuation curve.
    'multi_frequency' (2/8/16 kHz): one equation per band."""
    section = "atmospheric_inverse_joint"
    frequencies = JOINT_SINGLE_FREQUENCY_HZ if variant == "single_frequency" else JOINT_MULTI_FREQUENCY_HZ
    n_freq = len(frequencies)
    fs, max_time, spp, pressure = INVERSE_SAMPLING_RATE, INVERSE_MAX_TIME, INVERSE_SPP, cmp.ATMO_PRESSURE
    scene, sensor = _medium_scene(frequencies, fs, max_time)
    reference = _medium_integrator(JOINT_GOAL["temperature"], JOINT_GOAL["relative_humidity"],
                                   pressure, max_time)

    def loss_fn(etc, it_seed):
        # common random numbers: the reference is rendered with the same seed
        ref = mi.render(scene, integrator=reference, sensor=sensor, spp=spp, seed=it_seed) / spp
        total = mi.Float(0.0)
        for j in range(n_freq):
            e, r = _channel(etc, j, n_freq), _channel(ref, j, n_freq)
            total += dr.mean(dr.square(e - r)) / dr.mean(dr.square(r))
        return total / n_freq

    runs = []
    for k, (t_start, rh_start) in enumerate(starts):
        runs.append(_optimise_medium(
            ("temperature", "relative_humidity"),
            dict(temperature=t_start, relative_humidity=rh_start), {}, loss_fn, scene, sensor,
            pressure, max_time, spp, seed=k * JOINT_ITERATIONS))
        print(f"  {variant} start {k}: ({t_start}, {rh_start}) -> "
              f"({runs[-1]['trace'][-1, 0]:.2f}, {runs[-1]['trace'][-1, 1]:.3f})", flush=True)

    # loss landscape (same relative loss, see the block comment)
    raw = _attenuation_free_etc(frequencies, fs, max_time, spp=2 ** 17, seed=10 ** 6)
    ref = _attenuate(raw, frequencies, fs, JOINT_GOAL["temperature"], JOINT_GOAL["relative_humidity"], pressure)
    landscape = np.empty((len(LANDSCAPE_T), len(LANDSCAPE_RH)))
    for a, t in enumerate(LANDSCAPE_T):
        for b, rh in enumerate(LANDSCAPE_RH):
            e = _attenuate(raw, frequencies, fs, t, rh, pressure)
            landscape[a, b] = np.mean(np.mean((e - ref) ** 2, axis=0) / np.mean(ref ** 2, axis=0))
    alpha_grid = np.array([[[_alpha(f, t, rh, pressure) for f in frequencies] for rh in LANDSCAPE_RH]
                           for t in LANDSCAPE_T])                     # (nT, nRH, nf)
    alpha_goal = np.array([_alpha(f, JOINT_GOAL["temperature"], JOINT_GOAL["relative_humidity"], pressure)
                           for f in frequencies])
    return _save_medium_run(section, variant, runs, starts, ("temperature", "relative_humidity"),
                            frequencies, landscape, alpha_grid, goal=JOINT_GOAL, alpha_goal=alpha_goal,
                            extra=dict(goal=JOINT_GOAL, pressure_pa=pressure))


# ----------------------------- 3e -----------------------------------

def _grap_10ms(frequencies, n_bins):
    per_ms = int(round(RAVEN_BIN_S * 1e3))
    cols = []
    for f in frequencies:
        _, e = gw.grap_r1_histogram(f)
        e = e[:n_bins * per_ms]
        cols.append(e.reshape(n_bins, per_ms).sum(axis=1))
    return np.stack(cols, axis=1)                                  # (n_bins, nf)


def _raven_window(grap):
    start = int(round(RAVEN_WINDOW_START_S / RAVEN_BIN_S))
    masks = []
    for j in range(grap.shape[1]):
        ok = grap[:, j] > grap[:, j].max() * 10 ** (-RAVEN_DYNAMIC_RANGE_DB / 10)
        ok[:start] = False
        masks.append(ok)
    return np.stack(masks, axis=1)


def _raven_loss_np(e, grap, mask):
    """Mean squared dB difference per band after removing the best level
    offset -- scale-free, since misuka's and RAVEN's absolute energies differ,
    and it leaves the decay (which air attenuation changes) as the target."""
    total = 0.0
    for j in range(grap.shape[1]):
        k = mask[:, j]
        diff = 10 * np.log10(np.maximum(e[k, j], 1e-300)) - 10 * np.log10(grap[k, j])
        total += np.mean((diff - diff.mean()) ** 2)
    return total / grap.shape[1]


RAVEN_VARIANTS = {
    "temperature": (("temperature",), dict(relative_humidity=RAVEN_EXPECTED["relative_humidity"]),
                    [dict(temperature=t) for t in (0.0, 10.0, 40.0)]),
    "relative_humidity": (("relative_humidity",), dict(temperature=RAVEN_EXPECTED["temperature"]),
                          [dict(relative_humidity=rh) for rh in (0.15, 0.3, 0.9)]),
    "joint": (("temperature", "relative_humidity"), {},
              [dict(temperature=t, relative_humidity=rh) for t, rh in JOINT_STARTS]),
}


def run_inverse_raven(variant="joint"):
    """Retrieve T and/or RH so that misuka's decay matches GRAP's RAVEN
    histogram for R1 in RAVEN_FREQUENCIES_HZ. The atmosphere RAVEN used is not
    documented locally; RAVEN_EXPECTED (20 C / 50 % RH, Richter's RAVEN
    standard conditions) is drawn as the expected value, supported by GRAP's
    own T30: the air attenuation it implies at 8-16 kHz is 94-95 % of ISO
    9613-1's at 20 C / 50 %."""
    section = "atmospheric_inverse_raven"
    free, fixed, starts = RAVEN_VARIANTS[variant]
    frequencies, n_freq = RAVEN_FREQUENCIES_HZ, len(RAVEN_FREQUENCIES_HZ)
    fs, max_time, pressure = 1.0 / RAVEN_BIN_S, RAVEN_MAX_TIME, RAVEN_PRESSURE
    n_bins = int(round(max_time * fs))
    grap = _grap_10ms(frequencies, n_bins)
    mask = _raven_window(grap)
    scene, sensor = _medium_scene(frequencies, fs, max_time)
    targets = [10 * np.log10(grap[mask[:, j], j]) for j in range(n_freq)]
    index = [np.flatnonzero(mask[:, j]).astype(np.uint32) for j in range(n_freq)]

    def loss_fn(etc, it_seed):
        total = mi.Float(0.0)
        for j in range(n_freq):
            e = dr.gather(mi.Float, _channel(etc, j, n_freq), mi.UInt32(index[j]))
            e = dr.maximum(e, 1e-30)
            diff = 10 * dr.log(e) / np.log(10) - mi.Float(targets[j])
            total += dr.mean(dr.square(diff - dr.mean(diff)))
        return total / n_freq

    runs = []
    for k, start in enumerate(starts):
        runs.append(_optimise_medium(free, start, fixed, loss_fn, scene, sensor, pressure,
                                     max_time, RAVEN_SPP, seed=k * JOINT_ITERATIONS))
        print(f"  raven/{variant} start {k}: {start} -> "
              f"{dict(zip(free, np.round(runs[-1]['trace'][-1], 3)))}", flush=True)

    # landscape from a fine attenuation-free render, aggregated into 10 ms bins
    fine_fs, per = 8000.0, int(round(8000.0 * RAVEN_BIN_S))
    raw = _attenuation_free_etc(frequencies, fine_fs, max_time, spp=2 ** 18, seed=10 ** 6)
    def agg(x):
        return x[:n_bins * per].reshape(n_bins, per, -1).sum(axis=1)
    landscape = np.empty((len(LANDSCAPE_T), len(LANDSCAPE_RH)))
    for a, t in enumerate(LANDSCAPE_T):
        for b, rh in enumerate(LANDSCAPE_RH):
            landscape[a, b] = _raven_loss_np(agg(_attenuate(raw, frequencies, fine_fs, t, rh, pressure)),
                                             grap, mask)
    alpha_grid = np.array([[[_alpha(f, t, rh, pressure) for f in frequencies] for rh in LANDSCAPE_RH]
                           for t in LANDSCAPE_T])
    alpha_expected = np.array([_alpha(f, RAVEN_EXPECTED["temperature"], RAVEN_EXPECTED["relative_humidity"],
                                      pressure) for f in frequencies])
    starts_full = [dict(fixed, **s) for s in starts]
    return _save_medium_run(
        section, variant, runs, [(s["temperature"], s["relative_humidity"]) for s in starts_full],
        free, frequencies, landscape, alpha_grid, goal=RAVEN_EXPECTED, alpha_goal=alpha_expected,
        fixed=fixed, extra_arrays=dict(grap_10ms=grap, window_mask=mask, raw_landscape_etc_10ms=agg(raw)),
        extra=dict(expected=RAVEN_EXPECTED, expected_note=(
            "RAVEN's atmosphere for GRAP is not documented locally; 20 C / 50 % RH are Richter's "
            "RAVEN standard conditions. GRAP's T30 implies 94-95 % of ISO 9613-1's attenuation at "
            "20 C / 50 % at 8-16 kHz (Eyring estimate)."),
            loss="mean squared dB difference per band after removing the best level offset",
            window_start_s=RAVEN_WINDOW_START_S, dynamic_range_db=RAVEN_DYNAMIC_RANGE_DB,
            bin_s=RAVEN_BIN_S, pressure_pa=pressure, scattering="per material (GRAP tables)"))


def _save_medium_run(section, variant, runs, starts, free, frequencies, landscape, alpha_grid,
                     goal, alpha_goal, fixed=None, extra_arrays=None, extra=None):
    iterations = runs[0]["trace"].shape[0]
    full = np.empty((len(runs), iterations, 2))
    for k, (run, start) in enumerate(zip(runs, starts)):
        for c, name in enumerate(("temperature", "relative_humidity")):
            full[k, :, c] = run["trace"][:, list(free).index(name)] if name in free else start[c]
    finals = full[:, -1, :]
    arrays = dict(trajectories=full, losses=np.stack([r["losses"] for r in runs]),
                  starts=np.array(starts, dtype=float), finals=finals,
                  runtime_per_start_s=np.array([r["runtime_s"] for r in runs]),
                  landscape=landscape, landscape_temperature=LANDSCAPE_T,
                  landscape_relative_humidity=LANDSCAPE_RH, alpha_grid=alpha_grid,
                  alpha_goal=np.asarray(alpha_goal), frequencies_hz=np.array(frequencies),
                  etc_final_per_start=np.stack([r["etc_final"] for r in runs]))
    arrays.update(extra_arrays or {})
    a_best = np.unravel_index(np.argmin(landscape), landscape.shape)
    metrics = dict(free=list(free), fixed=fixed or {}, n_starts=len(runs), iterations=iterations,
                   goal_temperature=goal["temperature"], goal_relative_humidity=goal["relative_humidity"],
                   final_temperature_mean=float(finals[:, 0].mean()), final_temperature_std=float(finals[:, 0].std()),
                   final_relative_humidity_mean=float(finals[:, 1].mean()),
                   final_relative_humidity_std=float(finals[:, 1].std()),
                   landscape_min_temperature=float(LANDSCAPE_T[a_best[0]]),
                   landscape_min_relative_humidity=float(LANDSCAPE_RH[a_best[1]]),
                   final_alpha_per_start=[[_alpha(f, t, rh, (extra or {}).get("pressure_pa", cmp.ATMO_PRESSURE))
                                           for f in frequencies] for t, rh in finals],
                   mean_runtime_per_start_s=float(arrays["runtime_per_start_s"].mean()))
    storage.save_run(section, SCENARIO_NAME, metrics=metrics, arrays=arrays, variant=variant,
                     room=_room_metadata(), extra=extra)
    print(f"done: {section}/{variant} -- finals T={np.round(finals[:, 0], 2)}, RH={np.round(finals[:, 1], 3)}; "
          f"landscape minimum at T={metrics['landscape_min_temperature']:.1f}, "
          f"RH={metrics['landscape_min_relative_humidity']:.2f}", flush=True)
    return metrics


# =====================================================================
#                           shared metadata
# =====================================================================

def _room_metadata():
    return dict(
        source_pos=list(gw.RICHTER_SOURCE_POS), mic_pos=list(gw.RICHTER_MIC_POS),
        source_radius=gw.RICHTER_SOURCE_RADIUS, max_time=MAX_TIME,
        speed_of_sound=gw.RICHTER_SPEED_OF_SOUND,
    )


def _render_params(n_repeats, **kwargs):
    base = dict(
        misuka_spp=SPP, n_repeats=n_repeats,
        seeds=[_seed_for_repeat(i) for i in range(n_repeats)],
        # A runtime number is only interpretable together with how much
        # hardware produced it: the LLVM variant renders multi-threaded
        # (measured ~970% CPU on this machine during the 1c renders, i.e.
        # ~10 of 11 cores), which is the whole reason it beats the
        # single-threaded scalar variant by ~10x. Recorded here because
        # nothing else in the suite captures it.
        dr_thread_count=int(dr.thread_count()),
        host_cpu_count=os.cpu_count(),
        sampling_rate=SAMPLING_RATE, frequencies_hz=list(FREQUENCIES_HZ),
        speed_of_sound=gw.RICHTER_SPEED_OF_SOUND,
        atmo_medium=dict(STANDARD_MEDIUM),
    )
    base.update(kwargs)
    return base


def run_all():
    run_attenuation_frequency_dependence()
    run_attenuation_bin_width()
    run_attenuation_bin_width_etc_error()
    run_attenuation_runtime()
    run_speed_of_sound_extremes()
    run_inverse_retrieval()
    run_inverse_distance_frequency()
    run_inverse_out_of_range()


if __name__ == "__main__":
    run_all()
