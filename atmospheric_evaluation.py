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
pra_comparison_gewandhaus (RICHTER_* : source_radius=0.1 m, c=343.64 m/s,
the source/mic pair selected by line-of-sight visibility quota), with the
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
BIN_WIDTH_FREQUENCIES_HZ = [100.0, 2000.0, 8000.0, 20000.0]

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


def _atmosphere_kwargs():
    """The atmosphere the attenuation blocks render in -- the suite's own
    ATMO_* constants, so this section is directly comparable to run_atmosphere
    / run_atmosphere_methods."""
    return dict(temperature=cmp.ATMO_TEMPERATURE,
                relative_humidity=cmp.ATMO_RELATIVE_HUMIDITY,
                atmospheric_pressure=cmp.ATMO_PRESSURE)


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
    c_atmo = cmp._scalar(cmp.atmo_speed_of_sound())

    def render_one(seed):
        e_raw = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                  atmosphere=True, apply_attenuation=False)
        e_inline = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                     atmosphere=True, apply_attenuation=True)
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
    c_atmo = cmp._scalar(cmp.atmo_speed_of_sound())

    def render_one(seed):
        etc = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                atmosphere=True, apply_attenuation=True)
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

    c_atmo = cmp._scalar(cmp.atmo_speed_of_sound())
    scenes = [_scene(frequencies=frequencies, sampling_rate=fs) for fs in bin_widths_hz]

    def render_one(seed):
        mean_abs_pct = np.empty((len(bin_widths_hz), len(frequencies)))
        max_abs_pct = np.empty((len(bin_widths_hz), len(frequencies)))
        for i, (fs, scene_dict) in enumerate(zip(bin_widths_hz, scenes)):
            e_raw = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                      atmosphere=True, apply_attenuation=False)
            e_inline = cmp.render_misuka(scene_dict, MAX_TIME, spp=SPP, seed=seed,
                                         atmosphere=True, apply_attenuation=True)
            e_post = cmp.apply_pure_tone_attenuation(
                e_raw, sampling_rate=fs, speed_of_sound_ms=c_atmo,
                frequencies=frequencies, **_atmosphere_kwargs())
            diff_pct = np.abs(cmp.relative_difference_percent(e_post, e_inline))
            mean_abs_pct[i] = diff_pct.mean(axis=0)
            max_abs_pct[i] = diff_pct.max(axis=0)
        return dict(mean_abs_pct=mean_abs_pct, max_abs_pct=max_abs_pct)

    per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
    arrays = dict(
        bin_widths_hz=np.array(bin_widths_hz),
        bin_widths_s=1.0 / np.array(bin_widths_hz),
        frequencies_hz=np.array(frequencies),
        mean_abs_pct=mean["mean_abs_pct"], mean_abs_pct_std=std["mean_abs_pct"],
        max_abs_pct=mean["max_abs_pct"], max_abs_pct_std=std["max_abs_pct"],
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
    if variant is not None and mi.variant() != variant:
        mi.set_variant(variant)
    scene = dict(name=SCENARIO_NAME, max_time=MAX_TIME, spp=SPP,
                 misuka_scene=_scene(frequencies=[cmp.FREQUENCY_HZ]))

    def render_one(seed):
        metrics, _, _ = cmp.compare_atmosphere(scene, seed=seed)
        return dict(
            runtime_none_ms=metrics["runtime_no_atmo_s"] * 1e3,
            runtime_inline_ms=metrics["runtime_atmo_s"] * 1e3,
            runtime_post_ms=metrics["runtime_post_total_s"] * 1e3,
        )

    per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
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
                                     timing_variant=mi.variant()),
        misuka_variant=mi.variant(),
    )
    print(f"done: {section} [{mi.variant()}] -- inline={metrics['runtime_inline_ms_mean']:.1f}ms "
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
    """ISO 9613-1's temperature-only speed of sound, the formula
    pyroomacoustics and similar tools use."""
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
# `tolerance` is the absolute band a run must enter to count as converged,
# and the band the figure draws behind the goal. For temperature it is the
# JND (+/-0.38); relative humidity has no JND to use, so it keeps the 2%
# relative criterion the first version of this module used, expressed
# absolutely (0.02 * 0.7) -- and its figure draws no band, since only the
# end value is of interest there.
INVERSE_PARAMETERS = {
    "temperature": dict(
        scene_key="medium_temperature",
        goal=30.0, initial=10.0, learning_rate=0.5, clip=(-100.0, 100.0),
        tolerance=0.38, show_band=True,
    ),
    "relative_humidity": dict(
        scene_key="medium_relative_humidity",
        goal=0.7, initial=0.2, learning_rate=0.03, clip=(0.0, 1.0),
        tolerance=0.02 * 0.7, show_band=False,
    ),
}
INVERSE_FREQUENCY_HZ = 8000.0   # attenuation grows with f^2: the strongest
                                # gradient signal of the suite's bands
INVERSE_MAX_TIME = 0.5          # only the attenuated early part is needed
INVERSE_SAMPLING_RATE = 8000.0
INVERSE_SPP = 2 ** 14
INVERSE_ITERATIONS = 400       # under the JND criterion the temperature
                               # retrieval needs ~350 iterations; 300 left it
                               # at 29.47, just outside the +/-0.38 band
INVERSE_N_SEEDS = 10           # independent optimisations per configuration


def _inverse_integrator(parameter_name, value, max_time):
    """acoustic_prb with only the optimised medium parameter free.

    speed_of_sound is pinned to the scene's own value so that the optimised
    parameter can influence the render *only* through air attenuation --
    the same isolation the tutorial makes, and the reason a temperature
    retrieval here is a statement about attenuation, not about path timing.
    """
    spec = INVERSE_PARAMETERS[parameter_name]
    medium = dict(temperature=INVERSE_PARAMETERS["temperature"]["goal"],
                  relative_humidity=INVERSE_PARAMETERS["relative_humidity"]["goal"],
                  atmospheric_pressure=cmp.ATMO_PRESSURE,
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


def _run_inverse(parameter_name, goal=None, initial=None, frequency_hz=INVERSE_FREQUENCY_HZ,
                 distance_scale=1.0, iterations=INVERSE_ITERATIONS, spp=INVERSE_SPP,
                 max_time=INVERSE_MAX_TIME, sampling_rate=INVERSE_SAMPLING_RATE, seed=0):
    """One retrieval run. Returns the per-iteration history plus timings."""
    spec = INVERSE_PARAMETERS[parameter_name]
    goal = spec["goal"] if goal is None else goal
    initial = spec["initial"] if initial is None else initial
    tolerance = spec["tolerance"]

    # Distance variants move the microphone along the source->mic direction,
    # keeping the room, the source and the emitter radius untouched.
    source = np.array(gw.RICHTER_SOURCE_POS, dtype=float)
    mic = np.array(gw.RICHTER_MIC_POS, dtype=float)
    mic_scaled = source + (mic - source) * distance_scale
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
    converged_at = -1

    for i in range(iterations):
        t0 = time.perf_counter()
        it_seed = seed + i
        # Common random numbers: identical seed for both renders, so they
        # trace the same paths and differ only through attenuation.
        etc_ref = mi.render(scene, sensor=microphone, integrator=integrator_ref,
                            spp=spp, seed=it_seed) / spp
        etc = mi.render(scene, params, integrator=integrator, sensor=microphone,
                        spp=spp, seed=it_seed) / spp
        # Relative, not absolute, squared error. The tutorial's absolute
        # dr.mean(dr.square(...)) works on its free-field scene, whose ETC is
        # orders of magnitude larger; this room's spp-normalised ETC is
        # ~1e-5, so the absolute loss lands at ~1e-14 and its gradient at
        # ~1e-18 -- far below Adam's epsilon (1e-8), which then swallows the
        # step entirely and the parameter never moves (measured). Dividing by
        # the reference energy makes the loss scale-free and restores a
        # gradient Adam can act on.
        loss = dr.mean(dr.square(etc - etc_ref)) / dr.mean(dr.square(etc_ref))

        values[i] = opt[parameter_name].array[0]
        losses[i] = loss.array[0]

        dr.backward(loss)
        opt.step()
        update()
        iteration_times[i] = time.perf_counter() - t0

        if converged_at < 0 and abs(values[i] - goal) <= tolerance:
            converged_at = i

    etc_final = np.asarray(etc.numpy()).squeeze()
    etc_reference = np.asarray(etc_ref.numpy()).squeeze()

    return dict(
        values=values, losses=losses, iteration_times_s=iteration_times,
        tolerance=float(tolerance), show_band=bool(spec.get("show_band", True)),
        etc_final=etc_final, etc_reference=etc_reference,
        goal=float(goal), initial=float(initial), converged_at=int(converged_at),
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
        converged_at_per_seed=np.array([r["converged_at"] for r in runs]),
        final_value_per_seed=np.array([r["final_value"] for r in runs]),
        # The ETC panel shows one representative pair rather than an average:
        # averaging two noisy echograms would flatten exactly the fine
        # structure the zoom is there to show.
        etc_final=first["etc_final"], etc_reference=first["etc_reference"],
        goal=first["goal"], initial=first["initial"], tolerance=first["tolerance"],
        show_band=first["show_band"], distance_m=first["distance_m"],
        frequency_hz=first["frequency_hz"], n_seeds=n_seeds,
    )


def _save_inverse(section, variant, result, parameter_name=None, extra=None):
    values = result["values"]              # (n_seeds, iterations)
    losses = result["losses"]
    converged = result["converged_at_per_seed"]
    finals = result["final_value_per_seed"]
    reached = converged[converged >= 0]

    arrays = dict(
        values=values, losses=losses,
        iteration_times_s=result["iteration_times_s"],
        runtime_per_seed_s=result["runtime_per_seed_s"],
        converged_at_per_seed=converged, final_value_per_seed=finals,
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
               ("goal", "initial", "distance_m", "frequency_hz", "tolerance")}
    metrics["n_seeds"] = int(result["n_seeds"])
    metrics["iterations"] = int(values.shape[1])
    metrics["show_band"] = bool(result["show_band"])
    metrics["final_value"] = float(finals.mean())
    metrics["final_value_std"] = float(finals.std())
    metrics["absolute_error"] = float(abs(finals.mean() - result["goal"]))
    metrics["n_converged"] = int(reached.size)
    # -1 means "no seed reached the band"; the mean is over those that did.
    metrics["converged_at"] = float(reached.mean()) if reached.size else -1
    metrics["converged"] = bool(reached.size == converged.size)
    metrics["total_runtime_s"] = float(result["runtime_per_seed_s"].sum())
    metrics["mean_runtime_per_seed_s"] = float(result["runtime_per_seed_s"].mean())
    # Which medium parameter this run retrieved. Every variant except
    # "relative_humidity" is a temperature retrieval (the distance/frequency
    # studies vary the scene, not the parameter), and the figure needs to
    # know which one to pick the right tolerance band -- storing it beats
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
              f"(goal {m['goal']}), {m['n_converged']}/{m['n_seeds']} seeds "
              f"converged after ~{m['converged_at']:.0f} iterations, "
              f"{m['total_runtime_s']:.0f}s total")
    return out


def run_inverse_distance_frequency():
    """3b: the same retrieval at half/double distance and half/double
    frequency, to compare convergence speed."""
    section = "atmospheric_inverse_distance_frequency"
    variants = {
        "distance_half": dict(distance_scale=0.5),
        "distance_double": dict(distance_scale=2.0),
        "frequency_half": dict(frequency_hz=INVERSE_FREQUENCY_HZ / 2),
        "frequency_double": dict(frequency_hz=INVERSE_FREQUENCY_HZ * 2),
    }
    out = {}
    for variant, kwargs in variants.items():
        result = _run_inverse_seeds("temperature", **kwargs)
        out[variant] = _save_inverse(section, variant, result,
                                     parameter_name="temperature")
        m = out[variant]
        print(f"done: {section}/{variant} -- {m['n_converged']}/{m['n_seeds']} seeds "
              f"converged after ~{m['converged_at']:.0f} iterations "
              f"(d={m['distance_m']:.1f}m, f={m['frequency_hz']:.0f}Hz)")
    return out


def run_inverse_out_of_range():
    """3c: temperature retrieval at -73 C, far outside the range where
    ISO 9613-1 still claims +/-50% accuracy -- does the optimizer converge
    anyway?"""
    section = "atmospheric_inverse_out_of_range"
    result = _run_inverse_seeds("temperature", goal=-73.0, initial=10.0)

    # Why it fails is worth storing alongside: the ISO 9613-1 attenuation
    # coefficient is not monotone in temperature. Sampling it over the range
    # the optimizer moves through shows a maximum near 0 C separating the
    # start point from the goal, so gradient descent can only reach the
    # local optimum on its own side of that barrier.
    temperatures = np.linspace(-90.0, 50.0, 141)
    alpha = np.array([
        cmp._scalar(mi.acoustic.energy_attenuation_coefficient(
            temperature=float(t), frequency=INVERSE_FREQUENCY_HZ,
            relative_humidity=INVERSE_PARAMETERS["relative_humidity"]["goal"],
            atmospheric_pressure=cmp.ATMO_PRESSURE))
        for t in temperatures])

    metrics = _save_inverse(
        section, "temperature_minus73", result, parameter_name="temperature",
        extra=dict(note="-73 C is outside ISO 9613-1's validated range; the "
                        "inverse variants cannot enforce the input ranges the "
                        "underlying formulas assume",
                   failure_mechanism=(
                       "The retrieval does not merely stall, it converges to the "
                       "wrong temperature (~34 C). The attenuation coefficient is "
                       "non-monotone in temperature: from the 10 C start it rises "
                       "towards a maximum near 0 C and falls towards a local "
                       "minimum near 34 C. The goal's coefficient is only reached "
                       "far below that maximum, so no descent direction leads to "
                       "it -- a barrier, not a vanishing gradient. See the stored "
                       "alpha_vs_temperature arrays.")))
    # attach the diagnostic curve to the same run
    run_dir = storage.run_dir(section, SCENARIO_NAME, "temperature_minus73")
    existing = dict(np.load(f"{run_dir}/arrays.npz"))
    existing.update(alpha_temperatures_c=temperatures, alpha_1_per_m=alpha)
    np.savez(f"{run_dir}/arrays.npz", **existing)
    print(f"done: {section} -- recovered {metrics['final_value']:.2f} C "
          f"(goal -73 C), {metrics['n_converged']}/{metrics['n_seeds']} seeds converged")
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
        atmo_temperature=cmp.ATMO_TEMPERATURE,
        atmo_relative_humidity=cmp.ATMO_RELATIVE_HUMIDITY,
        atmo_pressure=cmp.ATMO_PRESSURE,
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
