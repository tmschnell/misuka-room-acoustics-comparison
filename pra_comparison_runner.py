"""Orchestration layer for the misuka-vs-pyroomacoustics comparison suite:
one run_<section>() function per comparison, each doing exactly what its
corresponding notebook cell used to do (same calls into
test_pyroomacoustics_comparison, same metric functions, same order of
operations) but saving the result via pra_comparison_storage instead of
leaving it in an in-notebook variable.

Two cross-cutting behaviors live here, applied uniformly to every section
before any section-specific content is reworked:

1. Variant selection: every render call goes through cmp's own
   render_misuka/render_pra, which read whatever variant mi.set_variant()
   last selected -- this module's own __main__ entrypoint (and every other
   place in this suite that calls mi.set_variant) requests
   ("cuda_acoustic", "llvm_ad_acoustic"), i.e. prefer the CUDA JIT backend,
   falling back to the LLVM one where no GPU/cuda_acoustic build is
   available (this machine: mi.variants() has no cuda_acoustic, so it falls
   back transparently -- verified, see the acceptance script).

2. Repeats for anything ray-tracing-based: N_REPEATS independent runs, each
   with its own seed (misuka's `seed` render parameter *and*
   pyroomacoustics.random.seed(), see _seed_for_repeat), aggregated via
   _repeat_and_aggregate() into per-repeat + mean + std arrays/metrics,
   stored alongside n_repeats/seeds as render_params metadata. Sections (or
   sub-parts of a section) that are purely analytic/deterministic --
   PRA's image_source_model() with ray_tracing never invoked -- are *not*
   repeated (repeating a deterministic computation would just waste time
   reproducing the exact same result n_repeats times): see
   run_appendix_ism_peak, where the ISM reference is computed once outside
   the repeat loop but misuka's own sweep is repeated inside it.

   For backward compatibility with the not-yet-reworked plotting functions
   in pra_comparison_plots.py (still expecting a single bare array/scalar
   per key, e.g. run["arrays"]["e_misuka"]), every repeated quantity is
   *also* stored under its original bare key, aliased to the repeat mean --
   existing plots keep working unmodified against freshly-regenerated data,
   until each section is reworked (in later prompts) to read `{key}_mean`/
   `{key}_std` explicitly and use plot_with_std_band
   (pra_comparison_plots.py) instead.

Run directly (`python pra_comparison_runner.py`) to (re)generate every
stored result under pra_comparison_data/. The notebook only ever reads
from there via pra_comparison_storage.load_run() -- it never calls
cmp.render_*/cmp.compare*/pyrato.* itself anymore.

run_main_comparison() and run_atmosphere() also return the same raw
(scenario/mode -> arrays) data they save, for standalone callers that want
to chain into further derived analysis without re-rendering.
"""

import numpy as np
import mitsuba as mi
import pyroomacoustics as pra

import test_pyroomacoustics_comparison as cmp
import pra_comparison_storage as storage
from pra_comparison_metrics import (
    pyrato_metrics, parameters_full, JND_ABS, JND_REL, jnd_band,
    cosine_similarity, rmspe, mpe, rel_pct,
    pra_ism_peak_times_and_energies, build_peak_windows,
    integrate_ism_energy_in_windows, integrate_etc_energy_in_windows,
    check_alignment_offset_negligible,
)

# Same ray-count sweep used throughout the original notebook's ray-count-
# convergence and appendix sections.
SPP_VALUES = [2 ** k for k in range(6, 25)]
SPP_SWEEP_SCENARIO = cmp.scenario_shoebox_diffuse

# ---------------------------------------------------------------------
# Repeats for ray-tracing-based renderings (Task 2). A single, central knob
# -- not hardcoded per call site, per section, or per render function.
# Change here to affect every run_<section>() function at once.
#
# Cost note: several sections already sweep 21 spp values per scenario
# (SPP_VALUES above); N_REPEATS multiplies that cost directly for those
# (run_ray_count_convergence, run_appendix_ism_peak, run_appendix_hybrid_
# params). Consider passing a smaller n_repeats explicitly to just those
# calls for an initial/interactive pass rather than raising N_REPEATS
# itself, which would also inflate the cheaper sections.
# ---------------------------------------------------------------------
N_REPEATS = 3
BASE_SEED = 0


def _seed_for_repeat(repeat_index):
    return BASE_SEED + repeat_index


def _repeat_and_aggregate(render_one, n_repeats):
    """Call render_one(seed) n_repeats times, each with a different,
    deterministic seed (_seed_for_repeat) -- and, since pyroomacoustics'
    ray tracer reads its own global RNG rather than taking a seed
    parameter, pyroomacoustics.random.seed(seed) is also set immediately
    before each call (harmless no-op for renders that don't use PRA's ray
    tracer at all, e.g. run_atmosphere/run_atmosphere_methods).

    render_one(seed) must return a dict of {name: value}, value a scalar or
    a numpy array, with the *same* keys and shapes on every call (misuka's
    render_misuka is deterministic regardless of seed on this build/variant
    -- verified previously -- so its own repeats mainly capture floating-
    point reduction-order noise from its parallel LLVM backend, not genuine
    sampling variance; pyroomacoustics' stochastic ray tracer, where used,
    *does* vary meaningfully between repeats).

    Returns (per_repeat, mean, std), each a dict with the same keys as
    render_one's return value: per_repeat[key] stacks all n_repeats raw
    values along a new leading axis, mean/std collapse that axis.
    """
    results = []
    for i in range(n_repeats):
        seed = _seed_for_repeat(i)
        pra.random.seed(seed)
        results.append(render_one(seed))

    per_repeat, mean, std = {}, {}, {}
    for key in results[0]:
        stacked = np.stack([np.asarray(r[key]) for r in results])
        per_repeat[key] = stacked
        mean[key] = stacked.mean(axis=0)
        std[key] = stacked.std(axis=0)
    return per_repeat, mean, std


def _repeated_arrays_and_metrics(per_repeat, mean, std):
    """Split _repeat_and_aggregate's output into (arrays, scalar_metrics)
    ready for storage.save_run -- every key gets `{key}_per_repeat` (the
    full stack), `{key}_mean`, `{key}_std` in `arrays`; scalar-valued keys
    (ndim == 0) additionally get `{key}_mean`/`{key}_std` in `metrics` (so a
    quick look at metadata.json doesn't require loading the npz). The bare
    `{key}` name is *also* kept (aliased to the mean) for backward
    compatibility with not-yet-reworked plotting code -- see this module's
    docstring.
    """
    arrays, metrics = {}, {}
    for key in per_repeat:
        mean_val = np.asarray(mean[key])
        arrays[key] = mean_val
        arrays[f"{key}_per_repeat"] = per_repeat[key]
        arrays[f"{key}_mean"] = mean_val
        arrays[f"{key}_std"] = np.asarray(std[key])
        if mean_val.ndim == 0:
            metrics[key] = float(mean_val)
            metrics[f"{key}_mean"] = float(mean_val)
            metrics[f"{key}_std"] = float(std[key])
    return arrays, metrics


def _room_metadata(scenario, pra_room=None):
    """Best-effort room/render metadata. auditorium_complex's misuka_scene is
    a loaded mi.Scene (not a dict) and has no pra_room_factory, so several
    fields are simply omitted for it -- readers must use .get().
    """
    room = {"max_time": scenario["max_time"]}
    misuka_scene = scenario["misuka_scene"]
    if isinstance(misuka_scene, dict):
        room["source_pos"] = list(misuka_scene["emitter"]["center"])
        room["source_radius"] = misuka_scene["emitter"]["radius"]
        room["mic_pos"] = list(misuka_scene["mic"]["origin"])
    if pra_room is not None:
        room["volume_m3"] = float(pra_room.get_volume())
        # Representative single value -- rooms with per-wall material
        # overrides (e.g. flutter_corridor's end walls) have more than one
        # distinct (absorption, scattering) pair; only the first wall's is
        # recorded here. Fine for descriptive metadata, not used by any
        # computation.
        room["absorption"] = float(pra_room.walls[0].absorption[0])
        room["scattering"] = float(pra_room.walls[0].scatter[0])
    return room


def _render_params(**kwargs):
    base = dict(misuka_seed=0, sampling_rate=cmp.SAMPLING_RATE,
                speed_of_sound=cmp.SPEED_OF_SOUND, n_repeats=1)
    base.update(kwargs)
    return base


def _repeat_render_params(n_repeats, **kwargs):
    return _render_params(n_repeats=n_repeats,
                          seeds=[_seed_for_repeat(i) for i in range(n_repeats)], **kwargs)


# =====================================================================
#           Section: main comparison -- ray-tracing-based (repeated)
# =====================================================================

def run_main_comparison(n_repeats=N_REPEATS):
    raw = {}
    for scenario_fn in cmp.PRA_SCENARIOS:
        scenario = scenario_fn()
        for mode_name, pra_order in cmp.PRA_MODES:

            def render_one(seed, scenario=scenario, pra_order=pra_order):
                metrics, e_misuka, e_pra = cmp.compare(scenario, pra_order, seed=seed)
                pm_misuka = pyrato_metrics(e_misuka)
                pm_pra = pyrato_metrics(e_pra)
                out = {
                    "e_misuka": e_misuka, "e_pra": e_pra,
                    "edc_corr": metrics["edc_correlation"],
                    "runtime_misuka_ms": metrics["runtime_misuka_s"] * 1e3,
                    "runtime_pra_ms": metrics["runtime_pra_s"] * 1e3,
                }
                for key in ("T20", "T30", "C50", "D50"):
                    out[f"{key}_misuka"] = pm_misuka[key]
                    out[f"{key}_pra"] = pm_pra[key]
                if "flutter_misuka" in metrics:
                    out["flutter_misuka"] = metrics["flutter_misuka"]
                    out["flutter_pra"] = metrics["flutter_pra"]
                return out

            per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
            arrays, result_metrics = _repeated_arrays_and_metrics(per_repeat, mean, std)
            for key in ("T20", "T30", "C50", "D50"):
                result_metrics[f"{key}_delta"] = (result_metrics[f"{key}_misuka_mean"]
                                                  - result_metrics[f"{key}_pra_mean"])

            raw[(scenario["name"], mode_name)] = (arrays["e_misuka"], arrays["e_pra"])

            room = scenario["pra_room_factory"](pra_order)
            storage.save_run(
                "main_comparison", scenario["name"], variant={"pra_mode": mode_name},
                metrics=result_metrics, arrays=arrays,
                room=_room_metadata(scenario, room),
                render_params=_repeat_render_params(
                    n_repeats, misuka_spp=cmp.MISUKA_SPP, pra_order=pra_order,
                    pra_mode=mode_name, pra_n_rays=cmp.PRA_N_RAYS),
            )
            print(f"done: {scenario['name']:24s} / {mode_name}  ({n_repeats} repeat(s))")
    return raw


# =====================================================================
#           Section: atmosphere on/off -- ray-tracing-based (repeated)
# =====================================================================

def run_atmosphere(n_repeats=N_REPEATS):
    raw_atmo = {}
    for scenario_fn in cmp.SCENARIOS:
        scenario = scenario_fn()

        def render_one(seed, scenario=scenario):
            metrics, e_no_atmo, e_atmo = cmp.compare_atmosphere(scenario, seed=seed)
            pm_no_atmo = pyrato_metrics(e_no_atmo)
            pm_atmo = pyrato_metrics(e_atmo)
            out = {
                "e_no_atmo": e_no_atmo, "e_atmo": e_atmo,
                "edc_corr": metrics["edc_correlation"],
                "runtime_no_atmo_ms": metrics["runtime_no_atmo_s"] * 1e3,
                "runtime_atmo_ms": metrics["runtime_atmo_s"] * 1e3,
                "runtime_overhead_pct": metrics["runtime_overhead_pct"],
                "runtime_post_total_ms": metrics["runtime_post_total_s"] * 1e3,
                "runtime_post_overhead_pct": metrics["runtime_post_overhead_pct"],
            }
            for key in ("T20", "T30", "C50", "D50"):
                out[f"{key}_no_atmo"] = pm_no_atmo[key]
                out[f"{key}_atmo"] = pm_atmo[key]
            return out

        per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
        arrays, result_metrics = _repeated_arrays_and_metrics(per_repeat, mean, std)
        for key in ("T20", "T30", "C50", "D50"):
            result_metrics[f"{key}_delta"] = (result_metrics[f"{key}_atmo_mean"]
                                              - result_metrics[f"{key}_no_atmo_mean"])

        raw_atmo[scenario["name"]] = (arrays["e_no_atmo"], arrays["e_atmo"])

        pra_room = None
        if scenario.get("pra_room_factory") is not None:
            pra_room = scenario["pra_room_factory"](cmp.RT_HYBRID_ORDER)
        storage.save_run(
            "atmosphere", scenario["name"],
            metrics=result_metrics, arrays=arrays,
            room=_room_metadata(scenario, pra_room),
            render_params=_repeat_render_params(n_repeats, misuka_spp=scenario.get("spp", cmp.MISUKA_SPP)),
        )
        print(f"done: {scenario['name']:24s} (atmosphere on/off, {n_repeats} repeat(s))")
    return raw_atmo


# =====================================================================
#   Section: atmosphere methods (inline vs. post-processing) -- RT-based
# =====================================================================

# Explicit frequencies the post-vs-inline attenuation discrepancy is
# reported at (Task: "Erweiterung Post-processing vs. inline attenuation"),
# in addition to the single FREQUENCY_HZ (1000 Hz) the rest of this suite
# renders at -- misuka's spectral "tape" film supports several simultaneous
# frequency bands natively (one column per frequency, see tape.cpp's
# comma/space-tokenized `frequencies` property), so all six are rendered in
# one call rather than six separate renders.
HIGHLIGHT_FREQUENCIES_HZ = [100.0, 1000.0, 4000.0, 8000.0, 14000.0, 20000.0]


def _scene_dict_with_frequencies(misuka_scene, frequencies):
    """Shallow-copies just the nested dicts needed to override the tape
    film's frequency list. Not a full deepcopy: the scene dict also holds
    mitsuba Transform objects elsewhere that aren't deepcopy/pickle-able.
    """
    scene_dict = dict(misuka_scene)
    scene_dict["mic"] = dict(misuka_scene["mic"])
    scene_dict["mic"]["film"] = dict(misuka_scene["mic"]["film"])
    scene_dict["mic"]["film"]["frequencies"] = ",".join(str(f) for f in frequencies)
    return scene_dict


def _multi_frequency_post_vs_inline(scenario, seed, frequencies=HIGHLIGHT_FREQUENCIES_HZ):
    """Same inline-vs-post-processing comparison as
    cmp.compare_atmosphere_methods, but rendered simultaneously at several
    frequency bands instead of the scenario's single default FREQUENCY_HZ --
    so the discrepancy between the two attenuation-application methods can
    be read off explicitly per frequency, not only implicitly in one curve
    at one frequency. Only called for dict-based scenes (the caller checks
    `isinstance(scenario["misuka_scene"], dict)` -- excludes
    auditorium_complex, which loads from an external mesh file with a
    separately-loaded sensor, not a plain dict that's cheap to retarget to a
    different frequency set).
    """
    scene_dict = _scene_dict_with_frequencies(scenario["misuka_scene"], frequencies)
    spp = scenario.get("spp", cmp.MISUKA_SPP)

    e_raw = cmp.render_misuka(scene_dict, scenario["max_time"], spp=spp, seed=seed,
                              atmosphere=True, apply_attenuation=False)
    e_inline = cmp.render_misuka(scene_dict, scenario["max_time"], spp=spp, seed=seed,
                                 atmosphere=True, apply_attenuation=True)
    c_atmo = cmp.atmo_speed_of_sound()
    e_post = cmp.apply_pure_tone_attenuation(
        e_raw, sampling_rate=cmp.SAMPLING_RATE, speed_of_sound_ms=c_atmo,
        temperature=cmp.ATMO_TEMPERATURE, frequencies=frequencies,
        relative_humidity=cmp.ATMO_RELATIVE_HUMIDITY, atmospheric_pressure=cmp.ATMO_PRESSURE)

    # (n_time, n_freq) -> EDC per frequency column, same as
    # cmp.compare_atmosphere_methods (schroeder_edc_db is 1D-only).
    edc_inline = np.apply_along_axis(cmp.schroeder_edc_db, 0, e_inline)
    edc_post = np.apply_along_axis(cmp.schroeder_edc_db, 0, e_post)
    diff_pct = cmp.relative_difference_percent(edc_post, edc_inline)  # (n_time, n_freq)
    diff_pct_abs = np.abs(diff_pct)

    return {
        "freq_e_inline": e_inline, "freq_e_post": e_post,
        "freq_diff_pct": diff_pct, "freq_diff_pct_abs": diff_pct_abs,
        # Per-frequency summaries (shape (n_freq,)): mean/max magnitude over
        # the whole render duration -- the explicit, table-ready numbers;
        # the curve above is for those who want the time evolution too.
        "freq_mean_abs_pct": np.mean(diff_pct_abs, axis=0),
        "freq_max_abs_pct": np.max(diff_pct_abs, axis=0),
    }


def run_atmosphere_methods(n_repeats=N_REPEATS):
    for scenario_fn in cmp.SCENARIOS:
        scenario = scenario_fn()
        is_dict_scene = isinstance(scenario["misuka_scene"], dict)

        def render_one(seed, scenario=scenario, is_dict_scene=is_dict_scene):
            out = cmp.compare_atmosphere_methods(scenario, seed=seed)
            if is_dict_scene:
                out.update(_multi_frequency_post_vs_inline(scenario, seed))
            return out

        per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
        arrays, _ = _repeated_arrays_and_metrics(per_repeat, mean, std)

        extra_render_params = {}
        if is_dict_scene:
            extra_render_params["highlight_frequencies_hz"] = HIGHLIGHT_FREQUENCIES_HZ
        storage.save_run(
            "atmosphere_methods", scenario["name"],
            arrays=arrays,
            room=_room_metadata(scenario),
            render_params=_repeat_render_params(n_repeats, misuka_spp=scenario.get("spp", cmp.MISUKA_SPP),
                                                **extra_render_params),
        )
        print(f"done: {scenario['name']:24s} (inline vs. post-processing, same speed of sound, "
              f"{n_repeats} repeat(s))")


# =====================================================================
#           Section: ray-count convergence -- ray-tracing-based
#                        (misuka sweep + PRA hybrid reference)
# =====================================================================

def run_ray_count_convergence(n_repeats=N_REPEATS):
    scenario = SPP_SWEEP_SCENARIO()

    def render_one(seed):
        sweep = cmp.ray_count_convergence(SPP_SWEEP_SCENARIO, SPP_VALUES, seed=seed)
        err_no_atmo = np.array(sweep["errors_no_atmo"])
        err_atmo = np.array(sweep["errors_atmo"])
        rel_err_no_atmo_pct = 100 * (10 ** (err_no_atmo / 10) - 1)
        rel_err_atmo_pct = 100 * (10 ** (err_atmo / 10) - 1)

        pra_ref_metrics = pyrato_metrics(sweep["e_pra"])
        out = dict(errors_no_atmo=err_no_atmo, errors_atmo=err_atmo,
                   rel_err_no_atmo_pct=rel_err_no_atmo_pct, rel_err_atmo_pct=rel_err_atmo_pct,
                   e_pra=np.asarray(sweep["e_pra"]))
        for k, v in pra_ref_metrics.items():
            out[f"{k}_pra"] = v
        for key in ("T20", "T30", "C50", "D50"):
            no_atmo_vals, atmo_vals, no_atmo_rel, atmo_rel = [], [], [], []
            for e_no_atmo_i, e_atmo_i in zip(sweep["energies_no_atmo"], sweep["energies_atmo"]):
                pm_no_atmo = pyrato_metrics(e_no_atmo_i)
                pm_atmo = pyrato_metrics(e_atmo_i)
                no_atmo_vals.append(pm_no_atmo[key])
                atmo_vals.append(pm_atmo[key])
                no_atmo_rel.append(rel_pct(pm_no_atmo[key], pra_ref_metrics[key]))
                atmo_rel.append(rel_pct(pm_atmo[key], pra_ref_metrics[key]))
            out[f"{key}_no_atmo"] = np.array(no_atmo_vals)
            out[f"{key}_atmo"] = np.array(atmo_vals)
            out[f"{key}_no_atmo_rel_pct"] = np.array(no_atmo_rel)
            out[f"{key}_atmo_rel_pct"] = np.array(atmo_rel)
        return out

    per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
    arrays, metrics = _repeated_arrays_and_metrics(per_repeat, mean, std)
    arrays["spp"] = np.array(SPP_VALUES)  # the sweep's x-axis is fixed, not repeated

    storage.save_run(
        "ray_count_convergence", scenario["name"],
        metrics=metrics, arrays=arrays,
        room=_room_metadata(scenario, scenario["pra_room_factory"](cmp.RT_HYBRID_ORDER)),
        render_params=_repeat_render_params(n_repeats, spp_values=SPP_VALUES,
                                            pra_order=cmp.RT_HYBRID_ORDER, pra_mode="hybrid"),
    )
    print(f"done: {scenario['name']:24s} (ray-count convergence, {len(SPP_VALUES)} spp values, "
          f"{n_repeats} repeat(s))")


# =====================================================================
#     Appendix 1: ISM peak level -- MIXED: PRA-ISM reference is purely
#     analytic (image_source_model() only, ray tracing never invoked, see
#     pra_ism_peak_times_and_energies's own docstring) and computed once;
#     misuka's own sweep is ray-tracing-based and repeated.
# =====================================================================

def run_appendix_ism_peak(n_repeats=N_REPEATS):
    ism_peak_order = cmp.RT_HYBRID_ORDER
    for scenario_fn in cmp.PRA_SCENARIOS:
        scenario = scenario_fn()
        source_radius = scenario["misuka_scene"]["emitter"]["radius"]

        # Deterministic -- no ray tracing, no repeat needed.
        peak_times, peak_energies = pra_ism_peak_times_and_energies(scenario, ism_peak_order)
        windows = build_peak_windows(peak_times, source_radius)
        ism_window_energy = integrate_ism_energy_in_windows(peak_times, peak_energies, windows)
        ref = ism_window_energy / ism_window_energy[0]

        def render_one(seed, scenario=scenario, windows=windows, ref=ref):
            misuka_window_energy = {}
            for n_rays in SPP_VALUES:
                etc = cmp.render_misuka(scenario["misuka_scene"], scenario["max_time"], spp=n_rays, seed=seed)
                etc_aligned = cmp.align_hull_to_center(etc, source_radius, cmp.SAMPLING_RATE)
                misuka_window_energy[n_rays] = integrate_etc_energy_in_windows(
                    etc_aligned, cmp.SAMPLING_RATE, windows)

            n_show = list(misuka_window_energy.keys())
            cs_vals, rmspe_vals, mpe_vals = [], [], []
            with np.errstate(divide="ignore", invalid="ignore"):
                for n_rays in n_show:
                    v = misuka_window_energy[n_rays] / misuka_window_energy[n_show[-1]][0]
                    cs_vals.append(cosine_similarity(v, ref) if np.all(np.isfinite(v)) else np.nan)
                    rmspe_vals.append(rmspe(v, ref) if np.all(np.isfinite(v)) else np.nan)
                    mpe_vals.append(mpe(v, ref) if np.all(np.isfinite(v)) else np.nan)

            out = dict(cosine_similarity=np.array(cs_vals), rmspe_pct=np.array(rmspe_vals),
                      mpe_pct=np.array(mpe_vals))
            for n_rays in n_show:
                out[f"misuka_window_energy_spp{n_rays}"] = misuka_window_energy[n_rays]
            return out

        per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
        arrays, result_metrics = _repeated_arrays_and_metrics(per_repeat, mean, std)
        arrays["spp"] = np.array(SPP_VALUES)
        arrays["ism_window_energy"] = ism_window_energy
        result_metrics["n_max"] = SPP_VALUES[-1]

        storage.save_run(
            "appendix_ism_peak", scenario["name"],
            metrics=result_metrics, arrays=arrays,
            room=_room_metadata(scenario),
            render_params=_repeat_render_params(n_repeats, spp_values=SPP_VALUES, pra_order=ism_peak_order),
        )
        print(f"done: {scenario['name']:24s} (ISM peak-level, n_max={SPP_VALUES[-1]}, {n_repeats} repeat(s))")


# =====================================================================
#           Appendix 2: Hybrid parameter level -- ray-tracing-based
#              (PRA hybrid reference *and* misuka sweep, repeated)
# =====================================================================

def run_appendix_hybrid_params(n_repeats=N_REPEATS):
    for scenario_fn in cmp.PRA_SCENARIOS:
        scenario = scenario_fn()
        src = np.array(scenario["misuka_scene"]["emitter"]["center"])
        mic = np.array(scenario["misuka_scene"]["mic"]["origin"])
        source_radius = scenario["misuka_scene"]["emitter"]["radius"]
        distance0 = float(np.linalg.norm(src - mic))
        n_time_bins = int(round(scenario["max_time"] * cmp.SAMPLING_RATE))
        param_keys = list(JND_ABS) + list(JND_REL)
        check_alignment_offset_negligible(source_radius, scenario_name=scenario["name"],
                                          speed_of_sound=scenario.get("speed_of_sound", cmp.SPEED_OF_SOUND))
        # per-scenario override -- e.g. scenario_gewandhaus's own, much
        # higher-reaching sweep (misuka is cheap there even well past its
        # necessary_rays; see that scenario's own module docstring) instead
        # of the shoebox-scale default.
        spp_values = scenario.get("spp_values", SPP_VALUES)
        c = scenario.get("speed_of_sound", cmp.SPEED_OF_SOUND)

        def render_one(seed, scenario=scenario, spp_values=spp_values, c=c):
            # PRA's hybrid room uses ray_tracing=True (see pra_box_room/
            # pra_lroom_room) -- rebuilding + re-computing it per repeat is
            # what makes this reference vary across repeats too.
            #
            # Both sides are time-aligned before parameters_full() (see the
            # H1 t0-alignment investigation): misuka's ETC is shifted later
            # by the hull-vs-center offset (lossless -- see
            # shift_later_lossless's docstring for why the lossy
            # shift_series/align_hull_to_center is unsuitable at this
            # source_radius), and PRA's RIR has its frac_delay_length//2
            # lead-in stripped. All six parameters (not just C50) are
            # computed from these aligned signals -- verified to be neutral
            # to mildly positive for T60/C80, and to close (C50) or
            # meaningfully shrink (TS) the previously-unaligned gap where a
            # real gap existed.
            room_hybrid = scenario["pra_room_factory"](cmp.RT_HYBRID_ORDER)
            e_pra_hybrid = cmp.render_pra_frac_delay_stripped(room_hybrid, n_time_bins)
            reference_params = parameters_full(e_pra_hybrid, distance0, source_radius, speed_of_sound=c,
                                               g_radius_term=0.0)

            out = {f"reference_{p}": reference_params[p] for p in param_keys}
            per_param_series = {p: [] for p in param_keys}
            for n_rays in spp_values:
                etc = cmp.render_misuka(scenario["misuka_scene"], scenario["max_time"], spp=n_rays, seed=seed,
                                        speed_of_sound=c)
                etc_aligned = cmp.align_hull_to_center_lossless(etc, source_radius, cmp.SAMPLING_RATE,
                                                                speed_of_sound=c)
                params = parameters_full(etc_aligned, distance0, source_radius, speed_of_sound=c,
                                         g_radius_term=2 * source_radius)
                for p in param_keys:
                    per_param_series[p].append(params[p])
            for p in param_keys:
                out[f"misuka_{p}"] = np.array(per_param_series[p])
            return out

        per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
        arrays, result_metrics = _repeated_arrays_and_metrics(per_repeat, mean, std)
        arrays["spp"] = np.array(spp_values)
        result_metrics["n_max"] = spp_values[-1]
        for p in param_keys:
            # misuka_{p} is per-spp (array-valued, so its mean/std only ever
            # land in `arrays`, not `metrics` -- see
            # _repeated_arrays_and_metrics); reference_{p} is a scalar
            # (present in both). Delta at the highest spp, matching the
            # original single-render code's own n_show2[-1] convention.
            result_metrics[f"{p}_delta"] = float(arrays[f"misuka_{p}_mean"][-1] - arrays[f"reference_{p}_mean"])
        # Backward-compatible nested dict, matching the original (pre-repeat)
        # single-dict reference_params -- plot_hybrid_jnd (not yet reworked
        # for repeats) reads exactly this key.
        result_metrics["reference_params"] = {p: float(arrays[f"reference_{p}_mean"]) for p in param_keys}

        extra = {
            "alignment": (
                "G/EDT/TS/C50/T60/C80 are computed from time-aligned "
                "signals: misuka's ETC shifted later by the hull-vs-center "
                "offset (source_radius/speed_of_sound, lossless shift) and "
                "PRA's RIR with its frac_delay_length//2-sample lead-in "
                "stripped -- see check_alignment_offset_negligible() and "
                "parameters_full()'s g_radius_term docstring. Applied "
                "uniformly to all six parameters, not selectively."
            ),
        }
        if scenario["name"] == "shoebox_specular_5000m3":
            extra["ts_note"] = (
                "TS_delta crosses the JND boundary in the noise band after "
                "this alignment fix (was ~0.0085, now ~0.0105, vs JND "
                "0.010) -- a boundary crossing within run-to-run noise, not "
                "a real regression caused by the alignment; do not read "
                "this as 'alignment made TS worse'."
            )

        storage.save_run(
            "appendix_hybrid_params", scenario["name"],
            metrics=result_metrics, arrays=arrays,
            room=_room_metadata(scenario, scenario["pra_room_factory"](cmp.RT_HYBRID_ORDER)),
            render_params=_repeat_render_params(n_repeats, spp_values=spp_values,
                                                pra_order=cmp.RT_HYBRID_ORDER, pra_mode="hybrid"),
            extra=extra,
        )
        print(f"done: {scenario['name']:24s} (Hybrid parameter-level, n_max={spp_values[-1]}, "
              f"{n_repeats} repeat(s))")


# =====================================================================
#           Appendix 3: RT-only histogram level -- ray-tracing-based
# =====================================================================

def run_appendix_rt_histogram(n_repeats=N_REPEATS):
    rt_only_order = dict(cmp.PRA_MODES)["rt_only"]
    for scenario_fn in cmp.PRA_SCENARIOS:
        scenario = scenario_fn()
        source_radius = scenario["misuka_scene"]["emitter"]["radius"]
        n_time_bins = int(round(scenario["max_time"] * cmp.SAMPLING_RATE))

        def render_one(seed, scenario=scenario):
            room_rt = scenario["pra_room_factory"](rt_only_order, hist_bin_size=1.0 / cmp.SAMPLING_RATE)
            room_rt.compute_rir()
            pra_rt_etc = cmp.pra_histogram_aligned_etc(room_rt, source_radius, cmp.SAMPLING_RATE,
                                                       n_bins=n_time_bins)
            etc_misuka = cmp.render_misuka(scenario["misuka_scene"], scenario["max_time"],
                                          spp=SPP_VALUES[-1], seed=seed)

            direct_bin_misuka = cmp.leading_edge_bin(etc_misuka)
            direct_bin_pra = cmp.leading_edge_bin(pra_rt_etc)
            early_bins = min(len(etc_misuka), len(pra_rt_etc), int(0.1 * cmp.SAMPLING_RATE))
            lag = cmp.cross_correlation_lag(etc_misuka[:early_bins], pra_rt_etc[:early_bins], max_lag=10)
            dt = 1.0 / cmp.SAMPLING_RATE
            return dict(etc_misuka=etc_misuka, pra_rt_etc=pra_rt_etc,
                       direct_ms_misuka=direct_bin_misuka * dt * 1e3,
                       direct_ms_pra=direct_bin_pra * dt * 1e3,
                       cross_corr_lag_ms=lag * dt * 1e3)

        per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
        arrays, result_metrics = _repeated_arrays_and_metrics(per_repeat, mean, std)

        print(f"{scenario['name']:24s} direct sound: misuka={result_metrics['direct_ms_misuka_mean']:.3f} ms  "
              f"PRA-histogram={result_metrics['direct_ms_pra_mean']:.3f} ms   "
              f"cross-corr lag (first 100 ms)={result_metrics['cross_corr_lag_ms_mean']:.3f} ms  "
              f"({n_repeats} repeat(s))")

        storage.save_run(
            "appendix_rt_histogram", scenario["name"],
            metrics=result_metrics, arrays=arrays,
            room=_room_metadata(scenario, scenario["pra_room_factory"](rt_only_order)),
            render_params=_repeat_render_params(n_repeats, misuka_spp=SPP_VALUES[-1],
                                                pra_order=rt_only_order, pra_mode="rt_only"),
        )


# =====================================================================
#   Richter Fig.14 reproduction: misuka (ours) vs. misuka (Richter's own
#   Masterarbeit results for Gewandhaus) -- NOT misuka-vs-PRA, so this is
#   deliberately not part of PRA_SCENARIOS/run_all/run_edc_runtime_and_
#   appendix (same exclusion pattern as CR4/scenario_gewandhaus). See
#   pra_comparison_gewandhaus.py's RICHTER_* constants for the full
#   derivation of every setting used here (source_radius, f=20Hz, c,
#   position, floor-split materials) and their remaining, documented
#   limitations (no real R1 coordinate, TS stays open).
# =====================================================================

def run_richter_gewandhaus_reproduction(n_repeats=50):
    import pra_comparison_gewandhaus as gw

    max_time = 4.0
    spp = 8_388_608  # already-converged value, see the conversation's own timing/convergence checks
    scene_dict = gw.build_misuka_scene_dict_richter(max_time, cmp.SAMPLING_RATE)
    distance0 = float(np.linalg.norm(np.array(gw.RICHTER_SOURCE_POS) - np.array(gw.RICHTER_MIC_POS)))
    param_keys = ["G", "EDT", "TS", "C50", "T60", "C80"]

    def render_one(seed):
        etc = cmp.render_misuka(scene_dict, max_time, spp=spp, seed=seed,
                                speed_of_sound=gw.RICHTER_SPEED_OF_SOUND)
        etc_aligned = cmp.align_hull_to_center_lossless(etc, gw.RICHTER_SOURCE_RADIUS, cmp.SAMPLING_RATE,
                                                        speed_of_sound=gw.RICHTER_SPEED_OF_SOUND)
        params = parameters_full(etc_aligned, distance0, gw.RICHTER_SOURCE_RADIUS,
                                 sampling_rate=cmp.SAMPLING_RATE, speed_of_sound=gw.RICHTER_SPEED_OF_SOUND,
                                 g_radius_term=2 * gw.RICHTER_SOURCE_RADIUS)
        return {p: params[p] for p in param_keys}

    per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
    arrays, result_metrics = _repeated_arrays_and_metrics(per_repeat, mean, std)
    for p in param_keys:
        ref = gw.RICHTER_FIG14_REFERENCE[p]
        jnd = gw.RICHTER_FIG14_JND[p] if p in gw.RICHTER_FIG14_JND else 0.156 * abs(ref)
        result_metrics[f"{p}_fig14_ref"] = ref
        result_metrics[f"{p}_fig14_jnd"] = jnd
        result_metrics[f"{p}_fig14_diff"] = float(arrays[f"{p}_mean"] - ref)

    storage.save_run(
        "richter_gewandhaus_reproduction", "gewandhaus",
        metrics=result_metrics, arrays=arrays,
        room=dict(max_time=max_time, source_pos=list(gw.RICHTER_SOURCE_POS),
                 source_radius=gw.RICHTER_SOURCE_RADIUS, mic_pos=list(gw.RICHTER_MIC_POS),
                 visibility_quota="26/43 (60.5%), see conversation for the candidate grid search"),
        render_params=_repeat_render_params(n_repeats, misuka_spp=spp,
                                            speed_of_sound=gw.RICHTER_SPEED_OF_SOUND,
                                            frequency_hz=gw.RICHTER_FREQUENCY_HZ),
        extra=dict(
            comparison_target="Richter Masterarbeit Fig.14 (p.33), reference/hybrid line, read off by eye",
            limitations=(
                "Source/mic position is NOT GRAP's real R1 (unrecoverable from any "
                "available source -- see pra_comparison_gewandhaus.py docstring); chosen "
                "instead by documented line-of-sight visibility quota. Materials use a "
                "targeted floor/wall split (10 of 212 'walls' triangles), not the full "
                "6 original GRAP material groups. TS remains outside JND and moves "
                "further from the reference under every absorption change tested -- an "
                "independent, unresolved discrepancy, not a sign either fix is wrong."
            ),
        ),
    )
    print(f"done: richter_gewandhaus_reproduction (n_repeats={n_repeats}, spp={spp})")
    for p in param_keys:
        diff = result_metrics[f"{p}_fig14_diff"]
        jnd = result_metrics[f"{p}_fig14_jnd"]
        status = "OK" if abs(diff) <= jnd else "outside JND"
        print(f"  {p:5s} diff={diff:8.4f}  JND={jnd:6.3f}  [{status}]")


# =====================================================================
#   Gewandhaus: misuka vs. PRA-Hybrid, GEWANDHAUS_CANONICAL only.
#   Asymmetric repeat counts (misuka's own variance needs 50 draws to
#   pin down; PRA-Hybrid's spread was already small at n=3 in the
#   appendix_hybrid_params runs, and Richter himself only ever used a
#   single hybrid reference run) -- so misuka and PRA are aggregated
#   independently via two separate _repeat_and_aggregate calls, unlike
#   every other hybrid section above where both sides share one repeat
#   loop and n_repeats.
# =====================================================================

def run_gewandhaus_canonical_hybrid_comparison(n_repeats_misuka=50, n_repeats_pra=3):
    import pra_comparison_gewandhaus as gw

    cfg = gw.GEWANDHAUS_CANONICAL
    max_time = 4.0
    spp = 8_388_608  # already-converged value, see run_richter_gewandhaus_reproduction
    source_radius = cfg["source_radius"]
    distance0 = float(np.linalg.norm(np.array(cfg["source_pos"]) - np.array(cfg["mic_pos"])))
    n_time_bins = int(round(max_time * cmp.SAMPLING_RATE))
    param_keys = ["G", "EDT", "T30", "TS", "C50", "C80"]

    scene_dict = gw.build_misuka_scene_dict_richter(max_time, cmp.SAMPLING_RATE)

    def render_misuka_one(seed):
        etc = cmp.render_misuka(scene_dict, max_time, spp=spp, seed=seed,
                                speed_of_sound=cfg["speed_of_sound"])
        etc_aligned = cmp.align_hull_to_center_lossless(etc, source_radius, cmp.SAMPLING_RATE,
                                                        speed_of_sound=cfg["speed_of_sound"])
        params = parameters_full(etc_aligned, distance0, source_radius,
                                 sampling_rate=cmp.SAMPLING_RATE, speed_of_sound=cfg["speed_of_sound"],
                                 g_radius_term=2 * source_radius)
        return {p: params[p] for p in param_keys}

    def render_pra_one(seed):
        room_hybrid = gw.build_pra_room(
            cmp.RT_HYBRID_ORDER, fs=int(cmp.BIN_FACTOR * cmp.SAMPLING_RATE),
            hist_bin_size=1.0 / cmp.SAMPLING_RATE, n_rays=3_000_000, energy_thres=cmp.PRA_ENERGY_THRES,
            time_thres=max_time)
        e_pra_hybrid = cmp.render_pra_frac_delay_stripped(room_hybrid, n_time_bins)
        params = parameters_full(e_pra_hybrid, distance0, source_radius, sampling_rate=cmp.SAMPLING_RATE,
                                 speed_of_sound=cfg["speed_of_sound"], g_radius_term=0.0)
        return {p: params[p] for p in param_keys}

    _, mean_misuka, std_misuka = _repeat_and_aggregate(render_misuka_one, n_repeats_misuka)
    _, mean_pra, std_pra = _repeat_and_aggregate(render_pra_one, n_repeats_pra)

    result_metrics = {}
    arrays = {}
    for p in param_keys:
        arrays[f"misuka_{p}_mean"] = np.asarray(mean_misuka[p])
        arrays[f"misuka_{p}_std"] = np.asarray(std_misuka[p])
        arrays[f"pra_hybrid_{p}_mean"] = np.asarray(mean_pra[p])
        arrays[f"pra_hybrid_{p}_std"] = np.asarray(std_pra[p])
        result_metrics[f"misuka_{p}_mean"] = float(mean_misuka[p])
        result_metrics[f"misuka_{p}_std"] = float(std_misuka[p])
        result_metrics[f"pra_hybrid_{p}_mean"] = float(mean_pra[p])
        result_metrics[f"pra_hybrid_{p}_std"] = float(std_pra[p])
        result_metrics[f"{p}_delta"] = float(mean_misuka[p] - mean_pra[p])
        jnd = JND_ABS[p] if p in JND_ABS else 0.05 * abs(float(mean_pra[p]))
        result_metrics[f"{p}_jnd"] = jnd
        result_metrics[f"{p}_grap"] = gw.GRAP_R1_20HZ_REFERENCE[p]

    storage.save_run(
        "gewandhaus_canonical_hybrid_comparison", "gewandhaus",
        metrics=result_metrics, arrays=arrays,
        room=dict(max_time=max_time, source_pos=list(cfg["source_pos"]),
                 source_radius=source_radius, mic_pos=list(cfg["mic_pos"]),
                 speed_of_sound=cfg["speed_of_sound"], frequency_hz=cfg["frequency_hz"]),
        render_params=dict(misuka_spp=spp, n_repeats_misuka=n_repeats_misuka,
                          n_repeats_pra=n_repeats_pra, pra_order=cmp.RT_HYBRID_ORDER,
                          pra_n_rays=3_000_000, pra_mode="hybrid"),
        extra=dict(
            alignment=(
                "Same alignment as run_appendix_hybrid_params: misuka's ETC "
                "shifted later (lossless) by the hull-vs-center offset, PRA's "
                "RIR with its frac_delay_length//2 lead-in stripped."
            ),
            asymmetric_repeats=(
                f"misuka x{n_repeats_misuka} (own variance needs the larger "
                f"sample), PRA-Hybrid x{n_repeats_pra} (spread was already "
                "small in prior appendix_hybrid_params runs; GRAP's own "
                "hybrid reference is itself a single RAVEN run)."
            ),
        ),
    )

    print(f"done: gewandhaus_canonical_hybrid_comparison "
         f"(misuka x{n_repeats_misuka}, PRA-Hybrid x{n_repeats_pra})")
    header = f"  {'param':6s} {'misuka':>18s} {'PRA-Hybrid':>18s} {'delta':>8s} {'JND':>6s} {'status':>12s} {'GRAP':>9s}"
    print(header)
    for p in param_keys:
        m_mean, m_std = result_metrics[f"misuka_{p}_mean"], result_metrics[f"misuka_{p}_std"]
        r_mean, r_std = result_metrics[f"pra_hybrid_{p}_mean"], result_metrics[f"pra_hybrid_{p}_std"]
        delta = result_metrics[f"{p}_delta"]
        jnd = result_metrics[f"{p}_jnd"]
        status = "OK" if abs(delta) <= jnd else "outside JND"
        grap = result_metrics[f"{p}_grap"]
        print(f"  {p:6s} {m_mean:8.4f}+/-{m_std:6.4f} {r_mean:8.4f}+/-{r_std:6.4f} "
             f"{delta:8.4f} {jnd:6.3f} {status:>12s} {grap:9.4f}")

    return result_metrics


# =====================================================================
#   Gewandhaus: misuka vs. GRAP's own RAVEN results for R1 at 20 Hz
#   (RNo01_R1_Parameters.csv + RNo01_R1_Histogram.csv), canonical config.
# =====================================================================

def _bin_1ms(etc, sampling_rate, n_bins):
    idx = (np.arange(len(etc)) / sampling_rate * 1e3).astype(int)
    keep = idx < n_bins
    return np.bincount(idx[keep], weights=etc[keep], minlength=n_bins)


def run_gewandhaus_grap_reference(n_repeats=3, spp=8_388_608, section="gewandhaus_grap_reference"):
    import pra_comparison_gewandhaus as gw

    cfg = gw.GEWANDHAUS_CANONICAL
    c, r = cfg["speed_of_sound"], cfg["source_radius"]
    max_time = 4.0
    distance0 = float(np.linalg.norm(np.subtract(cfg["source_pos"], cfg["mic_pos"])))
    ref = gw.GRAP_R1_20HZ_REFERENCE
    param_keys = ["G", "EDT", "T30", "TS", "C50", "C80"]
    grap_t, grap_e = gw.grap_r1_histogram(cfg["frequency_hz"])
    n_bins = int(round(max_time * 1e3))
    grap_e = grap_e[:n_bins]
    scene_dict = gw.build_misuka_scene_dict_richter(max_time, cmp.SAMPLING_RATE)

    def render_one(seed):
        etc = cmp.render_misuka(scene_dict, max_time, spp=spp, seed=seed, speed_of_sound=c)
        etc = cmp.align_hull_to_center_lossless(etc, r, cmp.SAMPLING_RATE, speed_of_sound=c)
        params = parameters_full(etc, distance0, r, sampling_rate=cmp.SAMPLING_RATE,
                                 speed_of_sound=c, g_radius_term=2 * r)
        # Absolute scale of GRAP's histogram: direct sound = 1/d^2. Same
        # direct-sound window as parameters_full's G, so the scaled curve's
        # own G equals params["G"].
        end_bin = int(round((distance0 + 2 * r) / c * cmp.SAMPLING_RATE))
        etc_1ms = _bin_1ms(etc, cmp.SAMPLING_RATE, n_bins) / distance0 ** 2 / etc[:end_bin + 1].sum()
        out = {p: params[p] for p in param_keys}
        out["etc_1ms"] = etc_1ms
        return out

    per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
    arrays, metrics = _repeated_arrays_and_metrics(per_repeat, mean, std)

    # Control: GRAP's own histogram through the same parameters_full must
    # reproduce RNo01_R1_Parameters.csv (up to its 1 ms resolution).
    grap_self = parameters_full(gw.grap_r1_histogram(cfg["frequency_hz"])[1], distance0, r,
                                sampling_rate=1000.0, speed_of_sound=c, g_radius_term=0.0,
                                expected_start_s=gw.GRAP_R1_DIRECT_DELAY_S)
    for p in param_keys:
        metrics[f"{p}_grap_selfcheck"] = grap_self[p]

    shift = int(np.argmax(grap_e > 0)) - int(np.argmax(arrays["etc_1ms_mean"] > 0))
    arrays["grap_etc_1ms"] = grap_e
    arrays["time_1ms_s"] = np.arange(n_bins) * 1e-3
    metrics["direct_bin_shift_1ms"] = shift  # applied to misuka in the plot if nonzero
    for p in param_keys:
        jnd = JND_ABS[p] if p in JND_ABS else 0.05 * abs(ref[p])
        metrics[f"{p}_grap"] = ref[p]
        metrics[f"{p}_jnd"] = jnd
        metrics[f"{p}_delta"] = float(metrics[f"{p}_mean"] - ref[p])

    storage.save_run(
        section, "gewandhaus", metrics=metrics, arrays=arrays,
        room=dict(max_time=max_time, source_pos=list(cfg["source_pos"]), mic_pos=list(cfg["mic_pos"]),
                 source_radius=r, speed_of_sound=c, frequency_hz=cfg["frequency_hz"],
                 materials={k: list(v) for k, v in cfg["materials"].items()}),
        render_params=_repeat_render_params(n_repeats, misuka_spp=spp, speed_of_sound=c,
                                            frequency_hz=cfg["frequency_hz"]),
        extra=dict(reference="GRAP RNo01_R1_Parameters.csv / _Histogram.csv, 20 Hz (RAVEN hybrid, "
                             "ISM order 3 + 200k rays); C50 from D50; T30 vs T30"),
    )
    print(f"done: {section} (misuka x{n_repeats}, spp={spp}); direct-sound 1ms-bin shift = {shift}")
    print(f"  {'param':5s} {'misuka':>18s} {'GRAP':>8s} {'GRAP-hist':>9s} {'delta':>8s} {'JND':>6s}  status")
    for p in param_keys:
        d, j = metrics[f"{p}_delta"], metrics[f"{p}_jnd"]
        print(f"  {p:5s} {metrics[f'{p}_mean']:8.4f}+/-{metrics[f'{p}_std']:6.4f} {ref[p]:8.4f} "
             f"{metrics[f'{p}_grap_selfcheck']:9.4f} {d:+8.4f} {j:6.3f}  {'OK' if abs(d) <= j else 'outside JND'}")
    return metrics


# =====================================================================
#   Gewandhaus at 125 Hz: expected results for the CHORAS comparison.
#   Mirrors what a CHORAS user can set: one scattering value for the whole
#   room (area-weighted GRAP mean), paths terminated after 100 dB energy
#   loss (Richter), response length = GRAP's RAVEN reference (4.65 s).
#   Otherwise GEWANDHAUS_CANONICAL (geometry, S1/R1, r=0.1 m, c=343.99).
# =====================================================================

CHORAS_125HZ_SCATTERING = 0.0195  # area-weighted GRAP scattering at 125 Hz
CHORAS_MAX_ENERGY_LOSS_DB = 100.0
CHORAS_MAX_TIME = 4.65
CHORAS_PRA_N_RAYS = 3_000_000
# CHORAS's PRA integration always applies air attenuation at fixed 20 C /
# 50 % RH (pyfar, reference pressure); misuka gets the same atmosphere.
CHORAS_AIR = dict(temperature=20.0, relative_humidity=0.5, atmospheric_pressure=101325.0)
CHORAS_MISUKA_MEDIUM = dict(CHORAS_AIR, apply_attenuation=True)
# PRA's RT histogram bin width for RIR synthesis: pyroomacoustics' default.
# One sampling period (~23 us) under-weights the RT part badly (verified:
# EDT 0.71 s instead of 1.66 s at 125 Hz) -- only use that for raw-histogram
# comparisons.
PRA_RIR_HIST_BIN_S = 0.004
# PRA in the 125 Hz runs at misuka's own rate: with 4 ms histogram bins the
# BIN_FACTOR > 1 constraint (one-sample histogram bins) does not apply.
PRA_125HZ_BIN_FACTOR = 1


def _extend_125hz_misuka(section, render_misuka_one, extra_spp, param_keys):
    run = storage.load_run(section, "gewandhaus")
    meta, arrays = run["metadata"], dict(run["arrays"])
    metrics, rp = dict(meta["metrics"]), dict(meta["render_params"])
    n_repeats = int(rp["n_repeats_misuka"])
    assert not set(extra_spp) & set(int(s) for s in arrays["spp"]), "ray counts already in the run"
    per, mean, std = _repeat_and_aggregate(lambda seed: render_misuka_one(seed, extra_spp), n_repeats)
    for p in param_keys:
        key = f"misuka_{p}"
        stacked = np.concatenate([arrays[f"{key}_per_repeat"], per[key]], axis=1)
        arrays[f"{key}_per_repeat"] = stacked
        arrays[key] = arrays[f"{key}_mean"] = stacked.mean(axis=0)
        arrays[f"{key}_std"] = stacked.std(axis=0)
        metrics[f"{p}_misuka_converged"] = float(arrays[key][-1])
        metrics[f"{p}_delta_misuka_grap"] = float(arrays[key][-1] - metrics[f"{p}_grap"])
    arrays["etc_1ms"] = arrays["etc_1ms_mean"] = mean["etc_1ms"]
    arrays["etc_1ms_std"] = std["etc_1ms"]
    arrays["etc_1ms_per_repeat"] = per["etc_1ms"]
    arrays["spp"] = np.concatenate([arrays["spp"], extra_spp])
    rp["misuka_spp_values"] = [int(s) for s in arrays["spp"]]
    rp["extended_spp"] = list(rp.get("extended_spp", [])) + [int(s) for s in extra_spp]
    extra = dict(meta.get("extra", {}), extension=(
        f"ray counts {extra_spp} rendered later with the same seeds and settings and appended; "
        "PRA and GRAP data unchanged"))
    storage.save_run(section, "gewandhaus", metrics=metrics, arrays=arrays, room=meta["room"],
                     render_params=rp, extra=extra)
    print(f"extended {section} by spp {extra_spp}")
    for p in param_keys:
        print(f"  {p:4s} " + "  ".join(f"2^{int(np.log2(s))}: {m:.4f}+/-{sd:.4f}"
              for s, m, sd in zip(arrays["spp"][-4:], arrays[f"misuka_{p}"][-4:], arrays[f"misuka_{p}_std"][-4:])))
    return metrics


def run_gewandhaus_125hz_expected(n_repeats_misuka=50, n_repeats_pra=3, spp_values=None,
                                  section="gewandhaus_125hz_expected", extend_spp=None,
                                  scattering=CHORAS_125HZ_SCATTERING, pra_receiver_radius=0.5,
                                  pra_n_rays=None, pra_fix=False, pra_only=False, pra_merge=False,
                                  pra_extend_to=None):
    """extend_spp: render only misuka at these additional ray counts (same seeds
    and settings) and append them to the stored run; PRA and GRAP data are kept.
    pra_merge: render only PRA again and replace it in the stored run; misuka
    and GRAP data are kept. pra_extend_to: add PRA runs with the next seeds to
    the stored ones until there are this many; everything else is kept.
    scattering: one value for all surfaces (CHORAS-like), or None for GRAP's
    per-material values (see run_gewandhaus_125hz_best)."""
    pra_n_rays = pra_n_rays or CHORAS_PRA_N_RAYS
    import pra_comparison_gewandhaus as gw

    cfg = gw.GEWANDHAUS_CANONICAL
    c, r, f = cfg["speed_of_sound"], cfg["source_radius"], 125.0
    max_time = CHORAS_MAX_TIME
    spp_values = spp_values or cmp.GEWANDHAUS_SPP_VALUES
    distance0 = float(np.linalg.norm(np.subtract(cfg["source_pos"], cfg["mic_pos"])))
    n_time_bins = int(round(max_time * cmp.SAMPLING_RATE))
    n_bins_1ms = int(round(max_time * 1e3))
    param_keys = ["G", "EDT", "T30", "TS", "C50", "C80", "D50"]

    def with_d50(p):
        p = dict(p)
        p["D50"] = float(100.0 / (1.0 + 10 ** (-p["C50"] / 10.0)))  # D50 in %, from C50
        return {k: p[k] for k in param_keys}

    materials_125 = gw.materials_at([f])
    absorption = {name: a for name, (a, _) in materials_125.items()}
    scattering_of = {name: (s if scattering is None else scattering) for name, (_, s) in materials_125.items()}
    import pyfar as pf
    air_m_125 = float(np.ravel(pf.constants.air_attenuation(
        CHORAS_AIR["temperature"], [f], relative_humidity=CHORAS_AIR["relative_humidity"],
        atmospheric_pressure=CHORAS_AIR["atmospheric_pressure"])[1].freq)[0])
    scene_dict = gw.build_misuka_scene_dict_richter(max_time, cmp.SAMPLING_RATE, frequencies=[f])
    for name in absorption:
        scene_dict[name]["bsdf"]["scattering"]["value"] = scattering_of[name]

    def render_misuka_one(seed, spp_list=None):
        out = {p: [] for p in param_keys}
        for spp in spp_list or spp_values:
            etc = cmp.render_misuka(scene_dict, max_time, max_energy_loss=CHORAS_MAX_ENERGY_LOSS_DB,
                                    spp=spp, seed=seed, speed_of_sound=c,
                                    acoustic_medium=CHORAS_MISUKA_MEDIUM)
            etc = cmp.align_hull_to_center_lossless(etc, r, cmp.SAMPLING_RATE, speed_of_sound=c)
            params = with_d50(parameters_full(etc, distance0, r, sampling_rate=cmp.SAMPLING_RATE,
                                              speed_of_sound=c, g_radius_term=2 * r))
            for p in param_keys:
                out[p].append(params[p])
        res = {f"misuka_{p}": np.array(v) for p, v in out.items()}
        # ETC at the highest ray count, 1 ms bins, direct sound scaled to 1/d^2 like GRAP's histogram
        end_bin = int(round((distance0 + 2 * r) / c * cmp.SAMPLING_RATE))
        res["etc_1ms"] = _bin_1ms(etc, cmp.SAMPLING_RATE, n_bins_1ms) / distance0 ** 2 / etc[:end_bin + 1].sum()
        return res

    def render_pra_one(seed):
        room = gw.build_pra_room(
            cmp.RT_HYBRID_ORDER, fs=int(PRA_125HZ_BIN_FACTOR * cmp.SAMPLING_RATE),
            hist_bin_size=PRA_RIR_HIST_BIN_S, n_rays=pra_n_rays,
            energy_thres=10 ** (-CHORAS_MAX_ENERGY_LOSS_DB / 10), time_thres=max_time,
            materials={n: (a, scattering_of[n]) for n, a in absorption.items()},
            require_canonical=False, quiet=True, receiver_radius=pra_receiver_radius)
        # CHORAS's PRA always attenuates with its fixed 20 C / 50 % RH air;
        # flat materials give a single PRA band, which gets the 125 Hz value.
        room.air_absorption = np.full(len(room.walls[0].absorption), air_m_125)
        if pra_fix:
            cmp.drop_double_counted_ism_order(room)
        e = cmp.render_pra_frac_delay_stripped(room, n_time_bins, factor=PRA_125HZ_BIN_FACTOR)
        params = with_d50(parameters_full(e, distance0, r, sampling_rate=cmp.SAMPLING_RATE,
                                          speed_of_sound=c, g_radius_term=0.0))
        res = {f"pra_{p}": params[p] for p in param_keys}
        res["pra_etc_1ms"] = _bin_1ms(e, cmp.SAMPLING_RATE, n_bins_1ms)
        return res

    if extend_spp:
        return _extend_125hz_misuka(section, render_misuka_one, list(extend_spp), param_keys)

    if pra_merge or pra_extend_to:
        run = storage.load_run(section, "gewandhaus")
        meta, arrays, metrics = run["metadata"], dict(run["arrays"]), dict(run["metadata"]["metrics"])
        if pra_extend_to:
            n_done = arrays["pra_G_per_repeat"].shape[0]
            new = []
            for i in range(n_done, pra_extend_to):
                seed = _seed_for_repeat(i)
                pra.random.seed(seed)
                new.append(render_pra_one(seed))
            per_p = {k: np.concatenate([arrays[f"{k}_per_repeat"], np.stack([np.asarray(r[k]) for r in new])])
                     for k in new[0]}
            mean_p = {k: v.mean(axis=0) for k, v in per_p.items()}
            std_p = {k: v.std(axis=0) for k, v in per_p.items()}
            n_repeats_pra = pra_extend_to
        else:
            per_p, mean_p, std_p = _repeat_and_aggregate(render_pra_one, n_repeats_pra)
        arrays_p, metrics_p = _repeated_arrays_and_metrics(per_p, mean_p, std_p)
        arrays.update(arrays_p); metrics.update(metrics_p)
        for p in param_keys:
            if f"{p}_grap" in metrics:
                metrics[f"{p}_delta_pra_grap"] = float(metrics[f"pra_{p}_mean"] - metrics[f"{p}_grap"])
        metrics["reference_params"] = {p: float(metrics[f"pra_{p}_mean"]) for p in param_keys}
        rp = dict(meta["render_params"], n_repeats_pra=n_repeats_pra,
                  pra_sampling_rate=PRA_125HZ_BIN_FACTOR * cmp.SAMPLING_RATE)
        storage.save_run(section, "gewandhaus", metrics=metrics, arrays=arrays, room=meta["room"],
                         render_params=rp, extra=dict(meta.get("extra", {}), pra_rerun=(
                             "PRA rendered again at misuka's sampling rate and replaced; misuka and GRAP unchanged")))
        print(f"done: {section} (PRA x{n_repeats_pra} merged)")
        for p in param_keys:
            print(f"  {p:5s} {metrics[f'pra_{p}_mean']:10.4f}+/-{metrics[f'pra_{p}_std']:7.4f}")
        return metrics

    if pra_only:
        per_p, mean_p, std_p = _repeat_and_aggregate(render_pra_one, n_repeats_pra)
        arrays, metrics = _repeated_arrays_and_metrics(per_p, mean_p, std_p)
        metrics["reference_params"] = {p: float(metrics[f"pra_{p}_mean"]) for p in param_keys}
        storage.save_run(
            section, "gewandhaus", metrics=metrics, arrays=arrays,
            room=dict(max_time=max_time, source_pos=list(cfg["source_pos"]), mic_pos=list(cfg["mic_pos"]),
                      source_radius=r, speed_of_sound=c, frequency_hz=f,
                      absorption=absorption, scattering=scattering_of),
            render_params=dict(n_repeats_pra=n_repeats_pra, pra_order=cmp.RT_HYBRID_ORDER,
                               pra_n_rays=pra_n_rays, pra_receiver_radius_m=pra_receiver_radius,
                               pra_energy_thres_db=-CHORAS_MAX_ENERGY_LOSS_DB, sampling_rate=cmp.SAMPLING_RATE,
                               pra_hist_bin_s=PRA_RIR_HIST_BIN_S, pra_fix_double_counted_ism_order=pra_fix,
                               air=dict(CHORAS_AIR, pra_energy_attenuation_1_per_m=air_m_125)),
            extra=dict(purpose="PRA only" + (", with drop_double_counted_ism_order" if pra_fix else "")),
        )
        print(f"done: {section} (PRA x{n_repeats_pra}, fix={pra_fix})")
        for p in param_keys:
            print(f"  {p:5s} {metrics[f'pra_{p}_mean']:10.4f}+/-{metrics[f'pra_{p}_std']:7.4f}")
        return metrics

    per_m, mean_m, std_m = _repeat_and_aggregate(render_misuka_one, n_repeats_misuka)
    arrays, metrics = _repeated_arrays_and_metrics(per_m, mean_m, std_m)
    per_p, mean_p, std_p = _repeat_and_aggregate(render_pra_one, n_repeats_pra)
    arrays_p, metrics_p = _repeated_arrays_and_metrics(per_p, mean_p, std_p)
    arrays.update(arrays_p); metrics.update(metrics_p)

    # GRAP's RAVEN reference at 125 Hz (third-octave row) and its histogram
    ref = dict(EDT=1.7277, T30=2.1554, C80=3.2739, D50=53.96, G=7.1529, TS=0.0872)
    ref["C50"] = float(10 * np.log10(0.5396 / (1 - 0.5396)))
    grap_t, grap_e = gw.grap_r1_histogram(f)
    grap_self = with_d50(parameters_full(grap_e, distance0, r, sampling_rate=1000.0, speed_of_sound=c,
                                         g_radius_term=0.0, expected_start_s=gw.GRAP_R1_DIRECT_DELAY_S))
    arrays["grap_etc_1ms"] = grap_e[:n_bins_1ms]
    arrays["time_1ms_s"] = np.arange(n_bins_1ms) * 1e-3
    arrays["spp"] = np.array(spp_values)
    for p in param_keys:
        jnd = JND_ABS.get(p, 5.0 if p == "D50" else None) or 0.05 * abs(ref[p])
        if p == "C50":  # derived from the D50 JND, asymmetric (jnd_band)
            lo, hi = jnd_band("C50", ref[p])
            metrics["C50_jnd_lo"], metrics["C50_jnd_hi"] = lo, hi
            jnd = (hi - lo) / 2
        metrics[f"{p}_grap"] = ref[p]
        metrics[f"{p}_grap_selfcheck"] = grap_self[p]
        metrics[f"{p}_jnd"] = jnd
        metrics[f"{p}_misuka_converged"] = float(arrays[f"misuka_{p}_mean"][-1])
        metrics[f"{p}_delta_misuka_grap"] = float(arrays[f"misuka_{p}_mean"][-1] - ref[p])
        metrics[f"{p}_delta_pra_grap"] = float(metrics[f"pra_{p}_mean"] - ref[p])
    # reference_params + misuka_{p}: lets plot_hybrid_jnd draw this run as well
    metrics["reference_params"] = {p: float(metrics[f"pra_{p}_mean"]) for p in param_keys}

    storage.save_run(
        section, "gewandhaus", metrics=metrics, arrays=arrays,
        room=dict(max_time=max_time, source_pos=list(cfg["source_pos"]), mic_pos=list(cfg["mic_pos"]),
                 source_radius=r, speed_of_sound=c, frequency_hz=f,
                 absorption=absorption, scattering=scattering_of),
        render_params=dict(misuka_spp_values=list(spp_values), n_repeats_misuka=n_repeats_misuka,
                          n_repeats_pra=n_repeats_pra, seeds_misuka=[_seed_for_repeat(i) for i in range(n_repeats_misuka)],
                          max_energy_loss_db=CHORAS_MAX_ENERGY_LOSS_DB, pra_order=cmp.RT_HYBRID_ORDER,
                          pra_n_rays=pra_n_rays, pra_receiver_radius_m=pra_receiver_radius, pra_energy_thres_db=-CHORAS_MAX_ENERGY_LOSS_DB,
                          sampling_rate=cmp.SAMPLING_RATE, pra_hist_bin_s=PRA_RIR_HIST_BIN_S,
                          air=dict(CHORAS_AIR, pra_energy_attenuation_1_per_m=air_m_125,
                                   misuka="inline ISO 9613-1", pra="pyfar ISO 9613-1, applied in RIR synthesis")),
        extra=dict(purpose=("expected results for the CHORAS comparison (CHORAS-like settings)" if scattering is not None else "local verification with the best settings for the comparison with RAVEN"),
                   reference="GRAP RNo01_R1_Parameters.csv, 125 Hz third-octave row (RAVEN hybrid)"),
    )
    print(f"done: {section} (misuka x{n_repeats_misuka} over {len(spp_values)} spp values, PRA x{n_repeats_pra})")
    print(f"  {'param':5s} {'misuka@max spp':>20s} {'PRA hybrid':>20s} {'GRAP':>8s} {'GRAP-hist':>9s} {'JND':>6s}")
    for p in param_keys:
        print(f"  {p:5s} {arrays[f'misuka_{p}_mean'][-1]:10.4f}+/-{arrays[f'misuka_{p}_std'][-1]:7.4f} "
              f"{metrics[f'pra_{p}_mean']:10.4f}+/-{metrics[f'pra_{p}_std']:7.4f} {ref[p]:8.4f} "
              f"{grap_self[p]:9.4f} {metrics[f'{p}_jnd']:6.3f}")
    return metrics


def run_gewandhaus_125hz_best(n_repeats_misuka=50, n_repeats_pra=3):
    """Local verification with the best settings for the comparison with RAVEN,
    not limited to what CHORAS exposes: GRAP's per-material scattering, PRA
    with a 0.1 m receiver (misuka's and Richter's detector size; 3e6 rays, about
    7x Richter's estimate of 414,582), misuka from 2^10 to 2^25 rays. Air at
    RAVEN's standard conditions (20 C / 50 % RH), 100 dB, 4.65 s as before."""
    return run_gewandhaus_125hz_expected(
        n_repeats_misuka, n_repeats_pra, spp_values=[2 ** k for k in range(10, 26)],
        section="gewandhaus_125hz_best", scattering=None, pra_receiver_radius=0.1,
        pra_n_rays=3_000_000)


def run_gewandhaus_125hz_best_prafix(n_repeats_pra=3):
    """PRA part of run_gewandhaus_125hz_best again, same settings and seeds,
    with pyroomacoustics' double-counted reflection order removed
    (cmp.drop_double_counted_ism_order). misuka is not re-rendered."""
    return run_gewandhaus_125hz_expected(
        n_repeats_pra=n_repeats_pra, section="gewandhaus_125hz_best_prafix", scattering=None,
        pra_receiver_radius=0.1, pra_n_rays=3_000_000, pra_fix=True, pra_only=True)


# =====================================================================
#                                run_all
# =====================================================================

def run_all(n_repeats=N_REPEATS):
    run_main_comparison(n_repeats)
    run_atmosphere(n_repeats)
    run_atmosphere_methods(n_repeats)
    run_ray_count_convergence(n_repeats)
    run_appendix_ism_peak(n_repeats)
    run_appendix_hybrid_params(n_repeats)
    run_appendix_rt_histogram(n_repeats)


def run_edc_runtime_and_appendix(n_repeats=N_REPEATS):
    """Only the sections needed for: the notebook's "### Runtime summary"
    (run_atmosphere -- also backs the on/off ETC/EDC plots right above it,
    same section, one render per scenario covers both), "### Post-processing
    vs. inline attenuation: relative and absolute EDC difference"
    (run_atmosphere_methods), and the whole "## Appendix: three additional
    misuka vs. PRA comparisons" (the three run_appendix_* sections).
    Skips run_main_comparison and run_ray_count_convergence, which back
    unrelated notebook sections and are the slowest part of run_all.
    """
    run_atmosphere(n_repeats)
    run_atmosphere_methods(n_repeats)
    run_appendix_ism_peak(n_repeats)
    run_appendix_hybrid_params(n_repeats)
    run_appendix_rt_histogram(n_repeats)


if __name__ == "__main__":
    import mitsuba as mi
    mi.set_variant("cuda_acoustic", "llvm_ad_acoustic")
    run_edc_runtime_and_appendix(n_repeats=50)
    #run_all()
