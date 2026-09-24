"""How much does the post-processing-vs-inline atmospheric attenuation
discrepancy (see test_pyroomacoustics_comparison.compare_atmosphere_methods
and its module docstring) depend on time-bin width (sampling_rate) and on
rendered frequency?

Theory (see pra_comparison_metrics/the notebook's own "Post-processing vs.
inline attenuation" section): post-processing attenuates each time bin's
*already-summed* energy using that bin's time-implied distance
(`t_bin * c`), while inline attenuation applies the correct, per-path
geometric distance before any binning happens. The two can only differ
when a bin actually spans more than one true distance's worth of
attenuation -- i.e. the discrepancy should grow with bin width (lower
sampling_rate) and with frequency (ISO 9613-1 attenuation grows sharply
with frequency, so the same distance spread maps to a larger attenuation
spread). This script tests both dependencies directly on one scenario
(shoebox_diffuse_200m3's geometry/materials, reused from
test_pyroomacoustics_comparison.py -- not re-defined here) rather than
just asserting it.

Metric: relative_difference_percent(e_post, e_inline) on the raw ETC
(*not* the Schroeder EDC pra_comparison_runner._multi_frequency_post_vs_
inline uses -- this script's brief specifically asked for the relative-ETC
version), summarized per (sampling_rate, frequency) as the mean and max
absolute percentage over the whole render, aggregated (mean +/- std) over
N_REPEATS independent seeds via pra_comparison_runner's own repeat
machinery.

Run directly to (re)generate the stored sweep; see plot_post_vs_inline_
attenuation_sweep in pra_comparison_plots.py for the corresponding figure.
"""
import numpy as np
import mitsuba as mi
mi.set_variant("cuda_acoustic", "llvm_ad_acoustic")

import test_pyroomacoustics_comparison as cmp
import pra_comparison_storage as storage
from pra_comparison_runner import _repeat_and_aggregate, _seed_for_repeat

SECTION = "attenuation_post_vs_inline"
SCENARIO_NAME = "shoebox_diffuse_200m3"

# Brackets the suite's own default (44100 Hz) on both sides, log-ish spaced.
SAMPLING_RATES_HZ = [2000.0, 4000.0, 8000.0, 16000.0, 22050.0, 44100.0, 88200.0, 176400.0]
# Same set pra_comparison_runner.HIGHLIGHT_FREQUENCIES_HZ already uses for the
# equivalent EDC-based, fixed-sampling-rate frequency sweep -- reused as-is.
FREQUENCIES_HZ = [100.0, 1000.0, 4000.0, 8000.0, 14000.0, 20000.0]
N_REPEATS = 10


def _shoebox_diffuse_200m3_geometry():
    """Geometry/materials of scenario_shoebox_diffuse() (test_
    pyroomacoustics_comparison.py), extracted rather than called directly:
    that function bakes in the module-level SAMPLING_RATE via
    misuka_scene_dict's default, and this sweep needs to build the same
    scene at several different sampling rates instead.
    """
    dim = (8.0, 5.0, 5.0)  # 200 m^3
    src, mic = (2.0, 2.5, 1.2), (5.0, 2.5, 1.2)
    absorption, scattering = 0.2, 0.9
    source_radius = 0.2
    max_time = 0.4
    return dim, src, mic, absorption, scattering, source_radius, max_time


def run_sweep(n_repeats=N_REPEATS):
    dim, src, mic, absorption, scattering, source_radius, max_time = _shoebox_diffuse_200m3_geometry()
    bsdf = cmp.acoustic_bsdf(absorption, scattering)
    shapes = cmp.misuka_box_shapes(dim, bsdf)
    freqs_str = ",".join(str(f) for f in FREQUENCIES_HZ)
    # atmo_speed_of_sound() returns the variant's native (drjit) Float under
    # the JIT variants this suite renders with -- not JSON-serializable as
    # a render_params value, see cmp._scalar's own docstring for why.
    c_atmo = cmp._scalar(cmp.atmo_speed_of_sound())

    n_fs, n_freq = len(SAMPLING_RATES_HZ), len(FREQUENCIES_HZ)
    mean_abs_pct = np.zeros((n_fs, n_freq))
    mean_abs_pct_std = np.zeros((n_fs, n_freq))
    max_abs_pct = np.zeros((n_fs, n_freq))
    max_abs_pct_std = np.zeros((n_fs, n_freq))
    example_diff_pct = {}  # one (mean-over-repeats) full diff curve per sampling rate, for illustration

    for i, fs in enumerate(SAMPLING_RATES_HZ):
        scene_dict = cmp.misuka_scene_dict(shapes, src, mic, source_radius, max_time,
                                           frequencies=freqs_str, sampling_rate=fs)

        def render_one(seed, scene_dict=scene_dict, fs=fs):
            e_raw = cmp.render_misuka(scene_dict, max_time, spp=cmp.MISUKA_SPP, seed=seed,
                                      atmosphere=True, apply_attenuation=False)
            e_inline = cmp.render_misuka(scene_dict, max_time, spp=cmp.MISUKA_SPP, seed=seed,
                                         atmosphere=True, apply_attenuation=True)
            e_post = cmp.apply_pure_tone_attenuation(
                e_raw, sampling_rate=fs, speed_of_sound_ms=c_atmo,
                temperature=cmp.ATMO_TEMPERATURE, frequencies=FREQUENCIES_HZ,
                relative_humidity=cmp.ATMO_RELATIVE_HUMIDITY, atmospheric_pressure=cmp.ATMO_PRESSURE)
            diff_pct = cmp.relative_difference_percent(e_post, e_inline)  # (n_time_bins, n_freq)
            diff_pct_abs = np.abs(diff_pct)
            return dict(mean_abs_pct=diff_pct_abs.mean(axis=0), max_abs_pct=diff_pct_abs.max(axis=0),
                       diff_pct=diff_pct)

        per_repeat, mean, std = _repeat_and_aggregate(render_one, n_repeats)
        mean_abs_pct[i] = mean["mean_abs_pct"]
        mean_abs_pct_std[i] = std["mean_abs_pct"]
        max_abs_pct[i] = mean["max_abs_pct"]
        max_abs_pct_std[i] = std["max_abs_pct"]
        example_diff_pct[f"diff_pct_curve_fs{int(fs)}"] = mean["diff_pct"]
        print(f"done: sampling_rate={fs:>9.0f} Hz  "
              f"mean|delta|%% (100Hz..20kHz) = {np.round(mean['mean_abs_pct'], 5)}")

    arrays = dict(
        sampling_rates_hz=np.array(SAMPLING_RATES_HZ),
        frequencies_hz=np.array(FREQUENCIES_HZ),
        mean_abs_pct=mean_abs_pct, mean_abs_pct_std=mean_abs_pct_std,
        max_abs_pct=max_abs_pct, max_abs_pct_std=max_abs_pct_std,
        **example_diff_pct,
    )
    metrics = dict(
        n_repeats=n_repeats,
        overall_max_abs_pct=float(max_abs_pct.max()),
        mean_abs_pct_at_44100hz=[float(x) for x in mean_abs_pct[SAMPLING_RATES_HZ.index(44100.0)]],
    )
    storage.save_run(
        SECTION, SCENARIO_NAME,
        metrics=metrics, arrays=arrays,
        room=dict(dim=list(dim), source_pos=list(src), mic_pos=list(mic),
                  source_radius=source_radius, absorption=absorption, scattering=scattering,
                  max_time=max_time, volume_m3=float(np.prod(dim))),
        render_params=dict(
            misuka_spp=cmp.MISUKA_SPP, n_repeats=n_repeats,
            seeds=[_seed_for_repeat(i) for i in range(n_repeats)],
            sampling_rates_hz=SAMPLING_RATES_HZ, frequencies_hz=FREQUENCIES_HZ,
            speed_of_sound_atmo=c_atmo, atmo_temperature=cmp.ATMO_TEMPERATURE,
            atmo_relative_humidity=cmp.ATMO_RELATIVE_HUMIDITY, atmo_pressure=cmp.ATMO_PRESSURE,
        ),
    )
    print(f"saved: {SECTION} / {SCENARIO_NAME}")
    return arrays


if __name__ == "__main__":
    run_sweep()
