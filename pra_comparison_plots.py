"""Plotting functions for the misuka-vs-pyroomacoustics comparison suite.

Every function here takes already-loaded data (a run dict from
pra_comparison_storage.load_run(), or a list of such runs/their metadata)
and draws a figure or builds a styled table -- none of them render, simulate,
or compute a derived metric themselves. That work happens once, in
pra_comparison_runner.py, and is persisted via pra_comparison_storage; the
notebook (or any other caller) just loads and calls these.

Relocated from pyroomacoustics_comparison.ipynb; the actual matplotlib/pandas
styling calls are unchanged from the notebook version, only the inputs
changed (loaded arrays/metadata dicts instead of in-notebook globals like
`raw[(name, mode)]`).
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyfar as pf

import test_pyroomacoustics_comparison as cmp
from pra_comparison_metrics import pyrato_edc, JND_ABS, JND_REL

# Suite-wide plot style: cmr10 (LaTeX's default serif) + vector (SVG) output
# for every fig.savefig() call, so figures drop straight into a LaTeX report
# without a font/format mismatch.
font = {'family': 'cmr10', 'size': 11}
plt.rc('font', **font)
plt.rc('mathtext', fontset='cm')
plt.rc('axes', unicode_minus=False)  # cmr10 has no unicode minus glyph
plt.rc('savefig', format='svg')

# Suite-wide color coding: misuka renders in green, pyroomacoustics/PRA in
# orange. `_ALT` shades distinguish a second same-origin series within one
# plot (e.g. "no atmosphere" vs. "inline atmosphere", both misuka).
MISUKA_COLOR = "#2ca02c"
MISUKA_COLOR_ALT = "#1b7837"
MISUKA_COLOR_ALT2 = "#78c679"
PRA_COLOR = "#ff7f0e"
PRA_COLOR_ALT = "#fdae61"

# Sequential map in the suite's own two colours, for the heat maps: PRA
# orange at the low end, misuka green at the high end. Not perceptually
# uniform the way viridis is -- the two hues meet in an olive midtone -- so
# it is offered as an alternative rather than as the default.
PRA_TO_MISUKA_CMAP = LinearSegmentedColormap.from_list(
    "pra_to_misuka", [PRA_COLOR, MISUKA_COLOR])

# Three-point version of the same idea, via a light midtone. Interpolating
# orange straight to green passes through a dark olive whose lightness
# barely changes, which flattens the contrast in the middle of the range;
# routing through a pale yellow restores it. The cost is that a light centre
# reads as a diverging map, so it suits data with a meaningful midpoint
# better than a pure magnitude.
PRA_TO_MISUKA_CMAP_3 = LinearSegmentedColormap.from_list(
    "pra_to_misuka_3", [PRA_COLOR, "#fff7bc", MISUKA_COLOR])

# Shades of green for plots with one curve per frequency (the bin-width
# blocks 1b/1c): all-green keeps every curve visually in the "misuka" family,
# with lightness carrying the frequency so the curves stay one colour family
# while still being told apart. A quantity that differs only by *marker*
# (e.g. avg vs. max) reuses this same colour for both markers at a given
# frequency, which is the point of keeping it separate from MISUKA_COLOR_ALT.
FREQUENCY_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "frequency_greens", ["#c7e9c0", "#00441b"])


def frequency_colors(frequencies):
    """One colour per frequency from FREQUENCY_GREEN_CMAP, lowest frequency
    lightest. Avoids the map's near-white and near-black extremes, which are
    hard to read against a white background or a black legend marker edge.
    """
    n = len(frequencies)
    positions = [0.5] if n == 1 else np.linspace(0.12, 0.92, n)
    return [FREQUENCY_GREEN_CMAP(p) for p in positions]


# =====================================================================
#                   Suite-wide settings overview
# =====================================================================
#
# Everything shown here is read from already-saved run metadata (via
# pra_comparison_storage.list_runs(), passed in by the caller -- this module
# never imports pra_comparison_storage itself, matching every other
# table/plot function below) -- never hardcoded. Re-render with a changed
# backend setting (pra_comparison_runner.py), re-run storage.list_runs() and
# this function in the notebook, and the numbers below update automatically:
# there is nothing in this function that could go stale on its own.

def _distinct_preserve_order(values):
    seen, out = [], []
    for v in values:
        key = tuple(v) if isinstance(v, list) else v
        if key not in seen:
            seen.append(key)
            out.append(v)
    return out


def suite_overview(runs):
    """Consolidated overview of every setting relevant across ALL stored
    comparison runs: misuka variant, sampling rate, speed of sound, repeat
    count (X), ray counts (single-render spp and sweep spp range, PRA ray
    count), and a per-scenario room-geometry table. Flags (rather than
    silently picking one value for) any setting that turns out to differ
    across the stored runs -- if that happens, the runs were produced under
    inconsistent settings and probably shouldn't be compared directly.

    `runs`: list of run metadata dicts, e.g. pra_comparison_storage.list_runs()
    (every section, or a filtered subset).

    Returns the per-scenario room-geometry table (a plain DataFrame) for
    further use/display; also prints the global summary as a side effect,
    so a bare `plots.suite_overview(storage.list_runs())` as a cell's last
    line shows both.
    """
    if not runs:
        print("No stored runs found -- run pra_comparison_runner.py first.")
        return None

    def field(getter):
        return _distinct_preserve_order(v for v in (getter(r) for r in runs) if v is not None)

    def warn_if_not_unique(label, values):
        suffix = "  [WARNING: differs across stored runs!]" if len(values) > 1 else ""
        print(f"{label}: {values[0] if len(values) == 1 else values}{suffix}")

    sections = sorted(set(r["section"] for r in runs))
    timestamps = sorted(r["timestamp"] for r in runs if r.get("timestamp"))

    print(f"Stored comparison runs: {len(runs)}  across {len(sections)} sections: {sections}")
    if timestamps:
        print(f"Generated: {timestamps[0]}  ..  {timestamps[-1]}")

    warn_if_not_unique("misuka variant", field(lambda r: r.get("misuka_variant")))
    warn_if_not_unique("Sampling rate [Hz]", field(lambda r: r["render_params"].get("sampling_rate")))
    warn_if_not_unique("Speed of sound [m/s]", field(lambda r: r["render_params"].get("speed_of_sound")))
    warn_if_not_unique("Repeats per rendering (X)", field(lambda r: r["render_params"].get("n_repeats")))
    warn_if_not_unique("PRA rays per render", field(lambda r: r["render_params"].get("pra_n_rays")))

    misuka_spp = sorted(field(lambda r: r["render_params"].get("misuka_spp")))
    print(f"misuka spp (single-render sections): {misuka_spp}")

    spp_sweeps = field(lambda r: r["render_params"].get("spp_values"))
    if spp_sweeps:
        sweep = spp_sweeps[0]
        suffix = "  [WARNING: differs across sweep sections!]" if len(spp_sweeps) > 1 else ""
        print(f"misuka spp sweep (convergence sections): {len(sweep)} values, "
              f"{min(sweep)} .. {max(sweep)}{suffix}")

    rooms_by_scenario = {}
    for r in runs:
        if r["scenario"] not in rooms_by_scenario and r.get("room"):
            rooms_by_scenario[r["scenario"]] = r["room"]
    room_table = pd.DataFrame([dict(scenario=name, **room)
                              for name, room in sorted(rooms_by_scenario.items())])
    return room_table


# =====================================================================
#                       Rendered geometries (static)
# =====================================================================
#
# Pure illustration of each scenario's geometry (2D outlines for the top-
# down/side-on views, mirrored here since the actual 3D geometry lives in
# test_pyroomacoustics_comparison.py's scenario_*() functions as individual
# mesh transforms, not a simple outline polygon) -- no rendering, no stored
# run data involved. Manually kept in sync with the scenario_*() functions'
# dim/src/mic values; not derived from them automatically (a known,
# pre-existing simplification carried over as-is from this notebook's
# previous version, not introduced here).

GEOMETRY_SCENARIOS = [
    dict(
        name="Shoebox 200m3",
        # axis-aligned box [0,8] x [0,5] x [0,5]
        top_outline=[(0, 0), (8.0, 0), (8.0, 5.0), (0, 5.0)],
        side_outline=[(0, 0), (8.0, 0), (8.0, 5.0), (0, 5.0)],
        height=5.0,
        src=(2.0, 2.5, 1.2),
        mic=(5.0, 2.5, 1.2),
    ),
    dict(
        name="L-Room 1000m3",
        # corners per misuka_lroom_shapes() docstring, a=5,b=10,c=10,d=5,height=10
        # footprint = a*b + c*d = 50 + 50 = 100 m^2, *height = 1000 m^3
        top_outline=[(0, 0), (15.0, 0), (15.0, 5.0), (5.0, 5.0), (5.0, 10.0), (0, 10.0)],
        # side view = silhouette when viewed along Y, i.e. the full a+c x height
        # bounding rectangle (the notch only shows up in the top view)
        side_outline=[(0, 0), (15.0, 0), (15.0, 10.0), (0, 10.0)],
        grayline=[(5.0, 5.0), (10.0, 0.0)],  # vertical line at the notch
        height=10.0,
        src=(4.0, 8.0, 1.2),
        mic=(14.0, 4.0, 1.2),
    ),
    dict(
        name="Coincident Reflections 216m3",
        top_outline=[(0, 0), (6.0, 0), (6.0, 6.0), (0, 6.0)],
        side_outline=[(0, 0), (6.0, 0), (6.0, 6.0), (0, 6.0)],
        height=6.0,
        src=(3.0, 3.0, 3.0),
        mic=(3.4, 3.0, 3.0),
    ),
    dict(
        name="Flutter Corridor 100m3",
        top_outline=[(0, 0), (20.0, 0), (20.0, 2.0), (0, 2.0)],
        side_outline=[(0, 0), (20.0, 0), (20.0, 2.5), (0, 2.5)],
        height=2.5,
        src=(2.0, 1.0, 1.25),
        mic=(3.0, 1.0, 1.25),
        aspect=(20, 8, 8),
    ),
]


def _plot_3d_guides(ax, point):
    x, y, z = point
    style = {"color": "gray", "linestyle": ":", "linewidth": 1}
    ax.plot([x, x], [y, y], [0, z], **style)       # z-axis
    ax.plot([x, x], [0, y], [z, z], **style)       # y-axis
    ax.plot([0, x], [y, y], [z, z], **style)       # x-axis


def _plot_3d_room(ax, outline, height, src, mic, aspect=None):
    outline = outline + [outline[0]]

    for x, y in outline[:-1]:
        ax.plot([x, x], [y, y], [0, height], color="black", linewidth=1)

    xs, ys = zip(*outline)
    ax.plot(xs, ys, [0] * len(xs), "k-", linewidth=1.5)
    ax.plot(xs, ys, [height] * len(xs), "k-", linewidth=1.5)

    walls = [
        [(x1, y1, 0), (x2, y2, 0), (x2, y2, height), (x1, y1, height)]
        for (x1, y1), (x2, y2) in zip(outline[:-1], outline[1:])
    ]
    ax.add_collection3d(Poly3DCollection(
        walls, alpha=0.08, edgecolor="black", linewidth=0.8
    ))

    ax.scatter(*src, color="red", s=60, label="source")
    ax.scatter(*mic, color="blue", s=60, label="receiver")

    _plot_3d_guides(ax, src)
    _plot_3d_guides(ax, mic)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    if aspect is not None:
        ax.set_box_aspect(aspect)
        ax.set_yticks([0, 1, 2])
        ax.set_xticks([0, 5, 10, 15, 20])
    else:
        ax.set_box_aspect((max(x for x, y in outline), max(y for x, y in outline), height))

    ax.view_init(elev=25, azim=-150)
    ax.legend(fontsize=8)


def _plot_room_view(ax, outline, src_xy, mic_xy, xlabel, ylabel, grayline=None):
    xs, ys = zip(*(outline + [outline[0]]))
    ax.plot(xs, ys, "k-", linewidth=1.5)
    if grayline is not None:
        ax.plot(*grayline, color="gray", linestyle="--", linewidth=1.5, zorder=-3)
    ax.plot(*src_xy, "ro", markersize=8, label="source")
    ax.plot(*mic_xy, "bo", markersize=8, label="receiver")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)


def plot_rendered_geometries(scenarios=None):
    """Top-down, side-on, and 3D views of each scenario's room geometry with
    source/receiver marked. `scenarios` defaults to GEOMETRY_SCENARIOS (every
    scenario with a simple-enough shape for a 2D outline -- excludes
    auditorium_complex, a real mesh); pass a subset of GEOMETRY_SCENARIOS to
    show fewer. Draws and shows each figure directly (unlike the other
    plot_* functions above, which return a figure for the caller to show) --
    there's no per-scenario result to hand back, only the illustration.
    """
    if scenarios is None:
        scenarios = GEOMETRY_SCENARIOS

    for scenario in scenarios:
        src, mic = scenario["src"], scenario["mic"]

        fig = plt.figure(figsize=(14, 5))
        ax_top = fig.add_subplot(1, 3, 1)
        ax_side = fig.add_subplot(1, 3, 2)
        ax_3d = fig.add_subplot(1, 3, 3, projection="3d")

        fig.suptitle(scenario["name"])
        ax_top.set_title("Top view (X-Y)")
        ax_side.set_title("Side view (X-Z)")
        ax_3d.set_title("3D view")

        _plot_room_view(ax_top, scenario["top_outline"], (src[0], src[1]), (mic[0], mic[1]), "x [m]", "y [m]")
        _plot_room_view(ax_side, scenario["side_outline"], (src[0], src[2]), (mic[0], mic[2]), "x [m]", "z [m]",
                       grayline=scenario.get("grayline"))
        _plot_3d_room(ax_3d, scenario["top_outline"], scenario["height"], src, mic, aspect=scenario.get("aspect"))

        fig.tight_layout()
        plt.show()


# =====================================================================
#                   Air attenuation: misuka vs. literature table
# =====================================================================
#
# Purely analytic/deterministic (misuka's apply_pure_tone_attenuation
# evaluates the continuous ISO 9613-1 formula, pyroomacoustics looks up its
# coarse literature table -- pyroomacoustics.parameters.air_absorption_table,
# Vorlaender 2008 -- both closed-form, no rendering/ray tracing/repeats
# involved) -- computation unchanged from the original single-shared-plot
# version; only the figure layout changed, one subplot per (temperature,
# humidity) condition instead of all four overlaid on one shared axes.

def plot_air_absorption_comparison(conditions=None):
    """One subplot per (temperature, humidity) condition, each showing
    exactly two curves over the shared frequency axis: misuka's continuous
    ISO 9613-1 coefficient vs. pyroomacoustics' literature-table coefficient,
    for that one condition -- instead of all conditions' curve pairs
    overlaid on a single shared plot.
    """
    if conditions is None:
        conditions = [(10.0, 40.0), (10.0, 80.0), (20.0, 40.0), (20.0, 80.0)]

    ncols = 2
    nrows = -(-len(conditions) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for ax, (temperature, humidity) in zip(axes.flat, conditions):
        physics = cmp.pra.parameters.Physics(temperature=temperature, humidity=humidity)
        air_abs = physics.get_air_absorption()
        freqs = air_abs["center_freqs"]
        pra_coeffs = air_abs["coeffs"]
        misuka_coeffs = [cmp._misuka_air_absorption_coeff(temperature, f, humidity) for f in freqs]

        ax.loglog(freqs, misuka_coeffs, "o-", color=MISUKA_COLOR, label="misuka (ISO 9613-1)")
        ax.loglog(freqs, pra_coeffs, "x--", color=PRA_COLOR, label="pra (literature table)")
        ax.set_title(f"T={temperature:.0f}°C  RH={humidity:.0f}%")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Attenuation coefficient [1/m]")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    for ax in axes.flat[len(conditions):]:
        ax.set_visible(False)

    fig.suptitle("Air attenuation: continuous ISO 9613-1 (misuka) vs. literature table (pyroomacoustics)")
    fig.tight_layout()
    return fig


# =====================================================================
#           Generic: mean +/- std band (repeats, Task 2)
# =====================================================================
#
# Not yet used by any plot_* function below -- individual sections will
# switch to this (reading the `{key}_mean`/`{key}_std` arrays every
# repeated run now stores, see pra_comparison_runner.py) in later,
# section-specific passes.

def plot_with_std_band(x, mean, std, ax=None, n_std=1.0, label=None, color=None, a=1.0, **line_kwargs):
    """Plot `mean` vs. `x` as a line, with a shaded +/- n_std*std band
    around it. Generic helper for any repeat-averaged sweep (ray-count
    convergence, JND convergence, per-scenario metrics, ...) -- takes plain
    arrays, not a load_run() dict, so it has no opinion about which section
    it's used from.

    Returns the Axes used (a new one is created if `ax` is None).
    """
    if ax is None:
        _, ax = plt.subplots()
    mean = np.asarray(mean)
    std = np.asarray(std)
    line, = ax.plot(x, mean, label=label, color=color, alpha=a, **line_kwargs)
    ax.fill_between(x, mean - n_std * std, mean + n_std * std, color=line.get_color(), alpha=0.2)
    return ax


# =====================================================================
#                   Section: main comparison (misuka vs. PRA rt_only/hybrid)
# =====================================================================

def plot_main_comparison(scenario_name, runs_by_mode):
    """runs_by_mode: {"rt_only": run, "hybrid": run}, each a load_run() dict
    for section="main_comparison". misuka's ETC is identical across modes
    (misuka has no ISM order), so it's only plotted once (rt_only's copy).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(scenario_name)
    pra_style = {"rt_only": dict(color=PRA_COLOR, linestyle="--"),
                 "hybrid": dict(color=PRA_COLOR_ALT, linestyle=":")}

    for mode_name, run in runs_by_mode.items():
        e_misuka = run["arrays"]["e_misuka"]
        e_pra = run["arrays"]["e_pra"]
        runtime_misuka_ms = run["metadata"]["metrics"]["runtime_misuka_ms"]
        runtime_pra_ms = run["metadata"]["metrics"]["runtime_pra_ms"]
        t = np.arange(len(e_misuka)) / cmp.SAMPLING_RATE

        misuka_label = f"misuka ({runtime_misuka_ms:.0f} ms)"
        pra_label = f"pra ({mode_name}, {runtime_pra_ms:.0f} ms)"

        with np.errstate(divide="ignore"):
            if mode_name == "rt_only":
                axes[0].plot(t, 10 * np.log10(np.maximum(e_misuka / e_misuka.max(), 1e-10)),
                             color=MISUKA_COLOR, label=misuka_label)
            axes[0].plot(t[:len(e_pra)], 10 * np.log10(np.maximum(e_pra / e_pra.max(), 1e-10)),
                         label=pra_label, **pra_style[mode_name])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            edc_m = pyrato_edc(e_misuka)
            edc_p = pyrato_edc(e_pra)
        if mode_name == "rt_only":
            pf.plot.time(edc_m, dB=True, log_prefix=10, ax=axes[1],
                         color=MISUKA_COLOR, label=misuka_label)
        pf.plot.time(edc_p, dB=True, log_prefix=10, ax=axes[1],
                     label=pra_label, **pra_style[mode_name])

    axes[0].set_title("Energy time curve (normalized, dB)")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylim(bottom=-105)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Schroeder EDC (pyrato, dB)")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def main_comparison_table(records):
    """records: list of metadata dicts (section="main_comparison"), one per
    (scenario, mode) -- e.g. from pra_comparison_storage.list_runs().
    """
    rows = []
    for meta in records:
        row = {"scenario": meta["scenario"], "pra_mode": meta["variant"]["pra_mode"]}
        row.update(meta["metrics"])
        rows.append(row)
    results = pd.DataFrame(rows)
    return results.style.format(precision=3).background_gradient(
        subset=[c for c in results.columns if c.endswith("_delta")],
        cmap="RdBu_r", vmin=-2, vmax=2)


# =====================================================================
#                   Section: atmosphere on/off (misuka only)
# =====================================================================

def _etc_db_mean_std(run, key):
    """Mean/std, in dB (normalized to the mean curve's own peak = 0 dB), of
    a stored ETC across repeats -- computed from the per-repeat array
    (`{key}_per_repeat`) if present, so the std band reflects genuine
    repeat-to-repeat spread in dB space, not a linear-space std naively
    relabeled. Falls back to treating the single stored mean as a "1
    repeat" if per-repeat data isn't there (older, pre-repeat stored runs;
    std is 0 in that case, so plot_with_std_band just draws a plain line).
    """
    mean = run["arrays"][key]
    per_repeat = run["arrays"].get(f"{key}_per_repeat")
    if per_repeat is None:
        per_repeat = mean[None, :]
    ref = mean.max()
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(np.maximum(per_repeat / ref, 1e-10))
    return db.mean(axis=0), db.std(axis=0)


def plot_atmosphere(scenario_name, run):
    e_no_atmo = run["arrays"]["e_no_atmo"]
    e_atmo = run["arrays"]["e_atmo"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(scenario_name)
    t = np.arange(len(e_no_atmo)) / cmp.SAMPLING_RATE

    mean_db_n, std_db_n = _etc_db_mean_std(run, "e_no_atmo")
    mean_db_a, std_db_a = _etc_db_mean_std(run, "e_atmo")
    plot_with_std_band(t, mean_db_n, std_db_n, ax=axes[0], color=MISUKA_COLOR, label="no atmosphere")
    plot_with_std_band(t[:len(mean_db_a)], mean_db_a, std_db_a, ax=axes[0], color=MISUKA_COLOR_ALT, label="inline atmosphere")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edc_n = pyrato_edc(e_no_atmo)
        edc_a = pyrato_edc(e_atmo)
    pf.plot.time(edc_n, dB=True, log_prefix=10, ax=axes[1], color=MISUKA_COLOR, label="no atmosphere")
    pf.plot.time(edc_a, dB=True, log_prefix=10, ax=axes[1], color=MISUKA_COLOR_ALT, label="inline atmosphere")

    axes[0].set_title("Energy time curve (normalized, dB)")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylim(bottom=-105)
    axes[0].legend()
    axes[1].set_title("Schroeder EDC (pyrato, dB)")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    fig.tight_layout()
    return fig


def atmosphere_table(records):
    rows = [dict(scenario=meta["scenario"], **meta["metrics"]) for meta in records]
    results_atmo = pd.DataFrame(rows)
    return results_atmo.style.format(precision=3).background_gradient(
        subset=["runtime_overhead_pct", "runtime_post_overhead_pct"], cmap="RdBu_r", vmin=-10, vmax=10)


def plot_atmosphere_runtime_summary(records):
    """records: list of metadata dicts (section="atmosphere"), one per
    scenario, in the desired display order.

    Bar chart and its runtime-results table side by side in one figure
    (table restricted to the runtime-related columns the bars themselves
    visualize -- ms values and overhead percentages -- not the full
    results_atmo table's T20/T30/C50/D50/edc_corr columns, which would be
    unreadably wide crammed next to a bar chart and aren't what this chart
    is about; see atmosphere_table() for the full table).
    """
    results_atmo = pd.DataFrame([dict(scenario=meta["scenario"], **meta["metrics"]) for meta in records])
    fig, (ax_bar, ax_table) = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    x = np.arange(len(results_atmo))
    width = 0.27
    bar_specs = [
        (x - width, "runtime_no_atmo_ms", "no atmosphere", MISUKA_COLOR),
        (x, "runtime_atmo_ms", "inline atmosphere", MISUKA_COLOR_ALT),
        (x + width, "runtime_post_total_ms", "no atmosphere + post-process", MISUKA_COLOR_ALT2),
    ]
    for xpos, col, label, color in bar_specs:
        ax_bar.bar(xpos, results_atmo[col], width, label=label, color=color)
        # Point at the bar top + std whisker above/below (runtime_*_std is
        # written by _repeated_arrays_and_metrics for every scalar metric).
        ax_bar.errorbar(xpos, results_atmo[col], yerr=results_atmo[f"{col}_std"],
                        fmt="o", color="gray", markersize=3, capsize=2,
                        linestyle="none",elinewidth=0.5,capthick=0.5, zorder=3)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(results_atmo["scenario"], rotation=30, ha="right")
    ax_bar.set_ylabel("Render time [ms]")
    ax_bar.set_title("misuka render time: no atmosphere vs. inline vs. post-processed atmosphere")
    ax_bar.legend(fontsize=8)

    table_cols = ["scenario", "runtime_no_atmo_ms", "runtime_atmo_ms", "runtime_overhead_pct",
                 "runtime_post_total_ms", "runtime_post_overhead_pct"]
    col_labels = ["scenario", "no atmo\n[ms]", "inline\n[ms]", "inline\noverhead [%]",
                 "post-total\n[ms]", "post\noverhead [%]"]
    table_values = results_atmo[table_cols].round(2).values
    ax_table.axis("off")
    tbl = ax_table.table(cellText=table_values, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(table_cols))))
    tbl.scale(1, 1.8)
    ax_table.set_title("Runtime results", fontsize=10)

    fig.tight_layout()

    for _, row in results_atmo.iterrows():
        print(f"{row['scenario']:24s} "
              f"inline_overhead={row['runtime_overhead_pct']:+6.2f}%  "
              f"post_process_overhead={row['runtime_post_overhead_pct']:+6.2f}%  "
              f"({row['runtime_no_atmo_ms']:.1f}ms -> "
              f"inline={row['runtime_atmo_ms']:.1f}ms / "
              f"post={row['runtime_post_total_ms']:.1f}ms)")
    return fig


# =====================================================================
#           Section: atmosphere methods (inline vs. post-processing)
# =====================================================================

def plot_atmosphere_methods(scenario_name, run):
    """Attenuation-only and post-vs-inline relative-error panels (as before,
    now with a repeat mean +/- std band instead of a single-render line --
    std is 0 for not-yet-repeated stored runs, so the band just draws as a
    plain line in that case). If the run also has multi-frequency data (see
    pra_comparison_runner.HIGHLIGHT_FREQUENCIES_HZ -- absent for
    auditorium_complex), a third panel shows the post-vs-inline *relative*
    error (unsigned, i.e. |%|) over time, one line per highlight frequency;
    see atmosphere_methods_frequency_table() for the explicit per-frequency
    summary numbers this panel's curves would otherwise only show visually.
    """
    diff_attenuation_only_pct = run["arrays"]["diff_attenuation_only_pct"]
    diff_attenuation_only_pct_std = run["arrays"].get(
        "diff_attenuation_only_pct_std", np.zeros_like(diff_attenuation_only_pct))
    diff_post_vs_inline_pct = run["arrays"]["diff_post_vs_inline_pct"]
    diff_post_vs_inline_pct_std = run["arrays"].get(
        "diff_post_vs_inline_pct_std", np.zeros_like(diff_post_vs_inline_pct))
    dt = 1.0 / cmp.SAMPLING_RATE
    t = np.arange(len(diff_attenuation_only_pct)) * dt

    has_freq_data = "freq_diff_pct_abs_mean" in run["arrays"]
    ncols = 3 if has_freq_data else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5))
    fig.suptitle(scenario_name)

    plot_with_std_band(t, diff_attenuation_only_pct, diff_attenuation_only_pct_std, ax=axes[0], color=MISUKA_COLOR)
    axes[0].set_title("Attenuation only\n(no attenuation vs. inline attenuation, same speed of sound)")

    plot_with_std_band(t, diff_post_vs_inline_pct, diff_post_vs_inline_pct_std, ax=axes[1], color=MISUKA_COLOR_ALT)
    axes[1].set_title("Post-processing vs. inline attenuation\n(same speed of sound)")

    for ax in axes[:2]:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Relative difference [%]")
        ax.set_xlim(0, t[-1])

    if has_freq_data:
        freqs = run["metadata"]["render_params"]["highlight_frequencies_hz"]
        diff_pct_abs_mean = run["arrays"]["freq_diff_pct_abs_mean"]
        diff_pct_abs_std = run["arrays"]["freq_diff_pct_abs_std"]
        for i, f in enumerate(freqs):
            plot_with_std_band(t, diff_pct_abs_mean[:, i], diff_pct_abs_std[:, i], ax=axes[2],
                               color=f"C{i}", a=0.5, label=f"{f:.0f} Hz")
        axes[2].set_title("Post-processing vs. inline attenuation\nrelative error (|%|), per frequency")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Relative difference [%] (abs)")
        axes[2].set_xlim(0, t[-1])
        axes[2].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    return fig


def atmosphere_methods_frequency_table(run):
    """Explicit, directly-readable numbers for the post-processing vs.
    inline attenuation discrepancy at each highlight frequency (see
    pra_comparison_runner.HIGHLIGHT_FREQUENCIES_HZ) -- relative error (%),
    unsigned, both the mean and the max over the render's whole duration --
    rather than only visible as a curve shape in plot_atmosphere_methods's
    third panel. Returns None if this run has no multi-frequency data
    (auditorium_complex).
    """
    if "freq_mean_abs_pct_mean" not in run["arrays"]:
        return None
    freqs = run["metadata"]["render_params"]["highlight_frequencies_hz"]
    a = run["arrays"]
    rows = [
        dict(
            frequency_hz=f,
            mean_abs_pct=a["freq_mean_abs_pct_mean"][i],
            max_abs_pct=a["freq_max_abs_pct_mean"][i],
        )
        for i, f in enumerate(freqs)
    ]
    return pd.DataFrame(rows).style.format(precision=5)


# =====================================================================
#                   Section: ray-count convergence
# =====================================================================

def plot_ray_count_convergence_error(scenario_name, run):
    spp = np.array(run["arrays"]["spp"])
    err_no_atmo = np.array(run["arrays"]["errors_no_atmo"])
    err_atmo = np.array(run["arrays"]["errors_atmo"])
    delta = err_atmo - err_no_atmo

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(scenario_name)

    axes[0].semilogx(spp, err_no_atmo, "o-", color=MISUKA_COLOR, label="no atmosphere")
    axes[0].semilogx(spp, err_atmo, "o-", color=MISUKA_COLOR_ALT, label="inline atmosphere")
    axes[0].set_xlabel("misuka spp (rays per frequency bin)")
    axes[0].set_ylabel("RMS EDC error vs. PRA hybrid [dB]")
    axes[0].set_title("Absolute error")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(spp, delta, "o-", color=MISUKA_COLOR_ALT2)
    axes[1].set_xlabel("misuka spp (rays per frequency bin)")
    axes[1].set_ylabel("extra error from atmosphere [dB]\n(inline atmosphere - no atmosphere)")
    axes[1].set_title("Atmosphere's own contribution")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_ray_count_convergence_relative_error(scenario_name, run):
    spp = np.array(run["arrays"]["spp"])
    rel_err_no_atmo_pct = np.array(run["arrays"]["rel_err_no_atmo_pct"])
    rel_err_atmo_pct = np.array(run["arrays"]["rel_err_atmo_pct"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(f"{scenario_name} -- relative RMS error vs. PRA hybrid")
    ax.semilogx(spp, rel_err_no_atmo_pct, "o-", color=MISUKA_COLOR, label="no atmosphere")
    ax.semilogx(spp, rel_err_atmo_pct, "o-", color=MISUKA_COLOR_ALT, label="inline atmosphere")
    ax.set_xlabel("misuka spp (rays per frequency bin)")
    ax.set_ylabel("relative RMS EDC error vs. PRA hybrid [%]")
    ax.set_title("Relative RMS error (100 * (10**(dB/10) - 1))")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def iso_convergence_table(run):
    spp = run["arrays"]["spp"]
    rows = []
    for i, spp_i in enumerate(spp):
        row = {"spp": int(spp_i)}
        for key in ("T20", "T30", "C50", "D50"):
            row[f"{key}_no_atmo"] = run["arrays"][f"{key}_no_atmo"][i]
            row[f"{key}_atmo"] = run["arrays"][f"{key}_atmo"][i]
            row[f"{key}_pra"] = run["metadata"]["metrics"][f"{key}_pra"]
            row[f"{key}_no_atmo_rel_pct"] = run["arrays"][f"{key}_no_atmo_rel_pct"][i]
            row[f"{key}_atmo_rel_pct"] = run["arrays"][f"{key}_atmo_rel_pct"][i]
        rows.append(row)
    results_conv = pd.DataFrame(rows)
    return results_conv.style.format(precision=3).background_gradient(
        subset=[c for c in results_conv.columns if c.endswith("_rel_pct")],
        cmap="RdBu_r", vmin=-50, vmax=50)


def plot_iso_convergence(scenario_name, run):
    spp = run["arrays"]["spp"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"{scenario_name} -- ISO 3382 descriptors vs. PRA hybrid, relative [%]")

    metric_colors = {"T20": "C0", "T30": "C1", "C50": "C2", "D50": "C3"}
    for key, color in metric_colors.items():
        axes[0].semilogx(spp, run["arrays"][f"{key}_no_atmo_rel_pct"], "o-", color=color, label=key)
        axes[1].semilogx(spp, run["arrays"][f"{key}_atmo_rel_pct"], "o-", color=color, label=key)

    axes[0].set_title("no atmosphere")
    axes[1].set_title("inline atmosphere")
    for ax in axes:
        ax.set_xlabel("misuka spp (rays per frequency bin)")
        ax.set_ylabel("relative to PRA hybrid [%]")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


# =====================================================================
#           Appendix 1: ISM peak level
# =====================================================================

def plot_ism_peak_bars(scenario_name, run):
    ism_window_energy = run["arrays"]["ism_window_energy"]
    ref = ism_window_energy / ism_window_energy[0]
    spp_values = run["arrays"]["spp"]
    n_show = list(spp_values)

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.get_cmap("Greens")
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, n_rays in enumerate(n_show):
            vals = run["arrays"][f"misuka_window_energy_spp{n_rays}"]
            vals = vals / run["arrays"][f"misuka_window_energy_spp{n_show[-1]}"][0]
            ax.bar(np.arange(len(vals)) + i / (len(n_show) + 2), vals,
                   width=1 / (len(n_show) + 2), color=cmap(0.4 + 0.5 * i / len(n_show)),
                   label=f"N=2^{int(np.log2(n_rays))}")
    ax.hlines(ref, range(len(ref)), range(1, len(ref) + 1), color=PRA_COLOR, lw=0.9, label="PRA-ISM")
    ax.set_yscale("log")
    ax.set_xlabel("Peak #")
    ax.set_ylabel("Normalized energy")
    ax.set_title(f"{scenario_name}: peak energies, misuka (bars) vs. PRA-ISM (line)")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    return fig


def plot_ism_peak_convergence_metrics(scenario_name, run):
    spp = run["arrays"]["spp"]
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 5))
    fig.suptitle(scenario_name)
    axs[0].semilogx(spp, run["arrays"]["cosine_similarity"], "o-", color=MISUKA_COLOR)
    axs[0].set_ylabel("Cosine similarity")
    axs[1].semilogx(spp, run["arrays"]["rmspe_pct"], "o-", color=MISUKA_COLOR)
    axs[1].set_ylabel("RMSPE [%]")
    axs[2].semilogx(spp, run["arrays"]["mpe_pct"], "o-", color=MISUKA_COLOR)
    axs[2].set_ylabel("MPE [%]")
    axs[2].set_xlabel("misuka spp (rays per frequency bin)")
    for ax in axs:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


# =====================================================================
#           Appendix 2: Hybrid parameter level
# =====================================================================

def plot_hybrid_jnd(scenario_name, run):
    spp = run["arrays"]["spp"]
    reference_params = run["metadata"]["metrics"]["reference_params"]

    UNITS = {"C50": "dB", "C80": "dB", "G": "dB", "TS": "s", "EDT": "s", "T60": "s"}

    # z-order, back to front: JND band (green) < repeat range (light blue) <
    # +/-1 std (dark blue) < the actual mean/reference lines on top.
    Z_JND, Z_RANGE, Z_STD, Z_LINES = 1, 3, 4, 2

    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    fig.suptitle(scenario_name)
    for ax, param in zip(axs.flat, list(JND_ABS) + list(JND_REL)):
        vals = run["arrays"][f"misuka_{param}"]
        per_repeat = run["arrays"][f"misuka_{param}_per_repeat"]  # (n_repeats, n_spp)
        std = run["arrays"][f"misuka_{param}_std"]
        pra_val = reference_params[param]

        jnd = JND_ABS[param] if param in JND_ABS else JND_REL[param] * abs(pra_val)
        # Band centered on misuka's own converged (highest-spp) mean, not on
        # PRA's value -- same comparison as before, just read as "is PRA
        # within misuka's own tolerance" instead of the other way round.
        # Falls back to PRA's value if misuka's last value is itself
        # non-finite (e.g. G for l_room_1000m3 -- no line of sight, see
        # parameters_full's g_radius_term docstring).
        band_center = vals[-1] if np.isfinite(vals[-1]) else pra_val
        band_lo, band_hi = band_center - jnd, band_center + jnd
        ax.axhspan(band_lo, band_hi, color="C0", alpha=0.15, zorder=Z_JND, label="JND")

        ax.fill_between(spp, per_repeat.min(axis=0), per_repeat.max(axis=0),
                        color="C2", alpha=0.15, zorder=Z_RANGE, label="misuka range")
        ax.fill_between(spp, vals - std, vals + std,
                        color="C2", alpha=0.35, zorder=Z_STD, label=r"misuka ±1 $\sigma$")

        ax.semilogx(spp, vals, marker="o", linestyle="-", color="C2", label="misuka", zorder=Z_LINES+1, markersize=2.0)
        ax.axhline(pra_val, color="C1", label="PRA-hybrid", zorder=Z_LINES)
        ax.set_ylabel(f"{param} / {UNITS[param]}")
        ax.set_xlabel("misuka spp")
        ax.grid(True, which="both", alpha=0.3)

        # spp is always well-defined and positive, but `vals` isn't -- a
        # parameter that's non-finite at every spp for this scenario (e.g.
        # G for l_room_1000m3: no line-of-sight between source and mic, see
        # parameters_full's g_radius_term docstring) leaves semilogx() with
        # nothing usable to autoscale from, and matplotlib silently falls
        # back to a default view straddling zero -- invalid for a log axis,
        # and only surfaces as a crash later, in fig.tight_layout(). Setting
        # this explicitly from spp itself sidesteps that regardless of what
        # the y-data looks like.
        ax.set_xlim(spp.min(), spp.max())

        # y-range: PRA's value must always be visible, and the (now
        # misuka-centered) JND band should be as large a fraction of the
        # visible height as possible without exceeding 2/5. The minimal
        # span covering both the band and PRA's value sets a lower bound on
        # the total height; if PRA sits far from the band (e.g. G), that
        # bound already exceeds what a 2/5 band would need, so the band
        # ends up smaller than 2/5 -- unavoidable while keeping PRA
        # visible. If PRA is close to/inside the band, the axis is padded
        # out evenly on both sides until the band shrinks to exactly 2/5.
        band_height = 2 * jnd
        base_lo, base_hi = min(band_lo, pra_val), max(band_hi, pra_val)
        total_height = max(base_hi - base_lo, band_height / 0.4)
        extra = total_height - (base_hi - base_lo)
        ax.set_ylim(base_lo - extra / 2, base_hi + extra / 2)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, 0), frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# =====================================================================
#           Appendix 3: RT-only histogram level
# =====================================================================

def plot_rt_histogram_comparison(scenario_name, run):
    etc_misuka = run["arrays"]["etc_misuka"]
    pra_rt_etc = run["arrays"]["pra_rt_etc"]
    dt = 1.0 / cmp.SAMPLING_RATE
    n = min(len(etc_misuka), len(pra_rt_etc))
    times_ms = np.arange(n) * dt * 1e3

    fig, ax = plt.subplots(figsize=(8, 3.5))
    with np.errstate(divide="ignore", invalid="ignore"):
        ax.semilogy(times_ms, etc_misuka[:n] / etc_misuka[:n].max(), color=MISUKA_COLOR, label="misuka RT")
        ax.semilogy(times_ms, pra_rt_etc[:n] / pra_rt_etc[:n].max(), color=PRA_COLOR, linestyle="--",
                   label="PRA RT histogram", alpha=0.8)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Normalized energy")
    ax.set_xlim(0, min(times_ms[-1], 100))
    ax.legend(fontsize=8)
    ax.set_title(f"{scenario_name}: RT-only, no sinc reconstruction")
    fig.tight_layout()
    return fig


# =====================================================================
#   Post-processing vs. inline attenuation: sampling-rate/frequency sweep
# =====================================================================

def plot_post_vs_inline_attenuation_sweep(scenario_name, run):
    """analyze_post_vs_inline_attenuation.py's sweep: how much the relative-
    ETC discrepancy between post-processing and inline atmospheric
    attenuation depends on time-bin width (sampling_rate) and frequency.
    Log-log, since the theory (see that script's docstring) predicts the
    discrepancy scales with bin width -- a straight line here is the
    signature of that power-law relationship, not just "smaller is better".
    """
    fs = run["arrays"]["sampling_rates_hz"]
    freqs = run["arrays"]["frequencies_hz"]
    mean_abs_pct = run["arrays"]["mean_abs_pct"]
    mean_abs_pct_std = run["arrays"]["mean_abs_pct_std"]
    max_abs_pct = run["arrays"]["max_abs_pct"]
    max_abs_pct_std = run["arrays"]["max_abs_pct_std"]

    fig, axs = plt.subplots(2, 1, figsize=(11, 4.5))
    for j, f in enumerate(freqs):
        color = f"C{j}"
        axs[0].errorbar(fs, mean_abs_pct[:, j], yerr=mean_abs_pct_std[:, j],
                        marker="o", markersize=3, color=color, label=f"{f:.0f} Hz")
        axs[1].errorbar(fs, max_abs_pct[:, j], yerr=max_abs_pct_std[:, j],
                        marker="o", markersize=3, color=color, label=f"{f:.0f} Hz")

    for ax, title in zip(axs, ("mean |relative ETC difference|", "max |relative ETC difference|")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        if title.startswith("mean"):
            ax.set_xlabel("sampling rate [Hz]  (bin width = 1/fs)")
        ax.set_ylabel("post vs. inline, |%|")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=8, title="frequency")
    #fig.suptitle(f"{scenario_name}: post-processing vs. inline attenuation error")
    fig.tight_layout()
    return fig


def plot_post_vs_inline_attenuation_time_profile(scenario_name, run, sampling_rate_hz=44100):
    """Is the post-vs-inline attenuation discrepancy spread evenly over
    time, or concentrated in the direct sound/early reflections vs. the
    late reverberant tail? Uses the full per-bin relative-ETC-difference
    curve analyze_post_vs_inline_attenuation.py already stores (one
    sampling rate's worth, default 44100 Hz -- the suite's own default),
    one line per frequency, against time. A light rolling-mean smooths
    per-bin Monte Carlo shot noise without hiding the time trend itself.
    The direct-sound arrival time (source-mic distance / speed of sound)
    is marked for reference.
    """
    freqs = run["arrays"]["frequencies_hz"]
    diff_pct = run["arrays"][f"diff_pct_curve_fs{sampling_rate_hz}"]  # (n_time_bins, n_freq)
    times_ms = np.arange(diff_pct.shape[0]) / sampling_rate_hz * 1e3

    room = run["metadata"]["room"]
    src, mic = np.array(room["source_pos"]), np.array(room["mic_pos"])
    direct_sound_ms = np.linalg.norm(mic - src) / cmp.SPEED_OF_SOUND * 1e3

    window = max(1, int(round(sampling_rate_hz * 0.002)))  # ~2 ms smoothing window
    smoothed_per_freq = [pd.Series(np.abs(diff_pct[:, j])).rolling(window, center=True, min_periods=1).mean()
                        for j in range(len(freqs))]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for j, f in enumerate(freqs):
        for ax in axs:
            ax.semilogy(times_ms, smoothed_per_freq[j], color=f"C{j}", label=f"{f:.0f} Hz")
    for ax in axs:
        ax.axvline(direct_sound_ms, color="black", linestyle=":", linewidth=1,
                  label=f"direct sound ({direct_sound_ms:.1f} ms)")
        ax.set_xlabel("Time [ms]")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel(f"|post vs. inline| [%]  ({window/sampling_rate_hz*1e3:.1f} ms rolling mean)")
    axs[0].set_title("full render")
    axs[1].set_title("direct sound / early reflections (zoom)")
    axs[1].set_xlim(0, 30)
    axs[0].legend(fontsize=8, title="frequency")
    fig.suptitle(f"{scenario_name}: error over time at {sampling_rate_hz:.0f} Hz sampling rate")
    fig.tight_layout()
    return fig


def appendix_summary_table(section1_records, section2_records, section3_records):
    return pd.concat([
        pd.DataFrame([dict(scenario=m["scenario"], **m["metrics"], comparison="misuka vs. PRA-ISM")
                     for m in section1_records]),
        pd.DataFrame([dict(scenario=m["scenario"], n_max=m["metrics"]["n_max"],
                           comparison="misuka vs. PRA-Hybrid") for m in section2_records]),
        pd.DataFrame([dict(scenario=m["scenario"], **m["metrics"], comparison="misuka vs. PRA-RT-only")
                     for m in section3_records]),
    ], ignore_index=True)


# =====================================================================
#          Evaluation of misuka's atmospheric extensions
# =====================================================================
#
# Figures for atmospheric_evaluation.py's three blocks (post-vs-inline
# attenuation, speed-of-sound method spread, inverse atmospheric
# rendering). Same contract as everything above: these read stored runs and
# draw, they never render or recompute.

def plot_attenuation_frequency_dependence(scenario_name, run):
    """How the post-vs-inline attenuation error grows with frequency.

    The ISO 9613-1 attenuation coefficient scales with f^2, so the same
    per-bin distance ambiguity buys a progressively larger energy error --
    log-log, where that power law shows up as a straight line.
    """
    frequencies = run["arrays"]["frequencies_hz"]
    mean_abs_pct = run["arrays"]["mean_abs_pct"]
    max_abs_pct = run["arrays"]["max_abs_pct"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(frequencies, mean_abs_pct, yerr=run["arrays"]["mean_abs_pct_std"],
                marker="o", markersize=4, color=MISUKA_COLOR, label="mean")
    # No error bars on the max series: relative_difference_percent() is
    # bounded to +/-100% by construction (it normalises by max(|a|,|b|)), so
    # as soon as one bin has fully decayed the per-seed maximum jumps to the
    # bound. Around 14 kHz only some seeds hit such a bin, giving a spread
    # (std 28.8 against a mean of 15.4) whose lower whisker ran off the log
    # axis as a vertical line. The spread is an artefact of that bound, not
    # a measurement uncertainty, so plotting it as one would be misleading.
    ax.plot(frequencies, max_abs_pct, marker="s", markersize=4,
            color=MISUKA_COLOR_ALT, label="max")
    if np.nanmax(max_abs_pct) >= 99.9:
        ax.axhline(100.0, linestyle="--", linewidth=0.8, color="0.6")
        ax.text(frequencies[0], 100.0, " metric bound (100 %)", fontsize=7,
                va="bottom", ha="left", color="0.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("post vs. inline, absolute difference [%]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _bin_width_x_axes(ax, bin_widths_s, bin_widths_hz):
    """Shared x-axis furniture for the bin-width plots: explicit ticks (the
    default log locator's 2/3/6 x 10^n minors collide at this figure size)
    and the sampling-rate axis mirrored on top, since bin width is 1/fs and
    fs is what the suite actually configures per render."""
    ax.set_xscale("log")
    bin_widths_ms = bin_widths_s * 1e3
    ax.set_xticks(bin_widths_ms)
    ax.set_xticklabels([f"{w:.3g}" for w in bin_widths_ms], fontsize=8)
    ax.set_xticks([], minor=True)

    sampling_axis = ax.secondary_xaxis(
        "top", functions=(lambda ms: 1e3 / np.where(ms == 0, np.nan, ms),
                          lambda hz: 1e3 / np.where(hz == 0, np.nan, hz)))
    sampling_axis.set_xlabel("sampling rate [Hz]")
    sampling_axis.set_xticks(bin_widths_hz)
    sampling_axis.set_xticklabels([f"{f:.0f}" for f in bin_widths_hz], fontsize=8)
    sampling_axis.set_xticks([], minor=True)


def plot_attenuation_bin_width(scenario_name, run):
    """max/avg |t_real - dt_i| against time-bin width, one curve pair per
    frequency.

    Both axes are the quantity the post-processing approximation actually
    gets wrong: the gap between a path's true arrival time and the bin time
    its energy is attenuated by. Shown in metres (the form that enters the
    attenuation exponent) with the seconds axis mirrored on the right.

    Frequency sets the colour (see frequency_colors); avg and max at the
    same frequency share that colour and differ only by marker (circle vs.
    triangle), so the plot reads as "one green family per frequency, shape
    tells you which statistic".
    """
    bin_widths_s = run["arrays"]["bin_widths_s"]
    bin_widths_hz = run["arrays"]["bin_widths_hz"]
    frequencies = run["arrays"]["frequencies_hz"]
    max_err_m = run["arrays"]["max_err_m"]  # (n_bin_widths, n_freq)
    avg_err_m = run["arrays"]["avg_err_m"]
    speed_of_sound = run["metadata"]["metrics"]["speed_of_sound_atmo"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = frequency_colors(frequencies)
    for j, (freq, color) in enumerate(zip(frequencies, colors)):
        ax.plot(bin_widths_s * 1e3, avg_err_m[:, j], marker="o", markersize=4,
                color=color, label=f"{freq:.0f} Hz, avg")
        ax.plot(bin_widths_s * 1e3, max_err_m[:, j], marker="^", markersize=5,
                color=color, linestyle="--", label=f"{freq:.0f} Hz, max")
    ax.set_yscale("log")
    ax.set_xlabel("time-bin width [ms]")
    ax.set_ylabel("distance error [m]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _bin_width_x_axes(ax, bin_widths_s, bin_widths_hz)

    secondary = ax.secondary_yaxis(
        "right", functions=(lambda m: m / speed_of_sound * 1e3,
                            lambda s: s * speed_of_sound / 1e3))
    secondary.set_ylabel("time error [ms]")
    fig.tight_layout()
    return fig


def plot_attenuation_bin_width_etc_error(scenario_name, run):
    """Relative ETC error (post-processing vs. inline attenuation) against
    time-bin width, one curve pair per frequency.

    Same layout and colour/marker convention as plot_attenuation_bin_width
    (frequency -> colour, avg/max -> marker) -- this is the quantity 1a
    reports (relative_difference_percent, in %), plotted against bin width
    instead of a fixed sampling rate.
    """
    bin_widths_s = run["arrays"]["bin_widths_s"]
    bin_widths_hz = run["arrays"]["bin_widths_hz"]
    frequencies = run["arrays"]["frequencies_hz"]
    mean_abs_pct = run["arrays"]["mean_abs_pct"]  # (n_bin_widths, n_freq)
    max_abs_pct = run["arrays"]["max_abs_pct"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = frequency_colors(frequencies)
    for j, (freq, color) in enumerate(zip(frequencies, colors)):
        ax.plot(bin_widths_s * 1e3, mean_abs_pct[:, j], marker="o", markersize=4,
                color=color, label=f"{freq:.0f} Hz, avg")
        ax.plot(bin_widths_s * 1e3, max_abs_pct[:, j], marker="^", markersize=5,
                color=color, linestyle="--", label=f"{freq:.0f} Hz, max")
    if np.nanmax(max_abs_pct) >= 99.9:
        ax.axhline(100.0, linestyle="--", linewidth=0.8, color="0.6")
        ax.text(bin_widths_s[0] * 1e3, 100.0, " metric bound (100 %)", fontsize=7,
                va="bottom", ha="left", color="0.4")
    ax.set_yscale("log")
    ax.set_xlabel("time-bin width [ms]")
    ax.set_ylabel("post vs. inline, absolute difference [%]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _bin_width_x_axes(ax, bin_widths_s, bin_widths_hz)
    fig.tight_layout()
    return fig


def plot_attenuation_runtime(scenario_name, run):
    """Runtime distributions of the three attenuation modes, with the
    significance result the comparison rests on.

    Violins show the shape of each distribution (runtimes are typically
    right-skewed by scheduling noise), the boxes inside give median and
    quartiles. The annotated p-value is the paired test
    atmospheric_evaluation picked per distribution -- see its
    `*_test` metric for which one.
    """
    keys = ["runtime_none_ms", "runtime_inline_ms", "runtime_post_ms"]
    labels = ["no attenuation", "inline", "post-processing"]
    colors = [MISUKA_COLOR_ALT2, MISUKA_COLOR, MISUKA_COLOR_ALT]
    samples = [np.asarray(run["arrays"][key], dtype=float) for key in keys]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    parts = ax.violinplot(samples, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.35)
    ax.boxplot(samples, widths=0.12, showfliers=False,
               medianprops=dict(color="black"))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("runtime [ms]")
    ax.grid(True, axis="y", alpha=0.3)

    metrics = run["metadata"]["metrics"]
    p_value = metrics.get("inline_vs_none_p")
    if p_value is not None:
        test_name = metrics.get("inline_vs_none_test", "paired test")
        verdict = "significant" if p_value < 0.05 else "not significant"
        ax.set_title(f"inline vs. no attenuation: p = {p_value:.3f} "
                     f"({test_name}, {verdict}, n = {metrics.get('n_repeats')})",
                     fontsize=10)
    fig.tight_layout()
    return fig


def plot_speed_of_sound_extremes(scenario_name, run, cmap="viridis"):
    """Where `cramer` / `ideal_gas` drift furthest from the ISO 9613-1
    (temperature-only) speed of sound, over a realistic indoor range.

    One panel per method: |method - ISO| over temperature and relative
    humidity at that method's worst-case CO2. CO2 is not an axis because its
    influence is far weaker than the other two -- see the stored metrics for
    its optimal value, and for the located maximum (which sits in the
    corner of the searched box, so the colour map already shows where it is).
    """
    metrics = run["metadata"]["metrics"]
    temperatures = run["arrays"]["temperatures_c"]
    humidities = run["arrays"]["relative_humidities"]
    methods = [m for m in ("cramer", "ideal_gas")
               if f"difference_grid_{m}" in run["arrays"]]

    fig, axs = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 4.2),
                            squeeze=False)
    for ax, method in zip(axs[0], methods):
        grid = run["arrays"][f"difference_grid_{method}"]
        mesh = ax.pcolormesh(humidities, temperatures, grid, shading="auto",
                             cmap=cmap)
        fig.colorbar(mesh, ax=ax, label=r"$|c_{method} - c_{ISO}|$ [m/s]")
        ax.grid(False)
        ax.set_xlabel("relative humidity")
        ax.set_ylabel(r"temperature [$^\circ$C]")
        ax.set_title(f"{method.replace('_', ' ')} vs. ISO 9613-1\n"
                     f"max {metrics[f'{method}_max_abs_difference_ms']:.3f} m/s "
                     f"at {metrics[f'{method}_co2_ppm']:.0f} ppm CO$_2$", fontsize=10)
    fig.tight_layout()
    return fig


# cmr10 carries no degree glyph (same reason axes.unicode_minus is off
# suite-wide), so the temperature label goes through mathtext.
# Band drawn behind the goal in the left panel of plot_inverse_rendering.
# Temperature uses its JND; relative humidity deliberately gets none (only
# its end value is of interest). Keyed by the run's stored `parameter`, with
# the leading token of a variant name as fallback for older runs.
INVERSE_PARAMETER_JND = {
    "temperature": 0.38,
    "relative_humidity": None,
}

INVERSE_PARAMETER_LABELS = {
    "temperature": r"temperature [$^\circ$C]",
    "temperature_minus73": r"temperature [$^\circ$C]",
    "relative_humidity": "relative humidity",
}


def plot_inverse_rendering(scenario_name, run, parameter_label=None,
                           extra_panels=None, etc_xlim=None,
                           etc_dynamic_range_db=80.0, goal_band=None,
                           etc_zoom=None,
                           etc_zoom_inset_bounds=(0.52, 0.50, 0.44, 0.44),
                           etc_zoom_span_fraction=0.06,
                           etc_zoom_center=0.1):
    """The tutorial's three-panel inverse-rendering figure, extensible.

    Panels, left to right: the retrieved parameter over the iterations
    (with the goal as a reference line), the loss over the iterations, and
    the reconstructed ETC against the target ETC.

    `extra_panels` appends further panels without touching that layout --
    pass a sequence of callables, each invoked as ``panel(ax, run)`` on its
    own axes to the right (e.g. a runtime-per-iteration panel). The figure
    widens to keep every panel the same size, so adding one never squeezes
    the existing three.
    """
    arrays, metrics = run["arrays"], run["metadata"]["metrics"]
    extra_panels = list(extra_panels or [])
    n_panels = 3 + len(extra_panels)

    fig, axs = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4),
                            layout="constrained", squeeze=False)
    axs = axs[0]

    values = arrays["values"]
    goal = metrics["goal"]
    if parameter_label is None:
        variant = run["metadata"].get("variant") or ""
        parameter_label = INVERSE_PARAMETER_LABELS.get(
            variant, INVERSE_PARAMETER_LABELS.get(variant.split("_")[0], "parameter"))
    label = parameter_label

    # JND band behind the goal line, per retrieved parameter (see
    # INVERSE_PARAMETER_JND): temperature gets its JND, relative humidity
    # gets none -- for humidity only the end value is of interest.
    band = goal_band
    if band is None:
        parameter = metrics.get("parameter") or (run["metadata"].get("variant") or "")
        band = INVERSE_PARAMETER_JND.get(
            parameter, INVERSE_PARAMETER_JND.get(parameter.split("_")[0]))
    if band:
        # mathtext for the sign: cmr10 has no +/- glyph, same reason
        # axes.unicode_minus is disabled suite-wide.
        axs[0].axhspan(goal - band, goal + band, color=PRA_COLOR, alpha=0.15,
                       label=rf"JND: $\pm${band:g}")
    # Mean over the seeds with their full range behind it. `values` is
    # (n_seeds, iterations) for a multi-seed run and 1-D for an older
    # single-run one, which is why both shapes are handled here.
    if values.ndim > 1:
        mean_values = arrays.get("values_mean", values.mean(axis=0))
        lo_values = arrays.get("values_min", values.min(axis=0))
        hi_values = arrays.get("values_max", values.max(axis=0))
        iterations = np.arange(len(mean_values))
        axs[0].fill_between(iterations, lo_values, hi_values,
                            color=MISUKA_COLOR, alpha=0.2,
                            label=f"range ({values.shape[0]} runs)")
    else:
        mean_values = values
        iterations = np.arange(len(mean_values))
    axs[0].plot(iterations, mean_values, color=MISUKA_COLOR, label="mean")
    axs[0].axhline(goal, linestyle="dotted", color=PRA_COLOR, label=f"goal: {goal:g}")
    axs[0].plot(iterations[-1], mean_values[-1], marker="o", color=MISUKA_COLOR,
                label=f"final: {mean_values[-1]:.3g}")
    converged_at = metrics.get("converged_at", -1)
    if converged_at is not None and converged_at >= 0:
        axs[0].axvline(converged_at, linestyle="--", color="0.5",
                       label=f"converged: {converged_at}")
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel(label)
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)

    losses = arrays["losses"]
    if losses.ndim > 1:
        axs[1].fill_between(np.arange(losses.shape[1]),
                            arrays.get("losses_min", losses.min(axis=0)),
                            arrays.get("losses_max", losses.max(axis=0)),
                            color=MISUKA_COLOR, alpha=0.2)
        axs[1].semilogy(arrays.get("losses_mean", losses.mean(axis=0)),
                        color=MISUKA_COLOR)
    else:
        axs[1].semilogy(losses, color=MISUKA_COLOR)
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel("loss (relative squared error)")
    axs[1].grid(True, which="both", alpha=0.3)

    # Runtime box plot, inset top right. Once these retrievals are repeated
    # over several seeds, the run carries one total runtime per seed
    # (`runtime_per_seed_s`) and that is what gets boxed; until then the
    # single run's per-iteration times are shown instead, which is the same
    # quantity at a finer granularity. The y label says which of the two.
    # One value per optimisation run, not per iteration.
    runtime_samples = np.asarray(arrays["runtime_per_seed_s"], dtype=float)
    runtime_title = "runtime / run [s]"
    inset = axs[1].inset_axes([0.76, 0.58, 0.16, 0.38])
    inset.boxplot(runtime_samples, widths=0.7, showfliers=False,
                  medianprops=dict(color=MISUKA_COLOR))
    # A box plot of a single sample carries no x information, so everything
    # but the y axis is removed rather than drawn and ignored.
    inset.set_xticks([])
    for side in ("top", "right", "bottom"):
        inset.spines[side].set_visible(False)
    inset.yaxis.set_major_locator(mticker.LinearLocator(numticks=4))
    # Precision follows the spread: runtimes that differ by ~1 s render as
    # four identical labels under a fixed "%.0f".
    span = float(np.ptp(runtime_samples))
    decimals = 0 if span >= 10 else (1 if span >= 1 else 2)
    inset.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{decimals}f"))
    # Unit in the title, not on a y label: an inset y label is drawn outside
    # the inset box and lands on top of the loss curve behind it.
    inset.set_title(runtime_title, fontsize=6)
    inset.tick_params(axis="y", labelsize=6)
    inset.patch.set_alpha(0.9)

    times = arrays["times_s"]
    with np.errstate(divide="ignore"):
        etc_final_db = 10 * np.nan_to_num(np.log10(np.abs(arrays["etc_final"])), neginf=-300)
        etc_ref_db = 10 * np.nan_to_num(np.log10(np.abs(arrays["etc_reference"])), neginf=-300)
    axs[2].plot(times, etc_ref_db, color=PRA_COLOR, linewidth=1.8, label="target")
    axs[2].plot(times, etc_final_db, color=MISUKA_COLOR, linewidth=0.8,
                alpha=0.9, label="reconstructed")
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("energy [dB]")
    if etc_xlim is not None:
        axs[2].set_xlim(*etc_xlim)
    # Empty time bins land at the -300 dB floor above; without an explicit
    # limit they compress the actual decay into a few pixels.
    finite_top = np.max(etc_ref_db)
    axs[2].set_ylim(finite_top - etc_dynamic_range_db, finite_top + 5)
    axs[2].legend(fontsize=8, loc="lower left")
    axs[2].grid(True, alpha=0.3)

    # Zoomed inset on the two curves, as in the tutorial's own result figure
    # (atmosphere_optimization.ipynb). The window defaults to the first few
    # milliseconds after the direct sound, where the attenuation difference
    # the optimisation acts on is largest and the two curves are otherwise
    # indistinguishable at full scale.
    if etc_zoom is None:
        # The reflection peaks around 0.1 s -- the part of the echogram the
        # retrieval's attenuation difference is read off, and far enough past
        # the direct sound not to be dominated by it.
        half = 0.5 * etc_zoom_span_fraction * float(times[-1])
        lo, hi = etc_zoom_center - half, etc_zoom_center + half
    else:
        lo, hi = etc_zoom
    if hi > lo:
        window = (times >= lo) & (times <= hi)
        if np.any(window):
            # inset_axes + indicate_inset_zoom rather than the tutorial's
            # zoomed_inset_axes: the zoom window spans a couple of percent of
            # the x axis but most of the y axis, and a fixed zoom *factor*
            # turns that aspect ratio into a sliver. Giving the inset an
            # explicit box instead keeps it readable whatever the window is.
            axins = axs[2].inset_axes(etc_zoom_inset_bounds)
            axins.plot(times, etc_ref_db, color=PRA_COLOR, linewidth=1.8)
            axins.plot(times, etc_final_db, color=MISUKA_COLOR, linewidth=0.8,
                       alpha=0.9)
            axins.set_xlim(lo, hi)
            # Ignore the -300 dB floor of empty bins when scaling, exactly as
            # the main panel does -- otherwise a single empty bin inside the
            # window stretches the inset over 3000 dB and flattens everything.
            floor = finite_top - etc_dynamic_range_db
            visible = etc_ref_db[window][etc_ref_db[window] > floor]
            top = np.max(visible) if visible.size else finite_top
            bottom = np.min(visible) if visible.size else floor
            pad = max(0.1 * (top - bottom), 1.0)
            axins.set_ylim(max(bottom - pad, floor), top + pad)
            axins.tick_params(labelbottom=False, labelleft=False, length=2)
            axins.patch.set_alpha(0.9)
            axs[2].indicate_inset_zoom(axins, edgecolor="0.5", linewidth=0.6)

    for ax, panel in zip(axs[3:], extra_panels):
        panel(ax, run)
    return fig


def inverse_convergence_table(records):
    """Iterations-to-convergence and runtime across inverse-rendering runs
    (the distance/frequency variants of block 3b, or any other set)."""
    rows = []
    for name, metrics in records:
        rows.append({
            "run": name,
            "goal": metrics.get("goal"),
            "final": metrics.get("final_value"),
            "abs. error": metrics.get("absolute_error"),
            "iterations to converge": (metrics.get("converged_at")
                                       if metrics.get("converged_at", -1) >= 0 else None),
            "seeds converged": (f"{metrics.get('n_converged')}/{metrics.get('n_seeds')}"
                                if metrics.get("n_seeds") else None),
            "distance [m]": metrics.get("distance_m"),
            "frequency [Hz]": metrics.get("frequency_hz"),
            "runtime / run [s]": metrics.get("mean_runtime_per_seed_s"),
        })
    return pd.DataFrame(rows).style.format(precision=3)


def _feature_edges(triangles, angle_tol_deg=5.0, quantize=6):
    """Outline edges of a triangle soup, with coplanar seams removed.

    A triangulated wall carries many edges that exist only because the
    surface was split into triangles; drawing them suggests structure the
    room does not have. An edge shared by exactly two triangles whose
    normals agree to within `angle_tol_deg` lies inside a flat region and is
    dropped. What remains are the borders of each planar face plus any
    genuine crease between differently oriented faces.

    Returns an (n, 2, 3) array of segment endpoints.
    """
    triangles = np.asarray(triangles, dtype=float)
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(a, b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals),
                        where=lengths > 0)

    edges = {}
    for i, tri in enumerate(triangles):
        keys = [tuple(np.round(v, quantize)) for v in tri]
        for k in range(3):
            edge = tuple(sorted([keys[k], keys[(k + 1) % 3]]))
            edges.setdefault(edge, []).append(i)

    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    segments = []
    for (p0, p1), owners in edges.items():
        if len(owners) == 2:
            coplanar = abs(float(np.dot(normals[owners[0]], normals[owners[1]]))) >= cos_tol
            if coplanar:
                continue
        segments.append([p0, p1])
    return np.array(segments, dtype=float) if segments else np.zeros((0, 2, 3))


def plot_mesh_geometry_3d(triangles_by_group, source, mic, group_colors=None,
                          face_alpha=0.10, edge_alpha=0.25, elev=22, azim=-60,
                          figsize=(8, 7), exclude_groups=(), show_axes=True,
                          markers_on_top=False):
    """3D view of a triangle-mesh room with source and receiver marked.

    The mesh counterpart of _plot_3d_room, which only handles the extruded
    outlines of the parametric scenarios. `triangles_by_group` maps a
    material-group name to an (n, 3, 3) array of triangle vertices, i.e.
    exactly what pra_comparison_gewandhaus._load_mesh_triangles returns.

    The meshes use misuka's Y-up convention while matplotlib's 3D axes are
    Z-up, so every coordinate is reordered to (x, z, y) on the way in --
    relabelling the axes alone would draw the room lying on its side.

    Faces are drawn nearly transparent on purpose: source and receiver lie
    *inside* the room, and at any usable opacity the near-side walls would
    hide them. Keeping the edges visible still conveys the shape.
    """
    def to_plot_frame(points):
        points = np.asarray(points, dtype=float)
        return np.stack([points[..., 0], points[..., 2], points[..., 1]], axis=-1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if markers_on_top:
        # mplot3d depth-sorts whole artists, so at high face opacity the floor
        # is drawn over the markers standing on it and both positions vanish.
        # Disabling the computed order makes zorder authoritative; the trade
        # is that the surfaces themselves are then drawn in insertion order.
        ax.computed_zorder = False

    default_cycle = [MISUKA_COLOR, MISUKA_COLOR_ALT, MISUKA_COLOR_ALT2,
                     PRA_COLOR_ALT, "#9ecae1"]
    group_colors = group_colors or {}

    all_vertices = []
    # Groups listed in `exclude_groups` are left out entirely -- removing the
    # roof is what makes an opaque view into the hall possible at all.
    drawn = {name: tris for name, tris in triangles_by_group.items()
             if name not in exclude_groups}
    for i, (name, triangles) in enumerate(drawn.items()):
        triangles = to_plot_frame(triangles)
        all_vertices.append(triangles.reshape(-1, 3))
        color = group_colors.get(name, default_cycle[i % len(default_cycle)])
        collection = Poly3DCollection([tri for tri in triangles],
                                      facecolor=color, alpha=face_alpha,
                                      linewidth=0.3, zorder=i)
        collection.set_edgecolor((*plt.matplotlib.colors.to_rgb(color), edge_alpha))
        ax.add_collection3d(collection)
        # proxy artist: a Poly3DCollection at this alpha gives no usable
        # legend swatch
        # underscores have no glyph in cmr10 (they render as a diaeresis)
        ax.plot([], [], [], color=color, linewidth=3, alpha=0.6,
                label=f"{name.replace('_', ' ')} ({len(triangles)})")

    vertices = np.concatenate(
        [to_plot_frame(t).reshape(-1, 3) for t in triangles_by_group.values()])
    source_p, mic_p = to_plot_frame(source), to_plot_frame(mic)
    distance = float(np.linalg.norm(np.asarray(source, dtype=float)
                                    - np.asarray(mic, dtype=float)))

    marker_z = 100 if markers_on_top else None
    ax.scatter(*source_p, color="red", s=90, depthshade=False,
               edgecolor="black", linewidth=0.5, label="source", zorder=marker_z)
    ax.scatter(*mic_p, color="blue", s=90, depthshade=False, marker="^",
               edgecolor="black", linewidth=0.5, label="receiver", zorder=marker_z)
    ax.plot(*zip(source_p, mic_p), color="black", linestyle="--", linewidth=1.0,
            label=f"direct path ({distance:.2f} m)", zorder=marker_z)

    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(tuple(upper - lower))
    if show_axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_zlabel("height [m]")
    else:
        # Nothing but the geometry and the legend: axes, ticks, grid and the
        # backdrop panes are all suppressed.
        ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    # Legend outside the axes: inside it covers a good part of the room at
    # this aspect ratio, and 3D axes do not reflow around it.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(-0.08, 0.98),
              framealpha=0.9)
    # tight_layout misjudges 3D axes and clips the z label; set the margins
    # explicitly instead.
    fig.subplots_adjust(left=0.02, right=0.94, top=0.98, bottom=0.04)
    return fig, ax


def plot_mesh_sections(triangles_by_group, source, mic, group_colors=None,
                       exclude_groups=(), figsize=(9, 8), linewidth=0.7,
                       merge_coplanar=True, angle_tol_deg=5.0):
    """Technical plan and elevation of a triangle-mesh room.

    Two orthographic projections of the same mesh: a plan view (looking down
    the height axis) and a side elevation. Both are drawn as projected
    triangle edges rather than filled faces -- overlapping surfaces would
    otherwise hide the interior, and the point of a section drawing is that
    everything is visible at once.

    Coordinates follow misuka's Y-up convention: x/z span the floor, y is
    height.
    """
    from matplotlib.collections import LineCollection

    default_cycle = [MISUKA_COLOR, MISUKA_COLOR_ALT, MISUKA_COLOR_ALT2,
                     PRA_COLOR_ALT, "#9ecae1"]
    group_colors = group_colors or {}
    source = np.asarray(source, dtype=float)
    mic = np.asarray(mic, dtype=float)

    # (axis index pair, axis labels, panel title) for plan and elevation
    views = [((0, 2), ("", "z [m]"), "plan view (from above)"),
             ((0, 1), ("x [m]", "height y [m]"), "side elevation")]

    # Stacked with a shared x axis and height ratios matching the true
    # extents: at equal scale the plan is far deeper than the elevation is
    # tall, so side-by-side panels waste most of the figure. Stacking also
    # lets a feature be read off both views at the same x.
    vertices = np.concatenate([np.asarray(t).reshape(-1, 3)
                               for t in triangles_by_group.values()])
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True,
                            gridspec_kw={"height_ratios": [extent[2], extent[1]]})
    for ax, ((a, b), (xlabel, ylabel), title) in zip(axs, views):
        for i, (name, triangles) in enumerate(triangles_by_group.items()):
            if name in exclude_groups:
                continue
            triangles = np.asarray(triangles)
            color = group_colors.get(name, default_cycle[i % len(default_cycle)])
            if merge_coplanar:
                # Only the outlines of actual faces -- see _feature_edges for
                # why the triangulation seams are misleading here.
                segments = _feature_edges(triangles, angle_tol_deg)[:, :, [a, b]]
            else:
                segments = np.concatenate([
                    np.stack([triangles[:, k, [a, b]], triangles[:, (k + 1) % 3, [a, b]]], axis=1)
                    for k in range(3)])
            ax.add_collection(LineCollection(segments, colors=color,
                                             linewidths=linewidth, alpha=0.8))
            if ax is axs[0]:
                ax.plot([], [], color=color, linewidth=2,
                        label=f"{name.replace('_', ' ')} ({len(triangles)})")


        ax.plot(*zip(source[[a, b]], mic[[a, b]]), color="black",
                linestyle="--", linewidth=1.0, zorder=4)
        ax.scatter(*source[[a, b]], color="red", s=70, zorder=5,
                   edgecolor="black", linewidth=0.5)
        ax.scatter(*mic[[a, b]], color="blue", s=70, marker="^", zorder=5,
                   edgecolor="black", linewidth=0.5)

        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid(True, alpha=0.25, linewidth=0.5)

    axs[0].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.9)
    fig.tight_layout()
    return fig, axs
