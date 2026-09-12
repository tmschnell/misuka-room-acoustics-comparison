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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyfar as pf

import test_pyroomacoustics_comparison as cmp
from pra_comparison_metrics import pyrato_edc, JND_ABS, JND_REL


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

        ax.loglog(freqs, misuka_coeffs, "o-", color="C0", label="misuka (ISO 9613-1)")
        ax.loglog(freqs, pra_coeffs, "x--", color="C1", label="pra (literature table)")
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
    pra_style = {"rt_only": dict(color="C1", linestyle="--"),
                 "hybrid": dict(color="C2", linestyle=":")}

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
                             color="C0", label=misuka_label)
            axes[0].plot(t[:len(e_pra)], 10 * np.log10(np.maximum(e_pra / e_pra.max(), 1e-10)),
                         label=pra_label, **pra_style[mode_name])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            edc_m = pyrato_edc(e_misuka)
            edc_p = pyrato_edc(e_pra)
        if mode_name == "rt_only":
            pf.plot.time(edc_m, dB=True, log_prefix=10, ax=axes[1],
                         color="C0", label=misuka_label)
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
    plot_with_std_band(t, mean_db_n, std_db_n, ax=axes[0], color="C0", label="no atmosphere")
    plot_with_std_band(t[:len(mean_db_a)], mean_db_a, std_db_a, ax=axes[0], color="C3", label="inline atmosphere")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edc_n = pyrato_edc(e_no_atmo)
        edc_a = pyrato_edc(e_atmo)
    pf.plot.time(edc_n, dB=True, log_prefix=10, ax=axes[1], color="C0", label="no atmosphere")
    pf.plot.time(edc_a, dB=True, log_prefix=10, ax=axes[1], color="C3", label="inline atmosphere")

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
        (x - width, "runtime_no_atmo_ms", "no atmosphere", "C0"),
        (x, "runtime_atmo_ms", "inline atmosphere", "C3"),
        (x + width, "runtime_post_total_ms", "no atmosphere + post-process", "C2"),
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

    plot_with_std_band(t, diff_attenuation_only_pct, diff_attenuation_only_pct_std, ax=axes[0], color="C0")
    axes[0].set_title("Attenuation only\n(no attenuation vs. inline attenuation, same speed of sound)")

    plot_with_std_band(t, diff_post_vs_inline_pct, diff_post_vs_inline_pct_std, ax=axes[1], color="C2")
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

    axes[0].semilogx(spp, err_no_atmo, "o-", color="C0", label="no atmosphere")
    axes[0].semilogx(spp, err_atmo, "o-", color="C3", label="inline atmosphere")
    axes[0].set_xlabel("misuka spp (rays per frequency bin)")
    axes[0].set_ylabel("RMS EDC error vs. PRA hybrid [dB]")
    axes[0].set_title("Absolute error")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(spp, delta, "o-", color="C2")
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
    ax.semilogx(spp, rel_err_no_atmo_pct, "o-", color="C0", label="no atmosphere")
    ax.semilogx(spp, rel_err_atmo_pct, "o-", color="C3", label="inline atmosphere")
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
    cmap = plt.get_cmap("inferno")
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, n_rays in enumerate(n_show):
            vals = run["arrays"][f"misuka_window_energy_spp{n_rays}"]
            vals = vals / run["arrays"][f"misuka_window_energy_spp{n_show[-1]}"][0]
            ax.bar(np.arange(len(vals)) + i / (len(n_show) + 2), vals,
                   width=1 / (len(n_show) + 2), color=cmap(0.2 + 0.6 * i / len(n_show)),
                   label=f"N=2^{int(np.log2(n_rays))}")
    ax.hlines(ref, range(len(ref)), range(1, len(ref) + 1), color="k", lw=0.9, label="PRA-ISM")
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
    axs[0].semilogx(spp, run["arrays"]["cosine_similarity"], "o-", color="C0")
    axs[0].set_ylabel("Cosine similarity")
    axs[1].semilogx(spp, run["arrays"]["rmspe_pct"], "o-", color="C0")
    axs[1].set_ylabel("RMSPE [%]")
    axs[2].semilogx(spp, run["arrays"]["mpe_pct"], "o-", color="C0")
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
        ax.semilogy(times_ms, etc_misuka[:n] / etc_misuka[:n].max(), color="C2", label="misuka RT")
        ax.semilogy(times_ms, pra_rt_etc[:n] / pra_rt_etc[:n].max(), color="C1", linestyle="--",
                   label="PRA RT histogram", alpha=0.8)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Normalized energy")
    ax.set_xlim(0, min(times_ms[-1], 100))
    ax.legend(fontsize=8)
    ax.set_title(f"{scenario_name}: RT-only, no sinc reconstruction")
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
