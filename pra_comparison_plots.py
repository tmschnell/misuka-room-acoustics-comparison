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


from matplotlib.patches import Rectangle
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.transforms as mtransforms
from colorspacious import cspace_convert

import test_pyroomacoustics_comparison as cmp
from pra_comparison_metrics import pyrato_edc, JND_ABS, JND_REL, jnd_band

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
MISUKA_COLOR_ALT3 = "#0c3017"
PRA_COLOR = "#ff7f0e"
PRA_COLOR_ALT = "#fdae61"


def hex_to_rgb(color):

    color = color.lstrip("#")

    return np.array([

        int(color[i:i+2], 16) / 255

        for i in (0, 2, 4)

    ])

def rgb_to_hex(rgb):

    rgb = np.clip(rgb, 0, 1)

    rgb = (rgb * 255).round().astype(int)

    return "#{:02x}{:02x}{:02x}".format(*rgb)

# ---------------------------------------------------------------------

# 1. Original colors

# ---------------------------------------------------------------------

colors = [

    MISUKA_COLOR_ALT3,

    MISUKA_COLOR_ALT,

    MISUKA_COLOR,

    MISUKA_COLOR_ALT2,

    PRA_COLOR,

    PRA_COLOR_ALT,

]

rgb = np.array([hex_to_rgb(c) for c in colors])

# RGB -> CIELAB

lab = cspace_convert(rgb, "sRGB1", "CIELab")

# ---------------------------------------------------------------------

# 2. Define where the colors should occur

# ---------------------------------------------------------------------


positions = np.array([

    0.00,

    0.1,

    0.25,

    0.55,

    0.7,

    1.00,

])

# ---------------------------------------------------------------------

# 3. Interpolate in CIELAB

# ---------------------------------------------------------------------

n = 256

x = np.linspace(0, 1, n)

lab_interp = np.empty((n, 3))

for channel in range(3):

    lab_interp[:, channel] = np.interp(

        x,

        positions,

        lab[:, channel]

    )

# ---------------------------------------------------------------------

# 4. Convert back to RGB

# ---------------------------------------------------------------------

rgb_interp = cspace_convert(

    lab_interp,

    "CIELab",

    "sRGB1"

)

rgb_interp = np.clip(rgb_interp, 0, 1)

# ---------------------------------------------------------------------

# 5. Create colormap

# ---------------------------------------------------------------------

CMAP_CHAT2 = LinearSegmentedColormap.from_list(

    "pra_misuka",

    rgb_interp

)

# Sequential map for magnitudes: JND blue (pale) -> misuka greens -> PRA
# oranges, interpolated in CIELAB at equal spacing so no hue change is abrupt.
_BGO_STOPS = ["#dde9f3", "#9ecae1", 
              MISUKA_COLOR_ALT2, 
              #MISUKA_COLOR, 
              "#b8b04a", PRA_COLOR_ALT, PRA_COLOR]
_bgo_lab = cspace_convert(np.array([hex_to_rgb(c) for c in _BGO_STOPS]), "sRGB1", "CIELab")
_bgo_x = np.linspace(0, 1, len(_BGO_STOPS))
_bgo_t = np.linspace(0, 1, 256)
_bgo = np.clip(cspace_convert(np.stack([np.interp(_bgo_t, _bgo_x, _bgo_lab[:, c]) for c in range(3)], axis=1),
                              "CIELab", "sRGB1"), 0, 1)
BLUE_GREEN_ORANGE_CMAP = LinearSegmentedColormap.from_list("jnd_misuka_pra", _bgo)

# Sequential map in the suite's own two colours, for the heat maps: PRA
# orange at the low end, misuka green at the high end. Not perceptually
# uniform the way viridis is -- the two hues meet in an olive midtone -- so
# it is offered as an alternative rather than as the default.
PRA_TO_MISUKA_CMAP = LinearSegmentedColormap.from_list(
    "pra_to_misuka", [PRA_COLOR, MISUKA_COLOR])

MISUKA_TO_PRA = LinearSegmentedColormap.from_list(
    "misuka_to_prd", [MISUKA_COLOR_ALT3, MISUKA_COLOR, MISUKA_COLOR_ALT2])

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

    ax.set_xlabel("x in m")
    ax.set_ylabel("y in m")
    ax.set_zlabel("z in m")

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
    show fewer.

    Returns ``{scenario_name: fig}`` for the caller to show/save, matching
    every other plot_* function's contract (this one used to draw and
    plt.show() each figure itself, the sole exception, since there was no
    single scenario name to key a return value on for a multi-scenario
    call -- a dict resolves that without giving up the "caller decides what
    happens to the figure" contract the rest of this module relies on).
    """
    if scenarios is None:
        scenarios = GEOMETRY_SCENARIOS

    figures = {}
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

        _plot_room_view(ax_top, scenario["top_outline"], (src[0], src[1]), (mic[0], mic[1]), "x in m", "y in m")
        _plot_room_view(ax_side, scenario["side_outline"], (src[0], src[2]), (mic[0], mic[2]), "x in m", "z in m",
                       grayline=scenario.get("grayline"))
        _plot_3d_room(ax_3d, scenario["top_outline"], scenario["height"], src, mic, aspect=scenario.get("aspect"))

        fig.tight_layout()
        figures[scenario["name"]] = fig
    return figures


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
        ax.set_xlabel("Frequency in Hz")
        ax.set_ylabel("Attenuation coefficient in 1/m")
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
    axes[0].set_xlabel("Time in s")
    axes[0].set_ylim(bottom=-105)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Schroeder EDC (pyrato, dB)")
    axes[1].set_xlabel("Time in s")
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
    axes[0].set_xlabel("Time in s")
    axes[0].set_ylim(bottom=-105)
    axes[0].legend()
    axes[1].set_title("Schroeder EDC (pyrato, dB)")
    axes[1].set_xlabel("Time in s")
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
    ax_bar.set_ylabel("Render time in ms")
    ax_bar.set_title("misuka render time: no atmosphere vs. inline vs. post-processed atmosphere")
    ax_bar.legend(fontsize=8)

    table_cols = ["scenario", "runtime_no_atmo_ms", "runtime_atmo_ms", "runtime_overhead_pct",
                 "runtime_post_total_ms", "runtime_post_overhead_pct"]
    col_labels = ["scenario", "no atmo\nin ms", "inline\nin ms", "inline\noverhead in %",
                 "post-total\nin ms", "post\noverhead in %"]
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
        ax.set_xlabel("Time in s")
        ax.set_ylabel("Relative difference in %")
        ax.set_xlim(0, t[-1])

    if has_freq_data:
        freqs = run["metadata"]["render_params"]["highlight_frequencies_hz"]
        diff_pct_abs_mean = run["arrays"]["freq_diff_pct_abs_mean"]
        diff_pct_abs_std = run["arrays"]["freq_diff_pct_abs_std"]
        for i, f in enumerate(freqs):
            plot_with_std_band(t, diff_pct_abs_mean[:, i], diff_pct_abs_std[:, i], ax=axes[2],
                               color=f"C{i}", a=0.5, label=f"{f:.0f} Hz")
        axes[2].set_title("Post-processing vs. inline attenuation\nrelative error (|%|), per frequency")
        axes[2].set_xlabel("Time in s")
        axes[2].set_ylabel("Relative difference in % (abs)")
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
    axes[0].set_ylabel("RMS EDC error vs. PRA hybrid in dB")
    axes[0].set_title("Absolute error")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(spp, delta, "o-", color=MISUKA_COLOR_ALT2)
    axes[1].set_xlabel("misuka spp (rays per frequency bin)")
    axes[1].set_ylabel("extra error from atmosphere in dB\n(inline atmosphere - no atmosphere)")
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
    ax.set_ylabel("relative RMS EDC error vs. PRA hybrid in %")
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
    fig.suptitle(f"{scenario_name} -- ISO 3382 descriptors vs. PRA hybrid, relative in %")

    metric_colors = {"T20": "C0", "T30": "C1", "C50": "C2", "D50": "C3"}
    for key, color in metric_colors.items():
        axes[0].semilogx(spp, run["arrays"][f"{key}_no_atmo_rel_pct"], "o-", color=color, label=key)
        axes[1].semilogx(spp, run["arrays"][f"{key}_atmo_rel_pct"], "o-", color=color, label=key)

    axes[0].set_title("no atmosphere")
    axes[1].set_title("inline atmosphere")
    for ax in axes:
        ax.set_xlabel("misuka spp (rays per frequency bin)")
        ax.set_ylabel("relative to PRA hybrid in %")
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
    axs[1].set_ylabel("RMSPE in %")
    axs[2].semilogx(spp, run["arrays"]["mpe_pct"], "o-", color=MISUKA_COLOR)
    axs[2].set_ylabel("MPE in %")
    axs[2].set_xlabel("misuka spp (rays per frequency bin)")
    for ax in axs:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


# =====================================================================
#           Appendix 2: Hybrid parameter level
# =====================================================================

def _three_ticks(axis, lo, hi):
    """Exactly three ticks at round values within [lo, hi]: steps of 1, 2 or
    5 x 10^k first, then 2.5, then 1.5, 3 or 4 x 10^k; falls back to
    matplotlib's choice if none fits."""
    decades = range(int(np.floor(np.log10(hi - lo))) - 2, int(np.ceil(np.log10(hi - lo))) + 1)
    for mantissas in ((1, 2, 5), (2.5,), (1.5, 3, 4)):
        for e in decades:
            for m in mantissas:
                step = m * 10.0 ** e
                ticks = np.arange(np.ceil(lo / step) * step, hi + 1e-9 * step, step)
                if len(ticks) == 3:
                    axis.set_major_locator(mticker.FixedLocator(ticks))
                    return
    axis.set_major_locator(mticker.MaxNLocator(3))


def plot_hybrid_jnd(scenario_name, run, external_reference=None, external_label="external reference",
                    params=None, pra_fixed=None, pra_fixed_per_repeat=None, reference_rays=None):
    """`external_reference` ({param: value}, e.g. GRAP's own RAVEN parameters)
    adds a further horizontal reference line per panel; params it lacks get none.
    If given, the JND band is centred on it. `pra_fixed` ({param: value}) adds
    PRA with the double-counted reflection order removed
    (cmp.drop_double_counted_ism_order), in a lighter orange. PRA's repeats
    (run's pra_<param>_per_repeat, and `pra_fixed_per_repeat`, {param: array})
    are drawn as range and +/-1 std bands like misuka's."""
    spp = run["arrays"]["spp"]
    reference_params = run["metadata"]["metrics"]["reference_params"]
    reference_rays = reference_rays or spp.max()

    UNITS = {"C50": "dB", "C80": "dB", "G": "dB", "TS": "s", "EDT": "s", "T60": "s", "T30": "s"}

    # z-order, back to front: JND band (green) < repeat range (light blue) <
    # +/-1 std (dark blue) < the actual mean/reference lines on top.
    Z_JND, Z_RANGE, Z_STD, Z_LINES = 1, 3, 4, 2

    fig, axs = plt.subplots(2, 3, figsize=(6.6, 5.0))
    if scenario_name != "Gewandhaus (125 Hz)":
        fig.suptitle(scenario_name, fontsize=22)
    for ax, param in zip(axs.flat, params or list(JND_ABS) + list(JND_REL)):
        vals = run["arrays"][f"misuka_{param}"]
        per_repeat = run["arrays"][f"misuka_{param}_per_repeat"]  # (n_repeats, n_spp)
        std = run["arrays"][f"misuka_{param}_std"]
        pra_val = reference_params[param]

        # Band centered on misuka's own converged (highest-spp) mean, not on
        # PRA's value -- same comparison as before, just read as "is PRA
        # within misuka's own tolerance" instead of the other way round.
        # Falls back to PRA's value if misuka's last value is itself
        # non-finite (e.g. G for l_room_1000m3 -- no line of sight, see
        # parameters_full's g_radius_term docstring).
        ext_val = (external_reference or {}).get(param)
        if ext_val is not None:
            band_lo, band_hi = jnd_band(param, ext_val)
        else:
            band_center = vals[-1] if np.isfinite(vals[-1]) else pra_val
            band_lo, band_hi = jnd_band(param, band_center, relative_to=pra_val)
        ax.axhspan(band_lo, band_hi, color="C0", alpha=0.15, zorder=Z_JND, label="JND")

        ax.fill_between(spp, per_repeat.min(axis=0), per_repeat.max(axis=0),
                        color="C2", alpha=0.15, zorder=Z_RANGE, label="misuka range")
        ax.fill_between(spp, vals - std, vals + std,
                        color="C2", alpha=0.35, zorder=Z_STD, label=r"misuka $\pm 1\,\sigma$")

        ax.semilogx(spp, vals, marker="o", linestyle="-", color="C2", label="misuka", zorder=Z_LINES+1, markersize=2.0)
        # PRA as small box plots at its own ray count: box +/-1 std, line the
        # range of the runs, dot the mean. Both are centred on that ray count;
        # if they overlap vertically, the boxes are cut there: hybrid keeps
        # the left half, fixed the right half, with the range lines side by
        # side on the cut and half dots for the means.
        fixed_val = (pra_fixed or {}).get(param)
        n_rays = run["metadata"].get("render_params", {}).get("pra_n_rays") or spp.max()
        hybrid_reps = run["arrays"].get(f"pra_{param}_per_repeat", [pra_val])
        fixed_reps = (pra_fixed_per_repeat or {}).get(param, None if fixed_val is None else [fixed_val])
        spread = [(np.asarray(r), color, label, side) for r, color, label, side in (
            (hybrid_reps, "C1", "PRA-hybrid", -1), (fixed_reps, PRA_COLOR_ALT, "PRA-hybrid (fixed)", +1))
            if r is not None]
        extents = [(min(r.min(), r.mean() - r.std()), max(r.max(), r.mean() + r.std())) for r, *_ in spread]
        overlap = len(extents) == 2 and extents[0][0] <= extents[1][1] and extents[1][0] <= extents[0][1]
        box_half = 2 ** 0.6
        for reps, color, label, side in spread:
            m, s = reps.mean(), reps.std()
            if overlap:
                x0, x1 = (n_rays / box_half, n_rays) if side < 0 else (n_rays, n_rays * box_half)
                # range line next to the cut, shifted by 0.8 pt into its own half
                line_transform = mtransforms.offset_copy(ax.transData, fig=fig, x=0.8 * side, units="points")
                dot_style = dict(marker=MarkerStyle("o", fillstyle="left" if side < 0 else "right"),
                                 markerfacecoloralt="none", markeredgewidth=0)
                # black outline of the same half only, an open arc on the
                # unit circle (no line along the cut)
                edge_arc = Path.arc(90, 270) if side < 0 else Path.arc(-90, 90)
            else:
                x0, x1 = n_rays / box_half, n_rays * box_half
                line_transform = ax.transData
                dot_style = dict(marker="o", markeredgecolor="black", markeredgewidth=0.5)
                edge_arc = None
            ax.add_patch(Rectangle((x0, m - s), x1 - x0, 2 * s, facecolor=color, alpha=0.6,
                                   edgecolor=color, linewidth=0.8, zorder=Z_LINES + 2))
            ax.plot([n_rays, n_rays], [reps.min(), reps.max()], color=color, linewidth=1.2,
                    transform=line_transform, zorder=Z_LINES + 3)
            ax.plot(n_rays, m, color=color, markersize=5, linestyle="none",
                    zorder=Z_LINES + 4, label=label, **dot_style)
            if edge_arc is not None:
                ax.plot(n_rays, m, marker=edge_arc, markersize=5, linestyle="none",
                        markerfacecolor="none", markeredgecolor="black",
                        markeredgewidth=0.5, zorder=Z_LINES + 4)
        if ext_val is not None:
            ax.axhline(ext_val, color="black", linestyle="--", linewidth=0.7, zorder=Z_LINES)
            # RAVEN's own result as a dot at its ray count (one point, so the
            # dashed style only shows in the legend)
            ax.plot(reference_rays, ext_val, marker="o", color="black", markersize=2.5,
                    linestyle="--", linewidth=0.7, label=external_label, zorder=Z_LINES + 4)
        ax.set_ylabel(f"{param} in {UNITS[param]}")

        if ax in axs[-1]:
            ax.set_xlabel("number of rays")
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
        # Three labelled ticks at powers of two (the thesis gives ray counts as
        # 2^n), small ticks at every power of two in between. The top row
        # keeps the ticks but drops the labels.
        lo, hi = int(np.ceil(np.log2(spp.min()))), int(np.floor(np.log2(spp.max())))
        ax.xaxis.set_major_locator(mticker.FixedLocator(2.0 ** np.round(np.linspace(lo + 2, hi - 1, 3))))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: rf"$2^{{{round(np.log2(v))}}}$"))
        ax.xaxis.set_minor_locator(mticker.FixedLocator(2.0 ** np.arange(lo, hi + 1)))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        if ax not in axs[-1]:
            ax.tick_params(axis="x", labelbottom=False)

        # y-range: PRA's value must always be visible, and the (now
        # misuka-centered) JND band should be as large a fraction of the
        # visible height as possible without exceeding 2/5. The minimal
        # span covering both the band and PRA's value sets a lower bound on
        # the total height; if PRA sits far from the band (e.g. G), that
        # bound already exceeds what a 2/5 band would need, so the band
        # ends up smaller than 2/5 -- unavoidable while keeping PRA
        # visible. If PRA is close to/inside the band, the axis is padded
        # out evenly on both sides until the band shrinks to exactly 2/5.
        band_height = band_hi - band_lo
        must_show = [v for v in (pra_val, ext_val, fixed_val) if v is not None]
        must_show += [f(r) for r, *_ in spread for f in (np.min, np.max)]
        base_lo, base_hi = min(band_lo, *must_show), max(band_hi, *must_show)
        total_height = max(base_hi - base_lo, band_height / 0.4)
        # 5% margin, so a reference line never sits exactly on the frame
        extra = total_height - (base_hi - base_lo) + 0.05 * total_height
        ax.set_ylim(base_lo - extra / 2, base_hi + extra / 2)
        _three_ticks(ax.yaxis, *ax.get_ylim())
    handles, labels = axs.flat[0].get_legend_handles_labels()
    # full dots for PRA in the legend, also when the first panel shows halves
    from matplotlib.lines import Line2D
    pra_colors = {"PRA-hybrid": "C1", "PRA-hybrid (fixed)": PRA_COLOR_ALT}
    handles = [Line2D([], [], marker="o", color=pra_colors[l], markeredgecolor="black",
                      markeredgewidth=0.5, markersize=5, linestyle="none") if l in pra_colors else h
               for h, l in zip(handles, labels)]
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.19),
               frameon=False, fontsize=10)
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
    ax.set_xlabel("Time in ms")
    ax.set_ylabel("Normalized energy")
    ax.set_xlim(0, min(times_ms[-1], 100))
    ax.legend(fontsize=8)
    ax.set_title(f"{scenario_name}: RT-only, no sinc reconstruction")
    fig.tight_layout()
    return fig


def plot_gewandhaus_grap_etc(run, xlim_s=None):
    """misuka's 20 Hz ETC (1 ms bins, scaled so the direct sound carries
    1/d^2 like GRAP's histogram) over GRAP's RNo01_R1_Histogram 20 Hz column."""
    a, m = run["arrays"], run["metadata"]["metrics"]
    t = a["time_1ms_s"]
    misuka = np.roll(a["etc_1ms_mean"], int(m.get("direct_bin_shift_1ms", 0)))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    with np.errstate(divide="ignore"):
        ax.semilogy(t, a["grap_etc_1ms"], color=PRA_COLOR, label="GRAP (RAVEN hybrid)", alpha=0.8)
        ax.semilogy(t, misuka, color=MISUKA_COLOR, label="misuka", alpha=0.8)
    ax.set_xlabel("Time in s")
    ax.set_ylabel("Energy per 1 ms bin")
    if xlim_s is not None:
        ax.set_xlim(*xlim_s)
    ax.legend(fontsize=8)
    ax.set_title("Gewandhaus R1, 20 Hz")
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
            ax.set_xlabel("sampling rate in Hz  (bin width = 1/fs)")
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
        ax.set_xlabel("Time in ms")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel(f"|post vs. inline| in %  ({window/sampling_rate_hz*1e3:.1f} ms rolling mean)")
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
    ax.set_xlabel("frequency in Hz")
    ax.set_ylabel("post vs. inline, |relative difference| in %")
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
    ax.set_xticklabels([f"{w:.3g}" for w in bin_widths_ms], fontsize=13)
    ax.set_xticks([], minor=True)

    sampling_axis = ax.secondary_xaxis(
        "top", functions=(lambda ms: 1e3 / np.where(ms == 0, np.nan, ms),
                          lambda hz: 1e3 / np.where(hz == 0, np.nan, hz)))
    sampling_axis.set_xlabel("sampling rate in Hz", fontsize=13)
    sampling_axis.set_xticks(bin_widths_hz)
    sampling_axis.set_xticklabels([f"{f:.0f}" for f in bin_widths_hz], fontsize=13)
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
    ax.set_xlabel("time-bin width in ms")
    ax.set_ylabel("distance error in m")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.45), frameon=False)
    _bin_width_x_axes(ax, bin_widths_s, bin_widths_hz)

    secondary = ax.secondary_yaxis(
        "right", functions=(lambda m: m / speed_of_sound * 1e3,
                            lambda s: s * speed_of_sound / 1e3))
    secondary.set_ylabel("time error in ms", fontsize=12)
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
    # if np.nanmax(max_abs_pct) >= 99.9:
    #     ax.axhline(100.0, linestyle="--", linewidth=0.8, color="0.6")
    #     ax.text(bin_widths_s[0] * 1e3, 100.0, " metric bound (100 %)", fontsize=7,
    #             va="bottom", ha="left", color="0.4")
    ax.set_yscale("log")
    ax.set_xlabel("time-bin width in ms", fontsize=12)
    strd="post vs. inline, " + r'$|$' + "relative difference" +r'$|$' + " in %"
    ax.set_ylabel(strd)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.45), frameon=False)
    _bin_width_x_axes(ax, bin_widths_s, bin_widths_hz)
    fig.tight_layout()
    return fig


def plot_attenuation_bin_width_eps(scenario_name, run, show_noise=False):
    """Energy-weighted error eps = sum|H_post - H_inline| / sum H_inline (in %)
    of postprocessing against inline attenuation over the time-bin width,
    with the distance of the bin start (solid, filled) and of the bin centre
    (dashed, open), mean +/- std over the seeds. `show_noise` adds eps between
    two inline renders with different seeds (dotted), the Monte Carlo noise of
    one render on the same scale."""
    from matplotlib.lines import Line2D
    a = run["arrays"]
    bin_widths_ms = a["bin_widths_s"] * 1e3
    frequencies = a["frequencies_hz"]
    colors = frequency_colors(frequencies)
    styles = [("eps_pct", dict(linestyle="-"), "bin start"),
              ("eps_pct_centre", dict(linestyle="--", markerfacecolor="white"), "bin center")]
    if show_noise:
        styles.append(("eps_noise_pct", dict(linestyle=":", marker="^", markerfacecolor="white"),
                       "Monte Carlo noise"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for key, style, _ in styles:
        for j, color in enumerate(colors):
            ax.errorbar(bin_widths_ms, a[key][:, j], yerr=a[key + "_std"][:, j], color=color,
                        **{"marker": "o", "markersize": 4, "capsize": 2, **style})
    ax.set_yscale("log")
    ax.set_xlabel("time bin width in ms", fontsize=13)
    ax.set_ylabel(r"error $\varepsilon$ in %", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    handles = [Line2D([], [], color=c, linewidth=6) for c in colors]
    labels = [f"{f:.0f} Hz" for f in frequencies]
    for _, style, label in styles:
        handles.append(Line2D([], [], color="black", **{"marker": "o", "markersize": 4, **style}))
        labels.append(label)
    ax.legend(handles, labels, fontsize=12, ncol=6, loc="lower center",
              bbox_to_anchor=(0.5, -0.27), frameon=False)
    _bin_width_x_axes(ax, a["bin_widths_s"], a["bin_widths_hz"])
    #fig.tight_layout()
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
    labels = ["no attenuation", "inline", "postprocessing"]
    colors = [MISUKA_COLOR_ALT2, MISUKA_COLOR, MISUKA_COLOR_ALT]
    samples = [np.asarray(run["arrays"][key], dtype=float) for key in keys]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    parts = ax.violinplot(samples, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.35)
    ax.boxplot(samples, widths=0.12, showfliers=False,
               medianprops=dict(color="black"))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("runtime in ms", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    metrics = run["metadata"]["metrics"]
    p_value = metrics.get("inline_vs_none_p")
    if p_value is not None:
        test_name = metrics.get("inline_vs_none_test", "paired test")
        verdict = "significant" if p_value < 0.05 else "not significant"
        ax.set_title(f"inline vs. no attenuation: p = {p_value:.3f} "
                     f"({test_name}, {verdict}, n = {metrics.get('n_repeats')})",
                     fontsize=14)
    fig.tight_layout()
    return fig


def plot_speed_of_sound_extremes(scenario_name, run, cmap='viridis'):
    """Where `cramer` / `ideal_gas` drift furthest from the ISO 9613-1
    (temperature-only) speed of sound, over a realistic indoor range.

    One panel per method: |method - ISO| over temperature and relative
    humidity at that method's worst-case CO2 (`ideal_gas` ignores CO2). Both
    panels share one colour scale and one colour bar, so the two methods can
    be compared directly.
    """
    metrics = run["metadata"]["metrics"]
    temperatures = run["arrays"]["temperatures_c"]
    humidities = run["arrays"]["relative_humidities"]
    methods = [m for m in ("cramer", "ideal_gas")
               if f"difference_grid_{m}" in run["arrays"]]
    vmax = max(float(np.max(run["arrays"][f"difference_grid_{m}"])) for m in methods)

    fig, axs = plt.subplots(1, len(methods), figsize=(2.7 * len(methods) + 0.6, 3.1),
                            squeeze=False, sharey=True)
    for ax, method in zip(axs[0], methods):
        grid = run["arrays"][f"difference_grid_{method}"]
        mesh = ax.pcolormesh(humidities, temperatures, grid, shading="auto",
                             cmap=cmap, vmin=0.0, vmax=vmax)
        ax.grid(False)
        ax.set_xlabel("relative humidity")
        title = (f"{method.replace('_', ' ')} vs. ISO 9613-1\n"
                 f"max {metrics[f'{method}_max_abs_difference_ms']:.3f} m/s")
        if method == "cramer":
            title += f" at {metrics[f'{method}_co2_ppm']:.0f} ppm CO$_2$"
        ax.set_title(title, fontsize=11)
    axs[0][0].set_ylabel(r"temperature in $^\circ$C")
    fig.tight_layout()
    fig.colorbar(mesh, ax=axs[0].tolist(), location="right", fraction=0.05, pad=0.03,
                 label=r"$|c_{method} - c_{ISO}|$ in m/s")
    return fig


# cmr10 carries no degree glyph (same reason axes.unicode_minus is off
# suite-wide), so the temperature label goes through mathtext.
INVERSE_PARAMETER_LABELS = {
    "temperature": r"temperature in $^\circ$C",
    "temperature_minus10": r"temperature in $^\circ$C",
    "relative_humidity": "relative humidity",
}


def plot_inverse_rendering(scenario_name, run, parameter_label=None,
                           etc_xlim=None,
                           etc_dynamic_range_db=80.0, etc_floor_db=-100.0,
                           etc_zoom=None,
                           etc_zoom_inset_bounds=(0.52, 0.50, 0.44, 0.44),
                           etc_zoom_span_fraction=0.06,
                           etc_zoom_center=0.1):
    """The tutorial's three-panel inverse-rendering figure, extensible.

    Panels, left to right: the retrieved parameter over the iterations
    (with the goal as a reference line), the loss over the iterations, and
    the reconstructed ETC against the target ETC. Laid out as a 2 x 2 grid
    of equal panels: parameter and loss on top, the ETC bottom left and the
    legend in the free slot bottom right.
    """
    arrays, metrics = run["arrays"], run["metadata"]["metrics"]

    # bottom row lower than the top row, just high enough for the legend; a
    # fixed gap between the two top panels keeps the loss label clear of the
    # parameter panel
    fig, grid = plt.subplots(2, 2, figsize=(6.6, 4.2), layout="constrained",
                             gridspec_kw={"height_ratios": (1.35, 1)})
    fig.get_layout_engine().set(w_pad=0.15, wspace=0.06)
    axs = [grid[0, 0], grid[0, 1], grid[1, 0]]
    legend_ax = grid[1, 1]

    values = arrays["values"]
    goal = metrics["goal"]
    if parameter_label is None:
        variant = run["metadata"].get("variant") or ""
        parameter_label = INVERSE_PARAMETER_LABELS.get(
            variant, INVERSE_PARAMETER_LABELS.get(variant.split("_")[0], "parameter"))
    label = parameter_label

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
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel(label)

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
    # Less than two decades: label 1-2-5 steps in plain numbers, the default
    # labels every minor tick in scientific notation.
    lo, hi = axs[1].get_ylim()
    if np.log10(hi / lo) < 2:
        axs[1].yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        axs[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axs[1].yaxis.set_minor_formatter(mticker.NullFormatter())



    times = arrays["times_s"]
    with np.errstate(divide="ignore"):
        etc_final_db = 10 * np.nan_to_num(np.log10(np.abs(arrays["etc_final"])), neginf=-300)
        etc_ref_db = 10 * np.nan_to_num(np.log10(np.abs(arrays["etc_reference"])), neginf=-300)
    axs[2].plot(times, etc_ref_db, color=PRA_COLOR, linewidth=1.8, label="target")
    axs[2].plot(times, etc_final_db, color=MISUKA_COLOR, linewidth=0.8,
                alpha=0.9, label="reconstructed")
    axs[2].set_xlabel("time in s")
    axs[2].set_ylabel("energy in dB")
    if etc_xlim is not None:
        axs[2].set_xlim(*etc_xlim)
    # Empty time bins land at the -300 dB floor above; without an explicit
    # limit they compress the actual decay into a few pixels.
    finite_top = np.max(etc_ref_db)
    # The lower limit is fixed (etc_floor_db); below it the ETCs are noise.
    axs[2].set_ylim(etc_floor_db, finite_top + 5)
    _three_ticks(axs[2].yaxis, *axs[2].get_ylim())

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

    # One legend for all three panels, in the free slot of the grid.
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center left", bbox_to_anchor=(-0.05, 0.46), frameon=False, fontsize=11)

    return fig


def inverse_convergence_table(records):
    """Final values, final change and runtime across inverse-rendering runs
    (the distance/frequency variants of block 3b, or any other set)."""
    rows = []
    for name, metrics in records:
        rows.append({
            "run": name,
            "goal": metrics.get("goal"),
            "final": metrics.get("final_value"),
            "abs. error": metrics.get("absolute_error"),
            "final std": metrics.get("final_value_std"),
            "final change (max)": metrics.get("final_change_max"),
            "distance in m": metrics.get("distance_m"),
            "frequency in Hz": metrics.get("frequency_hz"),
            "runtime per run in s": metrics.get("mean_runtime_per_seed_s"),
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


EXTRA_RECEIVER_COLORS = ["#6baed6", "#fd8d3c", "#756bb1"]


def content_bbox(fig, pad_inches=0.03):
    """Bounding box (inches) of everything drawn on `fig` that is not white,
    for savefig(bbox_inches=...). bbox_inches="tight" keeps the empty extent
    of 3D axes around the model; this crops to what is actually visible."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    ys, xs = np.nonzero((rgba[..., :3] < 250).any(axis=-1) & (rgba[..., 3] > 0))
    h, dpi = rgba.shape[0], fig.dpi
    return Bbox([[xs.min() / dpi - pad_inches, (h - ys.max() - 1) / dpi - pad_inches],
                 [(xs.max() + 1) / dpi + pad_inches, (h - ys.min()) / dpi + pad_inches]])


def plot_mesh_geometry_3d(triangles_by_group, source, mic, group_colors=None,
                          face_alpha=0.10, edge_alpha=0.25, elev=22, azim=-60,
                          figsize=(8, 7), exclude_groups=(), show_axes=True,
                          markers_on_top=False, extra_receivers=None):
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

    `extra_receivers` ({label: position}) draws further receivers, each with
    its own dashed path from the source.
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
                     PRA_COLOR_ALT, "#9ecae1", "#9e9ac8"]
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

    #marker_z = 100 if markers_on_top else None
    ax.scatter(*source_p, color="red", s=90, depthshade=False,
               edgecolor="black", linewidth=0.5, label="source", zorder=100)
    ax.scatter(*mic_p, color="blue", s=90, depthshade=False, marker="o",
               edgecolor="black", linewidth=0.5, label="receiver", zorder=100)
    ax.plot(*zip(source_p, mic_p), color="black", linestyle="--", linewidth=1.0,
            label=f"direct path ({distance:.2f} m)", zorder=100-10)
    for (label, point), color in zip((extra_receivers or {}).items(), EXTRA_RECEIVER_COLORS):
        point_p = to_plot_frame(point)
        extra_distance = float(np.linalg.norm(np.asarray(source, dtype=float) - np.asarray(point, dtype=float)))
        ax.scatter(*point_p, color=color, s=90, depthshade=False, marker="o",
                   edgecolor="black", linewidth=0.5, label=label, zorder=100)
        ax.plot(*zip(source_p, point_p), color=color, linestyle="--", linewidth=1.0,
                label=f"path ({extra_distance:.2f} m)", zorder=100-20)

    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(tuple(upper - lower))
    if show_axes:
        ax.set_xlabel("x in m")
        ax.set_ylabel("z in m")
        ax.set_zlabel("height in m")
    else:
        # Nothing but the geometry and the legend: axes, ticks, grid and the
        # backdrop panes are all suppressed.
        ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    # The white axes background reaches beyond the drawn room; Inkscape would
    # count it into the drawing and shift a cropped figure off centre.
    ax.patch.set_visible(False)
    # tight_layout misjudges 3D axes and clips the z label; set the margins
    # explicitly instead.
    fig.subplots_adjust(left=0.02, right=0.94, top=0.98, bottom=0.04)
    # Legend outside the axes: inside it covers a good part of the room at
    # this aspect ratio, and 3D axes do not reflow around it. Centred under
    # the drawn room, which does not sit in the middle of its 3D axes.
    drawn_box = content_bbox(fig, pad_inches=0)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=10,
              bbox_to_anchor=((drawn_box.x0 + drawn_box.x1) / 2 / fig.get_figwidth(),
                              drawn_box.y0 / fig.get_figheight() - 0.01),
              bbox_transform=fig.transFigure)
    return fig, ax


def plot_mesh_sections(triangles_by_group, source, mic, group_colors=None,
                       exclude_groups=(), figsize=(9, 8), linewidth=0.7,
                       merge_coplanar=True, angle_tol_deg=5.0, extra_receivers=None):
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
                     PRA_COLOR_ALT, "#9ecae1", "#9e9ac8"]
    group_colors = group_colors or {}
    source = np.asarray(source, dtype=float)
    mic = np.asarray(mic, dtype=float)

    # (axis index pair, axis labels, panel title) for plan and elevation
    views = [((0, 2), ("", "z in m"), ""),
             ((0, 1), ("x in m", "height y in m"), "")]

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
        ax.scatter(*mic[[a, b]], color="blue", s=70, marker="o", zorder=5,
                   edgecolor="black", linewidth=0.5)
        for (label, point), color in zip((extra_receivers or {}).items(), EXTRA_RECEIVER_COLORS):
            point = np.asarray(point, dtype=float)
            ax.plot(*zip(source[[a, b]], point[[a, b]]), color=color,
                    linestyle="--", linewidth=1.0, zorder=4)
            ax.scatter(*point[[a, b]], color=color, s=70, marker="o", zorder=5,
                       edgecolor="black", linewidth=0.5,
                       label=label if ax is axs[0] else None)

        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid(True, alpha=0.25, linewidth=0.5)

    handles, lables = axs[0].get_legend_handles_labels()

    fig.legend(handles=handles, labels=lables, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=10)
    fig.tight_layout()
    return fig, axs


# =====================================================================
#   Joint (T, RH) retrieval and retrieval from the RAVEN reference
#   (atmospheric_evaluation blocks 3d / 3e)
# =====================================================================

def plot_inverse_joint(run, title=None, goal_label="goal", cmap="viridis"):
    """Loss landscape over (T, RH) with the optimiser trajectories from every
    start, the iso-attenuation curves through the goal (one per band), and the
    retrieved T and RH over the iterations. Runs from the RAVEN retrieval get a
    fourth panel: misuka's final decay against GRAP's, level-aligned per band."""
    a, m = run["arrays"], run["metadata"]["metrics"]
    T, RH = a["landscape_temperature"], a["landscape_relative_humidity"]
    traj, freqs = a["trajectories"], a["frequencies_hz"]
    has_grap = "grap_10ms" in a
    n = 4 if has_grap else 3
    fig, axs = plt.subplots(1, n, figsize=(4.4 * n, 4.2), layout="constrained",
                            gridspec_kw={"width_ratios": [1.5] + [1] * (n - 1)})
    ax = axs[0]
    with np.errstate(divide="ignore"):
        z = np.log10(a["landscape"].T)
    mesh = ax.pcolormesh(T, RH, z, cmap=cmap, shading="auto")
    fig.colorbar(mesh, ax=ax, label=r"$\log_{10}$ loss")
    styles = ["--", ":", "-."]
    for j, f in enumerate(freqs):
        cs = ax.contour(T, RH, a["alpha_grid"][:, :, j].T, levels=[a["alpha_goal"][j]],
                        colors="white", linestyles=styles[j % 3], linewidths=1.2)
        ax.plot([], [], color="0.3", linestyle=styles[j % 3], label=f"equal attenuation, {f/1e3:g} kHz")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for k in range(traj.shape[0]):
        ax.plot(traj[k, :, 0], traj[k, :, 1], color=colors[k % 10], linewidth=1.2)
        ax.plot(*a["starts"][k], "o", color=colors[k % 10], markersize=5, markeredgecolor="k")
        ax.plot(*traj[k, -1], "X", color=colors[k % 10], markersize=8, markeredgecolor="k")
    ax.plot(m["goal_temperature"], m["goal_relative_humidity"], "*", color="red", markersize=14,
            markeredgecolor="k", label=goal_label)
    ax.set_xlabel(r"temperature in $^\circ$C")
    ax.set_ylabel("relative humidity")
    ax.set_xlim(T[0], T[-1]); ax.set_ylim(RH[0], RH[-1])
    ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
    it = np.arange(traj.shape[1])
    for ax, c, label, goal in ((axs[1], 0, r"temperature in $^\circ$C", m["goal_temperature"]),
                               (axs[2], 1, "relative humidity", m["goal_relative_humidity"])):
        for k in range(traj.shape[0]):
            ax.plot(it, traj[k, :, c], color=colors[k % 10], linewidth=1.0)
        ax.axhline(goal, color="red", linestyle="dotted", label=goal_label)
        ax.set_xlabel("iteration"); ax.set_ylabel(label)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    if has_grap:
        ax = axs[3]
        grap, mask = a["grap_10ms"], a["window_mask"].astype(bool)
        best = int(np.argmin(a["losses"][:, -1]))
        e = np.asarray(a["etc_final_per_start"][best]).reshape(grap.shape[0], -1)
        t = (np.arange(grap.shape[0]) + 0.5) * 0.010
        for j, f in enumerate(freqs):
            k = mask[:, j]
            with np.errstate(divide="ignore"):
                g_db, e_db = 10 * np.log10(grap[:, j]), 10 * np.log10(e[:, j])
            shift = np.mean(e_db[k] - g_db[k])
            ax.plot(t[k], g_db[k], color="k", linewidth=1.0, label="GRAP (RAVEN)" if j == 0 else None)
            ax.plot(t[k], e_db[k] - shift, color=colors[j], linewidth=1.0, alpha=0.85,
                    label=f"misuka, {f/1e3:g} kHz")
        ax.set_xlabel("time in s"); ax.set_ylabel("energy per 10 ms bin in dB")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
        ax.set_title("best start, level-aligned", fontsize=9)
    if title:
        fig.suptitle(title)
    return fig


# =====================================================================
#   CHORAS results (125 Hz)
# =====================================================================

def plot_window_energy(windows, results, labels=None):
    """Energy per time window relative to a reference, in dB, one line per
    method over the windows (categorical x-axis). The dashed line marks the
    reference.

    windows : list of window labels; results : {method: list of dB values}
    Ticks mark the window boundaries; labels and points sit in the middle of
    their window.
    """
    method_colors = {"misuka": MISUKA_COLOR, "PRA": PRA_COLOR, "PRA fixed": PRA_COLOR_ALT}
    x = np.arange(len(windows)) + 0.5
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, zorder=1, label="reference (GRAP)")
    for m, y in results.items():
        ax.plot(x, y, color=method_colors.get(m, "C0"), marker="o", markersize=7,
                markeredgecolor="black", markeredgewidth=0.6, linewidth=1.5, zorder=3,
                label=labels.get(m, m) if labels else m)
    ax.set_xticks(np.arange(len(windows) + 1), [""] * (len(windows) + 1))
    ax.set_xticks(x, windows, minor=True, fontsize=9)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(0, len(windows))
    ax.set_xlabel("time window after the direct sound in ms")
    ax.set_ylabel("energy relative to GRAP in dB")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(loc="outside lower center", ncol=len(results) + 1, frameon=False, fontsize=10)
    return fig


def plot_scattering_sweep(scattering, results, reference, jnd, units, labels=None, reference_at=None):
    """Room acoustic parameters over the scattering coefficient, one panel per
    parameter and one line per method, with the reference (dashed) and its JND
    band. `reference_at` marks the scattering coefficient the reference
    corresponds to (dotted vertical line).

    scattering : list of coefficients; results : {method: {param: list}}
    """
    params = list(reference)
    method_colors = {"misuka": MISUKA_COLOR, "PRA": PRA_COLOR, "PRA fixed": PRA_COLOR_ALT}
    ncols = 3
    nrows = len(params) // ncols + 1  # one slot left over for the legend
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.6, 2.25 * nrows), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12)  # small gap between the rows
    axs = axs.ravel()
    for ax, p in zip(axs, params):
        ref, band = reference[p], jnd[p]
        ax.axhspan(ref - band, ref + band, color="C0", alpha=0.15, linewidth=0, zorder=0)
        ax.axhline(ref, color="black", linestyle="--", linewidth=1.2, zorder=1)
        if reference_at is not None:
            ax.axvline(reference_at, color="black", linestyle=":", linewidth=1.0, zorder=1)
        for m, vals in results.items():
            ax.plot(scattering, vals[p], color=method_colors.get(m, "C0"), marker="o", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.6, linewidth=1.5, zorder=3)
        ax.set_title(f"{p} in {units[p]}")
        ax.set_xlabel("scattering coefficient", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.15),
               plt.Line2D([], [], color="black", linestyle="--")] + \
              [plt.Line2D([], [], color=method_colors.get(m, "C0"), marker="o", markersize=6,
                          markeredgecolor="black", markeredgewidth=0.6) for m in results]
    names = ["JND", "reference (RAVEN)"] + [labels.get(m, m) if labels else m for m in results]
    for ax in axs[len(params):]:
        ax.axis("off")
    axs[-1].legend(handles, names, loc="center", frameon=False, fontsize=13)
    return fig


def plot_choras_parameters(rows, units, labels=None, colors=None, reference_label="reference (RAVEN)",
                           hollow=()):
    """One panel per room acoustic parameter and frequency band: the value each
    CHORAS method reports (markers in the suite's method colours), the reference
    (black, dashed) and its JND band (blue, as in plot_hybrid_jnd). Each y-axis
    is zoomed to the values and the band, so that "within the JND" can be read
    off directly.

    rows : {band_label: (results, reference, jnd)}, one figure row per band, with
           results {method: {param: value}} and reference / jnd {param: value};
    units : {param: unit}; colors : {method: colour}, overrides the defaults;
    hollow : methods drawn with an open marker and a dashed stem
    """
    first_results, first_reference, _ = next(iter(rows.values()))
    params, methods = list(first_reference), list(first_results)
    method_colors = {"misuka": MISUKA_COLOR, "PRA": PRA_COLOR, "PRA fixed": PRA_COLOR_ALT, **(colors or {})}
    colors = [method_colors.get(m, "C0") for m in methods]
    names = [labels.get(m, m) if labels else m for m in methods]
    fig, axs = plt.subplots(len(params), len(rows), figsize=(6.6, 1.45 * len(params) + 0.5),
                            layout="constrained", squeeze=False)
    x = np.arange(len(methods))
    for r, (band_label, (results, reference, jnd)) in enumerate(rows.items()):
        for c, p in enumerate(params):
            ax = axs[c, r]
            ref, band = reference[p], jnd[p]
            ax.axhspan(ref - band, ref + band, color="C0", alpha=0.15, linewidth=0, zorder=0)
            ax.axhline(ref, color="black", linestyle="--", linewidth=1.2, zorder=1)
            values = [results[m][p] for m in methods]
            for xi, v, col, m in zip(x, values, colors, methods):
                open_ = m in hollow
                ax.plot([xi, xi], [ref, v], color=col, linewidth=1.5, zorder=2,
                        linestyle="--" if open_ else "-")
                ax.plot(xi, v, "o", color=col, markersize=9, zorder=3,
                        markerfacecolor="white" if open_ else col,
                        markeredgecolor=col if open_ else "black", markeredgewidth=1.5 if open_ else 0.6)
                ax.annotate(f"{v:.3g}", (xi, v), textcoords="offset points", xytext=(7, -3),
                            va="center", fontsize=10, color="black", zorder=4,
                            bbox=dict(boxstyle="square,pad=0.05", facecolor="white",
                                      edgecolor="none", alpha=0.7))
            lo, hi = min(min(values), ref - band), max(max(values), ref + band)
            pad = 0.12 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.6, len(methods) - 0.2)
            if c == len(params) - 1:
                ax.set_xticks(x, names, fontsize=10, rotation=30, ha="right", rotation_mode="anchor")
            else:
                ax.set_xticks(x, [""] * len(methods))
            if c == 0:
                ax.set_title(band_label.replace("\n", ", "))
            if r == 0:
                ax.set_ylabel(f"{p} in {units[p]}")
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.15),
               plt.Line2D([], [], color="black", linestyle="--")] + \
              [plt.Line2D([], [], marker="o", color=col, linestyle="", markersize=8,
                          markerfacecolor="white" if m in hollow else col,
                          markeredgecolor=col if m in hollow else "black",
                          markeredgewidth=1.5 if m in hollow else 0.6) for col, m in zip(colors, methods)]
    fig.legend(handles, ["JND", reference_label] + names, loc="outside lower center",
               ncol=3, frameon=False, fontsize=10)
    return fig
