"""Storage layer for the misuka-vs-pyroomacoustics comparison suite.

One "run" (one comparison result) = one directory under RESULTS_ROOT holding:
  - metadata.json: everything needed to interpret the result without
    re-running anything -- timestamp, misuka variant string, render
    parameters (spp, PRA n_rays/mode, seed, repeat count, ...), room geometry
    (dimensions, source/mic position, source_radius, absorption/scattering),
    and scalar metrics (T60, C50, edc_correlation, ...).
  - arrays.npz (only written if there are any arrays): the raw numpy arrays
    a plot needs (ETC/EDC curves, per-spp sweeps, ...). Kept separate from
    metadata.json so the (small, diffable, human-readable) metadata can be
    inspected/grepped without touching a binary file.

Directory names are deterministic (section/scenario/variant, no timestamp),
so re-running the runner refreshes a result in place rather than
accumulating an ever-growing history -- this mirrors what used to be a
plain in-memory dict (`raw[(scenario, mode)]`, etc.) in the notebook, now
just persisted to disk instead of living only in a kernel's memory.

Format is intentionally loose/additive for forward compatibility: new
metadata keys can be added at any time without touching old result
directories (readers use `dict.get(...)`, never assume a key exists), and a
`format_version` field is included so a future breaking change has
somewhere to branch on.
"""

import json
import os
import re

import numpy as np

FORMAT_VERSION = "1.0"

# Named distinctly from "pra_comparison_results" -- that name is already an
# existing convention for *plot* output (test_pyroomacoustics_comparison.py's
# own __main__ block writes PNGs there, plot_scenario_geometries.py writes
# SVGs there); this directory holds structured run data instead.
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pra_comparison_data")


def _slug(value):
    """Filesystem-safe fragment for a directory name component."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_") or "_"


def _variant_slug(variant):
    """`variant` identifies a run within (section, scenario), e.g. a PRA mode
    name, an spp value, or a dict of such keys for a multi-axis sweep. None
    means "the only run for this (section, scenario)".
    """
    if variant is None:
        return None
    if isinstance(variant, dict):
        return "_".join(f"{_slug(k)}-{_slug(v)}" for k, v in sorted(variant.items()))
    return _slug(variant)


def run_dir(section, scenario, variant=None):
    parts = [_slug(section), _slug(scenario)]
    variant_slug = _variant_slug(variant)
    if variant_slug is not None:
        parts.append(variant_slug)
    return os.path.join(RESULTS_ROOT, "__".join(parts))


def save_run(section, scenario, *, metrics=None, arrays=None, room=None,
             render_params=None, variant=None, extra=None, misuka_variant=None,
             timestamp=None):
    """Persist one comparison result.

    Parameters
    ----------
    section : str
        Which comparison this is, e.g. "main_comparison", "atmosphere",
        "ray_count_convergence", "appendix_ism_peak" -- see
        pra_comparison_runner.py for the fixed set of section names in use.
    scenario : str
        Scenario name (matches `cmp`'s own scenario["name"]).
    metrics : dict, optional
        Scalar results (floats/ints/strings/lists of those) -- goes straight
        into metadata.json, so it stays greppable without loading the npz.
    arrays : dict[str, np.ndarray], optional
        Raw arrays needed to reproduce a plot (ETC curves, per-spp sweeps,
        ...) -- written to arrays.npz.
    room : dict, optional
        Geometry/material metadata: dim, source_pos, mic_pos, source_radius,
        absorption, scattering, max_time, ...
    render_params : dict, optional
        Everything about *how* the render was produced: misuka spp, PRA
        n_rays, PRA mode/order, seed, repeat count, sampling_rate,
        speed_of_sound, ...
    variant : str or dict, optional
        Distinguishes multiple runs for the same (section, scenario), e.g.
        {"pra_mode": "hybrid"} or {"n_rays": 65536}. Stored in metadata too.
    extra : dict, optional
        Anything else worth keeping that doesn't fit the categories above.
    misuka_variant : str, optional
        `mi.variant()` at render time. Defaults to introspecting the
        currently-loaded mitsuba module if not given.
    timestamp : str, optional
        ISO 8601 timestamp. Defaults to now (UTC).

    Returns
    -------
    str : the run's directory path.
    """
    if misuka_variant is None:
        try:
            import mitsuba as mi
            misuka_variant = mi.variant()
        except Exception:
            misuka_variant = None
    if timestamp is None:
        import datetime
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    path = run_dir(section, scenario, variant)
    os.makedirs(path, exist_ok=True)

    metadata = {
        "format_version": FORMAT_VERSION,
        "section": section,
        "scenario": scenario,
        "variant": variant,
        "timestamp": timestamp,
        "misuka_variant": misuka_variant,
        "render_params": render_params or {},
        "room": room or {},
        "metrics": metrics or {},
        "extra": extra or {},
        "array_names": sorted(arrays.keys()) if arrays else [],
    }

    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    arrays_path = os.path.join(path, "arrays.npz")
    if arrays:
        np.savez(arrays_path, **arrays)
    elif os.path.exists(arrays_path):
        # A previous run at this same (section, scenario, variant) had
        # arrays but this one doesn't -- don't leave a stale npz behind.
        os.remove(arrays_path)

    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_run(section, scenario, variant=None):
    """Load one previously-saved result.

    Returns
    -------
    dict with keys "metadata" (the full metadata.json dict) and "arrays"
    (dict[str, np.ndarray], empty if this run had none).
    """
    path = run_dir(section, scenario, variant)
    with open(os.path.join(path, "metadata.json")) as f:
        metadata = json.load(f)

    arrays = {}
    arrays_path = os.path.join(path, "arrays.npz")
    if os.path.exists(arrays_path):
        with np.load(arrays_path) as npz:
            arrays = {k: npz[k] for k in npz.files}

    return {"metadata": metadata, "arrays": arrays}


def list_runs(section=None):
    """List metadata (only) for every saved run, optionally filtered by
    section. Does not load arrays.npz -- use load_run for that once you know
    which run you want.
    """
    if not os.path.isdir(RESULTS_ROOT):
        return []
    runs = []
    for name in sorted(os.listdir(RESULTS_ROOT)):
        meta_path = os.path.join(RESULTS_ROOT, name, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            metadata = json.load(f)
        if section is not None and metadata.get("section") != section:
            continue
        runs.append(metadata)
    return runs
