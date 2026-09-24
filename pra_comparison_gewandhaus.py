"""Gewandhaus (GRAP database RNo01) import for both misuka and
pyroomacoustics -- a real, measured/RAVEN-simulated concert hall, chosen
(per the Bachelor's thesis discussion) as a second real-geometry scenario
after CR4: smaller (296 triangles vs. CR4's ~5100), and -- per Richter's
Masterarbeit -- one of the two rooms whose parameters actually converge
cleanly into the JND against RAVEN (unlike "Komische Oper", where phase
effects cause persistent deviations no ray count fixes).

Two simplifications, explicitly agreed for this integration (not full
GRAP-accuracy work, see the conversation this was built in):

1. Materials reuse the existing 4-mesh grouping from a prior misuka
   experiment (experiments/DAGA 2026/Gewandhaus/misuka/, itself built from
   the GRAP .skp via the same Blender workflow as CR4): walls, roof,
   audience, orchestra. Each mesh's coefficients are verified number-for-
   number against the official per-room CSVs in
   GRAP_DATABASE_V2/01_Gewandhaus/1 Geometry/ (walls=GH_Wall.csv,
   roof=GH_Ceiling.csv, audience=GH_Seats.csv, orchestra=GH_Organ.csv) --
   no central/root-level GRAP material database exists, so these per-room
   CSVs are the authoritative source. GH_Floor.csv and "GH_Seats organ".csv
   are unused: no separate floor/organ-seating mesh exists in this 4-mesh
   geometry. This is a real, verified 6-CSV-to-4-mesh mapping, not a fitted
   average or a guess -- the only remaining simplification is that the
   original Blender export didn't split out a separate floor/organ-seating
   mesh, not the material values themselves.

2. Source/mic are a freshly chosen, self-consistent pair -- NOT GRAP's own
   R1 (whose exact coordinates couldn't be verified from the GRAP
   deliverables; a candidate reconstruction from a likely-unrelated RAVEN
   .rpf project file gave a materially different distance). Reasonable
   distance/height, clear of every surface, identical for both simulators.

Geometry is small enough (296 triangles total) that -- unlike CR4 -- no
coplanar-triangle merging is needed: pyroomacoustics builds one Wall per
triangle directly and stays fast (ISM order=3: ~9s; a 3e6-ray hybrid
compute_rir: ~12 min -- both verified empirically, see the conversation's
sanity-check timing).
"""
import os

import numpy as np
import mitsuba as mi
import pyroomacoustics as pra

from pra_comparison_bras_cr4 import _add_despite_flaky_is_inside

MESH_DIR = ("/Users/timon/Documents/Timon/Uni/SoSe26/Bachelor-Arbeit/benchmaring/experiments/"
           "experiments/DAGA 2026/Gewandhaus/misuka/meshes")

# GRAP 1kHz coefficients, verified number-for-number against the official
# per-room CSVs in GRAP_DATABASE_V2/01_Gewandhaus/1 Geometry/ (no central/
# root-level material database exists in GRAP_DATABASE_V2 -- checked):
# walls=GH_Wall.csv, roof=GH_Ceiling.csv, audience=GH_Seats.csv,
# orchestra=GH_Organ.csv. "roof" previously (incorrectly) reused GH_Wall's
# values -- fixed here to GH_Ceiling's. GH_Floor.csv / "GH_Seats organ".csv
# are unused: no separate floor/organ-seating mesh exists in this geometry.
MATERIALS_1KHZ = {
    "walls": (0.0529, 0.1901), "roof": (0.335, 0.7501),
    "audience": (0.4173, 0.8401), "orchestra": (0.3644, 0.7501),
}

SOURCE_POS = (-8.0, 2.0, -15.0)   # inside "orchestra" footprint, 1.34m above stage floor
MIC_POS = (5.0, 3.0, -15.0)       # well inside the audience volume, clear of every surface
SOURCE_RADIUS = 0.3


def _to_xyz(p):
    try:
        return float(p.x), float(p.y), float(p.z)
    except TypeError:
        return float(p.x[0]), float(p.y[0]), float(p.z[0])


def _to_ijk(f):
    try:
        return int(f.x), int(f.y), int(f.z)
    except TypeError:
        return int(f.x[0]), int(f.y[0]), int(f.z[0])


def _load_mesh_triangles(mesh_name):
    mesh = mi.load_dict({"type": "ply", "filename": os.path.join(MESH_DIR, f"{mesh_name}.ply")})
    verts = np.array([_to_xyz(mesh.vertex_position(i)) for i in range(mesh.vertex_count())])
    faces = np.array([_to_ijk(mesh.face_indices(i)) for i in range(mesh.face_count())])
    return verts[faces]  # (n_faces, 3, 3)


def acoustic_bsdf(absorption, scattering, specular_lobe_width=0.001):
    return {
        "type": "acousticbsdf", "specular_lobe_width": specular_lobe_width,
        "absorption": {"type": "spectrum", "value": absorption},
        "scattering": {"type": "spectrum", "value": scattering},
    }


def build_misuka_scene_dict(max_time, sampling_rate, frequencies="1000.0",
                            source_pos=SOURCE_POS, mic_pos=MIC_POS, source_radius=SOURCE_RADIUS):
    scene_dict = {"type": "scene"}
    for mesh_name, (absorption, scattering) in MATERIALS_1KHZ.items():
        scene_dict[mesh_name] = {
            "type": "ply", "filename": os.path.join(MESH_DIR, f"{mesh_name}.ply"),
            "face_normals": True, "bsdf": acoustic_bsdf(absorption, scattering),
        }
    scene_dict["emitter"] = {
        "type": "sphere", "radius": source_radius, "center": list(source_pos),
        "emitter": {"type": "area", "radiance": {"type": "uniform", "value": 1.0}},
    }
    scene_dict["mic"] = {
        "type": "microphone", "origin": list(mic_pos), "direction": [1.0, 0.0, 0.0],
        "film": {"type": "tape", "frequencies": frequencies,
                 "time_bins": int(round(max_time * sampling_rate))},
    }
    return scene_dict


# =====================================================================
#   Richter Fig.14 reproduction: misuka (ours) vs. Richter's own misuka
#   results for Gewandhaus, NOT misuka-vs-PRA -- see the conversation this
#   was built in for the full derivation. Every constant below was fixed
#   against Richter's Masterarbeit Section 4.2/4.5, not left at this
#   module's misuka-vs-PRA defaults above (0.3m/1kHz/self-chosen position).
#
# - source_radius=0.1m, c=343.64 m/s: Richter Sec. 4.2/4.5, verbatim.
# - f=20Hz: Richter Sec. 4.5 evaluates only the lowest third-octave band;
#   coefficients below are each material's real 20Hz row from the GRAP
#   CSVs, not the 1kHz values MATERIALS_1KHZ uses.
# - Source/mic position: GRAP's real R1 coordinate could not be recovered
#   from any available source (GRAP CSVs/.skp, BRIR/RIR FLAC metadata --
#   0 embedded comments --, RAQI Items/Factors -- perceptual ratings, not
#   coordinates --, a nearby RAVEN .rpf project file -- gave an unrelated,
#   non-matching distance --, and no GRAP paper or database documentation
#   is available locally to confirm a seat-height convention). The position
#   below is chosen instead by a documented, non-arbitrary criterion:
#   highest line-of-sight "visibility quota" (fraction of time-matching
#   early-reflection candidate points unobstructed from both source and
#   mic) among a small grid of on-stage/main-floor candidates -- 60.5%
#   (26/43), vs. 13.3% (6/45) for this module's misuka-vs-PRA position,
#   which was found to badly self-shadow the low, orchestra-pit-edge
#   source. This is a limitation, not a substitute for the real R1: if
#   this room enters a from-scratch-accuracy argument, replace with the
#   real coordinate should it ever surface.
# - Materials: "walls" is split into two meshes (walls_actual.ply,
#   walls_floor.ply) -- 10 of "walls"'s 212 triangles (87.54 m^2) sit
#   exactly at Y=0 with a near-vertical normal (a genuine flat floor patch
#   near the stage-front, geometrically distinct from the rest of "walls",
#   which spans much greater heights) and get GH_Floor.csv's real 20Hz
#   coefficient instead of GH_Wall.csv's. This closes most of a
#   ray-count-independent G gap without disturbing C50/C80/T60/EDT (which
#   were already within JND at the new position); it does NOT close TS,
#   which moves further from Richter's reference under every absorption
#   change tested (uniform or targeted) -- an independent, still-open
#   discrepancy, not a sign the floor split is wrong.
RICHTER_SOURCE_RADIUS = 0.1
RICHTER_SPEED_OF_SOUND = 343.64
RICHTER_FREQUENCY_HZ = "20.0"
RICHTER_SOURCE_POS = (-11.2, 0.66 + 1.8, -15.0)   # stage center, away from the pit-rail edge
RICHTER_MIC_POS = (-3.0, 0.78 + 1.2, -22.0)        # main-floor audience seating
RICHTER_MATERIALS_20HZ = {
    "walls_actual": (0.2422, 0.01),   # GH_Wall.csv, 20Hz row
    "walls_floor": (0.0346, 0.01),    # GH_Floor.csv, 20Hz row -- the extracted patch
    "roof": (0.2769, 0.01),           # GH_Ceiling.csv, 20Hz row
    "audience": (0.1302, 0.01),       # GH_Seats.csv, 20Hz row
    "orchestra": (0.4803, 0.01),      # GH_Organ.csv, 20Hz row
}
# Read off Fig.14 (Masterarbeit p.33), reference (hybrid-simulation) line,
# by eye -- no digitized numeric table exists in the thesis or GRAP data.
RICHTER_FIG14_REFERENCE = dict(C50=-0.4, C80=1.8, G=7.75, EDT=2.32, T60=2.66, TS=0.123)
# T60's JND is relative (~15.6% of its reference value, see JND_REL); the
# other five are Richter's fixed absolute JNDs (see JND_ABS).
RICHTER_FIG14_JND = dict(C50=1.0, C80=1.0, G=1.0, EDT=0.21, TS=0.010)


# Each mesh of the RICHTER_* scene and the GRAP database CSV its
# coefficients come from. The CSVs (copied verbatim into
# pra_comparison_assets/gewandhaus/, same pattern as bras_cr4_v2's
# materials_1khz.json) carry all 31 third-octave bands, so a scene can be
# built at any frequency rather than only at the 20 Hz row hardcoded in
# RICHTER_MATERIALS_20HZ -- needed by the atmospheric evaluation, whose
# whole subject (ISO 9613-1 air attenuation) scales with f^2 and is
# therefore meaningless at 20 Hz.
MATERIAL_CSV_BY_MESH = {
    "walls_actual": "GH_Wall.csv",
    "walls_floor": "GH_Floor.csv",
    "roof": "GH_Ceiling.csv",
    "audience": "GH_Seats.csv",
    "orchestra": "GH_Organ.csv",
}
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "pra_comparison_assets", "gewandhaus")


def _read_material_csv(csv_name):
    """(frequencies, absorption, scattering) of one GRAP material CSV."""
    frequencies, absorption, scattering = [], [], []
    with open(os.path.join(ASSET_DIR, csv_name)) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3 or not parts[0] or not parts[0][0].isdigit():
                continue  # header
            frequencies.append(float(parts[0]))
            absorption.append(float(parts[1]))
            scattering.append(float(parts[2]))
    return np.array(frequencies), np.array(absorption), np.array(scattering)


def materials_at(frequencies):
    """Per-mesh ``(absorption, scattering)`` spectra at `frequencies`.

    Interpolated log-linearly in frequency over each material's own GRAP
    third-octave table, so a requested band that is not a tabulated centre
    (e.g. the suite's 14 kHz highlight frequency) still gets a physically
    sourced coefficient rather than a nearby band's.

    Returns, per mesh, two lists of ``(frequency, value)`` pairs -- the form
    misuka's `spectrum` accepts for a frequency-dependent BSDF coefficient.
    For a *single* frequency the two coefficients are returned as plain
    scalars instead: mitsuba builds a ContinuousDistribution from a pair
    list and that needs at least two entries, so a one-element list fails
    to load ("needs at least two entries!").
    """
    frequencies = [float(f) for f in np.atleast_1d(frequencies)]
    out = {}
    for mesh_name, csv_name in MATERIAL_CSV_BY_MESH.items():
        table_f, table_a, table_s = _read_material_csv(csv_name)
        log_f, log_table = np.log10(frequencies), np.log10(table_f)
        absorption = np.interp(log_f, log_table, table_a)
        scattering = np.interp(log_f, log_table, table_s)
        if len(frequencies) == 1:
            out[mesh_name] = (float(absorption[0]), float(scattering[0]))
        else:
            out[mesh_name] = (
                [(f, float(a)) for f, a in zip(frequencies, absorption)],
                [(f, float(s)) for f, s in zip(frequencies, scattering)],
            )
    return out


def build_misuka_scene_dict_richter(max_time, sampling_rate, frequencies=None,
                                    source_pos=None, mic_pos=None,
                                    source_radius=None):
    """The validated Gewandhaus scene (position/radius, see the block above).

    `frequencies` defaults to the 20 Hz band RICHTER_FREQUENCY_HZ pins, with
    that band's tabulated coefficients; pass explicit frequencies (a list, or
    a comma-separated string) to render other bands, in which case the
    materials are taken from the GRAP tables at those frequencies via
    materials_at(). source_pos/mic_pos/source_radius override the validated
    defaults -- used by the atmospheric distance study, which needs the same
    room and source but a different source-receiver separation.
    """
    source_pos = RICHTER_SOURCE_POS if source_pos is None else source_pos
    mic_pos = RICHTER_MIC_POS if mic_pos is None else mic_pos
    source_radius = RICHTER_SOURCE_RADIUS if source_radius is None else source_radius

    if frequencies is None:
        frequencies_str = RICHTER_FREQUENCY_HZ
        materials = {name: (a, s) for name, (a, s) in RICHTER_MATERIALS_20HZ.items()}
    else:
        if isinstance(frequencies, str):
            frequencies = [float(f) for f in frequencies.split(",")]
        frequencies = [float(f) for f in np.atleast_1d(frequencies)]
        frequencies_str = ", ".join(str(f) for f in frequencies)
        materials = materials_at(frequencies)

    scene_dict = {"type": "scene"}
    for mesh_name, (absorption, scattering) in materials.items():
        scene_dict[mesh_name] = {
            "type": "ply", "filename": os.path.join(MESH_DIR, f"{mesh_name}.ply"),
            "face_normals": True, "bsdf": acoustic_bsdf(absorption, scattering),
        }
    scene_dict["emitter"] = {
        "type": "sphere", "radius": source_radius, "center": list(source_pos),
        "emitter": {"type": "area", "radiance": {"type": "uniform", "value": 1.0}},
    }
    scene_dict["mic"] = {
        "type": "microphone", "origin": list(mic_pos), "direction": [1.0, 0.0, 0.0],
        "film": {"type": "tape", "frequencies": frequencies_str,
                 "time_bins": int(round(max_time * sampling_rate))},
    }
    return scene_dict


def build_pra_room(max_order, fs, hist_bin_size, n_rays, energy_thres, time_thres,
                   source_pos=SOURCE_POS, mic_pos=MIC_POS, source_radius=SOURCE_RADIUS):
    """One pyroomacoustics Wall per mesh triangle (296 total) -- no merging
    needed at this size, unlike CR4 (see module docstring). Wall winding is
    flipped relative to the source mesh's own normal convention (verified
    via room volume: this flip is what turns a nonsensical negative volume
    into +22325 m^3, matching mat_CR4-style documentation... well, here
    matching the independently-computed mesh volume itself, since no
    external ground truth for Gewandhaus's exact modeled volume was
    cross-checked here the way CR4's was).
    """
    walls = []
    for mesh_name, (absorption, scattering) in MATERIALS_1KHZ.items():
        mat = pra.Material(energy_absorption=absorption, scattering=scattering)
        for i, tri in enumerate(_load_mesh_triangles(mesh_name)):
            tri = tri[::-1]  # flip winding
            walls.append(pra.wall_factory(tri.T, mat.energy_absorption["coeffs"],
                                          mat.scattering["coeffs"], name=f"{mesh_name}_{i}"))

    room = pra.Room(walls, fs=fs, max_order=max_order, ray_tracing=True)
    room.set_ray_tracing(n_rays=n_rays, energy_thres=energy_thres,
                         time_thres=time_thres, hist_bin_size=hist_bin_size)
    _add_despite_flaky_is_inside(room.add_source, np.array(source_pos))
    _add_despite_flaky_is_inside(room.add_microphone, np.array(mic_pos))
    return room
