"""Gewandhaus (GRAP database RNo01) import for both misuka and
pyroomacoustics -- a real, measured/RAVEN-simulated concert hall, chosen
(per the Bachelor's thesis discussion) as a second real-geometry scenario
after CR4: smaller (148 triangles vs. CR4's ~5100), and -- per Richter's
Masterarbeit -- one of the two rooms whose parameters actually converge
cleanly into the JND against RAVEN (unlike "Komische Oper", where phase
effects cause persistent deviations no ray count fixes).

There is exactly ONE validated Gewandhaus configuration: GEWANDHAUS_CANONICAL
below -- GRAP's own geometry with all six materials (one mesh each), GRAP's
real S1/R1 positions (both read from RNo01_GH_Gewandhaus.skp), f=20Hz with
each material's GRAP 20Hz coefficients, source_radius=0.1m, c=343.99 m/s
(derived from GRAP's direct-sound delay), no atmosphere. Every Gewandhaus
render in this suite is expected to use it -- build_pra_room()/
build_misuka_scene_dict_richter() default to it and print what they actually
used before rendering (see _print_gewandhaus_header), and the misuka-vs-PRA
scenario (scenario_gewandhaus() in test_pyroomacoustics_comparison.py)
enforces it via require_canonical=True.

Superseded configurations (removed, so none can be selected again):
1kHz materials / 0.3m radius / a self-chosen, badly self-shadowed position
(its 2026-09-24/25 50-repeat run is NOT relevant); then a 5-group geometry
from the ISMRA RAVEN .ac (simplified: 22325 m^3 instead of GRAP's 22051 m^3,
missing several of GRAP's early reflections) with a floor split and a
position chosen by line-of-sight visibility quota, because the real R1 was
not yet recovered.

Geometry is small enough (148 triangles total) that -- unlike CR4 -- no
coplanar-triangle merging is needed: pyroomacoustics builds one Wall per
triangle directly and stays fast.
"""
import os

import numpy as np
import mitsuba as mi
import pyroomacoustics as pra

from pra_comparison_bras_cr4 import _add_despite_flaky_is_inside

# One PLY per GRAP material, straight from RNo01_GH_Gewandhaus.skp (via an
# OBJ export whose mat0..mat5 were mapped to the GH_* names by their exact
# 8-bit colours in the .skp). RAVEN axes: (x, z, -y) * 0.0254 from SketchUp
# inches. Verified: closed mesh, 22051.29 m^3 / 5527.75 m^2 (= GRAP), and all
# 59 rows of RNo01_R1_PlaneWavesList_IS.csv reproduced by our order<=3 image
# sources (<0.035 ms, <0.07 deg, no extras). Winding is the source's own
# (inward) -- build_pra_room flips it, as with the previous meshes.
MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pra_comparison_assets", "gewandhaus", "meshes_grap")


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


def _ray_hit_distances(origin, direction, triangles, t_max=np.inf):
    """Möller-Trumbore against all triangles; inf where a triangle is missed."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    det = np.where(ok, det, 1.0)
    s = origin - a
    u = np.einsum("ij,ij->i", s, p) / det
    q = np.cross(s, e1)
    v = (q @ direction) / det
    t = np.einsum("ij,ij->i", e2, q) / det
    return np.where(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < t_max), t, np.inf)


def check_position(point, other):
    """Raise unless `point` lies inside the closed Gewandhaus mesh and sees
    `other` directly -- guards derived positions (e.g. a scaled
    source-receiver distance) against silently rendering outside the room."""
    triangles = np.concatenate([_load_mesh_triangles(n) for n in RICHTER_MATERIALS_20HZ])
    point, other = np.asarray(point, dtype=float), np.asarray(other, dtype=float)
    directions = np.array([[0.3, 0.2, 0.93], [-0.7, 0.5, 0.1], [0.1, -0.9, -0.4]])
    inside = all(np.isfinite(_ray_hit_distances(point, d / np.linalg.norm(d), triangles)).sum() % 2 == 1
                 for d in directions)
    distance = np.linalg.norm(other - point)
    line_of_sight = not np.isfinite(_ray_hit_distances(point, (other - point) / distance, triangles,
                                                       distance - 1e-3)).any()
    if not (inside and line_of_sight):
        raise ValueError(f"Gewandhaus position {tuple(np.round(point, 3))}: inside={inside}, "
                         f"line of sight to {tuple(np.round(other, 3))}={line_of_sight}")


def acoustic_bsdf(absorption, scattering, specular_lobe_width=0.001):
    return {
        "type": "acousticbsdf", "specular_lobe_width": specular_lobe_width,
        "absorption": {"type": "spectrum", "value": absorption},
        "scattering": {"type": "spectrum", "value": scattering},
    }


# =====================================================================
#   Richter Fig.14 reproduction: misuka (ours) vs. Richter's own misuka
#   results for Gewandhaus. Every constant below was fixed against
#   Richter's Masterarbeit Section 4.2/4.5 and the position/geometry
#   investigation in this suite's own conversation -- together they define
#   GEWANDHAUS_CANONICAL further down, the one Gewandhaus configuration this
#   whole suite uses.
#
# - source_radius=0.1m: Richter Sec. 4.2/4.5, verbatim. c: see
#   RICHTER_SPEED_OF_SOUND (GRAP-derived 343.99 m/s, was Richter's 343.64).
# - f=20Hz: Richter Sec. 4.5 evaluates only the lowest third-octave band;
#   coefficients below are each GRAP material CSV's 20Hz row.
# - Source/mic position: GRAP's real S1/R1, read from the three component-
#   instance transforms in RNo01_GH_Gewandhaus.skp (SketchUp inches, mapped
#   (x, z, -y) * 0.0254 into RAVEN/mesh axes). Verified: |S1-R1| = 11.4027 m
#   (GRAP doc Fig.4 / Richter's _parameters_plot.py), S1 1.500 m above the
#   stage floor (Fig.4: 1.50 m), R1 1.188 m above the seats (Fig.4: 1.20 m),
#   and the full R1 image-source list (see MESH_DIR). GRAP's plane-wave
#   angles are in a head frame pointed at S1 (receivers "frontally
#   aligned"), which is why its direct sound has elevation ~0 although R1
#   sits 1.45 m higher than S1.
# - Materials: all six GRAP materials, one mesh each (see MESH_DIR).
RICHTER_SOURCE_RADIUS = 0.1
# GRAP-derived: |S1-R1| = 11.4027 m / GRAP direct sound delay 0.0331486 s
# (RNo01_R1_PlaneWavesList_IS.csv row 2) -- replaces Richter's 343.64 m/s
# since this room is now compared directly against the GRAP data.
RICHTER_SPEED_OF_SOUND = 343.99
RICHTER_FREQUENCY_HZ = "20.0"
_INCH = 0.0254
# .skp component origins in SketchUp inches (x, y, z), z up.
_SKP_S1 = (-302.8085538303735, 635.5985774172516, 85.03937007874016)
_SKP_R1 = (142.49288145568823, 636.3328108865335, 141.95465719291474)
RICHTER_SOURCE_POS = (_SKP_S1[0] * _INCH, _SKP_S1[2] * _INCH, -_SKP_S1[1] * _INCH)   # GRAP S1
RICHTER_MIC_POS = (_SKP_R1[0] * _INCH, _SKP_R1[2] * _INCH, -_SKP_R1[1] * _INCH)      # GRAP R1
RICHTER_MATERIALS_20HZ = {           # (absorption, scattering), each CSV's 20Hz row
    "GH_Ceiling": (0.2769, 0.01),
    "GH_Floor": (0.0346, 0.01),
    "GH_Organ": (0.4803, 0.01),
    "GH_Seats": (0.1302, 0.01),
    "GH_Seats_organ": (0.1145, 0.01),
    "GH_Wall": (0.2422, 0.01),
}
# Receiver for the doubled-distance variant of the atmospheric inverse study
# (not a GRAP position -- chosen here). Scaling R1 along S1->R1 lands under
# the raked stalls, so instead: in plan view on the extended S1->R1 line,
# 1.20 m above the GH_Seats rake (R1's ear height per GRAP Fig.4), at the
# point where |P - S1| = 2 * |R1 - S1| = 22.805 m. Verified with
# check_position: inside, line of sight to S1, >=1.09 m from every surface.
DISTANCE_DOUBLE_MIC_POS = (14.232, 8.441, -16.1804)

# RNo01_R1_Parameters.csv, 20Hz row (RAVEN hybrid, ISM order 3 + 200k rays).
# C50 from D50: 10*log10(D50/(1-D50)). T30 is compared against T30, not T60.
GRAP_R1_20HZ_REFERENCE = dict(EDT=2.3551, T30=2.7103, C80=1.8272, D50=0.47384,
                              G=7.8536, TS=0.12253)
GRAP_R1_20HZ_REFERENCE["C50"] = float(10 * np.log10(GRAP_R1_20HZ_REFERENCE["D50"]
                                                   / (1 - GRAP_R1_20HZ_REFERENCE["D50"])))
GRAP_R1_DIRECT_DELAY_S = 0.0331
# Read off Fig.14 (Masterarbeit p.33), reference (hybrid-simulation) line,
# by eye -- no digitized numeric table exists in the thesis or GRAP data.
RICHTER_FIG14_REFERENCE = dict(C50=-0.4, C80=1.8, G=7.75, EDT=2.32, T60=2.66, TS=0.123)
# T60's JND is relative (~15.6% of its reference value, see JND_REL); the
# other five are Richter's fixed absolute JNDs (see JND_ABS).
RICHTER_FIG14_JND = dict(C50=1.0, C80=1.0, G=1.0, EDT=0.21, TS=0.010)

# The single canonical Gewandhaus configuration -- a consolidated view of
# the RICHTER_* constants above, not an independent set of values (so there
# is exactly one place any of these can drift). Anything in this suite that
# renders Gewandhaus should end up matching this, and both
# build_misuka_scene_dict_richter() and build_pra_room() print what they
# actually used (see _print_gewandhaus_header) so a mismatch is visible
# rather than silent.
GEWANDHAUS_CANONICAL = dict(
    frequency_hz=float(RICHTER_FREQUENCY_HZ),
    source_radius=RICHTER_SOURCE_RADIUS,
    speed_of_sound=RICHTER_SPEED_OF_SOUND,
    source_pos=RICHTER_SOURCE_POS,
    mic_pos=RICHTER_MIC_POS,
    materials=dict(RICHTER_MATERIALS_20HZ),
    atmosphere=False,
)


def _print_gewandhaus_header(label, frequency_hz, source_radius, speed_of_sound,
                             source_pos, mic_pos, materials, require_canonical=False):
    """Print the values a Gewandhaus render is about to use, and say whether
    they match GEWANDHAUS_CANONICAL. With require_canonical=True, a mismatch
    raises instead of just printing -- used by the misuka-vs-PRA scenario
    and the canonical PRA room builder, where nothing should ever legitimately
    deviate. Left False for the general-purpose scene builder, which the
    atmospheric-extensions study deliberately calls with other frequencies/
    positions (varying frequency and source-receiver distance is the whole
    point of that study) -- printing there is for visibility, not enforcement.
    """
    matches = (
        float(frequency_hz) == GEWANDHAUS_CANONICAL["frequency_hz"]
        and source_radius == GEWANDHAUS_CANONICAL["source_radius"]
        and speed_of_sound == GEWANDHAUS_CANONICAL["speed_of_sound"]
        and tuple(source_pos) == GEWANDHAUS_CANONICAL["source_pos"]
        and tuple(mic_pos) == GEWANDHAUS_CANONICAL["mic_pos"]
        and dict(materials) == GEWANDHAUS_CANONICAL["materials"]
    )
    status = "CANONICAL" if matches else "DEVIATES FROM CANONICAL"
    print(f"[Gewandhaus/{label}] f={frequency_hz} Hz, source_radius={source_radius} m, "
         f"c={speed_of_sound} m/s, source={tuple(source_pos)}, mic={tuple(mic_pos)}, "
         f"materials={sorted(materials)}  -- {status}")
    if require_canonical and not matches:
        raise ValueError(
            f"Gewandhaus/{label}: this render does not match GEWANDHAUS_CANONICAL. "
            "The misuka-vs-PRA comparison must use the canonical configuration -- "
            "pass no overrides, or fix GEWANDHAUS_CANONICAL itself if it is stale.")
    return matches


# Each mesh of the RICHTER_* scene and the GRAP database CSV its
# coefficients come from. The CSVs (copied verbatim into
# pra_comparison_assets/gewandhaus/, same pattern as bras_cr4_v2's
# materials_1khz.json) carry all 31 third-octave bands, so a scene can be
# built at any frequency rather than only at the 20 Hz row hardcoded in
# RICHTER_MATERIALS_20HZ -- needed by the atmospheric evaluation, whose
# whole subject (ISO 9613-1 air attenuation) scales with f^2 and is
# therefore meaningless at 20 Hz.
MATERIAL_CSV_BY_MESH = {
    "GH_Ceiling": "GH_Ceiling.csv",
    "GH_Floor": "GH_Floor.csv",
    "GH_Organ": "GH_Organ.csv",
    "GH_Seats": "GH_Seats.csv",
    "GH_Seats_organ": "GH_Seats organ.csv",
    "GH_Wall": "GH_Wall.csv",
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


def grap_r1_histogram(frequency_hz=20.0):
    """(bin_start_times_s, energy) of RNo01_R1_Histogram.csv's column for
    `frequency_hz` -- RAVEN's 1 ms energy histogram, absolutely scaled so the
    direct sound carries 1/d^2 (10*log10(sum*10^2) reproduces GRAP's G)."""
    with open(os.path.join(ASSET_DIR, "RNo01_R1_Histogram.csv")) as f:
        lines = f.read().splitlines()
    bands = [float(x) for x in lines[2].split(",")[1:] if x]
    col = 1 + bands.index(float(frequency_hz))
    rows = [l.split(",") for l in lines[7:] if l.strip()]
    return (np.array([float(r[0]) for r in rows]), np.array([float(r[col]) for r in rows]))


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
                                    source_radius=None, speed_of_sound=None,
                                    quiet=False):
    """The validated Gewandhaus scene (position/radius, see the block above).

    `frequencies` defaults to the 20 Hz band RICHTER_FREQUENCY_HZ pins, with
    that band's tabulated coefficients; pass explicit frequencies (a list, or
    a comma-separated string) to render other bands, in which case the
    materials are taken from the GRAP tables at those frequencies via
    materials_at(). source_pos/mic_pos/source_radius override the validated
    defaults -- used by the atmospheric distance study, which needs the same
    room and source but a different source-receiver separation.

    Prints what it actually built (frequency/radius/c/position/materials) and
    whether that matches GEWANDHAUS_CANONICAL, via _print_gewandhaus_header --
    never aborts here (require_canonical=False), since deliberately varying
    frequency/position is exactly what the atmospheric-extensions study does;
    pass quiet=True to suppress the print (e.g. inside a tight repeat loop).
    """
    source_pos = RICHTER_SOURCE_POS if source_pos is None else source_pos
    mic_pos = RICHTER_MIC_POS if mic_pos is None else mic_pos
    source_radius = RICHTER_SOURCE_RADIUS if source_radius is None else source_radius
    speed_of_sound = RICHTER_SPEED_OF_SOUND if speed_of_sound is None else speed_of_sound

    if frequencies is None:
        frequencies_str = RICHTER_FREQUENCY_HZ
        materials = {name: (a, s) for name, (a, s) in RICHTER_MATERIALS_20HZ.items()}
        header_freq = RICHTER_FREQUENCY_HZ
    else:
        if isinstance(frequencies, str):
            frequencies = [float(f) for f in frequencies.split(",")]
        frequencies = [float(f) for f in np.atleast_1d(frequencies)]
        frequencies_str = ", ".join(str(f) for f in frequencies)
        materials = materials_at(frequencies)
        # GEWANDHAUS_CANONICAL is single-frequency (20 Hz); a multi-band
        # request is definitionally a deviation, so any one of the requested
        # bands works for the header's frequency comparison -- use the first.
        header_freq = frequencies[0]

    if not quiet:
        _print_gewandhaus_header("misuka", header_freq, source_radius, speed_of_sound,
                                 source_pos, mic_pos, materials, require_canonical=False)

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
                   materials=None, source_pos=None, mic_pos=None, source_radius=None,
                   require_canonical=True, quiet=False, receiver_radius=0.5):
    """One pyroomacoustics Wall per mesh triangle (148 total, 6 GRAP
    materials) -- no merging needed at this size, unlike CR4. Wall winding
    is flipped relative to the source mesh's own (inward) convention, which
    gives +22051.29 m^3, matching GRAP.

    Defaults to GEWANDHAUS_CANONICAL and, with require_canonical=True (the
    default), raises if called with anything else -- this is the PRA side of
    the misuka-vs-PRA comparison, which unlike the atmospheric-extensions
    study has no legitimate reason to deviate. Pass require_canonical=False
    only for a deliberate, explicitly-acknowledged one-off.
    """
    materials = GEWANDHAUS_CANONICAL["materials"] if materials is None else materials
    source_pos = GEWANDHAUS_CANONICAL["source_pos"] if source_pos is None else source_pos
    mic_pos = GEWANDHAUS_CANONICAL["mic_pos"] if mic_pos is None else mic_pos
    source_radius = GEWANDHAUS_CANONICAL["source_radius"] if source_radius is None else source_radius

    if not quiet:
        _print_gewandhaus_header("PRA", GEWANDHAUS_CANONICAL["frequency_hz"], source_radius,
                                 GEWANDHAUS_CANONICAL["speed_of_sound"], source_pos, mic_pos,
                                 materials, require_canonical=require_canonical)

    # pyroomacoustics reads its speed of sound from this global, not from any
    # per-room argument -- previously never set here, so every earlier
    # Gewandhaus PRA room silently ran at pyroomacoustics' own default
    # (343.0 m/s) rather than the intended value, regardless of what the
    # rest of the config claimed. Set explicitly so misuka and PRA agree on
    # the one physical constant that affects every propagation delay in the
    # comparison.
    pra.constants.set("c", GEWANDHAUS_CANONICAL["speed_of_sound"])

    walls = []
    for mesh_name, (absorption, scattering) in materials.items():
        mat = pra.Material(energy_absorption=absorption, scattering=scattering)
        for i, tri in enumerate(_load_mesh_triangles(mesh_name)):
            tri = tri[::-1]  # flip winding
            walls.append(pra.wall_factory(tri.T, mat.energy_absorption["coeffs"],
                                          mat.scattering["coeffs"], name=f"{mesh_name}_{i}"))

    room = pra.Room(walls, fs=fs, max_order=max_order, ray_tracing=True)
    room.set_ray_tracing(n_rays=n_rays, energy_thres=energy_thres, time_thres=time_thres,
                         hist_bin_size=hist_bin_size, receiver_radius=receiver_radius)
    _add_despite_flaky_is_inside(room.add_source, np.array(source_pos))
    _add_despite_flaky_is_inside(room.add_microphone, np.array(mic_pos))
    return room
