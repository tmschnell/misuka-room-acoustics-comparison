"""BRAS CR4 auditorium import for both misuka and pyroomacoustics, from a
single shared source of geometry (PLY meshes) and materials
(materials_1khz.json, both under pra_comparison_assets/bras_cr4_v2/) -- see
the "H1 t0-alignment"-style investigation notes for how this was arrived at.

Two problems had to be solved to make the PRA side tractable and correct,
both found by actually running the naive approach rather than assuming it
would work:

1. A pyroomacoustics Wall per mesh triangle (5139 of them, the natural
   reading of pyroomacoustics' own STL-import examples) never finished
   building the Room object -- killed after >6 minutes at 99% CPU, before
   any actual ray tracing. pyroomacoustics' image-source method and ray
   tracer are built around a small number of large planar walls, not a
   fine triangulation.

2. Fixed by merging triangles into their real flat surfaces: group by
   *exact* plane (rounded normal AND rounded plane offset -- not just
   normal direction, so two parallel-but-different surfaces can never
   merge), then within each plane, chain edge-adjacent triangles into
   connected components, then trace each component's boundary (directed
   edges with no matching reverse edge) into its exact polygon outline.
   Deliberately not a convex hull: an earlier version used one, and it
   silently bridged over concave regions (confirmed: "linoleum" came out
   93% larger in merged-polygon area than the sum of the triangles that
   went into it, i.e. filling in real gaps with fake floor). Boundary
   tracing preserves the exact area by construction: merged wall area
   equals the original triangle-area sum, verified now within 0.01-2%
   (the residual is a handful of components with an actual inner hole,
   currently approximated as solid -- see merged_polygons' docstring).

Both fixes are validated against the one external ground truth available:
mat_CR4.txt documents the real room's volume as 8656.5 m^3; the
reconstructed PRA room (after also fixing a global wall-winding sign flip
pyroomacoustics needed relative to the source mesh's own normal
convention) comes out to ~8500 m^3, within 2%.
"""
import json
import os
from collections import defaultdict

import numpy as np
import mitsuba as mi
import pyroomacoustics as pra

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pra_comparison_assets", "bras_cr4_v2")
MESHES_DIR = os.path.join(ASSETS_DIR, "meshes")
MATERIALS_PATH = os.path.join(ASSETS_DIR, "materials_1khz.json")

MESH_NAMES = ("brickwall", "concrete", "linoleum", "parquet", "seating_upstairs",
             "seating_downstairs", "white_panels", "windows", "wood_panels",
             "reflector", "reflector_bottom")


def load_materials():
    with open(MATERIALS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------
#                       Mesh loading and welding
# ---------------------------------------------------------------------

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


def load_mesh(mesh_name, weld_ndigits=4):
    """PLY export uses per-face vertex duplication for flat shading
    (verified: white_panels has 3798 raw vertices but only 1146 distinct
    positions at 4-decimal precision) -- triangles that are geometrically
    edge-adjacent therefore don't share vertex *indices*, breaking any
    index-based connectivity check. Weld coincident positions into shared
    indices first.
    """
    path = os.path.join(MESHES_DIR, f"{mesh_name}.ply")
    mesh = mi.load_dict({"type": "ply", "filename": path})
    raw_verts = np.array([_to_xyz(mesh.vertex_position(i)) for i in range(mesh.vertex_count())])
    raw_faces = np.array([_to_ijk(mesh.face_indices(i)) for i in range(mesh.face_count())])

    welded_index, welded_verts = {}, []
    remap = np.empty(len(raw_verts), dtype=int)
    for i, p in enumerate(raw_verts):
        key = tuple(np.round(p, weld_ndigits))
        idx = welded_index.get(key)
        if idx is None:
            idx = len(welded_verts)
            welded_index[key] = idx
            welded_verts.append(p)
        remap[i] = idx
    return np.array(welded_verts), remap[raw_faces]


# ---------------------------------------------------------------------
#         Coplanar, edge-connected triangle merging (boundary trace)
# ---------------------------------------------------------------------

def _face_normals(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return n / norm


def _connected_components(faces_subset):
    edge_to_faces = defaultdict(list)
    for fi, f in enumerate(faces_subset):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_to_faces[frozenset((a, b))].append(fi)
    parent = list(range(len(faces_subset)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for fis in edge_to_faces.values():
        for a, b in zip(fis[:-1], fis[1:]):
            union(a, b)

    groups = defaultdict(list)
    for i in range(len(faces_subset)):
        groups[find(i)].append(i)
    return list(groups.values())


def _trace_boundary_loops(faces_subset):
    """Directed-edge boundary trace: an edge (u,v) is a boundary edge iff
    its reverse (v,u) doesn't occur elsewhere in this triangle set
    (interior edges are shared by exactly 2 triangles with opposite
    winding, in a consistently-oriented manifold patch). Chains boundary
    edges into closed loops, preserving the original winding direction
    exactly -- no convex hull, so concave outlines/notches are preserved
    and merged area is exact.
    """
    directed = set()
    for f in faces_subset:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            directed.add((a, b))
    boundary = [e for e in directed if (e[1], e[0]) not in directed]

    next_of = {a: b for a, b in boundary}
    loops, seen = [], set()
    for start in next_of:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = next_of[start]
        while cur != start and cur not in seen:
            loop.append(cur)
            seen.add(cur)
            cur = next_of.get(cur)
            if cur is None:
                loop = None
                break
        if loop:
            loops.append(loop)
    return loops


def _polygon_area(pts):
    if len(pts) < 3:
        return 0.0
    c = pts.mean(axis=0)
    total = np.zeros(3)
    for i in range(len(pts)):
        total += np.cross(pts[i] - c, pts[(i + 1) % len(pts)] - c)
    return 0.5 * np.linalg.norm(total)


def merged_polygons(mesh_name, normal_round=3, offset_round=2):
    """One entry per real flat surface patch in `mesh_name`: exactly
    coplanar (same rounded normal AND same rounded plane offset) and
    edge-connected triangles, merged via boundary tracing (not a convex
    hull -- see module docstring). `normal` is the *original* mesh
    convention (verified: points from solid into the room's air volume,
    e.g. (0,1,0)/"up" for a floor) -- pyroomacoustics wants the opposite
    (see build_pra_room's wall-winding flip), so use this raw `normal`
    whenever moving a point *away* from a wall and into the room (see
    nudge_into_room below), not the flipped one pyroomacoustics ends up
    using internally.

    A component with more than one boundary loop has a real hole (rare;
    the inner loop(s) are dropped and the hole is approximated as solid --
    a documented, minor simplification, not a silent area-inflation bug
    the way convex-hull merging was).
    """
    verts, faces = load_mesh(mesh_name)
    normals = _face_normals(verts, faces)
    offsets = np.einsum("ij,ij->i", normals, verts[faces[:, 0]])

    by_plane = defaultdict(list)
    for fi in range(len(faces)):
        key = (tuple(np.round(normals[fi], normal_round)), round(offsets[fi], offset_round))
        by_plane[key].append(fi)

    polygons = []
    for key, face_idxs in by_plane.items():
        normal = np.array(key[0])
        if np.linalg.norm(normal) < 0.5:
            continue  # degenerate (near-zero-area) triangle(s), not a real surface
        faces_subset = faces[face_idxs]
        for comp in _connected_components(faces_subset):
            comp_faces = faces_subset[comp]
            v0, v1, v2 = verts[comp_faces[:, 0]], verts[comp_faces[:, 1]], verts[comp_faces[:, 2]]
            orig_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()

            loops = _trace_boundary_loops(comp_faces)
            if not loops:
                continue
            loops.sort(key=len, reverse=True)
            corners = verts[loops[0]]
            polygons.append(dict(mesh=mesh_name, corners=corners, normal=normal,
                                 n_triangles=len(comp_faces), orig_area=orig_area,
                                 merged_area=_polygon_area(corners), n_loops=len(loops)))
    return polygons


def all_merged_polygons():
    return {mesh_name: merged_polygons(mesh_name) for mesh_name in MESH_NAMES}


# ---------------------------------------------------------------------
#      Point-source placement: misuka's finite emitter sphere vs.
#      pyroomacoustics' idealized point source (must be strictly inside)
# ---------------------------------------------------------------------

def nudge_into_room(point, radius, polygons_by_mesh, margin=1e-3, tol=1e-2, max_iter=10):
    """If `point` sits on (within `tol` of) a wall plane -- as CR4's own
    loudspeaker reference position does, exactly on the stage floor
    (verified: distance 0.0 m to the "linoleum" plane at y=1.0) -- push it
    `radius + margin` into the room along that wall's normal. Same root
    cause as the hull-vs-center offset already handled for the shoebox/
    L-room scenarios (align_hull_to_center in test_pyroomacoustics_
    comparison.py): misuka's emitter is a finite sphere that can
    physically sit flush with/recessed into a surface, but pyroomacoustics
    needs a strictly-interior point source.

    The reference position turned out to sit exactly on a junction of
    several different floor materials (linoleum/parquet/seating riser all
    within `tol`, verified) -- pushing along only the first match isn't
    always enough to clear every nearby surface in one step, so this
    repeats (summing every currently-matching normal, then re-checking)
    until no plane is within `tol` or `max_iter` is reached. A no-op for a
    point already clear of every wall.
    """
    point = np.asarray(point, dtype=float)
    for _ in range(max_iter):
        # Several differently-named surfaces can meet at exactly the same
        # point (verified: CR4's reference position sits on a linoleum/
        # parquet/seating-riser junction, all near-identical (0,1,0)
        # normals) -- dedupe by direction first, so the push is scaled by
        # radius+margin *once* per distinct direction, not once per
        # matching polygon (summing 5 near-identical full-length normals
        # would overshoot by ~5x instead of clearing the surface by one
        # radius).
        directions = {tuple(np.round(poly["normal"], 2))
                     for polys in polygons_by_mesh.values() for poly in polys
                     if abs(np.dot(point - poly["corners"][0], poly["normal"])) < tol}
        if not directions:
            break
        push = np.array([sum(d) for d in zip(*directions)])
        push = push / np.linalg.norm(push)
        point = point + push * (radius + margin)
    return point


# ---------------------------------------------------------------------
#                           Scene builders
# ---------------------------------------------------------------------

def build_misuka_scene_dict(materials, emitter_pos, mic_pos, source_radius,
                            max_time, sampling_rate, frequency_hz=1000.0):
    scene_dict = {"type": "scene"}
    for mesh_name, mat_name in materials["mesh_to_material"].items():
        coeffs = materials["materials"][mat_name]
        scene_dict[mesh_name] = {
            "type": "ply",
            "filename": os.path.join(MESHES_DIR, f"{mesh_name}.ply"),
            "face_normals": True,
            "bsdf": {
                "type": "acousticbsdf",
                "absorption": {"type": "spectrum", "value": coeffs["absorption"]},
                "scattering": {"type": "spectrum", "value": coeffs["scattering"]},
            },
        }
    scene_dict["emitter"] = {
        "type": "sphere", "radius": source_radius, "center": list(emitter_pos),
        "emitter": {"type": "area", "radiance": {"type": "uniform", "value": 1.0}},
    }
    scene_dict["mic"] = {
        "type": "microphone", "origin": list(mic_pos), "direction": [1.0, 0.0, 0.0],
        "film": {"type": "tape", "frequencies": str(frequency_hz),
                 "time_bins": int(round(max_time * sampling_rate))},
    }
    return scene_dict


def _add_despite_flaky_is_inside(add_fn, point, max_attempts=20):
    """pyroomacoustics.Room.is_inside() (called internally by add_source/
    add_microphone) is itself randomized: it casts a line from a randomly
    perturbed external reference point and counts wall crossings, retrying
    on ambiguous (edge/vertex-grazing) hits up to its own internal cap
    before giving up for that attempt. Verified empirically for this room:
    the *same* point, same room, alternates True/False across repeated
    calls (e.g. 10-19 "inside" out of 20 trials at various safe-looking
    margins) -- not proximity to a real gap (checked: the ~20-odd merged
    polygons pyroomacoustics itself rejects as "not planar enough", see
    build_pra_room, sit 9.6-26 m away from every position tested here).
    Root cause: ~1700 edge-adjacent merged polygons share far more edges
    than the small number of large simple walls pyroomacoustics' is_inside
    was designed around, so a random probe line has a much higher chance
    of grazing a shared edge and landing in the ambiguous case. Each call
    resamples its own random reference point independently, so simply
    retrying resolves it in practice.
    """
    last_exc = None
    for _ in range(max_attempts):
        try:
            return add_fn(list(point))
        except ValueError as exc:
            last_exc = exc
    raise RuntimeError(
        f"pyroomacoustics.Room.is_inside() failed for {point} in all {max_attempts} "
        "randomized attempts -- this point is very likely genuinely outside the room, "
        "not just an unlucky ambiguous probe (see docstring)."
    ) from last_exc


def build_pra_room(materials, max_order, fs, hist_bin_size, emitter_pos, mic_pos, source_radius,
                   ray_tracing=True, n_rays=None, energy_thres=1e-8):
    """pyroomacoustics room from the same materials/geometry as
    build_misuka_scene_dict. Wall winding is flipped relative to the
    source mesh's own normal convention (verified via room volume: this
    flip is what turns a nonsensical negative volume into ~8500 m^3,
    matching the real, documented 8656.5 m^3 within 2%) -- so
    nudge_into_room above must use the *unflipped* `normal` from
    merged_polygons, not a wall's own (flipped) .normal attribute.

    `fs`/`hist_bin_size` are taken as plain parameters rather than derived
    from a sampling_rate here (this module deliberately doesn't import
    test_pyroomacoustics_comparison, which will import *this* module for
    scenario_bras_cr4 -- avoiding a circular import). Callers should pass
    `fs=int(cmp.BIN_FACTOR * cmp.SAMPLING_RATE)` and
    `hist_bin_size=1.0/cmp.SAMPLING_RATE`, matching cmp.render_pra's own
    already-documented reason: pyroomacoustics' rt.py interp_hist computes
    `pad = (room.fs // n_bins) // 2` and slices `out[..., pad:-pad]`; if
    hist_bin_size equals 1/room.fs exactly, pad is 0 and that slice
    degenerates to an empty one -- a real crash (reproduced: `ValueError:
    could not broadcast input array from shape (N,) into shape (0,)`), not
    a precision concern. Use cmp.render_pra_frac_delay_stripped(room,
    n_time_bins) to get the resulting RIR back down to cmp.SAMPLING_RATE,
    aligned and frac_delay-stripped, exactly like the other scenarios.
    """
    polygons_by_mesh = all_merged_polygons()

    walls = []
    for mesh_name, mat_name in materials["mesh_to_material"].items():
        coeffs = materials["materials"][mat_name]
        mat = pra.Material(energy_absorption=coeffs["absorption"], scattering=coeffs["scattering"])
        for i, poly in enumerate(polygons_by_mesh[mesh_name]):
            corners = poly["corners"]
            if len(corners) < 3:
                continue
            corners = corners[::-1]  # flip to pyroomacoustics' winding convention
            try:
                walls.append(pra.wall_factory(corners.T, mat.energy_absorption["coeffs"],
                                              mat.scattering["coeffs"], name=f"{mesh_name}_{i}"))
            except Exception:
                continue  # a handful of near-degenerate merged polygons fail pra's own planarity check

    room = pra.Room(walls, fs=fs, max_order=max_order, ray_tracing=ray_tracing)
    if ray_tracing:
        room.set_ray_tracing(n_rays=n_rays, energy_thres=energy_thres, hist_bin_size=hist_bin_size)

    source = nudge_into_room(emitter_pos, source_radius, polygons_by_mesh)
    mic = nudge_into_room(mic_pos, source_radius, polygons_by_mesh)
    _add_despite_flaky_is_inside(room.add_source, source)
    _add_despite_flaky_is_inside(room.add_microphone, mic)
    return room
