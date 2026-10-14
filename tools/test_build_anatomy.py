"""Tests for the anatomy pipeline's geometry, using synthetic tubes whose true
axis is known analytically.

These exist because the pipeline's failures are the silent kind: a mesh that is
subtly wrong still renders perfectly happily. The earlier centreline-fitting
tests here caught two real bugs (the geodesic hugging one wall of the tube, and
the path coming out reversed so tapered vessels rendered thin-end-first), and
that extractor has since been removed - the display now shows the BodyParts3D
mesh itself rather than a tube swept along a recovered path.

What is left to get wrong, and so what is tested here:

* **Welding.** The source OBJs duplicate vertices per face, so topologically the
  mesh is a cloud of loose triangles. Everything that walks edges depends on
  welding running first.
* **Decimation winding.** Deduplicating faces by their sorted index triple would
  randomise triangle orientation, and with it the normals. Invisible while every
  surface was translucent and double-sided; very visible now they are solid.
* **The narrowing morph target.** It must point inward and be bounded, or a
  high-risk vessel turns inside out instead of thinning.
* **The GLB container**, including the morph target the viewer binds to.

Run from the repo root::

    python -m pytest tools/test_build_anatomy.py -v
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_anatomy import (  # noqa: E402
    CONTRACT_ITERATIONS,
    NARROW_FRACTION,
    contract_to_axis,
    local_radius,
    narrowing_delta,
    parse_obj,
    vertex_cluster_decimate,
    vertex_normals,
    weld,
    write_glb,
)


def make_tube(curve_fn, n_along=90, n_around=14, radius_fn=lambda t: 0.5):
    """Sweep a circle along a 3D curve to build a tubular mesh."""
    verts: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    for i in range(n_along):
        t = i / (n_along - 1)
        c = np.asarray(curve_fn(t), dtype=float)
        eps = 1e-4
        tangent = np.asarray(curve_fn(min(t + eps, 1.0)), dtype=float) - np.asarray(
            curve_fn(max(t - eps, 0.0)), dtype=float
        )
        tangent /= np.linalg.norm(tangent) or 1.0
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(helper, tangent))) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        n1 = np.cross(tangent, helper)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(tangent, n1)
        r = radius_fn(t)
        for j in range(n_around):
            a = 2 * np.pi * j / n_around
            verts.append(c + r * (np.cos(a) * n1 + np.sin(a) * n2))
    for i in range(n_along - 1):
        for j in range(n_around):
            a = i * n_around + j
            b = i * n_around + (j + 1) % n_around
            c_ = (i + 1) * n_around + j
            d = (i + 1) * n_around + (j + 1) % n_around
            faces.append((a, b, c_))
            faces.append((b, d, c_))
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def unweld(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Explode a mesh into per-face vertices, as the source OBJs store them."""
    flat = faces.reshape(-1)
    return verts[flat], np.arange(len(flat), dtype=np.int64).reshape(-1, 3)


STRAIGHT = lambda t: (0.0, 0.0, 10.0 * t)  # noqa: E731
# A helix bending in all three axes, standing in for a vessel wrapping a ventricle.
HELIX = lambda t: (  # noqa: E731
    3.0 * np.cos(2.2 * np.pi * t),
    3.0 * np.sin(2.2 * np.pi * t),
    6.0 * t,
)


# ---------------------------------------------------------------------------
# Welding
# ---------------------------------------------------------------------------
def test_welding_restores_connectivity_of_an_exploded_mesh():
    """The real failure this guards: merged raw, the LAD's largest connected
    component held 565 of 1518 vertices. Welding brought it to a single piece."""
    verts, faces = make_tube(HELIX, n_along=60, n_around=12)
    loose_v, loose_f = unweld(verts, faces)
    assert len(loose_v) == len(faces) * 3, "test fixture is not actually exploded"

    welded_v, welded_f = weld(loose_v, loose_f)

    assert len(welded_v) == len(verts), "welding did not recover the vertex count"
    assert len(welded_f) == len(faces), "welding dropped real triangles"

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    def components(v, f):
        e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v)))
        return connected_components(g, directed=False)[0]

    assert components(loose_v, loose_f) == len(faces), "fixture should be shattered"
    assert components(welded_v, welded_f) == 1, "welded tube should be one piece"


def test_welding_preserves_geometry():
    verts, faces = make_tube(STRAIGHT, n_along=40, n_around=10)
    welded_v, _ = weld(*unweld(verts, faces))
    # Same point set, possibly reordered.
    assert np.allclose(np.sort(welded_v, axis=0), np.sort(verts, axis=0), atol=1e-6)


def test_welding_leaves_an_already_welded_mesh_alone():
    verts, faces = make_tube(HELIX, n_along=30, n_around=10)
    welded_v, welded_f = weld(verts, faces)
    assert len(welded_v) == len(verts)
    assert len(welded_f) == len(faces)


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------
def test_decimation_respects_budget_and_keeps_the_shape():
    verts, faces = make_tube(HELIX, n_along=300, n_around=30)
    assert len(faces) > 10_000

    small_v, small_f = vertex_cluster_decimate(verts, faces, 2000)
    assert len(small_f) <= 2000, f"{len(small_f)} triangles exceeds the budget"

    before = verts.max(axis=0) - verts.min(axis=0)
    after = small_v.max(axis=0) - small_v.min(axis=0)
    assert np.abs(before - after).max() < 0.5, "bounding box changed materially"


def test_decimation_keeps_triangle_winding():
    """Deduplicating on np.sort(faces) would scramble orientation. On a closed
    surface that shows up as normals pointing inward for roughly half the mesh."""
    verts, faces = make_tube(STRAIGHT, n_along=120, n_around=24)
    small_v, small_f = vertex_cluster_decimate(verts, faces, 600)

    normals = vertex_normals(small_v, small_f)
    axis_point = np.array([0.0, 0.0, 0.0])
    radial = small_v - axis_point
    radial[:, 2] = 0.0  # the tube's axis is z, so outward is the xy direction
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    outward = radial / norms

    agreement = np.sum(normals * outward, axis=1)
    # A scrambled mesh gives a mean near zero; a consistently wound one is
    # strongly signed, whichever way the source happened to be oriented.
    assert abs(float(np.mean(agreement))) > 0.6, (
        f"normals are incoherent (mean outward agreement {np.mean(agreement):.3f}) "
        "- triangle winding was not preserved by decimation"
    )


# ---------------------------------------------------------------------------
# Narrowing morph target
# ---------------------------------------------------------------------------
def test_contraction_collapses_a_tube_onto_its_axis():
    verts, faces = make_tube(STRAIGHT, n_along=80, n_around=16, radius_fn=lambda t: 0.5)
    # A high pass count, to show the direction is genuinely toward the axis.
    # Production uses far fewer, because the distance comes from the measured
    # radius and only the direction is wanted from here.
    contracted = contract_to_axis(verts, faces, iterations=80)

    before = np.linalg.norm(verts[:, :2], axis=1)
    after = np.linalg.norm(contracted[:, :2], axis=1)
    assert before.mean() == pytest.approx(0.5, abs=1e-6)
    assert after.mean() < before.mean() * 0.5, "contraction barely moved anything"


def test_the_shipped_displacement_keeps_the_vessel_s_length():
    """Laplacian smoothing drags free ends inward as well as walls, and a vessel
    that shortens reads as an artery retracting - a different, wrong claim.

    Clipping the displacement to a fraction of the lumen radius is what bounds
    that. Note this fixture is harsher than the real data: its ends are open, so
    they smooth inward fast, where the merged BodyParts3D arteries are watertight
    and keep 98.9-99.5% of their extent under the same displacement.
    """
    verts, faces = make_tube(STRAIGHT, n_along=80, n_around=16, radius_fn=lambda t: 0.5)
    span = lambda v: float(v[:, 2].max() - v[:, 2].min())  # noqa: E731
    before = span(verts)

    raw = span(contract_to_axis(verts, faces, iterations=CONTRACT_ITERATIONS))
    shipped = span(verts + narrowing_delta(verts, faces))

    assert shipped > raw, "clipping should shorten the vessel less than the raw contraction"
    # The exact geometric guarantee rather than a round number: no vertex moves
    # further than radius x NARROW_FRACTION, so the two ends between them can
    # cost at most twice that.
    floor = before - 2 * 0.5 * NARROW_FRACTION
    assert shipped >= floor, (
        f"shipped displacement shortened the tube to {shipped:.3f}, past the "
        f"{floor:.3f} the clip guarantees"
    )


def test_local_radius_recovers_a_known_calibre():
    for radius in (0.25, 0.5, 1.0):
        verts, faces = make_tube(STRAIGHT, n_along=120, n_around=20,
                                 radius_fn=lambda t: radius)
        measured = np.median(local_radius(verts, faces))
        assert measured == pytest.approx(radius, rel=0.15), (
            f"measured {measured:.3f} for a tube of radius {radius}"
        )


def test_local_radius_tracks_a_taper():
    verts, faces = make_tube(STRAIGHT, n_along=160, n_around=20,
                             radius_fn=lambda t: 1.0 - 0.6 * t)
    radius = local_radius(verts, faces)
    wide = radius[verts[:, 2] < 2.0].mean()
    narrow = radius[verts[:, 2] > 8.0].mean()
    assert wide > narrow * 1.8, "radius did not follow the taper"


def test_narrowing_delta_points_inward_and_is_bounded():
    radius = 0.5
    verts, faces = make_tube(HELIX, n_along=100, n_around=16,
                             radius_fn=lambda t: radius)
    delta = narrowing_delta(verts, faces)

    magnitude = np.linalg.norm(delta, axis=1)
    # Hard bound: a vertex can never travel further than its own lumen radius
    # times the fraction, so the vessel cannot turn inside out at full influence.
    assert magnitude.max() <= radius * NARROW_FRACTION * 1.15, (
        f"a vertex moves {magnitude.max():.3f}, past the {radius * NARROW_FRACTION:.3f} limit"
    )
    assert magnitude.mean() > 0.02, "morph target is effectively a no-op"

    # Inward means toward the local axis: applying it must shrink the mesh's
    # cross-section without displacing its centre.
    moved = verts + delta
    centre_before = verts.mean(axis=0)
    centre_after = moved.mean(axis=0)
    assert np.linalg.norm(centre_after - centre_before) < 0.05, (
        "narrowing shifted the vessel instead of thinning it"
    )
    spread_before = np.linalg.norm(verts - centre_before, axis=1).std()
    spread_after = np.linalg.norm(moved - centre_after, axis=1).std()
    assert spread_after < spread_before, "narrowing did not reduce cross-section"


def test_narrowing_is_proportional_along_a_tapering_vessel():
    """The thin distal end must narrow by the same FRACTION as the thick proximal
    end, not by the same distance. Driving the distance off the contraction
    instead collapsed thin segments through their own axis while leaving thick
    ones barely touched."""
    verts, faces = make_tube(STRAIGHT, n_along=160, n_around=20,
                             radius_fn=lambda t: 1.0 - 0.6 * t)
    delta = np.linalg.norm(narrowing_delta(verts, faces), axis=1)
    radius = local_radius(verts, faces)

    proximal = verts[:, 2] < 2.0
    distal = verts[:, 2] > 8.0
    assert delta[proximal].mean() > delta[distal].mean(), (
        "the thick end should move further in absolute terms"
    )
    ratio_proximal = (delta[proximal] / radius[proximal]).mean()
    ratio_distal = (delta[distal] / radius[distal]).mean()
    assert ratio_proximal == pytest.approx(ratio_distal, abs=0.15), (
        f"narrowing is not proportional: {ratio_proximal:.2f} vs {ratio_distal:.2f}"
    )
    assert max(ratio_proximal, ratio_distal) <= NARROW_FRACTION * 1.1


# ---------------------------------------------------------------------------
# OBJ reading
# ---------------------------------------------------------------------------
def test_parse_obj_triangulates_and_ignores_normals():
    text = "\n".join([
        "# Concept ID : FMA1",
        "v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0",
        "vn 0 0 1",
        "f 1//1 2//1 3//1 4//1",   # a quad, must fan-triangulate
    ])
    verts, faces = parse_obj(text)
    assert verts.shape == (4, 3)
    assert faces.shape == (2, 3), "quad was not split into two triangles"
    assert set(map(tuple, faces)) == {(0, 1, 2), (0, 2, 3)}


# ---------------------------------------------------------------------------
# GLB container
# ---------------------------------------------------------------------------
CUBE_V = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float,
)
CUBE_F = np.array(
    [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
     [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]],
    dtype=np.int64,
)


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    magic, version, length = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67, "not a glTF binary"
    assert version == 2
    assert length == len(raw), "header length disagrees with file size"
    json_len, json_type = struct.unpack("<II", raw[12:20])
    assert json_type == 0x4E4F534A, "first chunk must be JSON"
    gltf = json.loads(raw[20 : 20 + json_len].decode("utf-8"))
    bin_len, bin_type = struct.unpack("<II", raw[20 + json_len : 28 + json_len])
    assert bin_type == 0x004E4942, "second chunk must be BIN"
    return gltf, raw[28 + json_len : 28 + json_len + bin_len]


def accessor_data(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5125: np.uint32}[acc["componentType"]]
    cols = {"VEC3": 3, "SCALAR": 1}[acc["type"]]
    chunk = blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    return np.frombuffer(chunk, dtype=dtype).reshape(-1, cols)


def test_glb_is_a_valid_container_and_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.glb"
        write_glb(path, {
            "cube": {"verts": CUBE_V, "faces": CUBE_F},
            "cube2": {"verts": CUBE_V + 2, "faces": CUBE_F},
        })
        gltf, blob = read_glb(path)

    assert [m["name"] for m in gltf["meshes"]] == ["cube", "cube2"]
    assert [n["name"] for n in gltf["nodes"]] == ["cube", "cube2"]
    assert len(gltf["accessors"]) == 6, "expected POSITION/NORMAL/indices per mesh"
    # The licence requires attribution to travel with the asset.
    assert "BodyParts3D" in gltf["asset"]["copyright"]

    for acc in gltf["accessors"]:
        view = gltf["bufferViews"][acc["bufferView"]]
        assert view["byteOffset"] + view["byteLength"] <= len(blob), "view overruns BIN"

    pos = accessor_data(gltf, blob, gltf["meshes"][0]["primitives"][0]
                        ["attributes"]["POSITION"])
    assert np.abs(pos - CUBE_V).max() < 1e-5, "positions did not round-trip"


def test_glb_writes_the_stenosis_morph_target():
    """The viewer drives narrowing through morphTargetInfluences[0], so the
    target has to exist, sit at index 0, and start at weight zero."""
    morph = np.full_like(CUBE_V, -0.1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.glb"
        write_glb(path, {
            "plain": {"verts": CUBE_V, "faces": CUBE_F},
            "artery_LAD": {"verts": CUBE_V, "faces": CUBE_F, "morph": morph},
        })
        gltf, blob = read_glb(path)

    plain, artery = gltf["meshes"]
    assert "targets" not in plain["primitives"][0], "non-artery got a morph target"
    assert "weights" not in plain

    targets = artery["primitives"][0]["targets"]
    assert len(targets) == 1, "exactly one morph target expected"
    assert artery["weights"] == [0.0], "morph must start fully relaxed"
    assert artery["extras"]["targetNames"] == ["stenosis"]

    delta = accessor_data(gltf, blob, targets[0]["POSITION"])
    assert delta.shape == CUBE_V.shape
    assert np.abs(delta - morph).max() < 1e-6, "morph deltas did not round-trip"


def test_glb_morph_accessor_declares_bounds():
    """glTF requires min/max on POSITION accessors; some loaders reject without."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.glb"
        write_glb(path, {
            "artery_LAD": {"verts": CUBE_V, "faces": CUBE_F,
                           "morph": np.full_like(CUBE_V, -0.1)},
        })
        gltf, _ = read_glb(path)

    primitive = gltf["meshes"][0]["primitives"][0]
    for index in (primitive["attributes"]["POSITION"],
                  primitive["targets"][0]["POSITION"]):
        acc = gltf["accessors"][index]
        assert "min" in acc and "max" in acc, "POSITION accessor is missing bounds"
        assert len(acc["min"]) == 3 and len(acc["max"]) == 3
