"""Build the 3D anatomy assets from the BodyParts3D heart part set.

Reads the OBJ files in ``heart-beat parts/`` and writes two committed assets into
``frontend/public/models/``:

* ``anatomy.glb``   -- every structure as a named node, real BodyParts3D geometry,
                       with a narrowing morph target on each coronary artery
* ``vessels.json``  -- the manifest: which FMA concepts and which source files went
                       into each node, its triangle count, label anchor and bounds

WHAT CHANGED, AND WHY IT MATTERED
---------------------------------
The first version of this script downloaded ``partof_BP3D_4.0_obj_99.zip`` and built
the display from it. Two consequences were visible on screen:

1. That archive is reduced to 99% fewer polygons and contains **no myocardium**. Its
   only heart solids are the four chamber *cavities* - the blood pool - so the "heart"
   it produced was a set of blood-shaped blobs rather than cardiac muscle.
2. Coronary arteries there were too pitted to render, so the script reduced each to a
   ~40-point centreline and swept a synthetic tube along it. The path came from the
   data but every surface shown was generated.

This set (``heart5.0i``, 129 files, 93 FMA concepts) has real ventricular and atrial
walls, individually named coronary branches, valve leaflets and the conduction system
at full resolution, so both workarounds are gone: what renders is the mesh itself.

THREE THINGS THE FILES THEMSELVES TELL YOU
------------------------------------------
Established by reading the OBJs, not by trusting any parts list:

1. **Vertices are not welded.** Faces reference per-face duplicate vertices, so a
   mesh that is one solid piece geometrically is a cloud of disconnected triangles
   topologically: merged raw, the LAD splits into 20 components whose largest holds
   565 of 1518 vertices. Welding at 1e-3 mm brings that to 3. Anything that walks
   edges - decimation, contraction, component counts - is wrong without it.
2. **Multi-file structures are sequential segments, not alternatives.** LAD trunk
   MM420 -> MM424 -> MM425 abut within 0.04 mm, RCA MM556 -> MM436 -> MM439 within
   0.12 mm, LCX MM426 -> MM635 within 0.22 mm. Merging per FMA gives one continuous
   artery.
3. **This donor is left-dominant.** The posterior interventricular branch is filed as
   FMA79742, "trunk of posterior interventricular branch of *left* coronary artery",
   so the PDA belongs to the LCX territory here. In roughly 85% of people it arises
   from the RCA instead. The UI says so; assigning it silently would misrepresent
   both this model and the typical patient.

Attribution (required by the licence):
    BodyParts3D, (c) The Database Center for Life Science
    licensed under CC Attribution 4.0 International

Usage::

    python tools/build_anatomy.py            # build the assets
    python tools/build_anatomy.py --report   # print source statistics and exit
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = REPO_ROOT / "heart-beat parts"
OUT_DIR = REPO_ROOT / "frontend" / "public" / "models"

ATTRIBUTION = (
    "BodyParts3D, (c) The Database Center for Life Science "
    "licensed under CC Attribution 4.0 International"
)
SOURCE_SET = "BodyParts3D heart part set, compatibility version heart5.0i"

# Narrowing morph target: at full influence every vertex moves inward by this
# share of its own local lumen radius. Expressing it per vertex rather than as a
# global distance is what makes it anatomically meaningful - a 0.3 mm shift is
# most of a distal branch and barely visible on the proximal RCA. Uniform ALONG
# the vessel on purpose: the dataset labels a vessel stenotic or not and never
# says where the lesion sits, so a localised pinch would be an invention.
NARROW_FRACTION = 0.45

# Laplacian contraction supplies the DIRECTION of that movement, not its
# distance. It cannot supply the distance: how far a fixed number of passes
# travels depends on how dense the mesh is, and on these meshes it badly
# overshoots - measured against a true median calibre of 0.59 mm (LAD), 0.75 mm
# (LCX) and 0.72 mm (RCA), 25 passes already moves the average LAD vertex 1.23
# radii, i.e. straight through the lumen and out the far side. So the pass count
# is kept low, where the direction is reliably radial and the free ends have not
# yet been dragged inward (which would read as the artery retracting rather than
# narrowing), and the magnitude comes from the measured radius instead.
CONTRACT_ITERATIONS = 20
CONTRACT_LAMBDA = 0.6

WELD_EPS_MM = 1e-3


# ---------------------------------------------------------------------------
# Structure table.
#
# "layer" decides what the viewer shows by default: "core" is always on, the
# others are user-toggled. "group" decides how the frontend materialises it.
# Keyed by FMA Concept ID, the stable identifier across releases.
#
# "ratio" is the share of triangles kept by decimation, and it is set per node
# rather than globally because vertex clustering TEARS thin-walled geometry. The
# merged source meshes are watertight - zero boundary edges - and every hole in
# the render was introduced here: at ratio 0.35 the LAD picked up 280 boundary
# edges (4.5% of its edges) and the ventricle walls showed visible slits with the
# interior surface glinting through them. Measured boundary edges by ratio on the
# LV free wall: 0.8 -> 74, 0.6 -> 166, 0.5 -> 167. So the arteries, which are the
# whole point of this view and the thinnest things in it, are not decimated at
# all, the bulk organ is only lightly reduced, and the heaviest reductions are
# reserved for the valve leaflets, which are modelled at an absurd density
# (the mitral valve alone is 70,480 triangles before reduction).
# ---------------------------------------------------------------------------
PARTS: dict[str, dict] = {
    # -- ventricular myocardium: the solid body of the heart -----------------
    "myo_lv_free": {
        "label": "Left ventricle, free wall",
        "fmas": ("FMA84850",), "group": "myocardium", "layer": "core", "ratio": 0.8,
    },
    "myo_lv_septal": {
        "label": "Left ventricle, septal wall",
        "fmas": ("FMA9345",), "group": "myocardium", "layer": "core", "ratio": 0.8,
    },
    "myo_septum": {
        "label": "Interventricular septum",
        "fmas": ("FMA7134", "FMA83468"), "group": "myocardium", "layer": "core",
        "ratio": 0.8,
    },
    "myo_rv_inflow": {
        "label": "Right ventricle, inflow wall",
        "fmas": ("FMA49207",), "group": "myocardium", "layer": "core", "ratio": 0.8,
    },
    "myo_rv_outflow": {
        "label": "Right ventricle, outflow wall",
        "fmas": ("FMA49208",), "group": "myocardium", "layer": "core", "ratio": 1.0,
    },
    # -- atria ---------------------------------------------------------------
    "atrium_left": {
        "label": "Left atrium",
        "fmas": ("FMA9306", "FMA84095", "FMA9562", "FMA49292", "FMA9542", "FMA9547"),
        "group": "atrium", "layer": "core", "ratio": 0.8,
    },
    "atrium_right": {
        "label": "Right atrium",
        "fmas": ("FMA49285", "FMA49286", "FMA84094", "FMA49299"),
        "group": "atrium", "layer": "core", "ratio": 0.8,
    },
    # -- great vessels -------------------------------------------------------
    "aorta": {
        "label": "Ascending aorta",
        "fmas": ("FMA23733", "FMA15098"), "group": "great_vessel", "layer": "core",
        "ratio": 0.8,
    },
    "pulm_trunk": {
        "label": "Pulmonary trunk",
        "fmas": ("FMA15086",), "group": "great_vessel", "layer": "core", "ratio": 0.8,
    },
    # -- coronary arteries: the three the model predicts ---------------------
    "artery_LAD": {
        "label": "Left anterior descending",
        "anatomical": "trunk of the anterior interventricular branch with its "
                      "diagonal, second and third anterior, and septal branches",
        "fmas": ("FMA74912", "FMA3860", "FMA3888", "FMA3890", "FMA3893"),
        "group": "artery", "layer": "core", "ratio": 1.0, "vessel": "LAD",
    },
    "artery_LCX": {
        "label": "Left circumflex",
        "anatomical": "trunk of the circumflex branch with the left marginal, first "
                      "posterior ventricular and posterior interventricular branches",
        "fmas": ("FMA74923", "FMA3902", "FMA3914", "FMA79742"),
        "group": "artery", "layer": "core", "ratio": 1.0, "vessel": "LCX",
    },
    "artery_RCA": {
        "label": "Right coronary artery",
        "anatomical": "trunk of the right coronary artery with its conus, sinoatrial "
                      "nodal, marginal, atrioventricular nodal, ventricular and "
                      "atrial branches",
        "fmas": ("FMA3802", "FMA3807", "FMA3815", "FMA3818", "FMA3823", "FMA3829",
                 "FMA3837", "FMA3851"),
        "group": "artery", "layer": "core", "ratio": 1.0, "vessel": "RCA",
    },
    "artery_left_main": {
        "label": "Left coronary stem",
        "anatomical": "stem of the left coronary artery, proximal to the LAD/LCX "
                      "bifurcation and so attributed to neither",
        "fmas": ("FMA4685",), "group": "artery", "layer": "core", "ratio": 1.0,
    },
    # -- valves: the structures behind VHD and the murmur features -----------
    "valve_mitral": {
        "label": "Mitral valve",
        "fmas": ("FMA7242", "FMA7243", "FMA9498"), "group": "valve",
        "layer": "valves", "ratio": 0.3,
    },
    "valve_tricuspid": {
        "label": "Tricuspid valve",
        "fmas": ("FMA7238", "FMA7239", "FMA7240", "FMA85237"), "group": "valve",
        "layer": "valves", "ratio": 0.4,
    },
    "valve_aortic": {
        "label": "Aortic valve",
        "fmas": ("FMA7252", "FMA7253", "FMA7254", "FMA9502"), "group": "valve",
        "layer": "valves", "ratio": 1.0,
    },
    "valve_pulmonary": {
        "label": "Pulmonary valve",
        "fmas": ("FMA7247", "FMA7249", "FMA7250"), "group": "valve",
        "layer": "valves", "ratio": 0.5,
    },
    # -- conduction system: the structures behind the BBB feature ------------
    "cond_left_bundle": {
        "label": "Left bundle branch",
        "anatomical": "anterior, posterior and septal divisions of the left branch "
                      "of the atrioventricular bundle",
        "fmas": ("FMA9488", "FMA9489", "FMA83331"), "group": "conduction",
        "layer": "conduction", "ratio": 0.7,
    },
    "cond_right_bundle": {
        "label": "Right bundle branch",
        "fmas": ("FMA9486",), "group": "conduction", "layer": "conduction",
        "ratio": 0.6,
    },
    "cond_nodes": {
        "label": "Sinoatrial & atrioventricular nodes",
        "anatomical": "sinuatrial node, atrioventricular node, atrioventricular "
                      "bundle and the anterior, middle and posterior internodal tracts",
        "fmas": ("FMA9477", "FMA9478", "FMA9484", "FMA9480", "FMA9482", "FMA9483"),
        "group": "conduction", "layer": "conduction", "ratio": 0.7,
    },
}

# Nodes whose bounding box defines the view centre and scale. The great vessels
# run away from the heart and would shrink the subject if they were included.
CORE_FRAME = [k for k, v in PARTS.items() if v["group"] in ("myocardium", "atrium")]

VESSEL_NODES = {v["vessel"]: k for k, v in PARTS.items() if "vessel" in v}

# three.js wants y up and +z toward the camera, and we look at the anterior
# surface. BodyParts3D is LPS (+x patient left, +y posterior, +z superior), all
# three confirmed from the data: the LV free wall sits at +x and the right atrium
# at -x; the LAD, which runs the anterior groove, is more negative in y than the
# posterior LCX; the heart sits at z ~ 1175-1293 mm up a standing body.
# So: x -> x, z -> y, -y -> z. Determinant +1, i.e. a rotation with no mirroring -
# which matters, because a mirrored heart renders perfectly happily and would put
# the LAD on the wrong side.
TRANSFORM = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]])


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def parse_obj(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ reader: vertices and triangulated faces."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in text.splitlines():
        if line.startswith("v "):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            # Handles "f a", "f a/b", "f a/b/c" and "f a//c"; 1-indexed, may be negative.
            idx = []
            for tok in line.split()[1:]:
                raw = tok.split("/")[0]
                if not raw:
                    continue
                i = int(raw)
                idx.append(i - 1 if i > 0 else len(verts) + i)
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def read_header(path: Path) -> dict[str, str]:
    """The leading comment block: File ID, Concept ID, English name, Bounds."""
    header: dict[str, str] = {}
    with path.open("r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*([^:]+?)\s*:\s*(.*)", line)
            if m:
                header[m.group(1).strip()] = m.group(2).strip()
    return header


def index_parts(directory: Path) -> dict[str, list[Path]]:
    """Map FMA Concept ID -> source files, cross-checking name against header.

    The filename encodes the FMA id, but a filename is a label while the header
    is the file's own claim about itself. Disagreement means the set is not what
    it says it is, so it fails hard rather than guessing.
    """
    if not directory.is_dir():
        raise SystemExit(
            f"Part set not found at {directory}.\n"
            "Expected the BodyParts3D heart OBJ files in that folder."
        )
    index: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*.obj")):
        m = re.search(r"_(FMA\d+)_", path.name)
        if not m:
            raise SystemExit(f"Cannot read an FMA id from the filename: {path.name}")
        from_name = m.group(1)
        from_header = read_header(path).get("Concept ID", "")
        if from_header != from_name:
            raise SystemExit(
                f"{path.name}: filename says {from_name} but the file's own header "
                f"says {from_header!r}. Refusing to guess which is right."
            )
        index.setdefault(from_name, []).append(path)
    return index


def weld(verts: np.ndarray, faces: np.ndarray,
         eps: float = WELD_EPS_MM) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices and drop the degenerate faces that result.

    The source OBJs store per-face duplicates, so a mesh that is one solid piece
    geometrically is a cloud of disconnected triangles topologically. Everything
    downstream that walks edges depends on this running first.
    """
    if len(verts) == 0:
        return verts, faces
    keys = np.round(verts / eps).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    # Average the duplicates rather than picking one, so welding cannot nudge a
    # surface by up to eps in an arbitrary direction.
    welded = np.zeros((len(unique), 3))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    np.add.at(welded, inverse, verts)
    welded /= counts[:, None]
    new_faces = inverse[faces]
    ok = ((new_faces[:, 0] != new_faces[:, 1])
          & (new_faces[:, 1] != new_faces[:, 2])
          & (new_faces[:, 0] != new_faces[:, 2]))
    return welded, new_faces[ok]


def merge_group(fmas: tuple[str, ...], index: dict[str, list[Path]],
                parts_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Concatenate every file of every FMA in the group, then weld."""
    chunks_v: list[np.ndarray] = []
    chunks_f: list[np.ndarray] = []
    names: list[str] = []
    offset = 0
    for fma in fmas:
        paths = index.get(fma)
        if not paths:
            raise SystemExit(f"{fma} is not present in {parts_dir}")
        for path in paths:
            v, f = parse_obj(path.read_text(errors="replace"))
            chunks_v.append(v)
            chunks_f.append(f + offset)
            offset += len(v)
            names.append(path.name)
    verts, faces = weld(np.vstack(chunks_v), np.vstack(chunks_f))
    return verts, faces, names


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def vertex_cluster_decimate(verts: np.ndarray, faces: np.ndarray,
                            target: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid-cluster vertices until the triangle count fits the budget.

    Deduplication keys on the sorted index triple but keeps each triangle in its
    original order, because sorting the stored indices randomises winding - and
    with winding goes the normal. That only stopped mattering while every surface
    was translucent and double-sided.
    """
    if len(faces) <= target:
        return verts, faces
    extent = verts.max(axis=0) - verts.min(axis=0)
    diag = float(np.linalg.norm(extent))
    reps, new_faces = verts, faces
    for divisions in (200, 160, 128, 100, 80, 64, 50, 40, 32, 25, 20, 16, 12):
        cell = diag / divisions
        keys = np.floor((verts - verts.min(axis=0)) / cell).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                       return_counts=True)
        inverse = inverse.reshape(-1)
        reps = np.zeros((len(counts), 3))
        np.add.at(reps, inverse, verts)
        reps /= counts[:, None]
        candidate = inverse[faces]
        ok = ((candidate[:, 0] != candidate[:, 1])
              & (candidate[:, 1] != candidate[:, 2])
              & (candidate[:, 0] != candidate[:, 2]))
        candidate = candidate[ok]
        _, keep = np.unique(np.sort(candidate, axis=1), axis=0, return_index=True)
        new_faces = candidate[np.sort(keep)]
        if len(new_faces) <= target:
            return reps, new_faces
    return reps, new_faces


def contract_to_axis(verts: np.ndarray, faces: np.ndarray,
                     iterations: int = CONTRACT_ITERATIONS,
                     lam: float = CONTRACT_LAMBDA) -> np.ndarray:
    """Laplacian mesh contraction: collapse a tube toward its own medial axis.

    Repeatedly moving every vertex toward the mean of its neighbours shrinks a
    tubular mesh onto its centreline while keeping its length, which is exactly
    what the narrowing morph target needs. It replaces the geodesic centreline
    fit this script used to do, which was the fragile step in the old pipeline:
    on these meshes that fit returned a 47 mm "left coronary stem" against a true
    length under 15 mm.
    """
    n = len(verts)
    if n == 0 or len(faces) == 0:
        return verts.copy()
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    adjacency = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    p = verts.copy()
    for _ in range(iterations):
        p += lam * (adjacency @ p / degree[:, None] - p)
    return p


def local_radius(verts: np.ndarray, faces: np.ndarray, k: int = 40) -> np.ndarray:
    """Per-vertex lumen radius: half the distance to the facing wall opposite.

    For each vertex, look at its nearest neighbours and keep those whose normal
    points back at it (dot < -0.5, i.e. the other side of the tube rather than
    further along the same side); half the nearest such distance is the local
    radius. Measured this way the three arteries come out at a median 0.59 mm
    (LAD), 0.75 mm (LCX) and 0.72 mm (RCA), which is the calibre a coronary
    angiogram reports, so the number means something outside this file.
    """
    normals = vertex_normals(verts, faces)
    tree = cKDTree(verts)
    radius = np.full(len(verts), np.inf)

    # Widen the neighbourhood until most vertices have found the far wall. A
    # fixed k is not enough: on a finely tessellated tube the nearest few dozen
    # neighbours all sit on the SAME side of the lumen, so every vertex comes
    # back unresolved and the whole mesh falls through to the fallback.
    for width in (32, 128, 512, 2048):
        width = min(width, len(verts))
        distances, neighbours = tree.query(verts, k=width)
        facing_back = (normals[neighbours[:, 1:]] * normals[:, None, :]).sum(-1) < -0.5
        candidate = np.where(facing_back, distances[:, 1:], np.inf)
        radius = candidate.min(axis=1) / 2.0
        if np.isfinite(radius).mean() > 0.8 or width == len(verts):
            break

    # A vertex with no opposing wall in range - a free edge, or a genuinely flat
    # sheet - takes the mesh median rather than infinity.
    unknown = ~np.isfinite(radius)
    if unknown.all():
        extent = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        return np.full(len(verts), extent * 0.01)
    radius[unknown] = float(np.median(radius[~unknown]))
    return radius


def narrowing_delta(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex displacement for the stenosis morph target.

    Direction from the contraction, distance from the measured local radius, so
    at full influence the vessel loses NARROW_FRACTION of its calibre wherever
    you look along it and can never turn inside out. Taking the distance from the
    contraction instead - which is what this did first - collapsed thin segments
    through their own axis while leaving the thick proximal ones barely changed.
    """
    raw = contract_to_axis(verts, faces) - verts
    magnitude = np.linalg.norm(raw, axis=1)
    moving = magnitude > 0
    if not moving.any():
        return raw
    direction = np.zeros_like(raw)
    direction[moving] = raw[moving] / magnitude[moving, None]
    limit = local_radius(verts, faces) * NARROW_FRACTION
    return direction * np.minimum(magnitude, limit)[:, None]


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)  # area-weighted by construction
    for k in range(3):
        np.add.at(normals, faces[:, k], face_n)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


# ---------------------------------------------------------------------------
# Minimal glTF 2.0 binary writer
# ---------------------------------------------------------------------------
def write_glb(path: Path, meshes: dict[str, dict]) -> None:
    """Write named triangle meshes to a .glb.

    Each entry is ``{"verts", "faces"}`` plus an optional ``"morph"`` array of
    per-vertex position deltas, written as glTF morph target 0 at weight 0.
    """
    blob = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    gltf_meshes: list[dict] = []
    nodes: list[dict] = []

    def pad(to: int = 4) -> None:
        while len(blob) % to:
            blob.append(0)

    def push(data: np.ndarray, target: int | None) -> int:
        pad()
        offset = len(blob)
        blob.extend(data.tobytes())
        view = {"buffer": 0, "byteOffset": offset, "byteLength": data.nbytes}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    for name, entry in meshes.items():
        verts, faces = entry["verts"], entry["faces"]
        pos = verts.astype(np.float32)
        nrm = vertex_normals(verts, faces).astype(np.float32)
        # uint16 indices where the mesh fits, which is every node here. At these
        # triangle counts indices are the single largest thing in the file, so
        # halving them halves the download.
        narrow = len(pos) <= 65535
        idx = faces.astype(np.uint16 if narrow else np.uint32).reshape(-1)

        acc_pos = len(accessors)
        accessors.append({
            "bufferView": push(pos, 34962), "componentType": 5126,
            "count": int(len(pos)), "type": "VEC3",
            "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist(),
        })
        acc_nrm = len(accessors)
        accessors.append({
            "bufferView": push(nrm, 34962), "componentType": 5126,
            "count": int(len(nrm)), "type": "VEC3",
        })
        acc_idx = len(accessors)
        accessors.append({
            "bufferView": push(idx, 34963),
            "componentType": 5123 if narrow else 5125,
            "count": int(len(idx)), "type": "SCALAR",
        })

        primitive = {
            "attributes": {"POSITION": acc_pos, "NORMAL": acc_nrm},
            "indices": acc_idx, "mode": 4, "material": 0,
        }
        mesh: dict = {"name": name, "primitives": [primitive]}

        morph = entry.get("morph")
        if morph is not None:
            delta = morph.astype(np.float32)
            acc_morph = len(accessors)
            accessors.append({
                "bufferView": push(delta, 34962), "componentType": 5126,
                "count": int(len(delta)), "type": "VEC3",
                "min": delta.min(axis=0).tolist(), "max": delta.max(axis=0).tolist(),
            })
            primitive["targets"] = [{"POSITION": acc_morph}]
            mesh["weights"] = [0.0]
            mesh["extras"] = {"targetNames": ["stenosis"]}

        gltf_meshes.append(mesh)
        nodes.append({"name": name, "mesh": len(gltf_meshes) - 1})

    pad()
    gltf = {
        "asset": {"version": "2.0", "generator": "build_anatomy.py",
                  "copyright": ATTRIBUTION},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
        # One neutral material; the viewer assigns the real ones by node name, so
        # tissue colours live with the rest of the palette instead of in a binary.
        "materials": [{
            "name": "tissue",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.78, 0.44, 0.42, 1.0],
                "metallicFactor": 0.0, "roughnessFactor": 0.7,
            },
            "doubleSided": True,
        }],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(blob)
    bin_bytes += b"\x00" * ((4 - len(bin_bytes) % 4) % 4)

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        fh.write(json_bytes)
        fh.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
        fh.write(bin_bytes)


# ---------------------------------------------------------------------------
# Anatomical checks
# ---------------------------------------------------------------------------
def check_anatomy(centroids: dict[str, np.ndarray]) -> list[str]:
    """Assert the scene is anatomically right before anyone looks at a picture.

    All in display space: +x patient left, +y superior, +z anterior. A mirrored or
    mis-rotated heart renders perfectly happily and would put the LAD on the wrong
    side, so this is checked numerically rather than by eye.
    """
    lad, lcx, rca = (centroids["artery_LAD"], centroids["artery_LCX"],
                     centroids["artery_RCA"])
    checks = [
        ("RCA lies to the patient's right of the LAD", rca[0] < lad[0]),
        ("RCA lies to the patient's right of the LCX", rca[0] < lcx[0]),
        ("LAD is anterior to the LCX", lad[2] > lcx[2]),
        ("aorta is superior to the mitral valve plane",
         centroids["aorta"][1] > centroids["valve_mitral"][1]),
        ("left atrium is superior to the left ventricle",
         centroids["atrium_left"][1] > centroids["myo_lv_free"][1]),
        ("left ventricle lies to the patient's left of the right ventricle",
         centroids["myo_lv_free"][0] > centroids["myo_rv_inflow"][0]),
        ("pulmonary trunk is anterior to the aorta",
         centroids["pulm_trunk"][2] > centroids["aorta"][2]),
        # The two checks that matter for the BBB feature: a swapped pair would
        # light the wrong bundle for a patient with LBBB.
        ("left bundle branch lies to the patient's left of the right bundle",
         centroids["cond_left_bundle"][0] > centroids["cond_right_bundle"][0]),
        ("the nodes sit superior to both bundle branches",
         centroids["cond_nodes"][1] > max(centroids["cond_left_bundle"][1],
                                          centroids["cond_right_bundle"][1])),
    ]
    for name, ok in checks:
        print(f"    [{'ok' if ok else 'FAIL'}] {name}")
    return [name for name, ok in checks if not ok]


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="print source mesh statistics and exit")
    ap.add_argument("--parts-dir", type=Path, default=PARTS_DIR,
                    help="folder holding the BodyParts3D heart OBJ files")
    args = ap.parse_args()

    index = index_parts(args.parts_dir)
    n_files = sum(len(v) for v in index.values())
    print(f"[*] {args.parts_dir}")
    print(f"    {n_files} files, {len(index)} distinct FMA concepts")

    print("\n[*] reading, merging and welding source meshes")
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sources: dict[str, list[str]] = {}
    for name, meta in PARTS.items():
        verts, faces, files = merge_group(meta["fmas"], index, args.parts_dir)
        raw[name] = (verts, faces)
        sources[name] = files
        extent = verts.max(axis=0) - verts.min(axis=0)
        print(f"    {name:18s} {len(files):2d} file(s) {len(verts):7,d} verts "
              f"{len(faces):7,d} tris  extent="
              f"({extent[0]:.0f}, {extent[1]:.0f}, {extent[2]:.0f}) mm")

    if args.report:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- shared normalisation -----------------------------------------------
    core = np.vstack([raw[k][0] for k in CORE_FRAME])
    center = (core.max(axis=0) + core.min(axis=0)) / 2
    scale = 2.0 / float(np.linalg.norm(core.max(axis=0) - core.min(axis=0)))
    assert np.isclose(np.linalg.det(TRANSFORM), 1.0), "transform mirrors the anatomy"

    def normalise(points: np.ndarray) -> np.ndarray:
        return ((points - center) * scale) @ TRANSFORM.T

    print(f"\n[*] normalisation: centre={np.round(center, 1).tolist()} mm, "
          f"scale={scale:.5f}")
    print("    display axes: +x = patient left, +y = superior, +z = anterior")

    # --- build every node ----------------------------------------------------
    print("\n[*] building nodes")
    meshes: dict[str, dict] = {}
    manifest: dict[str, dict] = {}
    centroids: dict[str, np.ndarray] = {}
    for name, meta in PARTS.items():
        verts, faces = raw[name]
        target = max(64, int(len(faces) * meta["ratio"]))
        verts, faces = vertex_cluster_decimate(verts, faces, target)
        display = normalise(verts)

        entry: dict = {"verts": display, "faces": faces}
        morph_mm = 0.0
        if meta["group"] == "artery":
            # Contract in millimetre space, then rotate and scale the delta the
            # same way as the positions so it stays a matching displacement.
            delta = narrowing_delta(verts, faces)
            morph_mm = float(np.linalg.norm(delta, axis=1).mean())
            entry["morph"] = (delta * scale) @ TRANSFORM.T
        meshes[name] = entry

        centroid = display.mean(axis=0)
        centroids[name] = centroid

        # Label anchor. For an artery the label names the TRUNK, so it is placed
        # mid-trunk: the first FMA in the group is the trunk and the rest are its
        # branches, and anchoring to the group as a whole put the label out on
        # the tip of the furthest diagonal, metres of screen away from the
        # vessel it names. Mid-trunk is also more likely to be facing the camera
        # than a tip is.
        anchor_source = display
        if meta["group"] == "artery" and len(meta["fmas"]) > 1:
            trunk, _, _ = merge_group(meta["fmas"][:1], index, args.parts_dir)
            anchor_source = normalise(trunk)
        anchor = anchor_source[
            int(np.argmin(np.linalg.norm(
                anchor_source - anchor_source.mean(axis=0), axis=1)))
        ]

        manifest[name] = {
            "label": meta["label"],
            "anatomical_name": meta.get("anatomical", meta["label"]),
            "group": meta["group"],
            "layer": meta["layer"],
            "vessel": meta.get("vessel"),
            "fma_ids": list(meta["fmas"]),
            "source_files": sources[name],
            "triangles": int(len(faces)),
            "anchor": np.round(anchor, 5).tolist(),
            "centroid": np.round(centroid, 5).tolist(),
            "radius": round(float(np.linalg.norm(display - centroid, axis=1).max()), 5),
        }
        note = f"  morph {morph_mm:.2f} mm mean" if morph_mm else ""
        print(f"    {name:18s} {len(raw[name][1]):7,d} -> {len(faces):6,d} tris{note}")

    # --- anatomical checks ---------------------------------------------------
    print("\n[*] anatomical checks (display space)")
    failures = check_anatomy(centroids)
    if failures:
        raise SystemExit(
            "Anatomy check failed: " + "; ".join(failures) +
            "\nRefusing to write an asset that is anatomically wrong."
        )

    # --- write ---------------------------------------------------------------
    glb = OUT_DIR / "anatomy.glb"
    write_glb(glb, meshes)
    total = sum(len(e["faces"]) for e in meshes.values())
    default_on = sum(len(meshes[k]["faces"]) for k, m in PARTS.items()
                     if m["layer"] == "core")
    print(f"\n[+] wrote {glb} ({glb.stat().st_size:,} bytes, {total:,} triangles, "
          f"{default_on:,} on by default)")

    (OUT_DIR / "vessels.json").write_text(json.dumps({
        "attribution": ATTRIBUTION,
        "source": SOURCE_SET,
        "note": ("Every node is BodyParts3D mesh geometry, decimated but not "
                 "reconstructed. Coronary arteries carry morph target 0, a uniform "
                 "contraction toward the vessel's own medial axis used to show "
                 "predicted stenosis; it is a severity cue and implies no lesion "
                 "position. This donor is left-dominant, so the posterior "
                 "interventricular branch is grouped with the LCX."),
        "normalisation": {
            "center_mm": np.round(center, 4).tolist(),
            "scale": scale,
            "transform": TRANSFORM.tolist(),
            "axes": "+x = patient left, +y = superior, +z = anterior",
            "source_axes": "BodyParts3D LPS: +x left, +y posterior, +z superior",
        },
        "narrow_fraction": NARROW_FRACTION,
        "vessel_nodes": VESSEL_NODES,
        "nodes": manifest,
    }, indent=1), encoding="utf-8")
    print(f"[+] wrote {OUT_DIR / 'vessels.json'}")

    lines = [
        ATTRIBUTION,
        "",
        f"Source set: {SOURCE_SET}",
        f"            {n_files} OBJ files, {len(index)} distinct FMA concepts",
        "",
        "PROVENANCE",
        "These files were supplied locally in 'heart-beat parts/' rather than",
        "downloaded by this script. Unlike the BodyParts3D PART-OF archive they",
        "carry no embedded licence notice of their own - their headers record only",
        "File ID, Representation ID, Concept ID, English name, bounds and volume -",
        "so the credit line above is the one the BodyParts3D licence page specifies.",
        "",
        "Structures used (node -> FMA Concept IDs -> source files):",
    ]
    for name, meta in PARTS.items():
        lines.append(f"  {name}")
        lines.append(f"      {meta['label']}  [{', '.join(meta['fmas'])}]")
        for filename in sources[name]:
            lines.append(f"      - {filename}")
    (OUT_DIR / "ATTRIBUTION.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[+] wrote {OUT_DIR / 'ATTRIBUTION.txt'}")


if __name__ == "__main__":
    main()
