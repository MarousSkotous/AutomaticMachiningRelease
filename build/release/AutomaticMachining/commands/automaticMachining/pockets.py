"""Recognizes millable pockets and builds 2D Pocket operations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import adsk.cam
import adsk.core
import adsk.fusion

from ...lib import fusionAddInUtils as futil
from . import geometry, tool_library
from .cam_utils import set_expression, set_safe_heights, set_zero_stock_to_leave

# A recognized pocket bottom narrower than this (cm) in its shortest planar
# direction, or shallower than this (cm) in depth, is not machinable and is
# skipped rather than treated as a real feature.
MIN_FEATURE_DIMENSION_CM = 1e-6

# Tolerance (cm) used when deciding whether a pocket tool "fits" inside a
# pocket's minimum span.
TOOL_FIT_EPSILON_CM = 1e-6

# A helper sketch used by a 2D Pocket must be parallel to the setup XY plane.
# This strict tolerance accepts import round-off but not a genuinely angled
# face.
SKETCH_PLANE_ALIGNMENT_TOLERANCE_DEG = 0.001
SKETCH_PLANE_ALIGNMENT_COS_THRESHOLD = math.cos(
    math.radians(SKETCH_PLANE_ALIGNMENT_TOLERANCE_DEG)
)

# How many adjacency hops _floor_faces_via_adjacency will search past a
# pocket's known wall/cap faces before giving up. A real floor can sit behind
# a corner-radius fillet or a chamfer rather than directly touching the wall
# (confirmed on real output — every wall's immediate neighbor was either the
# panel's own bottom face or a non-planar fillet, never the true floor), so
# more than one hop is needed; bounded to avoid wandering across the model.
# Raised from 4 to 8 after real output on bad_zij_schuif_breed_dL/dR showed
# the search still discovering new, unresolved transition faces (chamfers/
# fillets) at hop 4, the exact hop the old limit cut it off at — the true
# floor was one or more hops beyond where the search was forced to stop.
_FLOOR_ADJACENCY_MAX_HOPS = 8

# Sanity bound (cm) on how far a pocket wall/cap face's nearest edge may sit
# from the sketch plane and still be trusted as that face's "top" edge (see
# _closest_edge_to_plane). This is deliberately generous, not a precision
# tolerance: real parts can have the true opening edge sit a genuine
# sub-millimeter offset below the chosen sketch plane's face (confirmed on
# real Fusion output, not import round-off), so the closest edge is trusted
# rather than requiring a tight exact match. This bound only guards against
# picking an edge from a face that turns out not to border the sketch plane
# at all.
PLANE_EDGE_MAX_DISTANCE_CM = 0.5

# Distance (cm) within which two sketch curve endpoints are treated as the
# same point when closing a boundary loop. Real Fusion output showed that
# Point3D.isEqualTo's exact-match comparison fails to recognize two edges'
# endpoints as coincident even when they came from the same real BRep vertex
# (each Sketch.project() call can produce its own independent SketchPoint),
# so endpoints are merged by distance instead of exact equality.
ENDPOINT_MERGE_TOLERANCE_CM = 0.001

# An open/blind groove (see OpenPocketFeature) whose floor width is this wide
# (mm) is a specific customer feature cut with a saw-blade-shaped Slot Mill
# tool tilted 90 degrees instead of milled like an ordinary pocket. Width is
# measured the same way as any other floor (_face_average_width_cm), so real
# parts land within a fraction of a mm of the nominal 3.5 mm rather than
# exactly on it.
SAW_SLOT_WIDTH_MM = 3.5
SAW_SLOT_WIDTH_TOLERANCE_MM = 0.3


@dataclass
class PocketFeature:
    bottom_faces: List[adsk.fusion.BRepFace]
    depth_cm: float
    minimum_span_cm: float
    # A floor recovered from an edge-open recognized pocket must be passed to
    # CAM as an explicitly oriented boundary chain. Ordinary enclosed pockets
    # keep their proven face selection.
    use_floor_boundary: bool = False
    # Face ids reported by RecognizedPocket for an edge-open feature. They are
    # retained so the operation builder can distinguish real pocket walls
    # from the exterior side face at the pocket mouth without relying on a
    # part name, handedness or a fixed global direction.
    recognized_boundary_face_ids: Optional[Set[int]] = None


@dataclass
class ThroughSlotFeature:
    faces: List[adsk.fusion.BRepFace]
    depth_cm: float
    minimum_span_cm: float


@dataclass
class OpenPocketFeature:
    """A blind pocket whose recognized region touches the part's outer
    silhouette (e.g. a groove routed in from an edge, open at one end).

    Legacy fallback representation for a pocket without a usable floor.
    Automatic recognition now refuses to populate this path because its
    machining side and mouth cannot be proven from topology; such a feature
    is reported for manual machining instead.
    """

    faces: List[adsk.fusion.BRepFace]
    depth_cm: float
    minimum_span_cm: float


@dataclass
class SawSlotFeature:
    """A narrow open/blind groove (see OpenPocketFeature) exactly
    SAW_SLOT_WIDTH_MM wide, machined with a saw-blade-shaped Slot Mill tool
    tilted 90 degrees instead of an end mill.

    wall_face is one of the groove's own planar side walls. It is used as the
    reference face for Fusion CAM's per-operation tool orientation ("Select Z
    axis/plane and X axis" mode), which is how the tilted blade's rotation
    axis is derived from real part geometry — see _create_saw_slot_operation.
    """

    floor_faces: List[adsk.fusion.BRepFace]
    wall_face: adsk.fusion.BRepFace
    depth_cm: float
    width_cm: float
    # Set only for one arm of a "dogbone" groove (see _dogbone_saw_arms): the
    # single boundary edge to follow, when the floor's own longest-edge pick
    # would otherwise be ambiguous or wrong (a groove locally widened around
    # a hardware-clearance hole splits each wall into two same-length
    # collinear segments — one per arm — so "the longest edge" no longer
    # identifies a unique, full-length path). None keeps the original
    # longest-edge selection in _create_saw_slot_operation for every plain
    # groove, unchanged.
    path_edge: Optional[adsk.fusion.BRepEdge] = None


@dataclass
class NestedThroughFeature:
    """A closed inner boundary already present on a recognized pocket's own
    floor face — e.g. a counterbore's narrower through-opening in the middle
    of its shallow outer recess.

    Confirmed on real output (Hou_plafond_L3_achter_Hefdak_Boxer_OH-00198):
    RecognizedPocket returned the whole stepped shape as a single blind
    pocket with only the outer, shallow rebate as its floor face — the
    narrower through-opening in the middle of that same floor face (present
    as the face's own inner BRepLoop) was never surfaced as its own feature
    by RecognizedPocket and was left completely unmachined. Machined as a
    closed 2D Contour cutting from the floor's own depth down through the
    stock bottom, using the loop's real edges rather than a fixed
    circle/pocket shape, since the opening can be any shape.
    """

    floor_face: adsk.fusion.BRepFace
    inner_loop: adsk.fusion.BRepLoop
    depth_cm: float
    span_cm: float


BoundaryFeature = Union[ThroughSlotFeature, OpenPocketFeature]


PocketRecognitionResult = Tuple[
    List[PocketFeature],
    List[ThroughSlotFeature],
    List[OpenPocketFeature],
    List[SawSlotFeature],
    List[str],
]


def recognize_pocket_features(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
    z_axis_index: int,
    maximum_drill_diameter_mm: float = float('inf'),
) -> PocketRecognitionResult:
    attack_vector = z_axis.copy()
    attack_vector.scaleBy(-1.0)
    try:
        recognized = adsk.cam.RecognizedPocket.recognizePockets(body, attack_vector)
    except Exception as exc:
        return [], [], [], [f'Pocket recognition failed: {exc}']

    features: List[PocketFeature] = []
    through_features: List[ThroughSlotFeature] = []
    open_features: List[OpenPocketFeature] = []
    saw_features: List[SawSlotFeature] = []
    seen_face_sets = set()
    skipped_nonflat = 0
    skipped_degenerate = 0
    skipped_open_through = 0
    for pocket in recognized:
        circular_diameter_cm = _circular_pocket_diameter_cm(pocket)
        if (
            circular_diameter_cm is not None
            and circular_diameter_cm * 10.0 <= maximum_drill_diameter_mm + 1e-6
        ):
            # Circular features at or below the configured limit are handled
            # by drilling. Larger circles continue through the normal blind or
            # through-pocket path and are milled with the selected pocket tool.
            continue
        if pocket.isThrough:
            if _is_open_through_region(pocket):
                # Pocket recognition also returns open regions cut into the
                # outside silhouette.  The outer 2D Contour already follows
                # those edges; clearing the entire open region as a pocket is
                # redundant and can place linking/entry moves outside the part.
                skipped_open_through += 1
                continue
            through_pocket_faces = _recognized_pocket_faces(pocket)
            minimum_span_cm = _minimum_planar_span(through_pocket_faces, z_axis_index)
            if minimum_span_cm <= MIN_FEATURE_DIMENSION_CM:
                skipped_degenerate += 1
                continue
            through_features.append(
                ThroughSlotFeature(
                    faces=through_pocket_faces,
                    depth_cm=pocket.depth,
                    minimum_span_cm=minimum_span_cm,
                )
            )
            continue
        if _is_open_blind_pocket(pocket):
            # Not through, but the region still touches the outer silhouette
            # (e.g. a groove routed in from an edge). RecognizedPocket may
            # either include its real floor or expose only walls/caps. Prefer
            # a directly supplied floor; otherwise find one via BRep adjacency
            # and machine it exactly like a normal closed pocket, reusing the
            # proven face-based path.
            # This matters for a bending groove (e.g. a curved slide track):
            # its two walls can have very different lengths around the bend,
            # which made reconstructing a closed boundary sketch from just
            # the walls unreliable (see AI_NOTES.md) — the real floor,
            # wherever it's split into by the bend, doesn't have that problem.
            open_pocket_faces = _recognized_pocket_faces(pocket)
            # This pocket's OWN local top, not the body-global largest face:
            # on a multi-level part those can differ by centimeters, which
            # silently offset every depth measurement below and rejected the
            # real floor at every hop (confirmed on real output — see
            # _best_matching_z_face).
            top_face = _best_matching_z_face(body, z_axis, open_pocket_faces)
            # `isClosed=False` does not universally mean that Fusion omitted
            # the floor from faces/sharedFaces.  On
            # HPL_badkamer_kunststof_plaathouder_R the list contains the real
            # 27.2917 cm² floor (the same face area that the mirrored L part
            # already machines successfully), but the old open-pocket branch
            # never inspected it and instead projected one of its edges into
            # the unsafe open-chain fallback.  Prefer any directly reported,
            # attack-normal face at the expected local depth.  Only walk
            # adjacency when Fusion genuinely did omit the floor.
            floor_faces = (
                _floor_faces_at_expected_depth(
                    open_pocket_faces, attack_vector, top_face, pocket.depth
                )
                if top_face is not None
                else []
            )
            if floor_faces:
                futil.log(
                    'Open-pocket floor found directly in recognized faces: '
                    + ', '.join(
                        f'{face.tempId}: area {face.area:.4f} cm^2' for face in floor_faces
                    )
                    + '.',
                    force_console=True,
                )
            elif top_face is not None:
                floor_faces = _floor_faces_via_adjacency(
                    open_pocket_faces, attack_vector, top_face, pocket.depth
                )
            if not floor_faces:
                _log_floor_adjacency_rejections(
                    open_pocket_faces, attack_vector, top_face, pocket.depth
                )
            if floor_faces:
                # A groove that steps down partway along (e.g. a rabbet/
                # "polodrážka") can have floor fragments at two different
                # depths. pocket.depth is a single value and cannot be
                # trusted for all of them — measure each fragment's own
                # depth from the top surface and group by that instead of
                # assuming one uniform depth, so a shallower step is not cut
                # too deep (or a deeper one left uncut).
                top_plane = adsk.core.Plane.cast(top_face.geometry)
                depth_groups: Dict[int, List[adsk.fusion.BRepFace]] = {}
                for floor_face in floor_faces:
                    offset = top_plane.origin.vectorTo(floor_face.pointOnFace)
                    depth_cm = abs(offset.dotProduct(top_plane.normal))
                    depth_microns = int(round(depth_cm * 10.0 * 1000.0))
                    depth_groups.setdefault(depth_microns, []).append(floor_face)
                for depth_microns, group_faces in depth_groups.items():
                    depth_cm = depth_microns / 10000.0
                    # An axis-aligned bounding box (per fragment or combined)
                    # badly overstates the width of a fragment that sits
                    # diagonally in X/Y — a thin diagonal strip's bounding
                    # box can be nearly square. A bending groove's tightest
                    # point is exactly where a fragment is most likely to be
                    # diagonal (the turn itself), so that's where this most
                    # needs to be orientation-independent: use each
                    # fragment's own area / longest edge as an average-width
                    # estimate instead, and take the narrowest fragment.
                    minimum_span_cm = min(_face_average_width_cm(face) for face in group_faces)
                    if minimum_span_cm <= MIN_FEATURE_DIMENSION_CM or depth_cm <= MIN_FEATURE_DIMENSION_CM:
                        futil.log(
                            'Open-pocket floor fragment rejected as degenerate: '
                            f'span {minimum_span_cm * 10.0:.4f} mm, depth {depth_cm * 10.0:.4f} mm, '
                            f'{len(group_faces)} face(s): '
                            + ', '.join(f'{face.tempId}' for face in group_faces) + '.',
                            force_console=True,
                        )
                        skipped_degenerate += 1
                        continue
                    width_mm = minimum_span_cm * 10.0
                    if abs(width_mm - SAW_SLOT_WIDTH_MM) <= SAW_SLOT_WIDTH_TOLERANCE_MM:
                        wall_face = _wall_face_for_floor(
                            group_faces[0], attack_vector, z_axis_index, depth_cm
                        )
                        if wall_face is not None:
                            saw_features.append(
                                SawSlotFeature(
                                    floor_faces=group_faces,
                                    wall_face=wall_face,
                                    depth_cm=depth_cm,
                                    width_cm=minimum_span_cm,
                                )
                            )
                            continue
                        futil.log(
                            f'Saw slot candidate at {width_mm:.4f} mm has no adjacent planar '
                            'wall face for tool orientation; machining as an ordinary pocket '
                            'instead.',
                            adsk.core.LogLevels.WarningLogLevel,
                            force_console=True,
                        )
                    dogbone_arms = [
                        arm
                        for face in group_faces
                        for arm in _dogbone_saw_arms(face, attack_vector, depth_cm, z_axis_index)
                    ]
                    if dogbone_arms:
                        saw_features.extend(dogbone_arms)
                        continue
                    # Which direction of this floor's exact outer boundary
                    # keeps machining inside the floor region is resolved
                    # later from its ordered co-edges and the setup tool axis — see
                    # _resolve_open_floor_orientation in
                    # _create_pocket_operation. Recognition only records that
                    # this floor needs that resolution.
                    group_floor_ids = {face.tempId for face in group_faces}
                    features.append(
                        PocketFeature(
                            bottom_faces=group_faces,
                            depth_cm=depth_cm,
                            minimum_span_cm=minimum_span_cm,
                            use_floor_boundary=True,
                            recognized_boundary_face_ids={
                                face.tempId
                                for face in open_pocket_faces
                                if face.tempId not in group_floor_ids
                            },
                        )
                    )
                continue
            minimum_span_cm = _minimum_planar_span(open_pocket_faces, z_axis_index)
            if minimum_span_cm <= MIN_FEATURE_DIMENSION_CM or pocket.depth <= MIN_FEATURE_DIMENSION_CM:
                futil.log(
                    'Open blind pocket rejected as degenerate (no floor found via adjacency): '
                    f'span {minimum_span_cm * 10.0:.4f} mm, depth {pocket.depth * 10.0:.4f} mm, '
                    f'{len(open_pocket_faces)} wall/cap face(s).',
                    force_console=True,
                )
                skipped_degenerate += 1
                continue
            warnings.append(
                'Open pocket has no verifiable floor boundary, so its machining side and '
                'mouth cannot be proven safely — MACHINE MANUALLY.'
            )
            continue
        bottom_faces = _pocket_bottom_faces(pocket, attack_vector)
        if not bottom_faces:
            skipped_nonflat += 1
            continue
        face_key = tuple(sorted(face.tempId for face in bottom_faces))
        if face_key in seen_face_sets:
            continue
        seen_face_sets.add(face_key)
        # An axis-aligned bounding box overstates the true tool-fit width of a
        # floor that locally bulges out (e.g. a saw-slot groove widened by a
        # circular hardware-clearance area around a hole in its middle —
        # confirmed on real output: RecognizedPocket returned such a groove as
        # a genuine isClosed=True pocket, whose _minimum_planar_span measured
        # 7.2 cm end-to-end across the bulge even though the groove's own
        # straight arms on either side are only ~3.5-5 mm wide). That
        # oversized estimate let _select_tool_size below believe an ordinary
        # pocket end mill fit, when it cannot reach into the narrow arms,
        # silently leaving part of the groove uncut. _face_average_width_cm
        # (already used for the same reason on the open-pocket floor
        # fragments below) catches this: take whichever of the two measures
        # is smaller, since either can be the true pinch point depending on
        # the floor's shape.
        minimum_span_cm = min(
            _minimum_planar_span(bottom_faces, z_axis_index),
            min(_face_average_width_cm(face) for face in bottom_faces),
        )
        if minimum_span_cm <= MIN_FEATURE_DIMENSION_CM or pocket.depth <= MIN_FEATURE_DIMENSION_CM:
            skipped_nonflat += 1
            continue
        dogbone_arms = [
            arm
            for face in bottom_faces
            for arm in _dogbone_saw_arms(face, attack_vector, pocket.depth, z_axis_index)
        ]
        if dogbone_arms:
            # Same reasoning as the open-pocket branch above: a groove
            # widened locally around a hole comes back from RecognizedPocket
            # as one ordinary (isClosed=True) pocket instead of the missing
            # full-span groove _find_full_span_saw_slots exists for — but
            # it's still a real saw slot, just interrupted by the bulge.
            saw_features.extend(dogbone_arms)
            continue
        features.append(
            PocketFeature(
                bottom_faces=bottom_faces,
                depth_cm=pocket.depth,
                minimum_span_cm=minimum_span_cm,
            )
        )

    if skipped_nonflat or skipped_degenerate or skipped_open_through:
        # A rejected floor can mean the true floor face belongs to a
        # different RecognizedPocket entry entirely (e.g. filtered out
        # earlier as an open-through/circular region). Dumping every
        # recognized pocket here shows whether that floor is hiding
        # elsewhere in the same recognizePockets() result instead of only
        # showing the one pocket that failed.
        _log_recognition_summary(recognized)

    warnings: List[str] = []
    if skipped_nonflat:
        warnings.append(f'Pockets without a usable planar bottom skipped: {skipped_nonflat}.')
    if skipped_degenerate:
        warnings.append(
            f'Pocket(s)/slot(s) too narrow or shallow to machine skipped: {skipped_degenerate} '
            '(see the Text Commands console for which one).'
        )
    if skipped_open_through:
        warnings.append(
            f'Open through region(s) left to the outer contour: {skipped_open_through}.'
        )

    consumed_face_ids = {face.tempId for feature in features for face in feature.bottom_faces}
    consumed_face_ids.update(face.tempId for feature in through_features for face in feature.faces)
    consumed_face_ids.update(face.tempId for feature in open_features for face in feature.faces)
    consumed_face_ids.update(face.tempId for feature in saw_features for face in feature.floor_faces)
    extra_saw_features, skipped_full_span_candidates = _find_full_span_saw_slots(
        body, z_axis, z_axis_index, attack_vector, consumed_face_ids
    )
    if extra_saw_features:
        futil.log(
            f'{len(extra_saw_features)} saw slot(s) found via direct BRep scan (not returned by '
            'RecognizedPocket — the groove touches the outer silhouette on every side, e.g. runs '
            'edge-to-edge, which RecognizedPocket does not anchor a pocket region to).',
            force_console=True,
        )
        saw_features.extend(extra_saw_features)
    if skipped_full_span_candidates:
        warnings.append(
            f'{skipped_full_span_candidates} recessed groove(s) at the saw-slot depth were not '
            f'machined automatically: their average width fell outside the '
            f'{SAW_SLOT_WIDTH_MM:g}±{SAW_SLOT_WIDTH_TOLERANCE_MM:g} mm tolerance or no usable wall '
            'face was found (often a local widening, e.g. a hardware-clearance bulge around a hole '
            'in the middle of the groove). See the Text Commands console for which face(s). '
            'MACHINE MANUALLY.'
        )

    return features, through_features, open_features, saw_features, warnings


def _find_full_span_saw_slots(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
    z_axis_index: int,
    attack_vector: adsk.core.Vector3D,
    exclude_face_ids: set,
) -> Tuple[List[SawSlotFeature], int]:
    """Finds SAW_SLOT_WIDTH_MM grooves directly from BRep geometry, for
    grooves RecognizedPocket.recognizePockets() never returns at all.

    Confirmed on real output: a groove that spans the body's own full
    bounding-box extent along its length (i.e. touches the outer silhouette
    at both ends, not just one) is missing entirely from recognizePockets()'s
    result — not merely misclassified — while the identical narrow/shallow
    groove shape open at only one end is returned fine and already handled by
    the open-blind-pocket branch above. Since the real geometry (a planar
    floor at the expected width, recessed below a z-parallel reference
    surface, with its own adjacent wall face) is fully present on the body
    regardless, this reuses the same width/wall-face logic applied directly
    to every remaining candidate face instead of going through
    RecognizedPocket.
    """
    top_face = _largest_same_direction_planar_face(body, z_axis)
    if top_face is None:
        return [], 0
    top_plane = adsk.core.Plane.cast(top_face.geometry)
    if not top_plane:
        return [], 0

    features: List[SawSlotFeature] = []
    skipped = 0
    for face in body.faces:
        if face.tempId in exclude_face_ids:
            continue
        if not adsk.core.Plane.cast(face.geometry):
            continue
        ok, normal = face.evaluator.getNormalAtPoint(face.pointOnFace)
        # Same-direction, not just parallel: _directions_parallel's abs-cosine
        # test also matches the opposite (far) surface. Confirmed on real
        # output: with only a parallel check, every groove floor was returned
        # for BOTH z_axis signs — once with a genuine small recessed depth,
        # once with a bogus depth against the wrong-side top_face — instead
        # of only for the side the groove actually opens onto.
        if not ok or not _directions_parallel(normal, z_axis) or normal.dotProduct(z_axis) <= 0:
            continue
        offset = top_plane.origin.vectorTo(face.pointOnFace)
        depth_cm = abs(offset.dotProduct(top_plane.normal))
        if depth_cm <= MIN_FEATURE_DIMENSION_CM:
            continue  # Coplanar with the reference surface itself, not a recessed floor.
        width_cm = _face_average_width_cm(face)
        if width_cm <= MIN_FEATURE_DIMENSION_CM:
            continue
        width_mm = width_cm * 10.0
        if abs(width_mm - SAW_SLOT_WIDTH_MM) > SAW_SLOT_WIDTH_TOLERANCE_MM:
            # A recessed, correctly-oriented floor at the saw-slot depth that
            # still misses the width window is not necessarily junk: a groove
            # that locally widens (e.g. a circular bulge around a hardware
            # clearance hole in the middle of an otherwise SAW_SLOT_WIDTH_MM
            # groove — confirmed on real output, area/length-of-longest-edge
            # for such a face lands well outside tolerance even though most
            # of its length genuinely is the nominal width) is tried as a
            # dogbone groove (see _dogbone_saw_arms) before being given up on.
            dogbone_arms = _dogbone_saw_arms(face, attack_vector, depth_cm, z_axis_index)
            if dogbone_arms:
                features.extend(dogbone_arms)
                continue
            skipped += 1
            futil.log(
                f'Saw slot candidate at face {face.tempId} rejected: average width '
                f'{width_mm:.4f} mm outside {SAW_SLOT_WIDTH_MM:g}±{SAW_SLOT_WIDTH_TOLERANCE_MM:g} '
                f'mm tolerance (area {face.area:.4f} cm^2, depth {depth_cm * 10.0:.4f} mm), and no '
                'dogbone-arm pair found either.',
                adsk.core.LogLevels.WarningLogLevel,
                force_console=True,
            )
            continue
        wall_face = _wall_face_for_floor(face, attack_vector, z_axis_index, depth_cm)
        if wall_face is None:
            skipped += 1
            futil.log(
                f'Saw slot candidate at face {face.tempId} (width {width_mm:.4f} mm) rejected: no '
                'adjacent planar wall face found for tool orientation (or the only candidate(s) were '
                'the panel\'s own outer edge — an open rabbet, not a real groove wall).',
                adsk.core.LogLevels.WarningLogLevel,
                force_console=True,
            )
            continue
        features.append(
            SawSlotFeature(
                floor_faces=[face],
                wall_face=wall_face,
                depth_cm=depth_cm,
                width_cm=width_cm,
            )
        )
    return features, skipped


def _log_recognition_summary(recognized: Sequence[adsk.cam.RecognizedPocket]) -> None:
    lines = [f'Pocket recognition summary: {len(recognized)} pocket(s) returned by recognizePockets().']
    for index, pocket in enumerate(recognized):
        lines.append(f'  [{index}] {_pocket_summary_line(pocket)}')
    futil.log('\n'.join(lines), force_console=True)


def _pocket_summary_line(pocket: adsk.cam.RecognizedPocket) -> str:
    def safe(getter, label):
        try:
            return f'{label}={getter()}'
        except (AttributeError, RuntimeError) as exc:
            return f'{label}=<error: {exc}>'

    return ', '.join(
        [
            safe(lambda: pocket.isThrough, 'isThrough'),
            safe(lambda: pocket.isClosed, 'isClosed'),
            safe(lambda: f'{pocket.depth:.4f}cm', 'depth'),
            safe(lambda: len(list(pocket.faces)), 'faces'),
            safe(lambda: len(list(pocket.sharedFaces)), 'sharedFaces'),
            safe(lambda: len(list(pocket.boundaries)), 'boundaries'),
        ]
    )


def _is_open_through_region(pocket: adsk.cam.RecognizedPocket) -> bool:
    """True for an exterior/open region that belongs to the part contour."""
    return bool(pocket.isThrough and not pocket.isClosed)


def _is_open_blind_pocket(pocket: adsk.cam.RecognizedPocket) -> bool:
    """True for a blind pocket whose region touches the outer silhouette
    (e.g. a groove routed in from an edge, open at one end). Confirmed on a
    real part via _log_recognition_summary: isThrough=False, isClosed=False.
    Fusion may or may not include the real floor in faces/sharedFaces, so the
    caller first tries the face-based paths before using a boundary fallback."""
    return bool((not pocket.isThrough) and (not pocket.isClosed))


def recognized_pocket_count(recognized: PocketRecognitionResult) -> int:
    """Number of millable pockets used to prioritize a machining side."""
    features, through_features, open_features, saw_features, _ = recognized
    return len(features) + len(through_features) + len(open_features) + len(saw_features)


def _circular_pocket_diameter_cm(
    pocket: adsk.cam.RecognizedPocket,
) -> Optional[float]:
    diameter_cm: Optional[float] = None
    for boundary in pocket.boundaries:
        for index in range(boundary.count):
            segment = boundary.item(index)
            circle = adsk.core.Circle3D.cast(segment)
            if not circle:
                return None
            diameter_cm = max(diameter_cm or 0.0, circle.radius * 2.0)
    return diameter_cm


def _pocket_bottom_faces(
    pocket: adsk.cam.RecognizedPocket,
    attack_vector: adsk.core.Vector3D,
) -> List[adsk.fusion.BRepFace]:
    result: List[adsk.fusion.BRepFace] = []
    # Fusion can place a valid, manually selectable pocket floor in
    # RecognizedPocket.sharedFaces when the floor also belongs to another
    # recognized region.  Looking only at faces incorrectly rejected such
    # open/connected flat-bottom grooves as having no planar bottom.
    candidate_faces = _recognized_pocket_faces(pocket)
    planar_count = 0
    closest_angle_deg: Optional[float] = None
    closest_face: Optional[adsk.fusion.BRepFace] = None
    for face in candidate_faces:
        if not adsk.core.Plane.cast(face.geometry):
            continue
        planar_count += 1
        ok, normal = face.evaluator.getNormalAtPoint(face.pointOnFace)
        if not ok:
            continue
        if _directions_parallel(normal, attack_vector):
            result.append(face)
            continue
        angle_deg = _angle_between_deg(normal, attack_vector)
        if closest_angle_deg is None or angle_deg < closest_angle_deg:
            closest_angle_deg = angle_deg
            closest_face = face

    if not result:
        _log_rejected_pocket_floor(candidate_faces, planar_count, closest_angle_deg, closest_face)
    return result


def _log_rejected_pocket_floor(
    candidate_faces: Sequence[adsk.fusion.BRepFace],
    planar_count: int,
    closest_angle_deg: Optional[float],
    closest_face: Optional[adsk.fusion.BRepFace],
) -> None:
    """Diagnostic for 'no usable planar bottom': the geometry may be planar
    and manually selectable in Fusion's UI while still failing the automatic
    parallel-to-attack-vector test, or it may never reach RecognizedPocket's
    faces/sharedFaces at all. This distinguishes those two cases from the
    Text Command console instead of guessing."""
    if closest_face is not None:
        center = closest_face.pointOnFace
        location = f'near ({center.x:.2f}, {center.y:.2f}, {center.z:.2f}) cm'
        futil.log(
            f'Pocket floor rejected: {len(candidate_faces)} candidate face(s), '
            f'{planar_count} planar, tolerance {SKETCH_PLANE_ALIGNMENT_TOLERANCE_DEG:g} deg, '
            f'closest planar face was {closest_angle_deg:.4f} deg off the attack vector, '
            f'area {closest_face.area:.4f} cm^2, {location}.',
            force_console=True,
        )
    else:
        futil.log(
            f'Pocket floor rejected: {len(candidate_faces)} candidate face(s), '
            f'{planar_count} planar, none had a usable normal.',
            force_console=True,
        )


def _recognized_pocket_faces(
    pocket: adsk.cam.RecognizedPocket,
) -> List[adsk.fusion.BRepFace]:
    """Return ordinary and shared pocket faces without duplicate BRep faces."""
    faces = list(pocket.faces)
    try:
        # sharedFaces was added after the original pocket-recognition API.  A
        # guarded read preserves compatibility with Fusion builds that expose
        # faces but not the newer property.
        faces.extend(pocket.sharedFaces)
    except (AttributeError, RuntimeError):
        pass

    unique: List[adsk.fusion.BRepFace] = []
    seen_temp_ids = set()
    for face in faces:
        try:
            temp_id = face.tempId
        except (AttributeError, RuntimeError):
            temp_id = None
        if temp_id is not None:
            if temp_id in seen_temp_ids:
                continue
            seen_temp_ids.add(temp_id)
        elif any(face is existing for existing in unique):
            continue
        unique.append(face)
    return unique


def _directions_parallel(
    first: adsk.core.Vector3D,
    second: adsk.core.Vector3D,
) -> bool:
    """Parallel test that tolerates small STEP-import axis round-off."""
    if (
        first.length < geometry.DIRECTION_LENGTH_EPSILON
        or second.length < geometry.DIRECTION_LENGTH_EPSILON
    ):
        return False
    cosine = abs(first.dotProduct(second) / (first.length * second.length))
    return cosine >= SKETCH_PLANE_ALIGNMENT_COS_THRESHOLD


def _floor_depth_tolerance_cm(expected_depth_cm: float) -> float:
    """Tolerance used to distinguish a real floor from other parallel
    ledges or the body's opposite face while searching an open pocket."""
    return max(0.05, expected_depth_cm * 0.5)


def _floor_faces_at_expected_depth(
    faces: Sequence[adsk.fusion.BRepFace],
    attack_vector: adsk.core.Vector3D,
    top_face: adsk.fusion.BRepFace,
    expected_depth_cm: float,
) -> List[adsk.fusion.BRepFace]:
    """Returns directly supplied planar floor faces at the expected depth.

    Fusion usually omits a floor from an edge-open RecognizedPocket, but not
    always.  Checking the supplied faces before an adjacency walk preserves
    the same face-based CAM selection used by ordinary closed pockets.  The
    depth check is essential because the supplied collection can also contain
    a parallel top ledge or the body's far side.
    """
    top_plane = adsk.core.Plane.cast(top_face.geometry)
    if not top_plane:
        return []
    tolerance_cm = _floor_depth_tolerance_cm(expected_depth_cm)
    result: List[adsk.fusion.BRepFace] = []
    for face in faces:
        if not adsk.core.Plane.cast(face.geometry):
            continue
        ok, normal = face.evaluator.getNormalAtPoint(face.pointOnFace)
        if not ok or not _directions_parallel(normal, attack_vector):
            continue
        offset = top_plane.origin.vectorTo(face.pointOnFace)
        depth_cm = abs(offset.dotProduct(top_plane.normal))
        # The absolute tolerance intentionally has a 0.05 cm lower bound,
        # which is larger than some very shallow pockets.  Never let that
        # make the depth-zero top ledge qualify as a floor.
        if depth_cm <= MIN_FEATURE_DIMENSION_CM:
            continue
        if abs(depth_cm - expected_depth_cm) <= tolerance_cm:
            result.append(face)
    return result


def _floor_faces_via_adjacency(
    faces: Sequence[adsk.fusion.BRepFace],
    attack_vector: adsk.core.Vector3D,
    top_face: adsk.fusion.BRepFace,
    expected_depth_cm: float,
) -> List[adsk.fusion.BRepFace]:
    """Finds a pocket's real floor face(s) through real BRep adjacency to its
    known wall/cap faces, instead of relying on RecognizedPocket's own
    faces/sharedFaces (which can omit the floor entirely for a pocket whose
    region touches the part's outer silhouette). Each wall's own edges
    connect directly to the true floor in the model, wherever it is — this
    works even when the floor is split into multiple coplanar fragments
    (e.g. around a bend in a curved groove), since every fragment adjacent to
    a known wall gets collected.

    A wall face can also border a face that is planar and attack-direction-
    normal but is *not* the pocket's floor at all — e.g. a rabbet cut into a
    panel's outer edge, where the wall is the panel's full-thickness edge
    face and its far side borders the panel's own bottom face. Confirmed on
    real output (a "floor" spanning almost the whole part's perimeter, at a
    depth matching full material thickness rather than the shallow rabbet).
    Filtering candidates by how close their own measured depth is to
    RecognizedPocket's reported depth rejects that kind of unrelated
    full-depth face without needing to know the true floor's exact depth in
    advance.

    The floor is also not always one hop from a wall: a real machined/
    modeled rabbet can have a corner-radius fillet (Cone/Cylinder) or a
    chamfer between the vertical wall and the horizontal floor. Confirmed on
    real output — every wall's one-hop neighbor was either the panel's own
    bottom face or a non-planar Cone/Cylinder face, never the true floor.
    Faces that are themselves neither a valid nor a rejected floor candidate
    (non-planar, or planar but not attack-direction-normal, and not already a
    known wall) are treated as transitions and searched one hop further, up
    to _FLOOR_ADJACENCY_MAX_HOPS, instead of stopping at the first hop."""
    top_plane = adsk.core.Plane.cast(top_face.geometry)
    # A fixed 0.3 cm floor is far bigger than a shallow pocket's own expected
    # depth (e.g. 0.23 cm here): it let a face at the SAME height as the top
    # (depth 0) pass as a "match" for a 2.3 mm-deep pocket, short-circuiting
    # the search into a wrong, degenerate result instead of correctly finding
    # no floor and falling through to the OpenPocketFeature sketch-boundary
    # path (which uses RecognizedPocket's own accurate depth). Confirmed on
    # real output: candidate face 44 at depth 0.0000 cm was accepted against
    # expected_depth_cm=0.23 under the old max(0.3, ...) tolerance.
    depth_tolerance_cm = _floor_depth_tolerance_cm(expected_depth_cm)
    visited = {face.tempId for face in faces}
    result: List[adsk.fusion.BRepFace] = []
    frontier = list(faces)
    for _ in range(_FLOOR_ADJACENCY_MAX_HOPS):
        next_frontier: List[adsk.fusion.BRepFace] = []
        for face in frontier:
            for edge in face.edges:
                for adjacent in edge.faces:
                    if adjacent.tempId in visited:
                        continue
                    visited.add(adjacent.tempId)
                    plane = adsk.core.Plane.cast(adjacent.geometry)
                    if plane:
                        ok, normal = adjacent.evaluator.getNormalAtPoint(adjacent.pointOnFace)
                        if ok and _directions_parallel(normal, attack_vector):
                            offset = top_plane.origin.vectorTo(adjacent.pointOnFace)
                            depth_cm = abs(offset.dotProduct(top_plane.normal))
                            if abs(depth_cm - expected_depth_cm) <= depth_tolerance_cm:
                                # A real floor match — do not search past it.
                                result.append(adjacent)
                                continue
                            # Rejected on depth, but NOT necessarily a dead
                            # end: on a multi-level part the true floor can
                            # sit one more step beyond an intermediate ledge
                            # that is itself planar and attack-normal (e.g. a
                            # raised boss's own shoulder). Confirmed on real
                            # output — a real, manually-selectable floor
                            # existed for a pocket where this search found
                            # nothing, because it stopped at exactly this
                            # kind of rejected intermediate face instead of
                            # continuing past it. Treat it like a transition
                            # too instead of assuming it's always an
                            # unrelated flat like a rabbet's far side.
                    # Non-planar (fillet), planar-but-angled (chamfer/another
                    # wall), or planar-but-wrong-depth: a transition, not a
                    # resolved floor. Keep looking beyond it.
                    next_frontier.append(adjacent)
        frontier = next_frontier
        if not frontier:
            break
    return result


#  A wall face reached only via a short edge of the floor's boundary is
#  almost always a cross-cut end face of the whole part (shared by every
#  parallel groove that opens onto it), not this groove's own wall — see
#  _wall_face_for_floor. An edge shorter than this fraction of the floor's
#  longest boundary edge is treated as one of those short "end" edges.
_WALL_LONG_EDGE_MIN_RATIO = 0.5

# How far off exactly perpendicular (to the attack vector) a candidate
# wall's own normal may be and still count as a real wall rather than an
# angled chamfer/fillet transition. This is a plain dot-product magnitude
# (both vectors are unit length), not an angle, so smaller is stricter;
# 0.1 allows roughly 6 degrees of STEP-import slack.
_WALL_PERPENDICULAR_ALIGNMENT_MAX = 0.1

# How far (cm) a candidate wall face's own extent along the machining Z axis
# may exceed the groove's recess depth and still count as that groove's real
# side wall. A genuine internal wall only rises from the floor up to the top
# reference surface, so its Z extent is ~equal to depth_cm. Confirmed on real
# output (HPL_badkamer_zij_vast_R_DR_iso_H00218): a 9.00 mm deep groove's true
# side walls each measured 9.00 mm in Z, while a rabbet ("polodrážka") open on
# one long side to the panel's own outer edge instead picked up that edge
# face as its "wall" — which, being the panel's actual perimeter, spans the
# full 13.6 mm panel thickness. 2 mm safely separates the two (a ~4.6 mm gap
# on that part) without rejecting a real wall over a small chamfer/fillet
# lead-in at the top edge.
_WALL_MAX_EXTRA_DEPTH_CM = 0.2


def _wall_face_for_floor(
    floor_face: adsk.fusion.BRepFace,
    attack_vector: adsk.core.Vector3D,
    z_axis_index: int,
    depth_cm: float,
) -> Optional[adsk.fusion.BRepFace]:
    """Finds one of a groove floor's own planar side walls via direct BRep
    adjacency, for use as the saw operation's tool-orientation reference face
    (see SawSlotFeature).

    A groove open at both ends (touching the part's outer silhouette) has a
    rectangular floor with two long boundary edges — bordering its own real
    side walls — and two short ones, at the open mouths, bordering whatever
    cross-cuts the part there. Confirmed on real output: on a part with four
    parallel grooves, the two faces reached via every floor's short edges
    were the *same* two faces for all four floors (the part's own end faces,
    shared because every groove opens onto them), while the faces reached
    via the long edges were unique per floor — the real walls. An earlier
    version of this function considered every boundary edge and kept
    whichever neighbor had the smallest area, which picked one of those
    shared end faces instead (it happened to be smaller than the real wall)
    and produced a tool orientation reference face for a wall of the part,
    not of this groove. Restricting the search to the long edges only fixes
    this without depending on which candidate happens to be smaller.

    A long edge can also border the panel's own outer edge face instead of a
    real internal wall — a rabbet ("polodrážka") open on that side rather
    than a fully enclosed groove. That neighbor is still planar and still
    perpendicular to the attack vector, so it passes every check above; it is
    caught separately by its Z extent (see _WALL_MAX_EXTRA_DEPTH_CM). Confirmed
    on real output (HPL_badkamer_zij_vast_R_DR_iso_H00218): such a groove
    still has a genuine wall on its *other* long edge, so picking the first
    valid candidate would silently machine it anyway with a merely-corrected
    orientation. An edge rabbet is not a saw-slot feature at all regardless of
    what its other side looks like, so finding even one open-to-the-edge
    candidate rejects the whole floor rather than falling back to another
    edge's wall.
    """
    edges = list(floor_face.edges)
    if not edges:
        return None
    max_length_cm = max(edge.length for edge in edges)
    long_edges = [
        edge for edge in edges if edge.length >= max_length_cm * _WALL_LONG_EDGE_MIN_RATIO
    ]
    candidates: List[adsk.fusion.BRepFace] = []
    for edge in long_edges:
        for adjacent in edge.faces:
            if adjacent.tempId == floor_face.tempId:
                continue
            if not adsk.core.Plane.cast(adjacent.geometry):
                continue
            ok, normal = adjacent.evaluator.getNormalAtPoint(adjacent.pointOnFace)
            if not ok:
                continue
            alignment = abs(
                normal.dotProduct(attack_vector) / (normal.length * attack_vector.length)
            )
            if alignment > _WALL_PERPENDICULAR_ALIGNMENT_MAX:
                continue
            box = adjacent.boundingBox
            wall_extent_cm = geometry.point_coordinate(
                box.maxPoint, z_axis_index
            ) - geometry.point_coordinate(box.minPoint, z_axis_index)
            if wall_extent_cm > depth_cm + _WALL_MAX_EXTRA_DEPTH_CM:
                # This neighbor reaches deeper than the recess itself — it's
                # the panel's own outer edge, not this groove's real wall.
                # This side is open to the panel's perimeter, so the whole
                # floor is an edge rabbet, not a saw slot: reject it outright
                # rather than keep searching the remaining long edges.
                futil.log(
                    f'Saw slot candidate at floor {floor_face.tempId} rejected outright: wall '
                    f'candidate {adjacent.tempId} spans {wall_extent_cm * 10.0:.4f} mm in Z vs a '
                    f'{depth_cm * 10.0:.4f} mm deep recess — this is the panel\'s own outer edge (an '
                    'open rabbet, e.g. a "polodrážka"), so no saw slot belongs here at all.',
                    adsk.core.LogLevels.WarningLogLevel,
                    force_console=True,
                )
                return None
            candidates.append(adjacent)
    if not candidates:
        return None
    wall_face = candidates[0]
    futil.log(
        f'Saw slot wall face candidates for floor {floor_face.tempId} '
        f'(via {len(long_edges)} long edge(s) of {len(edges)}): '
        + ', '.join(f'{face.tempId} (area {face.area:.4f} cm^2)' for face in candidates)
        + f' -> chose {wall_face.tempId}.',
        force_console=True,
    )
    return wall_face


def _wall_face_for_edge(
    edge: adsk.fusion.BRepEdge,
    floor_face_temp_id: int,
    attack_vector: adsk.core.Vector3D,
    z_axis_index: int,
    depth_cm: float,
) -> Optional[adsk.fusion.BRepFace]:
    """Same match as _wall_face_for_floor's per-edge check, but for one
    specific edge instead of every long edge of a whole floor — used by
    _dogbone_saw_arms, where the two arms of a split groove need their own,
    separately identified wall face rather than whichever candidate a
    whole-face search happens to return first (see _dogbone_saw_arms)."""
    for adjacent in edge.faces:
        if adjacent.tempId == floor_face_temp_id:
            continue
        if not adsk.core.Plane.cast(adjacent.geometry):
            continue
        ok, normal = adjacent.evaluator.getNormalAtPoint(adjacent.pointOnFace)
        if not ok:
            continue
        alignment = abs(
            normal.dotProduct(attack_vector) / (normal.length * attack_vector.length)
        )
        if alignment > _WALL_PERPENDICULAR_ALIGNMENT_MAX:
            continue
        box = adjacent.boundingBox
        wall_extent_cm = geometry.point_coordinate(
            box.maxPoint, z_axis_index
        ) - geometry.point_coordinate(box.minPoint, z_axis_index)
        if wall_extent_cm > depth_cm + _WALL_MAX_EXTRA_DEPTH_CM:
            # Same open-rabbet rejection as _wall_face_for_floor.
            continue
        return adjacent
    return None


# How parallel two boundary-edge "wall lines" must be (plain dot product of
# unit directions) to be considered the same groove's two side walls in
# _dogbone_saw_arms. Matches _WALL_PERPENDICULAR_ALIGNMENT_MAX's STEP-import
# slack.
_DOGBONE_PARALLEL_MIN_DOT = 0.999

# How much two candidate wall-line edges' spans (projected onto the shared
# wall direction) must overlap, as a fraction of the shorter edge's own span,
# to be treated as the two sides of the same physical groove arm in
# _dogbone_saw_arms. Not 1.0: real STEP-import edges bordering the same arm
# were confirmed to differ very slightly in length/extent.
_DOGBONE_OVERLAP_MIN_RATIO = 0.9


def _dogbone_saw_arms(
    face: adsk.fusion.BRepFace,
    attack_vector: adsk.core.Vector3D,
    depth_cm: float,
    z_axis_index: int,
) -> List[SawSlotFeature]:
    """Recovers the individual straight saw-blade passes from a groove floor
    that locally widens partway along its length — e.g. a circular
    hardware-clearance bulge around a hole in the middle — and so fails the
    plain whole-face width check in the caller (_face_average_width_cm
    averages the bulge into the estimate, and even a correct width estimate
    would still leave _create_saw_slot_operation picking a single "longest
    edge" that only spans one arm, since both arms' matching walls end up as
    separate, equal-length collinear edge segments — a real saw blade cannot
    follow the bulge's curved relief itself, only straight passes).

    Confirmed on real output (Hou_plafond_L3_achter_boxer_OH-00224, the row
    of a saw-slot grid that crosses a bolt-clearance hole): the groove's two
    true side walls stay on exactly the same two infinite lines, exactly
    SAW_SLOT_WIDTH_MM apart, for the entire nominal length — RecognizedPocket
    just returns them as one BRep face whose boundary walks out around the
    hole on two arcs, splitting each wall into two same-length collinear
    edges (one per arm) with a large gap where the bulge sits. Each pair of
    directly opposite, equal-span collinear edges (one from each wall line)
    is exactly one arm's own boundary.

    General by construction, not specific to this one part: works from the
    floor's own boundary edges alone (cluster by collinearity, pair by
    parallel + nominal spacing + overlapping span), so it applies to however
    many arms/bulges a given groove has.
    """
    outer_loop = next((loop for loop in face.loops if loop.isOuter), None)
    if outer_loop is None:
        return []

    clusters: List[Dict[str, object]] = []
    for edge in outer_loop.edges:
        line = adsk.core.Line3D.cast(edge.geometry)
        if line is None:
            continue
        direction = line.startPoint.vectorTo(line.endPoint)
        if direction.length < geometry.DIRECTION_LENGTH_EPSILON:
            continue
        direction.normalize()
        for cluster in clusters:
            if line.isColinearTo(cluster['line']):
                cluster['edges'].append(edge)
                break
        else:
            clusters.append({'line': line, 'dir': direction, 'edges': [edge]})
    if len(clusters) < 2:
        return []

    def project(point: adsk.core.Point3D, origin: adsk.core.Point3D, direction: adsk.core.Vector3D) -> float:
        return origin.vectorTo(point).dotProduct(direction)

    arms: List[SawSlotFeature] = []
    used_edge_ids = set()
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            cluster_a, cluster_b = clusters[i], clusters[j]
            direction_a = cluster_a['dir']
            if abs(direction_a.dotProduct(cluster_b['dir'])) < _DOGBONE_PARALLEL_MIN_DOT:
                continue
            line_a = cluster_a['line']
            line_b = cluster_b['line']
            offset = line_a.startPoint.vectorTo(line_b.startPoint)
            along = offset.dotProduct(direction_a)
            along_vector = direction_a.copy()
            along_vector.scaleBy(along)
            perpendicular = offset.copy()
            perpendicular.subtract(along_vector)
            width_mm = perpendicular.length * 10.0
            if abs(width_mm - SAW_SLOT_WIDTH_MM) > SAW_SLOT_WIDTH_TOLERANCE_MM:
                continue
            for edge_a in cluster_a['edges']:
                if edge_a.tempId in used_edge_ids:
                    continue
                a_start = project(edge_a.startVertex.geometry, line_a.startPoint, direction_a)
                a_end = project(edge_a.endVertex.geometry, line_a.startPoint, direction_a)
                a_low, a_high = min(a_start, a_end), max(a_start, a_end)
                for edge_b in cluster_b['edges']:
                    if edge_b.tempId in used_edge_ids:
                        continue
                    b_start = project(edge_b.startVertex.geometry, line_a.startPoint, direction_a)
                    b_end = project(edge_b.endVertex.geometry, line_a.startPoint, direction_a)
                    b_low, b_high = min(b_start, b_end), max(b_start, b_end)
                    shorter_span = min(a_high - a_low, b_high - b_low)
                    overlap = min(a_high, b_high) - max(a_low, b_low)
                    if shorter_span <= MIN_FEATURE_DIMENSION_CM or overlap / shorter_span < _DOGBONE_OVERLAP_MIN_RATIO:
                        continue
                    wall_face = _wall_face_for_edge(
                        edge_a, face.tempId, attack_vector, z_axis_index, depth_cm
                    ) or _wall_face_for_edge(
                        edge_b, face.tempId, attack_vector, z_axis_index, depth_cm
                    )
                    if wall_face is None:
                        futil.log(
                            f'Dogbone saw slot arm at face {face.tempId} '
                            f'(edges {edge_a.tempId}/{edge_b.tempId}) has no adjacent planar wall '
                            'face for tool orientation; arm left unmachined.',
                            adsk.core.LogLevels.WarningLogLevel,
                            force_console=True,
                        )
                        used_edge_ids.add(edge_a.tempId)
                        used_edge_ids.add(edge_b.tempId)
                        break
                    futil.log(
                        f'Dogbone saw slot arm found at face {face.tempId}: edges '
                        f'{edge_a.tempId}/{edge_b.tempId}, span {overlap:.4f} cm, width '
                        f'{width_mm:.4f} mm, wall face {wall_face.tempId}.',
                        force_console=True,
                    )
                    arms.append(
                        SawSlotFeature(
                            floor_faces=[face],
                            wall_face=wall_face,
                            depth_cm=depth_cm,
                            width_cm=perpendicular.length,
                            path_edge=edge_a,
                        )
                    )
                    used_edge_ids.add(edge_a.tempId)
                    used_edge_ids.add(edge_b.tempId)
                    break
    return arms


def _log_floor_adjacency_rejections(
    faces: Sequence[adsk.fusion.BRepFace],
    attack_vector: adsk.core.Vector3D,
    top_face: Optional[adsk.fusion.BRepFace],
    expected_depth_cm: float,
) -> None:
    """Diagnostic twin of _floor_faces_via_adjacency, run only when it finds
    nothing. Walks the exact same hop-by-hop traversal and logs, per hop,
    why every candidate was turned away (not planar, normal not parallel to
    the attack vector, or its own measured depth too far from
    RecognizedPocket.depth) or kept on as a transition to search past, so a
    real cause can be read off the log instead of guessed at."""
    if top_face is None:
        futil.log('Floor-via-adjacency: no top face found on the body.', force_console=True)
        return
    top_plane = adsk.core.Plane.cast(top_face.geometry)
    # A fixed 0.3 cm floor is far bigger than a shallow pocket's own expected
    # depth (e.g. 0.23 cm here): it let a face at the SAME height as the top
    # (depth 0) pass as a "match" for a 2.3 mm-deep pocket, short-circuiting
    # the search into a wrong, degenerate result instead of correctly finding
    # no floor and falling through to the OpenPocketFeature sketch-boundary
    # path (which uses RecognizedPocket's own accurate depth). Confirmed on
    # real output: candidate face 44 at depth 0.0000 cm was accepted against
    # expected_depth_cm=0.23 under the old max(0.3, ...) tolerance.
    depth_tolerance_cm = _floor_depth_tolerance_cm(expected_depth_cm)
    visited = {face.tempId for face in faces}
    lines = [
        f'Floor-via-adjacency: found nothing for {len(faces)} wall/cap face(s), '
        f'expected depth {expected_depth_cm:.4f} cm (tolerance {depth_tolerance_cm:.4f} cm, '
        f'max {_FLOOR_ADJACENCY_MAX_HOPS} hop(s)).'
    ]
    frontier = list(faces)
    candidate_count = 0
    for hop in range(1, _FLOOR_ADJACENCY_MAX_HOPS + 1):
        next_frontier: List[adsk.fusion.BRepFace] = []
        for face in frontier:
            for edge in face.edges:
                for adjacent in edge.faces:
                    if adjacent.tempId in visited:
                        continue
                    visited.add(adjacent.tempId)
                    candidate_count += 1
                    plane = adsk.core.Plane.cast(adjacent.geometry)
                    if not plane:
                        lines.append(
                            f'  hop {hop}, candidate {adjacent.tempId}: not planar '
                            f'({adjacent.geometry.classType()}) -> transition, searching past it.'
                        )
                        next_frontier.append(adjacent)
                        continue
                    ok, normal = adjacent.evaluator.getNormalAtPoint(adjacent.pointOnFace)
                    if not ok:
                        lines.append(f'  hop {hop}, candidate {adjacent.tempId}: could not evaluate normal.')
                        next_frontier.append(adjacent)
                        continue
                    if not _directions_parallel(normal, attack_vector):
                        angle_deg = _angle_between_deg(normal, attack_vector)
                        lines.append(
                            f'  hop {hop}, candidate {adjacent.tempId}: normal {angle_deg:.2f} deg off '
                            f'attack vector -> transition, searching past it.'
                        )
                        next_frontier.append(adjacent)
                        continue
                    offset = top_plane.origin.vectorTo(adjacent.pointOnFace)
                    depth_cm = abs(offset.dotProduct(top_plane.normal))
                    delta_cm = abs(depth_cm - expected_depth_cm)
                    kept = delta_cm <= depth_tolerance_cm
                    lines.append(
                        f'  hop {hop}, candidate {adjacent.tempId}: planar & parallel, depth {depth_cm:.4f} cm '
                        f'(delta {delta_cm:.4f} cm vs tolerance {depth_tolerance_cm:.4f} cm) '
                        f'-> {"kept" if kept else "rejected, searching past it too"}.'
                    )
                    if not kept:
                        next_frontier.append(adjacent)
        frontier = next_frontier
        if not frontier:
            break
    if candidate_count == 0:
        lines.append('  no adjacent non-wall faces found at all.')
    futil.log('\n'.join(lines), force_console=True)


def _angle_between_deg(
    first: adsk.core.Vector3D,
    second: adsk.core.Vector3D,
) -> float:
    """Smallest angle (degrees) between two directions, ignoring sign."""
    cosine = abs(first.dotProduct(second) / (first.length * second.length))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _minimum_planar_span(
    faces: Sequence[adsk.fusion.BRepFace],
    z_axis_index: int,
) -> float:
    minimums = [float('inf'), float('inf'), float('inf')]
    maximums = [float('-inf'), float('-inf'), float('-inf')]
    for face in faces:
        box = face.boundingBox
        for axis_index in range(3):
            minimums[axis_index] = min(
                minimums[axis_index], geometry.point_coordinate(box.minPoint, axis_index)
            )
            maximums[axis_index] = max(
                maximums[axis_index], geometry.point_coordinate(box.maxPoint, axis_index)
            )
    planar_axes = [axis_index for axis_index in range(3) if axis_index != z_axis_index]
    return min(maximums[index] - minimums[index] for index in planar_axes)


def _face_average_width_cm(face: adsk.fusion.BRepFace) -> float:
    """Orientation-independent average-width estimate for a roughly
    rectangular strip face (area / its own length), used instead of an
    axis-aligned bounding box when a face can sit diagonally in X/Y — see
    _minimum_planar_span's caller for why that matters for a bending
    groove's floor fragments.

    Length is the total of every straight edge collinear (same infinite
    line, not just the same direction — two parallel edges on the strip's
    opposite long sides must not be summed together) with the single
    longest edge, not just that one edge. Confirmed on real output: another
    feature's edge landing on a long, otherwise straight boundary can split
    it into several shorter collinear segments (e.g. 42.8+11.7+42.8 cm
    instead of one 92.5 cm edge) without changing the face's real shape.
    Dividing by only the longest fragment (42.8 cm here) overstated a 3.5 mm
    saw-slot groove's width by ~4x (to 12.3 mm), pushing it out of
    SAW_SLOT_WIDTH_TOLERANCE_MM and misrouting it to ordinary pocket
    clearing instead of the saw operation."""
    edges = list(face.edges)
    if not edges:
        return 0.0
    longest_edge = max(edges, key=lambda edge: edge.length)
    if longest_edge.length <= MIN_FEATURE_DIMENSION_CM:
        return 0.0
    longest_line = adsk.core.Line3D.cast(longest_edge.geometry)
    if longest_line is None:
        return face.area / longest_edge.length
    length_cm = 0.0
    for edge in edges:
        line = adsk.core.Line3D.cast(edge.geometry)
        if line is not None and line.isColinearTo(longest_line):
            length_cm += edge.length
    if length_cm <= MIN_FEATURE_DIMENSION_CM:
        return face.area / longest_edge.length
    return face.area / length_cm


def _loop_circle_diameter_cm(loop: adsk.fusion.BRepLoop) -> Optional[float]:
    """Returns the diameter of a loop that is exactly one full circle, or
    None for any other shape (including a circular arc split across more
    than one edge — real closed circles from BRep/STEP import are already
    confirmed elsewhere in this module to sometimes come back as a single
    Circle3D edge, e.g. a drilled hole's own boundary)."""
    edges = list(loop.edges)
    if len(edges) != 1:
        return None
    circle = adsk.core.Circle3D.cast(edges[0].geometry)
    if circle is None:
        return None
    return circle.radius * 2.0


def _loop_bounding_span_cm(loop: adsk.fusion.BRepLoop, z_axis_index: int) -> float:
    """Same idea as _minimum_planar_span, but for one loop's own edges
    instead of a set of faces — used as this loop's tool-fit width when it
    isn't a plain circle."""
    minimums = [float('inf'), float('inf'), float('inf')]
    maximums = [float('-inf'), float('-inf'), float('-inf')]
    for edge in loop.edges:
        box = edge.boundingBox
        for axis_index in range(3):
            minimums[axis_index] = min(
                minimums[axis_index], geometry.point_coordinate(box.minPoint, axis_index)
            )
            maximums[axis_index] = max(
                maximums[axis_index], geometry.point_coordinate(box.maxPoint, axis_index)
            )
    planar_axes = [index for index in range(3) if index != z_axis_index]
    return min(maximums[index] - minimums[index] for index in planar_axes)


def find_nested_through_openings(
    floor_faces_with_depth: Sequence[Tuple[adsk.fusion.BRepFace, float]],
    z_axis_index: int,
    maximum_drill_diameter_mm: float = float('inf'),
) -> List[NestedThroughFeature]:
    """Finds closed inner boundaries on already-recognized floor faces that
    aren't already handled by drilling — see NestedThroughFeature for why
    these exist and why RecognizedPocket alone never surfaces them. A plain
    circular opening within the configured drilling diameter is skipped here
    on the assumption holes.py already drills it; anything else (a bigger
    circle, or any non-circular shape) is returned.

    Takes explicit (face, depth) pairs rather than PocketFeature objects: a
    dogbone saw-slot arm (see _dogbone_saw_arms) shares its floor face with
    another arm and no longer owns a PocketFeature at all once split out of
    `features`, but the hole through the *middle* of that same shared face
    (e.g. a bolt-clearance hole a saw-slot groove locally widens around)
    still needs to be found here exactly as before — confirmed on real
    output (Hou_plafond_L3_achter_boxer_OH-00224): before this was widened
    to include saw_features' own floor faces too, splitting that groove into
    its two arms silently dropped the already-working "Nested through
    openings" operation that machines this exact hole, since the floor face
    it lives on was no longer reachable from any PocketFeature."""
    results: List[NestedThroughFeature] = []
    seen_loop_ids = set()
    seen_face_ids = set()
    for face, depth_cm in floor_faces_with_depth:
        if face.tempId in seen_face_ids:
            continue
        seen_face_ids.add(face.tempId)
        for loop in face.loops:
            if loop.isOuter:
                continue
            loop_key = tuple(sorted(edge.tempId for edge in loop.edges))
            if loop_key in seen_loop_ids:
                continue
            seen_loop_ids.add(loop_key)
            diameter_cm = _loop_circle_diameter_cm(loop)
            if (
                diameter_cm is not None
                and diameter_cm * 10.0 <= maximum_drill_diameter_mm + 1e-6
            ):
                continue
            span_cm = (
                diameter_cm
                if diameter_cm is not None
                else _loop_bounding_span_cm(loop, z_axis_index)
            )
            if span_cm <= MIN_FEATURE_DIMENSION_CM:
                continue
            results.append(
                NestedThroughFeature(
                    floor_face=face,
                    inner_loop=loop,
                    depth_cm=depth_cm,
                    span_cm=span_cm,
                )
            )
    return results


def has_opposite_side_pockets(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
    z_axis_index: int,
    maximum_drill_diameter_mm: float = float('inf'),
) -> bool:
    """True if blind pockets exist that are only reachable by attacking from
    the opposite side of the body (i.e. the part would need to be flipped to
    machine them). Through slots are excluded: they are already fully cut by
    the single-sided operation and need no manual follow-up."""
    opposite_axis = z_axis.copy()
    opposite_axis.scaleBy(-1.0)
    features, _, open_features, saw_features, _ = recognize_pocket_features(
        body,
        opposite_axis,
        z_axis_index,
        maximum_drill_diameter_mm,
    )
    return bool(features) or bool(open_features) or bool(saw_features)


@dataclass
class PreparedPockets:
    grouped: Dict[Tuple[str, int], List[PocketFeature]]
    through_grouped: Dict[
        str,
        Tuple[List[ThroughSlotFeature], adsk.fusion.Sketch, Optional[float]],
    ]
    open_grouped: Dict[
        Tuple[str, int], Tuple[List[OpenPocketFeature], adsk.fusion.Sketch, List[adsk.fusion.SketchCurve]]
    ]
    saw_grouped: Dict[int, List[SawSlotFeature]]
    nested_grouped: Dict[Tuple[str, int], List[NestedThroughFeature]]
    warnings: List[str]


def prepare_pocket_operations(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
    z_axis_index: int,
    small_tool: adsk.cam.Tool,
    large_tool: adsk.cam.Tool,
    recognized: Optional[PocketRecognitionResult] = None,
    saw_tool: Optional[adsk.cam.Tool] = None,
    maximum_drill_diameter_mm: float = float('inf'),
) -> PreparedPockets:
    """Recognizes pockets and, for through slots, draws their boundary
    sketches. Called before the CAM Setup is created so Design geometry is
    complete before CAM operations reference it."""
    if recognized is None:
        recognized = recognize_pocket_features(body, z_axis, z_axis_index)
    features, through_features, open_features, saw_features, recognition_warnings = recognized
    warnings = list(recognition_warnings)

    saw_grouped: Dict[int, List[SawSlotFeature]] = {}
    if saw_tool is not None:
        for feature in saw_features:
            depth_microns = int(round(feature.depth_cm * 10.0 * 1000.0))
            saw_grouped.setdefault(depth_microns, []).append(feature)
        # No automatic width sanity check here: tool_diameter is the saw
        # blade's overall cutting-circle diameter (confirmed on real output —
        # a real ~350 mm blade), not its kerf/thickness, which is what
        # actually needs to match SAW_SLOT_WIDTH_MM. Which tool_* parameter
        # holds a slot mill's kerf width is not yet confirmed, so this is
        # left unchecked rather than compared against the wrong dimension.
    elif saw_features:
        warnings.append(
            f'{len(saw_features)} {SAW_SLOT_WIDTH_MM:g} mm saw slot(s) found but "Saw slots" '
            'is off or no saw tool was selected; left unmachined.'
        )

    small_diameter_cm = tool_library.tool_diameter_cm(small_tool)
    large_diameter_cm = tool_library.tool_diameter_cm(large_tool)
    small_flute_length_cm = tool_library.tool_flute_length_cm(small_tool)
    large_flute_length_cm = tool_library.tool_flute_length_cm(large_tool)
    skipped_too_small = 0
    skipped_too_deep = 0

    grouped: Dict[Tuple[str, int], List[PocketFeature]] = {}
    for feature in features:
        size, diameter_fits = _select_tool_size(
            feature.minimum_span_cm,
            feature.depth_cm,
            small_diameter_cm,
            small_flute_length_cm,
            large_diameter_cm,
            large_flute_length_cm,
        )
        if size is None:
            _log_tool_fit_rejection(
                feature.minimum_span_cm,
                feature.depth_cm,
                small_diameter_cm,
                small_flute_length_cm,
                large_diameter_cm,
                large_flute_length_cm,
            )
            if diameter_fits:
                skipped_too_deep += 1
            else:
                skipped_too_small += 1
            continue
        if size not in ('Small', 'Large'):
            # Defensive guard for future changes to _select_tool_size.
            skipped_too_small += 1
            continue
        _log_pocket_tool_selection(size, feature)
        depth_microns = int(round(feature.depth_cm * 10.0 * 1000.0))
        grouped.setdefault((size, depth_microns), []).append(feature)

    through_grouped_features: Dict[str, List[ThroughSlotFeature]] = {}
    for feature in through_features:
        # The generated path extends 1 mm below stock bottom, so the cutter
        # must have that extra usable flute length as well.
        size, diameter_fits = _select_tool_size(
            feature.minimum_span_cm,
            feature.depth_cm + 0.1,
            small_diameter_cm,
            small_flute_length_cm,
            large_diameter_cm,
            large_flute_length_cm,
        )
        if size is None:
            _log_tool_fit_rejection(
                feature.minimum_span_cm,
                feature.depth_cm + 0.1,
                small_diameter_cm,
                small_flute_length_cm,
                large_diameter_cm,
                large_flute_length_cm,
            )
            if diameter_fits:
                skipped_too_deep += 1
            else:
                skipped_too_small += 1
            continue
        through_grouped_features.setdefault(size, []).append(feature)

    through_grouped: Dict[
        str,
        Tuple[List[ThroughSlotFeature], adsk.fusion.Sketch, Optional[float]],
    ] = {}
    for size, group in through_grouped_features.items():
        sketch, _curves, reference_area_cm2 = _sketch_from_boundaries(
            body, z_axis, z_axis_index, group
        )
        through_grouped[size] = (group, sketch, reference_area_cm2)

    open_grouped_features: Dict[Tuple[str, int], List[OpenPocketFeature]] = {}
    for feature in open_features:
        size, diameter_fits = _select_tool_size(
            feature.minimum_span_cm,
            feature.depth_cm,
            small_diameter_cm,
            small_flute_length_cm,
            large_diameter_cm,
            large_flute_length_cm,
        )
        if size is None:
            _log_tool_fit_rejection(
                feature.minimum_span_cm,
                feature.depth_cm,
                small_diameter_cm,
                small_flute_length_cm,
                large_diameter_cm,
                large_flute_length_cm,
            )
            if diameter_fits:
                skipped_too_deep += 1
            else:
                skipped_too_small += 1
            continue
        depth_microns = int(round(feature.depth_cm * 10.0 * 1000.0))
        open_grouped_features.setdefault((size, depth_microns), []).append(feature)

    open_grouped: Dict[
        Tuple[str, int], Tuple[List[OpenPocketFeature], adsk.fusion.Sketch, List[adsk.fusion.SketchCurve]]
    ] = {}
    for key, group in open_grouped_features.items():
        sketch, curves, _reference_area_cm2 = _sketch_from_boundaries(
            body, z_axis, z_axis_index, group
        )
        open_grouped[key] = (group, sketch, curves)

    nested_grouped: Dict[Tuple[str, int], List[NestedThroughFeature]] = {}
    nested_source_faces: List[Tuple[adsk.fusion.BRepFace, float]] = [
        (face, feature.depth_cm) for feature in features for face in feature.bottom_faces
    ]
    nested_source_faces.extend(
        (face, feature.depth_cm) for feature in saw_features for face in feature.floor_faces
    )
    for nested_feature in find_nested_through_openings(
        nested_source_faces, z_axis_index, maximum_drill_diameter_mm
    ):
        size, diameter_fits = _select_tool_size(
            nested_feature.span_cm,
            nested_feature.depth_cm,
            small_diameter_cm,
            small_flute_length_cm,
            large_diameter_cm,
            large_flute_length_cm,
        )
        if size is None:
            _log_tool_fit_rejection(
                nested_feature.span_cm,
                nested_feature.depth_cm,
                small_diameter_cm,
                small_flute_length_cm,
                large_diameter_cm,
                large_flute_length_cm,
            )
            if diameter_fits:
                skipped_too_deep += 1
            else:
                skipped_too_small += 1
            continue
        depth_microns = int(round(nested_feature.depth_cm * 10.0 * 1000.0))
        nested_grouped.setdefault((size, depth_microns), []).append(nested_feature)

    if skipped_too_small:
        warnings.append(
            f'{skipped_too_small} pocket(s)/slot(s) were narrower than the selected small pocket tool.'
        )
    if skipped_too_deep:
        warnings.append(
            f'{skipped_too_deep} pocket(s)/slot(s) were deeper than the usable flute '
            'length of the selected pocket tools.'
        )
    return PreparedPockets(
        grouped, through_grouped, open_grouped, saw_grouped, nested_grouped, warnings
    )


def _log_tool_fit_rejection(
    minimum_span_cm: float,
    required_depth_cm: float,
    small_diameter_cm: float,
    small_flute_length_cm: float,
    large_diameter_cm: float,
    large_flute_length_cm: float,
) -> None:
    """Logs the exact numbers behind a 'too narrow'/'too deep' skip, so a
    real tool-reach limit can be told apart from a depth/width that was
    measured wrong for this feature's shape, instead of guessing which."""
    futil.log(
        f'Pocket tool fit rejected: span {minimum_span_cm * 10.0:.2f} mm, required depth '
        f'{required_depth_cm * 10.0:.2f} mm; small tool Ø{small_diameter_cm * 10.0:.2f} mm '
        f'/ flute {small_flute_length_cm * 10.0:.2f} mm; large tool '
        f'Ø{large_diameter_cm * 10.0:.2f} mm / flute {large_flute_length_cm * 10.0:.2f} mm.',
        force_console=True,
    )


def _log_pocket_tool_selection(size: str, feature: PocketFeature) -> None:
    """Logs which tool a pocket feature was assigned and the per-face width
    estimate behind its minimum_span_cm, so a 'wrong tool picked' report can
    be checked against the actual measured numbers instead of guessed at —
    e.g. whether a narrow real groove's span was overestimated by one oddly
    shaped floor fragment in the group."""
    per_face = ', '.join(
        f'{face.tempId}: area {face.area:.4f} cm^2 -> width {_face_average_width_cm(face):.4f} cm'
        for face in feature.bottom_faces
    )
    futil.log(
        f'Pocket tool selected: {size}, span {feature.minimum_span_cm * 10.0:.2f} mm, '
        f'depth {feature.depth_cm * 10.0:.2f} mm, {len(feature.bottom_faces)} floor face(s): {per_face}.',
        force_console=True,
    )


def _select_tool_size(
    minimum_span_cm: float,
    required_depth_cm: float,
    small_diameter_cm: float,
    small_flute_length_cm: float,
    large_diameter_cm: float,
    large_flute_length_cm: float,
) -> Tuple[Optional[str], bool]:
    """Chooses the largest selected tool that fits width and cutting depth.

    The boolean reports whether at least one tool fit by diameter.  This lets
    the caller distinguish a narrow feature from an insufficient flute length
    in the batch report.
    """
    diameter_fits = False
    for size, diameter_cm, flute_length_cm in (
        ('Large', large_diameter_cm, large_flute_length_cm),
        ('Small', small_diameter_cm, small_flute_length_cm),
    ):
        if diameter_cm > minimum_span_cm + TOOL_FIT_EPSILON_CM:
            continue
        diameter_fits = True
        if flute_length_cm + TOOL_FIT_EPSILON_CM >= required_depth_cm:
            return size, True
    return None, diameter_fits


def create_pocket_operations(
    cam: adsk.cam.CAM,
    setup: adsk.cam.Setup,
    prepared: PreparedPockets,
    small_tool: adsk.cam.Tool,
    large_tool: adsk.cam.Tool,
    stepdown_mm: float,
    saw_tool: Optional[adsk.cam.Tool] = None,
    attack_vector: Optional[adsk.core.Vector3D] = None,
) -> Tuple[List[adsk.cam.OperationBase], List[str]]:
    warnings = list(prepared.warnings)
    if (
        not prepared.grouped
        and not prepared.through_grouped
        and not prepared.open_grouped
        and not prepared.saw_grouped
        and not prepared.nested_grouped
    ):
        if not warnings:
            warnings.append('No millable pockets were found.')
        return [], warnings

    tool_axis: Optional[adsk.core.Vector3D] = None
    if attack_vector is not None:
        tool_axis = attack_vector.copy()
        tool_axis.scaleBy(-1.0)

    operations: List[adsk.cam.OperationBase] = []
    for (size, depth_microns), group in sorted(prepared.grouped.items()):
        tool = large_tool if size == 'Large' else small_tool
        depth_mm = depth_microns / 1000.0
        operation, group_warnings = _create_pocket_operation(
            setup, group, tool, size, depth_mm, stepdown_mm, tool_axis
        )
        warnings.extend(group_warnings)
        if operation is not None:
            operations.append(operation)

    for size, (group, sketch, reference_area_cm2) in sorted(
        prepared.through_grouped.items()
    ):
        tool = large_tool if size == 'Large' else small_tool
        operations.append(
            _create_through_slot_operation(
                setup,
                sketch,
                group,
                tool,
                size,
                stepdown_mm,
                reference_area_cm2,
            )
        )

    for (size, depth_microns), (group, _sketch, curves) in sorted(prepared.open_grouped.items()):
        tool = large_tool if size == 'Large' else small_tool
        depth_mm = depth_microns / 1000.0
        operations.append(
            _create_open_pocket_operation(setup, curves, group, tool, size, depth_mm, stepdown_mm)
        )

    if prepared.nested_grouped:
        if attack_vector is None:
            raise RuntimeError(
                'create_pocket_operations requires attack_vector to orient nested through openings.'
            )
        assert tool_axis is not None
        for (size, depth_microns), group in sorted(prepared.nested_grouped.items()):
            tool = large_tool if size == 'Large' else small_tool
            depth_mm = depth_microns / 1000.0
            operations.append(
                _create_nested_through_operation(
                    setup, group, tool, depth_mm, tool_axis
                )
            )

    if prepared.saw_grouped and saw_tool is not None:
        if attack_vector is None:
            raise RuntimeError('create_pocket_operations requires attack_vector to build saw slots.')
        for depth_microns, group in sorted(prepared.saw_grouped.items()):
            depth_mm = depth_microns / 1000.0
            operations.append(
                _create_saw_slot_operation(setup, group, saw_tool, depth_mm, attack_vector)
            )
        warnings.append(
            'Saw slot tool orientation is derived from real part geometry but has not been '
            'verified on a saw-equipped machine — check the toolpath, tool orientation and '
            'depth in Fusion\'s simulation before cutting real material.'
        )

    return operations, warnings


def _set_ramp_parameters(operation: adsk.cam.OperationBase) -> None:
    """Straight plunge entry instead of Fusion's default ramped entry, for a
    2D Pocket operation. Must be called AFTER setup.operations.add() —
    confirmed on real output that add() resets doRamp/rampClearanceHeight
    (like every other tool-preset-linked lead/ramp parameter — see
    entry_radius in _create_contour_operation and _create_saw_slot_operation)
    back to their tool-preset defaults no matter what operation_input was set
    to beforehand. rampClearanceHeight's own default (2.5 mm) exceeds the
    2 mm feed height from set_safe_heights, which otherwise raises a "Ramp
    Clearance Height" warning even with ramping disabled (it's a separate
    Linking-tab value)."""
    set_expression(operation.parameters, 'doRamp', 'false')
    set_expression(operation.parameters, 'rampClearanceHeight', '0 mm')


_FLOOR_TOOL_AXIS_ALIGNMENT_MIN = 0.98
_OPEN_BOUNDARY_BELOW_FLOOR_TOLERANCE_CM = 0.001


def _open_floor_chain_should_revert(
    first_coedge_is_opposed_to_edge: bool,
    floor_normal_dot_tool_axis: float,
) -> bool:
    """Returns the ChainSelection orientation whose pocket side is the
    owning floor face, independent of edge winding or mirroring.

    A face's outer-loop co-edges traverse its boundary with the face region
    on the consistent side. Fusion's API ChainSelection loses that co-edge
    orientation and instead starts from the first BRepEdge's natural
    direction. ``isOpposedToEdge`` bridges those two directions. Looking at
    the loop from the opposite setup side reverses the relationship once
    more, hence the normal/tool-axis sign below.
    """
    if abs(floor_normal_dot_tool_axis) < _FLOOR_TOOL_AXIS_ALIGNMENT_MIN:
        raise RuntimeError(
            'Open-pocket floor is not perpendicular to the setup tool axis.'
        )
    if floor_normal_dot_tool_axis > 0.0:
        return first_coedge_is_opposed_to_edge
    return not first_coedge_is_opposed_to_edge


def _open_floor_wall_coedges(
    floor_face: adsk.fusion.BRepFace,
    recognized_boundary_face_ids: Optional[Set[int]],
    tool_axis: adsk.core.Vector3D,
) -> Optional[List[adsk.fusion.BRepCoEdge]]:
    """Returns one unambiguous open run following the pocket's real walls.

    A valid mouth edge has an adjacent exterior face that continues below the
    pocket floor. A valid wall edge borders a face reported by
    RecognizedPocket and that face does not continue below the floor. Both
    tests must agree; missing or conflicting evidence is rejected rather than
    risking a toolpath on the wrong side. Multiple disconnected wall runs are
    also rejected because a single ChainSelection cannot represent them as
    one unambiguous open pocket.
    """
    if not recognized_boundary_face_ids:
        return None
    outer_loop = next((loop for loop in floor_face.loops if loop.isOuter), None)
    if outer_loop is None:
        return None
    coedges = list(outer_loop.coEdges)
    if not coedges:
        return None

    statuses: List[str] = []
    for coedge in coedges:
        adjacent_faces = [
            face for face in coedge.edge.faces if face.tempId != floor_face.tempId
        ]
        if not adjacent_faces:
            return None
        is_recognized_wall = any(
            face.tempId in recognized_boundary_face_ids for face in adjacent_faces
        )
        extends_below_floor = False
        for adjacent_face in adjacent_faces:
            for vertex in adjacent_face.vertices:
                offset = floor_face.pointOnFace.vectorTo(vertex.geometry)
                if (
                    offset.dotProduct(tool_axis)
                    < -_OPEN_BOUNDARY_BELOW_FLOOR_TOLERANCE_CM
                ):
                    extends_below_floor = True
                    break
            if extends_below_floor:
                break

        if is_recognized_wall and not extends_below_floor:
            statuses.append('wall')
        elif extends_below_floor and not is_recognized_wall:
            statuses.append('mouth')
        else:
            return None

    wall_starts = [
        index
        for index, status in enumerate(statuses)
        if status == 'wall' and statuses[index - 1] != 'wall'
    ]
    if len(wall_starts) != 1 or 'mouth' not in statuses:
        return None

    wall_coedges: List[adsk.fusion.BRepCoEdge] = []
    index = wall_starts[0]
    while statuses[index] == 'wall':
        wall_coedges.append(coedges[index])
        index = (index + 1) % len(coedges)
    return wall_coedges or None


def _resolve_open_floor_orientation(
    floor_face: adsk.fusion.BRepFace,
    tool_axis: Optional[adsk.core.Vector3D],
    recognized_boundary_face_ids: Optional[Set[int]],
) -> Optional[Tuple[str, List[adsk.fusion.BRepCoEdge]]]:
    """Returns the exact open wall chain and its safe machining orientation.

    A face-based PocketSelection is intentionally not used here. Fusion can
    resolve that selection to the floor on one imported part and to the
    surrounding material on its mirrored counterpart. Feed-distance probing
    cannot reliably detect the latter when both regions are of comparable
    size. The floor's ordered co-edges preserve topology for circular,
    rectangular, concave, mirrored and rotated imported STEP geometry. Only
    the proven wall run is retained; its mouth is left open for Fusion to
    extend to the stock boundary.
    """
    if tool_axis is None:
        return None
    wall_coedges = _open_floor_wall_coedges(
        floor_face, recognized_boundary_face_ids, tool_axis
    )
    if not wall_coedges:
        return None
    normal_ok, floor_normal = floor_face.evaluator.getNormalAtPoint(
        floor_face.pointOnFace
    )
    if not normal_ok:
        return None
    normal_alignment = floor_normal.dotProduct(tool_axis)
    try:
        is_reverted = _open_floor_chain_should_revert(
            wall_coedges[0].isOpposedToEdge,
            normal_alignment,
        )
    except RuntimeError:
        return None
    mode = 'chain_true' if is_reverted else 'chain_false'
    futil.log(
        f'Open floor {floor_face.tempId} orientation resolved geometrically: "{mode}" '
        f'(first co-edge opposed={wall_coedges[0].isOpposedToEdge}, '
        f'normal·toolZ={normal_alignment:.6f}, {len(wall_coedges)} wall edge(s)).',
        force_console=True,
    )
    return mode, wall_coedges


def _create_pocket_operation(
    setup: adsk.cam.Setup,
    features: Sequence[PocketFeature],
    tool: adsk.cam.Tool,
    size: str,
    depth_mm: float,
    stepdown_mm: float,
    tool_axis: Optional[adsk.core.Vector3D],
) -> Tuple[Optional[adsk.cam.OperationBase], List[str]]:
    warnings: List[str] = []
    resolved_features: List[PocketFeature] = []
    selection_by_face_id: Dict[
        int, Tuple[str, List[adsk.fusion.BRepCoEdge]]
    ] = {}
    for feature in features:
        if not feature.use_floor_boundary:
            resolved_features.append(feature)
            continue
        all_resolved = True
        for floor_face in feature.bottom_faces:
            resolved = _resolve_open_floor_orientation(
                floor_face,
                tool_axis,
                feature.recognized_boundary_face_ids,
            )
            if resolved is None:
                all_resolved = False
                break
            selection_by_face_id[floor_face.tempId] = resolved
        if all_resolved:
            resolved_features.append(feature)
        else:
            face_ids = [face.tempId for face in feature.bottom_faces]
            warnings.append(
                f'Open pocket floor {face_ids} has no single, unambiguous wall chain and '
                'exterior mouth — MACHINE MANUALLY.'
            )

    if not resolved_features:
        return None, warnings

    operation_input = setup.operations.createInput('pocket2d')
    pocket_count = len(resolved_features)
    operation_input.displayName = (
        f'AM – {size} pockets – {depth_mm:g} mm ({pocket_count})'
    )
    operation_input.tool = tool
    set_safe_heights(operation_input.parameters)
    set_expression(operation_input.parameters, 'doMultipleDepths', 'true')
    set_expression(operation_input.parameters, 'maximumStepdown', f'{stepdown_mm:g} mm')
    set_expression(operation_input.parameters, 'topHeight_mode', "'from contour'")
    set_expression(operation_input.parameters, 'topHeight_offset', f'{depth_mm:g} mm')
    set_expression(operation_input.parameters, 'bottomHeight_mode', "'from contour'")
    set_expression(operation_input.parameters, 'bottomHeight_offset', '0 mm')
    set_zero_stock_to_leave(operation_input.parameters)
    # See the same setting in entry.py's contour operation: the default
    # (~0.01 mm) chordal tolerance is far tighter than a CNC router needs and
    # measurably slows toolpath calculation on curved pocket boundaries.
    set_expression(operation_input.parameters, 'tolerance', '0.1 mm')

    pocket_parameter = operation_input.parameters.itemByName('pockets')
    if not pocket_parameter:
        raise RuntimeError('The 2D Pocket strategy does not expose pocket geometry selection.')
    pocket_value = pocket_parameter.value
    selections = pocket_value.getCurveSelections()
    _add_pocket_geometry_selections(
        selections, resolved_features, selection_by_face_id
    )
    pocket_value.applyCurveSelections(selections)
    operation = setup.operations.add(operation_input)
    _set_ramp_parameters(operation)
    return operation, warnings


def _add_pocket_geometry_selections(
    selections: adsk.cam.CurveSelections,
    features: Sequence[PocketFeature],
    selection_by_face_id: Dict[
        int, Tuple[str, List[adsk.fusion.BRepCoEdge]]
    ],
) -> None:
    """Adds normal floors to one PocketSelection and each verified open floor
    as an oriented wall chain extended to the closest stock boundary."""
    pocket_faces = [
        face for feature in features if not feature.use_floor_boundary
        for face in feature.bottom_faces
    ]
    if pocket_faces:
        pocket_selection = selections.createNewPocketSelection()
        pocket_selection.isSelectingSamePlaneFaces = False
        pocket_selection.inputGeometry = pocket_faces

    for feature in features:
        if not feature.use_floor_boundary:
            continue
        for floor_face in feature.bottom_faces:
            resolved = selection_by_face_id.get(floor_face.tempId)
            if resolved is None:
                raise RuntimeError(
                    f'Pocket floor face {floor_face.tempId} has no safe boundary orientation.'
                )
            mode, wall_coedges = resolved
            if mode not in ('chain_true', 'chain_false') or not wall_coedges:
                raise RuntimeError(
                    f'Pocket floor face {floor_face.tempId} has an invalid open boundary.'
                )
            chain_selection = selections.createNewChainSelection()
            chain_selection.isOpen = True
            chain_selection.isReverted = mode == 'chain_true'
            chain_selection.extensionType = adsk.cam.ExtensionTypes.BoundaryExtensionType
            chain_selection.extensionMethod = (
                adsk.cam.ExtensionMethods.ClosestBoundaryExtensionMethod
            )
            chain_selection.inputGeometry = [coedge.edge for coedge in wall_coedges]
            futil.log(
                f'Pocket geometry: open wall chain from face {floor_face.tempId}, extended '
                f'to the closest stock boundary (isReverted={chain_selection.isReverted}), '
                f'{len(wall_coedges)} edge(s), area {floor_face.area:.4f} cm^2.',
                force_console=True,
            )


_REFERENCE_FACE_AREA_TOLERANCE_CM2 = 0.5


def _select_boundary_profiles(
    sketch: adsk.fusion.Sketch,
    count: int,
    reference_area_cm2: Optional[float],
) -> List[adsk.fusion.Profile]:
    """Picks the `count` largest profiles in a boundary sketch, excluding
    one whose area matches the sketch's own reference face.

    The reference area is captured before the sketch is created and passed
    in here. Reading ``sketch.referencePlane`` after CAM operations exist can
    raise ``referencePlane is a BRefFace - need to roll timeline back before
    sketch`` in a parametric Fusion design.

    Confirmed on real output: _sketch_from_boundaries draws directly on a
    real body face (to avoid the construction-plane pitfalls documented
    there), and Fusion also treats that face's own full silhouette as one
    more sketch profile — completely unrelated to anything actually drawn,
    but indistinguishable from a real one by shape alone. Passing the whole
    sketch to a pocket/2D Contour operation (as a SketchSelection) let it
    include that spurious profile too. Combined with the real, much smaller
    boundary, Fusion could not resolve the two into one toolpath at all
    ("No passes to link. Generated toolpath is empty.") on a real part.
    Excluding it and keeping only the `count` largest of what remains
    assumes any other stray profile (e.g. one of the real boundary's own
    inner loops, also surfaced as its own standalone profile) is smaller
    than the genuine feature(s) — true on the real part this was diagnosed
    on (1900.88 cm^2 real vs. 24.19 cm^2 stray), not proven in general.
    """
    candidates: List[Tuple[float, adsk.fusion.Profile]] = []
    for profile in sketch.profiles:
        area = profile.areaProperties().area
        if (
            reference_area_cm2 is not None
            and abs(area - reference_area_cm2) <= _REFERENCE_FACE_AREA_TOLERANCE_CM2
        ):
            continue
        candidates.append((area, profile))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [profile for _area, profile in candidates[:count]]


def _create_through_slot_operation(
    setup: adsk.cam.Setup,
    sketch: adsk.fusion.Sketch,
    features: Sequence[ThroughSlotFeature],
    tool: adsk.cam.Tool,
    size: str,
    stepdown_mm: float,
    reference_area_cm2: Optional[float],
) -> adsk.cam.OperationBase:
    """Cuts a boundary that has no bottom face (RecognizedPocket
    isThrough=True) using a sketch (built by _sketch_from_boundaries, before
    the Setup existed) that traces it, since the face-based pocket selection
    used by _create_pocket_operation has no floor to select for this feature.
    This includes both non-circular slots and circular openings above the
    drilling diameter limit.

    A 2D Contour on the boundary, not a 2D Pocket clearing its interior: the
    region is already fully open (isThrough=True means there is no material
    there to clear), so pocketing it wastefully re-cuts empty space and — worse
    — a 2D Pocket fed a bare closed chain (no bounding face/profile of its
    own) can resolve which side is "inside" the wrong way and mill out
    everything OUTSIDE the boundary instead, gutting the surrounding part.
    Confirmed on real output (hou_plafond_L3_mid_klein_luik_boxer_OH-00201):
    tracing the same boundary as a 2D Contour with compensation='left' and
    isReverted=True keeps the tool on the boundary's inside — sharp toolpath
    corners hugging the loop — where isReverted=False rounds every corner
    outward (the tool offsetting away from the loop, cutting the wrong side),
    the same reversed-closed-loop behavior already confirmed in
    _create_nested_through_operation.

    NOTE: bottomHeight_mode/'from stock bottom' and the sign of
    bottomHeight_offset below are unverified against this Fusion version —
    confirm the actual cut depth in Fusion's simulation before machining real
    material.
    """
    operation_input = setup.operations.createInput('contour2d')
    operation_input.displayName = f'AM – {size} through slots ({len(features)})'
    operation_input.tool = tool
    set_safe_heights(operation_input.parameters)
    set_expression(operation_input.parameters, 'doMultipleDepths', 'true')
    set_expression(operation_input.parameters, 'maximumStepdown', f'{stepdown_mm:g} mm')
    # See the open-pocket operation below for why this is 'from contour' and
    # not 'from stock top': this slot's own local opening can sit lower than
    # another, taller feature elsewhere on the part. bottomHeight stays
    # stock-bottom-relative since a through cut must still reach the real
    # physical stock bottom regardless of the local top height.
    set_expression(operation_input.parameters, 'topHeight_mode', "'from contour'")
    set_expression(operation_input.parameters, 'topHeight_offset', '0 mm')
    set_expression(operation_input.parameters, 'bottomHeight_mode', "'from stock bottom'")
    # Small overcut past the stock bottom, mirroring the through-drill
    # breakThroughDepth convention in holes.py, so the cut reliably clears the
    # far side. UNVERIFIED sign — check the resulting depth in Fusion.
    set_expression(operation_input.parameters, 'bottomHeight_offset', '-1 mm')
    set_zero_stock_to_leave(operation_input.parameters)
    set_expression(operation_input.parameters, 'tolerance', '0.1 mm')
    set_expression(operation_input.parameters, 'compensation', "'left'")
    set_expression(operation_input.parameters, 'compensationType', "'computer'")

    contour_parameter = operation_input.parameters.itemByName('contours')
    if not contour_parameter:
        raise RuntimeError('The 2D Contour strategy does not expose contour geometry selection.')
    contour_value = contour_parameter.value
    profiles = _select_boundary_profiles(sketch, len(features), reference_area_cm2)
    if len(profiles) != len(features):
        raise RuntimeError(
            f'Expected {len(features)} through-slot boundary profile(s) in "{sketch.name}" but '
            f'found {len(profiles)} after excluding the reference face\'s own outline.'
        )
    selections = contour_value.getCurveSelections()
    for profile in profiles:
        chain_selection = selections.createNewChainSelection()
        chain_selection.isOpen = False
        chain_selection.inputGeometry = [
            profile_curve.sketchEntity
            for loop in profile.profileLoops
            for profile_curve in loop.profileCurves
        ]
        # See docstring: keeps the tool on the boundary's inside.
        chain_selection.isReverted = True
    contour_value.applyCurveSelections(selections)
    operation = setup.operations.add(operation_input)
    # Must be set AFTER operations.add(), and on 'entry_radius' (the real
    # parameter), not the read-only 'leadInRadius' mirror — see
    # _create_contour_operation.
    set_expression(operation.parameters, 'entry_radius', '0 mm')
    return operation


def _create_open_pocket_operation(
    setup: adsk.cam.Setup,
    curves: Sequence[adsk.fusion.SketchCurve],
    features: Sequence[OpenPocketFeature],
    tool: adsk.cam.Tool,
    size: str,
    depth_mm: float,
    stepdown_mm: float,
) -> adsk.cam.OperationBase:
    """Machines a blind pocket whose recognized region touches the outer
    silhouette (e.g. a groove routed in from an edge, open at one end).

    Confirmed on a real part via _log_recognition_summary that
    RecognizedPocket gives no floor face for these (isThrough=False,
    isClosed=False; faces/sharedFaces contain only walls and, for a rounded
    end, a non-planar cap) — _pocket_bottom_faces() can never succeed here.
    Machined from a sketch boundary via _sketch_from_boundaries, same as a
    through slot, but cut to the recognized depth instead of all the way
    through stock. Unlike a through slot's boundary, this one is genuinely
    open at the pocket's mouth (no wall face exists there to project an
    edge from), so it's fed through a ChainSelection with isOpen=True
    instead of a SketchSelection needing a closed loop — Fusion extends an
    open chain to the stock boundary itself to complete it. Confirmed
    correct via a manually built "Open Chain" 2D Pocket operation in Fusion
    on a real part with this exact shape (a shallow shelf next to a taller
    boss) before porting the selection logic here.

    NOTE: the sign of bottomHeight_offset below is unverified against this
    Fusion version, exactly like the through-slot warning above — confirm the
    actual cut depth in Fusion's simulation before machining real material.
    """
    operation_input = setup.operations.createInput('pocket2d')
    operation_input.displayName = (
        f'AM – {size} open pockets – {depth_mm:g} mm ({len(features)})'
    )
    operation_input.tool = tool
    set_safe_heights(operation_input.parameters)
    set_expression(operation_input.parameters, 'doMultipleDepths', 'true')
    set_expression(operation_input.parameters, 'maximumStepdown', f'{stepdown_mm:g} mm')
    # 'from stock top' assumes this pocket's own local opening sits at the
    # SAME height as the model's global stock top — false whenever another
    # feature elsewhere on the part is taller (confirmed on real output: a
    # 2.3 mm-deep shelf next to a raised boss with 1.36 cm-deep holes got a
    # toolpath cut relative to the boss's height, landing near the top
    # surface instead of down in the actual recess). 'from contour' anchors
    # to the selected sketch's own real Z instead, the same pattern already
    # used successfully in _create_pocket_operation for face-selected floors.
    set_expression(operation_input.parameters, 'topHeight_mode', "'from contour'")
    set_expression(operation_input.parameters, 'topHeight_offset', '0 mm')
    set_expression(operation_input.parameters, 'bottomHeight_mode', "'from contour'")
    set_expression(operation_input.parameters, 'bottomHeight_offset', f'-{depth_mm:g} mm')
    set_zero_stock_to_leave(operation_input.parameters)
    set_expression(operation_input.parameters, 'tolerance', '0.1 mm')

    if not curves:
        raise RuntimeError(
            'No boundary curves were projected for this open pocket — nothing to select.'
        )
    pocket_parameter = operation_input.parameters.itemByName('pockets')
    if not pocket_parameter:
        raise RuntimeError('The 2D Pocket strategy does not expose pocket geometry selection.')
    pocket_value = pocket_parameter.value
    selections = pocket_value.getCurveSelections()
    chain_selection = selections.createNewChainSelection()
    chain_selection.isOpen = True
    # isReverted picks which side of the open chain is "inside" the pocket
    # to clear. False cleared everything EXCEPT the small pocket (confirmed
    # on real output: the toolpath spanned the whole part) — the opposite of
    # what's wanted, so this is flipped to True. UNVERIFIED against a
    # passing real run yet — confirm in Fusion's simulation.
    chain_selection.isReverted = True
    chain_selection.inputGeometry = list(curves)
    pocket_value.applyCurveSelections(selections)
    operation = setup.operations.add(operation_input)
    _set_ramp_parameters(operation)
    return operation


def _nested_chain_should_revert(
    first_coedge_is_opposed_to_edge: bool,
    floor_normal_dot_tool_axis: float,
) -> bool:
    """Returns the ChainSelection direction that puts Left compensation
    inside an inner face loop.

    Fusion disables automatic tool-side detection for API-created
    ChainSelections and derives the initial side from the first input edge.
    A BRepLoop's co-edges, unlike its bare edges, retain the loop traversal
    direction. In that traversal the owning face lies on one side and an
    inner loop's opening on the other. The result must also be inverted when
    the owning face normal opposes the setup tool axis.

    ``isOpposedToEdge`` converts the co-edge traversal into the first
    BRepEdge's natural direction, which is the direction Fusion actually
    uses as its ChainSelection reference. This makes the result independent
    of STEP topology ordering and of mirrored left/right parts.
    """
    if abs(floor_normal_dot_tool_axis) <= geometry.DIRECTION_LENGTH_EPSILON:
        raise RuntimeError(
            'Nested through-opening floor normal is perpendicular to the setup tool axis.'
        )
    if floor_normal_dot_tool_axis > 0.0:
        return not first_coedge_is_opposed_to_edge
    return first_coedge_is_opposed_to_edge


def _create_nested_through_operation(
    setup: adsk.cam.Setup,
    features: Sequence[NestedThroughFeature],
    tool: adsk.cam.Tool,
    depth_mm: float,
    tool_axis: adsk.core.Vector3D,
) -> adsk.cam.OperationBase:
    """Cuts a closed inner boundary already present on a recognized pocket's
    own floor face — see NestedThroughFeature — from that floor's own depth
    down through the stock bottom. A 2D Contour with a closed ChainSelection
    over the loop's real edges (not a fixed pocket/hole shape), since the
    opening can be any shape, not just circular.

    The travel direction is resolved separately for every loop from its
    co-edge orientation and the floor normal relative to the setup tool
    axis. It must not be a fixed isReverted value: mirrored L/R parts can
    expose the same inner loop with opposite BRepEdge directions.
    """
    operation_input = setup.operations.createInput('contour2d')
    operation_input.displayName = f'AM – Nested through openings – {depth_mm:g} mm ({len(features)})'
    operation_input.tool = tool
    set_safe_heights(operation_input.parameters)
    set_expression(operation_input.parameters, 'topHeight_mode', "'from contour'")
    set_expression(operation_input.parameters, 'topHeight_offset', '0 mm')
    # Mirrors _create_through_slot_operation: this opening must reach the
    # real physical stock bottom regardless of this floor's own local depth.
    set_expression(operation_input.parameters, 'bottomHeight_mode', "'from stock bottom'")
    set_expression(operation_input.parameters, 'bottomHeight_offset', '-1 mm')
    set_expression(operation_input.parameters, 'doMultipleDepths', 'true')
    set_zero_stock_to_leave(operation_input.parameters)
    set_expression(operation_input.parameters, 'tolerance', '0.1 mm')
    set_expression(operation_input.parameters, 'compensation', "'left'")
    set_expression(operation_input.parameters, 'compensationType', "'computer'")

    contour_parameter = operation_input.parameters.itemByName('contours')
    if not contour_parameter:
        raise RuntimeError('The 2D Contour strategy does not expose contour geometry selection.')
    contour_value = contour_parameter.value
    selections = contour_value.getCurveSelections()
    for feature in features:
        coedges = list(feature.inner_loop.coEdges)
        if not coedges:
            raise RuntimeError(
                f'Nested through-opening loop on floor face '
                f'{feature.floor_face.tempId} has no co-edges.'
            )
        normal_ok, floor_normal = feature.floor_face.evaluator.getNormalAtPoint(
            feature.floor_face.pointOnFace
        )
        if not normal_ok:
            raise RuntimeError(
                f'Could not evaluate the normal of nested through-opening floor face '
                f'{feature.floor_face.tempId}.'
            )
        normal_alignment = floor_normal.dotProduct(tool_axis)
        chain_selection = selections.createNewChainSelection()
        chain_selection.isOpen = False
        # Preserve BRepLoop traversal order explicitly. loop.edges contains
        # the same entities but loses each edge's orientation in this face.
        chain_selection.inputGeometry = [coedge.edge for coedge in coedges]
        chain_selection.isReverted = _nested_chain_should_revert(
            coedges[0].isOpposedToEdge,
            normal_alignment,
        )
        futil.log(
            f'Nested through opening: closed chain on floor face {feature.floor_face.tempId}, '
            f'span {feature.span_cm * 10.0:.2f} mm, '
            f'first co-edge opposed={coedges[0].isOpposedToEdge}, '
            f'normal·toolZ={normal_alignment:.6f}, '
            f'isReverted={chain_selection.isReverted}.',
            force_console=True,
        )
    contour_value.applyCurveSelections(selections)
    operation = setup.operations.add(operation_input)
    # Must be set AFTER operations.add(), and on 'entry_radius' (the real
    # parameter), not the read-only 'leadInRadius' mirror — see
    # _create_contour_operation.
    set_expression(operation.parameters, 'entry_radius', '0 mm')
    return operation


def _saw_chain_should_revert(
    path_edge: adsk.fusion.BRepEdge,
    wall_face: adsk.fusion.BRepFace,
    attack_vector: adsk.core.Vector3D,
) -> bool:
    """Computes, from real geometry, which travel direction along path_edge
    puts the tilted saw blade's "compensation: left" cut on the correct side
    of the groove — replacing a blind, hardcoded `isReverted = True` that
    only happened to hold for the specific edge/wall pairing every plain
    groove on the confirmed-working parts (Hou_plafond_L3_achter_boxer_OH-
    00224/00198) uses.

    Calibrated against those same confirmed-correct grooves: for path_edge's
    own direction (start to end) crossed with wall_face's outward normal,
    dotted with attack_vector, the working grooves all measure exactly -1.0
    (cross(edge_dir, wall_normal) points opposite attack_vector) with
    isReverted=True. A dogbone groove arm (see _dogbone_saw_arms) measures
    +1.0 instead — confirmed on real output (Hou_plafond_L3_mid_fan_boxer_OH
    -00206 and hou_plafond_L3_mid_klein_luik_boxer_OH-00201, both regressed
    to cutting the wrong side of those arms until this was made geometric
    instead of a constant) — because path_edge there is one specific edge
    from a same-direction cluster pairing, not "whichever edge order
    happened to yield isReverted=True" the way the constant was tuned for.
    This sign, not the edge's own identity, is what should decide the
    travel direction, so it now decides isReverted directly instead of
    assuming every SawSlotFeature shares the same fixed convention.
    """
    start = path_edge.startVertex.geometry
    end = path_edge.endVertex.geometry
    edge_direction = start.vectorTo(end)
    if edge_direction.length < geometry.DIRECTION_LENGTH_EPSILON:
        return True
    edge_direction.normalize()
    ok, wall_normal = wall_face.evaluator.getNormalAtPoint(wall_face.pointOnFace)
    if not ok:
        return True
    cross = edge_direction.crossProduct(wall_normal)
    return cross.dotProduct(attack_vector) < 0.0


def _create_saw_slot_operation(
    setup: adsk.cam.Setup,
    features: Sequence[SawSlotFeature],
    tool: adsk.cam.Tool,
    depth_mm: float,
    attack_vector: adsk.core.Vector3D,
) -> adsk.cam.OperationBase:
    """Machines a SAW_SLOT_WIDTH_MM groove with a saw-blade-shaped Slot Mill
    tool tilted 90 degrees, instead of milling it like an ordinary pocket.

    Uses a 2D Contour strategy (not 2D Pocket) because the tool's own width
    already matches the groove exactly — there is no area to clear, only a
    kerf to cut — following the groove floor's own boundary edges, the same
    geometry _add_pocket_geometry_selections uses for ordinary open pockets.
    Unlike that pocket2d chain, this one must be an OPEN chain: a closed
    chain around the groove's thin floor loop produced a "Closed contour has
    0 area" empty toolpath on a real run, confirmed fixed by switching to
    Open. Every parameter set below, including the tool-orientation ones
    (overrideToolView/view_orientation_*), was read back from a manually
    built and visually confirmed 2D Contour operation in Fusion (Tool
    orientation: "Select Z axis/plane and X axis", referencing one of the
    groove's own wall faces) via a parameter-dump diagnostic script — not
    guessed.

    UNVERIFIED: this operation's behavior on a real saw-equipped machine.
    Confirm the toolpath, tool orientation and depth in Fusion's simulation
    before cutting real material.
    """
    operation_input = setup.operations.createInput('contour2d')
    operation_input.displayName = f'AM – Saw slots – {depth_mm:g} mm ({len(features)})'
    operation_input.tool = tool
    set_safe_heights(operation_input.parameters)
    set_expression(operation_input.parameters, 'doMultipleDepths', 'false')
    # 'from contour' anchors heights to this groove's own selected floor,
    # the same reasoning as _create_pocket_operation: the local top can sit
    # at a different height than another feature elsewhere on the part.
    set_expression(operation_input.parameters, 'topHeight_mode', "'from contour'")
    set_expression(operation_input.parameters, 'topHeight_offset', f'{depth_mm:g} mm')
    set_expression(operation_input.parameters, 'bottomHeight_mode', "'from contour'")
    set_expression(operation_input.parameters, 'bottomHeight_offset', '0 mm')
    set_zero_stock_to_leave(operation_input.parameters)
    set_expression(operation_input.parameters, 'tolerance', '0.1 mm')
    # Read back from the manually confirmed operation's parameter dump.
    set_expression(operation_input.parameters, 'compensation', "'left'")
    set_expression(operation_input.parameters, 'compensationType', "'computer'")

    # Tilts the tool 90 degrees off the setup's normal Z axis, referencing
    # this groove's own wall face as the "Select Z axis/plane and X axis"
    # tool orientation — the setting that makes a saw blade cut sideways
    # instead of straight down.
    set_expression(operation_input.parameters, 'overrideToolView', 'true')
    set_expression(operation_input.parameters, 'view_orientation_mode', "'axesZX'")
    axis_z_parameter = operation_input.parameters.itemByName('view_orientation_axisZ')
    if not axis_z_parameter:
        raise RuntimeError('The 2D Contour strategy does not expose view_orientation_axisZ.')
    # This ONE face sets the tool's actual physical tilt for every groove in
    # the operation, not just features[0]'s own groove: every chain's
    # isReverted below must be computed against this SAME face, not each
    # feature's own wall_face. Confirmed on real output
    # (hou_plafond_L3_achter_klein_luik_boxer_OH-00200): _wall_face_for_floor
    # /_wall_face_for_edge's "first adjacent candidate found" pick is not
    # consistently the same physical side across independently-recognized
    # grooves — some rows' own wall_face normal came back +Y, others -Y, on
    # an otherwise uniform parallel-groove pattern. Using each row's own
    # (inconsistent) wall_face to compute isReverted while the actual tilt
    # stays fixed to shared_wall_face produced exactly the reported
    # symptom: some rows cut correctly, others reversed.
    shared_wall_face = features[0].wall_face
    axis_z_parameter.value.value = [shared_wall_face]

    contour_parameter = operation_input.parameters.itemByName('contours')
    if not contour_parameter:
        raise RuntimeError('The 2D Contour strategy does not expose contour geometry selection.')
    contour_value = contour_parameter.value
    selections = contour_value.getCurveSelections()
    for feature in features:
        for floor_face in feature.floor_faces:
            if feature.path_edge is not None:
                # One arm of a dogbone groove (see _dogbone_saw_arms): the
                # floor's own "longest edge" is ambiguous or wrong here (both
                # arms' matching walls are equal-length collinear segments,
                # split apart by the bulge in the middle), so the specific
                # edge already identified for this arm is used directly
                # instead of recomputing it from the whole (shared) floor
                # face's boundary loop.
                path_edge = feature.path_edge
            else:
                outer_loop = next((loop for loop in floor_face.loops if loop.isOuter), None)
                if outer_loop is None:
                    raise RuntimeError(
                        f'Saw slot floor face {floor_face.tempId} has no outer boundary loop.'
                    )
                edges = list(outer_loop.edges)
                if not edges:
                    raise RuntimeError(
                        f'Saw slot floor face {floor_face.tempId} has an empty outer boundary loop.'
                    )
                # The saw travels the groove's length in a single pass, so the
                # path is just one of the floor's two long boundary edges (its
                # own walls) — not the full rectangle, which also includes the
                # two short end edges. Confirmed by the user against a real
                # generated toolpath, together with isOpen=True below (a closed
                # chain around the full rectangle produced an empty "Closed
                # contour has 0 area" toolpath).
                path_edge = max(edges, key=lambda edge: edge.length)
            chain_selection = selections.createNewChainSelection()
            chain_selection.isOpen = True
            chain_selection.inputGeometry = [path_edge]
            # The edge's own natural curve direction does not consistently
            # correspond to which side "compensation: left" ends up cutting
            # on — confirmed both on the original real output
            # (Hou_plafond_L3_achter_..._OH-00198, needed True) and later on
            # dogbone groove arms (needed False instead, see
            # _saw_chain_should_revert) — so this is computed per feature
            # from real geometry rather than assumed constant.
            # Computed against shared_wall_face (the actual tool-tilt
            # reference for this whole operation), not feature.wall_face —
            # see the comment above shared_wall_face's assignment.
            chain_selection.isReverted = _saw_chain_should_revert(
                path_edge, shared_wall_face, attack_vector
            )
            futil.log(
                f'Saw slot geometry: open chain from face {floor_face.tempId}, '
                f'edge length {path_edge.length:.2f} cm, wall face {feature.wall_face.tempId} '
                f'(tilt reference {shared_wall_face.tempId}).',
                force_console=True,
            )
    contour_value.applyCurveSelections(selections)
    operation = setup.operations.add(operation_input)
    # Fusion's default (nonzero) horizontal lead-in radius left a gap in the
    # cut contour near the lead-in point instead of a full pass around the
    # boundary. Confirmed on real output; requested by the user for every 2D
    # Contour operation this add-in creates. Must be set AFTER operations.add()
    # on 'entry_radius' — that's the true editable parameter behind the UI's
    # "Horizontal Lead-In Radius" field. 'leadInRadius' looks like the same
    # thing and accepts writes, but it's a read-only derived mirror that
    # Fusion recomputes from 'entry_radius' on every recompute (e.g. a
    # sibling operation being added to or removed from this setup), silently
    # reverting a direct write back to the tool preset. Confirmed on real
    # output.
    set_expression(operation.parameters, 'entry_radius', '0 mm')
    return operation


def _sketch_from_boundaries(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
    z_axis_index: int,
    features: Sequence[BoundaryFeature],
) -> Tuple[adsk.fusion.Sketch, List[adsk.fusion.SketchCurve], Optional[float]]:
    component = body.parentComponent
    _log_component_context(body)
    # A ConstructionPlaneInput defined with setByPlane can only be added in a
    # direct-modeling design. Most source files use parametric modeling, where
    # Fusion raises "3 : Environment is not supported" from
    # ConstructionPlanes.add(). Prefer a real planar model face parallel to
    # the setup XY plane. This is essential for STEP models whose own axes are
    # slightly rotated relative to the global origin planes.
    sketch_plane = _sketch_plane_for_features(body, z_axis, features)
    if sketch_plane is None:
        _log_no_sketch_plane_face(body, z_axis)
        # The recognized boundary points do not need a plane at their own Z
        # offset: SketchCurves projects 3D input points onto this parallel
        # fallback plane, following Autodesk's official recognition sample.
        if z_axis_index == 0:
            sketch_plane = component.yZConstructionPlane
        elif z_axis_index == 1:
            sketch_plane = component.xZConstructionPlane
        elif z_axis_index == 2:
            sketch_plane = component.xYConstructionPlane
        else:
            raise ValueError(f'Unsupported machining Z axis index: {z_axis_index}.')

    _log_sketch_plane_choice(sketch_plane)
    reference_face = adsk.fusion.BRepFace.cast(sketch_plane)
    reference_area_cm2 = reference_face.area if reference_face else None
    sketch = component.sketches.add(sketch_plane)
    sketch.name = 'AM – Through slot boundary'

    # Diagnostics on real Fusion output (see AI_NOTES.md) proved that hand-
    # built sketch points from pocket.boundaries' Point3D coordinates can
    # land far from the body: this body's owning component is not the
    # design's root component, and SketchPoint.worldGeometry applies the
    # full occurrence transform chain that raw Point3D input does not go
    # through. Projecting the pocket's real BRep edges instead of
    # reconstructing points lets Fusion handle that conversion itself, the
    # same way every other operation in this add-in references existing
    # geometry rather than rebuilding coordinates by hand.
    plane_origin, plane_normal = _plane_origin_and_normal(sketch_plane)
    label_by_token: Dict[str, str] = {}
    projected_curves: List[adsk.fusion.SketchCurve] = []
    for feature in features:
        for face in feature.faces:
            edge = _closest_edge_to_plane(face, plane_origin, plane_normal)
            if edge is None:
                _log_no_plane_edge(face)
                continue
            _log_projected_edge(face, edge)
            label = f'{face.area:.4f} cm^2 face / {edge.length:.2f} cm edge'
            for entity in sketch.project(edge):
                projected_curves.append(entity)
                try:
                    label_by_token[entity.entityToken] = label
                except (AttributeError, RuntimeError):
                    pass
    _merge_coincident_endpoints(sketch)
    _log_open_boundary_endpoints(sketch, label_by_token)
    return sketch, projected_curves, reference_area_cm2


def _plane_origin_and_normal(sketch_plane) -> Tuple[adsk.core.Point3D, adsk.core.Vector3D]:
    face = adsk.fusion.BRepFace.cast(sketch_plane)
    if face:
        plane = adsk.core.Plane.cast(face.geometry)
        return plane.origin, plane.normal
    construction_plane = adsk.fusion.ConstructionPlane.cast(sketch_plane)
    plane = construction_plane.geometry
    return plane.origin, plane.normal


def _closest_edge_to_plane(
    face: adsk.fusion.BRepFace,
    plane_origin: adsk.core.Point3D,
    plane_normal: adsk.core.Vector3D,
) -> Optional[adsk.fusion.BRepEdge]:
    """Returns this face's edge nearest to the sketch plane — its own top
    edge, where it meets the part's top surface, or a circular opening's top
    rim. Uses the closest edge rather than a fixed distance tolerance: a
    fixed 0.01 cm tolerance found zero matching edges on a real part because
    the true top edge sits a genuine ~0.05 cm offset below the chosen sketch
    plane's face (confirmed on real Fusion output, not import round-off — see
    AI_NOTES.md). The other edges of a simple wall/cap face (bottom, sides)
    sit much farther away (comparable to the pocket's own depth), so picking
    the nearest one reliably finds the top edge without needing to guess the
    right absolute tolerance."""
    best_edge: Optional[adsk.fusion.BRepEdge] = None
    best_distance: Optional[float] = None
    for edge in face.edges:
        point = edge.pointOnEdge
        offset = plane_origin.vectorTo(point)
        distance = abs(offset.dotProduct(plane_normal))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_edge = edge
    if best_edge is not None and best_distance is not None and best_distance <= PLANE_EDGE_MAX_DISTANCE_CM:
        return best_edge
    return None


def _log_projected_edge(face: adsk.fusion.BRepFace, edge: adsk.fusion.BRepEdge) -> None:
    """A pocket wall face should contribute one short edge (its own top
    edge) to the boundary sketch. If a wall face also shares an edge with
    the part's outer silhouette at the pocket's open mouth, that edge could
    be picked instead and would be much longer than the pocket itself — log
    the projected edge's length so an unexpectedly long one is obvious
    instead of inferred from the resulting toolpath."""
    futil.log(
        f'Projecting 1 edge ({edge.length:.2f} cm) from a face ({face.area:.4f} cm^2).',
        force_console=True,
    )


def _log_no_plane_edge(face: adsk.fusion.BRepFace) -> None:
    futil.log(
        f'No edge within {PLANE_EDGE_MAX_DISTANCE_CM:g} cm of the sketch plane on a face '
        f'({face.area:.4f} cm^2) — nothing projected for it.',
        force_console=True,
    )


def _largest_parallel_planar_face(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
) -> Optional[adsk.fusion.BRepFace]:
    candidates = _z_parallel_planar_faces(body, z_axis)
    return max(candidates, key=lambda face: face.area) if candidates else None


def _largest_same_direction_planar_face(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
) -> Optional[adsk.fusion.BRepFace]:
    """Like _largest_parallel_planar_face, but only considers faces whose
    normal actually points the same way as z_axis, not merely parallel to
    its line. _z_parallel_planar_faces (and therefore
    _largest_parallel_planar_face) matches by absolute cosine, so on an
    ordinary flat panel it treats the top and bottom face as equally
    "parallel" and can return the far, opposite-facing surface — e.g. the
    body-wide top face — instead of the near reference surface a caller
    measuring depth from z_axis's own side actually wants. Confirmed on real
    output: used unfiltered, this picked an 11339.87 cm^2 opposite-facing
    face over the intended 508.75 cm^2 near one, offsetting every depth
    measurement by the panel's full thickness."""
    best: Optional[adsk.fusion.BRepFace] = None
    best_area = -1.0
    for face in _z_parallel_planar_faces(body, z_axis):
        ok, normal = face.evaluator.getNormalAtPoint(face.pointOnFace)
        if not ok or normal.dotProduct(z_axis) <= 0:
            continue
        if face.area > best_area:
            best = face
            best_area = face.area
    return best


def _z_parallel_planar_faces(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
) -> List[adsk.fusion.BRepFace]:
    candidates: List[adsk.fusion.BRepFace] = []
    for face in body.faces:
        if not adsk.core.Plane.cast(face.geometry):
            continue
        ok, normal = face.evaluator.getNormalAtPoint(face.pointOnFace)
        if not ok or normal.length < geometry.DIRECTION_LENGTH_EPSILON:
            continue
        cosine = abs(normal.dotProduct(z_axis) / (normal.length * z_axis.length))
        if cosine >= SKETCH_PLANE_ALIGNMENT_COS_THRESHOLD:
            candidates.append(face)
    return candidates


def _sketch_plane_for_features(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
    features: Sequence[BoundaryFeature],
) -> Optional[adsk.fusion.BRepFace]:
    """Picks the body face that this feature's own walls actually reach,
    instead of unconditionally the single largest z-parallel face on the
    whole body. A stepped/tapered part can have several z-parallel faces at
    different heights; always picking the biggest one regardless of height
    left a lower step's wall faces farther than PLANE_EDGE_MAX_DISTANCE_CM
    from the chosen plane — every wall then failed _closest_edge_to_plane,
    the sketch ended up with no projected geometry, and the resulting 2D
    Pocket operation cleared the outer silhouette instead of the intended
    pocket (confirmed on real output: 6/6 wall faces logged "No edge within
    0.5 cm of the sketch plane"). Falls back to the old largest-face choice
    when no candidate matches any wall face, so behavior is unchanged for
    parts where that already worked."""
    boundary_faces = [face for feature in features for face in feature.faces]
    return _best_matching_z_face(body, z_axis, boundary_faces)


def _best_matching_z_face(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
    boundary_faces: Sequence[adsk.fusion.BRepFace],
) -> Optional[adsk.fusion.BRepFace]:
    """Shared by _sketch_plane_for_features and the open-blind-pocket depth
    reference in recognize_pocket_features: picks whichever z-parallel body
    face these specific boundary/wall faces actually reach (by edge
    proximity), instead of the single largest z-parallel face on the whole
    body. The same multi-level-part mismatch that broke the sketch plane
    (see above) also broke floor-adjacency depth matching — a pocket's real
    floor got rejected at every hop because its depth was measured against
    the body-global top instead of this pocket's own local top, offsetting
    every measured depth by a constant amount (confirmed on real output: a
    2.3 mm-deep pocket's real floor candidates were rejected at 0.78-1.36 cm
    instead of ~0.23 cm). Falls back to the largest candidate when none
    match, so behavior is unchanged for single-level parts."""
    candidates = _z_parallel_planar_faces(body, z_axis)
    if not candidates:
        return None
    candidates.sort(key=lambda face: face.area, reverse=True)

    def match_count(candidate: adsk.fusion.BRepFace) -> int:
        origin, normal = _plane_origin_and_normal(candidate)
        return sum(
            1 for face in boundary_faces
            if _closest_edge_to_plane(face, origin, normal) is not None
        )

    best = max(candidates, key=match_count)
    return best if match_count(best) > 0 else candidates[0]


def _log_no_sketch_plane_face(
    body: adsk.fusion.BRepBody,
    z_axis: adsk.core.Vector3D,
) -> None:
    """_sketch_from_boundaries() found no body face within
    SKETCH_PLANE_ALIGNMENT_TOLERANCE_DEG of z_axis and is about to fall back
    to a component-origin construction plane. For an off-origin STEP import
    that plane can sit far from the actual geometry, so surface the closest
    real candidate's angle to tell a too-tight tolerance apart from a
    genuinely missing face instead of guessing."""
    total_count = 0
    planar_count = 0
    closest_angle_deg: Optional[float] = None
    for face in body.faces:
        total_count += 1
        if not adsk.core.Plane.cast(face.geometry):
            continue
        planar_count += 1
        ok, normal = face.evaluator.getNormalAtPoint(face.pointOnFace)
        if not ok:
            continue
        angle_deg = _angle_between_deg(normal, z_axis)
        if closest_angle_deg is None or angle_deg < closest_angle_deg:
            closest_angle_deg = angle_deg
    if closest_angle_deg is None:
        futil.log(
            f'No sketch plane face: {total_count} face(s), {planar_count} planar, none had '
            'a usable normal. Falling back to a component-origin construction plane.',
            force_console=True,
        )
    else:
        futil.log(
            f'No sketch plane face within {SKETCH_PLANE_ALIGNMENT_TOLERANCE_DEG:g} deg: '
            f'{total_count} face(s), {planar_count} planar, closest was '
            f'{closest_angle_deg:.4f} deg off the machining Z axis. Falling back to a '
            'component-origin construction plane, which can land far from off-origin STEP '
            'geometry.',
            force_console=True,
        )


def _log_component_context(body: adsk.fusion.BRepBody) -> None:
    """A sketch lives in a Component's own local space, but SketchPoint's
    worldGeometry applies the full occurrence transform chain up to the
    root component. If this body's owning component is reached through a
    non-identity occurrence (assemblyContext), that transform alone could
    explain a trusted, on-plane point coming back shifted after being drawn
    into a sketch and read back via worldGeometry — check that theory
    directly instead of guessing further from coordinate arithmetic."""
    try:
        design = adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct)
        is_root = bool(design and body.parentComponent == design.rootComponent)
    except (AttributeError, RuntimeError) as exc:
        is_root = f'<error: {exc}>'
    context = body.assemblyContext
    context_desc = 'None (direct root geometry)' if context is None else context.fullPathName
    futil.log(
        f'Component context: parentComponent="{body.parentComponent.name}", '
        f'is root component={is_root}, assemblyContext={context_desc}.',
        force_console=True,
    )


def _log_sketch_plane_choice(sketch_plane) -> None:
    """Logs where the chosen sketch plane actually is, so a mis-placed
    boundary sketch can be told apart from a mis-chosen plane instead of
    inferred indirectly from where the drawn curves end up."""
    face = adsk.fusion.BRepFace.cast(sketch_plane)
    if face:
        point = face.pointOnFace
        futil.log(
            f'Sketch plane: real face, area {face.area:.4f} cm^2, point on face '
            f'({point.x:.2f}, {point.y:.2f}, {point.z:.2f}).',
            force_console=True,
        )
        return
    construction_plane = adsk.fusion.ConstructionPlane.cast(sketch_plane)
    if construction_plane:
        origin = construction_plane.geometry.origin
        futil.log(
            f'Sketch plane: construction plane {construction_plane.name}, origin '
            f'({origin.x:.2f}, {origin.y:.2f}, {origin.z:.2f}).',
            force_console=True,
        )
        return
    futil.log(f'Sketch plane: unrecognized type {type(sketch_plane)!r}.', force_console=True)


def _merge_coincident_endpoints(sketch: adsk.fusion.Sketch) -> None:
    """Joins separately drawn boundary segments into closed sketch loops."""
    endpoints = []
    for line in sketch.sketchCurves.sketchLines:
        endpoints.extend((line.startSketchPoint, line.endSketchPoint))
    for arc in sketch.sketchCurves.sketchArcs:
        endpoints.extend((arc.startSketchPoint, arc.endSketchPoint))

    for index, point in enumerate(endpoints):
        if not _is_valid_sketch_point(point):
            continue
        for other_index in range(index + 1, len(endpoints)):
            other = endpoints[other_index]
            if (
                not _is_valid_sketch_point(other)
                or _same_sketch_point(point, other)
            ):
                continue
            if (
                point.geometry.distanceTo(other.geometry) <= ENDPOINT_MERGE_TOLERANCE_CM
                and point.merge(other)
            ):
                endpoints[other_index] = None


def _log_open_boundary_endpoints(
    sketch: adsk.fusion.Sketch,
    label_by_token: Optional[Dict[str, str]] = None,
) -> None:
    """After drawing and merging a boundary's curves, count sketch points
    touched by only one curve end. A closed loop has none of these; a
    recognized region that touches the part's outer silhouette (isClosed=
    False) has no drawn curve across that open side, so its endpoints there
    stay dangling — expected for an open-pocket boundary consumed via
    ChainSelection(isOpen=True), which extends it to the stock boundary
    itself, but still worth surfacing for a closed-loop consumer (through
    slots) where it would be a genuine anomaly. label_by_token (curve
    entityToken -> a description of the face/edge it was projected from)
    lets a dangling point be traced back to which specific edge it belongs
    to, instead of guessing pairings from bare coordinates."""
    label_by_token = label_by_token or {}
    endpoints: List[Tuple[adsk.fusion.SketchPoint, str]] = []
    for line in sketch.sketchCurves.sketchLines:
        label = label_by_token.get(_entity_token(line), '?')
        endpoints.append((line.startSketchPoint, f'{label} (start)'))
        endpoints.append((line.endSketchPoint, f'{label} (end)'))
    for arc in sketch.sketchCurves.sketchArcs:
        label = label_by_token.get(_entity_token(arc), '?')
        endpoints.append((arc.startSketchPoint, f'{label} (start)'))
        endpoints.append((arc.endSketchPoint, f'{label} (end)'))

    groups: List[List[Tuple[adsk.fusion.SketchPoint, str]]] = []
    for point, label in endpoints:
        if not _is_valid_sketch_point(point):
            continue
        matched_group = next(
            (
                group
                for group in groups
                if _same_sketch_point(point, group[0][0])
                or point.geometry.distanceTo(group[0][0].geometry) <= ENDPOINT_MERGE_TOLERANCE_CM
            ),
            None,
        )
        if matched_group is None:
            groups.append([(point, label)])
        else:
            matched_group.append((point, label))

    dangling = [group for group in groups if len(group) == 1]
    if dangling:
        details = '; '.join(
            f'{label} at ({point.geometry.x:.2f}, {point.geometry.y:.2f})'
            for group in dangling[:8]
            for point, label in group
        )
        futil.log(
            f'Sketch boundary has {len(dangling)} open (unclosed) endpoint(s) after merging: '
            f'{details}. Expected for an open-pocket boundary (extended to the stock edge by '
            'ChainSelection.isOpen); unexpected for a through-slot boundary, which should '
            'always close on its own.',
            force_console=True,
        )


def _entity_token(entity) -> str:
    try:
        return entity.entityToken
    except (AttributeError, RuntimeError):
        return ''


def _is_valid_sketch_point(point: Optional[adsk.fusion.SketchPoint]) -> bool:
    """False for null or solver-deleted points without dereferencing them."""
    if point is None:
        return False
    try:
        return point.isValid
    except RuntimeError:
        return False


def _same_sketch_point(
    first: adsk.fusion.SketchPoint,
    second: adsk.fusion.SketchPoint,
) -> bool:
    """Detects repeated API wrappers for one point before calling merge."""
    if first is second or first == second:
        return True
    try:
        return first.entityToken == second.entityToken
    except RuntimeError:
        return False
