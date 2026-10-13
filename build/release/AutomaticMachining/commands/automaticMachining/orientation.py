"""Detects the machining orientation (top direction and X/Y axes) of a body.

Hole recognition and pocket recognition both need to know which side of the
body is "up" before they can classify features as blind vs. through, or
choose the pocket attack direction. This module infers that from the body's
own geometry: blind cylindrical holes determine the machining side when
available, and the thinnest bounding-box axis determines the panel-thickness
axis otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import adsk.core
import adsk.fusion

from . import geometry

# Score contributed by a single blind cylindrical hole that opens towards a
# candidate top direction.
BLIND_HOLE_SCORE_WEIGHT = 100.0

# Two candidate orientations are considered tied when their scores differ by
# less than this amount.
SCORE_TIE_EPSILON = 1e-6

_AXIS_NAMES = ('X', 'Y', 'Z')


@dataclass
class Orientation:
    x: adsk.core.Vector3D
    y: adsk.core.Vector3D
    z: adsk.core.Vector3D
    z_axis_index: int
    used_fallback: bool
    reason: str


def flipped(value: Orientation) -> Orientation:
    """Returns the same machining frame viewed from the opposite side."""
    x = value.x.copy()
    z = value.z.copy()
    z.scaleBy(-1.0)
    y = z.crossProduct(x)
    y.normalize()
    return Orientation(
        x=x,
        y=y,
        z=z,
        z_axis_index=value.z_axis_index,
        used_fallback=value.used_fallback,
        reason=value.reason,
    )


def detect_orientation(body: adsk.fusion.BRepBody) -> Orientation:
    scores: Dict[Tuple[int, int], float] = {}
    # Keep the real model-space directions as well as their coarse global-axis
    # keys. Imported STEP geometry is often rotated by a small amount. Snapping
    # such a direction back to an exact global axis makes Fusion reject the
    # cylindrical faces in a Drill operation even though recognition accepted
    # them as axis-aligned.
    direction_signals: Dict[
        Tuple[int, int], List[Tuple[float, adsk.core.Vector3D]]
    ] = {}
    planar_directions: List[Tuple[float, Tuple[int, int], adsk.core.Vector3D]] = []
    box = body.boundingBox
    dimensions = (
        box.maxPoint.x - box.minPoint.x,
        box.maxPoint.y - box.minPoint.y,
        box.maxPoint.z - box.minPoint.z,
    )

    # Blind cylindrical holes provide the strongest indication of the top.
    for face in body.faces:
        cylinder = adsk.core.Cylinder.cast(face.geometry)
        if not cylinder or not geometry.is_internal_cylinder(face, cylinder):
            continue
        openings = geometry.cylinder_opening_directions(face)
        if len(openings) == 1:
            key = geometry.axis_key(openings[0])
            if key:
                scores[key] = scores.get(key, 0.0) + BLIND_HOLE_SCORE_WEIGHT
                direction_signals.setdefault(key, []).append(
                    (BLIND_HOLE_SCORE_WEIGHT, openings[0].copy())
                )

    # Keep planar directions only so the selected coarse axis can follow the
    # small angular error of imported STEP geometry.  Do not score the faces:
    # a sloped/stepped outline can contain an ordinary profile face inside its
    # bounding box, and treating that as a pocket floor can swap panel Y/Z.
    for face in body.faces:
        if not adsk.core.Plane.cast(face.geometry):
            continue
        ok, normal = face.evaluator.getNormalAtPoint(face.pointOnFace)
        if not ok:
            continue
        key = geometry.axis_key(normal)
        if not key:
            continue
        planar_directions.append((face.area, key, normal.copy()))

    used_fallback = False
    reason = ''
    if scores:
        selected_key, opposite_side_tie = _select_scored_direction(scores)
        z_axis, z_sign = selected_key
        # Use one strongest real geometric direction rather than an average.
        # Averaging almost-parallel imported surfaces can introduce exactly the
        # tiny angular error that Fusion's cylindrical-face selection rejects.
        z = max(direction_signals[selected_key], key=lambda item: item[0])[1]
        z.normalize()
        if opposite_side_tie:
            used_fallback = True
            reason = (
                f'Blind-hole signals were equal on both {_AXIS_NAMES[z_axis]} sides; '
                f'+{_AXIS_NAMES[z_axis]} was selected initially. Check the reported '
                'opposite-side features before machining.'
            )
    else:
        z_axis = min(range(3), key=lambda index: dimensions[index])
        z_sign = 1
        z = _best_planar_axis(planar_directions, z_axis, z_sign)
        if z is None:
            z = geometry.unit_axis(z_axis, z_sign)
        used_fallback = True
        reason = (
            'No blind hole determined the top side; the thinnest axis in the positive '
            'direction was selected as Z. Check the orientation before machining.'
        )

    remaining = [index for index in range(3) if index != z_axis]
    x_axis = max(remaining, key=lambda index: dimensions[index])
    x = _best_planar_axis(planar_directions, x_axis, 1)
    if x is None:
        x = geometry.unit_axis(x_axis, 1)
    # Matrix3D.setWithCoordinateSystem requires orthogonal axes. Project the
    # selected model direction onto the plane normal to the real Z direction;
    # for a clean orthogonal model this changes nothing beyond round-off.
    z_component = z.copy()
    z_component.scaleBy(x.dotProduct(z))
    x.subtract(z_component)
    if x.length < geometry.DIRECTION_LENGTH_EPSILON:
        x = geometry.unit_axis(x_axis, 1)
        z_component = z.copy()
        z_component.scaleBy(x.dotProduct(z))
        x.subtract(z_component)
    x.normalize()
    y = z.crossProduct(x)
    y.normalize()
    return Orientation(x, y, z, z_axis, used_fallback, reason)


def _select_scored_direction(
    scores: Dict[Tuple[int, int], float],
) -> Tuple[Tuple[int, int], bool]:
    """Selects a scored axis and tolerates a tie between its two sides.

    Equal signals on + and - of one axis still identify the panel-thickness
    axis.  The positive side is a deterministic initial choice; later pocket
    recognition can flip it when the opposite side has more pockets.  A tie
    across different axes remains unsafe and is rejected.
    """
    best_score = max(scores.values())
    best = [
        key for key, score in scores.items()
        if abs(score - best_score) < SCORE_TIE_EPSILON
    ]
    if len(best) == 1:
        return best[0], False
    axes = {key[0] for key in best}
    if len(axes) == 1:
        return max(best, key=lambda key: key[1]), True
    candidates = ', '.join(_describe_axis(key) for key in sorted(best))
    raise RuntimeError(
        'The machining axis is ambiguous because matching features score equally '
        f'on multiple axes ({candidates}). Rework the model so a single axis is '
        'clearly distinguished, or process it manually.'
    )


def _best_planar_axis(
    candidates: List[Tuple[float, Tuple[int, int], adsk.core.Vector3D]],
    axis_index: int,
    sign: int,
) -> Optional[adsk.core.Vector3D]:
    """Returns the largest planar face normal matching a requested axis.

    Face area makes the broad sides/edges of a panel win over tiny imported
    detail faces. The returned direction is flipped to the requested sign.
    """
    matching = [item for item in candidates if item[1][0] == axis_index]
    if not matching:
        return None
    _, key, direction = max(matching, key=lambda item: item[0])
    result = direction.copy()
    if key[1] != sign:
        result.scaleBy(-1.0)
    result.normalize()
    return result


def _describe_axis(key: Tuple[int, int]) -> str:
    axis_index, sign = key
    return ('+' if sign > 0 else '-') + _AXIS_NAMES[axis_index]
