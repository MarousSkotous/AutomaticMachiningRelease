"""Geometry helpers shared by orientation, hole and pocket recognition."""

from __future__ import annotations

from typing import List, Optional, Tuple

import adsk.core
import adsk.fusion

# A vector is treated as aligned to a global axis when its dominant component
# is at least this large (the vectors involved are always unit length, so this
# is effectively cos(angle)). 0.98 corresponds to roughly 11 degrees of slack,
# loose enough to tolerate STEP-import noise while still rejecting genuinely
# angled features.
AXIS_ALIGNMENT_COS_THRESHOLD = 0.98

# A face is classified as the internal wall of a hole when its outward normal
# points at least this far against the radial (centre-to-surface) direction.
INTERNAL_FACE_DOT_THRESHOLD = -0.5

# Below this vector length a direction is considered degenerate (e.g. two
# opening centres coincide) and is skipped rather than normalized.
DIRECTION_LENGTH_EPSILON = 1e-8


def axis_key(vector: adsk.core.Vector3D) -> Optional[Tuple[int, int]]:
    values = (vector.x, vector.y, vector.z)
    axis = max(range(3), key=lambda index: abs(values[index]))
    if abs(values[axis]) < AXIS_ALIGNMENT_COS_THRESHOLD:
        return None
    return axis, 1 if values[axis] >= 0 else -1


def unit_axis(index: int, sign: int) -> adsk.core.Vector3D:
    values = [0.0, 0.0, 0.0]
    values[index] = float(sign)
    return adsk.core.Vector3D.create(*values)


def point_coordinate(point: adsk.core.Point3D, index: int) -> float:
    return (point.x, point.y, point.z)[index]


def is_internal_cylinder(
    face: adsk.fusion.BRepFace,
    cylinder: adsk.core.Cylinder,
) -> bool:
    point = face.pointOnFace
    ok, normal = face.evaluator.getNormalAtPoint(point)
    if not ok:
        return False
    axis = cylinder.axis.copy()
    axis.normalize()
    origin_to_point = cylinder.origin.vectorTo(point)
    axial = axis.copy()
    axial.scaleBy(origin_to_point.dotProduct(axis))
    radial = origin_to_point.copy()
    radial.subtract(axial)
    if radial.length < 1e-8:
        return False
    radial.normalize()
    return normal.dotProduct(radial) < INTERNAL_FACE_DOT_THRESHOLD


def cylinder_opening_directions(face: adsk.fusion.BRepFace) -> List[adsk.core.Vector3D]:
    """Directions (from the cylinder's centre) in which the given cylindrical
    face is open, i.e. bordered by a non-outer loop on an adjacent planar
    face. A blind hole has one direction, a through hole has two."""
    opening_centers: List[adsk.core.Point3D] = []
    all_centers: List[adsk.core.Point3D] = []
    for edge in face.edges:
        circle = adsk.core.Circle3D.cast(edge.geometry)
        if not circle:
            continue
        all_centers.append(circle.center)
        for coedge in edge.coEdges:
            loop = coedge.loop
            adjacent = loop.face
            if adjacent == face or not adsk.core.Plane.cast(adjacent.geometry):
                continue
            if not loop.isOuter:
                opening_centers.append(circle.center)
                break

    if not all_centers or not opening_centers:
        return []
    midpoint = adsk.core.Point3D.create(
        sum(point.x for point in all_centers) / len(all_centers),
        sum(point.y for point in all_centers) / len(all_centers),
        sum(point.z for point in all_centers) / len(all_centers),
    )
    directions: List[adsk.core.Vector3D] = []
    direction_keys = set()
    for center in opening_centers:
        direction = midpoint.vectorTo(center)
        if direction.length < DIRECTION_LENGTH_EPSILON:
            continue
        direction.normalize()
        key = axis_key(direction)
        # Opposite directions are distinct here: they identify the two mouths
        # of a through-hole. Vector3D.isParallelTo treats them as equivalent.
        if key and key not in direction_keys:
            directions.append(direction)
            direction_keys.add(key)
    return directions
