"""Recognizes drillable axis-aligned cylindrical holes and builds drill ops."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import adsk.cam
import adsk.core
import adsk.fusion

from . import geometry, tool_library
from .cam_utils import set_expression, set_safe_heights

# Cylinder centre coordinates (Fusion's internal unit is centimetres) are
# compared after rounding to this many units per centimetre to decide
# whether two holes share the same axis line. 100000 / cm is 0.1 micrometre:
# tight enough to avoid merging distinct holes while absorbing floating
# point noise.
LINE_POSITION_PRECISION_PER_CM = 100000.0

# A Drill operation can only consume cylindrical faces parallel to its tool
# axis. Recognition uses a deliberately broad near-global-axis tolerance, but
# operation geometry must be far stricter. 0.001 degree still absorbs numeric
# noise while preventing Fusion's "Cylindrical face not in tool orientation"
# failure.
DRILL_AXIS_ALIGNMENT_COS_THRESHOLD = math.cos(math.radians(0.001))


@dataclass
class HoleFace:
    face: adsk.fusion.BRepFace
    diameter_cm: float
    is_through: bool


def find_hole_faces(
    body: adsk.fusion.BRepBody,
    top: adsk.core.Vector3D,
) -> Tuple[List[HoleFace], List[str]]:
    candidates: Dict[Tuple[int, int, int, int], List[Tuple[HoleFace, float]]] = {}
    top_key = geometry.axis_key(top)
    skipped_misaligned = 0
    for face in body.faces:
        cylinder = adsk.core.Cylinder.cast(face.geometry)
        if not cylinder or not geometry.is_internal_cylinder(face, cylinder):
            continue
        cylinder_key = geometry.axis_key(cylinder.axis)
        if not cylinder_key or not top_key or cylinder_key[0] != top_key[0]:
            continue
        if not _directions_parallel(cylinder.axis, top):
            skipped_misaligned += 1
            continue

        openings = geometry.cylinder_opening_directions(face)
        if not any(
            _directions_parallel(direction, top, same_direction=True)
            for direction in openings
        ):
            continue
        is_through = any(
            _directions_parallel(direction, top, same_direction=False)
            and direction.dotProduct(top) < 0.0
            for direction in openings
        )
        line_key = _cylinder_line_key(cylinder, top_key[0])
        hole = HoleFace(face, cylinder.radius * 2.0, is_through)
        candidates.setdefault(line_key, []).append((hole, cylinder.radius))

    result: List[HoleFace] = []
    skipped_stepped = 0
    for same_axis_holes in candidates.values():
        radii = {round(radius, 5) for _, radius in same_axis_holes}
        if len(radii) > 1:
            # A counterbore or another stepped hole must not be mistaken for
            # multiple independent drilling operations.
            skipped_stepped += 1
            continue
        result.append(same_axis_holes[0][0])

    warnings = []
    if skipped_misaligned:
        warnings.append(
            f'Holes not parallel to the setup tool axis skipped: {skipped_misaligned}.'
        )
    if skipped_stepped:
        warnings.append(f'Stepped or complex holes skipped: {skipped_stepped}.')
    return result, warnings


def _cylinder_line_key(
    cylinder: adsk.core.Cylinder,
    axis_index: int,
) -> Tuple[int, int, int, int]:
    # Project the origin onto the plane perpendicular to the cylinder axis.
    # Unlike dropping one global coordinate, this remains constant along a
    # slightly rotated STEP cylinder and therefore still merges coaxial
    # segments of stepped holes.
    axis = cylinder.axis.copy()
    axis.normalize()
    origin_vector = adsk.core.Vector3D.create(
        cylinder.origin.x, cylinder.origin.y, cylinder.origin.z
    )
    axial = axis.copy()
    axial.scaleBy(origin_vector.dotProduct(axis))
    origin_vector.subtract(axial)
    return (
        axis_index,
        int(round(origin_vector.x * LINE_POSITION_PRECISION_PER_CM)),
        int(round(origin_vector.y * LINE_POSITION_PRECISION_PER_CM)),
        int(round(origin_vector.z * LINE_POSITION_PRECISION_PER_CM)),
    )


def _directions_parallel(
    first: adsk.core.Vector3D,
    second: adsk.core.Vector3D,
    same_direction: bool = False,
) -> bool:
    first_length = first.length
    second_length = second.length
    if (
        first_length < geometry.DIRECTION_LENGTH_EPSILON
        or second_length < geometry.DIRECTION_LENGTH_EPSILON
    ):
        return False
    cosine = first.dotProduct(second) / (first_length * second_length)
    if same_direction:
        return cosine >= DRILL_AXIS_ALIGNMENT_COS_THRESHOLD
    return abs(cosine) >= DRILL_AXIS_ALIGNMENT_COS_THRESHOLD


def has_opposite_side_holes(
    body: adsk.fusion.BRepBody,
    top: adsk.core.Vector3D,
    max_drill_diameter_mm: float = float('inf'),
) -> bool:
    """True if blind holes exist that open only from the opposite side of the
    body (i.e. the part would need to be flipped to drill them). Through
    holes are excluded: find_hole_faces already drills them in one pass."""
    bottom = top.copy()
    bottom.scaleBy(-1.0)
    bottom_key = geometry.axis_key(bottom)
    if not bottom_key:
        return False
    for face in body.faces:
        cylinder = adsk.core.Cylinder.cast(face.geometry)
        if not cylinder or not geometry.is_internal_cylinder(face, cylinder):
            continue
        if cylinder.radius * 2.0 * 10.0 > max_drill_diameter_mm + 1e-6:
            # Oversized circular features are assigned to pocket milling, so
            # they must not also drive the opposite-side drilling warning.
            continue
        cylinder_key = geometry.axis_key(cylinder.axis)
        if not cylinder_key or cylinder_key[0] != bottom_key[0]:
            continue
        if not _directions_parallel(cylinder.axis, bottom):
            continue
        openings = geometry.cylinder_opening_directions(face)
        if len(openings) == 1 and _directions_parallel(
            openings[0], bottom, same_direction=True
        ):
            return True
    return False


def create_drill_operations(
    setup: adsk.cam.Setup,
    holes: Sequence[HoleFace],
    tools: Sequence[adsk.cam.Tool],
    tolerance_mm: float,
) -> Tuple[List[adsk.cam.OperationBase], List[str]]:
    grouped: Dict[Tuple[int, bool], List[HoleFace]] = {}
    for hole in holes:
        diameter_microns = int(round(hole.diameter_cm * 10.0 * 1000.0))
        grouped.setdefault((diameter_microns, hole.is_through), []).append(hole)

    operations: List[adsk.cam.OperationBase] = []
    warnings: List[str] = []
    for (diameter_microns, is_through), group in sorted(grouped.items()):
        diameter_mm = diameter_microns / 1000.0
        tool = tool_library.nearest_drill(tools, diameter_mm, tolerance_mm)
        if not tool:
            warnings.append(
                f'No drill was found within ±{tolerance_mm:g} mm tolerance for '
                f'{len(group)} hole(s) with Ø{diameter_mm:g} mm diameter.'
            )
            continue

        operation_input = setup.operations.createInput('drill')
        suffix = 'through' if is_through else 'blind'
        operation_input.displayName = f'AM – Drilling Ø{diameter_mm:g} mm ({suffix})'
        operation_input.tool = tool
        set_safe_heights(operation_input.parameters)
        set_expression(
            operation_input.parameters,
            'drillTipThroughBottom',
            'true' if is_through else 'false',
        )
        if is_through:
            set_expression(operation_input.parameters, 'breakThroughDepth', '1 mm')
        selection = operation_input.parameters.itemByName('holeFaces').value
        selection.value = [hole.face for hole in group]
        operations.append(setup.operations.add(operation_input))

    return operations, warnings
