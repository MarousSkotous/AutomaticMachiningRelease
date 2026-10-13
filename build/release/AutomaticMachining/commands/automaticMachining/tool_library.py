"""Discovers CAM tool libraries and resolves tools used by the batch.

Discovery walks every tool-library location (Cloud/Hub/Local/Fusion),
recursing into folders — for the built-in Fusion stock libraries and cloud
locations this is a slow, network-bound tree walk. Results are cached at
module level for the lifetime of the add-in session (cleared on stop()) so
opening the command dialog repeatedly doesn't repeat that walk every time.
"""

from __future__ import annotations

import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import adsk.cam
import adsk.core

from ...lib import fusionAddInUtils as futil

# Tool types offered for the outer contour / pocket dropdowns. Anything else
# in the library (drills, chamfer mills, etc.) is not a viable milling tool
# for these strategies.
MILLING_TOOL_TYPES = {'flat end mill', 'bull nose end mill'}

# Tool type offered for the saw-blade slot dropdown (see pockets.SawSlotFeature).
# A "slot mill" is Fusion's disc-shaped cutter type, the closest match to a
# real saw blade tilted 90 degrees.
SAW_TOOL_TYPES = {'slot mill'}

# The lower-edge chamfer is cut from the top setup with an undercut/dovetail
# cutter.  It deliberately has its own dropdown: a dovetail mill is not safe
# for ordinary pockets or the through outer contour.
BOTTOM_CHAMFER_TOOL_TYPES = {'dovetail mill'}
_BOTTOM_CHAMFER_DIAMETER_MM = 20.0
_BOTTOM_CHAMFER_TAPER_DEG = 30.0

# Duplicated from pockets.SAW_SLOT_WIDTH_MM (importing it here would create a
# circular import: pockets already imports this module). Only used to pick a
# sensible default dropdown selection, not to validate the chosen tool.
_SAW_SLOT_WIDTH_MM = 3.5

# Fusion cannot safely calculate a pocket/contour toolpath for a milling tool
# whose cutter definition has no cutting length.  A few imported library tools
# have a valid diameter but tool_fluteLength=0, which otherwise makes them look
# selectable and later produces an empty toolpath.
MIN_FLUTE_LENGTH_CM = 1e-6

# A tool is considered a drill candidate when its tool_type is this.
DRILL_TOOL_TYPE = 'drill'

_library_cache: Optional[Dict[str, str]] = None
_tool_library_cache: Dict[str, adsk.cam.ToolLibrary] = {}


def clear_cache() -> None:
    global _library_cache, _tool_library_cache
    _library_cache = None
    _tool_library_cache = {}


def discover_tool_libraries() -> Dict[str, str]:
    global _library_cache
    if _library_cache is None:
        _library_cache = _discover_tool_libraries()
    return _library_cache


def _discover_tool_libraries() -> Dict[str, str]:
    result: Dict[str, str] = {}
    libraries = adsk.cam.CAMManager.get().libraryManager.toolLibraries
    locations = (
        ('Cloud', adsk.cam.LibraryLocations.CloudLibraryLocation),
        ('Hub', adsk.cam.LibraryLocations.HubLibraryLocation),
        ('Local', adsk.cam.LibraryLocations.LocalLibraryLocation),
        ('Fusion', adsk.cam.LibraryLocations.Fusion360LibraryLocation),
    )
    for location_name, location in locations:
        try:
            root_url = libraries.urlByLocation(location)
            if not root_url:
                continue
            for url_string in _library_urls(libraries, root_url):
                url = adsk.core.URL.create(url_string)
                base_label = f'{location_name} / {url.leafName}'
                label = base_label
                suffix = 2
                while label in result:
                    label = f'{base_label} ({suffix})'
                    suffix += 1
                result[label] = url_string
        except Exception:
            futil.log(
                f'Unable to load {location_name} tool libraries:\n{traceback.format_exc()}',
                adsk.core.LogLevels.WarningLogLevel,
            )
    return dict(sorted(result.items(), key=lambda item: item[0].lower()))


def _library_urls(
    libraries: adsk.cam.ToolLibraries,
    root_url: adsk.core.URL,
) -> List[str]:
    result = [url.toString() for url in libraries.childAssetURLs(root_url)]
    for folder_url in libraries.childFolderURLs(root_url):
        result.extend(_library_urls(libraries, folder_url))
    return result


def load_tool_library(url_string: str) -> adsk.cam.ToolLibrary:
    cached = _tool_library_cache.get(url_string)
    if cached is not None:
        return cached
    libraries = adsk.cam.CAMManager.get().libraryManager.toolLibraries
    library = libraries.toolLibraryAtURL(adsk.core.URL.create(url_string))
    if not library:
        raise RuntimeError('Unable to load the selected tool library.')
    _tool_library_cache[url_string] = library
    return library


def populate_milling_tools(
    url_string: str,
    small_contour_dropdown: adsk.core.DropDownCommandInput,
    large_contour_dropdown: adsk.core.DropDownCommandInput,
    small_pocket_dropdown: adsk.core.DropDownCommandInput,
    large_pocket_dropdown: adsk.core.DropDownCommandInput,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Fills the four tool dropdowns from the library at url_string and
    returns (index-by-label, diameter_cm-by-label) for the offered tools."""
    milling_tool_by_label: Dict[str, int] = {}
    milling_tool_diameter_by_label: Dict[str, float] = {}
    dropdowns = (
        small_contour_dropdown,
        large_contour_dropdown,
        small_pocket_dropdown,
        large_pocket_dropdown,
    )
    for dropdown in dropdowns:
        dropdown.listItems.clear()
    if not url_string:
        return milling_tool_by_label, milling_tool_diameter_by_label
    try:
        tools = [tool for tool in load_tool_library(url_string)]
        choices: List[Tuple[str, int, float]] = []
        for index, tool in enumerate(tools):
            if tool_type(tool) not in MILLING_TOOL_TYPES:
                continue
            if tool_flute_length_cm(tool) <= MIN_FLUTE_LENGTH_CM:
                continue
            base_label = tool_label(tool)
            label = base_label
            suffix = 2
            while label in milling_tool_by_label:
                label = f'{base_label} ({suffix})'
                suffix += 1
            diameter_cm = tool_diameter_cm(tool)
            milling_tool_by_label[label] = index
            milling_tool_diameter_by_label[label] = diameter_cm
            choices.append((label, index, diameter_cm))

        if not choices:
            return milling_tool_by_label, milling_tool_diameter_by_label
        small_choice = min(range(len(choices)), key=lambda item: choices[item][2])
        large_choice = max(range(len(choices)), key=lambda item: choices[item][2])
        for dropdown, selected_choice in zip(
            dropdowns, (small_choice, large_choice, small_choice, large_choice)
        ):
            for choice_index, (label, _, _) in enumerate(choices):
                dropdown.listItems.add(label, choice_index == selected_choice, '')
    except Exception:
        futil.log(traceback.format_exc(), adsk.core.LogLevels.ErrorLogLevel)
    return milling_tool_by_label, milling_tool_diameter_by_label


def populate_saw_tools(
    url_string: str,
    saw_dropdown: adsk.core.DropDownCommandInput,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Fills the saw-blade slot tool dropdown from the library at url_string
    and returns (index-by-label, diameter_cm-by-label) for the offered tools.
    Separate from populate_milling_tools because a 'slot mill' is not a
    viable outer-contour/pocket tool and should not appear in those
    dropdowns."""
    saw_tool_by_label: Dict[str, int] = {}
    saw_tool_diameter_by_label: Dict[str, float] = {}
    saw_dropdown.listItems.clear()
    if not url_string:
        return saw_tool_by_label, saw_tool_diameter_by_label
    try:
        tools = [tool for tool in load_tool_library(url_string)]
        choices: List[Tuple[str, int, float]] = []
        for index, tool in enumerate(tools):
            if tool_type(tool) not in SAW_TOOL_TYPES:
                continue
            if tool_flute_length_cm(tool) <= MIN_FLUTE_LENGTH_CM:
                continue
            base_label = tool_label(tool)
            label = base_label
            suffix = 2
            while label in saw_tool_by_label:
                label = f'{base_label} ({suffix})'
                suffix += 1
            diameter_cm = tool_diameter_cm(tool)
            saw_tool_by_label[label] = index
            saw_tool_diameter_by_label[label] = diameter_cm
            choices.append((label, index, diameter_cm))

        if not choices:
            return saw_tool_by_label, saw_tool_diameter_by_label
        # Default to whichever offered tool sits closest to the nominal saw
        # slot width instead of always the first one in the library.
        default_choice = min(
            range(len(choices)),
            key=lambda item: abs(choices[item][2] * 10.0 - _SAW_SLOT_WIDTH_MM),
        )
        for choice_index, (label, _, _) in enumerate(choices):
            saw_dropdown.listItems.add(label, choice_index == default_choice, '')
    except Exception:
        futil.log(traceback.format_exc(), adsk.core.LogLevels.ErrorLogLevel)
    return saw_tool_by_label, saw_tool_diameter_by_label


def populate_bottom_chamfer_tools(
    url_string: str,
    dropdown: adsk.core.DropDownCommandInput,
) -> Dict[str, int]:
    """Offers dovetail cutters and defaults to the requested Ø20/30° tool."""
    tool_by_label: Dict[str, int] = {}
    dropdown.listItems.clear()
    if not url_string:
        return tool_by_label
    try:
        tools = [tool for tool in load_tool_library(url_string)]
        choices: List[Tuple[str, int, float, float]] = []
        for index, tool in enumerate(tools):
            if tool_type(tool) not in BOTTOM_CHAMFER_TOOL_TYPES:
                continue
            base_label = tool_label(tool)
            label = base_label
            suffix = 2
            while label in tool_by_label:
                label = f'{base_label} ({suffix})'
                suffix += 1
            diameter_mm = tool_diameter_cm(tool) * 10.0
            taper_deg = tool_taper_angle_deg(tool)
            tool_by_label[label] = index
            choices.append((label, index, diameter_mm, taper_deg))

        if not choices:
            return tool_by_label
        default_choice = min(
            range(len(choices)),
            key=lambda item: (
                abs(choices[item][2] - _BOTTOM_CHAMFER_DIAMETER_MM)
                + abs(choices[item][3] - _BOTTOM_CHAMFER_TAPER_DEG)
            ),
        )
        for choice_index, (label, _, _, _) in enumerate(choices):
            dropdown.listItems.add(label, choice_index == default_choice, '')
    except Exception:
        futil.log(traceback.format_exc(), adsk.core.LogLevels.ErrorLogLevel)
    return tool_by_label


def tool_by_index(tools: Sequence[adsk.cam.Tool], index: int, purpose: str) -> adsk.cam.Tool:
    if index < 0 or index >= len(tools):
        raise RuntimeError(
            f'The selected {purpose} tool is no longer available in the library.'
        )
    return tools[index]


def nearest_drill(
    tools: Sequence[adsk.cam.Tool],
    diameter_mm: float,
    tolerance_mm: float,
) -> Optional[adsk.cam.Tool]:
    candidates: List[Tuple[float, adsk.cam.Tool]] = []
    for tool in tools:
        if tool_type(tool) != DRILL_TOOL_TYPE:
            continue
        tool_diameter_mm = tool_diameter_cm(tool) * 10.0
        difference = abs(tool_diameter_mm - diameter_mm)
        if difference <= tolerance_mm + 1e-9:
            candidates.append((difference, tool))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def tool_type(tool: adsk.cam.Tool) -> str:
    parameter = tool.parameters.itemByName('tool_type')
    return parameter.value.value if parameter else ''


def tool_diameter_cm(tool: adsk.cam.Tool) -> float:
    parameter = tool.parameters.itemByName('tool_diameter')
    return float(parameter.value.value) if parameter else 0.0


def tool_corner_radius_cm(tool: adsk.cam.Tool) -> float:
    """Returns the tool's corner (bull-nose) radius, 0 for a flat end mill."""
    parameter = tool.parameters.itemByName('tool_cornerRadius')
    return float(parameter.value.value) if parameter else 0.0


def tool_taper_angle_deg(tool: adsk.cam.Tool) -> float:
    """Returns Fusion's dovetail/chamfer taper angle in degrees."""
    parameter = tool.parameters.itemByName('tool_taperAngle')
    return float(parameter.value.value) if parameter else 0.0


def tool_flute_length_cm(tool: adsk.cam.Tool) -> float:
    """Returns usable cutter/flute length in Fusion's internal centimetres."""
    parameter = tool.parameters.itemByName('tool_fluteLength')
    return float(parameter.value.value) if parameter else 0.0


def tool_label(tool: adsk.cam.Tool) -> str:
    description_parameter = tool.parameters.itemByName('tool_description')
    description = description_parameter.value.value if description_parameter else 'Tool'
    number_parameter = tool.parameters.itemByName('tool_number')
    number = number_parameter.value.value if number_parameter else 0
    diameter_mm = tool_diameter_cm(tool) * 10.0
    return f'T{number} – Ø{diameter_mm:g} mm – {description}'
