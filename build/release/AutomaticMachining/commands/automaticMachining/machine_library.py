"""Discovers CAM machines and resolves the post processor bound to one.

Machines are addressed the same way as everything else in the Machine
Library dialog shown in Fusion (folders of machine assets across Cloud/Hub/
Local/Fusion locations). Each ``Machine`` can carry its own post processor
(``hasPost`` / ``postURL``), which is exactly what lets a user pick "Homag
Wood Wop" from a dropdown and have its bound woodWOP post used automatically,
instead of having to hunt for the .cps file separately in the Post Library.

Discovery recurses every location's folder tree, which is slow and
network-bound for Cloud/Hub locations. The result is cached at module level
for the lifetime of the add-in session (cleared on stop()) so opening the
command dialog repeatedly doesn't repeat that walk every time.
"""

from __future__ import annotations

import traceback
from typing import Dict, List, Optional

import adsk.cam
import adsk.core

from ...lib import fusionAddInUtils as futil

_machine_cache: Optional[List[adsk.cam.Machine]] = None


def clear_cache() -> None:
    global _machine_cache
    _machine_cache = None


def discover_machines() -> List[adsk.cam.Machine]:
    global _machine_cache
    if _machine_cache is None:
        _machine_cache = _discover_machines()
    return _machine_cache


def _discover_machines() -> List[adsk.cam.Machine]:
    """Flat list of every machine across all machine-library locations,
    recursing into folders (mirrors the Machine Library dialog's tree)."""
    library = adsk.cam.CAMManager.get().libraryManager.machineLibrary
    locations = (
        adsk.cam.LibraryLocations.CloudLibraryLocation,
        adsk.cam.LibraryLocations.HubLibraryLocation,
        adsk.cam.LibraryLocations.LocalLibraryLocation,
        adsk.cam.LibraryLocations.Fusion360LibraryLocation,
    )
    machines: List[adsk.cam.Machine] = []
    for location in locations:
        try:
            root_url = library.urlByLocation(location)
            if not root_url:
                continue
            machines.extend(_machines_in_folder(library, root_url))
        except Exception:
            futil.log(
                f'Unable to load machines for location {location}:\n{traceback.format_exc()}',
                adsk.core.LogLevels.WarningLogLevel,
            )
    return machines


def _machines_in_folder(
    machine_library: adsk.cam.MachineLibrary,
    folder_url: adsk.core.URL,
) -> List[adsk.cam.Machine]:
    result = list(machine_library.childMachines(folder_url) or [])
    for sub_url in machine_library.childFolderURLs(folder_url) or []:
        result.extend(_machines_in_folder(machine_library, sub_url))
    return result


def machine_label(machine: adsk.cam.Machine) -> str:
    name = ' '.join(part for part in (machine.vendor, machine.model) if part)
    name = name or 'Machine'
    return f'{name} — {machine.description}' if machine.description else name


def populate_machine_dropdown(dropdown: adsk.core.DropDownCommandInput) -> Dict[str, int]:
    """Fills the dropdown from discover_machines() and returns a label ->
    index mapping. The list is rebuilt in the same order at batch-run time
    (see machine_by_index), so persisting the chosen index is enough."""
    dropdown.listItems.clear()
    machines = discover_machines()
    index_by_label: Dict[str, int] = {}
    for index, machine in enumerate(machines):
        base_label = machine_label(machine)
        label = base_label
        suffix = 2
        while label in index_by_label:
            label = f'{base_label} ({suffix})'
            suffix += 1
        index_by_label[label] = index
        dropdown.listItems.add(label, index == 0, '')
    return index_by_label


def machine_by_index(index: int) -> adsk.cam.Machine:
    machines = discover_machines()
    if index < 0 or index >= len(machines):
        raise RuntimeError('The selected machine is no longer available in the library.')
    return machines[index]


def resolve_post_configuration(machine: adsk.cam.Machine) -> adsk.cam.PostConfiguration:
    if not machine.hasPost:
        raise RuntimeError(
            f'The machine "{machine_label(machine)}" has no post processor assigned '
            'in the Machine Library.'
        )
    post_library = adsk.cam.CAMManager.get().libraryManager.postLibrary
    post_configuration = post_library.postConfigurationAtURL(machine.postURL)
    if not post_configuration:
        raise RuntimeError(
            f'Unable to load the post processor assigned to "{machine_label(machine)}".'
        )
    return post_configuration
