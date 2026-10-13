"""Batch CAM creation for Fusion designs in a cloud or local source folder.

The implementation deliberately avoids Manufacturing Extension-only APIs.
Hole recognition is limited to axis-aligned cylindrical BRep faces, while
Fusion's basic pocket recognition is used for flat-bottom pockets. Source
files are never modified: each design is copied to a new cloud folder first.
An optional local source (including Google Drive for desktop) is imported by
Fusion locally, saved into a same-named Fusion folder, and then handed to the
same copy-based batch.  Importing locally avoids the much slower cloud-side
STEP-to-F3D translation performed by ``DataFolder.uploadFile``.

The recognition and tool-lookup logic lives in sibling modules:
``geometry`` (shared vector/face helpers), ``orientation`` (top/X/Y
detection), ``holes`` and ``pockets`` (feature recognition + operation
creation), ``tool_library`` (CAM tool library discovery and lookup), and
``machine_library`` (CAM machine discovery and resolving the post processor
bound to a machine). This module owns the command UI, batch orchestration
and CAM setup.

The selected machine is deliberately never assigned to a Setup or NCProgram
(see the comments in ``_create_setup``/``_create_nc_program``): Fusion
crashes inside ``Setups.add()`` while evaluating ``Machine.isSimulationReady()``
for machines that lack a full simulation model (e.g. post-only machine
definitions). Only the machine's bound post processor is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import os
import re
import threading
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import adsk.cam
import adsk.core
import adsk.fusion

from ... import config
from ...lib import fusionAddInUtils as futil
from . import (
    holes,
    local_source,
    machine_library,
    orientation,
    pdf_report,
    pockets,
    presets,
    tool_library,
)
from .cam_utils import set_expression, set_safe_heights, set_zero_stock_to_leave


app = adsk.core.Application.get()
ui = app.userInterface

CMD_ID = f'{config.COMPANY_NAME}_{config.ADDIN_NAME}_automaticMachining'
CMD_NAME = 'Automatic Machining'
CMD_DESCRIPTION = 'Creates CAM copies of Fusion models from the active cloud folder.'
CUSTOM_EVENT_ID = f'{CMD_ID}_runBatch'

WORKSPACE_ID = 'CAMEnvironment'
TAB_ID = 'MillingTab'
PANEL_ID = 'AutomaticMachiningPanel'
PANEL_NAME = 'AUTOMATIC MACHINING'
PANEL_BESIDE_ID = 'CAMActionPanel'
LEGACY_CAM_PANEL_ID = 'CAMToolpathPanel'
LEGACY_WORKSPACE_ID = 'FusionSolidEnvironment'
LEGACY_PANEL_ID = 'SolidScriptsAddinsPanel'
ICON_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'resources',
    '',
)

SUPPORTED_EXTENSIONS = {'f3d'}
SOURCE_MODE_FUSION = 'Active Fusion folder'
SOURCE_MODE_LOCAL = 'Local folder (Google Drive)'
PRESET_EMPTY_LABEL = '(no saved presets)'
ORIENTATION_COMPONENT_NAME = '_AM_WCS_ORIENTATION'
SETUP_NAME = 'Automatic Machining'

DEFAULT_NC_OUTPUT_FOLDER = os.path.join(os.path.expanduser('~'), 'Documents', 'AutomaticMachining_NC')

# How long to wait for a cloud copy of a source file to finish uploading
# before giving up on that file.
CLOUD_COPY_TIMEOUT_S = 120.0
CLOUD_COPY_POLL_INTERVAL_S = 0.1

# A locally imported document is already a native Fusion design.  Saving it to
# the cloud still finishes asynchronously, but it does not require the slow
# server-side STEP-to-F3D translation used by DataFolder.uploadFile.
LOCAL_SAVE_TIMEOUT_S = 180.0
LOCAL_SAVE_POLL_INTERVAL_S = 0.5
# Fusion occasionally finishes the WIP/PIM publish without delivering the
# matching dataFileComplete callback to the add-in.  Do not consult the cloud
# folder immediately: a newly created document can report a transient
# isComplete value before the publish pipeline is idle.  After a conservative
# delay, require the same freshly enumerated cloud item to remain complete for
# several seconds before accepting it as a fallback completion signal.
LOCAL_SAVE_FALLBACK_START_S = 12.0
LOCAL_SAVE_FOLDER_CHECK_INTERVAL_S = 2.0
LOCAL_SAVE_FALLBACK_STABLE_S = 4.0
# Keep the document open briefly after Application.dataFileComplete. Fusion's
# WIP upload can finish just before its PIM publish bookkeeping becomes idle;
# immediately closing and importing another document has caused native crashes.
LOCAL_SAVE_SETTLE_S = 2.0

# How long to wait for toolpath generation of a single document's operations
# before giving up on that file.
TOOLPATH_GENERATION_TIMEOUT_S = 300.0

# Document.activate() has been observed to return without error while
# Document.isActive still reads False immediately afterwards — a single
# doEvents() is not always enough for Fusion to finish the switch. Poll
# briefly instead of giving up on the first check: this only affects whether
# a processed copy gets closed for tidiness (it is already saved by this
# point), never whether it closes when unsafe to do so.
RETURN_DOCUMENT_ACTIVATE_TIMEOUT_S = 2.0
RETURN_DOCUMENT_ACTIVATE_POLL_INTERVAL_S = 0.1

# Maximum number of warning/failure lines shown in the summary message box.
# The full, untruncated list is always written to the Fusion log file.
MAX_REPORT_ITEMS = 12

local_handlers: List[object] = []
_custom_event = None
_pending_config: Optional['BatchConfig'] = None
_library_by_label: Dict[str, str] = {}
_milling_tool_by_label: Dict[str, int] = {}
_milling_tool_diameter_by_label: Dict[str, float] = {}
_saw_tool_by_label: Dict[str, int] = {}
_saw_tool_diameter_by_label: Dict[str, float] = {}
_bottom_chamfer_tool_by_label: Dict[str, int] = {}
_current_milling_library_url = ''
_machine_index_by_label: Dict[str, int] = {}
_presets_by_name: Dict[str, dict] = {}
_source_file_count = 0
_local_import_session: Optional['LocalImportSession'] = None
_local_import_timer: Optional[threading.Timer] = None


@dataclass(frozen=True)
class BatchConfig:
    source_folder_id: str
    source_folder_name: str
    destination_name: str
    library_url: str
    small_contour_tool_index: int
    large_contour_tool_index: int
    small_pocket_tool_index: int
    large_pocket_tool_index: int
    saw_tool_index: int
    bottom_chamfer_tool_index: int
    machine_index: int
    create_contour: bool
    create_pockets: bool
    create_saw_slots: bool
    create_bottom_chamfer: bool
    create_drilling: bool
    generate_toolpaths: bool
    create_nc_code: bool
    drill_tolerance_mm: float
    max_drill_diameter_mm: float
    contour_stepdown_mm: float
    pocket_stepdown_mm: float
    nc_output_folder: str
    local_source_folder: str = ''


@dataclass
class LocalImportSession:
    batch: BatchConfig
    cloud_folder: adsk.core.DataFolder
    local_files: List[str]
    progress: adsk.core.ProgressDialog
    saved_files: List[adsk.core.DataFile]
    original_document: Optional[adsk.core.Document]
    next_index: int = 0
    current_document: Optional[adsk.core.Document] = None
    current_data_file_id: str = ''
    completed_data_file: Optional[adsk.core.DataFile] = None
    completed_at: float = 0.0
    current_started_at: float = 0.0
    fallback_data_file_id: str = ''
    fallback_seen_at: float = 0.0
    last_folder_check_at: float = 0.0
    last_fallback_error: str = ''


def start() -> None:
    global _custom_event

    _delete_toolbar_control(LEGACY_WORKSPACE_ID, LEGACY_PANEL_ID)
    _delete_toolbar_control(WORKSPACE_ID, LEGACY_CAM_PANEL_ID, TAB_ID)
    panel = _ensure_toolbar_panel()
    if not panel:
        raise RuntimeError('Unable to create the Automatic Machining panel on the Milling tab.')
    old_control = panel.controls.itemById(CMD_ID) if panel else None
    old_def = ui.commandDefinitions.itemById(CMD_ID)
    if old_control:
        old_control.deleteMe()
    if old_def:
        old_def.deleteMe()

    cmd_def = ui.commandDefinitions.addButtonDefinition(
        CMD_ID, CMD_NAME, CMD_DESCRIPTION, ICON_FOLDER
    )
    futil.add_handler(cmd_def.commandCreated, command_created)

    if panel:
        control = panel.controls.addCommand(cmd_def)
        control.isPromoted = True

    try:
        app.unregisterCustomEvent(CUSTOM_EVENT_ID)
    except Exception:
        pass
    _custom_event = app.registerCustomEvent(CUSTOM_EVENT_ID)
    futil.add_handler(_custom_event, batch_event)
    # Unlike a cached DataFile.isComplete value, this application event is
    # raised when Fusion has finished the upload and all cloud-side processing.
    futil.add_handler(app.dataFileComplete, data_file_complete)


def stop() -> None:
    global _custom_event, _pending_config, _local_import_session

    _cancel_local_import_timer()
    if _local_import_session:
        _close_local_import_document(_local_import_session)
        _restore_local_import_document(_local_import_session)
        try:
            _local_import_session.progress.hide()
        except Exception:
            pass
        _local_import_session = None
    _delete_toolbar_control(WORKSPACE_ID, PANEL_ID, TAB_ID)
    _delete_toolbar_control(WORKSPACE_ID, LEGACY_CAM_PANEL_ID, TAB_ID)
    _delete_toolbar_control(LEGACY_WORKSPACE_ID, LEGACY_PANEL_ID)
    panel = _toolbar_panel(WORKSPACE_ID, PANEL_ID, TAB_ID)
    if panel:
        panel.deleteMe()
    cmd_def = ui.commandDefinitions.itemById(CMD_ID)
    if cmd_def:
        cmd_def.deleteMe()
    try:
        app.unregisterCustomEvent(CUSTOM_EVENT_ID)
    except Exception:
        pass
    _custom_event = None
    _pending_config = None
    tool_library.clear_cache()
    machine_library.clear_cache()


def _toolbar_panel(workspace_id: str, panel_id: str, tab_id: str = ''):
    workspace = ui.workspaces.itemById(workspace_id)
    if not workspace:
        return None
    if tab_id:
        tab = workspace.toolbarTabs.itemById(tab_id)
        if tab:
            panel = tab.toolbarPanels.itemById(panel_id)
            if panel:
                return panel
    return workspace.toolbarPanels.itemById(panel_id)


def _ensure_toolbar_panel():
    panel = _toolbar_panel(WORKSPACE_ID, PANEL_ID, TAB_ID)
    if panel:
        return panel
    workspace = ui.workspaces.itemById(WORKSPACE_ID)
    tab = workspace.toolbarTabs.itemById(TAB_ID) if workspace else None
    if not tab:
        return None
    return tab.toolbarPanels.add(PANEL_ID, PANEL_NAME, PANEL_BESIDE_ID, False)


def _delete_toolbar_control(
    workspace_id: str, panel_id: str, tab_id: str = ''
) -> None:
    panel = _toolbar_panel(workspace_id, panel_id, tab_id)
    control = panel.controls.itemById(CMD_ID) if panel else None
    if control:
        control.deleteMe()


def command_created(args: adsk.core.CommandCreatedEventArgs) -> None:
    global _library_by_label, _milling_tool_by_label
    global _milling_tool_diameter_by_label, _current_milling_library_url
    global _saw_tool_by_label, _saw_tool_diameter_by_label
    global _bottom_chamfer_tool_by_label
    global _machine_index_by_label, _pending_config
    global _presets_by_name, _source_file_count
    _library_by_label = {}
    _milling_tool_by_label = {}
    _milling_tool_diameter_by_label = {}
    _saw_tool_by_label = {}
    _saw_tool_diameter_by_label = {}
    _bottom_chamfer_tool_by_label = {}
    _current_milling_library_url = ''
    _machine_index_by_label = {}
    try:
        _presets_by_name = presets.load_presets()
    except Exception:
        _presets_by_name = {}
        futil.log(
            'Unable to load Automatic Machining presets:\n' + traceback.format_exc(),
            adsk.core.LogLevels.WarningLogLevel,
        )
    _pending_config = None

    command = args.command
    command.setDialogInitialSize(620, 900)
    command.setDialogMinimumSize(620, 400)
    inputs = command.commandInputs

    preset_input = inputs.addDropDownCommandInput(
        'preset_select',
        'Preset',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )
    _populate_preset_dropdown(preset_input)
    selected_preset_name = (
        preset_input.selectedItem.name
        if preset_input.selectedItem and preset_input.selectedItem.name != PRESET_EMPTY_LABEL
        else ''
    )
    inputs.addStringValueInput('preset_name', 'Preset name', selected_preset_name)
    inputs.addBoolValueInput('preset_load', 'Load preset', False, '', False)
    inputs.addBoolValueInput('preset_save', 'Save preset', False, '', False)
    inputs.addTextBoxCommandInput('preset_status', '', '', 1, True)

    active_folder = app.data.activeFolder
    folder_name = active_folder.name if active_folder else '(not available)'
    # Cached once here and reused by command_validate_inputs, which Fusion
    # calls on almost every dialog interaction — recomputing this from the
    # (cloud-backed) file list on every call made the whole dialog laggy.
    _source_file_count = _count_supported_files(active_folder) if active_folder else 0
    file_count = _source_file_count

    source_mode = inputs.addDropDownCommandInput(
        'source_mode',
        'Source type',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )
    source_mode.listItems.add(SOURCE_MODE_FUSION, True, '')
    source_mode.listItems.add(SOURCE_MODE_LOCAL, False, '')

    inputs.addTextBoxCommandInput(
        'source_info',
        'Source',
        f'<b>{_html(folder_name)}</b><br/>Fusion models: {file_count}',
        3,
        True,
    )
    local_folder_input = inputs.addStringValueInput('local_source_folder', 'Local folder', '')
    local_folder_input.isVisible = False
    browse_input = inputs.addBoolValueInput('browse_local_source', 'Browse', False, '', False)
    browse_input.isVisible = False

    default_destination = datetime.now().strftime('AutomaticMachining_%Y%m%d_%H%M%S')
    inputs.addStringValueInput('destination_name', 'Folder', default_destination)

    library_input = inputs.addDropDownCommandInput(
        'tool_library',
        'Tools',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )
    _library_by_label = tool_library.discover_tool_libraries()
    for index, label in enumerate(_library_by_label):
        library_input.listItems.add(label, index == 0, '')

    machine_input = inputs.addDropDownCommandInput(
        'machine',
        'Machine',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )
    _machine_index_by_label = machine_library.populate_machine_dropdown(machine_input)

    inputs.addBoolValueInput('create_contour', 'Contour', True, '', True)
    small_contour_input = inputs.addDropDownCommandInput(
        'small_contour_tool',
        'Small',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )
    large_contour_input = inputs.addDropDownCommandInput(
        'large_contour_tool',
        'Large',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )
    stepdown = adsk.core.ValueInput.createByString('5 mm')
    inputs.addValueInput('contour_stepdown', 'Step', 'mm', stepdown)

    inputs.addBoolValueInput('create_pockets', 'Pockets', True, '', True)
    small_pocket_input = inputs.addDropDownCommandInput(
        'small_pocket_tool',
        'Small',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )
    large_pocket_input = inputs.addDropDownCommandInput(
        'large_pocket_tool',
        'Large',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )
    pocket_stepdown = adsk.core.ValueInput.createByString('5 mm')
    inputs.addValueInput('pocket_stepdown', 'Step', 'mm', pocket_stepdown)

    inputs.addBoolValueInput('create_saw_slots', 'Saw slots', True, '', True)
    saw_tool_input = inputs.addDropDownCommandInput(
        'saw_tool',
        'Saw',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )

    inputs.addBoolValueInput('create_bottom_chamfer', 'Bottom chamfer', True, '', True)
    bottom_chamfer_tool_input = inputs.addDropDownCommandInput(
        'bottom_chamfer_tool',
        'Chamfer tool',
        adsk.core.DropDownStyles.TextListDropDownStyle,
    )

    if library_input.selectedItem:
        _current_milling_library_url = _library_by_label.get(
            library_input.selectedItem.name, ''
        )
        _milling_tool_by_label, _milling_tool_diameter_by_label = tool_library.populate_milling_tools(
            _current_milling_library_url,
            small_contour_input,
            large_contour_input,
            small_pocket_input,
            large_pocket_input,
        )
        _saw_tool_by_label, _saw_tool_diameter_by_label = tool_library.populate_saw_tools(
            _current_milling_library_url,
            saw_tool_input,
        )
        _bottom_chamfer_tool_by_label = tool_library.populate_bottom_chamfer_tools(
            _current_milling_library_url,
            bottom_chamfer_tool_input,
        )

    inputs.addBoolValueInput('create_drilling', 'Drilling', True, '', True)
    max_drill_diameter = adsk.core.ValueInput.createByString('12 mm')
    inputs.addValueInput(
        'max_drill_diameter',
        'Max. drill Ø',
        'mm',
        max_drill_diameter,
    )
    tolerance = adsk.core.ValueInput.createByString('0.5 mm')
    inputs.addValueInput('drill_tolerance', 'Tol.', 'mm', tolerance)
    inputs.addBoolValueInput(
        'generate_toolpaths',
        'Paths',
        True,
        '',
        True,
    )
    inputs.addBoolValueInput(
        'create_nc_code',
        'NC code',
        True,
        '',
        True,
    )
    inputs.addStringValueInput('nc_output_folder', 'Reports', DEFAULT_NC_OUTPUT_FOLDER)

    futil.add_handler(command.execute, command_execute, local_handlers=local_handlers)
    futil.add_handler(command.inputChanged, command_input_changed, local_handlers=local_handlers)
    futil.add_handler(command.validateInputs, command_validate_inputs, local_handlers=local_handlers)
    futil.add_handler(command.destroy, command_destroy, local_handlers=local_handlers)


def command_input_changed(args: adsk.core.InputChangedEventArgs) -> None:
    global _milling_tool_by_label, _milling_tool_diameter_by_label
    global _saw_tool_by_label, _saw_tool_diameter_by_label
    global _bottom_chamfer_tool_by_label
    global _current_milling_library_url, _presets_by_name

    if args.input.id == 'preset_select':
        selected = args.inputs.itemById('preset_select').selectedItem
        if selected and selected.name != PRESET_EMPTY_LABEL:
            args.inputs.itemById('preset_name').value = selected.name
        return

    if args.input.id == 'preset_save':
        args.inputs.itemById('preset_save').value = False
        name = args.inputs.itemById('preset_name').value.strip()
        try:
            values = _capture_dialog_preset(args.inputs)
            _presets_by_name = presets.save_preset(name, values)
            _populate_preset_dropdown(args.inputs.itemById('preset_select'), name)
            args.inputs.itemById('preset_name').value = name
            _set_preset_status(args.inputs, f'Preset "{name}" saved.')
        except Exception as exc:
            _set_preset_status(args.inputs, str(exc), is_error=True)
        return

    if args.input.id == 'preset_load':
        args.inputs.itemById('preset_load').value = False
        selected = args.inputs.itemById('preset_select').selectedItem
        name = selected.name if selected else ''
        if not name or name == PRESET_EMPTY_LABEL or name not in _presets_by_name:
            _set_preset_status(args.inputs, 'Select a saved preset first.', is_error=True)
            return
        try:
            _apply_dialog_preset(args.inputs, _presets_by_name[name])
            args.inputs.itemById('preset_name').value = name
            _set_preset_status(args.inputs, f'Preset "{name}" loaded.')
        except Exception as exc:
            _set_preset_status(args.inputs, str(exc), is_error=True)
        return

    if args.input.id == 'source_mode':
        _update_source_controls(args.inputs)
        return

    if args.input.id == 'browse_local_source':
        browse_input = args.inputs.itemById('browse_local_source')
        if not browse_input.value:
            return
        folder_input = args.inputs.itemById('local_source_folder')
        dialog = ui.createFolderDialog()
        dialog.title = 'Select Google Drive or another local model folder'
        current_path = local_source.normalize_folder(folder_input.value)
        if current_path and os.path.isdir(current_path):
            dialog.initialDirectory = current_path
        if dialog.showDialog() == adsk.core.DialogResults.DialogOK:
            folder_input.value = dialog.folder
        # A non-checkbox BoolValueCommandInput behaves like a push button, but
        # resetting its value keeps behavior consistent across Fusion versions.
        browse_input.value = False
        _update_source_controls(args.inputs)
        return

    if args.input.id == 'local_source_folder':
        _update_source_controls(args.inputs)
        return

    if args.input.id != 'tool_library':
        return
    library_input = args.inputs.itemById('tool_library')
    small_contour_input = args.inputs.itemById('small_contour_tool')
    large_contour_input = args.inputs.itemById('large_contour_tool')
    small_pocket_input = args.inputs.itemById('small_pocket_tool')
    large_pocket_input = args.inputs.itemById('large_pocket_tool')
    saw_tool_input = args.inputs.itemById('saw_tool')
    bottom_chamfer_tool_input = args.inputs.itemById('bottom_chamfer_tool')
    if library_input.selectedItem:
        selected_url = _library_by_label.get(library_input.selectedItem.name, '')
        if selected_url == _current_milling_library_url:
            return
        _current_milling_library_url = selected_url
        _milling_tool_by_label, _milling_tool_diameter_by_label = tool_library.populate_milling_tools(
            selected_url,
            small_contour_input,
            large_contour_input,
            small_pocket_input,
            large_pocket_input,
        )
        _saw_tool_by_label, _saw_tool_diameter_by_label = tool_library.populate_saw_tools(
            selected_url,
            saw_tool_input,
        )
        _bottom_chamfer_tool_by_label = tool_library.populate_bottom_chamfer_tools(
            selected_url,
            bottom_chamfer_tool_input,
        )


def _populate_preset_dropdown(
    dropdown: adsk.core.DropDownCommandInput,
    selected_name: str = '',
) -> None:
    dropdown.listItems.clear()
    names = sorted(_presets_by_name, key=str.casefold)
    if not names:
        dropdown.listItems.add(PRESET_EMPTY_LABEL, True, '')
        return
    if selected_name not in _presets_by_name:
        selected_name = names[0]
    for name in names:
        dropdown.listItems.add(name, name == selected_name, '')


def _set_preset_status(
    inputs: adsk.core.CommandInputs,
    message: str,
    is_error: bool = False,
) -> None:
    status = inputs.itemById('preset_status')
    color = '#c62828' if is_error else '#2e7d32'
    status.formattedText = f'<span style="color:{color}">{_html(message)}</span>'


def _capture_dialog_preset(inputs: adsk.core.CommandInputs) -> dict:
    library_item = inputs.itemById('tool_library').selectedItem
    machine_item = inputs.itemById('machine').selectedItem
    if not library_item or not machine_item:
        raise ValueError('Select a tool library and machine before saving a preset.')

    small_contour_item = inputs.itemById('small_contour_tool').selectedItem
    large_contour_item = inputs.itemById('large_contour_tool').selectedItem
    small_item = inputs.itemById('small_pocket_tool').selectedItem
    large_item = inputs.itemById('large_pocket_tool').selectedItem
    saw_item = inputs.itemById('saw_tool').selectedItem
    bottom_chamfer_item = inputs.itemById('bottom_chamfer_tool').selectedItem
    return {
        'tool_library_label': library_item.name,
        'tool_library_url': _library_by_label.get(library_item.name, ''),
        'machine_label': machine_item.name,
        'machine_index': _machine_index_by_label.get(machine_item.name, -1),
        'small_contour_tool_label': small_contour_item.name if small_contour_item else '',
        'small_contour_tool_index': _milling_tool_by_label.get(small_contour_item.name, -1)
        if small_contour_item
        else -1,
        'large_contour_tool_label': large_contour_item.name if large_contour_item else '',
        'large_contour_tool_index': _milling_tool_by_label.get(large_contour_item.name, -1)
        if large_contour_item
        else -1,
        'small_pocket_tool_label': small_item.name if small_item else '',
        'small_pocket_tool_index': _milling_tool_by_label.get(small_item.name, -1)
        if small_item
        else -1,
        'large_pocket_tool_label': large_item.name if large_item else '',
        'large_pocket_tool_index': _milling_tool_by_label.get(large_item.name, -1)
        if large_item
        else -1,
        'saw_tool_label': saw_item.name if saw_item else '',
        'saw_tool_index': _saw_tool_by_label.get(saw_item.name, -1) if saw_item else -1,
        'bottom_chamfer_tool_label': (
            bottom_chamfer_item.name if bottom_chamfer_item else ''
        ),
        'bottom_chamfer_tool_index': (
            _bottom_chamfer_tool_by_label.get(bottom_chamfer_item.name, -1)
            if bottom_chamfer_item else -1
        ),
        'create_contour': inputs.itemById('create_contour').value,
        'create_pockets': inputs.itemById('create_pockets').value,
        'create_saw_slots': inputs.itemById('create_saw_slots').value,
        'create_bottom_chamfer': inputs.itemById('create_bottom_chamfer').value,
        'create_drilling': inputs.itemById('create_drilling').value,
        'generate_toolpaths': inputs.itemById('generate_toolpaths').value,
        'create_nc_code': inputs.itemById('create_nc_code').value,
        'contour_stepdown_mm': inputs.itemById('contour_stepdown').value * 10.0,
        'pocket_stepdown_mm': inputs.itemById('pocket_stepdown').value * 10.0,
        'max_drill_diameter_mm': inputs.itemById('max_drill_diameter').value * 10.0,
        'drill_tolerance_mm': inputs.itemById('drill_tolerance').value * 10.0,
        'nc_output_folder': inputs.itemById('nc_output_folder').value.strip(),
    }


def _apply_dialog_preset(inputs: adsk.core.CommandInputs, values: dict) -> None:
    global _milling_tool_by_label, _milling_tool_diameter_by_label
    global _saw_tool_by_label, _saw_tool_diameter_by_label
    global _bottom_chamfer_tool_by_label
    global _current_milling_library_url

    library_label = next(
        (
            label
            for label, url in _library_by_label.items()
            if url == values.get('tool_library_url')
        ),
        '',
    )
    if not library_label and values.get('tool_library_label') in _library_by_label:
        library_label = str(values['tool_library_label'])
    if not library_label:
        raise ValueError('The preset tool library is no longer available.')

    library_input = inputs.itemById('tool_library')
    _select_dropdown_name(library_input, library_label)
    small_contour_input = inputs.itemById('small_contour_tool')
    large_contour_input = inputs.itemById('large_contour_tool')
    small_input = inputs.itemById('small_pocket_tool')
    large_input = inputs.itemById('large_pocket_tool')
    saw_input = inputs.itemById('saw_tool')
    bottom_chamfer_input = inputs.itemById('bottom_chamfer_tool')
    _milling_tool_by_label, _milling_tool_diameter_by_label = (
        tool_library.populate_milling_tools(
            _library_by_label[library_label],
            small_contour_input,
            large_contour_input,
            small_input,
            large_input,
        )
    )
    _saw_tool_by_label, _saw_tool_diameter_by_label = tool_library.populate_saw_tools(
        _library_by_label[library_label],
        saw_input,
    )
    _bottom_chamfer_tool_by_label = tool_library.populate_bottom_chamfer_tools(
        _library_by_label[library_label],
        bottom_chamfer_input,
    )
    _current_milling_library_url = _library_by_label[library_label]

    # Presets saved before the contour tool split into Small/Large carried a
    # single 'contour_tool_index'/'contour_tool_label'. Fall back to that one
    # tool for both fields so an old preset still loads instead of raising.
    if 'small_contour_tool_index' in values or 'small_contour_tool_label' in values:
        small_contour_index = values.get('small_contour_tool_index')
        small_contour_label = values.get('small_contour_tool_label')
    else:
        small_contour_index = values.get('contour_tool_index')
        small_contour_label = values.get('contour_tool_label')
    if 'large_contour_tool_index' in values or 'large_contour_tool_label' in values:
        large_contour_index = values.get('large_contour_tool_index')
        large_contour_label = values.get('large_contour_tool_label')
    else:
        large_contour_index = values.get('contour_tool_index')
        large_contour_label = values.get('contour_tool_label')
    _select_mapped_dropdown(
        small_contour_input,
        _milling_tool_by_label,
        small_contour_index,
        small_contour_label,
        'small contour tool',
    )
    _select_mapped_dropdown(
        large_contour_input,
        _milling_tool_by_label,
        large_contour_index,
        large_contour_label,
        'large contour tool',
    )
    _select_mapped_dropdown(
        small_input,
        _milling_tool_by_label,
        values.get('small_pocket_tool_index'),
        values.get('small_pocket_tool_label'),
        'small pocket tool (tools with zero/missing flute length are excluded)',
    )
    _select_mapped_dropdown(
        large_input,
        _milling_tool_by_label,
        values.get('large_pocket_tool_index'),
        values.get('large_pocket_tool_label'),
        'large pocket tool',
    )
    # Older presets predate the saw tool dropdown; leave populate_saw_tools's
    # own default selection alone instead of raising for a key that was never
    # saved.
    if 'saw_tool_index' in values or 'saw_tool_label' in values:
        _select_mapped_dropdown(
            saw_input,
            _saw_tool_by_label,
            values.get('saw_tool_index'),
            values.get('saw_tool_label'),
            'saw tool',
        )
    # Older presets predate this operation; keep the dropdown's Ø20/30°
    # default rather than rejecting an otherwise valid preset.
    if 'bottom_chamfer_tool_index' in values or 'bottom_chamfer_tool_label' in values:
        _select_mapped_dropdown(
            bottom_chamfer_input,
            _bottom_chamfer_tool_by_label,
            values.get('bottom_chamfer_tool_index'),
            values.get('bottom_chamfer_tool_label'),
            'bottom chamfer tool',
        )
    _select_mapped_dropdown(
        inputs.itemById('machine'),
        _machine_index_by_label,
        values.get('machine_index'),
        values.get('machine_label'),
        'machine',
    )

    for input_id in (
        'create_contour',
        'create_pockets',
        'create_saw_slots',
        'create_bottom_chamfer',
        'create_drilling',
        'generate_toolpaths',
        'create_nc_code',
    ):
        if input_id in values:
            inputs.itemById(input_id).value = bool(values[input_id])

    for input_id, key in (
        ('contour_stepdown', 'contour_stepdown_mm'),
        ('pocket_stepdown', 'pocket_stepdown_mm'),
        ('max_drill_diameter', 'max_drill_diameter_mm'),
        ('drill_tolerance', 'drill_tolerance_mm'),
    ):
        value = values.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            inputs.itemById(input_id).expression = f'{float(value):g} mm'

    nc_output_folder = values.get('nc_output_folder')
    if isinstance(nc_output_folder, str) and nc_output_folder.strip():
        inputs.itemById('nc_output_folder').value = nc_output_folder.strip()


def _select_dropdown_name(
    dropdown: adsk.core.DropDownCommandInput,
    name: str,
) -> bool:
    for item in dropdown.listItems:
        if item.name == name:
            item.isSelected = True
            return True
    return False


def _select_mapped_dropdown(
    dropdown: adsk.core.DropDownCommandInput,
    index_by_label: Dict[str, int],
    saved_index: object,
    saved_label: object,
    description: str,
) -> None:
    if isinstance(saved_index, int) and not isinstance(saved_index, bool):
        for item in dropdown.listItems:
            if index_by_label.get(item.name) == saved_index:
                item.isSelected = True
                return
    if isinstance(saved_label, str) and _select_dropdown_name(dropdown, saved_label):
        return
    raise ValueError(f'The preset {description} is no longer available.')


def _update_source_controls(inputs: adsk.core.CommandInputs) -> None:
    mode_input = inputs.itemById('source_mode')
    folder_input = inputs.itemById('local_source_folder')
    browse_input = inputs.itemById('browse_local_source')
    source_info = inputs.itemById('source_info')
    use_local_source = bool(
        mode_input.selectedItem and mode_input.selectedItem.name == SOURCE_MODE_LOCAL
    )

    folder_input.isVisible = use_local_source
    browse_input.isVisible = use_local_source
    active_folder = app.data.activeFolder
    active_name = active_folder.name if active_folder else '(not available)'

    if use_local_source:
        local_path = folder_input.value.strip()
        files = local_source.discover_model_files(local_path)
        source_name = local_source.folder_name(local_path) or '(select a folder)'
        source_info.formattedText = (
            f'<b>{_html(source_name)}</b><br/>'
            f'Local models: {len(files)}<br/>'
            f'Fusion parent: {_html(active_name)}'
        )
    else:
        source_info.formattedText = (
            f'<b>{_html(active_name)}</b><br/>Fusion models: {_source_file_count}'
        )


def command_validate_inputs(args: adsk.core.ValidateInputsEventArgs) -> None:
    inputs = args.inputs
    folder = app.data.activeFolder
    source_mode = inputs.itemById('source_mode').selectedItem
    use_local_source = bool(source_mode and source_mode.name == SOURCE_MODE_LOCAL)
    local_folder = inputs.itemById('local_source_folder').value.strip()
    source_is_valid = bool(
        folder
        and (
            (use_local_source and local_source.discover_model_files(local_folder))
            or (not use_local_source and _source_file_count > 0)
        )
    )
    destination = inputs.itemById('destination_name').value.strip()
    library = inputs.itemById('tool_library').selectedItem
    machine = inputs.itemById('machine').selectedItem
    contour_enabled = inputs.itemById('create_contour').value
    small_contour_tool = inputs.itemById('small_contour_tool').selectedItem
    large_contour_tool = inputs.itemById('large_contour_tool').selectedItem
    contour_stepdown = inputs.itemById('contour_stepdown').value
    pockets_enabled = inputs.itemById('create_pockets').value
    small_pocket_tool = inputs.itemById('small_pocket_tool').selectedItem
    large_pocket_tool = inputs.itemById('large_pocket_tool').selectedItem
    pocket_stepdown = inputs.itemById('pocket_stepdown').value
    saw_enabled = inputs.itemById('create_saw_slots').value
    saw_tool = inputs.itemById('saw_tool').selectedItem
    bottom_chamfer_enabled = inputs.itemById('create_bottom_chamfer').value
    bottom_chamfer_tool = inputs.itemById('bottom_chamfer_tool').selectedItem
    drilling_enabled = inputs.itemById('create_drilling').value
    max_drill_diameter = inputs.itemById('max_drill_diameter').value
    tolerance = inputs.itemById('drill_tolerance').value
    toolpaths_enabled = inputs.itemById('generate_toolpaths').value
    nc_code_enabled = inputs.itemById('create_nc_code').value
    nc_output_folder = inputs.itemById('nc_output_folder').value.strip()
    small_pocket_diameter = (
        _milling_tool_diameter_by_label.get(small_pocket_tool.name, 0.0)
        if small_pocket_tool else 0.0
    )
    large_pocket_diameter = (
        _milling_tool_diameter_by_label.get(large_pocket_tool.name, 0.0)
        if large_pocket_tool else 0.0
    )
    small_contour_diameter = (
        _milling_tool_diameter_by_label.get(small_contour_tool.name, 0.0)
        if small_contour_tool else 0.0
    )
    large_contour_diameter = (
        _milling_tool_diameter_by_label.get(large_contour_tool.name, 0.0)
        if large_contour_tool else 0.0
    )

    args.areInputsValid = bool(
        source_is_valid
        and local_source.is_safe_folder_name(destination)
        and library
        and machine
        and (not contour_enabled or small_contour_tool)
        and (not contour_enabled or large_contour_tool)
        and (not contour_enabled or contour_stepdown > 0)
        and (not contour_enabled or small_contour_diameter <= large_contour_diameter)
        and (not pockets_enabled or small_pocket_tool)
        and (not pockets_enabled or large_pocket_tool)
        and (not pockets_enabled or pocket_stepdown > 0)
        and (not pockets_enabled or small_pocket_diameter <= large_pocket_diameter)
        # Saw slot recognition piggybacks on pocket recognition (see
        # _process_document), so it needs Pockets enabled too.
        and (not saw_enabled or pockets_enabled)
        and (not saw_enabled or saw_tool)
        and (not bottom_chamfer_enabled or bottom_chamfer_tool)
        and (contour_enabled or pockets_enabled or drilling_enabled or bottom_chamfer_enabled)
        and (not drilling_enabled or max_drill_diameter > 0)
        and tolerance >= 0
        and (not nc_code_enabled or toolpaths_enabled)
        and nc_output_folder
    )


def command_execute(args: adsk.core.CommandEventArgs) -> None:
    global _pending_config
    inputs = args.command.commandInputs
    folder = app.data.activeFolder
    library_label = inputs.itemById('tool_library').selectedItem.name
    machine_item = inputs.itemById('machine').selectedItem
    machine_index = _machine_index_by_label.get(machine_item.name, -1) if machine_item else -1
    small_contour_item = inputs.itemById('small_contour_tool').selectedItem
    large_contour_item = inputs.itemById('large_contour_tool').selectedItem
    small_pocket_item = inputs.itemById('small_pocket_tool').selectedItem
    large_pocket_item = inputs.itemById('large_pocket_tool').selectedItem
    saw_item = inputs.itemById('saw_tool').selectedItem
    bottom_chamfer_item = inputs.itemById('bottom_chamfer_tool').selectedItem
    small_contour_index = (
        _milling_tool_by_label.get(small_contour_item.name, -1) if small_contour_item else -1
    )
    large_contour_index = (
        _milling_tool_by_label.get(large_contour_item.name, -1) if large_contour_item else -1
    )
    small_pocket_index = (
        _milling_tool_by_label.get(small_pocket_item.name, -1) if small_pocket_item else -1
    )
    large_pocket_index = (
        _milling_tool_by_label.get(large_pocket_item.name, -1) if large_pocket_item else -1
    )
    saw_index = _saw_tool_by_label.get(saw_item.name, -1) if saw_item else -1
    bottom_chamfer_index = (
        _bottom_chamfer_tool_by_label.get(bottom_chamfer_item.name, -1)
        if bottom_chamfer_item else -1
    )
    source_mode = inputs.itemById('source_mode').selectedItem
    local_folder = (
        local_source.normalize_folder(inputs.itemById('local_source_folder').value)
        if source_mode and source_mode.name == SOURCE_MODE_LOCAL
        else ''
    )

    _pending_config = BatchConfig(
        source_folder_id=folder.id,
        source_folder_name=folder.name,
        destination_name=inputs.itemById('destination_name').value.strip(),
        library_url=_library_by_label[library_label],
        small_contour_tool_index=small_contour_index,
        large_contour_tool_index=large_contour_index,
        small_pocket_tool_index=small_pocket_index,
        large_pocket_tool_index=large_pocket_index,
        saw_tool_index=saw_index,
        bottom_chamfer_tool_index=bottom_chamfer_index,
        machine_index=machine_index,
        create_contour=inputs.itemById('create_contour').value,
        create_pockets=inputs.itemById('create_pockets').value,
        create_saw_slots=inputs.itemById('create_saw_slots').value,
        create_bottom_chamfer=inputs.itemById('create_bottom_chamfer').value,
        create_drilling=inputs.itemById('create_drilling').value,
        generate_toolpaths=inputs.itemById('generate_toolpaths').value,
        create_nc_code=inputs.itemById('create_nc_code').value,
        drill_tolerance_mm=inputs.itemById('drill_tolerance').value * 10.0,
        max_drill_diameter_mm=inputs.itemById('max_drill_diameter').value * 10.0,
        contour_stepdown_mm=inputs.itemById('contour_stepdown').value * 10.0,
        pocket_stepdown_mm=inputs.itemById('pocket_stepdown').value * 10.0,
        nc_output_folder=inputs.itemById('nc_output_folder').value.strip(),
        local_source_folder=local_folder,
    )


def command_destroy(_args: adsk.core.CommandEventArgs) -> None:
    global local_handlers
    should_start = _pending_config is not None
    local_handlers = []
    if should_start:
        # Custom events execute outside the command transaction.  Opening,
        # saving and closing documents is not supported inside command events.
        app.fireCustomEvent(CUSTOM_EVENT_ID, '')


def batch_event(_args: adsk.core.CustomEventArgs) -> None:
    global _pending_config

    # Local cloud saves must be allowed to finish outside this event callback.
    # A short timer fires this same custom event to advance the import state
    # machine without blocking Fusion's native save/publish workflow.
    if _local_import_session:
        try:
            _poll_local_import()
        except Exception:
            _abort_local_import()
            futil.log(traceback.format_exc(), adsk.core.LogLevels.ErrorLogLevel)
            ui.messageBox(
                'Local import failed:\n\n' + traceback.format_exc(),
                CMD_NAME,
            )
        return

    config_to_run = _pending_config
    _pending_config = None
    if not config_to_run:
        return
    try:
        if config_to_run.local_source_folder:
            _start_local_import(config_to_run)
            return
        _run_batch(config_to_run)
    except Exception:
        futil.log(traceback.format_exc(), adsk.core.LogLevels.ErrorLogLevel)
        ui.messageBox(
            'Batch processing failed:\n\n' + traceback.format_exc(),
            CMD_NAME,
        )


def data_file_complete(args: adsk.core.DataEventArgs) -> None:
    """Wake the local-import state machine when its native save completes."""
    session = _local_import_session
    data_file = args.file
    if not session or not session.current_document or not data_file:
        return

    if not session.current_data_file_id:
        document_file = session.current_document.dataFile
        if document_file:
            session.current_data_file_id = document_file.id

    if _matches_current_local_save(session, data_file):
        session.completed_data_file = data_file
        session.completed_at = time.monotonic()
        futil.log(f'Local import: dataFileComplete received for {data_file.name}.')
        app.fireCustomEvent(CUSTOM_EVENT_ID, '')


def _start_local_import(batch: BatchConfig) -> None:
    """Create the cloud folder and start local import plus native cloud save."""
    global _local_import_session

    parent_folder = app.data.findFolderById(batch.source_folder_id)
    if not parent_folder:
        raise RuntimeError('The Fusion destination folder is no longer available.')

    local_files = local_source.discover_model_files(batch.local_source_folder)
    if not local_files:
        raise RuntimeError(
            'The local folder contains no supported STEP, STP, or F3D models.'
        )
    duplicate_names = local_source.duplicate_model_stems(local_files)
    if duplicate_names:
        raise RuntimeError(
            'Multiple local files would have the same Fusion name after import: '
            + ', '.join(duplicate_names)
        )

    unready_files = local_source.unready_local_files(local_files)
    if unready_files:
        details = '; '.join(
            f'{os.path.basename(path)}: {reason}' for path, reason in unready_files
        )
        raise RuntimeError(
            'Some local files are not fully available yet, which usually means Google '
            'Drive has not finished downloading them. In Google Drive, right-click the '
            'job folder, choose "Available offline", wait for sync to finish, then '
            'retry. Affected files: ' + details
        )

    cloud_folder_name = local_source.folder_name(batch.local_source_folder)
    if not cloud_folder_name:
        raise RuntimeError('The local source folder has no usable name.')
    if parent_folder.dataFolders.itemByName(cloud_folder_name):
        raise RuntimeError(
            f'The Fusion folder "{cloud_folder_name}" already exists in '
            f'"{parent_folder.name}". Nothing was imported to avoid overwriting data.'
        )

    cloud_folder = parent_folder.dataFolders.add(cloud_folder_name)
    if not cloud_folder:
        raise RuntimeError(f'Unable to create the Fusion folder "{cloud_folder_name}".')

    progress = ui.createProgressDialog()
    progress.isCancelButtonShown = True
    progress.show('Import local models', '%v of %m – %p %', 0, len(local_files))
    _local_import_session = LocalImportSession(
        batch=batch,
        cloud_folder=cloud_folder,
        local_files=local_files,
        progress=progress,
        saved_files=[],
        original_document=app.activeDocument,
    )
    try:
        _begin_next_local_import()
    except Exception:
        _abort_local_import()
        raise


def _begin_next_local_import() -> None:
    """Import one model locally and start saving the native Fusion document."""
    session = _local_import_session
    if not session:
        return
    if session.progress.wasCancelled:
        raise RuntimeError(
            f'Import cancelled after {len(session.saved_files)} of '
            f'{len(session.local_files)} files. The partially imported Fusion folder '
            f'"{session.cloud_folder.name}" was kept.'
        )
    if session.next_index >= len(session.local_files):
        _finish_local_import()
        return

    filename = session.local_files[session.next_index]
    display_name = os.path.basename(filename)
    fusion_name = os.path.splitext(display_name)[0]
    extension = os.path.splitext(filename)[1].lower().lstrip('.')
    session.progress.progressValue = session.next_index
    session.progress.message = f'{display_name}: importing locally in Fusion'
    adsk.doEvents()

    import_manager = app.importManager
    if extension in {'step', 'stp'}:
        import_options = import_manager.createSTEPImportOptions(filename)
    elif extension == 'f3d':
        import_options = import_manager.createFusionArchiveImportOptions(filename)
    else:
        raise RuntimeError(f'{display_name}: unsupported local model format.')
    if not import_options:
        raise RuntimeError(f'{display_name}: Fusion could not create import options.')

    document = import_manager.importToNewDocument(import_options)
    if not document:
        raise RuntimeError(f'{display_name}: Fusion could not import the model locally.')
    session.current_document = document
    session.current_data_file_id = ''
    session.completed_data_file = None
    session.completed_at = 0.0
    session.current_started_at = time.monotonic()
    session.fallback_data_file_id = ''
    session.fallback_seen_at = 0.0
    session.last_folder_check_at = 0.0
    session.last_fallback_error = ''
    session.progress.message = f'{display_name}: saving native Fusion file to cloud'
    adsk.doEvents()
    if not document.saveAs(
        fusion_name,
        session.cloud_folder,
        'Imported locally by Automatic Machining',
        '',
    ):
        raise RuntimeError(f'{display_name}: Fusion could not start the cloud save.')

    data_file = document.dataFile
    if data_file:
        session.current_data_file_id = data_file.id

    futil.log(
        f'Local import: imported {display_name} locally and started native cloud save '
        f'({session.next_index + 1}/{len(session.local_files)}).'
    )
    _schedule_local_import_poll()


def _poll_local_import() -> None:
    """Advance a native document save from a short custom event callback."""
    session = _local_import_session
    if not session:
        return
    if not session.current_document:
        _begin_next_local_import()
        return

    if session.progress.wasCancelled:
        raise RuntimeError(
            f'Import cancelled after {len(session.saved_files)} of '
            f'{len(session.local_files)} files. The partially imported Fusion folder '
            f'"{session.cloud_folder.name}" was kept.'
        )

    elapsed = time.monotonic() - session.current_started_at
    filename = session.local_files[session.next_index]
    display_name = os.path.basename(filename)
    document = session.current_document

    data_file = session.completed_data_file
    if not data_file:
        document_file = document.dataFile
        if document_file:
            # A new saveAs starts with a temporary document GUID. Fusion
            # replaces it with the final DataFile lineage ID only after the
            # cloud item has been created, so keep this value fresh instead of
            # permanently retaining the ID observed immediately after saveAs.
            document_file_id = document_file.id
            if document_file_id and document_file_id != session.current_data_file_id:
                session.current_data_file_id = document_file_id

        data_file = _completed_local_save_fallback(session, elapsed)
        if data_file:
            session.completed_data_file = data_file
            session.completed_at = time.monotonic()
            futil.log(
                f'Local import: accepted stable cloud-folder completion for '
                f'{data_file.name} after {elapsed:.1f} seconds.'
            )

    if not data_file:
        if elapsed >= LOCAL_SAVE_TIMEOUT_S:
            fallback_detail = (
                f' Last cloud-folder check error: {session.last_fallback_error}'
                if session.last_fallback_error
                else ''
            )
            raise RuntimeError(
                f'{display_name}: Fusion did not report the native save as complete within '
                f'{LOCAL_SAVE_TIMEOUT_S:g} seconds. The folder '
                f'"{session.cloud_folder.name}" was kept.{fallback_detail}'
            )
        session.progress.message = (
            f'{display_name}: saving native Fusion file to cloud ({elapsed:.0f} s)'
        )
        _schedule_local_import_poll()
        return

    settle_elapsed = time.monotonic() - session.completed_at
    if settle_elapsed < LOCAL_SAVE_SETTLE_S:
        session.progress.message = (
            f'{display_name}: finalizing cloud save '
            f'({settle_elapsed:.1f}/{LOCAL_SAVE_SETTLE_S:g} s)'
        )
        _schedule_local_import_poll()
        return

    session.saved_files.append(data_file.latestVersion)
    _close_local_import_document(session)
    _restore_local_import_document(session)
    session.next_index += 1
    session.progress.progressValue = session.next_index
    futil.log(
        f'Local import: completed {display_name} '
        f'({session.next_index}/{len(session.local_files)}).'
    )
    # Return to Fusion's event loop before importing the next document. This
    # lets native close/upload cleanup finish and avoids leaving Fusion in a
    # transient zero-document state while another import starts.
    _schedule_local_import_poll()


def _matches_current_local_save(
    session: LocalImportSession,
    data_file: adsk.core.DataFile,
) -> bool:
    """Return whether a cloud DataFile belongs to the current native save."""
    if session.current_data_file_id and data_file.id == session.current_data_file_id:
        return True

    # Some Fusion builds expose a version ID on the completion event while the
    # open Document initially exposes the item/lineage ID.  The target folder
    # is new and duplicate stems are rejected, so folder + exact saved name is
    # an equally strict identity check for this importer.
    filename = session.local_files[session.next_index]
    expected_name = os.path.splitext(os.path.basename(filename))[0]
    try:
        parent_folder = data_file.parentFolder
        return bool(
            data_file.name == expected_name
            and parent_folder
            and parent_folder.id == session.cloud_folder.id
        )
    except Exception:
        return False


def _completed_local_save_fallback(
    session: LocalImportSession,
    elapsed: float,
) -> Optional[adsk.core.DataFile]:
    """Confirm a missed completion event using a stable, refreshed folder item."""
    now = time.monotonic()
    if elapsed < LOCAL_SAVE_FALLBACK_START_S:
        return None
    if (
        session.last_folder_check_at
        and now - session.last_folder_check_at < LOCAL_SAVE_FOLDER_CHECK_INTERVAL_S
    ):
        return None
    session.last_folder_check_at = now

    filename = session.local_files[session.next_index]
    expected_name = os.path.splitext(os.path.basename(filename))[0]
    candidate = None
    try:
        # Autodesk documents that getting DataFolder.dataFiles forces Fusion's
        # local cloud representation to update for every file in the folder.
        data_files = session.cloud_folder.dataFiles
        # DataFiles has item/itemById/asArray, but unlike DataFolders it has no
        # itemByName method. The old call raised AttributeError and was
        # swallowed by this fallback's transient-refresh guard, making every
        # successfully published local save time out after 180 s.
        candidate = local_source.find_cloud_data_file(
            data_files,
            session.current_data_file_id,
            expected_name,
        )
        session.last_fallback_error = ''
    except Exception as exc:
        # A temporary Data Panel/cloud refresh failure is not a failed import;
        # the authoritative event or a later refresh can still finish it. Keep
        # the first distinct error visible so a programming/API mismatch does
        # not degrade into an unexplained 180-second timeout again.
        error = f'{type(exc).__name__}: {exc}'
        if error != session.last_fallback_error:
            session.last_fallback_error = error
            futil.log(
                f'Local import: cloud-folder completion check failed: {error}',
                adsk.core.LogLevels.WarningLogLevel,
            )
        candidate = None

    if not candidate or not _matches_current_local_save(session, candidate):
        session.fallback_data_file_id = ''
        session.fallback_seen_at = 0.0
        return None

    try:
        is_complete = candidate.isComplete
    except Exception:
        is_complete = False
    if not is_complete:
        session.fallback_data_file_id = ''
        session.fallback_seen_at = 0.0
        return None

    candidate_id = candidate.id
    if candidate_id != session.fallback_data_file_id:
        session.fallback_data_file_id = candidate_id
        session.fallback_seen_at = now
        futil.log(
            f'Local import: cloud-folder completion candidate found for '
            f'{candidate.name}; waiting for stability.'
        )
        return None
    if now - session.fallback_seen_at < LOCAL_SAVE_FALLBACK_STABLE_S:
        return None
    return candidate


def _finish_local_import() -> None:
    """Hand the completed native Fusion files to the existing batch unchanged."""
    global _local_import_session

    session = _local_import_session
    if not session:
        return
    _cancel_local_import_timer()
    session.progress.hide()
    _local_import_session = None
    _restore_local_import_document(session)

    batch = replace(
        session.batch,
        source_folder_id=session.cloud_folder.id,
        source_folder_name=session.cloud_folder.name,
        local_source_folder='',
    )
    try:
        _run_batch(batch, session.saved_files)
    except Exception:
        futil.log(traceback.format_exc(), adsk.core.LogLevels.ErrorLogLevel)
        ui.messageBox(
            'Batch processing failed after the local import completed:\n\n'
            + traceback.format_exc(),
            CMD_NAME,
        )


def _abort_local_import() -> None:
    global _local_import_session

    _cancel_local_import_timer()
    if _local_import_session:
        _close_local_import_document(_local_import_session)
        _restore_local_import_document(_local_import_session)
        try:
            _local_import_session.progress.hide()
        except Exception:
            pass
    _local_import_session = None


def _schedule_local_import_poll() -> None:
    global _local_import_timer

    _cancel_local_import_timer()
    timer = threading.Timer(
        LOCAL_SAVE_POLL_INTERVAL_S,
        lambda: app.fireCustomEvent(CUSTOM_EVENT_ID, ''),
    )
    timer.daemon = True
    _local_import_timer = timer
    timer.start()


def _cancel_local_import_timer() -> None:
    global _local_import_timer

    if _local_import_timer:
        _local_import_timer.cancel()
        _local_import_timer = None


def _close_local_import_document(session: LocalImportSession) -> None:
    """Close the temporary local-import document without another save."""
    document = session.current_document
    session.current_document = None
    session.current_data_file_id = ''
    session.completed_data_file = None
    session.completed_at = 0.0
    session.fallback_data_file_id = ''
    session.fallback_seen_at = 0.0
    session.last_folder_check_at = 0.0
    session.last_fallback_error = ''
    if document and document.isValid:
        try:
            document.close(False)
        except Exception:
            futil.log(
                'Local import: unable to close the temporary imported document.',
                adsk.core.LogLevels.WarningLogLevel,
            )


def _restore_local_import_document(session: LocalImportSession) -> None:
    """Return Fusion to the document active before the local import."""
    document = session.original_document
    if document and document.isValid:
        try:
            document.activate()
        except Exception:
            pass


def _create_batch_return_document() -> Tuple[adsk.core.Document, bool]:
    """Return a document that can stay active while CAM copies are closed.

    Fusion can be on its Home/Data panel with no active document when a local
    import hands its saved files to the CAM batch.  Closing the only open CAM
    copy in that state makes Fusion create a replacement document internally;
    that native ``CreateNewDocTask`` has been observed to crash.  Keep one
    ordinary design open for the whole batch instead.

    The boolean reports whether this function created the safety document.  A
    safety document is intentionally left open at the end because closing it
    would recreate the same zero-document transition this guard prevents.
    """
    document = app.activeDocument
    if document and document.isValid:
        return document, False

    futil.log('Batch CAM: no active document; creating a safety document.')
    document = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    if not document:
        raise RuntimeError('Unable to create the safety document for batch processing.')
    adsk.doEvents()
    return document, True


def _activate_batch_return_document(document: Optional[adsk.core.Document]) -> bool:
    """Activate the persistent document before an individual CAM copy closes."""
    if not document or not document.isValid:
        return False
    try:
        document.activate()
        deadline = time.monotonic() + RETURN_DOCUMENT_ACTIVATE_TIMEOUT_S
        while True:
            adsk.doEvents()
            if not document.isValid:
                return False
            if document.isActive:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(RETURN_DOCUMENT_ACTIVATE_POLL_INTERVAL_S)
    except Exception:
        futil.log(
            'Batch CAM: unable to activate the return document before closing a copy.\n'
            + traceback.format_exc(),
            adsk.core.LogLevels.WarningLogLevel,
        )
        return False


def _run_batch(
    batch: BatchConfig,
    source_files_override: Optional[Sequence[adsk.core.DataFile]] = None,
) -> None:
    if not local_source.is_safe_folder_name(batch.destination_name):
        raise RuntimeError(
            'The destination folder name is unsafe or is not portable. Use a single '
            'folder name without path separators, reserved characters, or dot segments.'
        )
    source_folder = app.data.findFolderById(batch.source_folder_id)
    if not source_folder:
        raise RuntimeError('The source cloud folder is no longer available.')

    source_files = (
        [data_file.latestVersion for data_file in source_files_override]
        if source_files_override is not None
        else [
            data_file.latestVersion
            for data_file in source_folder.dataFiles
            if data_file.fileExtension.lower().lstrip('.') in SUPPORTED_EXTENSIONS
        ]
    )
    if not source_files:
        raise RuntimeError('The source folder contains no supported Fusion models.')

    destination = source_folder.dataFolders.itemByName(batch.destination_name)
    if destination:
        raise RuntimeError(
            f'The destination folder "{batch.destination_name}" already exists. '
            'Choose a new name to avoid overwriting previous results.'
        )
    destination = source_folder.dataFolders.add(batch.destination_name)
    if not destination:
        raise RuntimeError('Unable to create the folder for copies.')

    tool_library_handle = tool_library.load_tool_library(batch.library_url)
    tools = [tool for tool in tool_library_handle]
    if batch.create_contour:
        small_contour_tool = tool_library.tool_by_index(
            tools, batch.small_contour_tool_index, 'small contour'
        )
        large_contour_tool = tool_library.tool_by_index(
            tools, batch.large_contour_tool_index, 'large contour'
        )
        if (
            tool_library.tool_diameter_cm(small_contour_tool)
            > tool_library.tool_diameter_cm(large_contour_tool)
        ):
            raise RuntimeError(
                'The small contour tool must not be larger than the large contour tool.'
            )
    else:
        small_contour_tool = None
        large_contour_tool = None
    if batch.create_pockets:
        small_pocket_tool = tool_library.tool_by_index(
            tools, batch.small_pocket_tool_index, 'small pocket'
        )
        large_pocket_tool = tool_library.tool_by_index(
            tools, batch.large_pocket_tool_index, 'large pocket'
        )
        if (
            tool_library.tool_diameter_cm(small_pocket_tool)
            > tool_library.tool_diameter_cm(large_pocket_tool)
        ):
            raise RuntimeError(
                'The small pocket tool must not be larger than the large pocket tool.'
            )
    else:
        small_pocket_tool = None
        large_pocket_tool = None
    if batch.create_saw_slots:
        saw_tool = tool_library.tool_by_index(tools, batch.saw_tool_index, 'saw')
    else:
        saw_tool = None
    if batch.create_bottom_chamfer:
        bottom_chamfer_tool = tool_library.tool_by_index(
            tools,
            batch.bottom_chamfer_tool_index,
            'bottom chamfer',
        )
    else:
        bottom_chamfer_tool = None
    drill_tools = [tool for tool in tools if tool_library.tool_type(tool) == 'drill']

    # This local folder always exists (holds the PDF report at minimum), and
    # additionally holds the per-part NC files when create_nc_code is on.
    local_folder = os.path.join(batch.nc_output_folder, batch.destination_name)
    os.makedirs(local_folder, exist_ok=True)

    machine = machine_library.machine_by_index(batch.machine_index)
    post_configuration: Optional[adsk.cam.PostConfiguration] = None
    if batch.create_nc_code:
        post_configuration = machine_library.resolve_post_configuration(machine)

    return_document, safety_document_created = _create_batch_return_document()

    progress = ui.createProgressDialog()
    progress.isCancelButtonShown = True
    progress.show('Automatic Machining', '%v of %m – %p %', 0, len(source_files))

    completed: List[str] = []
    warnings: List[str] = []
    failed: List[str] = []

    try:
        for index, source_file in enumerate(source_files):
            if progress.wasCancelled:
                warnings.append('The batch was cancelled by the user.')
                break
            progress.progressValue = index
            progress.message = f'{source_file.name}: creating copy'
            adsk.doEvents()
            futil.log(
                f'Batch CAM: {index + 1}/{len(source_files)} {source_file.name} '
                '– creating cloud copy.'
            )

            document = None
            try:
                copied_file = source_file.copy(destination)
                if not copied_file:
                    raise RuntimeError('The cloud copy was not created.')
                deadline = time.monotonic() + CLOUD_COPY_TIMEOUT_S
                while not copied_file.isComplete and time.monotonic() < deadline:
                    adsk.doEvents()
                    time.sleep(CLOUD_COPY_POLL_INTERVAL_S)
                if not copied_file.isComplete:
                    raise RuntimeError(
                        f'The cloud copy was not ready within {CLOUD_COPY_TIMEOUT_S:g} seconds.'
                    )
                progress.message = f'{source_file.name}: opening copy'
                adsk.doEvents()
                document = app.documents.open(copied_file, True)
                if not document:
                    raise RuntimeError('Unable to open the copy.')

                futil.log(f'Batch CAM: {source_file.name} – adding CAM operations.')
                result_warnings = _process_document(
                    document,
                    small_contour_tool,
                    large_contour_tool,
                    small_pocket_tool,
                    large_pocket_tool,
                    saw_tool,
                    bottom_chamfer_tool,
                    drill_tools,
                    post_configuration,
                    local_folder,
                    batch,
                )
                warnings.extend(f'{source_file.name}: {msg}' for msg in result_warnings)

                progress.message = f'{source_file.name}: saving CAM'
                adsk.doEvents()
                if not document.save('Automatic Machining – automatically generated CAM'):
                    raise RuntimeError('Unable to save CAM to the copy.')
                completed.append(source_file.name)
                futil.log(f'Batch CAM: {source_file.name} – CAM saved.')
            except Exception as exc:
                frames = traceback.extract_tb(exc.__traceback__)
                own_frame = next(
                    (
                        frame for frame in reversed(frames)
                        if 'automaticMachining' in frame.filename
                    ),
                    None,
                )
                location = ''
                if own_frame:
                    location = (
                        f' [{os.path.basename(own_frame.filename)}:{own_frame.lineno}'
                        f' in {own_frame.name}]'
                    )
                elif frames:
                    last = frames[-1]
                    location = f' [{os.path.basename(last.filename)}:{last.lineno} in {last.name}]'
                failed.append(f'{source_file.name}: {exc}{location}')
                futil.log(
                    f'Automatic Machining failed for {source_file.name}:\n{traceback.format_exc()}',
                    adsk.core.LogLevels.ErrorLogLevel,
                )
            finally:
                if document and document.isValid:
                    progress.message = f'{source_file.name}: closing copy'
                    adsk.doEvents()
                    futil.log(f'Batch CAM: {source_file.name} – preparing to close copy.')
                    if _activate_batch_return_document(return_document):
                        try:
                            if document.close(False):
                                futil.log(f'Batch CAM: {source_file.name} – copy closed.')
                            else:
                                warnings.append(
                                    f'{source_file.name}: Fusion refused to close the copy.'
                                )
                        except Exception:
                            warnings.append(f'{source_file.name}: unable to close the copy.')
                    else:
                        warnings.append(
                            f'{source_file.name}: copy left open because no safe return '
                            'document could be activated.'
                        )
                        futil.log(
                            f'Batch CAM: {source_file.name} – copy left open to avoid '
                            'closing the last active document.',
                            adsk.core.LogLevels.WarningLogLevel,
                        )
            progress.progressValue = index + 1
    finally:
        progress.hide()
        if return_document and return_document.isValid:
            try:
                return_document.activate()
            except Exception:
                pass
        if safety_document_created:
            futil.log(
                'Batch CAM: safety document left open intentionally to avoid a '
                'zero-document transition.'
            )

    _show_report(batch, destination, local_folder, completed, warnings, failed)


def _process_document(
    document: adsk.core.Document,
    small_contour_tool: Optional[adsk.cam.Tool],
    large_contour_tool: Optional[adsk.cam.Tool],
    small_pocket_tool: Optional[adsk.cam.Tool],
    large_pocket_tool: Optional[adsk.cam.Tool],
    saw_tool: Optional[adsk.cam.Tool],
    bottom_chamfer_tool: Optional[adsk.cam.Tool],
    drill_tools: Sequence[adsk.cam.Tool],
    post_configuration: Optional[adsk.cam.PostConfiguration],
    nc_folder: Optional[str],
    batch: BatchConfig,
) -> List[str]:
    warnings: List[str] = []
    design = adsk.fusion.Design.cast(document.products.itemByProductType('DesignProductType'))
    if not design:
        raise RuntimeError('The file does not contain a Fusion Design.')

    bodies = _solid_bodies(design)
    if len(bodies) != 1:
        names = ', '.join(body.name for body in bodies) or 'none'
        raise RuntimeError(
            f'Expected one solid body, found {len(bodies)} ({names}). The file was skipped.'
        )
    body = bodies[0]

    body_orientation = orientation.detect_orientation(body)
    body, body_orientation = _normalize_y_up_model(
        design,
        body,
        body_orientation,
    )

    cam_workspace = ui.workspaces.itemById('CAMEnvironment')
    if cam_workspace:
        cam_workspace.activate()
    cam = adsk.cam.CAM.cast(document.products.itemByProductType('CAMProductType'))
    if not cam:
        raise RuntimeError('Fusion did not create a CAM product.')

    # Pocket count has priority over holes when choosing which face receives
    # the one automatic setup. Recognize both directions before creating any
    # Design/CAM geometry, flip the frame when the opposite side has more
    # pockets, and reuse the recognition results below.
    selected_pocket_recognition: Optional[pockets.PocketRecognitionResult] = None
    opposite_pocket_recognition: Optional[pockets.PocketRecognitionResult] = None
    if batch.create_pockets:
        # If drilling is disabled, every circular feature is eligible for
        # pocket milling. Otherwise only circles above the configured drilling
        # limit are handed to the pocket recognizer.
        pocket_drill_limit_mm = (
            batch.max_drill_diameter_mm if batch.create_drilling else 0.0
        )
        selected_pocket_recognition = pockets.recognize_pocket_features(
            body,
            body_orientation.z,
            body_orientation.z_axis_index,
            pocket_drill_limit_mm,
        )
        opposite_z = body_orientation.z.copy()
        opposite_z.scaleBy(-1.0)
        opposite_pocket_recognition = pockets.recognize_pocket_features(
            body,
            opposite_z,
            body_orientation.z_axis_index,
            pocket_drill_limit_mm,
        )
        selected_count = pockets.recognized_pocket_count(selected_pocket_recognition)
        opposite_count = pockets.recognized_pocket_count(opposite_pocket_recognition)
        if opposite_count > selected_count:
            body_orientation = orientation.flipped(body_orientation)
            selected_pocket_recognition, opposite_pocket_recognition = (
                opposite_pocket_recognition,
                selected_pocket_recognition,
            )

    if body_orientation.used_fallback:
        warnings.append(body_orientation.reason)

    # Through-slot pockets need a boundary sketch drawn on the body's own
    # component (see pockets.prepare_pocket_operations). Prepare it before the
    # Setup so all Design geometry is complete before CAM operations refer to
    # it.
    prepared_pockets: Optional[pockets.PreparedPockets] = None
    if batch.create_pockets and small_pocket_tool and large_pocket_tool:
        prepared_pockets = pockets.prepare_pocket_operations(
            body,
            body_orientation.z,
            body_orientation.z_axis_index,
            small_pocket_tool,
            large_pocket_tool,
            recognized=selected_pocket_recognition,
            saw_tool=saw_tool,
            maximum_drill_diameter_mm=pocket_drill_limit_mm,
        )

    setup = _create_setup(design, cam, body, body_orientation)
    operations: List[adsk.cam.OperationBase] = []

    # Opposite-side pocket recognition was cached before setup creation so the
    # same geometry does not need to be recognized repeatedly.
    opposite_side_kinds: List[str] = []
    if (
        batch.create_pockets
        and opposite_pocket_recognition is not None
        and (
            bool(opposite_pocket_recognition[0])
            or bool(opposite_pocket_recognition[2])
            or bool(opposite_pocket_recognition[3])
        )
    ):
        opposite_side_kinds.append('pockets')
    if batch.create_drilling and holes.has_opposite_side_holes(
        body,
        body_orientation.z,
        batch.max_drill_diameter_mm,
    ):
        opposite_side_kinds.append('holes')
    if opposite_side_kinds:
        warnings.append(
            f"Found {' and '.join(opposite_side_kinds)} on the opposite side of the part; "
            'only this side was machined automatically. MACHINE THE OTHER SIDE MANUALLY.'
        )

    if batch.create_drilling:
        found_holes, recognition_warnings = holes.find_hole_faces(body, body_orientation.z)
        if not found_holes and not recognition_warnings:
            recognition_warnings.append('No simple holes were found for drilling.')
        drillable_holes = [
            hole
            for hole in found_holes
            if hole.diameter_cm * 10.0 <= batch.max_drill_diameter_mm + 1e-6
        ]
        oversized_holes = len(found_holes) - len(drillable_holes)
        if oversized_holes and not batch.create_pockets:
            recognition_warnings.append(
                f'{oversized_holes} hole(s) above the Ø{batch.max_drill_diameter_mm:g} mm '
                'drilling limit were not machined because pocket milling is disabled.'
            )
        drill_operations, drill_warnings = holes.create_drill_operations(
            setup,
            drillable_holes,
            drill_tools,
            batch.drill_tolerance_mm,
        )
        operations.extend(drill_operations)
        warnings.extend(recognition_warnings)
        warnings.extend(drill_warnings)

    if prepared_pockets is not None:
        pocket_attack_vector = body_orientation.z.copy()
        pocket_attack_vector.scaleBy(-1.0)
        pocket_operations, pocket_warnings = pockets.create_pocket_operations(
            cam,
            setup,
            prepared_pockets,
            small_pocket_tool,
            large_pocket_tool,
            batch.pocket_stepdown_mm,
            saw_tool=saw_tool,
            attack_vector=pocket_attack_vector,
        )
        operations.extend(pocket_operations)
        warnings.extend(pocket_warnings)

    if batch.create_contour and large_contour_tool:
        operations.append(
            _create_contour_operation(
                setup,
                body,
                large_contour_tool,
                batch.contour_stepdown_mm,
            )
        )
        # Only add a second pass when the small tool can actually reach
        # somewhere the large one can't (strictly smaller diameter) — picking
        # the same tool for both fields keeps the old single-tool behavior
        # without an empty, no-op finishing operation.
        if (
            small_contour_tool
            and tool_library.tool_diameter_cm(small_contour_tool)
            < tool_library.tool_diameter_cm(large_contour_tool)
        ):
            operations.append(
                _create_contour_operation(
                    setup,
                    body,
                    small_contour_tool,
                    batch.contour_stepdown_mm,
                    display_name='AM – Outer contour (corners)',
                    rest_machining_reference_tool=large_contour_tool,
                )
            )

    # The undercut pass must remain the final operation: the dovetail cutter
    # reaches the lower edge from this same top setup after all drilling,
    # pocketing and through-contour work is complete.
    if batch.create_bottom_chamfer and bottom_chamfer_tool:
        bottom_edge = _find_bottom_chamfer_edge(body, body_orientation)
        if bottom_edge is None:
            futil.log(
                'Bottom chamfer was skipped because the model has no planar chamfer '
                'face adjacent to its lower outside contour.',
                force_console=True,
            )
        else:
            operations.append(
                _create_bottom_chamfer_operation(setup, bottom_edge, bottom_chamfer_tool)
            )

    if not operations:
        warnings.append('The setup was created, but no operations were created.')

    if batch.generate_toolpaths and operations:
        collection = adsk.core.ObjectCollection.create()
        for operation in operations:
            collection.add(operation)
        future = cam.generateToolpath(collection)
        deadline = time.monotonic() + TOOLPATH_GENERATION_TIMEOUT_S
        while not future.isGenerationCompleted and time.monotonic() < deadline:
            adsk.doEvents()
        if not future.isGenerationCompleted:
            raise RuntimeError(
                f'Toolpath generation did not finish within {TOOLPATH_GENERATION_TIMEOUT_S:g} seconds.'
            )
        invalid = [_operation_failure(op) for op in operations if not op.hasToolpath]
        if invalid:
            warnings.append('No toolpath was generated for: ' + ', '.join(invalid))

        if batch.create_nc_code:
            warnings.extend(
                _create_nc_program(cam, setup, post_configuration, nc_folder, document.name)
            )
    elif batch.create_nc_code:
        warnings.append('NC code was not generated because no toolpaths were created.')

    return warnings


def _operation_failure(operation: adsk.cam.OperationBase) -> str:
    """Formats Fusion's actual generation error/warning for the PDF report."""
    messages: List[str] = []
    try:
        if operation.hasError and operation.error:
            messages.append(' '.join(operation.error.split()))
    except Exception:
        pass
    try:
        if operation.hasWarning and operation.warning:
            warning = ' '.join(operation.warning.split())
            if warning not in messages:
                messages.append(warning)
    except Exception:
        pass
    return f"{operation.name} ({'; '.join(messages)})" if messages else operation.name


def _normalize_y_up_model(
    design: adsk.fusion.Design,
    body: adsk.fusion.BRepBody,
    body_orientation: orientation.Orientation,
) -> Tuple[adsk.fusion.BRepBody, orientation.Orientation]:
    """Lays an XZ-oriented panel into XY while leaving other models alone.

    Some STEP exporters use Y for panel thickness, which makes an otherwise
    valid part stand vertically in Fusion.  Rotate only that recognizable
    case so the detected machining direction becomes global +Z.  Models whose
    machining direction is already X or Z keep their original geometry.

    The move must happen before pocket/hole preparation because a MoveFeature
    replaces the body's faces and invalidates previously captured references.
    """
    if body_orientation.z_axis_index != 1:
        return body, body_orientation

    if body.assemblyContext:
        raise RuntimeError(
            'Automatic Y/Z model correction currently requires the single solid '
            'body to be in the root component.'
        )

    root = design.rootComponent
    if body.parentComponent != root:
        raise RuntimeError(
            'Automatic Y/Z model correction currently requires the single solid '
            'body to be in the root component.'
        )

    transform = adsk.core.Matrix3D.create()
    global_z = adsk.core.Vector3D.create(0.0, 0.0, 1.0)
    if not transform.setToRotateTo(body_orientation.z, global_z):
        raise RuntimeError('Unable to calculate the automatic Y/Z model correction.')

    # Rotate around the body centre instead of the design origin.  Imported
    # STEP files are often located far from (0, 0, 0), and rotating there would
    # make the part jump unnecessarily across the canvas.
    box = body.boundingBox
    centre = adsk.core.Point3D.create(
        (box.minPoint.x + box.maxPoint.x) / 2.0,
        (box.minPoint.y + box.maxPoint.y) / 2.0,
        (box.minPoint.z + box.maxPoint.z) / 2.0,
    )
    rotated_centre = centre.copy()
    if not rotated_centre.transformBy(transform):
        raise RuntimeError('Unable to calculate the automatic Y/Z model correction.')
    transform.translation = rotated_centre.vectorTo(centre)

    entities = adsk.core.ObjectCollection.create()
    entities.add(body)
    move_features = root.features.moveFeatures
    move_input = move_features.createInput2(entities)
    if not move_input or not move_input.defineAsFreeMove(transform):
        raise RuntimeError('Unable to define the automatic Y/Z model correction.')
    move_features.add(move_input)

    # A parametric MoveFeature creates replacement topology, while a direct
    # model updates the body in place.  Resolve it again in both cases rather
    # than retaining a potentially invalid BRep handle.
    moved_bodies = _solid_bodies(design)
    if len(moved_bodies) != 1:
        raise RuntimeError('Unable to resolve the body after automatic Y/Z correction.')
    moved_body = moved_bodies[0]
    moved_orientation = orientation.detect_orientation(moved_body)
    if moved_orientation.z_axis_index != 2:
        raise RuntimeError('Automatic Y/Z model correction produced an invalid orientation.')
    return moved_body, moved_orientation


def _create_nc_program(
    cam: adsk.cam.CAM,
    setup: adsk.cam.Setup,
    post_configuration: Optional[adsk.cam.PostConfiguration],
    nc_folder: Optional[str],
    document_name: str,
) -> List[str]:
    if not post_configuration or not nc_folder:
        return ['NC code was not generated because no post processor/output folder was resolved.']

    file_name = _safe_filename(document_name)
    nc_input = cam.ncPrograms.createInput()
    nc_input.displayName = f'AM – {file_name}'
    nc_input.operations = [setup]
    nc_input.parameters.itemByName('nc_program_output_folder').value.value = (
        nc_folder.replace('\\', '/')
    )
    nc_input.parameters.itemByName('nc_program_filename').value.value = file_name
    open_in_editor = nc_input.parameters.itemByName('nc_program_openInEditor')
    if open_in_editor:
        open_in_editor.value.value = False

    nc_program = cam.ncPrograms.add(nc_input)
    if not nc_program:
        return ['Unable to create the NC program.']
    # Deliberately not setting nc_program.machine: the same Fusion crash that
    # affects Setup.machine (see _create_setup) may apply here too, since it
    # goes through the same machine-simulation-model binding internally.
    # postConfiguration alone is sufficient to post NC code.
    nc_program.postConfiguration = post_configuration

    options = adsk.cam.NCProgramPostProcessOptions.create()
    if not nc_program.postProcess(options):
        return ['Post-processing failed; no NC file was written.']
    return []


def _create_setup(
    design: adsk.fusion.Design,
    cam: adsk.cam.CAM,
    body: adsk.fusion.BRepBody,
    body_orientation: orientation.Orientation,
) -> adsk.cam.Setup:
    root = design.rootComponent
    transform = adsk.core.Matrix3D.create()
    transform.setWithCoordinateSystem(
        adsk.core.Point3D.create(0, 0, 0),
        body_orientation.x,
        body_orientation.y,
        body_orientation.z,
    )
    # Supplying the transform during creation also works when Fusion is set to
    # ground the first component automatically.
    occurrence = root.occurrences.addNewComponent(transform)
    component = occurrence.component
    component.name = ORIENTATION_COMPONENT_NAME

    setup_input = cam.setups.createInput(adsk.cam.OperationTypes.MillingOperation)
    setup_input.models = [body]
    setup_input.name = SETUP_NAME
    setup_input.stockMode = adsk.cam.SetupStockModes.RelativeBoxStock
    # Deliberately not setting setup_input.machine: for machines without a full
    # simulation model (e.g. post-only machine definitions like "Homag Wood
    # Wop"), Fusion crashes inside Setups.add() while it evaluates
    # Machine.isSimulationReady(). The machine's post processor is still used
    # for NC generation via resolve_post_configuration(); only the cosmetic
    # "assigned machine" shown in the Setup browser is skipped.
    setup = cam.setups.add(setup_input)
    setup.name = SETUP_NAME

    set_expression(setup.parameters, 'job_stockOffsetMode', "'simple'")
    set_expression(setup.parameters, 'job_stockOffsetSides', '0 mm')
    set_expression(setup.parameters, 'job_stockOffsetTop', '0 mm')

    axis_x = component.xConstructionAxis.createForAssemblyContext(occurrence)
    axis_y = component.yConstructionAxis.createForAssemblyContext(occurrence)
    set_expression(setup.parameters, 'wcs_orientation_mode', "'axesXY'")
    setup.parameters.itemByName('wcs_orientation_axisX').value.value = [axis_x]
    setup.parameters.itemByName('wcs_orientation_axisY').value.value = [axis_y]

    # WoodWOP convention requested by the user: lower-left corner on the
    # underside of the material, not on the top surface.
    setup.parameters.itemByName('wcs_origin_mode').value.value = 'modelPoint'
    setup.parameters.itemByName('wcs_origin_boxPoint').value.value = 'bottom 1'
    return setup


def _create_contour_operation(
    setup: adsk.cam.Setup,
    body: adsk.fusion.BRepBody,
    tool: adsk.cam.Tool,
    stepdown_mm: float,
    display_name: str = 'AM – Outer contour',
    rest_machining_reference_tool: Optional[adsk.cam.Tool] = None,
) -> adsk.cam.OperationBase:
    operation_input = setup.operations.createInput('contour2d')
    operation_input.displayName = display_name
    operation_input.tool = tool
    set_safe_heights(operation_input.parameters)
    set_expression(operation_input.parameters, 'topHeight_mode', "'from surface top'")
    set_expression(operation_input.parameters, 'topHeight_offset', '0 mm')
    set_expression(operation_input.parameters, 'bottomHeight_mode', "'from surface bottom'")
    set_expression(operation_input.parameters, 'bottomHeight_offset', '0 mm')
    set_expression(operation_input.parameters, 'doMultipleDepths', 'true')
    set_zero_stock_to_leave(operation_input.parameters)
    # Machine-control cutter compensation outputs G41/G42. Fusion's internal
    # enum value is "control" (the UI label is "In control").
    set_expression(operation_input.parameters, 'compensationType', "'control'")
    set_expression(
        operation_input.parameters,
        'maximumStepdown',
        f'{stepdown_mm:g} mm',
    )
    # Fusion's default (~0.01 mm) linearizes the silhouette's arcs/splines into
    # far more segments than a CNC router needs, which measurably slows
    # toolpath calculation on rounded profiles. 0.1 mm is still well within
    # router accuracy.
    set_expression(operation_input.parameters, 'tolerance', '0.1 mm')
    if rest_machining_reference_tool is not None:
        # Rest Machining compares this tool's reach against a reference
        # cutter diameter/corner radius and only cuts what that reference
        # tool could not — i.e. inside corners tighter than its radius. This
        # lets a small finishing tool clean up just the corners a larger
        # roughing tool leaves, instead of re-tracing the whole perimeter.
        reference_diameter_mm = (
            tool_library.tool_diameter_cm(rest_machining_reference_tool) * 10.0
        )
        reference_corner_radius_mm = (
            tool_library.tool_corner_radius_cm(rest_machining_reference_tool) * 10.0
        )
        set_expression(operation_input.parameters, 'useRestMachining', 'true')
        set_expression(
            operation_input.parameters,
            'restMaterialCutterDiameter',
            f'{reference_diameter_mm:g} mm',
        )
        set_expression(
            operation_input.parameters,
            'restMaterialCornerRadius',
            f'{reference_corner_radius_mm:g} mm',
        )

    contour_value = operation_input.parameters.itemByName('contours').value
    selections = contour_value.getCurveSelections()
    silhouette = selections.createNewSilhouetteSelection()
    silhouette.isSetupModelSelected = False
    silhouette.loopType = adsk.cam.LoopTypes.OnlyOutsideLoops
    silhouette.inputGeometry = [body]
    contour_value.applyCurveSelections(selections)
    operation = setup.operations.add(operation_input)
    # Fusion's default (nonzero) horizontal lead-in radius left a gap in the
    # cut contour near the lead-in point instead of a full pass around the
    # boundary. Confirmed on real output; requested by the user for every 2D
    # Contour operation this add-in creates. Must be set AFTER operations.add()
    # on 'entry_radius' — that's the true editable parameter behind the UI's
    # "Horizontal Lead-In Radius" field. 'leadInRadius' looks like the same
    # thing and accepts writes, but it's a read-only derived mirror
    # (parameters.itemByName('leadInRadius').isEditable is False) that Fusion
    # recomputes from 'entry_radius' on every recompute — e.g. the instant a
    # sibling operation is added to or removed from this setup. Writing
    # 'leadInRadius' directly looked like it worked (read-back matched) right
    # up until the next recompute silently reverted it back to the tool
    # preset. Confirmed on real output.
    set_expression(operation.parameters, 'entry_radius', '0 mm')
    return operation


def _find_bottom_chamfer_edge(
    body: adsk.fusion.BRepBody,
    body_orientation: orientation.Orientation,
) -> Optional[adsk.fusion.BRepEdge]:
    """Finds a modeled chamfer edge on the underside of the part.

    In setup coordinates ``orientation.z`` points out of the top machining
    face.  The dovetail cutter must reference the opposite planar face so
    ``bottomHeight = from contour`` places its tip at the physical underside,
    not at the top surface. Before choosing an edge, require its other
    adjacent face to be planar and
    genuinely angled relative to setup Z. A normal vertical side has zero Z
    alignment, while the underside has unit alignment; only a face between
    those cases is a modeled chamfer. This prevents the enabled UI option from
    inventing a chamfer on every ordinary square-edged part.

    Within the underside profile, ``orientation.y`` points toward the top of
    the screen. When several modeled chamfer edges exist, choose the one whose
    two endpoints have the smallest setup-Y coordinate. Requiring both endpoints
    to sit at that level avoids selecting a sloping side which merely touches
    the bottom at one corner.
    """
    setup_faces: List[adsk.fusion.BRepFace] = []
    for face in body.faces:
        if not adsk.core.Plane.cast(face.geometry):
            continue
        ok, normal = face.evaluator.getNormalAtPoint(face.pointOnFace)
        if ok and normal.dotProduct(body_orientation.z) < -0.98:
            setup_faces.append(face)
    if not setup_faces:
        return None

    profile_face = max(setup_faces, key=lambda face: face.area)
    outer_loop = next((loop for loop in profile_face.loops if loop.isOuter), None)
    if outer_loop is None:
        return None
    edges = [
        edge
        for edge in outer_loop.edges
        if edge.startVertex is not None and edge.endVertex is not None
    ]
    chamfer_edges: List[adsk.fusion.BRepEdge] = []
    for edge in edges:
        for adjacent_face in edge.faces:
            if not adsk.core.Plane.cast(adjacent_face.geometry):
                continue
            ok, adjacent_normal = adjacent_face.evaluator.getNormalAtPoint(
                adjacent_face.pointOnFace
            )
            if not ok:
                continue
            z_alignment = abs(adjacent_normal.dotProduct(body_orientation.z))
            # STEP imports are rarely mathematically exact. 0.05 rejects a
            # vertical side with generous tolerance; 0.98 rejects the bottom
            # face itself and any effectively horizontal ledge.
            if 0.05 < z_alignment < 0.98:
                chamfer_edges.append(edge)
                break
    edges = chamfer_edges
    if not edges:
        return None

    def endpoint_y(edge: adsk.fusion.BRepEdge) -> Tuple[float, float]:
        start = edge.startVertex.geometry
        end = edge.endVertex.geometry
        return (
            start.x * body_orientation.y.x
            + start.y * body_orientation.y.y
            + start.z * body_orientation.y.z,
            end.x * body_orientation.y.x
            + end.y * body_orientation.y.y
            + end.z * body_orientation.y.z,
        )

    edge_levels = [(edge, *endpoint_y(edge)) for edge in edges]
    minimum_y = min(min(start_y, end_y) for _, start_y, end_y in edge_levels)
    tolerance_cm = 1e-5
    horizontal_candidates = [
        edge
        for edge, start_y, end_y in edge_levels
        if abs(start_y - minimum_y) <= tolerance_cm
        and abs(end_y - minimum_y) <= tolerance_cm
    ]
    if horizontal_candidates:
        return max(horizontal_candidates, key=lambda edge: edge.length)

    # Non-orthogonal profiles may not contain a perfectly level lower edge.
    # Pick the boundary segment whose higher endpoint is lowest; its length is
    # only a tie-breaker, so a long sloping side cannot beat a lower segment.
    return min(
        edge_levels,
        key=lambda item: (max(item[1], item[2]), -item[0].length),
    )[0]


def _create_bottom_chamfer_operation(
    setup: adsk.cam.Setup,
    bottom_edge: adsk.fusion.BRepEdge,
    tool: adsk.cam.Tool,
) -> adsk.cam.OperationBase:
    """Cuts the selected lower profile edge with a dovetail cutter.

    These settings mirror the user-confirmed ``2D Contour2`` operation on
    HPL_Lengtebed_kast_hoekversteviging_achter_H00159.
    """
    operation_input = setup.operations.createInput('contour2d')
    operation_input.displayName = 'AM – Bottom chamfer'
    operation_input.tool = tool
    set_safe_heights(operation_input.parameters)
    set_expression(operation_input.parameters, 'topHeight_mode', "'from stock top'")
    set_expression(operation_input.parameters, 'topHeight_offset', '0 mm')
    set_expression(operation_input.parameters, 'bottomHeight_mode', "'from contour'")
    set_expression(operation_input.parameters, 'bottomHeight_offset', '0 mm')
    set_expression(operation_input.parameters, 'doMultipleDepths', 'false')
    set_zero_stock_to_leave(operation_input.parameters)
    set_expression(operation_input.parameters, 'compensation', "'left'")
    set_expression(operation_input.parameters, 'compensationType', "'computer'")
    set_expression(operation_input.parameters, 'tolerance', '0.1 mm')

    contour_parameter = operation_input.parameters.itemByName('contours')
    if not contour_parameter:
        raise RuntimeError('The 2D Contour strategy does not expose contour selection.')
    contour_value = contour_parameter.value
    selections = contour_value.getCurveSelections()
    chain = selections.createNewChainSelection()
    chain.isOpen = True
    # The underside face has the opposite loop orientation from the top face.
    # Reverse its lower edge so Left compensation keeps the dovetail cutter
    # outside the material (confirmed on the H00159 sample).
    chain.isReverted = True
    chain.inputGeometry = [bottom_edge]
    contour_value.applyCurveSelections(selections)
    operation = setup.operations.add(operation_input)
    set_expression(operation.parameters, 'entry_radius', '0 mm')
    return operation


def _solid_bodies(design: adsk.fusion.Design) -> List[adsk.fusion.BRepBody]:
    bodies: List[adsk.fusion.BRepBody] = []
    for body in design.rootComponent.bRepBodies:
        if body.isSolid and body.isVisible:
            bodies.append(body)
    for occurrence in design.rootComponent.allOccurrences:
        if not occurrence.isVisible:
            continue
        for body in occurrence.bRepBodies:
            if body.isSolid and body.isVisible:
                bodies.append(body)
    return bodies


def _count_supported_files(folder: Optional[adsk.core.DataFolder]) -> int:
    if not folder:
        return 0
    return sum(
        1
        for data_file in folder.dataFiles
        if data_file.fileExtension.lower().lstrip('.') in SUPPORTED_EXTENSIONS
    )


def _show_report(
    batch: BatchConfig,
    destination: adsk.core.DataFolder,
    report_folder: str,
    completed: Sequence[str],
    warnings: Sequence[str],
    failed: Sequence[str],
) -> None:
    summary_lines = [
        f'Source: {batch.source_folder_name}',
        f'Copies: {destination.name}',
        f'Completed: {len(completed)}',
        f'Errors: {len(failed)}',
    ]

    # The full, untruncated list always goes to the Fusion log file (and the
    # PDF report below) so large batches don't silently lose detail behind
    # the message box's cutoff.
    full_lines = list(summary_lines)
    if warnings:
        full_lines.append('\nWarnings:')
        full_lines.extend(f'• {item}' for item in warnings)
    if failed:
        full_lines.append('\nFailed files:')
        full_lines.extend(f'• {item}' for item in failed)
    full_report = '\n'.join(full_lines)
    futil.log(full_report, force_console=True)
    app.log(full_report, adsk.core.LogLevels.InfoLogLevel, adsk.core.LogTypes.FileLogType)

    pdf_path = os.path.join(report_folder, f'{batch.destination_name} - Report.pdf')
    try:
        pdf_lines = [line for raw_line in full_lines for line in raw_line.split('\n')]
        pdf_report.write_report_pdf(pdf_path, f'Automatic Machining – {batch.destination_name}', pdf_lines)
        pdf_note = f'PDF report: {pdf_path}'
    except Exception:
        futil.log(traceback.format_exc(), adsk.core.LogLevels.ErrorLogLevel)
        pdf_note = 'PDF report could not be written (see the Fusion log file).'

    dialog_lines = list(summary_lines)
    if warnings:
        dialog_lines.append('\nWarnings:')
        dialog_lines.extend(f'• {item}' for item in warnings[:MAX_REPORT_ITEMS])
        if len(warnings) > MAX_REPORT_ITEMS:
            dialog_lines.append(
                f'• ... and {len(warnings) - MAX_REPORT_ITEMS} more (see the Fusion log file for the full list)'
            )
    if failed:
        dialog_lines.append('\nFailed files:')
        dialog_lines.extend(f'• {item}' for item in failed[:MAX_REPORT_ITEMS])
        if len(failed) > MAX_REPORT_ITEMS:
            dialog_lines.append(
                f'• ... and {len(failed) - MAX_REPORT_ITEMS} more (see the Fusion log file for the full list)'
            )
    dialog_lines.append(f'\n{pdf_note}')
    ui.messageBox('\n'.join(dialog_lines), CMD_NAME)


def _html(value: str) -> str:
    return (
        value.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )


def _safe_filename(name: str) -> str:
    base_name = os.path.splitext(name)[0]
    return re.sub(r'[^A-Za-z0-9 _.-]', '_', base_name).strip() or 'AutomaticMachining'
