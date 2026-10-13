"""Fusion UI for checking and installing Automatic Machining releases."""

from __future__ import annotations

import json
import os
import threading
import traceback
import webbrowser

import adsk.core

from ... import config
from ... import version
from ...lib import fusionAddInUtils as futil
from ...lib import update_service


app = adsk.core.Application.get()
ui = app.userInterface

CMD_ID = f'{config.COMPANY_NAME}_{config.ADDIN_NAME}_update'
CMD_NAME = 'Update'
CMD_DESCRIPTION = 'Checks for and installs Automatic Machining updates.'
CUSTOM_EVENT_ID = f'{CMD_ID}_result'
WORKSPACE_ID = 'CAMEnvironment'
TAB_ID = 'MillingTab'
PANEL_ID = 'AutomaticMachiningPanel'
ICON_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'automaticMachining',
    'resources',
    '',
)

local_handlers = []
_custom_event = None
_checking = False
_downloading = False
_latest = None
_active_inputs = None
_notify_when_complete = False


def start() -> None:
    global _custom_event
    panel = _toolbar_panel()
    if not panel:
        raise RuntimeError('Unable to add the update command to the Automatic Machining panel.')
    old_control = panel.controls.itemById(CMD_ID)
    old_definition = ui.commandDefinitions.itemById(CMD_ID)
    if old_control:
        old_control.deleteMe()
    if old_definition:
        old_definition.deleteMe()
    definition = ui.commandDefinitions.addButtonDefinition(
        CMD_ID, CMD_NAME, CMD_DESCRIPTION, ICON_FOLDER
    )
    futil.add_handler(definition.commandCreated, command_created)
    control = panel.controls.addCommand(definition)
    control.isPromoted = True
    try:
        app.unregisterCustomEvent(CUSTOM_EVENT_ID)
    except Exception:
        pass
    _custom_event = app.registerCustomEvent(CUSTOM_EVENT_ID)
    futil.add_handler(_custom_event, update_event)
    check_for_updates(notify=True)


def stop() -> None:
    global _custom_event, _active_inputs
    panel = _toolbar_panel()
    control = panel.controls.itemById(CMD_ID) if panel else None
    if control:
        control.deleteMe()
    definition = ui.commandDefinitions.itemById(CMD_ID)
    if definition:
        definition.deleteMe()
    try:
        app.unregisterCustomEvent(CUSTOM_EVENT_ID)
    except Exception:
        pass
    _custom_event = None
    _active_inputs = None
    local_handlers.clear()


def _toolbar_panel():
    workspace = ui.workspaces.itemById(WORKSPACE_ID)
    if not workspace:
        return None
    tab = workspace.toolbarTabs.itemById(TAB_ID)
    if not tab:
        return None
    return tab.toolbarPanels.itemById(PANEL_ID)


def command_created(args: adsk.core.CommandCreatedEventArgs) -> None:
    global _active_inputs
    command = args.command
    command.setDialogInitialSize(470, 270)
    inputs = command.commandInputs
    inputs.addTextBoxCommandInput(
        'current_version', 'Installed', f'<b>{version.VERSION}</b>', 1, True
    )
    inputs.addTextBoxCommandInput('available_version', 'Available', _available_text(), 1, True)
    inputs.addTextBoxCommandInput('update_status', '', _status_text(), 3, True)
    inputs.addBoolValueInput('check_now', 'Check now', False, '', False)
    install = inputs.addBoolValueInput('install_update', 'Install update', False, '', False)
    install.isEnabled = _can_install()
    _active_inputs = inputs
    futil.add_handler(command.inputChanged, input_changed, local_handlers=local_handlers)
    futil.add_handler(command.destroy, command_destroy, local_handlers=local_handlers)
    if not _checking and _latest is None:
        check_for_updates()


def input_changed(args: adsk.core.InputChangedEventArgs) -> None:
    changed = args.input
    if changed.id == 'check_now':
        changed.value = False
        check_for_updates()
    elif changed.id == 'install_update':
        changed.value = False
        install_update()


def command_destroy(_args: adsk.core.CommandEventArgs) -> None:
    global _active_inputs
    _active_inputs = None
    local_handlers.clear()


def _available_text() -> str:
    if _checking:
        return 'Checking…'
    if not _latest:
        return '—'
    return f'<b>{_latest.version}</b>'


def _status_text() -> str:
    if _checking:
        return 'Checking GitHub Releases…'
    if _downloading:
        return 'Downloading and verifying the installer…'
    if not _latest:
        return 'Use Check now to query the latest published release.'
    if update_service.is_newer(_latest.version, version.VERSION):
        return 'A newer version is ready to install.'
    return 'Automatic Machining is up to date.'


def _can_install() -> bool:
    return bool(
        os.name == 'nt'
        and _latest
        and update_service.is_newer(_latest.version, version.VERSION)
        and not _downloading
    )


def _refresh_dialog(status: str = '') -> None:
    inputs = _active_inputs
    if not inputs:
        return
    available = inputs.itemById('available_version')
    status_input = inputs.itemById('update_status')
    install = inputs.itemById('install_update')
    if available:
        available.formattedText = _available_text()
    if status_input:
        status_input.formattedText = status or _status_text()
    if install:
        install.isEnabled = _can_install()


def _fire(payload: dict) -> None:
    app.fireCustomEvent(CUSTOM_EVENT_ID, json.dumps(payload))


def check_for_updates(notify: bool = False) -> None:
    global _checking, _notify_when_complete
    if _checking:
        return
    _checking = True
    _notify_when_complete = _notify_when_complete or notify
    _refresh_dialog()

    def worker():
        try:
            release = update_service.fetch_latest_release()
            _fire({'kind': 'check', 'ok': True, 'release': release.__dict__})
        except Exception as exc:
            _fire({'kind': 'check', 'ok': False, 'error': str(exc)})

    threading.Thread(target=worker, name='AutomaticMachiningUpdateCheck', daemon=True).start()


def install_update() -> None:
    global _downloading
    if not _can_install() or _downloading:
        return
    release = _latest
    _downloading = True
    _refresh_dialog()

    def worker():
        try:
            path = update_service.download_installer(release)
            _fire({'kind': 'download', 'ok': True, 'path': path})
        except Exception as exc:
            _fire({'kind': 'download', 'ok': False, 'error': str(exc)})

    threading.Thread(target=worker, name='AutomaticMachiningUpdateDownload', daemon=True).start()


def update_event(args: adsk.core.CustomEventArgs) -> None:
    global _checking, _downloading, _latest, _notify_when_complete
    try:
        payload = json.loads(args.additionalInfo)
        if payload.get('kind') == 'check':
            _checking = False
            if payload.get('ok'):
                _latest = update_service.ReleaseInfo(**payload['release'])
                has_update = update_service.is_newer(_latest.version, version.VERSION)
                _refresh_dialog()
                definition = ui.commandDefinitions.itemById(CMD_ID)
                if definition:
                    definition.name = f'Update to {_latest.version}' if has_update else CMD_NAME
                if has_update and _notify_when_complete:
                    ui.messageBox(
                        f'Automatic Machining {_latest.version} is available.\n\n'
                        'Use the Update button in the AUTOMATIC MACHINING panel to install it.',
                        'Automatic Machining update',
                    )
            else:
                _refresh_dialog('Update check failed: ' + payload.get('error', 'Unknown error'))
            _notify_when_complete = False
            return
        if payload.get('kind') == 'download':
            _downloading = False
            if not payload.get('ok'):
                _refresh_dialog('Download failed: ' + payload.get('error', 'Unknown error'))
                return
            update_service.launch_installer(payload['path'])
            _refresh_dialog('Installer started. Restart Fusion after it finishes.')
            ui.messageBox(
                'The verified update installer has started. Restart Fusion after installation '
                'to load the new version.',
                'Automatic Machining update',
            )
    except Exception:
        futil.log(
            'Automatic Machining updater failed:\n' + traceback.format_exc(),
            adsk.core.LogLevels.ErrorLogLevel,
        )
        _checking = False
        _downloading = False


def open_releases_page() -> None:
    webbrowser.open(_latest.page_url if _latest else update_service.RELEASES_URL)
