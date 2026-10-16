"""Display the bundled Automatic Machining HTML guide in a Fusion palette."""

from __future__ import annotations

import os

import adsk.core

from ... import config
from ...lib import fusionAddInUtils as futil


app = adsk.core.Application.get()
ui = app.userInterface

CMD_ID = f'{config.COMPANY_NAME}_{config.ADDIN_NAME}_help'
CMD_NAME = 'Help'
CMD_DESCRIPTION = 'Opens the offline Automatic Machining guide.'
PALETTE_ID = f'{config.COMPANY_NAME}_{config.ADDIN_NAME}_help_palette'
PALETTE_NAME = 'Automatic Machining – Help'
WORKSPACE_ID = 'CAMEnvironment'
TAB_ID = 'MillingTab'
PANEL_ID = 'AutomaticMachiningPanel'
HELP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'help.html')
HELP_URL = HELP_FILE.replace('\\', '/')
ICON_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'automaticMachining',
    'resources',
    '',
)

local_handlers = []


def start() -> None:
    panel = _toolbar_panel()
    if not panel:
        raise RuntimeError('Unable to add Help to the Automatic Machining panel.')
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


def stop() -> None:
    panel = _toolbar_panel()
    control = panel.controls.itemById(CMD_ID) if panel else None
    if control:
        control.deleteMe()
    definition = ui.commandDefinitions.itemById(CMD_ID)
    if definition:
        definition.deleteMe()
    palette = ui.palettes.itemById(PALETTE_ID)
    if palette:
        palette.deleteMe()
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
    futil.add_handler(args.command.execute, command_execute, local_handlers=local_handlers)
    futil.add_handler(args.command.destroy, command_destroy, local_handlers=local_handlers)


def command_execute(_args: adsk.core.CommandEventArgs) -> None:
    palette = ui.palettes.itemById(PALETTE_ID)
    if palette is None:
        palette = ui.palettes.add(
            id=PALETTE_ID,
            name=PALETTE_NAME,
            htmlFileURL=HELP_URL,
            isVisible=True,
            showCloseButton=True,
            isResizable=True,
            width=720,
            height=760,
            useNewWebBrowser=True,
        )
        futil.add_handler(palette.navigatingURL, palette_navigating)
    palette.isVisible = True


def palette_navigating(args: adsk.core.NavigationEventArgs) -> None:
    if args.navigationURL.lower().startswith(('http://', 'https://')):
        args.launchExternally = True


def command_destroy(_args: adsk.core.CommandEventArgs) -> None:
    local_handlers.clear()
