"""Commands exposed by the add-in."""

import importlib

from .. import config
from .automaticMachining import entry as automaticMachining
from .help import entry as help_command
from .update import entry as update

fusion_debug_bridge = None


commands = [automaticMachining, update, help_command]


# Assumes you defined a "start" function in each of your modules.
# The start function will be run when the add-in is started.
def start():
    global automaticMachining, help_command, update, commands, fusion_debug_bridge

    # Fusion can keep imported child modules alive across Scripts and Add-Ins
    # Stop/Run cycles. During development, reload the command implementation so
    # the code on disk is the code that is actually tested. The module has not
    # registered handlers yet on the initial run, and stop() cleans them before
    # subsequent runs.
    if config.DEBUG:
        from ..lib import fusion_debug_bridge as debug_bridge

        automaticMachining = importlib.reload(automaticMachining)
        help_command = importlib.reload(help_command)
        update = importlib.reload(update)
        fusion_debug_bridge = importlib.reload(debug_bridge)
        commands = [automaticMachining, update, help_command]

    for command in commands:
        command.start()

    # Dev-only: lets the fusion-debug MCP server run Python inside this
    # process for interactive debugging. Never starts outside config.DEBUG.
    if config.DEBUG:
        fusion_debug_bridge.start()


# Assumes you defined a "stop" function in each of your modules.
# The stop function will be run when the add-in is stopped.
def stop():
    for command in commands:
        command.stop()
    if config.DEBUG and fusion_debug_bridge:
        fusion_debug_bridge.stop()
