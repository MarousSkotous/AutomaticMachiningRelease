"""Persistent named machining presets for the Automatic Machining dialog."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Dict, Mapping, Optional


SCHEMA_VERSION = 1
MAX_PRESET_FILE_BYTES = 2 * 1024 * 1024


def default_file_path() -> str:
    """Returns a user-specific path that survives add-in updates."""
    if sys.platform == 'darwin':
        base = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support')
    elif os.name == 'nt':
        base = os.environ.get('APPDATA') or os.path.expanduser('~')
    else:
        base = os.environ.get('XDG_CONFIG_HOME') or os.path.join(
            os.path.expanduser('~'), '.config'
        )
    return os.path.join(base, 'AutomaticMachining', 'presets.json')


def load_presets(path: Optional[str] = None) -> Dict[str, dict]:
    filename = path or default_file_path()
    if not os.path.isfile(filename):
        return {}
    if os.path.getsize(filename) > MAX_PRESET_FILE_BYTES:
        raise ValueError('The Automatic Machining preset file is unexpectedly large.')
    with open(filename, 'r', encoding='utf-8') as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get('version') != SCHEMA_VERSION:
        raise ValueError('Unsupported Automatic Machining preset file format.')
    stored = document.get('presets')
    if not isinstance(stored, dict):
        raise ValueError('The Automatic Machining preset file is invalid.')
    return {
        str(name): dict(values)
        for name, values in stored.items()
        if isinstance(name, str) and name.strip() and isinstance(values, dict)
    }


def save_preset(
    name: str,
    values: Mapping[str, object],
    path: Optional[str] = None,
) -> Dict[str, dict]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError('Enter a preset name before saving.')

    filename = path or default_file_path()
    stored = load_presets(filename)
    stored[clean_name] = dict(values)
    folder = os.path.dirname(os.path.abspath(filename))
    os.makedirs(folder, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.presets-', suffix='.tmp', dir=folder)
    try:
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass  # Windows ACLs are not represented by POSIX mode bits.
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            descriptor = -1
            json.dump(
                {'version': SCHEMA_VERSION, 'presets': stored},
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, filename)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return stored
