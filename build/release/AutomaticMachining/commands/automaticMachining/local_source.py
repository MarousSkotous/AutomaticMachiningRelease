"""Discovery and validation for locally synchronized source folders.

This module deliberately has no dependency on the Fusion ``adsk`` package so
the path handling can be tested outside Fusion.  Google Drive for desktop is
treated like any other local filesystem provider; opening a streamed file is
left to the operating system and the Drive client.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Tuple


SUPPORTED_EXTENSIONS = frozenset({'f3d', 'step', 'stp'})
_WINDOWS_RESERVED_NAMES = frozenset(
    {'CON', 'PRN', 'AUX', 'NUL'}
    | {f'COM{number}' for number in range(1, 10)}
    | {f'LPT{number}' for number in range(1, 10)}
)
_INVALID_FOLDER_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def normalize_folder(path: str) -> str:
    """Return a normalized absolute path without requiring it to exist."""
    return os.path.abspath(os.path.expanduser(path.strip())) if path.strip() else ''


def folder_name(path: str) -> str:
    """Return the final directory name used for the Fusion cloud folder."""
    normalized = normalize_folder(path)
    return os.path.basename(normalized) if normalized else ''


def is_safe_folder_name(name: str, max_length: int = 120) -> bool:
    """Return whether *name* is one portable, non-traversing folder component.

    The destination is used both as a Fusion cloud-folder name and as a local
    report subdirectory.  Apply Windows' stricter filename rules on every
    platform so a preset/job created on macOS remains safe and usable there.
    """
    if not isinstance(name, str) or name != name.strip():
        return False
    if not name or len(name) > max_length or name in {'.', '..'}:
        return False
    if _INVALID_FOLDER_CHARACTERS.search(name) or name.endswith(('.', ' ')):
        return False
    stem = name.split('.', 1)[0].upper()
    return stem not in _WINDOWS_RESERVED_NAMES


def discover_model_files(path: str) -> List[str]:
    """Return supported files directly inside *path*, sorted by filename.

    Subfolders are intentionally not traversed.  A Drive job folder maps to
    one Fusion source folder, which keeps the subsequent batch deterministic.
    """
    normalized = normalize_folder(path)
    if not normalized or not os.path.isdir(normalized):
        return []

    files: List[str] = []
    try:
        with os.scandir(normalized) as entries:
            for entry in entries:
                # Never let a model-looking symlink upload data from outside
                # the directory the user explicitly selected.
                if not entry.is_file(follow_symlinks=False):
                    continue
                extension = os.path.splitext(entry.name)[1].lower().lstrip('.')
                if extension in SUPPORTED_EXTENSIONS:
                    files.append(entry.path)
    except OSError:
        return []
    return sorted(files, key=lambda value: os.path.basename(value).casefold())


def duplicate_model_stems(files: List[str]) -> List[str]:
    """Return names that would collide after Fusion strips file extensions."""
    names_by_stem = {}
    for filename in files:
        stem = os.path.splitext(os.path.basename(filename))[0]
        names_by_stem.setdefault(stem.casefold(), []).append(stem)
    return sorted(
        {names[0] for names in names_by_stem.values() if len(names) > 1},
        key=str.casefold,
    )


def unready_local_files(files: List[str], probe_bytes: int = 65536) -> List[Tuple[str, str]]:
    """Return (path, reason) for files that are missing, empty, or unreadable.

    Google Drive for desktop can list a "stream"/on-demand file in the folder
    before its content is actually present on disk. Probing a small read here
    forces the same access Fusion's importer will need and fails fast with a
    plain Python error, before any cloud folder or upload is created, instead
    of failing unpredictably inside ``ImportManager.importToNewDocument``.
    """
    problems: List[Tuple[str, str]] = []
    for path in files:
        try:
            size = os.path.getsize(path)
        except OSError as error:
            problems.append((path, f'not accessible ({error.strerror or error})'))
            continue
        if size == 0:
            problems.append((path, 'file is empty (0 bytes) — likely not downloaded yet'))
            continue
        try:
            with open(path, 'rb') as handle:
                handle.read(probe_bytes)
        except OSError as error:
            problems.append((path, f'could not be read ({error.strerror or error})'))
    return problems


def find_cloud_data_file(
    data_files: Any,
    preferred_id: str,
    expected_name: str,
) -> Any:
    """Find a refreshed Fusion DataFile without relying on itemByName.

    Fusion's DataFiles collection has no ``itemByName`` method. Keeping this
    small lookup API-independent makes the save-completion regression testable
    outside Fusion.
    """
    candidate = data_files.itemById(preferred_id) if preferred_id else None
    if candidate:
        return candidate
    return next(
        (
            data_file
            for data_file in data_files.asArray()
            if data_file.name == expected_name
        ),
        None,
    )
