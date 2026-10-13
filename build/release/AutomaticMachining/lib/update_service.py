"""Release discovery, download, verification, and installer launching.

This module deliberately has no Fusion API dependency so it can be tested with
ordinary Python and used by the Fusion-facing update command.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import subprocess
import tempfile
from typing import Callable, Iterable, Optional, Tuple
from urllib.request import Request, urlopen


RELEASE_REPOSITORY = 'MarousSkotous/AutomaticMachiningRelease'
LATEST_RELEASE_API = f'https://api.github.com/repos/{RELEASE_REPOSITORY}/releases/latest'
RELEASES_URL = f'https://github.com/{RELEASE_REPOSITORY}/releases'
INSTALLER_ASSET = 'AutomaticMachining-Windows-Setup.exe'
CHECKSUM_ASSET = INSTALLER_ASSET + '.sha256'
USER_AGENT = 'AutomaticMachining-Updater'
_VERSION_RE = re.compile(r'^v?(\d+)\.(\d+)\.(\d+)(?:[-+]([0-9A-Za-z.-]+))?$')


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    page_url: str
    installer_url: str
    checksum_url: str
    notes: str = ''


def parse_version(value: str) -> Tuple[int, int, int, int, str]:
    """Return a comparison key for the supported semantic-version subset."""
    match = _VERSION_RE.fullmatch((value or '').strip())
    if not match:
        raise ValueError(f'Unsupported version: {value!r}')
    major, minor, patch = (int(match.group(index)) for index in range(1, 4))
    suffix = match.group(4) or ''
    # A stable version sorts after a prerelease with the same numeric triplet.
    return major, minor, patch, 1 if not suffix else 0, suffix


def is_newer(candidate: str, current: str) -> bool:
    return parse_version(candidate) > parse_version(current)


def _read_url(url: str, timeout: float = 15.0) -> bytes:
    request = Request(
        url,
        headers={
            'Accept': 'application/vnd.github+json',
            'User-Agent': USER_AGENT,
            'X-GitHub-Api-Version': '2022-11-28',
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def release_from_payload(payload: dict) -> ReleaseInfo:
    tag = str(payload.get('tag_name', '')).strip()
    parse_version(tag)
    assets = {
        asset.get('name'): asset.get('browser_download_url')
        for asset in payload.get('assets', [])
        if isinstance(asset, dict)
    }
    installer_url = assets.get(INSTALLER_ASSET, '')
    checksum_url = assets.get(CHECKSUM_ASSET, '')
    if not installer_url or not checksum_url:
        raise RuntimeError(
            f'Release {tag} does not contain {INSTALLER_ASSET} and its checksum.'
        )
    return ReleaseInfo(
        version=tag.lstrip('v'),
        page_url=payload.get('html_url') or RELEASES_URL,
        installer_url=installer_url,
        checksum_url=checksum_url,
        notes=payload.get('body') or '',
    )


def fetch_latest_release(timeout: float = 15.0) -> ReleaseInfo:
    payload = json.loads(_read_url(LATEST_RELEASE_API, timeout).decode('utf-8'))
    return release_from_payload(payload)


def _expected_checksum(contents: bytes) -> str:
    text = contents.decode('ascii', errors='strict').strip()
    match = re.search(r'(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', text)
    if not match:
        raise RuntimeError('The release checksum file is invalid.')
    return match.group(0).lower()


def download_installer(
    release: ReleaseInfo,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Download the installer and verify it against the release checksum."""
    if os.name != 'nt':
        raise RuntimeError('The automatic installer is available only on Windows.')
    destination_dir = os.path.join(
        os.environ.get('LOCALAPPDATA', tempfile.gettempdir()),
        'AutomaticMachining',
        'Updates',
        release.version,
    )
    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, INSTALLER_ASSET)
    partial = destination + '.part'
    if progress:
        progress('Downloading checksum…')
    expected = _expected_checksum(_read_url(release.checksum_url, 30.0))
    if progress:
        progress('Downloading installer…')
    request = Request(release.installer_url, headers={'User-Agent': USER_AGENT})
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=60.0) as response, open(partial, 'wb') as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
        if digest.hexdigest().lower() != expected:
            raise RuntimeError('Installer checksum verification failed; nothing was installed.')
        os.replace(partial, destination)
    finally:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
    return destination


def launch_installer(path: str) -> None:
    """Launch a per-user, silent Inno Setup update."""
    if os.name != 'nt':
        raise RuntimeError('The automatic installer is available only on Windows.')
    subprocess.Popen(
        [
            path,
            '/SP-',
            '/SILENT',
            '/SUPPRESSMSGBOXES',
            '/NORESTART',
        ],
        close_fds=True,
    )
