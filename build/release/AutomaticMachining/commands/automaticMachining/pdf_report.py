"""Minimal, dependency-free PDF writer for the batch report.

Fusion's embedded Python interpreter has no guaranteed pip/network access,
so this hand-writes a PDF (no reportlab/weasyprint/a browser) using only the
standard library, plus a small bundled JPEG (resources/report/vanjorn_logo.jpg)
for the VAN-Jorn letterhead. Text uses the Helvetica core fonts (no
embedding needed) with real AFM advance widths, so wrapping is accurate for
a proportional font instead of falling back to a fixed-width one.
"""

from __future__ import annotations

import datetime
import os
import tempfile
from typing import Dict, List, NamedTuple, Sequence, Tuple

Color = Tuple[float, float, float]

PAGE_WIDTH = 595.0  # A4, points
PAGE_HEIGHT = 842.0
MARGIN = 48.0
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

# VAN-Jorn brand palette (from vanjorn.com's Elementor theme colors).
ACCENT = (0.890, 0.494, 0.169)  # #E37E2B - primary orange
RULE_SAND = (0.827, 0.663, 0.306)  # #D3A94E - secondary gold, quiet accents
TEXT_DARK = (0.20, 0.20, 0.20)  # #333333
TEXT_MUTED = (0.412, 0.447, 0.490)  # #69727D

TITLE_SIZE = 16.0
TITLE_LINE_HEIGHT = TITLE_SIZE * 1.25
SUBTITLE_SIZE = 8.5
SUBTITLE_LINE_HEIGHT = SUBTITLE_SIZE * 1.3
SECTION_SIZE = 11.5
SECTION_LINE_HEIGHT = SECTION_SIZE * 1.3
SECTION_GAP_BEFORE = 10.0
BODY_SIZE = 9.5
BODY_LINE_HEIGHT = BODY_SIZE * 1.35
BLANK_LINE_HEIGHT = BODY_LINE_HEIGHT * 0.6
BULLET_INDENT = 12.0
CONT_HEADER_SIZE = 8.0
FOOTER_SIZE = 8.0

TOP_BAND_HEIGHT = 8.0
LOGO_WIDTH = 168.0

_RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'resources', 'report')
_LOGO_PATH = os.path.join(_RESOURCES_DIR, 'vanjorn_logo.jpg')


def _jpeg_size(data: bytes) -> Tuple[int, int]:
    """Reads (width, height) from a JPEG's SOFn marker without any library."""
    if data[0:2] != b'\xff\xd8':
        raise ValueError('not a JPEG')
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = (data[i + 2] << 8) + data[i + 3]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = (data[i + 5] << 8) + data[i + 6]
            width = (data[i + 7] << 8) + data[i + 8]
            return width, height
        i += 2 + seg_len
    raise ValueError('SOF marker not found')


# --- Standard Helvetica / Helvetica-Bold AFM advance widths (1/1000 em),
# keyed by WinAnsi/cp1252 code point. Used for accurate proportional wrapping.
_HELVETICA_WIDTHS: Dict[int, int] = {
    32: 278, 33: 278, 34: 355, 35: 556, 36: 556, 37: 889, 38: 667, 39: 191, 40: 333, 41: 333,
    42: 389, 43: 584, 44: 278, 45: 333, 46: 278, 47: 278, 48: 556, 49: 556, 50: 556, 51: 556,
    52: 556, 53: 556, 54: 556, 55: 556, 56: 556, 57: 556, 58: 278, 59: 278, 60: 584, 61: 584,
    62: 584, 63: 556, 64: 1015, 65: 667, 66: 667, 67: 722, 68: 722, 69: 667, 70: 611, 71: 778,
    72: 722, 73: 278, 74: 500, 75: 667, 76: 556, 77: 833, 78: 722, 79: 778, 80: 667, 81: 778,
    82: 722, 83: 667, 84: 611, 85: 722, 86: 667, 87: 944, 88: 667, 89: 667, 90: 611, 91: 278,
    92: 278, 93: 278, 94: 469, 95: 556, 96: 333, 97: 556, 98: 556, 99: 500, 100: 556, 101: 556,
    102: 278, 103: 556, 104: 556, 105: 222, 106: 222, 107: 500, 108: 222, 109: 833, 110: 556, 111: 556,
    112: 556, 113: 556, 114: 333, 115: 500, 116: 278, 117: 556, 118: 500, 119: 722, 120: 500, 121: 500,
    122: 500, 123: 334, 124: 260, 125: 334, 126: 584, 127: 350, 128: 556, 129: 350, 130: 222, 131: 556,
    132: 333, 133: 1000, 134: 556, 135: 556, 136: 333, 137: 1000, 138: 667, 139: 333, 140: 1000, 141: 350,
    142: 611, 143: 350, 144: 350, 145: 222, 146: 222, 147: 333, 148: 333, 149: 350, 150: 556, 151: 1000,
    152: 333, 153: 1000, 154: 500, 155: 333, 156: 944, 157: 350, 158: 500, 159: 667, 160: 278, 161: 333,
    162: 556, 163: 556, 164: 556, 165: 556, 166: 260, 167: 556, 168: 333, 169: 737, 170: 370, 171: 556,
    172: 584, 173: 333, 174: 737, 175: 333, 176: 400, 177: 584, 178: 333, 179: 333, 180: 333, 181: 556,
    182: 537, 183: 278, 184: 333, 185: 333, 186: 365, 187: 556, 188: 834, 189: 834, 190: 834, 191: 611,
    192: 667, 193: 667, 194: 667, 195: 667, 196: 667, 197: 667, 198: 1000, 199: 722, 200: 667, 201: 667,
    202: 667, 203: 667, 204: 278, 205: 278, 206: 278, 207: 278, 208: 722, 209: 722, 210: 778, 211: 778,
    212: 778, 213: 778, 214: 778, 215: 584, 216: 778, 217: 722, 218: 722, 219: 722, 220: 722, 221: 667,
    222: 667, 223: 611, 224: 556, 225: 556, 226: 556, 227: 556, 228: 556, 229: 556, 230: 889, 231: 500,
    232: 556, 233: 556, 234: 556, 235: 556, 236: 278, 237: 278, 238: 278, 239: 278, 240: 556, 241: 556,
    242: 556, 243: 556, 244: 556, 245: 556, 246: 556, 247: 584, 248: 611, 249: 556, 250: 556, 251: 556,
    252: 556, 253: 500, 254: 556, 255: 500,
}

_HELVETICA_BOLD_WIDTHS: Dict[int, int] = {
    32: 278, 33: 333, 34: 474, 35: 556, 36: 556, 37: 889, 38: 722, 39: 238, 40: 333, 41: 333,
    42: 389, 43: 584, 44: 278, 45: 333, 46: 278, 47: 278, 48: 556, 49: 556, 50: 556, 51: 556,
    52: 556, 53: 556, 54: 556, 55: 556, 56: 556, 57: 556, 58: 333, 59: 333, 60: 584, 61: 584,
    62: 584, 63: 611, 64: 975, 65: 722, 66: 722, 67: 722, 68: 722, 69: 667, 70: 611, 71: 778,
    72: 722, 73: 278, 74: 556, 75: 722, 76: 611, 77: 833, 78: 722, 79: 778, 80: 667, 81: 778,
    82: 722, 83: 667, 84: 611, 85: 722, 86: 667, 87: 944, 88: 667, 89: 667, 90: 611, 91: 333,
    92: 278, 93: 333, 94: 584, 95: 556, 96: 333, 97: 556, 98: 611, 99: 556, 100: 611, 101: 556,
    102: 333, 103: 611, 104: 611, 105: 278, 106: 278, 107: 556, 108: 278, 109: 889, 110: 611, 111: 611,
    112: 611, 113: 611, 114: 389, 115: 556, 116: 333, 117: 611, 118: 556, 119: 778, 120: 556, 121: 556,
    122: 500, 123: 389, 124: 280, 125: 389, 126: 584, 127: 350, 128: 556, 129: 350, 130: 278, 131: 556,
    132: 500, 133: 1000, 134: 556, 135: 556, 136: 333, 137: 1000, 138: 667, 139: 333, 140: 1000, 141: 350,
    142: 611, 143: 350, 144: 350, 145: 278, 146: 278, 147: 500, 148: 500, 149: 350, 150: 556, 151: 1000,
    152: 333, 153: 1000, 154: 556, 155: 333, 156: 944, 157: 350, 158: 500, 159: 667, 160: 278, 161: 333,
    162: 556, 163: 556, 164: 556, 165: 556, 166: 280, 167: 556, 168: 333, 169: 737, 170: 370, 171: 556,
    172: 584, 173: 333, 174: 737, 175: 333, 176: 400, 177: 584, 178: 333, 179: 333, 180: 333, 181: 611,
    182: 556, 183: 278, 184: 333, 185: 333, 186: 365, 187: 556, 188: 834, 189: 834, 190: 834, 191: 611,
    192: 722, 193: 722, 194: 722, 195: 722, 196: 722, 197: 722, 198: 1000, 199: 722, 200: 667, 201: 667,
    202: 667, 203: 667, 204: 278, 205: 278, 206: 278, 207: 278, 208: 722, 209: 722, 210: 778, 211: 778,
    212: 778, 213: 778, 214: 778, 215: 584, 216: 778, 217: 722, 218: 722, 219: 722, 220: 722, 221: 667,
    222: 667, 223: 611, 224: 556, 225: 556, 226: 556, 227: 556, 228: 556, 229: 556, 230: 889, 231: 556,
    232: 556, 233: 556, 234: 556, 235: 556, 236: 278, 237: 278, 238: 278, 239: 278, 240: 611, 241: 611,
    242: 611, 243: 611, 244: 611, 245: 611, 246: 611, 247: 584, 248: 611, 249: 611, 250: 611, 251: 611,
    252: 611, 253: 556, 254: 611, 255: 556,
}

_DEFAULT_CHAR_WIDTH = 556


def _escape(text: str) -> bytes:
    escaped = text.replace('\\', r'\\').replace('(', r'\(').replace(')', r'\)')
    # cp1252 (Windows-1252) rather than latin-1: same printable range plus
    # the bullet (0x95) and en/em dash (0x96/0x97) our report text uses.
    return escaped.encode('cp1252', errors='replace')


def _text_width(text: str, widths: Dict[int, int], size: float) -> float:
    total = 0
    for byte in text.encode('cp1252', errors='replace'):
        total += widths.get(byte, _DEFAULT_CHAR_WIDTH)
    return total * size / 1000.0


def _wrap(text: str, widths: Dict[int, int], size: float, max_width: float) -> List[str]:
    if not text:
        return ['']
    words = text.split(' ')
    wrapped: List[str] = []
    current = ''
    for word in words:
        candidate = f'{current} {word}'.strip()
        if current and _text_width(candidate, widths, size) > max_width:
            wrapped.append(current)
            current = word
        else:
            current = candidate
    if current:
        wrapped.append(current)
    return wrapped or ['']


class _Row(NamedTuple):
    kind: str  # 'blank' | 'header' | 'bullet_first' | 'bullet_cont' | 'body'
    text: str
    gap_before: float


_ROW_HEIGHT = {
    'blank': BLANK_LINE_HEIGHT,
    'header': SECTION_LINE_HEIGHT,
    'bullet_first': BODY_LINE_HEIGHT,
    'bullet_cont': BODY_LINE_HEIGHT,
    'body': BODY_LINE_HEIGHT,
}


def _build_rows(lines: Sequence[str]) -> List[_Row]:
    rows: List[_Row] = []
    for raw_line in lines:
        if raw_line == '':
            rows.append(_Row('blank', '', 0.0))
            continue
        if raw_line.rstrip().endswith(':') and not raw_line.startswith('• '):
            for sub in _wrap(raw_line, _HELVETICA_BOLD_WIDTHS, SECTION_SIZE, CONTENT_WIDTH):
                rows.append(_Row('header', sub, SECTION_GAP_BEFORE))
            continue
        if raw_line.startswith('• '):
            content = raw_line[2:]
            sublines = _wrap(content, _HELVETICA_WIDTHS, BODY_SIZE, CONTENT_WIDTH - BULLET_INDENT)
            for i, sub in enumerate(sublines):
                kind = 'bullet_first' if i == 0 else 'bullet_cont'
                rows.append(_Row(kind, sub, 0.0))
            continue
        for sub in _wrap(raw_line, _HELVETICA_WIDTHS, BODY_SIZE, CONTENT_WIDTH):
            rows.append(_Row('body', sub, 0.0))
    return rows


class _Op(NamedTuple):
    kind: str  # 'rect' | 'text'
    x: float
    y: float
    a: float  # rect: width | text: unused
    b: float  # rect: height | text: unused
    color: Color
    font: str  # 'regular' | 'bold', text only
    size: float  # text only
    text: str  # text only


def _rect(x: float, y: float, w: float, h: float, color: Color) -> _Op:
    return _Op('rect', x, y, w, h, color, '', 0.0, '')


def _text(x: float, y: float, font: str, size: float, color: Color, text: str) -> _Op:
    return _Op('text', x, y, 0.0, 0.0, color, font, size, text)


def _header_geometry(title: str, has_logo: bool) -> Tuple[List[_Op], float]:
    """Builds the page-1 letterhead ops and returns (ops, body_start_y)."""
    ops: List[_Op] = []
    y = PAGE_HEIGHT
    ops.append(_rect(0.0, y - TOP_BAND_HEIGHT, PAGE_WIDTH, TOP_BAND_HEIGHT, ACCENT))
    y -= TOP_BAND_HEIGHT

    logo_height = 0.0
    if has_logo:
        try:
            with open(_LOGO_PATH, 'rb') as handle:
                jpeg_bytes = handle.read()
            logo_w_px, logo_h_px = _jpeg_size(jpeg_bytes)
            logo_height = LOGO_WIDTH * (logo_h_px / logo_w_px)
        except (OSError, ValueError):
            has_logo = False

    if has_logo:
        y -= 22.0
        y -= logo_height  # logo occupies [y, y + logo_height]
        ops.append(_Op('image', MARGIN, y, LOGO_WIDTH, logo_height, ACCENT, '', 0.0, ''))
        y -= 20.0
    else:
        y -= 28.0

    title_lines = _wrap(title, _HELVETICA_BOLD_WIDTHS, TITLE_SIZE, CONTENT_WIDTH)
    for line in title_lines:
        ops.append(_text(MARGIN, y, 'bold', TITLE_SIZE, TEXT_DARK, line))
        y -= TITLE_LINE_HEIGHT

    y -= 2.0
    generated = datetime.datetime.now().strftime('Generated %Y-%m-%d %H:%M')
    ops.append(_text(MARGIN, y, 'regular', SUBTITLE_SIZE, TEXT_MUTED, generated))
    y -= SUBTITLE_LINE_HEIGHT

    y -= 8.0
    ops.append(_rect(MARGIN, y - 1.2, CONTENT_WIDTH, 1.2, ACCENT))
    y -= 1.2
    y -= 16.0
    return ops, y


def _continuation_header_geometry(title: str) -> Tuple[List[_Op], float]:
    ops: List[_Op] = []
    y = PAGE_HEIGHT - 34.0
    label_lines = _wrap(title, _HELVETICA_WIDTHS, CONT_HEADER_SIZE, CONTENT_WIDTH)
    ops.append(_text(MARGIN, y, 'regular', CONT_HEADER_SIZE, TEXT_MUTED, 'VAN JORN · ' + (label_lines[0] if label_lines else title)))
    y -= 10.0
    ops.append(_rect(MARGIN, y, CONTENT_WIDTH, 1.0, RULE_SAND))
    y -= 16.0
    return ops, y


FOOTER_BOTTOM_LIMIT = MARGIN + 34.0


def _footer_ops(page_index: int, page_count: int) -> List[_Op]:
    ops: List[_Op] = []
    rule_y = MARGIN + 20.0
    ops.append(_rect(MARGIN, rule_y, CONTENT_WIDTH, 1.0, RULE_SAND))
    baseline = MARGIN + 8.0
    ops.append(_text(MARGIN, baseline, 'regular', FOOTER_SIZE, TEXT_MUTED, 'Automatic Machining'))
    page_label = f'Page {page_index + 1} of {page_count}'
    label_width = _text_width(page_label, _HELVETICA_WIDTHS, FOOTER_SIZE)
    ops.append(_text(PAGE_WIDTH - MARGIN - label_width, baseline, 'regular', FOOTER_SIZE, TEXT_MUTED, page_label))
    return ops


def write_report_pdf(path: str, title: str, lines: Sequence[str]) -> None:
    """Writes a VAN-Jorn-branded, multi-page report PDF."""
    has_logo = os.path.isfile(_LOGO_PATH)
    page1_header_ops, page1_body_start = _header_geometry(title, has_logo)
    cont_header_ops, cont_body_start = _continuation_header_geometry(title)

    rows = _build_rows(lines)

    pages_rows: List[List[_Row]] = []
    current_page: List[_Row] = []
    remaining = page1_body_start - FOOTER_BOTTOM_LIMIT
    for row in rows:
        h = row.gap_before + _ROW_HEIGHT[row.kind]
        if h > remaining and current_page:
            pages_rows.append(current_page)
            current_page = []
            remaining = cont_body_start - FOOTER_BOTTOM_LIMIT
        current_page.append(row)
        remaining -= h
    pages_rows.append(current_page)

    page_count = len(pages_rows)

    pages_ops: List[List[_Op]] = []
    for page_index, page_rows in enumerate(pages_rows):
        ops: List[_Op] = list(page1_header_ops if page_index == 0 else cont_header_ops)
        cursor_y = page1_body_start if page_index == 0 else cont_body_start
        for row in page_rows:
            cursor_y -= row.gap_before
            if row.kind == 'header':
                ops.append(_text(MARGIN, cursor_y, 'bold', SECTION_SIZE, ACCENT, row.text))
                ops.append(_rect(MARGIN, cursor_y - 5.0, 24.0, 2.0, ACCENT))
            elif row.kind == 'bullet_first':
                ops.append(_rect(MARGIN + 2.0, cursor_y + 2.2, 3.2, 3.2, ACCENT))
                ops.append(_text(MARGIN + BULLET_INDENT, cursor_y, 'regular', BODY_SIZE, TEXT_DARK, row.text))
            elif row.kind == 'bullet_cont':
                ops.append(_text(MARGIN + BULLET_INDENT, cursor_y, 'regular', BODY_SIZE, TEXT_DARK, row.text))
            elif row.kind == 'body':
                ops.append(_text(MARGIN, cursor_y, 'regular', BODY_SIZE, TEXT_DARK, row.text))
            cursor_y -= _ROW_HEIGHT[row.kind]
        ops.extend(_footer_ops(page_index, page_count))
        pages_ops.append(ops)

    # Fixed object numbering: 1=Catalog 2=Pages 3=Font-regular 4=Font-bold
    # 5=Logo image (only if present), then per page i (0-based):
    # Page object = base+2*i, Content object = base+1+2*i.
    catalog_id, pages_id, font_regular_id, font_bold_id = 1, 2, 3, 4
    image_id = 5 if has_logo else None
    next_id = 6 if has_logo else 5
    page_ids = [next_id + 2 * i for i in range(page_count)]
    content_ids = [next_id + 1 + 2 * i for i in range(page_count)]
    total_objects = (next_id - 1) + 2 * page_count

    object_bodies: List[bytes] = [b''] * (total_objects + 1)  # 1-indexed

    object_bodies[catalog_id] = f'<< /Type /Catalog /Pages {pages_id} 0 R >>'.encode('ascii')
    kids = ' '.join(f'{pid} 0 R' for pid in page_ids)
    object_bodies[pages_id] = (
        f'<< /Type /Pages /Kids [{kids}] /Count {page_count} >>'.encode('ascii')
    )
    object_bodies[font_regular_id] = (
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>'
    )
    object_bodies[font_bold_id] = (
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>'
    )

    jpeg_bytes = b''
    if has_logo:
        with open(_LOGO_PATH, 'rb') as handle:
            jpeg_bytes = handle.read()
        logo_w_px, logo_h_px = _jpeg_size(jpeg_bytes)
        object_bodies[image_id] = (
            f'<< /Type /XObject /Subtype /Image /Width {logo_w_px} /Height {logo_h_px} '
            f'/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode '
            f'/Length {len(jpeg_bytes)} >>\nstream\n'
        ).encode('ascii') + jpeg_bytes + b'\nendstream'

    font_resource = f'/F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R'
    for i, ops in enumerate(pages_ops):
        stream_parts: List[bytes] = []
        for op in ops:
            r, g, b = op.color
            if op.kind == 'rect':
                stream_parts.append(
                    f'{r:.3f} {g:.3f} {b:.3f} rg\n{op.x:.2f} {op.y:.2f} {op.a:.2f} {op.b:.2f} re f'.encode('ascii')
                )
            elif op.kind == 'image':
                stream_parts.append(
                    f'q {op.a:.2f} 0 0 {op.b:.2f} {op.x:.2f} {op.y:.2f} cm /Im1 Do Q'.encode('ascii')
                )
            else:  # text
                font_key = '/F2' if op.font == 'bold' else '/F1'
                stream_parts.append(
                    f'BT\n{r:.3f} {g:.3f} {b:.3f} rg\n{font_key} {op.size:g} Tf\n'
                    f'1 0 0 1 {op.x:.2f} {op.y:.2f} Tm\n'.encode('ascii')
                    + b'(' + _escape(op.text) + b') Tj\nET'
                )
        stream = b'\n'.join(stream_parts)

        content_id = content_ids[i]
        object_bodies[content_id] = (
            f'<< /Length {len(stream)} >>\nstream\n'.encode('ascii') + stream + b'\nendstream'
        )

        page_id = page_ids[i]
        xobject_resource = f' /XObject << /Im1 {image_id} 0 R >>' if (has_logo and i == 0) else ''
        object_bodies[page_id] = (
            f'<< /Type /Page /Parent {pages_id} 0 R '
            f'/MediaBox [0 0 {PAGE_WIDTH:g} {PAGE_HEIGHT:g}] '
            f'/Resources << /Font << {font_resource} >>{xobject_resource} >> '
            f'/Contents {content_id} 0 R >>'
        ).encode('ascii')

    buffer = bytearray()
    buffer += b'%PDF-1.4\n'
    offsets = [0] * (total_objects + 1)
    for obj_id in range(1, total_objects + 1):
        offsets[obj_id] = len(buffer)
        buffer += f'{obj_id} 0 obj\n'.encode('ascii')
        buffer += object_bodies[obj_id]
        buffer += b'\nendobj\n'

    xref_offset = len(buffer)
    buffer += f'xref\n0 {total_objects + 1}\n'.encode('ascii')
    buffer += b'0000000000 65535 f \n'
    for obj_id in range(1, total_objects + 1):
        buffer += f'{offsets[obj_id]:010d} 00000 n \n'.encode('ascii')
    buffer += (
        f'trailer\n<< /Size {total_objects + 1} /Root {catalog_id} 0 R >>\n'
        f'startxref\n{xref_offset}\n%%EOF'
    ).encode('ascii')

    output_folder = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(
        prefix='.automatic-machining-report-', suffix='.tmp', dir=output_folder
    )
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            descriptor = -1
            handle.write(bytes(buffer))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
