import asyncio
import ctypes
import re
import sys
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime
import flet as ft
from rapidocr_onnxruntime import RapidOCR

ASSETS = Path(__file__).resolve().parent / "assets"
CURRENCY = "₱"
PHOTO = "[photo]"
CHECKED, UNCHECKED = "[✓]", "[ ]"
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]
CLIP_DIR = Path(tempfile.gettempdir()) / "img2text"
_engine = None

def clipboard_seq():
    return ctypes.windll.user32.GetClipboardSequenceNumber() if sys.platform == "win32" else 0

def make_engine():
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        try:
            onnxruntime.preload_dlls()
            return RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        except Exception:
            pass
    return RapidOCR()

def bounds(box):
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def ring_color(img, x0, y0, x1, y1, near, far):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (max(x0 - far, 0), max(y0 - far, 0)), (min(x1 + far, w - 1), min(y1 + far, h - 1)), 255, -1)
    cv2.rectangle(mask, (max(x0 - near, 0), max(y0 - near, 0)), (min(x1 + near, w - 1), min(y1 + near, h - 1)), 0, -1)
    return np.median(img[mask > 0], axis=0)

def is_chip(img, x0, y0, x1, y1, lh):
    near, far = max(1, round(0.1 * lh)), round(0.75 * lh)
    return np.abs(ring_color(img, x0, y0, x1, y1, near, near + 2) - ring_color(img, x0, y0, x1, y1, far, far + 4)).max() > 18

def word_gap(gaps):
    values = sorted(set(gaps))
    best, best_score = float("inf"), 0
    for t in values[1:]:
        low, high = [g for g in gaps if g < t], [g for g in gaps if g >= t]
        score = len(low) * len(high) * (np.mean(high) - np.mean(low)) ** 2
        if score > best_score and min(high) >= 3 and min(high) >= 2.5 * max(np.mean(low), 1):
            best, best_score = t, score
    return best

def ink_words(gray, x0, y0, x1, y1, gap=None):
    crop = gray[y0:y1 + 1, x0:x1 + 1].astype(np.int16)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    ink = (np.abs(crop - np.median(border)) > 40).any(axis=0)
    cols = np.where(ink)[0]
    if not len(cols):
        return [(x0, x1)]
    if gap is None:
        gap = word_gap([b - a - 1 for a, b in zip(cols, cols[1:]) if b - a > 1])
    words, start = [], cols[0]
    for a, b in zip(cols, cols[1:]):
        if b - a - 1 >= gap:
            words.append((x0 + start, x0 + a))
            start = b
    words.append((x0 + start, x0 + cols[-1]))
    return words

def split_words(item, gray, cw):
    box, text = item[0], item[1]
    char_boxes, chars = (item[3], item[4]) if len(item) > 4 else (None, None)
    x0, y0, x1, y1 = bounds(box)
    if not char_boxes or len(char_boxes) != len(chars):
        return [(x0, x1, text)]
    intervals = ink_words(gray, x0, y0, x1, y1, cw if " " in text.strip() else None)
    groups, last = [], None
    for cb, ch in zip(char_boxes, chars):
        if ch.isspace():
            last = None
            continue
        cx0, _, cx1, _ = bounds(cb)
        center = (cx0 + cx1) / 2
        i = min(range(len(intervals)), key=lambda k: 0 if intervals[k][0] <= center <= intervals[k][1] else min(abs(center - intervals[k][0]), abs(center - intervals[k][1])))
        if i != last:
            groups.append([i, cx0, cx1, ""])
        groups[-1][2] = cx1
        groups[-1][3] += ch
        last = i
    words = []
    for n, (i, cx0, cx1, word) in enumerate(groups):
        first = n == 0 or groups[n - 1][0] != i
        final = n == len(groups) - 1 or groups[n + 1][0] != i
        wx0 = intervals[i][0] if first else cx0
        wx1 = intervals[i][1] if final else cx1
        if first and word[0].isdigit() and wx0 < cx0 - 0.6 * cw:
            word = CURRENCY + word
        words.append((wx0, wx1, word))
    return words

def clean(text):
    text = re.sub(r"(?<![A-Za-z])[PD](?=\d{1,3}(?:[,.]\d{3})*[.,]\d{2}\b)", CURRENCY, text)
    text = re.sub(r"(\d)\s*[-.·•]\s*(?=" + CURRENCY + r")", r"\1 · ", text)
    text = re.sub(r"(\d)\s*([.,:\-/])\s*(?=\d)", r"\1\2", text)
    text = re.sub(r"(\d)\.(\d{3})(?=[.,]\d{2}\b)", r"\1,\2", text)
    text = text.replace("√", "✓")
    return re.sub(r"(\d)\s*[x×�]\s*(?=" + CURRENCY + r"?\d)", r"\1 × ", text)

def is_number(text):
    return bool(re.fullmatch(r"[" + CURRENCY + r"\d.,:/×\- ]+", text))

def recover_glyphs(img, gray, text_boxes, lh):
    background = cv2.medianBlur(gray, max(3, int(lh * 2) | 1))
    ink = (cv2.absdiff(gray, background) > 40).astype(np.uint8)
    pad = round(0.3 * lh)
    for x0, y0, x1, y1 in text_boxes:
        ink[max(y0 - pad, 0):y1 + pad, max(x0 - 2, 0):x1 + 2] = 0
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    found = []
    for x, y, w, h, _ in stats[1:n]:
        if not (0.35 * lh <= h <= 1.2 * lh and w <= 1.5 * lh):
            continue
        cy, cx = y + h // 2, x + w // 2
        y0, y1 = max(cy - round(0.6 * lh), 0), cy + round(0.6 * lh)
        x0, x1 = max(cx - round(0.6 * lh), 0), cx + round(0.6 * lh)
        result, _ = _engine(img[y0:y1, x0:x1], use_det=False, use_cls=False)
        if result and result[0][0].strip() and result[0][1] >= 0.8:
            found.append((x, y, x + w, y + h, result[0][0].strip()))
    return found

def find_checkboxes(gray, lh):
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(edges)
    boxes = []
    for x, y, w, h, _ in stats[1:n]:
        if not (1.1 * lh <= w <= 2 * lh and 1.1 * lh <= h <= 2 * lh and 0.8 < w / h < 1.25):
            continue
        crop = gray[y:y + h, x:x + w].astype(np.int16)
        frame = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
        inner = crop[h // 4:h - h // 4, w // 4:w - w // 4]
        fill = np.median(inner)
        mixed = (np.abs(inner - fill) > 40).mean()
        if frame.std() < 25 and abs(np.median(frame) - fill) > 60 and mixed < 0.5:
            boxes.append((x, y, x + w - 1, y + h - 1, CHECKED if mixed > 0.1 else UNCHECKED))
    return boxes

def find_photos(img, text_boxes, lh):
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    for x0, y0, x1, y1 in text_boxes:
        edges[max(y0 - 4, 0):y1 + 4, max(x0 - 4, 0):x1 + 4] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(edges, np.ones((5, 5), np.uint8)))
    photos = []
    for x, y, w, h, _ in stats[1:n]:
        density = np.count_nonzero(edges[y:y + h, x:x + w]) / (w * h)
        if w >= 1.5 * lh and h >= 1.5 * lh and 0.33 < w / h < 3 and density > 0.08:
            photos.append((x, y, x + w, y + h))
    return photos

def find_rules(img, top, bottom):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    up, down = gray[2:-2] - gray[:-4], gray[2:-2] - gray[4:]
    thin = ((up > 6) & (down > 6)) | ((up < -6) & (down < -6))
    rows = [y + 2 for y in np.where(thin.mean(axis=1) > 0.5)[0] if top < y + 2 < bottom]
    return [y for i, y in enumerate(rows) if i == 0 or y - rows[i - 1] > 3]

def group_lines(words, lh):
    lines = []
    for word in sorted(words):
        if lines and word[0] - lines[-1][0][0] <= 0.6 * lh:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w[3]) for line in lines]

def is_cell_line(line, cw):
    return any(b[3] - a[4] > 2 * cw for a, b in zip(line, line[1:]))

def find_columns(lines, cw, width):
    if len(lines) < 3:
        return []
    coverage = np.zeros(width + 1)
    for line in lines:
        mark = np.zeros(width + 1, bool)
        for w in line:
            mark[w[3]:w[4] + 1] = True
        coverage += mark
    active = np.where(coverage >= max(2, 0.2 * len(lines)))[0]
    columns = []
    for x in active:
        if columns and x - columns[-1][1] <= 0.9 * cw:
            columns[-1][1] = x
        else:
            columns.append([x, x])
    return columns

def column_of(columns, word, cw):
    overlap = lambda c: min(word[4], c[1]) - max(word[3], c[0])
    distance = lambda c: max(c[0] - word[4], word[3] - c[1], 0)
    best = max(range(len(columns)), key=lambda i: (overlap(columns[i]), -distance(columns[i])))
    bridges = sum(overlap(c) > 0.3 * cw for c in columns) > 1
    return best, bridges

def is_mark(word):
    return len(word[5]) == 1 and not word[5].isdigit()

def line_fits(line, columns, cw):
    if not is_cell_line(line, cw):
        return False
    placed = [column_of(columns, w, cw) for w in line]
    if any(bridges for _, bridges in placed):
        return False
    for a, b, (ca, _), (cb, _) in zip(line, line[1:], placed, placed[1:]):
        gap = b[3] - a[4]
        if ca != cb and gap <= cw and not (is_mark(a) or is_mark(b)):
            return False
        if ca == cb and gap > 2 * cw:
            return False
    return True

def plain_row(row_lines, cw):
    parts = []
    for line in row_lines:
        text = line[0][5]
        for a, b in zip(line, line[1:]):
            text += ("   " if b[3] - a[4] > 2 * cw else " ") + b[5]
        parts.append(clean(text))
    return "\n".join(parts)

def render_table(lines, columns, lh, cw):
    rows = []
    for line in lines:
        top, bottom = min(w[1] for w in line), max(w[2] for w in line)
        cell_line = is_cell_line(line, cw)
        if rows and top - rows[-1][1] < 0.25 * lh and (cell_line or rows[-1][2]):
            rows[-1][0].append(line)
            rows[-1][1] = max(rows[-1][1], bottom)
            rows[-1][2] = rows[-1][2] or cell_line
        else:
            rows.append([[line], bottom, cell_line])
    cells = []
    for row_lines, _, cell_line in rows:
        row = [[] for _ in columns]
        fits = cell_line and all(line_fits(line, columns, cw) for line in row_lines if is_cell_line(line, cw))
        for line in row_lines:
            for word in line:
                c, bridges = column_of(columns, word, cw)
                fits = fits and not bridges
                row[c].append(word)
        for ws in row:
            ws.sort(key=lambda w: (w[0], w[3]))
        if fits:
            cells.append([(clean(" ".join(w[5] for w in ws)), min(w[3] for w in ws), max(w[4] for w in ws)) if ws else None for ws in row])
        else:
            cells.append(plain_row(row_lines, cw))
    table_rows = [row for row in cells if not isinstance(row, str)]
    for row in table_rows:
        if any(cell and re.search(r"\d · " + CURRENCY, cell[0]) for cell in row):
            for c, cell in enumerate(row):
                match = cell and re.fullmatch(r"(\d+)\s*[-.·:]?\s*[9PD0]?(\d[\d,]*\.\d{2})", cell[0])
                if match and CURRENCY not in cell[0]:
                    row[c] = (f"{match[1]} · {CURRENCY}{match[2]}", cell[1], cell[2])
    for c in range(len(columns)):
        filled = [row[c][0] for row in table_rows if row[c]]
        numeric = sum(is_number(t) for t in filled) > len(filled) / 2
        marked = [t for t in filled if re.search(r"[A-Za-z]{2,} [,)s:o>?;›»]$", t)]
        for row in table_rows:
            if not row[c]:
                continue
            text = row[c][0]
            if numeric and re.fullmatch(r"[\d.,]+( [\d.,]+)+", text):
                text = text.replace(" ", "")
            if len(marked) >= 3 and len(marked) >= len(filled) / 2:
                text = re.sub(r"([A-Za-z]{2,}) [,)s:o>?;›»]$", r"\1 ›", text)
            if numeric and text in ("。", "O", "o"):
                text = "0"
            row[c] = (text, row[c][1], row[c][2])
        filled = [row[c][0] for row in table_rows if row[c]]
        if numeric and sum(bool(re.fullmatch(r"\d+|\d{1,3}([.,]\d{3})+", t)) for t in filled) > len(filled) / 2:
            for row in table_rows:
                if row[c] and re.fullmatch(r"\d{1,3}(\.\d{3})+", row[c][0]):
                    row[c] = (row[c][0].replace(".", ","), row[c][1], row[c][2])
    used = [c for c in range(len(columns)) if any(row[c] for row in table_rows)]
    widths = {c: max(len(row[c][0]) for row in table_rows if row[c]) for c in used}
    right = {}
    for c in used:
        filled = [row[c] for row in table_rows if row[c]]
        right[c] = len(filled) > 1 and max(f[2] for f in filled) - min(f[2] for f in filled) < max(f[1] for f in filled) - min(f[1] for f in filled)
    out, header_done = [], False
    for row in cells:
        if isinstance(row, str):
            out.append(row)
            continue
        parts = [(row[c][0] if row[c] else "").rjust(widths[c]) if right[c] else (row[c][0] if row[c] else "").ljust(widths[c]) for c in used]
        out.append("  ".join(parts).rstrip())
        texts = [row[c][0] for c in used if row[c]]
        if not header_done and len(texts) >= 3 and not any(is_number(t) for t in texts):
            out.append("─" * (sum(widths.values()) + 2 * (len(used) - 1)))
            header_done = True
    return "\n".join(out)

def render_layout(img, items, words, lines, lh, cw, boxes):
    segments = []
    for item_index, (x0, y0, x1, y1) in items.items():
        own = [w for w in words if w[6] == item_index]
        chip = is_chip(img, x0, y0, x1, y1, lh)
        merged = []
        for w in own:
            if merged and w[3] - merged[-1][2] <= 2.5 * cw:
                merged[-1] = [merged[-1][0], merged[-1][1] + " " + w[5], w[4]]
            else:
                merged.append([w[3], w[5], w[4]])
        for x, text, _ in merged:
            text = clean(text)
            segments.append(((y0 + y1) / 2, x - cw if chip else x, f"[{text}]" if chip else text))
    segments += [(w[0], w[3], w[5]) for w in words if w[6] < 0]
    text_boxes = list(items.values())
    for x0, y0, x1, y1 in find_photos(img, text_boxes + boxes, lh):
        segments.append((y0 + lh / 2, x0, PHOTO))
    lines = []
    for yc, x, text in sorted(segments):
        if lines and yc - lines[-1][0] <= 0.6 * lh:
            lines[-1][1].append((x, text))
        else:
            lines.append([yc, [(x, text)]])
    for line in lines:
        line[1].sort()
    xs = sorted(x for _, segs in lines for x, _ in segs)
    anchor = {}
    for i, x in enumerate(xs):
        anchor[x] = anchor[xs[i - 1]] if i and x - xs[i - 1] < 0.6 * cw else x
    left = xs[0]
    shifts = []
    for _, segs in lines:
        for (px, ptext), (nx, _) in zip(segs, segs[1:]):
            if ptext == PHOTO:
                need = len(PHOTO) + 1 - round((nx - px) / cw)
                if need > 0:
                    shifts.append((nx, need))
    col = lambda x: round((anchor[x] - left) / cw) + sum(n for sx, n in shifts if anchor[x] >= anchor[sx])
    top, bottom = min(y0 for _, y0, _, _ in text_boxes), max(y1 for _, _, _, y1 in text_boxes)
    rules = [y for y in find_rules(img, top, bottom) if not any(y0 - 2 <= y <= y1 + 2 for _, y0, _, y1 in text_boxes)]
    rendered = []
    for yc, segs in lines:
        row = ""
        for x, text in segs:
            c = col(x)
            row += " " * (max(c, len(row) + 1) - len(row)) if row else " " * c
            row += text
        rendered.append((yc, row.rstrip()))
    width = max(len(r) for _, r in rendered)
    rendered += [(y, "─" * width) for y in rules]
    output = [r for _, r in sorted(rendered)]
    return "\n".join(r for i, r in enumerate(output) if not (i and r[:1] == "─" and output[i - 1][:1] == "─"))

def image_to_text(path: Path) -> str:
    global _engine
    if _engine is None:
        _engine = make_engine()
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    result, _ = _engine(img, return_word_box=True)
    if not result:
        return ""
    lh = float(np.median([bounds(item[0])[3] - bounds(item[0])[1] for item in result]))
    dark = np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < 110
    if lh < 16 or dark:
        scale = min(4.0, 36 / lh) if lh < 16 else 1.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if dark:
            img = 255 - img
        result, _ = _engine(img, return_word_box=True)
        if not result:
            return ""
    widths = [bounds(cb)[2] - bounds(cb)[0] for item in result if len(item) > 4 for cb in item[3]]
    cw = max(float(np.median(widths)) if widths else 8.0, 1.0)
    items = {i: bounds(item[0]) for i, item in enumerate(result)}
    lh = float(np.median([y1 - y0 for _, y0, _, y1 in items.values()]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    words = []
    for i, item in enumerate(result):
        x0, y0, x1, y1 = items[i]
        for wx0, wx1, text in split_words(item, gray, cw):
            words.append(((y0 + y1) / 2, y0, y1, wx0, wx1, text, i))
    checkboxes = find_checkboxes(gray, lh)
    boxes = [box[:4] for box in checkboxes]
    centers = [(y0 + y1) / 2 for _, y0, _, y1 in items.values()]
    for x0, y0, x1, y1, text in checkboxes:
        yc = min([c for c in centers if y0 - lh <= c <= y1], key=lambda c: abs(c - y0), default=(y0 + y1) / 2)
        words.append((yc, yc - lh / 2, yc + lh / 2, x0, x1, text, -1))
    for x0, y0, x1, y1, text in recover_glyphs(img, gray, list(items.values()) + boxes, lh):
        words.append(((y0 + y1) / 2, y0, y1, x0, x1, text, -1))
    lines = group_lines(words, lh)
    candidates = [line for line in lines if is_cell_line(line, cw)]
    columns = find_columns(candidates, cw, img.shape[1])
    if columns:
        columns = find_columns([line for line in candidates if line_fits(line, columns, cw)], cw, img.shape[1])
    line_sets = [set() for _ in columns]
    for n, line in enumerate(lines):
        if columns and line_fits(line, columns, cw):
            for word in line:
                if word[5] not in (CHECKED, UNCHECKED):
                    line_sets[column_of(columns, word, cw)[0]].add(n)
    if sum(len(s) >= 3 for s in line_sets) >= 4:
        return render_table(lines, columns, lh, cw)
    return render_layout(img, items, words, lines, lh, cw, boxes)

def card(title, icon, body, actions=None, expand=1):
    header = ft.Row(
        [ft.Icon(icon, size=18, color=ft.Colors.PRIMARY), ft.Text(title, size=15, weight=ft.FontWeight.W_600), ft.Container(expand=True), *(actions or [])],
        spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    return ft.Container(
        content=ft.Column([header, ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT), body], spacing=12, expand=True),
        padding=16, expand=expand, border_radius=16,
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW, border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
    )

async def main(page: ft.Page):
    page.title = "img2text"
    page.window.icon = str(ASSETS / "icon.ico")
    page.padding = 24
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
    page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
    page.window.min_width, page.window.min_height = 820, 560
    page.window.maximized = True
    current = {"path": None, "busy": False, "typing": False}

    preview = ft.Image(src="", visible=False, fit=ft.BoxFit.CONTAIN, border_radius=10)
    placeholder = ft.Column(
        [
            ft.Icon(ft.Icons.ADD_PHOTO_ALTERNATE_OUTLINED, size=56, color=ft.Colors.PRIMARY),
            ft.Text("Click to choose an image", size=15, weight=ft.FontWeight.W_500),
            ft.Text("or copy / snip an image (Win+Shift+S), or press Ctrl+V", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text("PNG, JPG, WEBP or BMP", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER, spacing=6,
    )
    file_name = ft.Text("No image selected", size=12, color=ft.Colors.ON_SURFACE_VARIANT, expand=True, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS)
    status_icon = ft.Icon(ft.Icons.INFO_OUTLINE, size=16, color=ft.Colors.ON_SURFACE_VARIANT)
    status = ft.Text("Pick an image to convert", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
    progress = ft.ProgressRing(visible=False, width=16, height=16, stroke_width=2)
    stats = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
    output = ft.TextField(
        multiline=True, expand=True, min_lines=12, hint_text="Extracted text will appear here",
        border=ft.OutlineInputBorder(border_radius=10, side=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT)), bgcolor=ft.Colors.SURFACE,
        text_style=ft.TextStyle(font_family="Consolas", size=13), on_change=lambda e: refresh_stats(e),
        on_focus=lambda e: set_typing(True), on_blur=lambda e: set_typing(False),
    )

    def set_status(text, icon, color):
        status.value, status.color = text, color
        status_icon.name, status_icon.color = icon, color

    def refresh_stats(e=None):
        text = output.value or ""
        stats.value = f"{len(text.splitlines())} lines · {len(text)} chars" if text else ""
        copy_btn.disabled = save_btn.disabled = not text
        if e:
            page.update()

    def set_output(text):
        output.value = text
        refresh_stats()

    async def convert(path, name=None):
        if current["busy"]:
            return
        current["busy"] = True
        name = name or path.name
        current["path"] = path
        preview.src = str(path)
        preview.visible, placeholder.visible = True, False
        file_name.value = name
        set_status(f"Converting {name}...", ft.Icons.HOURGLASS_TOP, ft.Colors.PRIMARY)
        progress.visible, status_icon.visible = True, False
        pick_btn.disabled = paste_btn.disabled = True
        set_output("")
        page.update()
        try:
            set_output(await asyncio.to_thread(image_to_text, path))
            set_status(f"Done: {name}" if output.value else "No text found", ft.Icons.CHECK_CIRCLE, ft.Colors.GREEN)
        except Exception as ex:
            set_status(f"Error: {ex}", ft.Icons.ERROR_OUTLINE, ft.Colors.ERROR)
        progress.visible, status_icon.visible = False, True
        pick_btn.disabled = paste_btn.disabled = False
        current["busy"] = False
        page.update()

    async def pick(e):
        files = await ft.FilePicker().pick_files(dialog_title="Choose an image", file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=IMAGE_TYPES)
        if files and files[0].path:
            await convert(Path(files[0].path))

    async def paste(e=None, quiet=False):
        clipboard = ft.Clipboard()
        data = await clipboard.get_image()
        if data:
            CLIP_DIR.mkdir(exist_ok=True)
            for old in CLIP_DIR.glob("clipboard-*.png"):
                old.unlink(missing_ok=True)
            path = CLIP_DIR / f"clipboard-{time.strftime('%Y%m%d-%H%M%S')}.png"
            path.write_bytes(data)
            return await convert(path, "Clipboard image")
        images = [Path(f) for f in await clipboard.get_files() if Path(f).suffix.lower().lstrip(".") in IMAGE_TYPES]
        if images:
            return await convert(images[0])
        if not quiet:
            page.show_dialog(ft.SnackBar(ft.Text("No image in the clipboard")))

    async def watch_clipboard():
        seq = clipboard_seq()
        while True:
            await asyncio.sleep(0.5)
            new = clipboard_seq()
            if new != seq:
                seq = new
                if auto_paste.value and not current["busy"]:
                    try:
                        await paste(quiet=True)
                    except Exception:
                        pass

    async def on_key(e: ft.KeyboardEvent):
        if e.ctrl and e.key.upper() == "V" and not current["typing"]:
            await paste()

    def set_typing(value):
        current["typing"] = value

    async def copy(e):
        if output.value:
            await ft.Clipboard().set(output.value)
            page.show_dialog(ft.SnackBar(ft.Text("Copied to clipboard")))

    async def save(e):
        if not output.value:
            return
        name = (current["path"].stem if current["path"] else "output") + ".txt"
        target = await ft.FilePicker().save_file(dialog_title="Save text", file_name=name, file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=["txt"])
        if target:
            Path(target).write_text(output.value, encoding="utf-8")
            page.show_dialog(ft.SnackBar(ft.Text(f"Saved to {target}")))

    def toggle_theme(e):
        dark = page.theme_mode == ft.ThemeMode.LIGHT
        page.theme_mode = ft.ThemeMode.DARK if dark else ft.ThemeMode.LIGHT
        theme_btn.icon = ft.Icons.LIGHT_MODE if dark else ft.Icons.DARK_MODE
        page.update()

    pick_btn = ft.FilledButton("Choose image", icon=ft.Icons.UPLOAD_FILE, on_click=pick)
    paste_btn = ft.OutlinedButton("Paste", icon=ft.Icons.CONTENT_PASTE, on_click=paste, tooltip="Paste image from clipboard (Ctrl+V)")
    auto_paste = ft.Switch(label="Auto-paste copied images", value=True, label_text_style=ft.TextStyle(size=12))
    copy_btn = ft.IconButton(ft.Icons.COPY, tooltip="Copy text", on_click=copy, disabled=True)
    save_btn = ft.IconButton(ft.Icons.SAVE_ALT, tooltip="Save as .txt", on_click=save, disabled=True)
    theme_btn = ft.IconButton(ft.Icons.LIGHT_MODE, tooltip="Toggle theme", on_click=toggle_theme)

    header = ft.Row(
        [
            ft.Image(src="icon.svg", width=42, height=42),
            ft.Column([ft.Text("Image to Text", size=22, weight=ft.FontWeight.BOLD), ft.Text("Extract text, tables and layout from screenshots", size=13, color=ft.Colors.ON_SURFACE_VARIANT)], spacing=0, expand=True),
            theme_btn,
        ],
        spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    drop_zone = ft.Container(
        content=ft.Stack([placeholder, preview], alignment=ft.Alignment.CENTER, expand=True),
        expand=True, padding=12, border_radius=12, on_click=pick, ink=True, alignment=ft.Alignment.CENTER,
        bgcolor=ft.Colors.SURFACE, border=ft.Border.all(1.5, ft.Colors.OUTLINE_VARIANT),
    )
    source = card("Source image", ft.Icons.IMAGE_OUTLINED, ft.Column([drop_zone, ft.Row([file_name, paste_btn, pick_btn], spacing=12)], spacing=12, expand=True), actions=[auto_paste], expand=2)
    result = card(
        "Extracted text", ft.Icons.TEXT_SNIPPET_OUTLINED,
        ft.Column([output, ft.Row([progress, status_icon, status, ft.Container(expand=True), stats], spacing=8)], spacing=10, expand=True),
        actions=[copy_btn, save_btn], expand=3,
    )
    page.add(ft.Column([header, ft.Row([source, result], spacing=20, expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH)], spacing=20, expand=True))
    page.on_keyboard_event = on_key
    page.run_task(watch_clipboard)

if __name__ == "__main__":
    ft.run(main, assets_dir=str(ASSETS))