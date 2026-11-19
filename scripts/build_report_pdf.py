#!/usr/bin/env python3
"""Typeset the evidence-backed Markdown report without changing scientific inputs.

Run with the existing pinned study environment:
    MPLCONFIGDIR=/tmp/bt4012-mpl .venv/bin/python scripts/build_report_pdf.py
This layout-only builder requires matplotlib and Pillow, already in that environment.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import RendererAgg
from functools import lru_cache
MEASURER = RendererAgg(1, 1, 72)
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
W, H = 595.276, 841.89
LEFT, RIGHT, TOP, BOTTOM = 49, 49, 57, 48
WIDTH = W - LEFT - RIGHT
BODY = 10.5
LEADING = 14.6
INK, BLUE, GREY = '#17232f', '#143b58', '#52616f'


def plain(text):
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
    return text.replace('**', '').replace('`', '').replace('\\_', '_')


@lru_cache(maxsize=32768)
def text_width(text, size=BODY, bold=False):
    if not text:
        return 0
    prop = FontProperties(family='DejaVu Sans', size=size, weight='bold' if bold else 'normal')
    return MEASURER.get_text_width_height_descent(text, prop, ismath=False)[0]


def wrap(text, width, size=BODY, bold=False):
    """Wrap using font measurements, including long identifiers and URLs."""
    words = plain(text).split()
    lines, current = [], ''
    for word in words:
        pieces = [word]
        if text_width(word, size, bold) > width:
            pieces, piece = [], ''
            for char in word:
                if piece and text_width(piece + char, size, bold) > width:
                    pieces.append(piece)
                    piece = char
                else:
                    piece += char
            pieces.append(piece)
        for piece in pieces:
            trial = (current + ' ' + piece).strip()
            if current and text_width(trial, size, bold) > width:
                lines.append(current)
                current = piece
            else:
                current = trial
    if current:
        lines.append(current)
    return lines or ['']


def parse(source):
    lines, out, i = source.splitlines(), [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c.replace(' ', '')) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(('table', rows))
            continue
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            out.append(('heading', (len(m[1]), m[2])))
            i += 1
            continue
        m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', line)
        if m:
            out.append(('image', (m[1], m[2])))
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            out.append(('break', None))
            i += 1
            continue
        if re.match(r'^[-*]\s+', line) or re.match(r'^\d+\.\s+', line):
            out.append(('bullet', re.sub(r'^[-*]\s+', '', line)))
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|[-*]\s|\d+\.\s)', lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        out.append(('paragraph', ' '.join(para)))
    return out


class Report:
    def __init__(self, destination, source_dir):
        self.pdf = PdfPages(destination, metadata={
            'Title': 'Robust Bank Account Fraud Detection under Temporal Drift and Class Imbalance',
            'Author': 'BT4012 project group — author details pending',
            'Subject': 'Evidence-backed submission draft; exploratory consumed-month results',
        })
        self.source_dir = source_dir
        self.page = 0
        self.main_pages = 0
        self.appendix = False
        self.current_section = 'BT4012 Fraud Analytics'
        self.log = []
        self.fig = None
        self.new_page()

    def new_page(self):
        if self.fig is not None:
            self.finish_page()
        self.page += 1
        self.fig = plt.figure(figsize=(W / 72, H / 72), dpi=110, facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis('off')
        self.ax.text(LEFT, 26, 'BT4012  |  FRAUD ANALYTICS', fontsize=8, color=BLUE,
                     fontweight='bold', va='top', fontfamily='DejaVu Sans')
        self.ax.text(W - RIGHT, 26, 'SUBMISSION DRAFT', fontsize=7.5, color=GREY,
                     ha='right', va='top', fontfamily='DejaVu Sans')
        self.ax.plot([LEFT, W - RIGHT], [42, 42], color='#ccd7df', linewidth=.65)
        self.y = TOP
        self.page_elements = []

    def finish_page(self):
        self.ax.plot([LEFT, W - RIGHT], [H - 34, H - 34], color='#ccd7df', linewidth=.65)
        self.ax.text(LEFT, H - 25, self.current_section, fontsize=7, color=GREY, va='top')
        self.ax.text(W - RIGHT, H - 25, str(self.page), fontsize=8, color=GREY, ha='right', va='top')
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.log.append({'page': self.page, 'appendix': self.appendix,
                         'bottom_content_y': round(self.y, 2), 'elements': self.page_elements})
        if not self.appendix:
            self.main_pages += 1
        self.fig = None

    def need(self, height):
        if self.y + height > H - BOTTOM:
            self.new_page()

    def lines(self, lines, size=BODY, leading=LEADING, bold=False, color=INK, x=LEFT):
        for line in lines:
            self.ax.text(x, self.y, line, fontsize=size, color=color, va='top',
                         fontweight='bold' if bold else 'normal', fontfamily='DejaVu Sans',
                         parse_math=False)
            self.y += leading

    def paragraph(self, text, bullet=False):
        indent = 12 if bullet else 0
        lines = wrap(text, WIDTH - indent)
        # Keep at least two lines of a paragraph together at page transitions.
        self.need(min(2, len(lines)) * LEADING + 5)
        if bullet:
            self.ax.text(LEFT + 1, self.y, '•', fontsize=BODY, va='top', color=BLUE)
        for n, line in enumerate(lines):
            if self.y + LEADING > H - BOTTOM:
                self.new_page()
            self.lines([line], x=LEFT + indent)
        self.y += 6.4
        self.page_elements.append('paragraph')

    def heading(self, level, title):
        if level <= 2 and re.match(r'^(References|Appendix)', plain(title), re.I):
            if self.y > TOP + 10:
                self.new_page()
            self.appendix = True
        elif level == 2 and re.match(r'^\d+[.\s]', title) and not (self.page == 1 and title.startswith('1.')):
            if self.y > TOP + 10:
                self.new_page()
        size = 19 if level == 1 else 15 if level == 2 else 11.4
        gap = 11 if level <= 2 else 7
        lines = wrap(title, WIDTH, size, True)
        self.need(len(lines) * (size + 5) + 2 * LEADING + gap)
        if self.y > TOP:
            self.y += gap
        self.lines(lines, size=size, leading=size + 5, bold=True, color=BLUE)
        self.y += gap
        if level == 2:
            self.current_section = plain(title)
        self.page_elements.append('heading: ' + plain(title))

    def table(self, rows):
        if not rows:
            return
        ncols = max(map(len, rows))
        rows = [r + [''] * (ncols - len(r)) for r in rows]
        # Relative weights reduce unnecessary numeric whitespace.
        weights = []
        for c in range(ncols):
            lengths = sorted(text_width(plain(r[c]), 8.5) for r in rows)
            representative = lengths[min(len(lengths) - 1, int(.8 * len(lengths)))]
            weights.append(max(48, min(205, representative + 12)))
        widths = [WIDTH * v / sum(weights) for v in weights]
        size = 8.7 if ncols <= 5 else 8.0
        lead = size + 3.7
        header = rows[0]
        def row_height(row):
            cell_lines = [wrap(c, w - 12, size, False) for c, w in zip(row, widths)]
            return cell_lines, max(map(len, cell_lines)) * lead + 10
        def draw(row, is_header=False):
            cell_lines, height = row_height(row)
            self.need(height)
            fill = '#e8eff4' if is_header else ('#f7f9fb' if draw.counter % 2 else 'white')
            self.ax.add_patch(plt.Rectangle((LEFT, self.y), WIDTH, height, facecolor=fill,
                                            edgecolor='#d6e0e7', linewidth=.4))
            x, start = LEFT, self.y
            for lines, width in zip(cell_lines, widths):
                self.y = start + 5
                self.lines(lines, size=size, leading=lead, bold=is_header, x=x + 6)
                x += width
            self.y = start + height
            draw.counter += 1
        draw.counter = 0
        header_h = row_height(header)[1]
        self.need(header_h + (row_height(rows[1])[1] if len(rows)>1 else 0))
        draw(header, True)
        for row in rows[1:]:
            if self.y + row_height(row)[1] > H - BOTTOM:
                self.new_page()
                draw(header, True)
            draw(row)
        self.y += 10
        self.page_elements.append('table')

    def image(self, caption, reference):
        path = (self.source_dir / reference).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        im = Image.open(path)
        height = min(WIDTH * im.height / im.width, 295)
        width = height * im.width / im.height
        caption_lines = wrap(caption, WIDTH, 8.8)
        self.need(height + len(caption_lines) * 12 + 14)
        self.ax.imshow(im, extent=(LEFT + (WIDTH-width)/2, LEFT + (WIDTH+width)/2,
                                   self.y + height, self.y), aspect='auto')
        self.y += height + 6
        self.lines(caption_lines, size=8.8, leading=12, color=GREY)
        self.y += 12
        self.page_elements.append('figure: ' + str(path.relative_to(ROOT)))

    def close(self):
        if self.fig is not None:
            self.finish_page()
        self.pdf.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'submission/report.md')
    parser.add_argument('--output', type=Path, default=ROOT / 'submission/reportXX-draft.pdf')
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = Report(args.output, args.source.parent)
    for kind, value in parse(args.source.read_text()):
        if kind == 'heading': document.heading(*value)
        elif kind == 'paragraph': document.paragraph(value)
        elif kind == 'bullet': document.paragraph(value, True)
        elif kind == 'table': document.table(value)
        elif kind == 'image': document.image(*value)
        elif kind == 'break': document.new_page()
    document.close()
    result = {'output': str(args.output.relative_to(ROOT)), 'main_pages': document.main_pages,
              'total_pages': document.page, 'body_font_pt': BODY, 'layout': document.log}
    args.output.with_suffix('.layout.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'layout'}))
    if not 8 <= document.main_pages <= 12:
        raise SystemExit('Main report must be 8–12 pages; revise layout/source and rebuild.')
    if any(page['bottom_content_y'] > H - BOTTOM + .1 for page in document.log):
        raise SystemExit('Content exceeds bottom margin.')

if __name__ == '__main__':
    main()
