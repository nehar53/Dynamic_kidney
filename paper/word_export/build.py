import json, re, subprocess, copy
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FONT = 'Times New Roman'

# ------------------------------------------------------------------ reference doc
subprocess.run('pandoc -o ref.docx --print-default-data-file reference.docx', shell=True, check=True)
ref = Document('ref.docx')


def setfont(style, size=None, bold=None, italic=None, color=True):
    f = style.font
    f.name = FONT
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rpr.append(rf)
    for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rf.set(qn(a), FONT)
    for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:cstheme', 'w:eastAsiaTheme'):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    if size: f.size = Pt(size)
    if bold is not None: f.bold = bold
    if italic is not None: f.italic = italic
    if color: f.color.rgb = RGBColor(0, 0, 0)


st = ref.styles


class _S:
    def __init__(self, styles): self.styles = styles
    def __getitem__(self, n):
        for x in self.styles:
            if x.name == n: return x
        raise KeyError(n)


st = _S(ref.styles)
for name in ['Normal', 'Body Text', 'First Paragraph', 'Compact', 'Title', 'Heading 1', 'Heading 2',
             'Heading 3', 'Image Caption', 'Table Caption', 'Captioned Figure', 'Figure', 'Abstract',
             'Block Text', 'Hyperlink']:
    try:
        s = st[name]
    except KeyError:
        continue
    setfont(s, color=(name != 'Hyperlink'))
setfont(st['Normal'], 11)
for name in ['Body Text', 'First Paragraph']:
    s = st[name]
    setfont(s, 11)
    pf = s.paragraph_format
    pf.space_before = Pt(0); pf.space_after = Pt(6); pf.line_spacing = 1.15
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
s = st['Compact']; setfont(s, 8.5); s.paragraph_format.space_after = Pt(0); s.paragraph_format.space_before = Pt(0)
s.paragraph_format.line_spacing = 1.0
s = st['Title']; setfont(s, 20, bold=False); s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
s.paragraph_format.space_after = Pt(18)
# remove the title's bottom border of the default template
ppr = s.element.get_or_add_pPr()
for b in ppr.findall(qn('w:pBdr')):
    ppr.remove(b)
s = st['Heading 1']; setfont(s, 11, bold=False, italic=False)
s.font.small_caps = True
s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
s.paragraph_format.space_before = Pt(14); s.paragraph_format.space_after = Pt(6)
s = st['Heading 2']; setfont(s, 11, bold=False, italic=True)
s.paragraph_format.space_before = Pt(10); s.paragraph_format.space_after = Pt(4)
s = st['Heading 3']; setfont(s, 11, bold=False, italic=True)
s.paragraph_format.space_before = Pt(6); s.paragraph_format.space_after = Pt(3)
s = st['Image Caption']; setfont(s, 9, italic=False)
s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; s.paragraph_format.space_after = Pt(12)
s = st['Table Caption']; setfont(s, 9, italic=False)
s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
s.paragraph_format.space_before = Pt(12); s.paragraph_format.space_after = Pt(4)
s.paragraph_format.keep_with_next = True
s = st['Captioned Figure']; s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
s.paragraph_format.keep_with_next = True
try:
    s = st['Figure']; s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
except KeyError:
    pass
sec = ref.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
    setattr(sec, side, Inches(1))
ref.save('ref.docx')

# ------------------------------------------------------------------ pandoc
subprocess.run('pandoc pre.tex -f latex -t docx --reference-doc ref.docx -o raw.docx', shell=True, check=True)
doc = Document('raw.docx')
body = doc.element.body

# ------------------------------------------------------------------ helpers


def run(p, text, bold=False, italic=False, size=None, sub=False, sup=False):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.name = FONT
    r._element.get_or_add_rPr()
    rf = r._element.rPr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); r._element.rPr.insert(0, rf)
    for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rf.set(qn(a), FONT)
    if size: r.font.size = Pt(size)
    if sub: r.font.subscript = True
    if sup: r.font.superscript = True
    return r


def strip_prefix(p, prefix):
    for r in p.runs:
        if prefix in r.text:
            r.text = r.text.replace(prefix, '', 1).lstrip()
            return


def new_par_after(el, style=None):
    from docx.text.paragraph import Paragraph
    np_ = OxmlElement('w:p')
    el.addnext(np_)
    para = Paragraph(np_, doc._body)
    if style:
        para.style = _S(doc.styles)[style]
    return para


def border(cell_or_tcPr, side, sz):
    tcPr = cell_or_tcPr
    b = tcPr.find(qn('w:tcBorders'))
    if b is None:
        b = OxmlElement('w:tcBorders'); tcPr.append(b)
    e = OxmlElement(f'w:{side}')
    e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), str(sz)); e.set(qn('w:space'), '0'); e.set(qn('w:color'), '000000')
    b.append(e)


# ------------------------------------------------------------------ abstract / keywords / notes
for p in list(doc.paragraphs):
    t = p.text
    if t.startswith('ABSTRACTMARK'):
        strip_prefix(p, 'ABSTRACTMARK')
        for r in p.runs:
            r.bold = True; r.font.size = Pt(10)
        lead = run(p, 'Abstract', bold=True, italic=True, size=10)
        dash = run(p, '—', bold=True, size=10)
        p._p.remove(lead._element); p._p.remove(dash._element)
        first = p._p.find(qn('w:r'))
        first.addprevious(lead._element); first.addprevious(dash._element)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif t.startswith('KEYWORDSMARK'):
        strip_prefix(p, 'KEYWORDSMARK')
        for r in p.runs:
            r.bold = True; r.font.size = Pt(10)
        lead = run(p, 'Index Terms', bold=True, italic=True, size=10)
        dash = run(p, '—', bold=True, size=10)
        p._p.remove(lead._element); p._p.remove(dash._element)
        first = p._p.find(qn('w:r'))
        first.addprevious(lead._element); first.addprevious(dash._element)
        p.paragraph_format.space_after = Pt(12)
    elif t.startswith('TABLENOTE'):
        strip_prefix(p, 'TABLENOTE')
        for r in p.runs:
            r.font.size = Pt(8.5)
        pf = p.paragraph_format
        pf.space_before = Pt(0); pf.space_after = Pt(0); pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.line_spacing = 1.0

# ------------------------------------------------------------------ tables: booktabs look
for tbl in doc.tables:
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = tbl._tbl.tblPr
    for b in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(b)
    rows = tbl.rows
    n = len(rows)
    for i, row in enumerate(rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement('w:cantSplit'); trPr.append(cs)
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for old in tcPr.findall(qn('w:tcBorders')):
                tcPr.remove(old)
            if i == 0:
                border(tcPr, 'top', 12); border(tcPr, 'bottom', 6)
            if i == n - 1:
                border(tcPr, 'bottom', 12)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(8.5)
    # rows that are a group heading (one merged cell): a thin rule above
    for i, row in enumerate(rows):
        if i > 1 and len(row._tr.findall(qn('w:tc'))) == 1:
            tcPr = row._tr.find(qn('w:tc')).get_or_add_tcPr()
            border(tcPr, 'top', 4)
    # keep the table on one page where possible
    for row in rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True

# ---- column widths from content
def set_widths(tbl, maxw=6.5):
    from docx.shared import Emu
    ncol = len(tbl.columns)
    need = [0.0] * ncol
    for row in tbl.rows:
        tcs = row._tr.findall(qn('w:tc'))
        if len(tcs) != ncol:
            continue
        for j, tc in enumerate(tcs):
            txt = ''.join(t.text or '' for t in tc.iter(qn('w:t'))) + ''.join(t.text or '' for t in tc.iter(qn('m:t')))
            need[j] = max(need[j], min(len(txt), 34))
    size = 7.5 if ncol >= 10 else 8.5
    cw = size * 0.62 / 72
    hdr = [max((len(x) for x in ''.join(t.text or '' for t in tc.iter(qn('w:t'))).split()), default=3)
           for tc in tbl.rows[0]._tr.findall(qn('w:tc'))]
    if len(hdr) != ncol:
        hdr = [3] * ncol
    w = [max(n, 3) * cw + 0.24 for n in need]   # inches
    mn = [max(h, 3) * cw * 1.15 + 0.24 for h in hdr]
    while sum(w) > maxw:
        flex = [j for j in range(ncol) if w[j] > mn[j] + 0.01]
        if not flex:
            break
        excess = sum(w) - maxw
        tot_flex = sum(w[j] - mn[j] for j in flex)
        for j in flex:
            w[j] -= min(w[j] - mn[j], excess * (w[j] - mn[j]) / tot_flex)
    tblPr = tbl._tbl.tblPr
    for old in tblPr.findall(qn('w:tblW')):
        tblPr.remove(old)
    tw = OxmlElement('w:tblW'); tw.set(qn('w:w'), str(int(sum(w) * 1440))); tw.set(qn('w:type'), 'dxa')
    tblPr.append(tw)
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
    grid = tbl._tbl.tblGrid
    for j, gc in enumerate(grid.findall(qn('w:gridCol'))):
        gc.set(qn('w:w'), str(int(w[j] * 1440)))
    for row in tbl.rows:
        tcs = row._tr.findall(qn('w:tc'))
        k = 0
        for tc in tcs:
            tcPr = tc.get_or_add_tcPr()
            span = tcPr.find(qn('w:gridSpan'))
            sp = int(span.get(qn('w:val'))) if span is not None else 1
            for old in tcPr.findall(qn('w:tcW')):
                tcPr.remove(old)
            e = OxmlElement('w:tcW'); e.set(qn('w:w'), str(int(sum(w[k:k+sp]) * 1440))); e.set(qn('w:type'), 'dxa')
            tcPr.insert(0, e)
            k += sp


for tbl in doc.tables:
    set_widths(tbl, 9.0 if len(tbl.columns) >= 10 else 6.5)


# ---- math inside tables at the table text size; group rules in Table IV
MNS = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
for tbl in doc.tables:
    hp = '15' if len(tbl.columns) >= 10 else '17'
    for mr in tbl._tbl.iter(MNS + 'r'):
        rpr = mr.find(qn('w:rPr'))
        if rpr is None:
            rpr = OxmlElement('w:rPr')
            t_el = mr.find(MNS + 't')
            t_el.addprevious(rpr)
        for o in rpr.findall(qn('w:sz')):
            rpr.remove(o)
        sz = OxmlElement('w:sz'); sz.set(qn('w:val'), hp); rpr.append(sz)
    if len(tbl.columns) >= 10:
        for i, row in enumerate(tbl.rows):
            first = row.cells[0].text.strip()
            if i > 1 and first:
                for tc in row._tr.findall(qn('w:tc')):
                    border(tc.get_or_add_tcPr(), 'top', 4)
# Table IV (10 columns) is wide: smaller text
big = [t for t in doc.tables if len(t.columns) >= 10]
for t in big:
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(7.5)

# ------------------------------------------------------------------ algorithm
ALG = [
    (0, [('b', 'Require: '), ('', 'initial pool '), ('i', 'V'), ('sub', '1'), ('', '; policy '), ('i', 'π'),
         ('', ' ∈ {'), ('i', 'π'), ('sub', 'M'), ('', ', '), ('i', 'π'), ('sub', 'B'), ('', '}; bandit '),
         ('i', 'B'), ('', '; rounds '), ('i', 'T'), ('', '; population size '), ('i', 'N'), ('', '; generations '),
         ('i', 'G'), ('sub', 'max')]),
    (0, [('', '1:  '), ('i', 'F'), ('sub', '0'), ('', ' ← ∅'), ('c', '    ▷ previous front')]),
    (0, [('', '2:  '), ('b', 'for '), ('i', 't'), ('', ' = 1, …, '), ('i', 'T'), ('b', ' do')]),
    (1, [('', '3:  '), ('i', 'M'), ('sub', 't'), ('', ' ← ∅')]),
    (1, [('', '4:  '), ('b', 'if '), ('i', 'π'), ('', ' clears in round '), ('i', 't'), ('b', ' then')]),
    (2, [('', '5:  build '), ('i', 'G'), ('sub', 't'), ('', ' and the structure list '), ('i', 'K'), ('sub', 't')]),
    (2, [('', '6:  ('), ('i', 'M'), ('sub', 't'), ('sup', 'ILP'), ('', ', '), ('i', 'F'), ('sub', '1'), ('sup', 'ILP'),
         ('', '('), ('i', 't'), ('', ')) ← solve (9)')]),
    (2, [('', '7:  '), ('b', 'if '), ('i', 'K'), ('sub', 't'), ('', ' admits at most 2'), ('i', 'N'),
         ('', ' matchings'), ('b', ' then')]),
    (3, [('', '8:  '), ('i', 'F'), ('sub', 't'), ('', ' ← exact Pareto front by enumeration')]),
    (2, [('', '9:  '), ('b', 'else')]),
    (3, [('', '10: '), ('i', 'Q'), ('', ' ← {'), ('i', 'M'), ('sub', 't'), ('sup', 'ILP'), ('', '} ∪ '), ('i', 'F'),
         ('sub', 't−1'), ('', ' ∪ random individuals')]),
    (3, [('', '11: '), ('b', 'for '), ('i', 'g'), ('', ' = 1, …, '), ('i', 'G'), ('sub', 'max'), ('b', ' do')]),
    (4, [('', '12: '), ('i', 'p'), ('sub', 'm'), ('', ' ← '), ('i', 'B'), ('', '.select('), ('b', 'x'), ('sub', 't'),
         ('', ')')]),
    (4, [('', '13: create offspring by uniform crossover and bit-flip mutation with rate '), ('i', 'p'),
         ('sub', 'm')]),
    (4, [('', '14: repair; apply NSGA-III environmental selection')]),
    (4, [('', '15: update the archive; '), ('i', 'B'), ('', '.update('), ('i', 'r'), ('sub', 'g'), ('', ')')]),
    (3, [('', '16: '), ('b', 'end for')]),
    (3, [('', '17: '), ('i', 'F'), ('sub', 't'), ('', ' ← archive')]),
    (2, [('', '18: '), ('b', 'end if')]),
    (2, [('', '19: '), ('i', 'M'), ('sub', 't'), ('', ' ← select('), ('i', 'F'), ('sub', 't'), ('', '); record PoF'),
         ('sub', 't')]),
    (1, [('', '20: '), ('b', 'end if')]),
    (1, [('', '21: '), ('i', 'V'), ('sub', 't+1'), ('', ' ← update the pool by (1)')]),
    (0, [('', '22: '), ('b', 'end for')]),
]
for p in list(doc.paragraphs):
    if p.text.strip() == 'ALGORITHMMARK':
        cap = p
        for r in list(cap.runs):
            r._element.getparent().remove(r._element)
        cap.style = _S(doc.styles)['Body Text']
        run(cap, 'Algorithm 1', bold=True, size=10); run(cap, '  One dynamic trajectory', size=10)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        cap.paragraph_format.space_before = Pt(10); cap.paragraph_format.space_after = Pt(2)
        cap.paragraph_format.keep_with_next = True
        t = doc.add_table(rows=1, cols=1)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = t.rows[0].cells[0]
        cell.width = Inches(6.5)
        tcPr = cell._tc.get_or_add_tcPr()
        border(tcPr, 'top', 8); border(tcPr, 'bottom', 8)
        first = True
        for lvl, segs in ALG:
            para = cell.paragraphs[0] if first else cell.add_paragraph()
            first = False
            para.paragraph_format.left_indent = Inches(0.3 * lvl + (0 if segs[0][0] == 'b' else 0.0))
            para.paragraph_format.space_after = Pt(1); para.paragraph_format.space_before = Pt(1)
            para.paragraph_format.keep_with_next = True
            for kind, txt in segs:
                if kind == 'c':
                    run(para, txt, size=9.5)
                else:
                    run(para, txt, bold=(kind == 'b'), italic=(kind == 'i'), size=9.5,
                        sub=(kind == 'sub'), sup=(kind == 'sup'))
        cap._p.addnext(t._tbl)
        sp = new_par_after(t._tbl, 'Body Text')
        break

# ------------------------------------------------------------------ references
refs = json.load(open('refs.json', encoding='utf-8'))
fix = {'branch-andprice': 'branch-and-price', 'cardinalityconstrained': 'cardinality-constrained',
       'Qnetwork': 'Q-network', 'fairnessaware': 'fairness-aware', '– ': '–'}
refs = [re.sub(r'(\d)– (\d)', r'\1–\2', r) for r in refs]
for i, r in enumerate(refs):
    for a, b in fix.items():
        r = r.replace(a, b)
    refs[i] = r

ital = json.load(open('italics.json', encoding='utf-8'))
order = json.load(open('cite_order.json'))
for p in list(doc.paragraphs):
    if p.text.strip() == 'REFERENCESMARK':
        for r in list(p.runs):
            r._element.getparent().remove(r._element)
        p.style = _S(doc.styles)['Heading 1']
        run(p, 'References')
        prev = p._p
        for n, (key, text) in enumerate(zip(order, refs), 1):
            para = new_par_after(prev, 'Body Text')
            prev = para._p
            pf = para.paragraph_format
            pf.left_indent = Inches(0.4); pf.first_line_indent = Inches(-0.4)
            pf.space_after = Pt(3); pf.line_spacing = 1.0; pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.tab_stops.add_tab_stop(Inches(0.4))
            run(para, f'[{n}]\t', size=9.5)
            it = ital.get(key)
            if it and it in text:
                a = text.index(it, text.index('”'))
                run(para, text[:a], size=9.5); run(para, it, italic=True, size=9.5)
                run(para, text[a + len(it):], size=9.5)
            else:
                if it:
                    print('ITALIC NOT FOUND', n, key, it)
                run(para, text, size=9.5)
        break


# ------------------------------------------------------------------ numbered display equations
from docx.text.paragraph import Paragraph
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
for p_el in list(body.iter(qn('w:p'))):
    mp = p_el.find(M + 'oMathPara')
    if mp is None:
        continue
    kids = [c for c in p_el if c.tag != qn('w:pPr')]
    k = kids.index(mp)
    before, after = kids[:k], kids[k + 1:]
    num = None
    for c in after:
        for t in c.iter(qn('w:t')):
            m = re.search(r'EQNO(\d+)QE', t.text or '')
            if m:
                num = m.group(1); t.text = (t.text[:m.start()] + t.text[m.end():])
    if num is None:
        nxt = p_el.getnext()
        if nxt is not None and nxt.tag == qn('w:p'):
            ts = list(nxt.iter(qn('w:t')))
            for t in ts:
                m = re.search(r'EQNO(\d+)QE', t.text or '')
                if m:
                    num = m.group(1); t.text = (t.text[:m.start()] + t.text[m.end():])
                    break
            # drop leading blank runs; remove the paragraph if nothing is left
            for r_ in list(nxt.findall(qn('w:r'))):
                tt = ''.join(x.text or '' for x in r_.iter(qn('w:t')))
                if tt.strip() == '':
                    nxt.remove(r_)
                else:
                    break
            if not any((x.text or '').strip() for x in nxt.iter(qn('w:t'))) and nxt.find('.//' + M + 'oMath') is None:
                nxt.getparent().remove(nxt)
    pPr = p_el.find(qn('w:pPr'))

    def mkpar(children):
        np_ = OxmlElement('w:p')
        if pPr is not None:
            np_.append(copy.deepcopy(pPr))
        for c in children:
            np_.append(c)
        return np_

    has_text = lambda ch: any((t.text or '').strip() for c in ch for t in c.iter(qn('w:t'))) or \
        any(c.tag == M + 'oMath' for c in ch)
    tbl = doc.add_table(rows=1, cols=3)._tbl
    widths = (0.55, 5.4, 0.55)
    grid = tbl.tblGrid
    for j, gc in enumerate(grid.findall(qn('w:gridCol'))):
        gc.set(qn('w:w'), str(int(widths[j] * 1440)))
    tblPr = tbl.tblPr
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
    jc = OxmlElement('w:jc'); jc.set(qn('w:val'), 'center'); tblPr.append(jc)
    tcs = tbl.findall('.//' + qn('w:tc'))
    for j, tc in enumerate(tcs):
        tcPr = tc.get_or_add_tcPr()
        for o in tcPr.findall(qn('w:tcW')):
            tcPr.remove(o)
        e = OxmlElement('w:tcW'); e.set(qn('w:w'), str(int(widths[j] * 1440))); e.set(qn('w:type'), 'dxa'); tcPr.append(e)
        va = OxmlElement('w:vAlign'); va.set(qn('w:val'), 'center'); tcPr.append(va)
    mid = Paragraph(tcs[1].find(qn('w:p')), doc._body)
    mid.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mid.paragraph_format.space_before = Pt(3); mid.paragraph_format.space_after = Pt(3)
    mid._p.append(mp)
    right = Paragraph(tcs[2].find(qn('w:p')), doc._body)
    right.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if num:
        run(right, f'({num})', size=11)
    anchor = p_el
    if has_text(before):
        nb = mkpar(before); p_el.addprevious(nb)
    p_el.addprevious(tbl)
    if has_text(after):
        na = mkpar(after)
        # the text after an equation continues the paragraph: no first-line indent
        p_el.addprevious(na)
    p_el.getparent().remove(p_el)


# ------------------------------------------------------------------ Table IV on a landscape page
body_sect = body.find(qn('w:sectPr'))
for tbl in [t for t in doc.tables if len(t.columns) >= 10]:
    cap = tbl._tbl.getprevious()
    pre = OxmlElement('w:p'); cap.addprevious(pre)
    ppr = OxmlElement('w:pPr'); pre.append(ppr)
    ppr.append(copy.deepcopy(body_sect))
    post = OxmlElement('w:p'); tbl._tbl.addnext(post)
    ppr2 = OxmlElement('w:pPr'); post.append(ppr2)
    ls = copy.deepcopy(body_sect)
    pg = ls.find(qn('w:pgSz'))
    w_, h_ = pg.get(qn('w:w')), pg.get(qn('w:h'))
    pg.set(qn('w:w'), h_); pg.set(qn('w:h'), w_); pg.set(qn('w:orient'), 'landscape')
    ppr2.append(ls)

# ------------------------------------------------------------------ schema order (Word is strict)
ORD = {
    'tblPr': ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize', 'tblStyleColBandSize',
              'tblW', 'jc', 'tblCellSpacing', 'tblInd', 'tblBorders', 'shd', 'tblLayout', 'tblCellMar',
              'tblLook', 'tblCaption', 'tblDescription'],
    'tcPr': ['cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge', 'tcBorders', 'shd', 'noWrap', 'tcMar',
             'textDirection', 'tcFitText', 'vAlign', 'hideMark'],
    'tcBorders': ['top', 'start', 'left', 'bottom', 'end', 'right', 'insideH', 'insideV', 'tl2br', 'tr2bl'],
}
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
for tag, order_ in ORD.items():
    for el in body.iter(W + tag):
        kids = list(el)
        # merge duplicate tcBorders / keep last of duplicated singletons
        if tag == 'tcPr':
            bs = el.findall(W + 'tcBorders')
            for extra in bs[1:]:
                for c in list(extra):
                    bs[0].append(c)
                el.remove(extra)
            kids = list(el)
        seen = {}
        for c in kids:
            nm = c.tag.replace(W, '')
            seen[nm] = c          # last one wins for duplicates
        for c in kids:
            el.remove(c)
        for nm in order_:
            if nm in seen:
                el.append(seen[nm])
        for nm, c in seen.items():
            if nm not in order_:
                el.append(c)

# ------------------------------------------------------------------ page numbers, figures
footer = doc.sections[0].footer
fp = footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run()
for tag, txt in (('begin', None), (None, 'PAGE'), ('end', None)):
    if tag:
        fc = OxmlElement('w:fldChar'); fc.set(qn('w:fldCharType'), tag); r._r.append(fc)
    else:
        it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = txt; r._r.append(it)
r.font.size = Pt(9); r.font.name = FONT

# no empty paragraph before the title, and doc properties anonymous
doc.core_properties.author = ''
doc.core_properties.last_modified_by = ''
doc.core_properties.title = 'Dynamic Kidney Exchange Optimization Under Operational Constraints for Equitable Allocation'
doc.save('Dynamic_Kidney_Exchange_Manuscript_v3.docx')
print('saved')
