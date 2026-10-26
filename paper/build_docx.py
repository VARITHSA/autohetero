"""main.tex -> main.docx in the IEEE conference layout of main.pdf (two columns, Times 10 pt, roman headings).

    tectonic main.tex && python paper/build_docx.py      # needs pandoc, python-docx and pymupdf

Pandoc converts the text, tables and equations; the system diagram and the algorithms are cropped from main.pdf
(Word has no TikZ or algorithmic), and the figures are rasterised from fig/*.pdf. The rest is styling.
"""
import copy
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import docx
import pymupdf
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
COLUMN, WIDE = 3.45, 7.0  # image widths in inches
AUTHORS = ["Srivathsa B S", "Manas Verma", "Pavithra N"]
AFFILIATION = ["Manipal Institute of Technology, Bengaluru", "Manipal Academy of Higher Education", "Manipal, India"]
FIGS = ["arch", "crossover", "rq1", "confidence", "rq2", "hybrid"]  # in source order


def roman(n):
    return ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"][n]


# ---------------------------------------------------------------- images


def crop_images(pdf, out):
    """The system diagram and the three algorithms, cut from the compiled PDF at 300 dpi."""
    d = pymupdf.open(pdf)
    for page in d:
        hit = page.search_for("Fig. 1.")
        if hit:  # the diagram: every drawing in its column above the caption
            cap = hit[0]
            rs = [dr["rect"] for dr in page.get_drawings() if abs(dr["rect"].x0 - cap.x0) < 260 and dr["rect"].y1 < cap.y0 and dr["rect"].x0 >= cap.x0 - 5]
            u = rs[0]
            for r in rs:
                u |= r
            page.get_pixmap(dpi=300, clip=pymupdf.Rect(u.x0 - 4, u.y0 - 4, u.x1 + 4, u.y1 + 4)).save(out / "arch.png")
            break
    for n in (1, 2, 3):
        for page in d:
            rules = sorted((dr["rect"] for dr in page.get_drawings() if dr["rect"].height < 1.5 and dr["rect"].width > 200), key=lambda r: r.y0)
            found = None
            for cap in page.search_for(f"Algorithm {n} "):
                col = [r for r in rules if abs(r.x0 - cap.x0) < 5]
                above = [r for r in col if 0 <= cap.y0 - r.y0 < 6]  # a float's caption sits right under its top rule
                if above:
                    below = [r for r in col if r.y0 > cap.y0]
                    found = (above[0], below[1])  # rule under the caption, then the closing rule
                    break
            if found:
                top, bottom = found
                page.get_pixmap(dpi=300, clip=pymupdf.Rect(top.x0 - 2, top.y0 - 2, top.x1 + 2, bottom.y0 + 2)).save(out / f"alg{n}.png")
                break
        else:
            sys.exit(f"Algorithm {n} not found in {pdf}")
    for name in FIGS[1:]:
        pymupdf.open(HERE / "fig" / f"{name}.pdf")[0].get_pixmap(dpi=300).save(out / f"{name}.png")


# ---------------------------------------------------------------- LaTeX for pandoc


def to_pandoc_tex(s, img):
    """main.tex with numbering, citations and cross-references resolved, ready for pandoc."""
    keys = re.findall(r"\\bibitem\{([^}]+)\}", s)
    cite = {k: i + 1 for i, k in enumerate(keys)}
    s = re.sub(r"~?\\cite\{([^}]+)\}", lambda m: "~[" + ", ".join(str(cite[k.strip()]) for k in m.group(1).split(",")) + "]", s)

    # numbers for every label, in source order
    labels, counters, last = {}, {"sec": 0, "sub": 0, "fig": 0, "tab": 0, "alg": 0, "eq": 0}, None
    for m in re.finditer(r"\\section\{|\\subsection\{|\\begin\{(figure\*?|table\*?|algorithm|equation)\}|\\label\{([^}]+)\}", s):
        t = m.group(0)
        if t.startswith("\\section"):
            counters["sec"] += 1; counters["sub"] = 0; last = roman(counters["sec"])
        elif t.startswith("\\subsection"):
            counters["sub"] += 1; last = "ABCDEFGHIJ"[counters["sub"] - 1]
        elif m.group(1):
            kind = {"figure": "fig", "table": "tab", "algorithm": "alg", "equation": "eq"}[m.group(1).rstrip("*")]
            counters[kind] += 1
            last = roman(counters[kind]) if kind == "tab" else str(counters[kind])
        else:
            labels[m.group(2)] = last
    s = re.sub(r"\\eqref\{([^}]+)\}", lambda m: f"({labels[m.group(1)]})", s)
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m.group(1)], s)

    # headings and captions carry their numbers as text
    state = {"sec": 0, "sub": 0, "env": None, "fig": 0, "tab": 0}

    def number(m):
        t = m.group(0)
        if t.startswith("\\section{"):
            state["sec"] += 1; state["sub"] = 0
            return f"\\section*{{{roman(state['sec'])}. {m.group(1)}}}"
        if t.startswith("\\subsection{"):
            state["sub"] += 1
            return f"\\subsection*{{{'ABCDEFGHIJ'[state['sub'] - 1]}. {m.group(2)}}}"
        if t.startswith("\\begin"):
            state["env"] = m.group(3).rstrip("*")
            return t
        if state["env"] == "figure":
            state["fig"] += 1
            return f"\\caption{{Fig. {state['fig']}. {m.group(4)}}}"
        if state["env"] == "table":
            state["tab"] += 1
            return f"\\caption{{TABLE {roman(state['tab'])}|{m.group(4)}}}"
        return t

    s = re.sub(r"\\section\{([^}]*)\}|\\subsection\{([^}]*)\}|\\begin\{(figure\*?|table\*?|algorithm)\}|\\caption\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
               number, s)

    # pieces Word can't draw become images
    s = re.sub(r"\\resizebox\{\\columnwidth\}\{!\}\{\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}\}", r"\\includegraphics{arch}", s, flags=re.S)
    n = iter(range(1, 10))
    s = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", lambda m: f"\\begin{{figure}}\n\\includegraphics{{alg{next(n)}}}\n\\end{{figure}}", s, flags=re.S)
    s = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([a-z0-9]+)\}", lambda m: f"\\includegraphics{{{img}/{m.group(2)}.png}}", s)

    # equations: one block, number at the right
    def eq(m):
        body = re.sub(r"\\label\{[^}]*\}", "", m.group(1))
        body = re.sub(r",?\s*\n\s*\n", "", body).strip().rstrip(",").replace("\\,", " ").replace("\\!", "")
        return body

    eqn = iter(range(1, 10))
    s = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", lambda m: f"\\begin{{equation}}\n{eq(m)} \\qquad ({next(eqn)})\n\\end{{equation}}", s, flags=re.S)

    # small things pandoc doesn't know
    s = s.replace("{table*}", "{table}").replace("{figure*}", "{figure}")  # width is set in Word
    s = re.sub(r"\\label\{[^}]*\}|\\renewcommand\{\\arraystretch\}\{[^}]*\}|\\setlength\{\\tabcolsep\}\{[^}]*\}", "", s)  # resolved above; pandoc trips on them
    s = s.replace("\\cmark", "✓").replace("\\xmark", "✗").replace("\\textmu ", "µ").replace("\\textmu", "µ")
    s = re.sub(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\end\{tabular\}\\\\\[2pt\]\s*\\footnotesize\{(.*?)\}\n", r"\\end{tabular}\n\n\\noindent NOTE: \1\n\n", s)
    s = re.sub(r"\\begin\{abstract\}\s*(.*?)\\end\{abstract\}", r"\\noindent ABSTRACT: \1", s, flags=re.S)
    s = re.sub(r"\\begin\{IEEEkeywords\}\s*(.*?)\\end\{IEEEkeywords\}", r"\\noindent KEYWORDS: \1", s, flags=re.S)
    refs = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", s, flags=re.S)
    items = [t.strip() for t in re.split(r"\\bibitem\{[^}]+\}", refs.group(1))[1:]]
    s = s[: refs.start()] + "\\section*{References}\n\n" + "\n\n".join(f"\\noindent REF[{i}] {t}" for i, t in enumerate(items, 1)) + "\n" + s[refs.end():]
    s = re.sub(r"\\title\{.*?\}\n|\\author\{.*?\n(?=\\maketitle)|\\maketitle", "", s, flags=re.S)
    _, body = s.split("\\begin{document}", 1)
    return "\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\begin{document}" + body



# ---------------------------------------------------------------- Word styling


def sect_pr(cols):
    """Section properties: US Letter, IEEE margins, continuous start, `cols` columns."""
    s = OxmlElement("w:sectPr")
    for tag, attrs in (("w:type", {"w:val": "continuous"}),
                       ("w:pgSz", {"w:w": "12240", "w:h": "15840"}),
                       ("w:pgMar", {"w:top": "1080", "w:right": "893", "w:bottom": "1440", "w:left": "893", "w:header": "720", "w:footer": "720", "w:gutter": "0"}),
                       ("w:cols", {"w:num": str(cols), "w:space": "259"})):
        e = OxmlElement(tag)
        for k, v in attrs.items():
            e.set(qn(k), v)
        s.append(e)
    return s


def end_section(paragraph, cols):
    """Make `paragraph` the last one of a section with `cols` columns."""
    paragraph._p.get_or_add_pPr().append(sect_pr(cols))


def times(style_or_run, size, bold=None, italic=None, caps=None):
    f = style_or_run.font
    f.name, f.size, f.color.rgb = "Times New Roman", Pt(size), RGBColor(0, 0, 0)
    rpr = style_or_run.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), "Times New Roman")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        fonts.attrib.pop(qn(a), None)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if caps is not None:
        f.small_caps = caps


def spacing(style, before=0, after=0, indent=None, align=None, hanging=None):
    pf = style.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), 1.0
    if indent is not None:
        pf.first_line_indent = Inches(indent)
    if hanging is not None:
        pf.left_indent, pf.first_line_indent = Inches(hanging), Inches(-hanging)
    if align is not None:
        pf.alignment = align


def style_document(d):
    st = d.styles
    times(st["Normal"], 10)
    for name in ("Body Text", "First Paragraph"):
        times(st[name], 10)
        spacing(st[name], indent=0.14, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    times(st["Compact"], 10)
    spacing(st["Compact"], align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    times(st["Heading 1"], 10, bold=False, italic=False, caps=True)
    spacing(st["Heading 1"], before=8, after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    times(st["Heading 2"], 10, bold=False, italic=True, caps=False)
    spacing(st["Heading 2"], before=6, after=3, align=WD_ALIGN_PARAGRAPH.LEFT)
    for name in ("Heading 1", "Heading 2"):
        st[name].paragraph_format.keep_with_next = True
    for name in ("Image Caption", "Table Caption"):
        times(st[name], 8, italic=False)
    spacing(st["Image Caption"], before=3, after=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    spacing(st["Table Caption"], before=6, after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    code = st["Verbatim Char"]
    code.font.name, code.font.size = "Courier New", Pt(8.5)
    code.element.get_or_add_rPr().find(qn("w:rFonts")).set(qn("w:hAnsi"), "Courier New")
    link = st["Hyperlink"]
    link.font.size, link.font.color.rgb = Pt(8), RGBColor(0, 0, 0)
    for name in ("Captioned Figure", "Figure"):
        spacing(st[name], before=6, after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
        st[name].paragraph_format.keep_with_next = True


def strip_marker(p, marker):
    """Remove a leading text marker from paragraph p (it can span runs)."""
    left = marker
    for r in p.runs:
        if not left:
            break
        take = min(len(left), len(r.text))
        if r.text[:take] == left[:take]:
            r.text, left = r.text[take:], left[take:]
        else:
            break
    if p.runs and p.runs[0].text.startswith(" "):
        p.runs[0].text = p.runs[0].text.lstrip()


def lead(p, text, italic=True, bold=False, size=None):
    """Insert a run at the start of p."""
    r = p.add_run(text)
    r.italic, r.bold = italic, bold
    if size:
        r.font.size = Pt(size)
    p._p.insert(1 if p._p.pPr is not None else 0, r._r)
    return r


def front_matter(d):
    """Title and the three author blocks, full width, before the abstract."""
    first = next(p for p in d.paragraphs if p.text.startswith("ABSTRACT:"))
    title = first.insert_paragraph_before()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(14)
    times(title.add_run("Trust or Measure? Confidence-Gated CPU–GPU Execution Selection"), 24)
    t = d.add_table(rows=1, cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, name in zip(t.rows[0].cells, AUTHORS):
        cell.width = Inches(2.4)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)
        times(p.add_run(name), 11)
        for i, line in enumerate(AFFILIATION):
            q = cell.add_paragraph()
            q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.space_after = Pt(0)
            times(q.add_run(line), 9, italic=i < 2)
        q.paragraph_format.space_after = Pt(16)  # Word ignores spacing on the section-break paragraph below
    title._p.addnext(t._tbl)
    gap = first.insert_paragraph_before()
    gap.paragraph_format.space_before = Pt(10)
    gap.paragraph_format.space_after = Pt(10)
    end_section(gap, 1)


def mark_paragraphs(d):
    """Abstract, index terms, the table note and references, from the markers to_pandoc_tex left."""
    for p in d.paragraphs:
        t = p.text
        if t.startswith("ABSTRACT:"):
            strip_marker(p, "ABSTRACT:")
            for r in p.runs:
                r.bold, r.font.size = True, Pt(9)
            lead(p, "Abstract", bold=True, size=9).font.italic = True
            _dash(p, 9)
            spacing_p(p, indent=0.14)
        elif t.startswith("KEYWORDS:"):
            strip_marker(p, "KEYWORDS:")
            for r in p.runs:
                r.bold, r.font.size = True, Pt(9)
            lead(p, "Index Terms", bold=True, size=9)
            _dash(p, 9)
            spacing_p(p, indent=0.14, after=6)
        elif t.startswith("NOTE:"):
            strip_marker(p, "NOTE:")
            for r in p.runs:
                r.font.size = Pt(8)
            spacing_p(p, after=6)
        elif t.startswith("REF["):
            num = t[4: t.index("]")]
            strip_marker(p, f"REF[{num}]")
            for r in p.runs:
                r.font.size = Pt(8)
            lead(p, f"[{num}]\t", italic=False, size=8)
            pf = p.paragraph_format
            pf.left_indent, pf.first_line_indent, pf.space_after = Inches(0.3), Inches(-0.3), Pt(1)
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def _dash(p, size):
    """Em dash right after the lead-in run."""
    r = p.add_run("—")
    r.bold, r.italic, r.font.size = True, True, Pt(size)
    p._p.insert(list(p._p).index(p.runs[0]._r) + 1, r._r)


def spacing_p(p, indent=0.0, after=0):
    pf = p.paragraph_format
    pf.first_line_indent, pf.space_after, pf.space_before = Inches(indent), Pt(after), Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def table_captions(d):
    """'TABLE I|Title' -> 'TABLE I' on one line, the title in small caps below it, as IEEE sets them."""
    for p in d.paragraphs:
        if p.style.name == "Table Caption" and "|" in p.text:
            head, title = p.text.split("|", 1)
            for r in list(p.runs):
                r._r.getparent().remove(r._r)
            times(p.add_run(head), 8)
            p.runs[-1].add_break(WD_BREAK.LINE)
            times(p.add_run(title), 8, caps=True)


def style_tables(d):
    """Booktabs look: rules above, below and under the header; 8 pt text; no other borders."""
    for t in d.tables[1:]:  # the first is the author block
        tblpr = t._tbl.tblPr
        for old in tblpr.findall(qn("w:tblBorders")):
            tblpr.remove(old)
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single" if side in ("top", "bottom") else "nil")
            if side in ("top", "bottom"):
                e.set(qn("w:sz"), "8"); e.set(qn("w:color"), "000000")
            borders.append(e)
        tblpr.append(borders)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell in t.rows[0].cells:
            tcpr = cell._tc.get_or_add_tcPr()
            b = OxmlElement("w:tcBorders")
            e = OxmlElement("w:bottom")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
            b.append(e)
            tcpr.append(b)
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.first_line_indent = Inches(0)
                    for r in p.runs:
                        r.font.size, r.font.name = Pt(8), "Times New Roman"
            for mr in row._tr.iter(qn("m:r")):  # equations in cells keep their own run properties
                rpr = mr.find(qn("w:rPr"))
                if rpr is None:
                    rpr = OxmlElement("w:rPr")
                    mr.insert(0 if mr.find(qn("m:rPr")) is None else 1, rpr)
                sz = OxmlElement("w:sz")
                sz.set(qn("w:val"), "16")
                rpr.append(sz)
    _no_borders(d.tables[0])


def _no_borders(t):
    tblpr = t._tbl.tblPr
    for old in tblpr.findall(qn("w:tblBorders")) + tblpr.findall(qn("w:tblStyle")):
        tblpr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "nil")
        borders.append(e)
    tblpr.append(borders)
    look = tblpr.find(qn("w:tblLook"))
    if look is not None:
        look.set(qn("w:firstRow"), "0")  # the default table style rules off a header row
    for tc in t._tbl.iter(qn("w:tc")):
        b = OxmlElement("w:tcBorders")
        for side in ("top", "bottom"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "nil")
            b.append(e)
        tc.get_or_add_tcPr().append(b)


def size_images(d):
    """Column-wide images, except the crossover figure, which spans the page as in the PDF."""
    for shape, name in zip(d.inline_shapes, ["arch", "alg1", "alg2", "alg3", "crossover", "rq1", "confidence", "rq2", "hybrid"]):
        w = WIDE if name == "crossover" else COLUMN
        ratio = shape.height / shape.width
        shape.width, shape.height = Inches(w), Inches(w * ratio)


def wide_blocks(d):
    """Put the crossover figure and the claim map (TABLE IV) in one-column sections, like figure* and table*."""
    body = d.element.body
    paras = d.paragraphs
    # crossover: its image paragraph and caption
    cap = next(p for p in paras if p.text.startswith("Fig. 2."))
    img = cap._p.getprevious()
    _wrap(body, img, cap._p)
    tcap = next(p for p in d.paragraphs if p.style.name == "Table Caption" and p.text.startswith("TABLE IV"))
    tbl = tcap._p.getnext()
    _wrap(body, tcap._p, tbl)


def _wrap(body, first, last):
    """Sections: two columns end just before `first`; one column runs from `first` to `last`."""
    before = OxmlElement("w:p")
    first.addprevious(before)
    before_ppr = OxmlElement("w:pPr")
    before.append(before_ppr)
    before_ppr.append(sect_pr(2))
    after = OxmlElement("w:p")
    last.addnext(after)
    ppr = OxmlElement("w:pPr")
    after.append(ppr)
    ppr.append(sect_pr(1))


def final_section(d):
    body = d.element.body
    old = body.find(qn("w:sectPr"))
    if old is not None:
        body.remove(old)
    body.append(sect_pr(2))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        crop_images(HERE / "main.pdf", tmp)
        src = to_pandoc_tex((HERE / "main.tex").read_text(), tmp)
        (tmp / "src.tex").write_text(src)
        raw = tmp / "raw.docx"
        subprocess.run(["pandoc", str(tmp / "src.tex"), "-s", "-o", str(raw)], check=True, cwd=tmp)
        d = docx.Document(raw)
    style_document(d)
    front_matter(d)
    mark_paragraphs(d)
    table_captions(d)
    style_tables(d)
    size_images(d)
    wide_blocks(d)
    final_section(d)
    d.save(HERE / "main.docx")
    print("wrote", HERE / "main.docx")


if __name__ == "__main__":
    main()
