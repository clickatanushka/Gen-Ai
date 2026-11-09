"""Build one Gen AI module (or all) from its content file mNN.py.

usage: python build.py 1 2 ...   (no args = every mNN.py found)
Each module file defines MODULE (see m01.py for the full schema). Outputs go to
../03_Modules/Module_NN_<Folder>/: 1_Module_Guide.docx, 2_Theory_Notes.docx, 3_Video_Scripts.docx,
4_Slides_ModuleNN.pptx, 5_Quiz.docx, 5_Quiz.csv. Diagrams render to out/mNN_<key>.png.

The deck clones the look of the Data Science kit: slides are copied from template.pptx prototypes
(Module 3 deck), then the 31 template slides are dropped. Recording cues never go on a slide: they
live in the speaker notes (SEGMENTS line on video title slides, DRAW note on diagram slides).
"""
import copy, csv, glob, importlib, io, os, re, sys

import pypandoc
from pptx import Presentation
from pptx.util import Emu, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
COURSE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from style import save   # noqa: E402

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
# template prototypes (1-based slide numbers in template.pptx)
P_COVER, P_TITLE, P_TWO, P_RECAP, P_STATS, P_CODE, P_TABLE, P_DIAGRAM, P_CARDS = 1, 2, 3, 5, 8, 9, 14, 4, 24
LEFT, WIDTH = 548640, 11094415
FMT = {"S": "Slides", "W": "Handwriting board", "R": "Screen recording", "F": "Face cam"}


# ---------------------------------------------------------------- pptx helpers
def set_lines(shape, lines, size=None, bullets=None):
    """Replace a text box's paragraphs with `lines`, keeping the first paragraph's look."""
    tb = shape.text_frame._txBody
    paras = tb.findall(A + "p")
    proto = copy.deepcopy(paras[0])
    for p in paras:
        tb.remove(p)
    for line in lines:
        p = copy.deepcopy(proto)
        runs = p.findall(A + "r")
        for r in runs[1:]:
            p.remove(r)
        if runs:
            runs[0].find(A + "t").text = line
            if size:
                runs[0].find(A + "rPr").set("sz", str(int(size * 100)))
        tb.append(p)


def by_name(slide, name):
    return [s for s in slide.shapes if s.name == name]


def one(slide, name, i=0):
    return by_name(slide, name)[i]


def drop(shape):
    shape._element.getparent().remove(shape._element)


class Deck:
    def __init__(self):
        self.prs = Presentation(os.path.join(HERE, "template.pptx"))
        self.n_template = len(self.prs.slides)
        self.protos = list(self.prs.slides)

    def clone(self, proto_no, notes):
        src = self.protos[proto_no - 1]
        s = self.prs.slides.add_slide(src.slide_layout)
        for sh in list(s.shapes):
            drop(sh)
        for sh in src.shapes:
            if sh.shape_type != 13:   # pictures are added fresh
                s.shapes._spTree.append(copy.deepcopy(sh._element))
        s.notes_slide.notes_text_frame.text = notes
        return s

    def finish(self, path):
        ids = self.prs.slides._sldIdLst
        for sid in list(ids)[: self.n_template]:
            self.prs.part.drop_rel(sid.rId)
            ids.remove(sid)
        for i, s in enumerate(self.prs.slides, 1):   # slide-number boxes keep the template's numbers
            for sh in by_name(s, "Slide Number Placeholder 0"):
                set_lines(sh, [str(i)])
        self.prs.save(path)
        return len(self.prs.slides)

    # ---- slide types
    def cover(self, kicker, title, subtitle):
        s = self.clone(P_COVER, "Deck title.")
        t = by_name(s, "Text 0")
        set_lines(t[0], [title]); set_lines(t[1], [subtitle]); set_lines(one(s, "Text 2"), [kicker])

    def video_title(self, label, title, segments, notes):
        s = self.clone(P_TITLE, f"SEGMENTS: {segments}\n{notes}")
        set_lines(by_name(s, "Text 0")[0], [title]); set_lines(one(s, "Text 1"), [label])

    def recap(self, title, points, task, notes, task_label="PRACTICE TASK"):
        s = self.clone(P_RECAP, notes)
        set_lines(one(s, "Text 0"), [title]); set_lines(one(s, "Text 1"), points)
        set_lines(one(s, "Text 3"), [task_label]); set_lines(one(s, "Text 4"), [task])

    def two_col(self, sl, notes):
        s = self.clone(P_TWO, notes)
        set_lines(one(s, "Text 0"), [sl["title"]])
        (h1, l1), (h2, l2) = sl["left"], sl["right"]
        set_lines(one(s, "Text 2"), [h1]); set_lines(one(s, "Text 3"), l1)
        set_lines(one(s, "Text 5"), [h2]); set_lines(one(s, "Text 6"), l2)

    def stats(self, sl, notes):
        s = self.clone(P_STATS, notes)
        set_lines(one(s, "Text 0"), [sl["title"]])
        items, gap = sl["stats"], 365760
        n = len(items)
        assert 2 <= n <= 4, "stats: 2 to 4 items"
        w = (WIDTH - (n - 1) * gap) // n
        groups = [("stat0", "Text 2", "Text 3"), ("stat1", "Text 5", "Text 6"), ("stat2", "Text 8", "Text 9")]
        while len(groups) < n:   # 4th stat: copy the 3rd group
            new = tuple(copy.deepcopy(one(s, nm)._element) for nm in groups[2])
            for el in new:
                s.shapes._spTree.append(el)
            groups.append(None)
        shapes = []
        for g in groups[:3]:
            shapes.append([one(s, nm) for nm in g])
        if n == 4:
            extra = list(s.shapes)[-3:]
            shapes.append(extra)
        for g in shapes[n:]:
            for sh in g:
                drop(sh)
        big = 54 if n <= 3 else 44
        for i, (box, num, lab) in enumerate(shapes[:n]):
            x = LEFT + i * (w + gap)
            box.left, box.width = Emu(x), Emu(w)
            for t in (num, lab):
                t.left, t.width = Emu(x + 274320), Emu(w - 548640)
            value, label = items[i]
            set_lines(num, [value], size=big if len(value) <= 7 else big - 14)
            set_lines(lab, [label])

    def cards(self, sl, notes):
        s = self.clone(P_CARDS, notes)
        set_lines(one(s, "Text 0"), [sl["title"]])
        items, gap = sl["cards"], 320040
        n = len(items)
        assert 2 <= n <= 4, "cards: 2 to 4 items"
        w = (WIDTH - (n - 1) * gap) // n
        groups = [["card0", "Shape 2", "Text 3", "Text 4", "Text 5"], ["card1", "Shape 7", "Text 8", "Text 9", "Text 10"],
                  ["card2", "Shape 12", "Text 13", "Text 14", "Text 15"], ["card3", "Shape 17", "Text 18", "Text 19", "Text 20"]]
        for g in groups[n:]:
            for nm in g:
                drop(one(s, nm))
        for i, g in enumerate(groups[:n]):
            card, dot, num, head, body = (one(s, nm) for nm in g)
            x = LEFT + i * (w + gap)
            card.left, card.width = Emu(x), Emu(w)
            for t in (dot, num):
                t.left = Emu(x + 274320)
            for t in (head, body):
                t.left, t.width = Emu(x + 274320), Emu(w - 548640)
            h, b = items[i]
            set_lines(head, [h], size=20 if n == 4 else 22)
            set_lines(body, [b], size=16 if n == 4 else 17)
            body.height = Emu(1645920)

    def code(self, sl, notes):
        s = self.clone(P_CODE, notes)
        set_lines(one(s, "Text 0"), [sl["title"]])
        set_lines(one(s, "Text 2"), [sl.get("label", "PYTHON")])
        lines = sl["code"].strip("\n").split("\n")
        panel, code, side = one(s, "codepanel"), one(s, "Text 3"), one(s, "Text 4")
        bullets = sl.get("bullets")
        if not bullets:
            drop(side)
            panel.width = Emu(WIDTH); code.width = Emu(WIDTH - 548640)
        width_in = code.width / 914400
        longest = max(len(l) for l in lines)
        size = min(17, width_in * 72 / (0.61 * max(longest, 1)), 3.9 * 72 / (1.22 * len(lines)))
        set_lines(code, lines, size=round(max(size, 9), 1))
        if bullets:
            set_lines(side, bullets)

    def bullets(self, sl, notes):
        s = self.clone(P_CODE, notes)
        set_lines(one(s, "Text 0"), [sl["title"]])
        for nm in ("codepanel", "Text 2", "Text 3"):
            drop(one(s, nm))
        side = one(s, "Text 4")
        side.left, side.width, side.top = Emu(LEFT), Emu(WIDTH), Emu(1463040)
        set_lines(side, sl["bullets"], size=sl.get("size", 22))

    def table(self, sl, notes):
        s = self.clone(P_TABLE, notes)
        set_lines(one(s, "Text 0"), [sl["title"]])
        frame = one(s, "Table 0")
        tbl = frame._element.find(".//" + A + "tbl")
        grid = tbl.find(A + "tblGrid")
        rows = tbl.findall(A + "tr")
        head_t, body_t = rows[0], rows[1:3] if len(rows) > 2 else [rows[1], rows[1]]
        for r in rows:
            tbl.remove(r)
        header, data = sl["header"], sl["rows"]
        ncol = len(header)
        widths = sl.get("widths") or [1] * ncol
        for g in grid.findall(A + "gridCol"):
            grid.remove(g)
        for wt in widths:
            el = grid.makeelement(A + "gridCol", {"w": str(int(WIDTH * wt / sum(widths)))})
            grid.append(el)
        size = sl.get("size", 15 if len(data) <= 6 else 13)

        def make(tpl, cells, bold=None):
            tr = copy.deepcopy(tpl)
            tcs = tr.findall(A + "tc")
            for tc in tcs:
                tr.remove(tc)
            for text in cells:
                tc = copy.deepcopy(tcs[0])
                ps = tc.findall(".//" + A + "p")
                for p in ps[1:]:
                    p.getparent().remove(p)
                r = ps[0].find(A + "r")
                r.find(A + "t").text = str(text)
                r.find(A + "rPr").set("sz", str(size * 100))
                tr.append(tc)
            return tr

        tbl.append(make(head_t, header))
        for i, row in enumerate(data):
            tbl.append(make(body_t[i % 2], row))
        frame.height = Emu(384048 * (len(data) + 1))

    def diagram(self, sl, notes, png):
        s = self.clone(P_DIAGRAM, notes)
        set_lines(one(s, "Text 0"), [sl["title"]])
        s.shapes.add_picture(png, Emu(LEFT), Emu(1325880), width=Emu(WIDTH))


# ---------------------------------------------------------------- docs
def docx(md, path):
    pypandoc.convert_text(md, "docx", format="markdown", outputfile=path,
                          extra_args=["--reference-doc", os.path.join(HERE, "reference.docx")])


def md_table(header, rows):
    out = ["| " + " | ".join(f"**{h}**" for h in header) + " |", "|" + "|".join(":---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


def esc(text):
    """Script text → markdown: keep [cues] literal."""
    return text.replace("[", "\\[").replace("]", "\\]")


# ---------------------------------------------------------------- build
def build(num):
    mod = importlib.import_module(f"m{num:02d}")
    importlib.reload(mod)
    M = mod.MODULE
    nn = f"{num:02d}"
    folder = os.path.join(COURSE, "03_Modules", f"Module_{nn}_{M['folder']}")
    os.makedirs(folder, exist_ok=True)

    # 1. slide numbers for every key, then the deck
    keys, n = {"cover": 1}, 1
    for v in M["videos"]:
        vk = v["num"].replace(".", "")
        n += 1; keys[f"v{vk}"] = n
        for sl in v["slides"]:
            n += 1
            if "key" in sl:
                assert sl["key"] not in keys, f"duplicate key {sl['key']}"
                keys[sl["key"]] = n
        n += 1; keys[f"r{vk}"] = n

    def fill(text):   # {key} → slide number; any other braces (code, dicts) are left alone
        return re.sub(r"\{(\w+)\}", lambda m: str(keys[m.group(1)]) if m.group(1) in keys else m.group(0), text)

    deck = Deck()
    deck.cover(f"MODULE {num} · {M['kicker']}", M["title"], M["subtitle"])
    for v in M["videos"]:
        vk = v["num"].replace(".", "")
        segs = " + ".join(FMT[c] for c in v["segments"])
        deck.video_title(f"VIDEO {v['num']} · {v['minutes']} MIN", v["title"], segs, f"Video {v['num']}.")
        for sl in v["slides"]:
            notes = f"Video {v['num']}." + (" " + sl["notes"] if sl.get("notes") else "")
            t = sl["type"]
            if t == "diagram":
                fig = getattr(mod, sl["draw"])()
                png = os.path.join(HERE, "out", f"m{nn}_{sl['key']}.png")
                save(fig, png)
                notes = ("✏ DRAW THIS SLIDE – hand-draw it live; the slide diagram is your reference\n" + notes +
                         "\n\nWhat to draw:\n" + "\n".join("• " + d for d in sl["draw_list"]))
                deck.diagram(sl, notes, png)
            else:
                getattr(deck, t)(sl, notes)
        r = v["recap"]
        deck.recap(f"Recap – Video {v['num']}", r["points"], fill(r["task"]), f"Video {v['num']}.",
                   r.get("label", "PRACTICE TASK"))
    total = deck.finish(os.path.join(folder, f"4_Slides_Module{nn}.pptx"))


    # 2. video scripts
    mins = sum(v["minutes"] for v in M["videos"])
    out = [f"# Module {num} – {M['title']}: Video Scripts", "",
           f"Slide numbers refer to `4_Slides_Module{nn}.pptx` ({total} slides). \\[Brackets\\] are on-screen "
           "actions: \\[Slide N\\] shows a slide, \\[Board: …\\] is drawn live on the handwriting board, "
           "\\[Screen: …\\] is a screen recording. " + M.get("script_intro", "") +
           f" Spoken length is about 70–80% of each runtime; drawing and pauses take the rest. Module total: {mins} minutes.",
           ""]
    for v in M["videos"]:
        out += ["------------------------------------------------------------------------", "",
                f"## Video {v['num']} – {v['title']} ({v['minutes']} min)", "", "### Shot plan", "",
                md_table(["Time", "On screen", "Format"], [(a, fill(b), c) for a, b, c in v["shot_plan"]]), "",
                "### Script", ""]
        for para in v["script"]:
            out += [esc(fill(para)), ""]
    docx("\n".join(out), os.path.join(folder, "3_Video_Scripts.docx"))

    # 3. guide + notes
    head = md_table(["Video", "Title", "Length", "Format"],
                    [(v["num"], v["title"], f"{v['minutes']} min", " + ".join(FMT[c] for c in v["segments"]))
                     for v in M["videos"]])
    guide = (f"# Module {num} – {M['title']}: Module Guide (for the instructor)\n\n"
             f"Total: {mins} minutes, {len(M['videos'])} videos. {M['guide_intro']}\n\n{head}\n\n{fill(M['guide_md']).strip()}\n")
    docx(guide, os.path.join(folder, "1_Module_Guide.docx"))
    docx(f"# Module {num} – {M['title']}: Theory Notes\n\n{fill(M['notes_md']).strip()}\n",
         os.path.join(folder, "2_Theory_Notes.docx"))

    # 4. quiz
    Q = M["quiz"]
    md = [f"# Module {num} – {M['title']} – Quiz", "", f"{len(Q)} questions. One correct answer each.", ""]
    for i, q in enumerate(Q, 1):
        assert q["correct"] in "ABCD"
        md += [f"### Q{i}. {q['q']}", ""] + [f"- {k}\\) {q[k.lower()]}" for k in "ABCD"] + \
              ["", f"Correct answer: {q['correct']}", "", f"Explanation: {q['why']}", ""]
    docx("\n".join(md), os.path.join(folder, "5_Quiz.docx"))
    with open(os.path.join(folder, "5_Quiz.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "option_a", "option_b", "option_c", "option_d", "correct", "explanation"])
        for q in Q:
            w.writerow([q["q"], q["a"], q["b"], q["c"], q["d"], q["correct"], q["why"]])
    print(f"Module {nn}: {total} slides, {len(M['videos'])} videos, {mins} min, {len(Q)} quiz -> {folder}")
    return total, mins


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1:]] or sorted(int(re.search(r"m(\d+)", f).group(1))
                                                   for f in glob.glob(os.path.join(HERE, "m[0-9][0-9].py")))
    for k in nums:
        build(k)
