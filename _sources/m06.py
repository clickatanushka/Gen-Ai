"""Module 6 – Document ingestion and chunking: PDF text extraction, cleaning, token chunking with overlap.

Schema: see m01.py. The code on the slides is a small generic reference implementation (pypdf + tiktoken).
"""
from style import *


def pdf_text():   # PDF page → extract_text() → clean_text()
    fig, ax = canvas()
    cx, w, h, yc = [2.75, 8.0, 13.25], 4.7, 4.5, 3.05
    heads = ["1 · The PDF page", "2 · page.extract_text()", "3 · after clean_text()"]
    cols = [BLUE, ORANGE, GREEN]
    for x, hd, c in zip(cx, heads, cols):
        box(ax, x, yc, w, h, "", color=c)
        text(ax, x, yc + h / 2 + 0.33, hd, fs=17, bold=True, color=c)
    for a, b, lab in [(cx[0], cx[1], "pypdf"), (cx[1], cx[2], "clean")]:
        arrow(ax, a + w / 2 + 0.05, yc, b - w / 2 - 0.05, yc, color=INK, lw=2)
        text(ax, (a + b) / 2, yc + 0.3, lab, fs=12, color=MUTED)
    # panel 1: what a reader sees
    x0, top = cx[0] - w / 2 + 0.25, yc + h / 2 - 0.35
    text(ax, x0, top, "City College · Student Handbook", fs=12, color=MUTED, ha="left")
    text(ax, x0, top - 0.6, "Fee refunds", fs=14, bold=True, ha="left")
    body = ["Students may request a fee", "refund within 14 days of", "joining. The manage-",
            "ment office approves each", "request."]
    for i, line in enumerate(body):
        text(ax, x0, top - 1.1 - i * 0.4, line, fs=13, ha="left")
    text(ax, cx[0], yc - h / 2 + 0.3, "Page 3", fs=12, color=MUTED)
    # panel 2: raw string, line breaks kept
    x1 = cx[1] - w / 2 + 0.2
    raw = ["City College · Student\\n", "Handbook\\nFee   refunds\\n", "Students may request a\\n",
           "fee refund within 14\\n", "days of joining. The\\n", "manage-\\nment office\\n",
           "approves...\\nPage 3"]
    for i, line in enumerate(raw):
        text(ax, x1, top - i * 0.5, line, fs=12, ha="left", family="monospace")
    text(ax, cx[1], yc - h / 2 + 0.3, "line breaks, double spaces, split word", fs=12, color=RED)
    # panel 3: one clean line (wrapped here only to fit)
    x2 = cx[2] - w / 2 + 0.25
    clean = ["City College · Student Handbook", "Fee refunds Students may request", "a fee refund within 14 days of",
             "joining. The management office", "approves each request. Page 3"]
    for i, line in enumerate(clean):
        text(ax, x2, top - i * 0.42, line, fs=13, ha="left")
    text(ax, x2, yc - 0.95, "+ one line, single spaces", fs=12, color=GREEN, bold=True, ha="left")
    text(ax, x2, yc - 1.35, "+ manage- ment → management", fs=12, color=GREEN, bold=True, ha="left")
    text(ax, x2, yc - 1.75, "− header and “Page 3” still inside", fs=12, color=RED, bold=True, ha="left")
    return fig


def window():   # 400-token windows sliding by 350 over 1,000 tokens and three pages
    fig, ax = canvas()
    X0, X1, N = 0.9, 15.1, 1000
    tx = lambda t: X0 + (X1 - X0) * t / N
    ys, sh = 1.35, 0.62   # page strip
    for a, b, c, lab in [(0, 300, BLUE, "page 1 · 300 tokens"), (300, 800, TEAL, "page 2 · 500 tokens"),
                         (800, 1000, PURPLE, "page 3 · 200 tokens")]:
        ax.add_patch(Rectangle((tx(a), ys), tx(b) - tx(a), sh, fc=TINT[c], ec=c, lw=2))
        text(ax, (tx(a) + tx(b)) / 2, ys + sh / 2, lab, fs=13, bold=True, color=c)
    for t in (0, 350, 400, 700, 750, 1000):
        ax.plot([tx(t)] * 2, [ys - 0.1, ys], color=INK, lw=1.5)
        text(ax, tx(t), ys - 0.32, f"{t:,}", fs=12, color=MUTED)
    text(ax, X0, ys - 0.75, "token number in the whole PDF (all pages flattened into one list)", fs=12,
         color=MUTED, ha="left")
    # overlap bands
    for a in (350, 700):
        ax.add_patch(Rectangle((tx(a), ys + sh), tx(a + 50) - tx(a), 3.65, fc="#FCE9C8", ec="none", zorder=0))
    text(ax, tx(375), ys + sh + 0.3, "50 overlap", fs=12, color=ORANGE, bold=True)
    text(ax, tx(725), ys + sh + 0.3, "50 overlap", fs=12, color=ORANGE, bold=True)
    bars = [(0, 400, 4.55, CORAL, "chunk 0 · tokens 0–399 · 400", "cites p.1"),
            (350, 750, 3.6, ORANGE, "chunk 1 · tokens 350–749 · 400", "cites p.2"),
            (700, 1000, 2.65, GREEN, "chunk 2 · 700–999 · 300", "cites p.2, though most of it is page 3")]
    for a, b, y, c, lab, cite in bars:
        ax.add_patch(FancyBboxPatch((tx(a), y), tx(b) - tx(a), 0.6, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=TINT[c], ec=c, lw=2))
        text(ax, (tx(a) + tx(b)) / 2, y + 0.3, lab, fs=13, bold=True, color=INK)
        ax.plot([tx(a)] * 2, [ys + sh, y], color=c, lw=1.5, ls=(0, (3, 3)))
        if b < 900:
            text(ax, tx(b) + 0.15, y + 0.3, cite, fs=13, bold=True, color=c, ha="left")
        else:
            text(ax, tx(a) - 0.15, y + 0.3, cite, fs=13, bold=True, color=c, ha="right")
    arrow(ax, tx(0), 5.6, tx(350), 5.6, color=INK, lw=2, ms=14)
    text(ax, tx(175), 5.88, "step = 400 − 50 = 350", fs=13, bold=True)
    arrow(ax, tx(350), 5.6, tx(700), 5.6, color=INK, lw=2, ms=14)
    text(ax, tx(525), 5.88, "next start = 700", fs=13, bold=True)
    text(ax, tx(1000), 5.88, "cited page = page of the FIRST token", fs=13, bold=True, color=RED, ha="right")
    return fig


def trade():   # chunk size trade-off
    fig, ax = canvas()
    X0, X1, y = 1.2, 14.8, 3.1
    tx = lambda s: X0 + (X1 - X0) * (s - 100) / 700
    arrow(ax, X0 - 0.3, y, X1 + 0.3, y, color=INK, lw=2.5, style="<|-|>")
    for s in (100, 200, 400, 600, 800):
        ax.plot([tx(s)] * 2, [y - 0.12, y + 0.12], color=INK, lw=2)
        text(ax, tx(s), y - 0.42, f"{s}", fs=14, color=MUTED, bold=True)
    text(ax, X1, y - 0.85, "chunk size in tokens", fs=13, color=MUTED, ha="right")
    box(ax, 4.0, 5.0, 6.6, 2.0, "", color=BLUE)
    text(ax, 0.95, 5.65, "Small chunks (100–200)", fs=17, bold=True, color=BLUE, ha="left")
    for i, (s, c) in enumerate([("+ sharp match: one idea per vector", GREEN),
                                ("− the answer may lose its context", RED),
                                ("− more vectors to store and search", RED)]):
        text(ax, 0.95, 5.15 - i * 0.42, s, fs=14, color=c, ha="left")
    box(ax, 12.0, 5.0, 6.6, 2.0, "", color=PURPLE)
    text(ax, 8.95, 5.65, "Large chunks (600–800)", fs=17, bold=True, color=PURPLE, ha="left")
    for i, (s, c) in enumerate([("+ the full rule stays in one piece", GREEN),
                                ("− one vector for many topics: diluted", RED),
                                ("− more tokens in every prompt", RED)]):
        text(ax, 8.95, 5.15 - i * 0.42, s, fs=14, color=c, ha="left")
    pill(ax, tx(400), 1.55, "common starting point: 400 tokens, 50 overlap", color=CORAL, fs=15)
    arrow(ax, tx(400), 2.05, tx(400), y - 0.62, color=CORAL, lw=2)
    text(ax, 0.95, 0.55, "“…within 14 days of delivery.”  14 days for what?", fs=13, color=BLUE,
         ha="left")
    text(ax, 15.05, 0.55, "returns + shipping + warranty in one chunk: what is it about?", fs=13, color=PURPLE,
         ha="right")
    return fig


V61 = dict(
    num="6.1", title="Getting text out of PDFs", minutes=9, segments="SW",
    slides=[
        dict(type="cards", key="pipe", title="Ingestion: four steps, once per document",
             cards=[("Extract", "Pull the text out of the file, page by page, and keep the page numbers."),
                    ("Clean", "Fix what extraction breaks: line breaks, extra spaces, split words."),
                    ("Chunk", "Cut the text into small, overlapping pieces of a fixed size."),
                    ("Embed + store", "Turn each chunk into a vector; save it with its text and page.")]),
        dict(type="two_col", key="stores", title="What a PDF really stores",
             left=("What you see", ["Headings and paragraphs", "Tables with rows and columns",
                                    "Two columns of text", "Page numbers, headers, footers"]),
             right=("What the file stores", ["Characters drawn at x, y positions", "Fonts and sizes",
                                             "No “paragraph” or “table” object",
                                             "Sometimes only a photo of the text"])),
        dict(type="code", key="extract", title="Step 1: read every page with pypdf",
             code=r'''
from pypdf import PdfReader

def read_pdf_pages(path):
    reader = PdfReader(path)
    pages = []
    for num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:                    # skip blank pages
            pages.append((num, text))
    if not pages:
        raise ValueError(f"No extractable text in {path}")
    return pages
''',
             bullets=["Pages are numbered from 1, like a human reads them",
                      "Result: a list of (page number, text) pairs",
                      "Blank pages are dropped, page numbers are kept",
                      "No text at all? Fail loudly, do not store nothing"]),
        dict(type="diagram", key="clean", title="From PDF page to clean text", draw="pdf_text",
             draw_list=["Three panels: the PDF page, extract_text(), after clean_text()",
                        "Page: header City College · Student Handbook, heading Fee refunds, body with "
                        "“manage-” / “ment” split over a line, footer Page 3",
                        "Raw text: \\n after every line, double spaces, the split word",
                        "Clean text: one line, single spaces, “management” joined",
                        "Red note: header and Page 3 are still inside the text"]),
        dict(type="code", key="cleancode", title="clean_text: three rules",
             code=r'''
import re

def clean_text(text):
    # 1. any run of spaces/newlines -> one space
    text = re.sub(r"\s+", " ", text)
    # 2. remove invisible control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # 3. join words split at a line end
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text.strip()
''',
             bullets=["“manage- ment” → “management”",
                      "Side effect: “self- service” → “selfservice”",
                      "Rule 3 only works because rule 1 ran first"]),
        dict(type="table", key="wrong", title="When extraction goes wrong",
             header=["Problem", "What you get", "Usual fix"], widths=[1.3, 2.6, 2.3],
             rows=[["Scanned PDF", "No text layer: extract_text() returns nothing", "OCR before ingestion"],
                   ["Tables", "Cells run into one line; which number is whose?", "Table-aware extraction"],
                   ["Two columns", "Lines from both columns mixed together", "Layout-aware extraction"],
                   ["Headers, footers", "Same title and page number on every page", "Detect and drop repeated lines"]]),
    ],
    recap=dict(points=["Ingestion = extract, clean, chunk, embed + store; once per document",
                       "A PDF stores positioned characters, not paragraphs",
                       "pypdf extract_text() gives one string per page; number pages from 1",
                       "clean_text: single spaces, no control characters, joined hyphen-splits",
                       "Scanned pages, tables, columns and headers need extra care"],
               task="Open any PDF from your own files. Copy one page into a text editor. Find one table, one "
                    "split word or one repeated header in the pasted text.", label="TRY THIS"),
    shot_plan=[("0:00–0:50", "Slide {cover}, Slide {v61}, hook", "S"),
               ("0:50–2:00", "Slide {pipe} the four ingestion steps", "S"),
               ("2:00–3:00", "Slide {stores} what a PDF stores", "S"),
               ("3:00–4:10", "Slide {extract} read_pdf_pages", "S"),
               ("4:10–6:10", "Slide {clean} → board: page → raw → clean", "W"),
               ("6:10–7:10", "Slide {cleancode} three cleaning rules", "S"),
               ("7:10–8:20", "Slide {wrong} when extraction goes wrong", "S"),
               ("8:20–9:00", "Slide {r61} recap", "S")],
    script=[
        "[Slide {cover}, Slide {v61}]",
        "Welcome to Module 6. In the last two modules we learned how embeddings turn text into numbers, and "
        "how a vector database finds the closest ones. But we skipped one question: where does that text come "
        "from? A company HR policy, a college handbook, a shop's returns guide: they "
        "usually arrive as PDFs. Before anyone can search them, something has to read the file, clean the text and "
        "cut it into pieces. That work is called ingestion, and it is the topic of this module.",
        "[Slide {pipe}]",
        "Ingestion has four steps. One, extract: pull the text out of the file, page by page. Two, clean: fix "
        "the mess that extraction leaves behind. Three, chunk: cut the text into small pieces of a fixed size. "
        "Four, embed and store: turn every chunk into a vector and save it in the vector database, together "
        "with its text and its page number.",
        "Notice the heading: once per document. Ingestion happens ahead of time, not when a user asks a "
        "question. And remember: garbage in, garbage out. If the text from step one is messy, a better embedding model or a bigger LLM "
        "cannot fix it. In this video we cover steps one and two.",
        "[Slide {stores}]",
        "Here is a surprise for many people. When you look at a PDF, you see headings, paragraphs, tables, "
        "maybe two columns. But the file does not store any of that. A PDF stores characters, each drawn at "
        "an x and y position on the page, with a font and a size. There is no object called paragraph, and no "
        "object called table. And if someone scanned a "
        "paper document, the page is just a photograph of text. To a computer, that is pixels, not words.",
        "So getting text out of a PDF means rebuilding words and lines from positions. Libraries do this for "
        "us. In Python, a popular free one is pypdf, and that is what we will use.",
        "[Slide {extract}]",
        "Here is a small reference function, read_pdf_pages. PdfReader opens the file. Then we loop over "
        "reader.pages. Notice start equals 1 in enumerate. Python normally counts from zero, but people count "
        "pages from one, and these numbers will end up in citations. Page 3 in an answer must mean page 3 in "
        "the PDF viewer.",
        "For each page, extract_text gives us one string. The “or empty string” part protects us when a page "
        "returns nothing. We clean the text, and if anything is left, we keep a pair: page number and text. "
        "Blank pages are skipped, but the other pages keep their real numbers. And if no page had any text at "
        "all, we raise an error instead of quietly storing nothing.",
        "[Slide {clean} → Board]",
        "Let me draw what happens to one page. [Board: panel 1, a page with a header, a heading, five lines "
        "of body text and Page 3 at the bottom] Here is page 3 of a college handbook. At the top, a header: "
        "City College, Student Handbook. Then a heading, fee refunds, and the body. Look at the third line of the body: "
        "the word management did not fit, so it was split. Manage, hyphen, and then ment on the next line.",
        "[Board: panel 2, the raw string with backslash n after every line] This is what extract_text gives "
        "us. One string, with a line break, backslash n, after every line. And the "
        "split word: manage, hyphen, line break, ment. If we embed this as it is, the word management does not "
        "exist in our text, so a question about the management office matches a little worse.",
        "[Board: panel 3, one clean line, management joined] After cleaning: one long line, single spaces, and "
        "management is one word again. [Board: red note, header and Page 3 still inside] But the header and "
        "the words Page 3 are still there. Our simple cleaning does not remove them.",
        "[Slide {cleancode}]",
        "The cleaning function has three rules, each a regular expression, a pattern for finding text. Rule "
        "one: any run of spaces, tabs or line breaks becomes one space. Rule two: remove invisible control "
        "characters that some PDFs contain. Rule three: a letter, a hyphen, some space, and a letter become the "
        "two letters joined. The order matters: rule three works because rule one already turned the line "
        "break into a space.",
        "Every rule has a side effect. If the phrase self-service broke right after its hyphen, rule three "
        "cannot tell a real hyphen from a split word, so we get selfservice. A small loss; joining split words "
        "helps far more than it hurts.",
        "[Slide {wrong}]",
        "Now the four classic problems. One: scanned PDFs have no text layer, so extract_text returns nothing. "
        "The fix is OCR, optical character recognition, which reads text from images. Two: tables. A fee table "
        "might come out as “plan basic premium price 499 999”. The numbers are there, but which price belongs "
        "to which plan is lost. Three: two-column layouts, where lines from both columns get mixed. Four: "
        "headers and footers that repeat on every page and add noise to many chunks. A common fix is to find "
        "lines that appear on most pages and drop them before rule one joins everything into one line.",
        "When an answer looks strange, the first thing a good engineer checks is the extracted text.",
        "[Slide {r61}]",
        "Recap. Ingestion is extract, clean, chunk, embed and store, done once per document. A PDF stores "
        "characters at positions, not paragraphs. extract_text gives one string per page, and we number pages "
        "from one. clean_text makes single spaces, removes control characters and joins split words. And "
        "scanned pages, tables, columns and headers need extra care.",
        "Try this: copy one page of any PDF into a text editor, and look for a broken table, a split word or a "
        "repeated header. In the next video: why we cut text into chunks, and exactly how.",
    ],
)

V62 = dict(
    num="6.2", title="Chunking with tokens and overlap", minutes=12, segments="SW",
    slides=[
        dict(type="cards", key="why", title="Why cut documents into chunks?",
             cards=[("Input limits", "An embedding model accepts a limited input: 8,191 tokens for "
                                     "text-embedding-3-small at the time of writing. A long PDF does not fit."),
                    ("Precise retrieval", "One vector for a whole PDF is a blur. One vector per passage can "
                                          "point at the exact rule."),
                    ("Prompt cost", "We send only the best few passages to the LLM, not every page. Fewer "
                                    "tokens: cheaper and faster.")]),
        dict(type="code", key="flatten", title="Flatten the pages, remember the pages",
             code=r'''
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def chunk_pages(pages, size=400, overlap=50):
    if not 0 <= overlap < size:
        raise ValueError("overlap must be smaller than size")
    tokens, page_of = [], []
    for page_num, text in pages:
        ids = enc.encode(text)
        tokens.extend(ids)
        page_of.extend([page_num] * len(ids))
    ...
''',
             bullets=["All pages become one long list of tokens",
                      "page_of has one entry per token: its page",
                      "Page 1 with 300 tokens → three hundred 1s",
                      "The guard stops a loop that never moves"]),
        dict(type="code", key="loop", title="The sliding window",
             code=r'''
    chunks, start = [], 0
    while start < len(tokens):
        piece = tokens[start:start + size]
        chunks.append({
            "text": enc.decode(piece),
            "page": page_of[start],     # first token
            "index": len(chunks),
            "token_count": len(piece),
        })
        start += size - overlap         # 400 - 50 = 350
    return chunks
''',
             bullets=["Take 400 tokens, turn them back into text",
                      "Cite the page of the FIRST token",
                      "Move forward 400 − 50 = 350 tokens",
                      "The last slice may be shorter"]),
        dict(type="diagram", key="window", title="400-token windows over a 3-page PDF", draw="window",
             draw_list=["A strip of 1,000 tokens: page 1 = 0–299, page 2 = 300–799, page 3 = 800–999",
                        "Chunk 0: tokens 0–399, cites p.1",
                        "Chunk 1: starts at 350, tokens 350–749, cites p.2",
                        "Chunk 2: starts at 700, tokens 700–999, only 300 tokens, cites p.2",
                        "Shade the two 50-token overlaps; write step = 400 − 50 = 350",
                        "Write in red: cited page = page of the FIRST token"]),
        dict(type="table", key="worked", title="Worked example: 1,000 tokens, size 400, overlap 50",
             header=["Chunk", "Start", "Tokens", "Count", "Pages inside", "Cited page"],
             widths=[0.8, 0.8, 1.2, 0.8, 1.5, 1.1],
             rows=[["0", "0", "0–399", "400", "1 and 2", "p.1"],
                   ["1", "350", "350–749", "400", "2", "p.2"],
                   ["2", "700", "700–999", "300", "2 and 3", "p.2"],
                   ["–", "1,050", "stop: 1,050 ≥ 1,000", "", "", ""]]),
        dict(type="code", key="chunk", title="One chunk, ready for the vector database",
             code=r'''
record = {
    "id": f"{doc_id}-{chunk['index']}",  # "handbook-2"
    "values": vector,                    # 1,536 floats
    "metadata": {
        "text": chunk["text"],
        "source": doc_id,
        "page": chunk["page"],
        "chunk_index": chunk["index"],
        "token_count": chunk["token_count"],
    },
}
index.upsert(vectors=[record])
''',
             bullets=["The text travels with the vector, so the answer can quote it",
                      "source + page become the citation",
                      "source also lets you filter or delete one document"]),
        dict(type="table", key="others", title="Other chunking strategies",
             header=["Strategy", "How it splits", "Good for"], widths=[1.3, 2.6, 2.2], size=14,
             rows=[["Fixed tokens", "Every N tokens, with overlap", "A simple, predictable baseline"],
                   ["Fixed characters", "Every N characters", "Quick scripts; sizes vary in tokens"],
                   ["Recursive", "Paragraphs, then sentences, then words", "Keeping sentences whole"],
                   ["By structure", "Headings, sections, clauses", "Policies, manuals, contracts"],
                   ["Semantic", "Where the topic changes (embeddings)", "Long, mixed documents"],
                   ["Small-to-big", "Search small chunks, send the bigger parent", "Precise search, full context"]]),
    ],
    recap=dict(points=["Chunk because of input limits, precise retrieval and prompt cost",
                       "Flatten pages into one token list plus a token → page map",
                       "Window of 400 tokens, step 400 − 50 = 350",
                       "Each chunk cites the page of its first token",
                       "Smarter splits exist: recursive, by structure, semantic, small-to-big"],
               task="A PDF has 1,500 tokens. With size 400 and overlap 50, write down where each chunk starts "
                    "and how many tokens it holds. Careful with the last ones: apply the loop exactly."),
    shot_plan=[("0:00–0:40", "Slide {v62}, hook", "S"),
               ("0:40–2:10", "Slide {why} why chunk", "S"),
               ("2:10–3:40", "Slide {flatten} flatten the pages", "S"),
               ("3:40–4:50", "Slide {loop} the sliding window", "S"),
               ("4:50–7:50", "Slide {window} → board: windows over 3 pages", "W"),
               ("7:50–9:00", "Slide {worked} worked example and edge case", "S"),
               ("9:00–10:10", "Slide {chunk} one chunk record", "S"),
               ("10:10–11:20", "Slide {others} other strategies", "S"),
               ("11:20–12:00", "Slide {r62} recap", "S")],
    script=[
        "[Slide {v62}]",
        "We now have clean text for every page. The next question sounds simple: why not store each page, or "
        "the whole PDF, as one piece? Why cut it into chunks at all? Let us answer that, and then build a "
        "chunker step by step.",
        "[Slide {why}]",
        "Three reasons. First, input limits. An embedding model can only read a limited amount of text at "
        "once. For OpenAI's text-embedding-3-small, at the time of writing, the limit is 8,191 tokens. A "
        "100-page manual is far bigger than that.",
        "Second, precise retrieval. An embedding is one point that summarises the meaning of a text. If that "
        "text is a whole policy about leave, travel, laptops and refunds, the point lands somewhere in the "
        "middle of all those topics. It is a blur. If each passage gets its own vector, a question about "
        "refunds can land right next to the refund passage.",
        "Third, prompt cost. In the end we send retrieved text to the LLM, and we pay for every token we send. "
        "Five focused passages are much cheaper and faster than fifty pages. So chunking is not a small "
        "technical detail. It decides what can be found, and what each answer costs.",
        "[Slide {flatten}]",
        "Our chunker counts tokens, not characters or pages. A token is the unit a model reads, roughly "
        "three-quarters of an English word; we met tokens in Module 2. We use tiktoken with the cl100k_base "
        "encoding, which is the tokenizer of the OpenAI embedding models we use. So 400 tokens here means 400 "
        "tokens to the model. The rule: chunk with the tokenizer of your embedding model.",
        "The function starts with a guard: overlap must be at least zero and smaller than the chunk size. You "
        "will see why in a moment. Then all pages are flattened into one long list of tokens. Page boundaries "
        "disappear, so a chunk can run from the end of one page into the next. But we must not lose the page "
        "numbers, because citations need them. So in parallel we build page_of: one entry for every token, "
        "holding its page number. If page 1 has 300 tokens, the list starts with three hundred ones.",
        "[Slide {loop}]",
        "Now the sliding window. Start at zero. Take a slice of size tokens, 400. Python slicing simply stops at "
        "the end of the list, so the last slice may be shorter. Decode the slice back into text. Look up the "
        "page of the first token: that is the page this chunk will cite. Save the chunk with its index and "
        "token count. Then move start forward by size minus overlap: 400 minus 50, so 350. Repeat while start "
        "is still inside the document.",
        "And here is the reason for the guard. If overlap equals the size, the step is zero, start never moves, "
        "and the loop runs forever. One line of checking prevents a frozen program.",
        "[Slide {window} → Board]",
        "Let me draw it, because this is where the details hide. [Board: a long strip, split into page 1 = 0 "
        "to 299, page 2 = 300 to 799, page 3 = 800 to 999] Here is a small PDF of 1,000 tokens. Page 1 has 300 "
        "tokens, page 2 has 500, page 3 has 200.",
        "[Board: chunk 0 bar over tokens 0 to 399] Chunk 0 starts at token 0 and takes 400 tokens, 0 to 399. "
        "It covers all of page 1 and the start of page 2. Its first token is on page 1, so it cites page 1.",
        "[Board: arrow, step = 400 − 50 = 350] Now we step forward 350, not 400. [Board: chunk 1 bar over 350 "
        "to 749, shade 350 to 399] Chunk 1 runs from 350 to 749. The shaded part, 350 to 399, is in both "
        "chunks. That is the overlap: 50 tokens. Why repeat them? Because a sentence that sits across the cut "
        "now appears whole in at least one chunk. Token 350 is on page 2, so chunk 1 cites page 2.",
        "[Board: chunk 2 bar over 700 to 999] Next start: 700. The slice would end at 1,100, but the document "
        "ends at 1,000, so chunk 2 has only 300 tokens. The next start would be 1,050, past the end, so we "
        "stop. Three chunks.",
        "[Board: red, cited page = page of the FIRST token] Now look at chunk 2 carefully. It starts on page 2, "
        "so it cites page 2. But 200 of its 300 tokens are on page 3. If the answer comes from that part, the "
        "citation points one page early. That is a real limit of this simple rule. A reader who checks page 2 "
        "and does not find the sentence should look at the next page too.",
        "[Slide {worked}]",
        "Here is the same example as a table: starts at 0, 350 and 700; counts 400, 400 and 300; cited pages "
        "1, 2 and 2. When you can fill this table by hand, you really understand the loop.",
        "One more case, because it shows how literally a computer follows a loop. Take a document of 750 "
        "tokens. Chunk 0 is 0 to 399. Chunk 1 is 350 to 749, which already reaches the end. But the loop only "
        "asks: is the next start, 700, smaller than 750? Yes. So it makes a third chunk, tokens 700 to 749, only "
        "50 tokens, and all of them are already inside chunk 1. A tiny duplicate. Harmless, but it wastes a "
        "vector.",
        "[Slide {chunk}]",
        "When we store a chunk, it becomes one record in the vector database. The id is the document id, a "
        "dash, and the chunk index, like handbook-2. The values are the embedding: 1,536 numbers for "
        "text-embedding-3-small. And the metadata holds the text, the source document, the page, the chunk "
        "index and the token count. The text travels with the vector, so the prompt can include it later. "
        "Source and page become the citation. And the source field lets you filter by document, or delete one "
        "document's chunks.",
        "One honest detail: because we cut by tokens, a chunk can start or end in the middle of a sentence, or "
        "even a word. Overlap reduces the damage, but it does not remove it.",
        "[Slide {others}]",
        "Fixed token windows are only one strategy. Fixed characters is similar but counts letters, so sizes "
        "vary in tokens. Recursive splitting tries paragraphs first, then sentences, then words, so sentences "
        "usually stay whole. Splitting by structure follows headings and sections, which suits policies and "
        "contracts. Semantic chunking uses embeddings to cut where the topic changes. And small-to-big: search "
        "small, precise chunks, but send the bigger parent section to the LLM. Fixed tokens is a good first "
        "choice because it is predictable and easy to debug. Start simple, measure, then improve.",
        "[Slide {r62}]",
        "Recap. We chunk because of input limits, precise retrieval and prompt cost. We flatten all pages into "
        "one token list with a token-to-page map, take 400-token windows, and step by 350. Each chunk cites the "
        "page of its first token. And smarter strategies exist: recursive, by structure, semantic and "
        "small-to-big.",
        "Practice: a PDF has 1,500 tokens. With size 400 and overlap 50, write down where each chunk starts and "
        "how many tokens each one has. In the next video: how to choose chunk size, overlap and top-k.",
    ],
)

V63 = dict(
    num="6.3", title="Choosing chunk size, overlap and top-k", minutes=9, segments="SW",
    slides=[
        dict(type="diagram", key="trade", title="The chunk-size trade-off", draw="trade",
             draw_list=["An axis of chunk size: 100, 200, 400, 600, 800 tokens",
                        "Left box, small chunks: + sharp match; − loses context; − more vectors",
                        "Right box, large chunks: + full rule in one piece; − diluted meaning; − more prompt tokens",
                        "Arrow at 400: a common starting point, 400 / 50",
                        "Examples: “within 14 days” (for what?) vs returns + shipping + warranty in one chunk"]),
        dict(type="stats", key="cost", title="Context sent with every question",
             stats=[("400", "tokens per chunk"), ("× 5", "chunks retrieved (top-k)"),
                    ("2,000", "tokens of context in every prompt")]),
        dict(type="table", key="when", title="When each setting takes effect",
             header=["Setting", "Typical range", "Example", "Takes effect"], widths=[1.5, 1.4, 1.0, 2.6],
             rows=[["Chunk size", "200–800 tokens", "400", "At ingestion: re-ingest to change"],
                   ["Chunk overlap", "10–20% of size", "50", "At ingestion: re-ingest to change"],
                   ["Top-k", "3–8 chunks", "5", "At query time: the very next question"],
                   ["Embedding model", "one per index", "same for both", "At ingestion: re-embed everything"]]),
        dict(type="cards", key="gotchas", title="Two gotchas",
             cards=[("Remove old chunks first", "Ids are document + chunk number. Re-ingest with fewer chunks "
                                                "and the old high numbers stay behind. Delete by source, then "
                                                "ingest again."),
                    ("Overlap < chunk size", "The step is size − overlap. Size 100 with overlap 100 gives a "
                                             "step of 0: the loop never moves forward.")]),
        dict(type="bullets", key="choose", title="Choosing settings like an engineer",
             bullets=["Write 10–20 real questions you know the answers to",
                      "Ingest with one setting; ask every question; look at the retrieved chunks",
                      "Count: was the right passage in the top-k?",
                      "Change one setting, re-ingest, count again",
                      "Keep the setting that finds the right passage most often"]),
    ],
    recap=dict(points=["Small chunks: precise but lose context; large: context but diluted",
                       "Overlap protects boundaries; it costs about 14% more vectors at 400/50",
                       "Context per question ≈ top-k × chunk size (5 × 400 = 2,000)",
                       "Size and overlap need re-ingesting (delete old chunks first); top-k does not",
                       "Choose with real test questions, not by guessing"],
               task="Pick one PDF and five questions. Chunk it at size 200 and at size 800, retrieve top-5 for each "
                    "question, and note whether the right passage appears."),
    shot_plan=[("0:00–0:40", "Slide {v63}, hook", "S"),
               ("0:40–3:30", "Slide {trade} → board: the trade-off", "W"),
               ("3:30–4:50", "Slide {cost} context per question", "S"),
               ("4:50–5:50", "Slide {when} when each setting takes effect", "S"),
               ("5:50–7:10", "Slide {gotchas} two gotchas", "S"),
               ("7:10–8:15", "Slide {choose} choosing settings", "S"),
               ("8:15–9:00", "Slide {r63} recap", "S")],
    script=[
        "[Slide {v63}]",
        "Every document assistant has three numbers to set: chunk size, chunk overlap and top-k, the number of "
        "chunks we retrieve for each question. Every learner asks the same thing: what are the right numbers? "
        "The honest answer is: it depends on your documents. In this video you will learn what each number "
        "trades away, and how to choose with evidence instead of guessing.",
        "[Slide {trade} → Board]",
        "[Board: a horizontal axis, chunk size from 100 to 800 tokens] Let me draw chunk size as a line. Small "
        "on the left, large on the right.",
        "[Board: left box, small chunks] Small chunks, 100 or 200 tokens. Good news: each chunk holds about one "
        "idea, so its vector is sharp, and a question can match it very precisely. Bad news: the answer may "
        "lose its context. [Board: write “within 14 days of delivery”, 14 days for what?] Imagine a chunk from "
        "a shop's FAQ that says only: within 14 days of delivery. Fourteen days for what? Returns? Exchanges? "
        "The heading that explained it is in another chunk. And small chunks mean many more vectors to store "
        "and search.",
        "[Board: right box, large chunks] Large chunks, 600 or 800 tokens. Good news: a whole rule, with its "
        "heading and its exceptions, stays in one piece. Bad news: one vector now has to represent many "
        "topics. [Board: write returns + shipping + warranty in one chunk] If one chunk covers returns, "
        "shipping and warranty, its vector is a mix of all three, and it matches none of them strongly. We call "
        "that a diluted embedding. Large chunks also put more tokens into every prompt.",
        "[Board: arrow at 400, common starting point 400 / 50] Many teams start in the middle: 400 tokens, "
        "roughly 300 words, one or two paragraphs, with 50 tokens of overlap.",
        "What about overlap? It protects facts that sit on a boundary, as we saw in the last video. But it is "
        "not free. With size 400 and step 350, we create one chunk every 350 tokens instead of every 400. 400 "
        "divided by 350 is about 1.14, so about 14 percent more vectors to embed, store and pay for. Overlap of "
        "10 to 20 percent of the chunk size is a common starting range.",
        "[Slide {cost}]",
        "Now top-k. Multiply: 400 tokens per chunk, times top-k of 5, gives 2,000 tokens of context in every "
        "prompt, plus the instructions and the question. Double the chunk size, and every question carries "
        "about twice as much context. Raise top-k to 15 and you send 6,000 tokens. More context can help the "
        "model find the answer, but it costs more, answers slower, and adds passages that are not relevant, "
        "which can distract the model. We come back to that in Module 7.",
        "Too small a top-k has the opposite problem. With top-k of 1, if the best chunk comes second instead of "
        "first, the model never sees it, and a well-built assistant will honestly say it could not find the "
        "answer. A top-k of 3 to 8 is a common range; five is a sensible middle.",
        "[Slide {when}]",
        "The last column of this table is the important one. Chunk size and overlap are used only when a "
        "document is ingested. Changing them does nothing to chunks already in the database. Top-k is used at "
        "question time, so it takes effect on the very next question. And the embedding model is fixed per "
        "index: chunks and questions must use the same model, so changing it means re-embedding everything.",
        "[Slide {gotchas}]",
        "That gives us two gotchas. First: after changing size or overlap, re-ingest the document, but delete "
        "its old chunks first. Why? Ids are the document name plus the chunk number. Say a handbook made 30 "
        "chunks before, and the new setting makes 20. The upsert overwrites ids 0 to 19, but old chunks 20 to "
        "29 stay behind and can still be retrieved, with out-of-date text. So delete every vector whose source "
        "is that document, then ingest again.",
        "Second: overlap must be smaller than chunk size. The step is size minus overlap. With size 100 and "
        "overlap 100 the step is zero, the start never moves, and the loop never ends. That is why our chunker "
        "checks it on the first line.",
        "[Slide {choose}]",
        "So how do you choose? With a small experiment. Write 10 to 20 real questions about your documents, "
        "where you already know the answer. Ingest with one setting. Ask every question and look at the "
        "retrieved chunks. Count: was the right passage in the top-k? Then change one setting, re-ingest, and "
        "count again. Keep the winner. In Module 8 this idea becomes a proper evaluation metric.",
        "[Slide {r63}]",
        "Recap. Small chunks are precise but lose context; large chunks keep context but dilute meaning. "
        "Overlap protects boundaries and costs about 14 percent more vectors at 400 and 50. Context per "
        "question is roughly top-k times chunk size. Size and overlap need re-ingesting, with old chunks "
        "deleted first; top-k does not. And choose with real test questions.",
        "Practice: take one PDF and five questions, try chunk size 200 and then 800, and note whether the right "
        "passage is retrieved. In the capstone project, DocuMind, you will use this exact ingestion step "
        "inside one real system. Next, Module 7 puts every piece together: RAG from end to end.",
    ],
)

MODULE = dict(
    folder="Ingestion_and_Chunking", kicker="INGESTION & CHUNKING", title="Document ingestion and chunking",
    subtitle="From PDF pages to token chunks with page numbers",
    guide_intro="Slides with a small generic reference implementation (pypdf + tiktoken), plus three board "
                "segments: page → raw → clean text, the sliding window over three pages, and the chunk-size "
                "trade-off.",
    videos=[V61, V62, V63],
    guide_md="""
## Goal of this module

The learner can explain what an ingestion pipeline does to a PDF, step by step: extract text page by page
with pypdf, clean it with three rules, flatten the pages into one token list with a token→page map, and cut
it into fixed-size token windows with overlap (for example 400 tokens stepping by 350). They can work out
chunk starts, sizes and cited pages by hand, name the classic extraction problems (scanned pages, tables,
columns, headers), compare chunking strategies, and choose chunk size, overlap and top-k with a small
experiment instead of guessing.

## Where this sits in the course

- Before: Module 2 explained tokens and tiktoken; Modules 4 and 5 explained embeddings and vector
  databases, which store the chunks this module creates.
- Now: the offline half of a document assistant: extract → clean → chunk → embed + store, once per
  document.
- Next: Module 7 puts ingestion, retrieval and generation together and shows how bad chunks cause bad
  answers. Module 8 turns "was the right passage retrieved?" into an evaluation metric. After the course,
  the capstone project show this ingestion step inside one complete system.

## What we teach

Video 6.1 – Getting text out of PDFs

- The four ingestion steps, run once per document; garbage in, garbage out
- A PDF stores positioned characters, not paragraphs or tables
- `read_pdf_pages`: pages numbered from 1, blank pages skipped, page numbers kept, an error when no page
  has text
- `clean_text`: whitespace → one space, control characters removed, hyphen-split words joined (and the
  "selfservice" side effect)
- Scanned PDFs (no text layer, OCR needed), tables, two columns, repeated headers and footers

Video 6.2 – Chunking with tokens and overlap

- Why chunk: embedding input limit (8,191 tokens for text-embedding-3-small at the time of writing),
  precise retrieval, prompt cost
- Chunk with the embedding model's tokenizer (`cl100k_base`)
- Flatten pages into `tokens` plus `page_of`; guard `0 <= overlap < size`
- Sliding window: 400 tokens, step 400 − 50 = 350, decode back to text, cited page = page of the first
  token
- Worked example: 1,000 tokens over pages of 300 / 500 / 200 → 3 chunks (400, 400, 300 tokens), cited
  pages 1, 2, 2; edge case of 750 tokens → a 50-token duplicate tail
- The chunk record: id `doc-index`, 1,536 values, metadata text / source / page / chunk_index /
  token_count
- Other strategies: fixed characters, recursive, by structure, semantic, small-to-big

Video 6.3 – Choosing chunk size, overlap and top-k

- Small vs large chunk trade-off; diluted embeddings
- Overlap cost: at 400/50 about 14% more vectors (400 / 350 ≈ 1.14)
- Context per question ≈ top-k × chunk size = 5 × 400 = 2,000 tokens
- Which settings take effect at ingestion (size, overlap, embedding model) and at query time (top-k)
- Gotchas: delete a document's old chunks before re-ingesting; overlap must be smaller than chunk size
- Choosing with 10–20 test questions and the retrieved chunks

## Recording notes

- Board 6.1 (slide {clean}): three panels side by side. Write the raw string with visible \\n so learners
  see the line breaks. Circle the split word "manage- / ment" in both the page and the raw text.
- Board 6.2 (slide {window}): draw the page strip first, to scale if you can (300 / 500 / 200), then the
  three chunk bars one by one, with dashed lines down from each start to the strip to "look up" the
  page. Finish with the red rule: page of the FIRST token.
- Practice answer for 6.2 (1,500 tokens, 400 / 50): starts 0, 350, 700, 1,050, 1,400 → five chunks of
  400, 400, 400, 400 and 100 tokens (the 4th chunk is 1,050–1,449; the 5th is 1,400–1,499). Learners often
  stop at four or make the last chunk 450; walk through the loop literally if they ask.
- The code on the slides is a reference implementation, not a library. Keep it on screen long enough to
  read; the important lines are the guard, `page_of[start]` and `start += size - overlap`.
- Version-dependent facts (8,191-token input limit, 1,536 dimensions, which tokenizer a model uses) are
  hedged "at the time of writing"; check the provider's docs before recording.

## Common learner questions

- "Why tokens and not words?" Models read tokens, limits and prices are in tokens, and we chunk with the
  embedding model's own tokenizer (cl100k_base), so 400 tokens here is 400 tokens there.
- "Why does my answer cite page 4 when the sentence is on page 5?" A chunk cites the page of its first
  token; it can run onto the next page.
- "Can I just use bigger chunks to be safe?" Bigger chunks dilute the vector and cost more per question.
  Measure with test questions.
- "My PDF gave 'No extractable text'." It is almost certainly scanned. Run OCR first.
- "Does changing top-k need re-ingesting?" No. Only chunk size, overlap and the embedding model do.
""",
    notes_md="""
------------------------------------------------------------------------

## Video 6.1 – Getting text out of PDFs

### The ingestion pipeline

| **Step** | **What happens** |
|:---|:---|
| 1. Extract | Read the text of each page; keep page numbers |
| 2. Clean | Normalise whitespace, remove control characters, join split words |
| 3. Chunk | Cut into fixed-size token windows with overlap |
| 4. Embed + store | One vector per chunk, upserted with its text, source and page |

Ingestion runs once per document, ahead of time. Questions later only search what was stored, so the
quality of the extracted text limits everything after it: garbage in, garbage out.

### What a PDF stores

A PDF describes how a page looks: characters (glyphs) drawn at x, y positions with fonts and sizes, plus
lines and images. It has no "paragraph", "heading" or "table" objects. A text extractor rebuilds words and
lines from the positions. A scanned PDF may contain only an image of each page, with no characters at all.

### Step 1: read every page (pypdf)

```python
from pypdf import PdfReader

def read_pdf_pages(path):
    reader = PdfReader(path)
    pages = []
    for num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:                    # skip blank pages
            pages.append((num, text))
    if not pages:
        raise ValueError(f"No extractable text in {path}")
    return pages
```

- Pages are numbered from 1 (`start=1`), so citations match the page numbers in a PDF viewer.
- `or ""` guards against a page that returns nothing.
- Blank pages are skipped, but every kept page keeps its real number.
- Output: a list of `(page_number, text)` pairs.
- If no page has text, fail loudly instead of storing an empty document.

### Step 2: cleaning

```python
import re

def clean_text(text):
    text = re.sub(r"\\s+", " ", text)                          # 1
    text = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]", "", text)   # 2
    text = re.sub(r"(\\w)-\\s+(\\w)", r"\\1\\2", text)              # 3
    return text.strip()
```

| **Rule** | **Pattern** | **Effect** |
|:---|:---|:---|
| 1 | `\\s+` → one space | Line breaks, tabs and double spaces become single spaces |
| 2 | `[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]` → nothing | Invisible control characters are removed |
| 3 | `(\\w)-\\s+(\\w)` → `\\1\\2` | "manage- ment" becomes "management" |

Rule 3 works only because rule 1 has already turned the line break after the hyphen into a space. Side
effect: a real hyphen that falls at a line end is removed too ("self- service" → "selfservice"). Joining
split words helps retrieval more than this occasional loss hurts.

These rules do not remove headers, footers or page numbers. An optional extra step: find lines that
appear on most pages and drop them, before rule 1 joins every line into one.

### What goes wrong

| **Problem** | **What you get** | **Usual fix** |
|:---|:---|:---|
| Scanned PDF | Empty text, "No extractable text" | OCR (optical character recognition) before ingestion |
| Tables | Cells run together in one line; structure lost | Table-aware extraction, or convert tables to text rows |
| Two-column layout | Lines from both columns may be interleaved | Layout-aware extraction |
| Headers and footers | Company name, page numbers in many chunks | Detect and strip repeated lines |

First debugging step for a strange answer: look at the extracted text.

------------------------------------------------------------------------

## Video 6.2 – Chunking with tokens and overlap

### Why chunk

1. Input limits: `text-embedding-3-small` accepts up to 8,191 tokens per input (at the time of writing).
2. Precise retrieval: one vector per passage points at the exact rule; one vector per document is a blur
   of all its topics.
3. Prompt cost and speed: only the top few chunks go to the LLM.

### A reference chunker

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def chunk_pages(pages, size=400, overlap=50):
    if not 0 <= overlap < size:
        raise ValueError("overlap must be smaller than size")
    tokens, page_of = [], []
    for page_num, text in pages:
        ids = enc.encode(text)
        tokens.extend(ids)
        page_of.extend([page_num] * len(ids))

    chunks, start = [], 0
    while start < len(tokens):
        piece = tokens[start:start + size]
        chunks.append({
            "text": enc.decode(piece),
            "page": page_of[start],     # page of the first token
            "index": len(chunks),
            "token_count": len(piece),
        })
        start += size - overlap
    return chunks
```

- Chunk with the embedding model's tokenizer. `cl100k_base` is the tokenizer of OpenAI's
  `text-embedding-3` models, so these counts are the embedding model's counts. (At the time of writing,
  chat models such as the GPT-4o family use a newer tokenizer, `o200k_base`, so their counts for the same
  text differ slightly.)
- `page_of` has one entry per token: the page that token came from.
- Window = `size` tokens (400). Step = `size − overlap` = 350. The slice simply stops at the end of the
  list, so the last chunk can be shorter.
- The cited page is the page of the chunk's first token.
- Chunk starts are 0, 350, 700, … so the number of chunks is ⌈N / 350⌉ for N tokens.
- The guard prevents a step of 0 or less, which would make the loop run forever.

### Worked example

A PDF of 1,000 tokens: page 1 = tokens 0–299, page 2 = 300–799, page 3 = 800–999.

| **Chunk** | **Start** | **Tokens** | **Count** | **Pages inside** | **Cited page** |
|:---|:---|:---|:---|:---|:---|
| 0 | 0 | 0–399 | 400 | 1 and 2 | p.1 |
| 1 | 350 | 350–749 | 400 | 2 | p.2 |
| 2 | 700 | 700–999 | 300 | 2 and 3 | p.2 |
| – | 1,050 | stop (1,050 ≥ 1,000) | | | |

Tokens 350–399 and 700–749 appear in two chunks (the overlaps). Chunk 2 cites page 2 although 200 of its
300 tokens are on page 3: citations can point one page early.

Second example, 1,500 tokens: starts 0, 350, 700, 1,050, 1,400 → five chunks of 400, 400, 400, 400 and
100 tokens (⌈1,500 / 350⌉ = 5).

Edge case, 750 tokens: starts 0, 350, 700 → the third chunk (700–749, 50 tokens) lies entirely inside
chunk 1 (350–749). A tiny duplicate tail chunk is harmless but wasteful.

### Token boundaries

The window cuts at token positions, not at sentence ends, so a chunk can start or end mid-sentence or even
mid-word. Overlap reduces the damage; structure-aware splitting avoids it.

### The chunk record

```python
record = {
    "id": f"{doc_id}-{chunk['index']}",   # e.g. "handbook-2"
    "values": vector,                     # 1,536 floats (text-embedding-3-small)
    "metadata": {"text": chunk["text"], "source": doc_id, "page": chunk["page"],
                 "chunk_index": chunk["index"], "token_count": chunk["token_count"]},
}
index.upsert(vectors=[record])            # Pinecone-style client; batch many records per call
```

| **Field** | **Meaning** |
|:---|:---|
| `id` | Document id + chunk index; stable, so re-ingesting overwrites the same ids |
| `values` | The chunk's embedding |
| `text` | The chunk text, sent to the LLM later |
| `source` | Which document; used for citations, filters and deleting one document |
| `page` | Page of the first token, counted from 1 |
| `chunk_index` | Position of the chunk in the document: 0, 1, 2 … |
| `token_count` | Tokens in this chunk (≤ chunk size) |

### Other strategies

| **Strategy** | **How it splits** | **Good for** | **Watch out** |
|:---|:---|:---|:---|
| Fixed tokens | Every N tokens with overlap | Predictable, debuggable baseline | Cuts sentences |
| Fixed characters | Every N characters | Quick scripts | Token size varies |
| Recursive | Paragraphs, then sentences, then words | Keeping sentences whole | Uneven sizes |
| By structure | Headings, sections, clauses | Policies, manuals, contracts | Needs reliable headings |
| Semantic | Where embeddings show a topic change | Long, mixed documents | Slower, more embedding calls |
| Small-to-big (parent–child) | Search small chunks, return their larger parent | Precise search with full context | More storage and logic |

------------------------------------------------------------------------

## Video 6.3 – Choosing chunk size, overlap and top-k

### The trade-off

| | **Small chunks (100–200)** | **Large chunks (600–800)** |
|:---|:---|:---|
| Match quality | Sharp: one idea per vector | Diluted: one vector for many topics |
| Context | Answer may lose its heading or condition | Whole rule stays together |
| Storage | More vectors | Fewer vectors |
| Prompt size | Smaller per chunk | Larger per chunk |

A common starting point is 400 tokens (about 300 English words) with 50 overlap.

### Overlap cost

Chunks are created every `size − overlap` tokens instead of every `size`. At 400 / 50: 400 / 350 ≈ 1.14,
so about 14% more chunks to embed, store and search. A common starting range is 10–20% of the chunk size
(50 / 400 = 12.5%).

### Context per question

Context tokens ≈ top-k × chunk size = 5 × 400 = 2,000 tokens, plus the instructions and the question.
Top-k 15 → about 6,000 tokens. More context costs more, answers more slowly, and can add distracting,
irrelevant passages (Module 7). Too small a top-k (1) misses the answer whenever the best chunk ranks
second. A common range is 3–8.

### When each setting takes effect

| **Setting** | **Typical range** | **Takes effect** |
|:---|:---|:---|
| Chunk size | 200–800 tokens | At ingestion: re-ingest to change |
| Chunk overlap | 10–20% of size | At ingestion: re-ingest to change |
| Top-k | 3–8 | At query time: the next question |
| Embedding model | one per index | At ingestion: re-embed every document; questions must use the same model |

### Gotchas

1. Delete old chunks before re-ingesting. Ids are `doc-index`; if the new setting creates fewer chunks
   (say 20 instead of 30), ids 0–19 are overwritten but 20–29 remain with stale text and can still be
   retrieved. Delete every vector whose `source` is that document (by metadata filter or id prefix,
   depending on the database), then ingest again.
2. Overlap must be smaller than chunk size. Step = size − overlap; with size 100 and overlap 100 the step
   is 0 and the loop never ends. Check it before the loop.

### Choosing with evidence

1. Write 10–20 questions whose answers you know.
2. Ingest with one setting; ask each question; look at the retrieved chunks.
3. Count how often the right passage is in the top-k.
4. Change one setting, re-ingest, count again; keep the best.

Module 8 turns this count into an evaluation metric (retrieval recall / context recall).
""",
    quiz=[
        dict(q="A PDF was made by scanning paper pages. What will pypdf's extract_text() most likely return?",
             a="The text, perfectly", b="Little or nothing, because the pages have no text layer",
             c="400-character chunks", d="A translated version", correct="B",
             why="A scanned page is an image. pypdf finds no characters, so a careful pipeline raises 'No extractable text'. OCR is needed first."),
        dict(q="After extract_text(), a page contains 'The manage-\\nment office'. What does the three-rule clean_text produce?",
             a="The manage- ment office", b="The manage-ment office", c="The management office", d="The manage ment office",
             correct="C", why="Rule 1 turns the line break into a space; rule 3 joins letter-hyphen-space-letter into one word."),
        dict(q="Why should ingestion number pages from 1 rather than from 0?",
             a="pypdf cannot read page 0", b="So a cited 'page 3' matches page 3 in the PDF viewer",
             c="It makes chunks smaller", d="Embedding models require it", correct="B",
             why="Page numbers end up in citations, and people count pages from one: enumerate(reader.pages, start=1)."),
        dict(q="With chunk size 400 and overlap 50, how far does the window move each time?",
             a="50 tokens", b="400 tokens", c="450 tokens", d="350 tokens", correct="D",
             why="start += size − overlap = 400 − 50 = 350."),
        dict(q="A document has 1,000 tokens. With size 400 and overlap 50, how many chunks are made and how big is the last?",
             a="3 chunks; the last has 300 tokens", b="2 chunks; the last has 600 tokens",
             c="3 chunks; the last has 400 tokens", d="4 chunks; the last has 50 tokens", correct="A",
             why="Starts at 0, 350 and 700. The last chunk runs from 700 to the end (999): 300 tokens. The next start, 1,050, is past the end."),
        dict(q="Page 2 holds tokens 300–799 and page 3 holds 800–999. A chunk covers tokens 700–999. Using the first-token rule, which page does it cite?",
             a="Page 3, because most tokens are there", b="Page 2, the page of its first token",
             c="Both pages", d="No page", correct="B",
             why="page_of[start] gives the page of the first token, even if most of the chunk is on the next page."),
        dict(q="Top-k is 5 and chunk size is 400 tokens. Roughly how many context tokens go into each prompt?",
             a="400", b="405", c="2,000", d="20,000", correct="C",
             why="Context ≈ top-k × chunk size = 5 × 400 = 2,000 tokens, plus the instructions and the question."),
        dict(q="You change chunk size from 400 to 200. What happens to documents that are already in the vector database?",
             a="Their chunks are re-cut automatically", b="Nothing, until you delete their old chunks and ingest them again",
             c="Their answers become shorter", d="The database deletes them", correct="B",
             why="Chunk size and overlap are used only at ingestion. Top-k is the setting that applies on the next question."),
        dict(q="A handbook made 30 chunks (ids handbook-0 to handbook-29). You re-ingest with a setting that makes 20 chunks, without deleting first. What is the problem?",
             a="Nothing; upsert replaces the whole document", b="Old chunks handbook-20 to handbook-29 stay behind with stale text and can still be retrieved",
             c="The new chunks are rejected", d="All 30 old chunks are duplicated", correct="B",
             why="Upsert only overwrites the ids it writes (0–19). Delete every vector whose source is that document before re-ingesting."),
        dict(q="Why must chunk overlap be smaller than chunk size?",
             a="Vector databases reject overlapping chunks", b="Otherwise the step size − overlap is 0 or negative and the loop never moves forward",
             c="tiktoken cannot decode overlaps", d="It is not required", correct="B",
             why="start += size − overlap. With size 100 and overlap 100 the start never changes, so the loop runs forever."),
    ],
)
