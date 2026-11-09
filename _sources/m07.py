"""Module 7 – Retrieval-Augmented Generation: why RAG, the pipeline step by step, how RAG fails.

Schema: see m01.py.
"""
from style import *


def where():   # where the knowledge lives: RAG vs fine-tuning vs long context
    fig, ax = canvas()
    cols = [(2.75, TEAL, "RAG", ["Your documents,\nchunked in a vector DB", "Top-5 passages\nfound per question",
                                 "Prompt = rules +\npassages + question"],
             "lives in: your documents"),
            (8.0, PURPLE, "Fine-tuning", ["Many example\nQ&A pairs", "Training run\n(hours, costs money)",
                                         "New model weights\nprompt = question only"],
             "lives in: the weights"),
            (13.25, ORANGE, "Long context", ["The whole document,\nevery page", "Pasted into the\nprompt every time",
                                             "Prompt = rules +\n100K tokens + question"],
             "lives in: one huge prompt")]
    for x, c, head, steps, lives in cols:
        box(ax, x, 5.75, 4.6, 0.7, head, color=c, fill=c, fs=19, bold=True, tc="white")
        ys = [4.6, 3.35, 2.1]
        for y, s in zip(ys, steps):
            box(ax, x, y, 3.9, 0.95, s, color=c, fs=14)
        for a, b in zip(ys, ys[1:]):
            arrow(ax, x, a - 0.48, x, b + 0.48, color=c, lw=2)
        arrow(ax, x, ys[-1] - 0.48, x, 1.12, color=c, lw=2)
        pill(ax, x, 0.72, "model  →  answer", color=c, fs=13)
        text(ax, x, 0.12, lives, fs=14, bold=True, color=c)
    ax.plot([5.38, 5.38], [0.2, 6.1], color=LINE, lw=1, ls=(0, (4, 4)))
    ax.plot([10.62, 10.62], [0.2, 6.1], color=LINE, lw=1, ls=(0, (4, 4)))
    return fig


def paths():   # the two paths of any RAG system, with the module that taught each box
    fig, ax = canvas()
    bw, bh = 2.45, 0.95

    def node(x, y, s, c, f):
        box(ax, x, y, bw, bh, s, color=c, fs=14, bold=False)
        text(ax, x, y - bh / 2 - 0.2, f, fs=12, color=MUTED)

    y1, y2, y3 = 5.05, 3.05, 1.05
    text(ax, 0.2, y1 + 0.78, "INGEST PATH · once per document", fs=13, bold=True, color=BLUE, ha="left")
    text(ax, 0.2, y2 + 0.78, "QUERY PATH · every question", fs=13, bold=True, color=TEAL, ha="left")
    text(ax, 0.2, y3 + 0.78, "ANSWER", fs=13, bold=True, color=CORAL, ha="left")
    xs = [1.45, 4.3, 7.15, 10.0]
    for x, s, f in zip(xs, ["Load document\n(PDF, web page)", "Extract + clean\ntext", "Chunk: ~400 tokens\n50 overlap",
                            "Embed chunks\n(batched)"],
                       ["Module 6", "Module 6", "Module 6", "Module 4"]):
        node(x, y1, s, BLUE, f)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + bw / 2, y1, b - bw / 2, y1, color=BLUE, lw=2)
    for x, s, f in zip(xs[:3], ["User question", "Embed question\n(same model)", "Search: top-k\nby cosine"],
                       ["the app", "Module 4", "Module 5"]):
        node(x, y2, s, TEAL, f)
    for a, b in zip(xs[:2], xs[1:3]):
        arrow(ax, a + bw / 2, y2, b - bw / 2, y2, color=TEAL, lw=2)
    px, pw = 14.1, 2.7
    ax.add_patch(FancyBboxPatch((px - pw / 2, 2.35), pw, 3.4, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=TINT[PURPLE], ec=PURPLE, lw=2.5))
    text(ax, px, 4.45, "Vector DB", fs=19, bold=True, color=PURPLE)
    text(ax, px, 3.55, "one vector per chunk\n+ text, source, page\n(the metadata)", fs=13, color=INK)
    text(ax, px - 0.75, 2.12, "Module 5", fs=12, color=MUTED)
    arrow(ax, xs[3] + bw / 2, y1, px - pw / 2, y1, color=BLUE, lw=2)
    text(ax, 12.0, y1 + 0.25, "upsert", fs=12, color=BLUE)
    arrow(ax, xs[2] + bw / 2, y2, px - pw / 2, y2, color=TEAL, lw=2)
    text(ax, 10.6, y2 + 0.25, "query vector", fs=12, color=TEAL)
    xa = [14.1, 10.6, 7.1, 3.6]
    for x, s, f in zip(xa, ["Prompt = rules\n+ chunks + question", "LLM, low\ntemperature",
                            "Answer streams in\nwith citations", "Answer + sources\nshown to the user"],
                       ["Module 3", "Module 2", "Module 2", "the app"]):
        node(x, y3, s, CORAL, f)
    arrow(ax, px, 2.35, px, y3 + bh / 2, color=PURPLE, lw=2)
    text(ax, px + 0.15, 1.95, "top-k chunks", fs=12, color=PURPLE, ha="left")
    for a, b in zip(xa, xa[1:]):
        arrow(ax, a - bw / 2, y3, b + bw / 2, y3, color=CORAL, lw=2)
    return fig


def trace():   # one question through a generic RAG system, as a sequence diagram
    fig, ax = canvas()
    lanes = [("App (UI)", BLUE), ("Your pipeline\ncode", INK), ("Embedding\nmodel", TEAL),
             ("Vector\ndatabase", PURPLE), ("LLM", ORANGE)]
    xs = [1.5 + i * 3.25 for i in range(5)]
    for x, (name, c) in zip(xs, lanes):
        box(ax, x, 5.85, 2.5, 0.85, name, color=c, fs=14, bold=True)
        ax.plot([x, x], [5.42, 0.15], color=c, lw=1.4, ls=(0, (4, 4)))
    dash = (0, (5, 3))
    msgs = [(0, 1, "1  question", BLUE, "-"),
            (1, 2, "2  question text", TEAL, "-"),
            (2, 1, "3  query vector: 1,536 numbers", TEAL, dash),
            (1, 3, "4  vector, top_k = 5, filter", PURPLE, "-"),
            (3, 1, "5  5 chunks: text, source, page, score", PURPLE, dash),
            (1, 4, "6  prompt = rules + 5 chunks + question, stream", CORAL, "-"),
            (1, 0, "7  the sources list", INK, dash),
            (4, 0, "8  answer tokens, one by one", ORANGE, dash)]
    y = 4.9
    bg = dict(fc="white", ec="white", lw=0, pad=1.5)
    for a, b, label, c, ls in msgs:
        arrow(ax, xs[a], y, xs[b], y, color=c, lw=2, ls=ls)
        text(ax, (xs[a] + xs[b]) / 2, y + 0.22, label, fs=12, color=c, bold=True, bbox=bg)
        y -= 0.6
    return fig


def failmap():   # where RAG breaks + debug decision
    fig, ax = canvas()
    stages = [(BLUE, "Ingestion", ["Scanned PDF: no text", "Tables, columns garbled", "Wrong or blank pages"]),
              (TEAL, "Chunking", ["Answer split across\ntwo chunks", "Chunk too big: topic\ndiluted"]),
              (PURPLE, "Retrieval", ["Right chunk not in top-k", "Different words, same\nmeaning missed",
                                     "Near-duplicate chunks"]),
              (CORAL, "Generation", ["Ignores the context", "Wrong page cited", "Refuses though the\nanswer is there"])]
    for i, (c, head, items) in enumerate(stages):
        x = 2.05 + i * 3.97
        box(ax, x, 5.8, 3.6, 0.62, head, color=c, fill=c, fs=17, bold=True, tc="white")
        box(ax, x, 4.2, 3.6, 2.45, "", color=c)
        for j, it in enumerate(items):
            text(ax, x - 1.6, 5.08 - j * 0.72, "• " + it, fs=13, ha="left", va="top")
        if i < 3:
            arrow(ax, x + 1.82, 5.8, x + 2.15, 5.8, color=LINE, lw=2)
    yd = 1.35
    box(ax, 1.75, yd, 2.9, 1.0, "Wrong or missing\nanswer", color=RED, fs=14, bold=True)
    box(ax, 5.35, yd, 3.0, 1.0, "Inspect the retrieved\nsources FIRST", color=INK, fs=14, bold=True)
    box(ax, 9.05, yd, 3.1, 1.0, "Is the right passage\nin the top-k?", color=PURPLE, fs=14, bold=True)
    arrow(ax, 3.2, yd, 3.85, yd, color=INK, lw=2)
    arrow(ax, 6.85, yd, 7.5, yd, color=INK, lw=2)
    box(ax, 13.75, 2.2, 4.0, 0.9, "NO → fix ingestion,\nchunking or retrieval", color=BLUE, fs=14)
    box(ax, 13.75, 0.5, 4.0, 0.9, "YES → fix the prompt\nor the model", color=CORAL, fs=14)
    arrow(ax, 10.6, yd + 0.2, 11.75, 2.2, color=BLUE, lw=2)
    arrow(ax, 10.6, yd - 0.2, 11.75, 0.5, color=CORAL, lw=2)
    return fig


V71 = dict(
    num="7.1", title="Why RAG? RAG vs fine-tuning vs long context", minutes=10, segments="SW",
    slides=[
        dict(type="cards", key="three", title="Three ways to give a model your knowledge",
             cards=[("RAG", "Change the prompt. Find the few passages that matter and add them to every question."),
                    ("Fine-tuning", "Change the model. Train it further on your examples so the weights themselves "
                                    "change."),
                    ("Long context", "Change the prompt size. Paste the whole document into the prompt, every "
                                     "single time.")]),
        dict(type="diagram", key="where", title="Where the knowledge lives", draw="where",
             draw_list=["Three columns: RAG, Fine-tuning, Long context",
                        "RAG: documents chunked in a vector DB → top-5 passages → prompt = rules + passages + "
                        "question → model",
                        "Fine-tuning: example Q&A pairs → training run → new weights; prompt = question only",
                        "Long context: whole document → pasted every time → prompt with ~100K tokens → model",
                        "Bottom of each column: lives in documents / weights / one huge prompt"]),
        dict(type="two_col", key="ft", title="Fine-tuning: good at, bad at",
             left=("Good at", ["Style and tone: always formal, always short", "Fixed output formats",
                               "Domain wording and behaviour", "Shorter prompts once trained"]),
             right=("Bad at", ["Facts that change: retrain every update", "No citations: no page to check",
                               "Needs many good examples", "Hard to remove one wrong fact"])),
        dict(type="stats", key="long", title="Why not paste the whole document?",
             stats=[("128K", "tokens in the context window of a small model like gpt-4o-mini (at the time of "
                             "writing)"),
                    ("~2,000", "tokens of context per question with RAG (5 chunks × 400)"),
                    ("~50×", "more input tokens to paste a 200-page manual (~100K tokens)")],
             notes="~500 tokens per dense page is a rough rule of thumb; real documents vary. Context window "
                   "sizes change with every model release: check the provider's current documentation."),
        dict(type="table", key="compare", title="RAG vs fine-tuning vs long context",
             header=["", "RAG", "Fine-tuning", "Long context"], widths=[1.3, 1.6, 1.6, 1.6], size=14,
             rows=[["Knowledge lives in", "Your documents", "Model weights", "The prompt"],
                   ["Update a fact", "Re-ingest one document", "Retrain", "Paste the new version"],
                   ["Citations", "Yes: source + page", "No", "Possible, less reliable"],
                   ["Cost per question", "Low: few chunks", "Low", "High: whole document"],
                   ["Setup effort", "Medium: pipeline", "High: data + training", "Low"],
                   ["Access control", "Filter by metadata", "Everyone sees all", "Per request"]]),
        dict(type="bullets", key="combine", title="Not either-or: combining them",
             bullets=["Most company assistants start with RAG: facts stay in documents",
                      "Add fine-tuning later for tone or a strict format, not for facts",
                      "Long context helps when one document must be read whole: a contract review",
                      "Bigger windows let RAG send more chunks, not zero chunks",
                      "A sensible first version: pure RAG, top-k = 5"]),
    ],
    recap=dict(points=["RAG changes the prompt, fine-tuning changes the weights, long context changes the "
                       "prompt size", "Fine-tuning: style and format, not changing facts",
                       "Long context: costly per question; facts in the middle can be missed",
                       "RAG: fresh, cited, cheap per question"],
               task="Your company's leave policy changes every quarter and employees need the page to check. "
                    "Write two sentences: which approach would you choose, and why not the other two?",
               label="TRY THIS"),
    shot_plan=[("0:00–0:50", "Slide {cover}, Slide {v71}, hook", "S"),
               ("0:50–2:00", "Slide {three} three ways", "S"),
               ("2:00–4:40", "Slide {where} → board: where the knowledge lives", "W"),
               ("4:40–5:50", "Slide {ft} fine-tuning good / bad", "S"),
               ("5:50–7:10", "Slide {long} long context numbers", "S"),
               ("7:10–8:40", "Slide {compare} comparison table", "S"),
               ("8:40–9:20", "Slide {combine} combining them", "S"),
               ("9:20–10:00", "Slide {r71} recap", "S")],
    script=[
        "[Slide {cover}, Slide {v71}]",
        "Welcome to Module 7. Over the last few modules we learned the parts one by one: tokens and the LLM, "
        "prompts, embeddings, vector databases, and chunking. Now we put the parts together into one system, "
        "called RAG, Retrieval-Augmented Generation. And we start with a question every interviewer asks: why "
        "RAG at all? Why not simply train the model on your documents? Or paste the whole document into the "
        "prompt? Both sound simpler. Let us see why most document assistants do neither.",
        "[Slide {three}]",
        "There are three ways to give a language model knowledge it does not have. Imagine an HR help desk "
        "that must answer questions about your company's policies. The first way is RAG. RAG changes the "
        "prompt. For every question, we find the few passages that matter and add them to the prompt. The model "
        "itself never changes.",
        "The second is fine-tuning. Fine-tuning changes the model. You take an existing model and train it a "
        "little more on your own examples, so the numbers inside it, the weights, change.",
        "The third is long context. Long context changes the size of the prompt. Modern models accept very long "
        "inputs, so you could paste the entire policy document into the prompt with every question.",
        "[Slide {where} → Board]",
        "Let me draw where the knowledge lives in each case, because that one idea explains everything else.",
        "[Board: three columns, headings RAG, Fine-tuning, Long context]",
        "[Board: RAG column: documents in a vector DB → top-5 passages → prompt → model] In RAG, the knowledge "
        "lives in your documents. They are chunked and stored in a vector database, outside the model. For each "
        "question we fetch the top five passages, put them in the prompt with the rules and the question, and "
        "the model writes the answer. If the policy changes tomorrow, you ingest the new document. Nothing else "
        "changes.",
        "[Board: Fine-tuning column: example Q&A pairs → training run → new weights] In fine-tuning, you first "
        "prepare many examples, question and ideal answer. A training run, which takes time and money, adjusts "
        "the weights. Now the knowledge lives inside the weights. The prompt carries only the question. There is "
        "no document in the prompt, so there is nothing to cite.",
        "[Board: Long context column: whole document → pasted every time → huge prompt] In long context, the "
        "whole document goes into the prompt, every time. A 200-page manual is roughly a hundred thousand "
        "tokens. The knowledge lives in one huge prompt that you pay for on every single question.",
        "[Board: write under each column: documents / weights / huge prompt] So: documents, weights, or a huge "
        "prompt. Keep that picture in mind.",
        "[Slide {ft}]",
        "Fine-tuning is genuinely useful, just not for this job. It is good at style and tone: make every answer "
        "formal and three lines long. It is good at fixed output formats, and at the wording and behaviour of a "
        "domain. And once trained, your prompts can be shorter.",
        "But look at the right side. Facts that change are a problem: every policy update means another "
        "training run. There are no citations, because the answer comes from the weights, not from a page. You "
        "need many good examples. And if the model learned one wrong fact, it is hard to remove it. In a "
        "company, policies change every quarter and people want to check the page. That is exactly where "
        "fine-tuning is weakest.",
        "[Slide {long}]",
        "What about long context? At the time of writing, a small model like gpt-4o-mini has a context window "
        "of about 128 thousand tokens, so a short document really can fit. But compare the numbers. A typical "
        "RAG system sends about two thousand tokens of context per question: five chunks of four hundred "
        "tokens. Pasting a 200-page manual sends around a hundred thousand. That is roughly fifty times more "
        "input tokens, on every question, so roughly fifty times the input cost, and the answer also starts "
        "more slowly.",
        "There is a quality problem too. Researchers found that models use information at the start and end "
        "of a long prompt better than information buried in the middle. The paper is called “Lost in the "
        "Middle”. Newer models have improved, but a short prompt with the right five passages is still easier "
        "for the model to use well than a hundred pages where the answer hides on page 73.",
        "[Slide {compare}]",
        "Here is the full comparison. Where does the knowledge live? Documents, weights, prompt. To update a "
        "fact: RAG re-ingests one document, fine-tuning retrains, long context pastes the new version. "
        "Citations: RAG gives source and page; fine-tuning gives none; long context can cite, but less "
        "reliably. Cost per question: low for RAG, low for a fine-tuned model, high for long context. Setup "
        "effort: RAG needs a pipeline; fine-tuning needs data and training; long context is easy to start.",
        "And the last row is easy to forget: access control. With RAG, every chunk carries metadata, so you can "
        "filter which documents a user may search. An intern sees the general handbook; the finance team also "
        "sees finance policies. Once a fact is inside fine-tuned weights, everyone who uses the model can get "
        "at it.",
        "[Slide {combine}]",
        "This is not either-or. Most company assistants start with RAG, because facts belong in documents. Some "
        "add fine-tuning later for tone or a strict format. Never for facts. Long context is great when one "
        "document must be read as a whole, like reviewing a single contract. And bigger context windows do not "
        "kill RAG; they let RAG send more chunks. A sensible first version is pure RAG with a top-k of five.",
        "[Slide {r71}]",
        "Recap. RAG changes the prompt, fine-tuning changes the weights, long context changes the prompt size. "
        "Fine-tuning is for style and format, not for changing facts. Long context costs a lot per question and "
        "can miss facts in the middle. RAG is fresh, cited and cheap per question.",
        "Try this: your company's leave policy changes every quarter and employees need the page to check. In two "
        "sentences, which approach do you choose, and why not the other two? Next: we build the RAG pipeline "
        "step by step and follow one question all the way through it.",
    ],
)

V72 = dict(
    num="7.2", title="The RAG pipeline step by step", minutes=10, segments="SW",
    slides=[
        dict(type="diagram", key="paths", title="Two paths: ingest once, query every time", draw="paths",
             draw_list=["Top row (blue), INGEST: load document → extract + clean → chunk 400 / 50 → embed "
                        "(batched) → upsert into the vector DB",
                        "Middle row (teal), QUERY: user question → embed (same model) → search top-k by cosine",
                        "Vector DB box on the right: one vector per chunk + text, source, page",
                        "Bottom row (coral), right to left: prompt = rules + chunks + question → LLM → answer "
                        "streams with citations → answer + sources shown",
                        "Under each box, write the module that taught it (2, 3, 4, 5, 6)"]),
        dict(type="diagram", key="trace", title="One question, step by step", draw="trace",
             notes="Draw the five lanes first, then add the arrows one at a time, in order, while you talk.",
             draw_list=["Five lanes across the top: App, Your pipeline code, Embedding model, Vector database, LLM",
                        "1 app → pipeline: question", "2–3 pipeline ↔ embedding model: question text, back 1,536 "
                        "numbers", "4–5 pipeline ↔ vector DB: vector + top_k + filter, back 5 chunks with text, "
                        "source, page, score", "6 pipeline → LLM: rules + chunks + question, stream on",
                        "7 pipeline → app: the sources list; 8 LLM → app: answer tokens one by one (dashed)"]),
        dict(type="code", key="sketch1", title="Pipeline sketch, part 1: ingest and retrieve",
             label="PYTHON",
             code='''from openai import OpenAI
from pinecone import Pinecone
client = OpenAI()                       # reads OPENAI_API_KEY
index = Pinecone().Index("handbook")    # reads PINECONE_API_KEY

def embed(texts):                       # 1,536 numbers per text
    r = client.embeddings.create(model="text-embedding-3-small",
                                 input=texts)
    return [d.embedding for d in r.data]

def ingest(chunks):     # once per document; chunk = {id, text, source, page}
    vecs = embed([c["text"] for c in chunks])        # one batched call
    index.upsert(vectors=[(c["id"], v, c) for c, v in zip(chunks, vecs)])

def retrieve(question, k=5):            # every question
    res = index.query(vector=embed([question])[0], top_k=k,
                      include_metadata=True)
    return [{**m.metadata, "score": m.score} for m in res.matches]'''),
        dict(type="code", key="sketch2", title="Pipeline sketch, part 2: build the prompt and generate",
             label="PYTHON · OPENAI SDK v1",
             code='''RULES = ("Answer ONLY from the context passages. Cite (source, p.N). "
         "If the answer is not there, say you could not find it.")

def answer(question, k=5):
    chunks = retrieve(question, k)
    if not chunks:                      # nothing found: no LLM call
        return iter(["Sorry, I could not find that."]), []
    context = "\\n\\n".join(f"[{i}] ({c['source']}, p.{c['page']})\\n{c['text']}"
                          for i, c in enumerate(chunks, 1))
    stream = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.1, stream=True,
        messages=[{"role": "system", "content": RULES},
                  {"role": "user",
                   "content": f"Context:\\n{context}\\n\\nQuestion: {question}"}])
    tokens = (p.choices[0].delta.content or "" for p in stream if p.choices)
    return tokens, chunks'''),
        dict(type="code", key="msgs", title="What the LLM actually receives", label="MESSAGES",
             code='''system: Answer ONLY from the context passages.
        Cite (source, p.N). If the answer is not
        there, say you could not find it.

user:   Context:
        [1] (refund_policy, p.3)
        Refunds are accepted within 14 days of delivery...

        [2] (refund_policy, p.4)
        Items marked final sale cannot be returned...

        Question: What is the refund window?''',
             bullets=["Rules in the system message", "Each chunk numbered, with source and page",
                      "The model never sees the whole document: only these passages"]),
        dict(type="code", key="show", title="The app: stream the answer, then show the sources",
             label="PYTHON",
             code='''tokens, sources = answer("What is the refund window?")

for t in tokens:                    # words appear as they arrive
    print(t, end="", flush=True)

for i, s in enumerate(sources, 1):
    print(f"\\n[{i}] {s['source']} p.{s['page']}"
          f"  score {s['score']:.2f}")''',
             bullets=["A web app does the same: write tokens as they come",
                      "The score is cosine similarity: compare chunks, not a probability"]),
        dict(type="table", key="steps", title="Each step: what goes in, what comes out",
             header=["Step", "Input", "Output"], widths=[1.3, 2.2, 2.4], size=14,
             rows=[["Embed question", "Question text", "One vector, 1,536 numbers"],
                   ["Search", "Vector, top-k, optional filter", "k chunks: text, source, page, score"],
                   ["Build prompt", "Rules + chunks + question", "System and user messages"],
                   ["Generate", "Messages, low temperature, stream", "Answer tokens, one by one"],
                   ["Show", "Tokens + sources list", "Answer with citations you can check"]]),
    ],
    recap=dict(points=["Ingest path runs once per document; query path runs for every question",
                       "Query: embed → search top-k → build prompt → generate → show sources",
                       "Same embedding model for chunks and questions",
                       "The LLM sees only numbered chunks with source and page; stream the answer, keep the sources"],
               task="Take the sketch on slides {sketch1}–{sketch2}. Write down what changes if you set k = 1, "
                    "and what changes with k = 10: the prompt size, the cost, and the risk of missing the answer."),
    shot_plan=[("0:00–0:40", "Slide {v72}, hook", "S"),
               ("0:40–2:30", "Slide {paths} → board: ingest path and query path", "W"),
               ("2:30–4:50", "Slide {trace} → board: one question, step by step", "W"),
               ("4:50–6:10", "Slides {sketch1}–{sketch2} pipeline sketch", "S"),
               ("6:10–7:10", "Slide {msgs} what the LLM receives", "S"),
               ("7:10–8:10", "Slide {show} stream and show sources", "S"),
               ("8:10–9:20", "Slide {steps} input / output of each step", "S"),
               ("9:20–10:00", "Slide {r72} recap", "S")],
    script=[
        "[Slide {v72}]",
        "In the last video we decided why RAG. Now let us see how. We will draw the whole pipeline, follow one "
        "question through it, and then write the whole thing as a short Python sketch. Our example is an online "
        "store's help desk, and the question is: “What is the refund window?”",
        "[Slide {paths} → Board]",
        "Every RAG system has two paths. [Board: top row, blue, INGEST] The ingest path runs once per "
        "document. Load the document, extract and clean the text, cut it into chunks of about four hundred "
        "tokens with fifty tokens of overlap, embed every chunk in batches, and upsert the vectors into the "
        "vector database, together with metadata: the chunk text, the source file and the page number.",
        "[Board: middle row, teal, QUERY] The query path runs for every question. Embed the question with the "
        "same embedding model, then search the database for the top-k closest chunks by cosine similarity.",
        "[Board: bottom row, coral, ANSWER] Then build a prompt from the rules, the chunks and the question, "
        "send it to the LLM at a low temperature, stream the answer, and show the sources under it. [Board: "
        "write module numbers under the boxes] Notice: there is no new idea here. Every box is something you "
        "already learned in Modules 2 to 6. RAG is the glue.",
        "[Slide {trace} → Board]",
        "Now let us follow one question in time. [Board: five boxes across the top, a dashed line down from "
        "each] Five lanes: the app, your pipeline code, the embedding model, the vector database and the LLM. "
        "Time flows downwards.",
        "[Board: arrow 1, app → pipeline] The customer types the question and presses Ask. The app hands it "
        "to your pipeline code.",
        "[Board: arrows 2 and 3, pipeline ↔ embedding model] The pipeline sends the question text to the "
        "embedding model. One API call later, back comes a list of 1,536 numbers: the question's position in "
        "meaning-space.",
        "[Board: arrows 4 and 5, pipeline ↔ vector database] The pipeline sends that vector to the database, "
        "with top-k of five and an optional filter. The database returns the five closest chunks. Each comes "
        "back with its text, source, page and a similarity score.",
        "[Board: arrow 6, pipeline → LLM] Now the pipeline builds the prompt: rules, the five chunks, the "
        "question. It sends this to the LLM with streaming switched on.",
        "[Board: arrows 7 and 8, back to the app, dashed] Two things go back to the app. The list of sources, "
        "and the answer, which arrives a few tokens at a time. That is why the first words appear after a "
        "moment, instead of after the whole answer is written.",
        "[Slide {sketch1}]",
        "Here is the same thing in code. First, two clients; both read their API keys from environment "
        "variables, never from the code. The embed function sends a list of texts and gets back one vector per "
        "text. Ingest embeds all chunks in one batched call and upserts each one as an ID, a vector and its "
        "metadata. Retrieve embeds the question with the same model and asks for the top k matches, with the "
        "metadata included, so we get the text, source and page back. Here I used the Pinecone client, but "
        "Chroma or any other vector database works the same way: create the index once, upsert vectors with "
        "metadata, then query with a vector and a top-k. Only the method names change. And remember why one "
        "batched call matters: sending a hundred chunks in one request is much faster than a hundred separate "
        "requests.",
        "[Slide {sketch2}]",
        "Part two. The rules go in the system message: answer only from the context, cite source and page, and "
        "say so if the answer is missing. If retrieval finds nothing, we do not call the model at all. No "
        "context, no guessing. Otherwise we number each chunk with its source and page, and call the chat model "
        "with temperature 0.1 and stream set to true. We return two things: the stream of tokens, and the "
        "chunks, which are our sources. Ingest, retrieve, generate: the whole pipeline in about thirty lines.",
        "[Slide {msgs}]",
        "And this is what the model actually receives. The rules, then the numbered passages, then the "
        "question. Notice that the model has never seen the refund policy document. It only sees these few "
        "passages. That is why the retrieval step matters so much: the model can only be as good as what we "
        "hand it.",
        "[Slide {show}]",
        "In the app, we print the tokens as they arrive, then list the sources: number, source, page and score. "
        "A web app does exactly the same, just in a browser. Be careful with that score. It is a cosine "
        "similarity. A score of 0.62 does not mean 62 percent likely to be right. It is a closeness score, "
        "useful for comparing chunks with each other. Showing the sources lets the customer open page 3 and "
        "check, and, as we will see next video, it is your best debugging tool.",
        "[Slide {steps}]",
        "Here is the query path as a table: what goes into each step and what comes out. Clear inputs and "
        "outputs like these make each step easy to test on its own. If the search step returns the wrong "
        "chunks, you can see it without ever calling the LLM. So when you build your own, test retrieval first, "
        "with ten real questions, and only then connect the model. A good habit is to log every step: the "
        "question, the retrieved sources with their scores, and the final answer.",
        "[Slide {r72}]",
        "Recap. The ingest path runs once per document; the query path runs for every question. Embed, search, "
        "build the prompt, generate, show the sources. Use the same embedding model for chunks and questions. "
        "And the model only sees numbered chunks with source and page.",
        "Practice: take the sketch and write down what changes with k equal to 1, and with k equal to 10: the "
        "prompt size, the cost, and the risk of missing the answer. Next: how RAG fails, and how to find out "
        "where.",
    ],
)

V73 = dict(
    num="7.3", title="How RAG fails and how to debug it", minutes=10, segments="SW",
    slides=[
        dict(type="diagram", key="failmap", title="Where RAG can break", draw="failmap",
             draw_list=["Four stage boxes in a row: Ingestion → Chunking → Retrieval → Generation",
                        "Under each, 2–3 failures (scanned PDF; answer split; right chunk not in top-k; ignores context)",
                        "Bottom row: Wrong answer → Inspect the retrieved sources FIRST → Is the right passage in "
                        "the top-k?",
                        "NO → fix ingestion / chunking / retrieval. YES → fix prompt / model"]),
        dict(type="table", key="worked", title="Worked example: reading the retrieved sources",
             header=["Rank", "Source", "Page", "Score", "Has the answer?"], widths=[0.8, 2.2, 0.8, 0.9, 1.8],
             size=14, notes="Question: “What is the minimum attendance to sit the exam?” Illustrative "
                            "numbers. With k = 10, page 9 appears at rank 7.",
             rows=[["1", "student_handbook.pdf", "4", "0.71", "No: leave rules"],
                   ["2", "student_handbook.pdf", "5", "0.69", "No: medical leave"],
                   ["3", "exam_rules.pdf", "2", "0.66", "No: exam timetable"],
                   ["4", "student_handbook.pdf", "12", "0.64", "No: holidays"],
                   ["5", "exam_rules.pdf", "3", "0.63", "No: hall tickets"],
                   ["7 (at k = 10)", "student_handbook.pdf", "9", "0.61", "Yes: “75% attendance”"]]),
        dict(type="table", key="symptoms", title="Symptom → likely cause → first fix",
             header=["Symptom", "Likely cause", "First thing to try"], widths=[2.0, 2.0, 2.2], size=14,
             rows=[["“Couldn't find”, but it is in the document", "Text not extracted, or chunk not retrieved",
                    "Check the chunk text; raise top-k"],
                   ["Half an answer", "Answer split across chunks", "More overlap or bigger chunks"],
                   ["Right topic, wrong detail", "Similar chunks crowd out the right one", "Metadata filter; re-ranking"],
                   ["Fact not in any source", "Model added prior knowledge", "Tighten the prompt; lower temperature"],
                   ["Wrong page cited", "Chunk spans two pages", "Store the page range, not only the first page"],
                   ["Synonyms missed", "Vocabulary mismatch", "Hybrid search; query rewriting"]]),
        dict(type="cards", key="improve", title="Upgrades beyond a first version",
             cards=[("Hybrid search", "Combine vector search with keyword search, so exact terms like "
                                      "“Form 16” are never missed."),
                    ("Re-ranking", "Fetch 20 chunks, let a second model re-score them, keep the best 5."),
                    ("Query rewriting", "Let an LLM rephrase or expand the question before searching."),
                    ("Metadata filters", "Search only the right document, department or year, using the "
                                         "metadata stored with each chunk.")]),
        dict(type="cards", key="ragas", title="Measure it: the four RAGAS metrics",
             cards=[("Faithfulness", "Is every claim in the answer supported by the retrieved context?"),
                    ("Answer relevancy", "Does the answer actually address the question?"),
                    ("Context precision", "Are the useful chunks ranked near the top?"),
                    ("Context recall", "Did retrieval fetch everything needed for the reference answer?")]),
    ],
    recap=dict(points=["Failures happen at four stages: ingest, chunk, retrieve, generate",
                       "Always inspect the retrieved sources first", "Right passage missing → fix retrieval side; "
                       "present → fix prompt / model", "Measure with faithfulness, relevancy, context precision "
                       "and recall"],
               task="Write five questions about a document you know well, such as your college handbook. For "
                    "each wrong answer from a RAG app, look at the retrieved sources and name the stage that "
                    "failed."),
    shot_plan=[("0:00–0:40", "Slide {v73}, hook", "S"),
               ("0:40–4:00", "Slide {failmap} → board: failure map + decision", "W"),
               ("4:00–5:10", "Slide {worked} worked example: retrieved sources", "S"),
               ("5:10–6:50", "Slide {symptoms} symptom table", "S"),
               ("6:50–8:10", "Slide {improve} upgrades", "S"),
               ("8:10–9:20", "Slide {ragas} RAGAS metrics", "S"),
               ("9:20–10:00", "Slide {r73} recap", "S")],
    script=[
        "[Slide {v73}]",
        "Every RAG system gives wrong answers sometimes. Yours will too. The difference between a beginner and "
        "an engineer is not that the engineer's system never fails. It is that the engineer can find out where "
        "it failed, quickly, instead of randomly changing the prompt and hoping.",
        "[Slide {failmap} → Board]",
        "[Board: four boxes in a row: Ingestion, Chunking, Retrieval, Generation] A RAG answer passes through "
        "four stages, and each one can break.",
        "[Board: under Ingestion: scanned PDF, garbled tables] Ingestion. If the PDF is a scanned image, the "
        "text extractor finds no text at all, so there is nothing to search. Tables and two-column layouts can "
        "come out with words in the wrong order. If the text never got in correctly, no later step can save "
        "you.",
        "[Board: under Chunking: answer split, chunk too big] Chunking. The answer might sit right on the border "
        "between two chunks: the first half in chunk 7, the second half in chunk 8. Retrieval finds one and "
        "the answer comes out half complete. Or the chunk is so big that it covers five topics, and its meaning "
        "is diluted.",
        "[Board: under Retrieval: not in top-k, different words, duplicates] Retrieval. The right chunk exists "
        "but is not in the top five. Maybe the question uses different words: the customer asks about “money "
        "back” and the document says “reimbursement”. Embeddings handle much of this, but not all. Or many "
        "near-identical chunks, say the same clause from three versions of a policy, crowd out the one you need.",
        "[Board: under Generation: ignores context, wrong page, refuses] Generation. The model ignores the "
        "context and adds something from its own memory. It cites the wrong page. Or it is so careful that it "
        "refuses, even though the answer is sitting in chunk 2.",
        "[Board: bottom row: Wrong answer → Inspect the retrieved sources FIRST] So when an answer is wrong, "
        "where do you start? Not with the prompt. Start with the retrieved sources: the chunks the model "
        "actually saw. [Board: diamond: Is the right passage in the top-k?] Ask one question: is the right "
        "passage there?",
        "[Board: NO arrow → fix ingestion, chunking, retrieval] If no, the model never had a chance. Changing "
        "the prompt will not help. Look at ingestion, chunking and retrieval. [Board: YES arrow → fix prompt or "
        "model] If yes, retrieval did its job, and the problem is in the prompt or the model. This one question "
        "cuts your debugging work in half.",
        "[Slide {worked}]",
        "Let us try it on a college help desk. A student asks: what is the minimum attendance to sit the exam? "
        "The app says it could not find the answer. We look at the five sources. Leave rules, medical leave, "
        "the exam timetable, holidays, hall tickets. All about the right topic, none with the answer. So this "
        "is a retrieval problem, not a model problem. We raise top-k to ten and look again: page 9 of the "
        "handbook, the one that says seventy-five percent attendance, appears at rank seven. Now we know where "
        "to work. A higher top-k, re-ranking, or better chunking, but not the prompt. Notice also why the "
        "right page scored lower than the others: the handbook says “attendance”, while pages about leave "
        "share many words with the question. Similar topic is not the same as the right answer. Without "
        "looking at the sources, we might have spent an hour rewriting the prompt for nothing.",
        "[Slide {symptoms}]",
        "Here is a cheat sheet. The model says it couldn't find something that is in the document: check "
        "whether the text was extracted, and whether the chunk was retrieved; try a higher top-k. Half an "
        "answer: probably split across chunks, so try more overlap or bigger chunks. Right topic, wrong detail: "
        "similar chunks are crowding out the right one; filter by metadata, or re-rank. A fact that appears in "
        "no source: the model used its own memory; tighten the prompt and keep the temperature low.",
        "Wrong page cited: many simple pipelines store only the page where a chunk starts. If a chunk runs from "
        "page 3 onto page 4 and the answer is on page 4, the citation says page 3. Storing the page range fixes "
        "that. And synonyms missed: hybrid search or query rewriting, which we see next.",
        "[Slide {improve}]",
        "A simple first version is a clean baseline. Here are the upgrades professional systems add. Hybrid "
        "search: combine vector search with old-fashioned keyword search, so exact terms like a form number or "
        "a product code are never missed. Re-ranking: fetch twenty chunks, let a second, more careful model "
        "score each one against the question, and keep the best five. Query rewriting: let an LLM rephrase the "
        "question, or split it into parts, before searching. And metadata filters: search only the right "
        "document, department or year, using the metadata you stored with every chunk.",
        "[Slide {ragas}]",
        "Finally: how do you know that a change made things better and not worse? You measure. A popular "
        "open-source library called RAGAS has four main metrics. Faithfulness: is every claim in the answer "
        "supported by the retrieved context? That catches hallucination. Answer relevancy: does the answer "
        "actually address the question? Context precision: are the useful chunks ranked near the top? And "
        "context recall: did retrieval fetch everything needed, compared with a reference answer? Two metrics "
        "check generation, two check retrieval. Exactly the split from our board. Module 8 goes deeper.",
        "[Slide {r73}]",
        "Recap. RAG can fail at four stages: ingest, chunk, retrieve, generate. Always inspect the retrieved "
        "sources first. If the right passage is missing, fix the retrieval side; if it is there, fix the prompt "
        "or the model. And measure with faithfulness, relevancy, context precision and context recall.",
        "Practice: write five questions about a document you know well, such as your college handbook. For "
        "every wrong answer, look at the retrieved sources and decide which stage failed. In the capstone "
        "project, DocuMind, you will see this whole pipeline inside one real system. Next, in Module 8: running "
        "GenAI in production, with cost, latency, safety and evaluation.",
    ],
)

MODULE = dict(
    folder="RAG", kicker="RAG", title="Retrieval-Augmented Generation",
    subtitle="Why RAG, the pipeline step by step, and how RAG fails",
    guide_intro="Slides with board segments (where knowledge lives, the two paths and the one-question "
                "trace, the failure map), a short generic Python sketch of the whole pipeline, and a worked "
                "debugging example.",
    videos=[V71, V72, V73],
    guide_md="""
## Goal of this module

The learner can explain why document assistants use RAG instead of fine-tuning or pasting whole documents
into a long prompt; describe the ingest path and the query path of any RAG system and say what is passed at
each step; read a short Python sketch of the whole pipeline; and debug a wrong answer by stage, starting
from the retrieved sources.

## Where this sits in the course

- Before: Modules 2–6 explained each part on its own: LLMs, tokens and streaming; prompts; embeddings;
  vector databases; ingestion and chunking.
- Now: the parts joined into one pipeline, the case for RAG, and how RAG fails.
- Next: Module 8 covers running GenAI in production (cost, latency, safety, evaluation), including the
  RAGAS metrics previewed in Video 7.3.

## What we teach

Video 7.1 – Why RAG

- Three approaches: RAG changes the prompt, fine-tuning changes the weights, long context changes the
  prompt size
- Board: where the knowledge lives in each
- Fine-tuning: good for style and format, bad for changing facts, no citations
- Long context: ~128K-token window for a small current model (at the time of writing), but ~50× the input
  tokens for a 200-page manual, slower, and the "Lost in the Middle" effect
- Comparison table, including access control through metadata filters; combining approaches

Video 7.2 – The RAG pipeline step by step

- Board: ingest path (once per document) and query path (every question), each box labelled with the
  module that taught it
- Board: sequence diagram across app → pipeline → embedding model → vector DB → LLM → app
- Generic Python sketch (OpenAI SDK v1 + Pinecone client): `embed`, `ingest`, `retrieve`, `answer`
- What the LLM receives: rules in the system message, numbered chunks with source and page
- No chunks → no LLM call, fixed "could not find" message
- Streaming the tokens, then listing the sources; the score is cosine similarity, not a probability
- Input / output table for each query step

Video 7.3 – How RAG fails

- Board: failures at four stages + "inspect the retrieved sources first" decision
- Worked example (college handbook): the right page is missing at k = 5, appears at rank 7 at k = 10
- Symptom → cause → first fix table
- Upgrades: hybrid search, re-ranking, query rewriting, metadata filters
- RAGAS metrics: faithfulness, answer relevancy, context precision, context recall

## Recording notes

- Board 7.1: draw the three columns with the same three-box shape so the contrast is visual. Write the
  "lives in" line last, underlined.
- Board 7.2 (slide {paths}): draw the three rows, then write the module number under each box; the point
  is that RAG needs no new theory.
- Board 7.2 (slide {trace}): draw all five lanes first, then add arrows one at a time, numbered. Use dashed
  arrows for returns.
- Code slides {sketch1}–{sketch2}: walk top to bottom; do not read every character. Point at the same model
  name in `embed` for both ingest and retrieve, at `if not chunks`, and at `stream=True`.
- Model names and library calls are correct at the time of writing; if a newer SDK changes a call, say so
  on camera rather than editing the slide.
- The refund-policy text on slide {msgs} and the numbers on slide {worked} are illustrative.

## Common learner questions

- "Can't I just fine-tune on the documents?" You can train on them, but the model will not reliably recall
  exact facts, cannot cite a page, and needs retraining for every change. Use fine-tuning for style or
  format on top of RAG.
- "Context windows keep growing. Will RAG die?" Bigger windows let RAG send more passages. Cost per
  question, speed and the difficulty of finding facts in very long prompts still favour sending the
  relevant passages only.
- "Why does the score never reach 1.0?" It is a cosine similarity between a short question and a 400-token
  chunk; they are never the same text. Compare scores with each other, not with 1.
- "Why must the question use the same embedding model as the chunks?" Different models place text in
  different spaces (often with different numbers of dimensions); comparing across them is meaningless.
- "Why does the citation say page 3 when the answer is on page 4?" A simple pipeline stores the page where
  each chunk starts. A chunk that crosses a page break cites its first page; store a page range to fix it.
""",
    notes_md="""
------------------------------------------------------------------------

## Video 7.1 – Why RAG? RAG vs fine-tuning vs long context

### Three ways to give a model your knowledge

| **Approach** | **What changes** | **Where the knowledge lives** |
|:---|:---|:---|
| RAG | The prompt: relevant passages are added to each question | In your documents (chunked in a vector database) |
| Fine-tuning | The model: extra training changes its weights | Inside the weights |
| Long context | The prompt size: the whole document is pasted in every time | In one very large prompt |

### Fine-tuning

Good at: style and tone, fixed output formats, domain wording and behaviour, shorter prompts.

Bad at (for document Q&A): facts that change (each update needs a new training run), citations (the
answer comes from weights, not a page), data needs (many good examples), and removing a fact once learned.
Rule of thumb: fine-tune for how the model answers, not for what it knows.

### Long context

At the time of writing, a small model such as `gpt-4o-mini` has a context window of about 128K tokens
(check the provider's current documentation). A short document fits. But:

| **Per question** | **Input tokens (rough)** |
|:---|:---|
| RAG: 5 chunks × 400 tokens + rules + question | ~2,000–2,500 |
| Whole 200-page manual (~500 tokens per dense page) | ~100,000 |

That is roughly 50× the input tokens (and input cost) for every question, with a slower first token.
Research ("Lost in the Middle", Liu et al., 2023) found that models use information at the beginning and
end of long inputs better than information in the middle. Newer models do better, but a short, focused
prompt is still easier to use well.

### Comparison

| | **RAG** | **Fine-tuning** | **Long context** |
|:---|:---|:---|:---|
| Update a fact | Re-ingest one document | Retrain | Paste the new version |
| Citations | Yes: source + page | No | Possible, less reliable |
| Cost per question | Low | Low | High |
| Setup effort | Medium (a pipeline) | High (data + training) | Low |
| Access control | Metadata filters per user | None once in the weights | Per request |

### Combining

Start with RAG for facts. Add fine-tuning for tone or a strict format if needed. Use long context when one
document must be read whole (for example a contract review). Larger windows let RAG send more chunks.

------------------------------------------------------------------------

## Video 7.2 – The RAG pipeline step by step

### Two paths

| **Path** | **When it runs** | **Steps** |
|:---|:---|:---|
| Ingest | Once per document (and again when it changes) | Load → extract + clean text → chunk (~400 tokens, 50 overlap) → embed in batches → upsert vectors with metadata (text, source, page) |
| Query | Every question | Embed the question (same model) → search top-k by cosine → build prompt → LLM (low temperature, streamed) → show answer + sources |

### One question, step by step

| **Step** | **From → to** | **What is passed** |
|:---|:---|:---|
| 1 | App → pipeline | The question |
| 2 | Pipeline → embedding model | Question text |
| 3 | Embedding model → pipeline | Query vector (1,536 numbers for `text-embedding-3-small`) |
| 4 | Pipeline → vector DB | Vector, top_k (e.g. 5), optional metadata filter |
| 5 | Vector DB → pipeline | k chunks: text, source, page, similarity score |
| 6 | Pipeline → LLM | System message (rules) + user message (numbered chunks + question), stream on |
| 7 | Pipeline → app | The sources list |
| 8 | LLM → app | Answer tokens, as they are generated |

### The sketch

```python
from openai import OpenAI
from pinecone import Pinecone
client = OpenAI()                       # reads OPENAI_API_KEY
index = Pinecone().Index("handbook")    # reads PINECONE_API_KEY

def embed(texts):
    r = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in r.data]

def ingest(chunks):                     # chunk = {id, text, source, page}
    vecs = embed([c["text"] for c in chunks])
    index.upsert(vectors=[(c["id"], v, c) for c, v in zip(chunks, vecs)])

def retrieve(question, k=5):
    res = index.query(vector=embed([question])[0], top_k=k, include_metadata=True)
    return [{**m.metadata, "score": m.score} for m in res.matches]

RULES = ("Answer ONLY from the context passages. Cite (source, p.N). "
         "If the answer is not there, say you could not find it.")

def answer(question, k=5):
    chunks = retrieve(question, k)
    if not chunks:
        return iter(["Sorry, I could not find that."]), []
    context = "\\n\\n".join(f"[{i}] ({c['source']}, p.{c['page']})\\n{c['text']}"
                          for i, c in enumerate(chunks, 1))
    stream = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.1, stream=True,
        messages=[{"role": "system", "content": RULES},
                  {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: {question}"}])
    tokens = (p.choices[0].delta.content or "" for p in stream if p.choices)
    return tokens, chunks
```

Points to notice:

- API keys come from environment variables, never from the code.
- The same embedding model embeds the chunks and the question.
- `ingest` embeds all chunks in one batched call (very large documents are sent in several batches).
- `include_metadata=True` brings back the text, source and page stored at ingest time.
- No chunks → no LLM call: no context, no guessing.
- `answer` returns two things: the token stream and the sources. The app prints the tokens as they arrive
  (`print(t, end="", flush=True)` or a web framework's streaming widget), then lists the sources.
- Library calls follow the OpenAI Python SDK v1 and the Pinecone client at the time of writing; Chroma or
  another vector database works the same way with different method names.

### What the model receives

- System message: the rules (use only the context, cite source and page, say when the answer is missing).
- User message: "Context:" + numbered passages, each labelled `(source, p.N)`, + "Question: …".
- The model never sees the whole document, only these passages. It can only be as good as what retrieval
  hands it.

### The score

The score shown next to each source is the cosine similarity between the question vector and the chunk
vector. It is not a probability of being correct; use it to compare chunks with each other.

------------------------------------------------------------------------

## Video 7.3 – How RAG fails and how to debug it

### Failures by stage

| **Stage** | **Typical failures** |
|:---|:---|
| Ingestion | Scanned PDF with no text layer; tables and columns extracted in the wrong order; blank pages |
| Chunking | Answer split across two chunks; chunks so large that the topic is diluted |
| Retrieval | Right chunk not in the top-k; different words for the same idea; near-duplicate chunks crowd it out |
| Generation | Ignores the context and uses prior knowledge; cites the wrong page; refuses although the answer is present |

### Debugging order

1. Inspect the retrieved sources first: the chunks the model actually saw.
2. Is the passage that contains the answer among them?
   - No → the model never saw it. Investigate ingestion (is the text there at all?), chunking (is it
     split?) and retrieval (top-k, wording, filters).
   - Yes → retrieval worked. Investigate the prompt and the model (instructions, temperature, model size).

### Worked example

Question to a college help desk: "What is the minimum attendance to sit the exam?" At k = 5 the sources
are all about nearby topics (leave, medical leave, timetable, holidays, hall tickets), none with the answer.
At k = 10, handbook page 9 ("75% attendance") appears at rank 7. Diagnosis: retrieval, not the model. Fixes
to try: a higher top-k, re-ranking, or better chunking.

### Debugging checklist

- [ ] Does the PDF have selectable text? (Try copying a sentence from it.)
- [ ] Does any chunk contain the answer sentence? (Search the chunk texts.)
- [ ] Is that chunk in the top-k? At what rank with top-k = 10 or 15?
- [ ] Is the answer split across two neighbouring chunks?
- [ ] Does the cited page match the chunk's text? (Simple pipelines cite the starting page.)
- [ ] Does the answer contain any claim that appears in no source? (Faithfulness problem.)

### Symptom table

| **Symptom** | **Likely cause** | **First fix** |
|:---|:---|:---|
| "Couldn't find" but it is in the document | Not extracted or not retrieved | Inspect chunks; raise top-k |
| Half an answer | Split across chunks | More overlap or bigger chunks |
| Right topic, wrong detail | Similar chunks crowd out the right one | Metadata filter; re-ranking |
| Fact in no source | Model used prior knowledge | Tighten prompt; keep temperature low |
| Wrong page cited | Chunk spans a page break | Store a page range |
| Synonyms missed | Vocabulary mismatch | Hybrid search; query rewriting |

### Upgrades beyond a first version

- Hybrid search: vector search + keyword search (for example BM25), so exact codes and names are not missed.
- Re-ranking: retrieve more chunks (say 20), re-score them with a cross-encoder or LLM, keep the best 5.
- Query rewriting: rephrase, expand or split the question before searching.
- Metadata filters: restrict search to a document, department or year using the metadata stored with each
  chunk.
- Better chunking: split on headings and paragraphs instead of fixed token counts (Module 6).

### Measuring: RAGAS

| **Metric** | **Question it answers** | **Checks** |
|:---|:---|:---|
| Faithfulness | Is every claim in the answer supported by the retrieved context? | Generation |
| Answer relevancy | Does the answer address the question? | Generation |
| Context precision | Are the relevant chunks ranked near the top? | Retrieval |
| Context recall | Did retrieval fetch everything needed for the reference answer? | Retrieval |

Details and the evaluation workflow are in Module 8.
""",
    quiz=[
        dict(q="What does RAG change, compared with fine-tuning?",
             a="RAG changes the model's weights; fine-tuning changes the prompt",
             b="RAG adds retrieved passages to the prompt; fine-tuning changes the model's weights",
             c="Both change the weights", d="Both only change the temperature", correct="B",
             why="RAG leaves the model untouched and changes what goes into the prompt. Fine-tuning trains the "
                 "model further, changing its weights."),
        dict(q="A company's policies change every quarter and employees must see the source page. Which approach fits best?",
             a="Fine-tuning on the policies each quarter", b="RAG over the policy documents",
             c="Asking a plain LLM with no documents", d="Raising the temperature", correct="B",
             why="RAG updates by re-ingesting the new document and cites source and page. Fine-tuning needs "
                 "retraining and cannot cite."),
        dict(q="Why is pasting a whole 200-page manual into every prompt usually a poor choice?",
             a="It is impossible: no model accepts more than 4,000 tokens",
             b="It costs far more input tokens per question, is slower, and facts in the middle can be missed",
             c="Long prompts make the model refuse to answer", d="Vector databases cannot store long prompts",
             correct="B", why="Around 100K tokens per question vs ~2,000 for five 400-token chunks, plus the "
                              "'lost in the middle' effect."),
        dict(q="What is fine-tuning genuinely good for?",
             a="Storing facts that change every week", b="Giving page-level citations",
             c="Teaching a consistent style, tone or output format", d="Replacing the vector database",
             correct="C", why="Fine-tuning shapes how a model answers; facts that change belong in documents."),
        dict(q="Which steps run on the ingest path, once per document?",
             a="Embed the question and search top-k", b="Extract and clean text, chunk, embed chunks, upsert",
             c="Build the prompt and stream the answer", d="Show the sources to the user", correct="B",
             why="Ingestion prepares the documents once. The query path (embed question, search, prompt, "
                 "generate) runs for every question."),
        dict(q="Why must the question be embedded with the same model that embedded the chunks?",
             a="It is cheaper", b="Different models place text in different vector spaces, so the "
                                   "comparison would be meaningless",
             c="The LLM requires it", d="It makes streaming faster", correct="B",
             why="Cosine similarity only means something when both vectors come from the same embedding model "
                 "(and so have the same number of dimensions and the same space)."),
        dict(q="In a well-built RAG pipeline, what should happen if the vector search returns no chunks?",
             a="The model answers from its own knowledge", b="The app crashes",
             c="Return a fixed 'could not find it' message without calling the LLM",
             d="Raise the temperature and try again", correct="C",
             why="No context means no grounded answer. Skipping the LLM call avoids guessing and saves cost."),
        dict(q="A source is shown with 'score 0.62'. What does that mean?",
             a="There is a 62% chance the answer is correct", b="The cosine similarity to the question is about 0.62",
             c="62% of the chunk was used", d="The document is 62% indexed", correct="B",
             why="It is a closeness score for comparing chunks with each other, not a probability."),
        dict(q="An answer is wrong. You inspect the retrieved sources and the passage with the right answer is not there. Where is the problem?",
             a="In the prompt wording", b="In ingestion, chunking or retrieval", c="In the temperature setting",
             d="In the streaming code", correct="B",
             why="If the right passage never reached the model, no prompt change can fix it. Look at the "
                 "retrieval side."),
        dict(q="Which RAGAS metric checks whether every claim in the answer is supported by the retrieved context?",
             a="Context recall", b="Context precision", c="Answer relevancy", d="Faithfulness", correct="D",
             why="Faithfulness catches hallucinated claims. Context precision and recall check retrieval; answer "
                 "relevancy checks whether the question was addressed."),
    ],
)
