"""Course-level documents, built from the module files (run after build.py).

Writes 00_START_HERE.docx, 01_Course_Guide/*.docx, 04_Assessments/Final_Assessment.docx + .csv,
05_Cheat_Sheets/GenAI_Glossary.docx.
"""
import csv, glob, importlib, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
COURSE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build import docx, md_table, FMT   # noqa: E402

mods = []
for f in sorted(glob.glob(os.path.join(HERE, "m[0-9][0-9].py"))):
    n = int(re.search(r"m(\d+)", f).group(1))
    mods.append((n, importlib.import_module(f"m{n:02d}").MODULE))
total = sum(v["minutes"] for _, M in mods for v in M["videos"])
nvid = sum(len(M["videos"]) for _, M in mods)


def out(rel, md):
    path = os.path.join(COURSE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    docx(md, path)
    print("wrote", rel)


def fmt(v):
    return " + ".join(FMT[c] for c in v["segments"])


mod_rows = [(n, M["title"], len(M["videos"]), f"{sum(v['minutes'] for v in M['videos'])} min",
             f"`03_Modules/Module_{n:02d}_{M['folder']}`") for n, M in mods]

# ------------------------------------------------------------------ 00_START_HERE
out("00_START_HERE.docx", f"""# Generative AI Foundations: Start Here

A theory course of {len(mods)} modules, {nvid} videos and about {total // 60} hours {total % 60} minutes of recorded
content. Modules 1–8 teach the concepts of Generative AI step by step: what LLMs are and how they work,
prompting, embeddings, vector databases, document ingestion, Retrieval-Augmented Generation, and running
GenAI in production. After Module 8, learners build the capstone project, DocuMind: a document assistant
that answers questions about uploaded PDFs with citations. It uses every concept from this course.

## Folder map

| **Folder** | **What is inside** |
|:---|:---|
| `00_START_HERE.docx` | This page |
| `01_Course_Guide/` | Course overview, prep and recording plan |
| `03_Modules/Module_NN_*/` | Per module: `1_Module_Guide`, `2_Theory_Notes`, `3_Video_Scripts`, `4_Slides_ModuleNN.pptx`, `5_Quiz` (.docx + .csv) |
| `04_Assessments/` | Final assessment drawn from all modules |
| `05_Cheat_Sheets/` | Generative AI glossary |
| `_sources/` | Generators: `build.py`, one `mNN.py` content file per module, diagram style, slide template |

## Modules

{md_table(["Module", "Title", "Videos", "Length", "Folder"], mod_rows)}

## How to use the files when recording

1. Read the module's `1_Module_Guide.docx`: goal, what each video teaches, recording notes.
2. Open `4_Slides_ModuleNN.pptx` in presenter view. Recording cues are only in the speaker notes,
   never on the slides: video title slides say which segments the video uses, and diagram slides marked
   "✏ DRAW THIS SLIDE" list what to draw on the handwriting board.
3. Record from `3_Video_Scripts.docx`: each video has a shot plan (time, what is on screen, format) and
   the full spoken script with \\[Slide N\\], \\[Board: …\\] and \\[Screen: …\\] cues.
4. `2_Theory_Notes.docx` is the learner hand-out; `5_Quiz.csv` imports into most LMS quiz tools.

## Changing content

Edit the module's `_sources/mNN.py` and rebuild with
`python build.py NN` (then `python course_docs.py` for these course-level files). Needs
`python-pptx`, `python-docx`, `matplotlib`, `pypandoc_binary`.
""")

# ------------------------------------------------------------------ 1_Course_Overview
rows = []
for n, M in mods:
    for v in M["videos"]:
        rows.append((v["num"], v["title"], f"{v['minutes']} min", fmt(v)))
out("01_Course_Guide/1_Course_Overview.docx", f"""# Course Overview

## What this course is

A theory course on Generative AI, from first principles to Retrieval-Augmented Generation (RAG) and
production. Modules 1–8 teach each concept on its own, with everyday examples and short generic code.
The course ends by introducing the capstone project, which learners build next: DocuMind, a document
assistant that answers questions about uploaded PDFs with citations, using every concept taught here.

## Who it is for

- Beginners to Generative AI who want to understand how modern LLM applications work.
- Comfortable with basic Python; no machine-learning maths required. Every term is explained the first
  time it is used.

## Learning outcomes

By the end, a learner can:

1. Place Generative AI within AI, ML and deep learning, and name the three ways a plain LLM fails in a
   company (hallucination, knowledge cutoff, no private data).
2. Explain tokens, next-token prediction, attention, the context window, temperature and streaming.
3. Design a grounded prompt with citations and a refusal rule, and test it.
4. Explain embeddings, compute cosine similarity by hand, and read similarity scores correctly.
5. Explain what a vector database adds over brute-force search, and how records are stored and queried.
6. Trace text from a PDF page to a chunk, and choose chunk size, overlap and top-k with reasons.
7. Trace one question through a RAG system end to end, compare RAG with fine-tuning and long context, and
   debug a bad answer stage by stage.
8. Estimate cost and latency per question, and explain async processing, guardrails, semantic caching,
   RAGAS evaluation and observability.
9. Recognise every one of these concepts in a complete RAG project, ready for the capstone.

## Format

{nvid} videos, about {total} minutes in total. Segments: slides (S), handwriting board (W) where diagram
slides are drawn live, and a few short screen recordings (R) of a Python shell. No
coding assignments: each video ends with a "Try this" or practice task, and each module has a quiz.

## Modules

{md_table(["Module", "Title", "Videos", "Length", "Folder"], mod_rows)}

## Every video

{md_table(["Video", "Title", "Length", "Format"], rows)}
""")

# ------------------------------------------------------------------ 3_Prep_and_Recording_Plan
plan = []
for n, M in mods:
    for v in M["videos"]:
        board = any(s["type"] == "diagram" for s in v["slides"])
        plan.append((v["num"], v["title"], f"{v['minutes']} min", fmt(v), "yes" if board else ""))
out("01_Course_Guide/2_Prep_and_Recording_Plan.docx", f"""# Prep and Recording Plan

## Before recording

- A Python shell with `openai`, `tiktoken`, `numpy` and `pypdf` for the short code demos.
- One small, text-based sample PDF (2–5 pages, e.g. a short company policy) and one scanned PDF (no text
  layer) for Module 6.
- Handwriting board or tablet. Each diagram slide's speaker notes list exactly what to draw.

## Segment codes

| **Code** | **Meaning** |
|:---|:---|
| S | Slides on screen, voice over |
| W | Handwriting board: draw the diagram slide live, using the slide as your reference |
| R | Screen recording (a Python shell) |

## Recording order and checklist

Record module by module; each video is self-contained and ends with its recap slide.

{md_table(["Video", "Title", "Length", "Format", "Board"], plan)}

## Timing

Scripts are written at about 95 spoken words per runtime minute; the rest of the time is drawing,
pauses and screen work. If a video runs long, shorten the board segment first, not the recap.

## Prices and versions

Model prices, context sizes and library APIs change. The scripts say "at the time of writing" wherever a
number can go out of date; check the OpenAI and Pinecone pricing pages before recording Module 8.
""")

# ------------------------------------------------------------------ Final assessment
qs = []
for n, M in mods:
    for q in M["quiz"][:4]:
        qs.append((n, q))
md = ["# Final Assessment", "",
      f"{len(qs)} questions, four from each module. One correct answer each. Pass mark: 70%.", ""]
for i, (n, q) in enumerate(qs, 1):
    md += [f"### Q{i}. (Module {n}) {q['q']}", ""] + [f"- {k}\\) {q[k.lower()]}" for k in "ABCD"] + \
          ["", f"Correct answer: {q['correct']}", "", f"Explanation: {q['why']}", ""]
out("04_Assessments/Final_Assessment.docx", "\n".join(md))
with open(os.path.join(COURSE, "04_Assessments", "Final_Assessment.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["module", "question", "option_a", "option_b", "option_c", "option_d", "correct", "explanation"])
    for n, q in qs:
        w.writerow([n, q["q"], q["a"], q["b"], q["c"], q["d"], q["correct"], q["why"]])
print("wrote 04_Assessments/Final_Assessment.csv")

# ------------------------------------------------------------------ Glossary
G = [
    ("Generative AI", "Deep-learning models that create new content (text, images, code, audio).", 1),
    ("LLM (large language model)", "A generative model for text, trained to predict the next token.", 1),
    ("Hallucination", "A fluent, confident output that is false or unsupported.", 1),
    ("Knowledge cutoff", "The date after which a model has no training data.", 1),
    ("RAG", "Retrieval-Augmented Generation: retrieve relevant passages, add them to the prompt, generate.", 1),
    ("Token", "A piece of text (word, part of a word, punctuation) that models read and write; the unit of cost and limits.", 2),
    ("Tokenizer / BPE", "Splits text into tokens; byte-pair encoding merges frequent character pairs into pieces.", 2),
    ("cl100k_base", "A common OpenAI tokenizer; used by text-embedding-3-small and older GPT-4 models.", 2),
    ("Next-token prediction", "The model scores every possible next token, picks one, appends it, repeats.", 2),
    ("Transformer / attention", "Architecture where each token weighs every other token to build context.", 2),
    ("Context window", "Maximum tokens per call, prompt and answer together.", 2),
    ("Temperature", "Reshapes next-token probabilities: low = focused and repeatable, high = varied.", 2),
    ("Streaming", "Sending tokens to the user as they are generated.", 2),
    ("System / user / assistant roles", "Message roles in a chat request: rules, the request, the model's replies.", 3),
    ("Few-shot prompting", "Including a few worked examples in the prompt.", 3),
    ("Grounding", "Instructing and checking that answers come only from the given context.", 3),
    ("Prompt injection", "Text that tries to override the system's instructions, e.g. hidden inside a PDF.", 3),
    ("Embedding", "A list of numbers that represents the meaning of a text.", 4),
    ("Dimension", "How many numbers an embedding has; 1,536 for text-embedding-3-small.", 4),
    ("Cosine similarity", "Similarity of two vectors by the angle between them; 1 = same direction.", 4),
    ("Vector database", "Stores vectors with metadata and finds nearest neighbours quickly.", 5),
    ("ANN / HNSW", "Approximate nearest-neighbour search; HNSW is a layered-graph index for it.", 5),
    ("Index / record / upsert", "A Pinecone collection; one stored item (id, values, metadata); insert-or-overwrite.", 5),
    ("top-k", "How many most-similar chunks retrieval returns (often 3–10).", 5),
    ("Metadata filter", "Restricting a search by stored fields, e.g. doc_name = refund_policy.", 5),
    ("Chunk / overlap", "A passage cut from a document; tokens shared with the previous chunk.", 6),
    ("OCR", "Optical character recognition: reading text from images of pages.", 6),
    ("Fine-tuning", "Further training a model's weights on your examples; changes behaviour, not a fact store.", 7),
    ("Re-ranking / hybrid search", "Re-scoring retrieved chunks with a stronger model; combining keyword and vector search.", 7),
    ("Semantic cache", "Reusing an earlier answer when a new question's embedding is similar enough.", 8),
    ("Guardrails", "Input and output checks: injection, PII, grounding, off-topic.", 8),
    ("RAGAS", "Library scoring faithfulness, answer relevancy, context precision and context recall.", 8),
    ("p50 / p95 latency", "Median and 95th-percentile response times.", 8),
    ("Observability", "Measuring tokens, cost, latency and quality for every request.", 8),
]
out("05_Cheat_Sheets/GenAI_Glossary.docx", "# Generative AI Glossary\n\nTerms in the order the course introduces them.\n\n" +
    md_table(["Term", "Meaning", "Module"], G) + "\n")
