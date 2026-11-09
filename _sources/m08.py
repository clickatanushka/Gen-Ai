"""Module 8 – GenAI in production: observability and running GenAI for real. One video; closes the course
and hands over to the capstone project. Schema: see m01.py."""
from style import *


def observe():   # one request, seen three ways: trace, logs, metrics
    fig, ax = canvas()
    # 1. trace: one request, step by step on a time line
    text(ax, 0.3, 6.0, "1 · TRACE  one request, step by step", fs=15, bold=True, color=BLUE, ha="left")
    steps = [("embed", 0.1, TEAL), ("search", 0.15, PURPLE), ("LLM writes the answer · 2.0 s", 2.0, CORAL)]
    x0, scale, y = 0.5, 4.0, 5.1
    x = x0
    for name, sec, c in steps:
        w = sec * scale
        ax.add_patch(Rectangle((x, y - 0.3), w, 0.6, fc=TINT[c], ec=c, lw=2))
        if w > 2:
            text(ax, x + w / 2, y, name, fs=13, color=INK)
        x += w
    text(ax, x0, 4.45, "embed 0.1 s + search 0.15 s (the two thin bars)", fs=13, color=MUTED, ha="left")
    text(ax, x + 0.2, y, "total ≈ 2.25 s:\nalmost all of it\nis the LLM", fs=13, color=MUTED, ha="left")

    # 2. log lines
    text(ax, 0.3, 3.55, "2 · LOGS  what happened", fs=15, bold=True, color=TEAL, ha="left")
    ax.add_patch(FancyBboxPatch((0.4, 0.45), 7.6, 2.75, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#1B2433", ec="#1B2433"))
    logs = ["10:02:11  question received",
            "10:02:11  5 chunks found, best score 0.81",
            "10:02:13  answer sent · 2,350 tokens",
            "10:02:13  cost ₹0.036 · 2.3 s",
            "10:05:40  ERROR  LLM timeout, retried"]
    for i, line in enumerate(logs):
        text(ax, 0.65, 2.85 - i * 0.5, line, fs=13, color="#FF8A80" if "ERROR" in line else "white",
             ha="left", family="monospace")

    # 3. metrics dashboard
    text(ax, 8.6, 3.55, "3 · METRICS  numbers over time", fs=15, bold=True, color=CORAL, ha="left")
    tiles = [("p95 latency", "3.1 s", BLUE), ("cost today", "₹410", ORANGE),
             ("tokens today", "4.1 M", PURPLE), ("cache hit rate", "18%", TEAL)]
    for i, (lab, val, c) in enumerate(tiles):
        tx, ty = 10.05 + (i % 2) * 3.55, 2.45 - (i // 2) * 1.45
        box(ax, tx, ty, 3.25, 1.25, "", color=c)
        text(ax, tx, ty + 0.2, val, fs=20, bold=True, color=c)
        text(ax, tx, ty - 0.33, lab, fs=12.5, color=INK)
    box(ax, 13.75, 5.05, 4.1, 1.5, "Like a car dashboard:\nyou see trouble while driving,\nnot after the crash",
        color=GREEN, fs=13)
    return fig


V81 = dict(
    num="8.1", title="Running GenAI in the real world", minutes=10, segments="SW",
    slides=[
        dict(type="two_col", key="demo", title="A demo is not a product",
             left=("A demo", ["One person, a few questions", "You watch every answer", "Costs a few paise",
                              "If it breaks, you just try again"]),
             right=("A product", ["Hundreds of users, all day", "Nobody watches each answer",
                                  "A real monthly bill", "Things go wrong quietly"])),
        dict(type="cards", key="pillars", title="Observability: three things you record",
             cards=[("Logs", "A diary of what happened: question received, chunks found, answer sent, "
                             "errors."),
                    ("Metrics", "Numbers over time: tokens, cost, response time, cache hit rate. Shown on "
                                "a dashboard."),
                    ("Traces", "One request followed step by step: search took 0.15 s, the LLM took 2 s. "
                               "Shows where the time goes.")]),
        dict(type="diagram", key="observe", title="One request, seen three ways", draw="observe",
             draw_list=["Top: a time line for one request: embed 0.1 s, search 0.15 s, LLM 2 s (the long bar)",
                        "Write: total ≈ 2.25 s, almost all of it is the LLM",
                        "Bottom left: a dark box with 4–5 log lines, one ERROR line in red",
                        "Bottom right: four dashboard tiles: p95 latency, cost today, tokens today, cache hit rate",
                        "Top right: 'like a car dashboard'"]),
        dict(type="table", key="watch", title="What to watch in a GenAI app",
             header=["Metric", "Why it matters", "Example alert"], widths=[1.3, 2.0, 2.0],
             rows=[["Tokens per question", "Tokens are the bill", "Average jumps by 50%"],
                   ["Cost per day", "Stay inside the budget", "Above ₹500 today"],
                   ["Latency p50 / p95", "The wait users feel", "p95 above 6 seconds"],
                   ["Errors and timeouts", "Users get nothing", "More than 2% of requests fail"],
                   ["Retrieval score", "Did we find good passages?", "Best score drops below 0.5"],
                   ["User feedback", "Thumbs up or down", "More thumbs down than usual"]]),
        dict(type="cards", key="more", title="Four more things a real app needs",
             cards=[("Cost", "About ₹36 per 1,000 questions for a small RAG app (prices at the time of "
                             "writing). Watch it every day."),
                    ("Secrets", "API keys live in a .env file, never in your code or on GitHub."),
                    ("Guardrails", "Block tricks like “ignore your rules” and hide personal data "
                                   "before it reaches the model."),
                    ("Quality", "Check that answers are supported by the sources (faithfulness) using a "
                                "fixed set of test questions.")]),
        dict(type="table", key="journey", title="The whole journey", header=["Module", "What you can now explain"],
             widths=[1.3, 4.0], size=14,
             rows=[["1 · Generative AI", "What GenAI is, and why plain LLMs hallucinate"],
                   ["2 · How LLMs work", "Tokens, next-token prediction, temperature, streaming"],
                   ["3 · Prompting", "Roles, grounded prompts, citations, polite refusal"],
                   ["4 · Embeddings", "Text as numbers, cosine similarity"],
                   ["5 · Vector databases", "Index, records, metadata, top-k search"],
                   ["6 · Ingestion", "PDF to clean text to overlapping chunks"],
                   ["7 · RAG", "Retrieve, augment, generate, and how it fails"],
                   ["8 · Production", "Observability, cost, secrets, guardrails, quality"]]),
        dict(type="bullets", key="project", title="Let's start the project: DocuMind",
             bullets=["An assistant that reads company PDFs and answers questions",
                      "Every answer shows the document and page as proof",
                      "It uses every idea from this course: chunks, embeddings, a vector database, a grounded "
                      "prompt, RAG",
                      "No new theory needed, now we build it"], size=23,
             notes="First time students hear what DocuMind is. Keep it short and exciting; do not explain its "
                   "code here."),
    ],
    recap=dict(points=["A product needs eyes on it: logs, metrics, traces",
                       "Watch tokens, cost, latency, errors and quality",
                       "Keep keys secret; add guardrails; test answers against sources",
                       "Next: build DocuMind with everything you learned"],
               task="As you build DocuMind, note which module taught each part you build.", label="NEXT"),
    shot_plan=[("0:00–0:40", "Slide {cover}, Slide {v81}, hook", "S"),
               ("0:40–1:50", "Slide {demo} demo vs product", "S"),
               ("1:50–3:10", "Slide {pillars} logs, metrics, traces", "S"),
               ("3:10–5:30", "Slide {observe} → board: one request, seen three ways", "W"),
               ("5:30–6:40", "Slide {watch} what to watch", "S"),
               ("6:40–7:50", "Slide {more} four more things", "S"),
               ("7:50–8:50", "Slide {journey} the whole journey", "S"),
               ("8:50–9:40", "Slide {project} let's start the project", "S"),
               ("9:40–10:00", "Slide {r81} recap", "S")],
    script=[
        "[Slide {cover}, Slide {v81}]",
        "Welcome to the last module. You now know how an LLM writes, how to prompt it, how to find the right "
        "passages with embeddings and a vector database, and how RAG joins it all together. Today is short and "
        "simple: what changes when real people start using your app, and how do you keep an eye on it?",
        "[Slide {demo}]",
        "First, a demo is not a product. In a demo, one person asks a few questions and watches every answer. "
        "It costs a few paise, and if something breaks, you just try again. In a product, hundreds of people ask "
        "questions all day. Nobody reads every answer. There is a real bill at the end of the month. And the "
        "dangerous part: things go wrong quietly. An answer gets slower, a document stops being found, the cost "
        "doubles, and nobody notices until a user complains.",
        "[Slide {pillars}]",
        "So we need eyes on the system. That is called observability: being able to see what your app is doing "
        "from the outside. We record three things. Logs are a diary: question received, five chunks found, "
        "answer sent, and any errors. Metrics are numbers over time: how many tokens, how much money, how long "
        "people wait, how often the cache helped. We put them on a dashboard. And traces follow one single "
        "request step by step, so you can see exactly where the time went.",
        "[Slide {observe} → Board]",
        "Let me draw one request, seen three ways. [Board: a time line with three bars] At the top, the trace. "
        "Embedding the question takes about a tenth of a second. The vector search takes about fifteen "
        "hundredths. And then the long bar: the LLM writing the answer, about two seconds. [Board: write total "
        "about 2.25 seconds] So if users say the app is slow, you already know where to look. It is almost "
        "never the search; it is the model writing.",
        "[Board: dark box with log lines] Bottom left, the logs. Each line is a small fact with a time: question "
        "received, five chunks found with a best score of 0.81, answer sent with 2,350 tokens, cost and time. "
        "And here, in red, an error: the LLM timed out and we retried. Without this line, you would never know.",
        "[Board: four tiles] Bottom right, the metrics: response time for the slowest five percent, which we "
        "call p95, cost today, tokens today, and the cache hit rate. [Board: write 'like a car dashboard'] Think "
        "of a car dashboard. The speed and fuel gauges show trouble while you are driving, not after the crash.",
        "[Slide {watch}]",
        "So what should you watch in a GenAI app? Tokens per question, because tokens are the bill. Cost per "
        "day, to stay inside the budget. Latency: p50 is the typical wait, p95 is the slow one. Errors and "
        "timeouts. The retrieval score, which tells you whether good passages are still being found. And user "
        "feedback, a simple thumbs up or down. For each one, set an alert: for example, tell me if today's cost "
        "goes above five hundred rupees, or if the slow wait goes above six seconds.",
        "[Slide {more}]",
        "Four more things, one line each. Cost: for a small RAG app it is roughly thirty-six rupees for a "
        "thousand questions at today's prices, so watch it daily. Secrets: your API keys live in a dot env file, "
        "never in your code and never on GitHub. Guardrails: block tricks like “ignore your rules”, and "
        "hide personal data like phone or Aadhaar numbers before it reaches the model. And quality: keep a fixed "
        "set of test questions and check that every answer is supported by its sources. That is called "
        "faithfulness.",
        "[Slide {journey}]",
        "Now look how far you have come. Module 1, what generative AI is. Module 2, how LLMs work. Module 3, "
        "prompting. Modules 4 to 6, embeddings, vector databases and preparing documents. Module 7, RAG. And "
        "today, running it in the real world. Concepts first, all eight of them.",
        "[Slide {project}]",
        "And now it is time to use them. Our capstone project is DocuMind: an assistant that reads company PDFs "
        "and answers questions about them, and every answer shows the document and page as proof. It needs no "
        "new theory. Every piece of it is something you already understand: chunks, embeddings, a vector "
        "database, a grounded prompt, and RAG.",
        "[Slide {r81}]",
        "Quick recap. A real product needs eyes on it: logs, metrics and traces. Watch tokens, cost, latency, "
        "errors and quality. Keep keys secret, add guardrails, and test answers against their sources. Your task: "
        "as you build DocuMind, note which module taught each part you build. Thank you for learning with me. "
        "So let's start the project.",
    ],
)

MODULE = dict(
    folder="Production", kicker="PRODUCTION", title="Running GenAI in the real world",
    subtitle="Observability, and then we start the project",
    guide_intro="One short video: slides plus one board segment, then the hand-over to the capstone project.",
    videos=[V81],
    guide_md="""
## Goal of this module

The learner can explain, in simple words, why a demo is not a product, what observability means (logs,
metrics, traces), which numbers to watch in a GenAI app, and four basics of a real deployment: cost,
secrets, guardrails and quality checks. The video ends the course and introduces the capstone project.

## Where this sits in the course

- Before: Modules 1–7 taught GenAI, LLMs, prompting, embeddings, vector databases, ingestion and RAG.
- Now: one short look at running GenAI for real users.
- Next: the capstone project, DocuMind, which learners build using every concept from the course.

## What we teach

Video 8.1 – Running GenAI in the real world

- Demo vs product: many users, real money, silent failures
- Observability: logs, metrics, traces; the car-dashboard analogy
- Board: one request seen three ways (trace time line, log lines, metric tiles)
- What to watch, with example alerts
- One line each: cost, secrets in .env, guardrails, quality (faithfulness)
- The whole journey, Modules 1–8; introducing DocuMind; "So let's start the project."

## Recording notes

- Keep it light and quick; this is a wrap-up, not a deep dive.
- Board: draw the trace bars to scale so the LLM bar is clearly the longest.
- Prices and the ₹36 per 1,000 questions figure are approximate; check the provider's pricing page before
  recording.
- Students hear about DocuMind here for the first time. Describe what it does, not its code.

## Common learner questions

- "Do I need Grafana or a big tool?" No. A log file and a simple dashboard page are enough to start.
- "What is p95?" Sort all response times; p95 is the time that 95 out of 100 requests beat. It shows the
  slow experiences an average hides.
- "Is observability only for big companies?" No. Even a class project benefits from logging tokens, cost
  and errors.
""",
    notes_md="""
## Video 8.1 – Running GenAI in the real world

### A demo is not a product

| **Demo** | **Product** |
|:---|:---|
| One person, a few questions | Hundreds of users, all day |
| You read every answer | Nobody reads every answer |
| Costs a few paise | A real monthly bill |
| Broken? Try again | Things fail quietly until users complain |

### Observability

Observability means being able to see what your app is doing from the outside, while it runs.

| **Signal** | **What it is** | **GenAI example** |
|:---|:---|:---|
| Logs | Timestamped lines describing events | "5 chunks found, best score 0.81", "ERROR: LLM timeout" |
| Metrics | Numbers tracked over time, shown on a dashboard | tokens per day, cost per day, p95 latency, cache hit rate |
| Traces | One request followed step by step with timings | embed 0.1 s → search 0.15 s → LLM 2 s |

Analogy: a car dashboard shows speed and fuel while you drive, so you act before the crash.

### What to watch

| **Metric** | **Why** | **Example alert** |
|:---|:---|:---|
| Tokens per question | Tokens are the bill | Average jumps by 50% |
| Cost per day | Budget | Above ₹500 today |
| Latency p50 / p95 | Wait users feel (typical / slow) | p95 above 6 s |
| Errors and timeouts | Users get nothing | More than 2% fail |
| Retrieval score | Are good passages still found? | Best score below 0.5 |
| User feedback | Direct signal of quality | More thumbs down than usual |

p50 is the median response time. p95 is the time that 95 of every 100 requests are faster than.

### Four basics

- Cost: roughly ₹36 per 1,000 questions for a small RAG app (about 2,200 input and 150 output tokens per
  question on a small model, prices at the time of writing).
- Secrets: API keys in a `.env` file that is never committed; share a `.env.example` with fake values.
- Guardrails: block prompt injection ("ignore your rules", including text hidden inside documents) and mask
  personal data such as phone or Aadhaar numbers before sending text to the model.
- Quality: keep a fixed set of test questions and check faithfulness, that every claim is supported by the
  retrieved sources.

### Next: the capstone project

DocuMind is an assistant that reads company PDFs and answers questions about them, showing the document and
page as proof. It uses chunks, embeddings, a vector database, a grounded prompt and RAG: everything from
this course.
""",
    quiz=[
        dict(q="What does observability mean for a GenAI app?",
             a="Making the app look good", b="Being able to see what the app is doing while it runs",
             c="Training the model on more data", d="Hiding errors from users", correct="B",
             why="Observability is seeing the app's behaviour from outside through logs, metrics and traces."),
        dict(q="Which signal follows ONE request step by step with timings?",
             a="A log", b="A metric", c="A trace", d="A guardrail", correct="C",
             why="A trace shows each step of one request, for example search 0.15 s and LLM 2 s."),
        dict(q="In a typical RAG request, which step usually takes the most time?",
             a="Embedding the question", b="Vector search", c="The LLM writing the answer",
             d="Reading the .env file", correct="C",
             why="Generation writes the answer token by token and usually takes most of the time."),
        dict(q="What is p95 latency?",
             a="The average response time", b="The time 95 out of 100 requests are faster than",
             c="95% of the cost", d="The fastest response", correct="B",
             why="p95 shows the slow experiences that an average can hide."),
        dict(q="Why are tokens per question worth watching?",
             a="Tokens decide the bill", b="Tokens decide the font size", c="Tokens are passwords",
             d="Tokens never change", correct="A",
             why="Providers charge per token, so a jump in tokens is a jump in cost."),
        dict(q="Where should API keys be kept?",
             a="In the code", b="In a public GitHub repo", c="In a .env file that is not committed",
             d="In the prompt", correct="C",
             why="Keys in code or on GitHub can be stolen; a git-ignored .env keeps them private."),
        dict(q="What does faithfulness check?",
             a="Whether the answer is short", b="Whether every claim is supported by the retrieved sources",
             c="Whether the user is happy", d="Whether the PDF is long", correct="B",
             why="Faithfulness asks if the answer is backed by the passages the system found."),
        dict(q="What is the capstone project, DocuMind?",
             a="A new LLM", b="An assistant that answers questions from company PDFs with the page as proof",
             c="An image generator", d="A vector database company", correct="B",
             why="DocuMind is a RAG assistant that uses every concept from this course."),
    ],
)
