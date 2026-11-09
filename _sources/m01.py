"""Module 1 – Introduction to Generative AI: what it is, where LLMs fail, and the landscape.

SCHEMA (every mNN.py follows it; build.py reads MODULE):
  folder, kicker, title, subtitle, guide_intro, script_intro (optional)
  videos: list of
    num "1.2", title, minutes, segments "SW" (S slides, W handwriting board, R screen recording, F face cam)
    slides: list of dicts, each with a type and optional key (for {key} slide numbers) and notes:
      bullets  title, bullets [..] (≤ 6, short), size (optional, default 22)
      two_col  title, left (heading, [lines]), right (heading, [lines])
      cards    title, cards [(heading, body)] 2–4 cards, numbered 1..n
      stats    title, stats [(big value, label)] 2–4
      code     title, label "PYTHON", code "…", bullets [..] optional (side column)
      table    title, header [..], rows [[..]], widths [..] optional, size optional
      diagram  title, draw "function name in this file", draw_list [..] (what to hand-draw, goes in notes)
    recap: points [..] (3–5), task "…", label (optional, default "PRACTICE TASK")
    shot_plan: [(time, on screen, format code)] – may use {v11} {r11} {key} for slide numbers
    script: [paragraphs]; cue paragraphs look like "[Slide {v11}]", "[Slide {anatomy} → Board]", "[Board: …]",
            "[Screen: …]". {key} works in scripts, shot plans, recap tasks, guide_md and notes_md;
            unknown {words} are left as they are.
    Auto keys: cover = 1, v<num> = video title slide (v11 for 1.1), r<num> = its recap slide.
  guide_md, notes_md: markdown. quiz: list of dicts q, a, b, c, d, correct ("A".."D"), why.
"""
from style import *


def nest():   # AI ⊃ ML ⊃ DL ⊃ GenAI
    fig, ax = canvas()
    cx, cy = 4.2, 3.05
    rings = [(8.0, 6.0, BLUE, "Artificial Intelligence"), (6.6, 4.6, TEAL, "Machine Learning"),
             (4.8, 3.25, PURPLE, "Deep Learning"), (2.9, 1.75, CORAL, "Generative AI")]
    for w, h, c, label in rings:
        ax.add_patch(Ellipse((cx, cy), w, h, fc=TINT[c], ec=c, lw=2.5))
    for (w, h, c, label), dy in zip(rings, (2.6, 1.95, 1.27, 0)):
        text(ax, cx, cy + dy, label, fs=17 if label != "Generative AI" else 18, bold=True, color=c)
    items = [(BLUE, "AI", "any machine that acts smart", "a chess engine with hand-written rules"),
             (TEAL, "Machine Learning", "learns patterns from examples", "a spam filter trained on labelled emails"),
             (PURPLE, "Deep Learning", "ML with many-layer neural networks", "face unlock on your phone"),
             (CORAL, "Generative AI", "deep models that create new content", "ChatGPT, Claude, Gemini")]
    for i, (c, head, what, ex) in enumerate(items):
        y = 5.55 - i * 1.5
        box(ax, 12.15, y - 0.3, 7.3, 1.38, "", color=c)
        text(ax, 8.7, y + 0.15, head, fs=16, bold=True, color=c, ha="left")
        text(ax, 8.7, y - 0.47, f"{what}\ne.g. {ex}", fs=13, color=INK, ha="left")
    return fig


def anatomy():   # generic GenAI app: user -> app -> (optional retrieval) -> LLM API -> answer
    fig, ax = canvas()
    yt = 4.55
    box(ax, 1.55, yt, 2.5, 1.35, "User\nasks a question", color=BLUE, fs=15)
    box(ax, 6.0, yt, 3.2, 1.35, "Your app\nbuilds the prompt", color=TEAL, fs=15)
    box(ax, 12.55, yt, 4.3, 1.35, "LLM API\nGPT · Claude · Gemini · Llama", color=CORAL, fs=15)
    # user <-> app
    arrow(ax, 2.8, yt + 0.3, 4.4, yt + 0.3, color=BLUE)
    text(ax, 3.6, yt + 0.62, "question", fs=13, color=BLUE)
    arrow(ax, 4.4, yt - 0.3, 2.8, yt - 0.3, color=CORAL)
    text(ax, 3.6, yt - 0.62, "answer", fs=13, color=CORAL)
    # app <-> LLM
    arrow(ax, 7.6, yt + 0.3, 10.4, yt + 0.3, color=TEAL)
    text(ax, 9.0, yt + 0.62, "prompt = tokens in", fs=13, color=TEAL)
    arrow(ax, 10.4, yt - 0.3, 7.6, yt - 0.3, color=CORAL)
    text(ax, 9.0, yt - 0.62, "answer = tokens out", fs=13, color=CORAL)
    text(ax, 12.55, yt + 1.0, "you pay per token, in and out", fs=13, color=MUTED)
    # optional retrieval
    yb = 1.6
    box(ax, 6.0, yb, 3.6, 1.35, "Retrieval (optional)\nfind relevant passages", color=PURPLE, fs=15, ls="--")
    arrow(ax, 5.6, yt - 0.68, 5.6, yb + 0.68, color=PURPLE)
    text(ax, 5.45, 3.1, "search", fs=13, color=PURPLE, ha="right")
    arrow(ax, 6.4, yb + 0.68, 6.4, yt - 0.68, color=PURPLE)
    text(ax, 6.55, 3.1, "top passages", fs=13, color=PURPLE, ha="left")
    box(ax, 12.55, yb, 4.3, 1.35, "Your documents\npolicies · FAQs · handbooks", color=GREEN, fs=15)
    arrow(ax, 10.4, yb, 7.8, yb, color=GREEN)
    text(ax, 9.1, yb + 0.3, "prepared in advance", fs=13, color=GREEN)
    pill(ax, 12.55, 0.42, "Retrieval + LLM = RAG (Module 7)", color=PURPLE, fs=15)
    return fig


V11 = dict(
    num="1.1", title="What Generative AI is", minutes=8, segments="SW",
    slides=[
        dict(type="diagram", key="nest", title="From AI to Generative AI", draw="nest",
             draw_list=["Four nested circles: AI, Machine Learning, Deep Learning, Generative AI",
                        "AI: chess engine with rules", "ML: spam filter learns from labelled emails",
                        "DL: face unlock, many-layer neural network",
                        "GenAI: ChatGPT / Claude / Gemini – creates new content"]),
        dict(type="two_col", key="disc", title="Two kinds of model",
             left=("Discriminative: picks an answer", ["Is this email spam? Yes / No",
                                                       "Which category is this ticket?",
                                                       "Output: a label or a number",
                                                       "Learns the boundary between groups"]),
             right=("Generative: creates new things", ["Write a reply to this customer",
                                                          "Summarise this 40-page policy",
                                                          "Output: text, image, code, audio",
                                                          "Learns what the data looks like"])),
        dict(type="cards", key="what", title="What Generative AI produces",
             cards=[("Text", "Answers, summaries, emails, reports. ChatGPT, Claude, Gemini."),
                    ("Code", "Functions, tests, explanations. GitHub Copilot, Claude Code."),
                    ("Images", "Pictures from a sentence. DALL·E, Midjourney, Imagen."),
                    ("Audio & video", "Voices, music, short clips. ElevenLabs, Sora, Veo.")]),
        dict(type="bullets", key="why", title="Why it took off now",
             bullets=["2017: the Transformer, a new neural-network design (Module 2)",
                      "Enormous amounts of text from the web, books and code",
                      "Powerful GPUs to train on all of it",
                      "2022: ChatGPT puts it in everyone's hands",
                      "Today: any developer can call a model with a few lines of code"]),
    ],
    recap=dict(points=["AI ⊃ Machine Learning ⊃ Deep Learning ⊃ Generative AI",
                       "Discriminative models choose a label; generative models create new content",
                       "Large Language Models (LLMs) are the text branch of Generative AI",
                       "Transformers + data + GPUs made it possible; APIs made it available"],
               task="Write down three tasks from your own work or study. For each, decide: is it a "
                    "discriminative job (pick a label) or a generative job (create something)?",
               label="TRY THIS"),
    shot_plan=[("0:00–0:50", "Slide {cover}, Slide {v11}, hook", "S"),
               ("0:50–3:30", "Slide {nest} → board: four nested circles", "W"),
               ("3:30–5:20", "Slide {disc} two kinds of model", "S"),
               ("5:20–6:30", "Slide {what} what GenAI produces", "S"),
               ("6:30–7:20", "Slide {why} why it took off now", "S"),
               ("7:20–8:00", "Slide {r11} recap", "S")],
    script=[
        "[Slide {cover}, Slide {v11}]",
        "Welcome to Generative AI Foundations. Over the next eight modules we will understand, step by step, "
        "how tools like ChatGPT actually work, and how companies build useful assistants on top of them. "
        "Assistants that answer questions from a policy handbook, help customer support teams, or explain a "
        "college rule book. You do not need any background in AI. If you can read a little Python, you are "
        "ready. We go concept first, with simple examples, and we build up the vocabulary slowly.",
        "Let us start at the very top. People use the words AI, machine learning, deep learning and "
        "generative AI as if they mean the same thing. They do not. They fit inside each other, like rings.",
        "[Slide {nest} → Board]",
        "[Board: draw a big circle, label it AI] The biggest ring is artificial intelligence: any machine that "
        "behaves in a way we would call smart. An old chess engine with thousands of hand-written rules is AI. "
        "Nobody taught it from examples; a programmer typed every rule.",
        "[Board: a smaller circle inside, label it Machine Learning] Inside it, machine learning. Here we stop "
        "writing rules. We give the machine examples and it finds the pattern itself. A spam filter sees "
        "thousands of emails labelled spam or not spam, and learns which words and senders give spam away.",
        "[Board: a smaller circle inside, label it Deep Learning] Inside that, deep learning: machine learning "
        "with neural networks that have many layers. Deep learning is what made computers good at messy data "
        "like photos, speech and language. Face unlock on your phone is deep learning.",
        "[Board: the smallest circle, label it Generative AI] And in the centre, generative AI: deep learning "
        "models that do not just recognise things, they create new things. New text, new images, new code. "
        "ChatGPT, Claude and Gemini live here. Everything that is generative AI is also deep learning, machine "
        "learning and AI. Not the other way round.",
        "[Slide {disc}]",
        "So what is really new? Most machine learning before 2020 was discriminative. A discriminative model "
        "picks an answer from a fixed set. Is this email spam, yes or no? Which department should handle this "
        "ticket? Its output is a label or a number. It learns where the boundary between the groups is.",
        "A generative model learns what the data itself looks like, well enough to produce new examples. Ask it "
        "to write a reply to an angry customer, and it writes one that has never existed before. Ask it to "
        "summarise a 40-page policy, and it produces a summary. The output is open-ended: a paragraph, an "
        "image, a function. A simple way to remember it: discriminative models answer “which one?”, "
        "generative models answer “write me one”.",
        "Real systems often use both ideas together. Think of a help desk assistant. First it has to find the "
        "few passages in the handbook that relate to your question. That is a ranking job: score every passage, "
        "keep the best ones. Then it writes an answer from those passages. That is a generative job. We will "
        "study both halves in this course.",
        "[Slide {what}]",
        "Generative AI produces four main kinds of output. Text: answers, summaries, emails, reports. Code: "
        "tools like GitHub Copilot and Claude Code write and explain programs. Images: you type a sentence and "
        "get a picture. And audio and video: synthetic voices, music, short clips. Models that handle several "
        "of these at once are called multimodal.",
        "In this course we focus on text, because most business assistants are text systems. The models that "
        "generate text are called large language models, or LLMs. In Module 2 we open one up and see how it "
        "actually writes.",
        "[Slide {why}]",
        "Why did all this happen in the last few years? Three ingredients came together. In 2017 researchers "
        "published a new neural-network design called the Transformer; we will meet it properly in Module 2. "
        "Second, enormous amounts of text from the web, books and code to learn from. Third, powerful GPUs that "
        "can train on all of that text. Then in late 2022 ChatGPT put this technology in everyone's hands. And "
        "today any developer can use a model with a few lines of code, which is exactly what we will do.",
        "[Slide {r11}]",
        "Recap. AI contains machine learning, which contains deep learning, which contains generative AI. "
        "Discriminative models choose a label; generative models create new content. LLMs are the text branch "
        "of generative AI. Transformers, data and GPUs made them possible, and simple APIs made them available "
        "to everyone.",
        "Try this: write down three tasks from your own work or study and decide for each one: is it picking a "
        "label, or creating something new? In the next video: what LLMs are brilliant at, and the three ways "
        "they fail when a company tries to use them.",
    ],
)

V12 = dict(
    num="1.2", title="What LLMs can and cannot do", minutes=8, segments="S",
    slides=[
        dict(type="two_col", key="strong", title="Strong at, weak at",
             left=("Strong at", ["Writing fluent, well-structured text", "Summarising and rewriting",
                                 "Following instructions and formats", "Combining ideas from many passages"]),
             right=("Weak at", ["Facts they never saw in training", "Anything after their knowledge cutoff",
                                "Your private files and policies", "Saying “I don't know” on their own"])),
        dict(type="cards", key="fail", title="Three ways a plain LLM fails",
             cards=[("Hallucination", "It produces a confident, fluent answer that is simply false, because it "
                                      "writes what sounds likely, not what is checked."),
                    ("Knowledge cutoff", "Training stopped on a date. Anything newer, like this year's policy, "
                                         "does not exist for the model."),
                    ("No private data", "It never read your company's documents. Ask about your own leave rules "
                                        "and it can only guess.")]),
        dict(type="two_col", key="leave", title="Same question, two systems",
             left=("Plain chatbot", ["Q: How many leave days can I carry over?",
                                     "A: “Usually 5 to 10 days, depending on the company.”",
                                     "Sounds right. It is a guess.", "No source. You cannot check it."]),
             right=("Document assistant",
                    ["Q: How many leave days can I carry over?",
                     "A: “Up to 8 days; they expire on 31 March (Source: leave_policy, p.4).”",
                     "Answer comes from your own policy", "Citation lets anyone verify it"])),
        dict(type="bullets", key="fix", title="The general fix: Retrieval-Augmented Generation",
             bullets=["No private data → find the right passages in your documents and give them to the model",
                      "Knowledge cutoff → the passages are as new as the documents you add",
                      "Hallucination → rules: use only the passages, cite them, say so if the answer is missing",
                      "Retrieve → Augment the prompt → Generate the answer",
                      "Modules 4–7 build this up piece by piece"]),
    ],
    recap=dict(points=["LLMs write fluent text, summarise, follow formats", "They hallucinate, have a cutoff, and "
                       "never saw your files", "RAG retrieves your passages and puts them in the prompt",
                       "Citations make answers checkable"],
               task="Ask any chatbot a question about a document only you have (your college rules, a lease). "
                    "Note what it does: guess, refuse, or ask for the text?", label="TRY THIS"),
    shot_plan=[("0:00–0:40", "Slide {v12}, hook", "S"), ("0:40–2:30", "Slide {strong} strong / weak", "S"),
               ("2:30–5:00", "Slide {fail} three failures", "S"), ("5:00–6:20", "Slide {leave} two systems", "S"),
               ("6:20–7:20", "Slide {fix} the general fix: RAG", "S"), ("7:20–8:00", "Slide {r12} recap", "S")],
    script=[
        "[Slide {v12}]",
        "Large language models can feel like magic. They write emails, explain code, summarise long reports. "
        "So why can a company not simply give every employee a chatbot and stop there? Because LLMs have a few "
        "weak spots, and in a business those weak spots are expensive. Let us look at both sides honestly.",
        "[Slide {strong}]",
        "First, what they are strong at. Writing fluent, well-organised text. Summarising and rewriting: make "
        "this shorter, make it formal, turn it into bullet points. Following instructions and formats, like "
        "“answer in three sentences and cite the page”. And combining ideas from several passages into one "
        "answer. Keep that last strength in mind; it becomes very important later in the course.",
        "Now the weak side. An LLM only knows what was in its training data. It knows nothing after its "
        "knowledge cutoff. It has never seen your private files. And, left alone, it rarely says “I don't "
        "know”. It prefers to produce something.",
        "[Slide {fail}]",
        "These weak spots show up as three failures. The first is hallucination. The model gives a confident, "
        "fluent, well-formatted answer that is false. Why? Because an LLM writes the words that are most likely "
        "to come next. Likely and true are not the same thing. In Module 2 you will see exactly how this "
        "word-by-word writing works, and why it makes hallucination possible.",
        "The second failure is the knowledge cutoff. Training stopped on some date. A policy that changed last "
        "month simply does not exist for the model. It will happily tell you the old rule. "
        "Every model has a cutoff date, and providers usually publish it, so check it before you trust an "
        "answer about recent events.",
        "The third failure matters most for a business: no private data. The model never read your HR handbook, "
        "your product manuals, your contracts or your standard operating procedures. Ask it about them and the "
        "best it can do is guess what a typical company might say. The same is true for a college: a "
        "general chatbot has never read your college's attendance rules, so any number it gives you is a "
        "guess dressed up as a fact.",
        "[Slide {leave}]",
        "Here is the same question asked to two systems. An employee asks: how many leave days can I carry over "
        "into next year? A plain chatbot answers: usually 5 to 10 days, depending on the company. It sounds "
        "reasonable. It is a guess, and there is no source to check. If your policy actually says 8 days, and "
        "they expire on 31 March, the employee may plan a holiday they cannot take.",
        "Now an assistant that can read the company's own documents. It answers: up to 8 days, they expire on "
        "31 March, source leave policy, page 4. The answer comes from your own policy, and the citation means "
        "anyone can open page 4 and verify it. That difference, checkable instead of plausible, is the whole "
        "reason document assistants are so popular with companies, colleges and online shops.",
        "[Slide {fix}]",
        "So how do we fix the three failures in general? For missing private data: before asking the model, we "
        "find the right passages in your documents and put them into the prompt. For the cutoff: those passages "
        "are as fresh as the documents you added this morning. For hallucination: we give the model clear "
        "rules. Use only the given passages, cite them, and say clearly when the answer is not there. Module 3 "
        "teaches you how to write those rules.",
        "This pattern, find the passages first and then let the model write, has a name: Retrieval-Augmented "
        "Generation, or RAG. Retrieval finds; augmentation adds what was found to the prompt; generation writes. "
        "RAG is the backbone of this course. Modules 4, 5 and 6 build the retrieval half, and Module 7 puts "
        "the whole pipeline together. Notice one more thing: RAG does not change the model at all. We do not "
        "retrain anything. We only change what we put into the prompt. That is why it is cheap, quick to "
        "update, and so widely used.",
        "[Slide {r12}]",
        "Recap. LLMs write, summarise and follow formats very well. They hallucinate, they have a cutoff, and "
        "they never saw your files. RAG retrieves your passages and puts them in the prompt, and citations make "
        "the answer checkable.",
        "Try this: ask any chatbot a question about a document only you have, like your college rules or a "
        "rental agreement. Does it guess, refuse, or ask you for the text? In the next video: the GenAI "
        "landscape, how apps actually talk to a model, and the map of this course.",
    ],
)

V13 = dict(
    num="1.3", title="The GenAI landscape and this course", minutes=9, segments="SWR",
    slides=[
        dict(type="table", key="families", title="Model families and providers (at the time of writing)",
             header=["Provider", "Model family", "How you use it"], widths=[1.3, 1.6, 3.1],
             rows=[["OpenAI", "GPT (e.g. gpt-4o-mini)", "Closed: API and the ChatGPT app"],
                   ["Anthropic", "Claude", "Closed: API and the Claude apps"],
                   ["Google", "Gemini (and open Gemma)", "Closed API; Gemma weights are downloadable"],
                   ["Meta", "Llama", "Open weights: download and run it yourself"],
                   ["Mistral AI", "Mistral", "Both: open-weight models and a paid API"]]),
        dict(type="two_col", key="open", title="Closed API models vs open-weight models",
             left=("Closed model, via an API", ["Runs on the provider's servers",
                                                         "No hardware to manage; pay per token",
                                                         "Often the strongest models",
                                                         "Your text leaves your building: read the terms"]),
             right=("Open-weight model, run yourself", ["You download the weights (the learned numbers)",
                                                        "Run on your own GPUs or a cloud host",
                                                        "Data stays with you; full control",
                                                        "You handle updates, scaling and safety"])),
        dict(type="code", key="call", title="Calling a model through an API", label="PYTHON · OPENAI SDK v1",
             code='''from openai import OpenAI

client = OpenAI()   # reads the OPENAI_API_KEY environment variable

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful HR assistant."},
        {"role": "user", "content": "Explain notice period in two lines."},
    ],
    temperature=0.2,
)
print(response.choices[0].message.content)   # the generated text
print(response.usage)                        # tokens in, tokens out'''),
        dict(type="stats", key="cost", title="Paying per token: a quick sum",
             stats=[("1,000", "questions a day, each 2,000 tokens in + 300 tokens out"),
                    ("$0.30", "input: 2,000,000 tokens × $0.15 per million"),
                    ("$0.18", "output: 300,000 tokens × $0.60 per million"),
                    ("$0.48", "per day (illustrative small-model prices)")]),
        dict(type="diagram", key="anatomy", title="Anatomy of a GenAI app", draw="anatomy",
             notes="Draw the top row first (user, app, LLM API), then add the dashed retrieval box and documents.",
             draw_list=["Top row: User → Your app (builds the prompt) → LLM API",
                        "Arrows back: answer = tokens out, from LLM API to app to user",
                        "Write 'you pay per token, in and out' above the LLM API box",
                        "Dashed box under the app: Retrieval (optional), arrows 'search' and 'top passages'",
                        "Your documents (policies, FAQs, handbooks) → Retrieval, labelled 'prepared in advance'",
                        "Bottom pill: Retrieval + LLM = RAG"]),
        dict(type="table", key="map", title="The course map", size=13,
             header=["Module", "Topic", "Key ideas", "Min"], widths=[0.75, 2.1, 3.4, 0.5],
             rows=[["1", "Introduction to Generative AI", "GenAI, LLM limits, RAG, the landscape", "25"],
                   ["2", "How LLMs work", "Tokens, next-token prediction, attention, temperature, streaming", "40"],
                   ["3", "Prompt engineering", "Roles, instructions, grounding, citations, refusals", "30"],
                   ["4", "Embeddings and similarity", "Meaning as vectors, dimensions, cosine similarity", "30"],
                   ["5", "Vector databases", "Index, upsert, metadata, filters, top-k search", "30"],
                   ["6", "Document ingestion and chunking", "PDF text, cleaning, token chunks with overlap", "30"],
                   ["7", "Retrieval-Augmented Generation", "The full pipeline and how it fails", "30"],
                   ["8", "Running GenAI in the real world", "Observability, then we start the project", "10"],
                   ["Then", "Capstone project: DocuMind", "You build a PDF question-answering assistant using every idea",
                    "–"]]),
    ],
    recap=dict(points=["Model families: GPT, Claude, Gemini, Llama, Mistral",
                       "Closed API models vs open-weight models you run yourself",
                       "An app sends a request, gets tokens back, and pays per token",
                       "Anatomy: user → app → optional retrieval → LLM API → answer"],
               task="Pick an assistant for a college handbook. Draw its anatomy from memory, mark the optional "
                    "step, then compare with slide {anatomy}."),
    shot_plan=[("0:00–0:30", "Slide {v13}, hook", "S"), ("0:30–1:50", "Slide {families} model families", "S"),
               ("1:50–3:00", "Slide {open} closed vs open-weight", "S"),
               ("3:00–4:20", "Slide {call} API call, then screen: run it", "S + R"),
               ("4:20–5:10", "Slide {cost} paying per token", "S"),
               ("5:10–7:30", "Slide {anatomy} → board: anatomy of a GenAI app", "W"),
               ("7:30–8:20", "Slide {map} course map", "S"),
               ("8:20–9:00", "Slide {r13} recap", "S")],
    script=[
        "[Slide {v13}]",
        "You now know what generative AI is and where LLMs fail. In this video we look at the landscape: who "
        "makes these models, the two ways you can use them, how an app actually talks to a model, and what "
        "most GenAI apps look like inside. We finish with the map of the whole course.",
        "[Slide {families}]",
        "A handful of companies build the best-known model families. These names change fast, so treat this "
        "as a snapshot at the time of writing. OpenAI makes the GPT family, for example gpt-4o-mini, a small, "
        "cheap model we will use in our examples. Anthropic makes Claude. Google makes Gemini, and also a "
        "smaller open family called Gemma. Meta makes Llama, whose weights anyone can download. And Mistral AI, "
        "a French company, offers both open models and a paid API. The good news: the core ideas in this "
        "course work the same way with all of them.",
        "[Slide {open}]",
        "That brings us to the two ways of using a model. A closed model runs on the provider's servers. You "
        "never see its insides; you send your text through an API and pay for what you use. No hardware to "
        "manage, and these are often the strongest models. The trade-off: your text leaves your building, so "
        "read the provider's data terms before sending anything sensitive.",
        "An open-weight model is different. The weights, the billions of learned numbers that make up the "
        "model, are published. You download them and run the model on your own GPUs or on a cloud machine you "
        "rent. Your data stays with you and you have full control. But now updates, scaling and safety are "
        "your job. People often call these open source, but strictly, open weights means the trained model is "
        "shared, not always the training data or code.",
        "[Slide {call}]",
        "Most apps use a model through an API, so let us see one call. This is the OpenAI Python library, "
        "version 1. We create a client, which reads the secret API key from an environment variable. Never "
        "type the key into your code. Then we send a request: which model, and a list of messages. The system "
        "message sets the model's role, and the user message is the actual question. Temperature controls how "
        "creative the answer is; low means focused. Module 2 explains it properly. Other providers have "
        "their own libraries, but the shape is almost the same: a model name, a list of messages, a few "
        "settings, and text back.",
        "[Screen: run the script in a terminal; show the answer and the usage line]",
        "When I run it, the answer comes back, and so does a usage record: how many tokens went in, and how "
        "many came out. A token is a small piece of text, roughly three quarters of an English word.",
        "[Slide {cost}]",
        "Those token counts are your bill. Let us do a quick sum with illustrative prices for a small model: "
        "15 cents per million input tokens and 60 cents per million output tokens. Say a help desk answers "
        "1,000 questions a day, each with 2,000 tokens in and 300 tokens out. Input: two million tokens, so 30 "
        "cents. Output: three hundred thousand tokens, so 18 cents. Total: 48 cents a day. Always check the "
        "provider's current price page, but notice that the input side usually dominates.",
        "[Slide {anatomy} → Board]",
        "Now let us draw what most GenAI apps look like inside. [Board: box User, box Your app, box LLM API] A "
        "user asks a question. Your app, your own code, builds a prompt and sends it to the LLM API. "
        "[Board: arrows back, label tokens out] The model returns the answer as tokens, the app shows it to the "
        "user, and you pay for every token in and out.",
        "[Board: dashed box Retrieval under the app] For a document assistant, one optional step sits in the "
        "middle. Before calling the model, the app searches your documents and picks the most relevant "
        "passages. [Board: box Your documents, arrow labelled prepared in advance] The documents, say a college "
        "handbook or an e-commerce FAQ, are prepared in advance so this search is fast. Those passages go into "
        "the prompt. [Board: pill Retrieval + LLM = RAG] Retrieval plus the LLM: that is RAG. "
        "Without the dashed box you have a plain chatbot; with it you have an assistant that answers from "
        "your own documents. Most of this course is about building that dashed box well.",
        "[Slide {map}]",
        "Here is our route. Module 2: how LLMs work, tokens, attention, temperature and streaming. Module 3: "
        "prompt engineering. Modules 4, 5 and 6 build retrieval: embeddings, vector databases, and preparing "
        "documents. Module 7 joins everything into a RAG pipeline. Module 8 shows how to keep an eye on GenAI in "
        "the real world, observability, and then we start the project. After that, you put it all into practice: you build the capstone project, DocuMind, an assistant that answers questions from company PDFs, with the page it found each answer on.",
        "[Slide {r13}]",
        "Recap. The main families are GPT, Claude, Gemini, Llama and Mistral. Closed models run behind an API; "
        "open-weight models you run yourself. An app sends a request, gets tokens back and pays per token. And "
        "the anatomy: user, app, optional retrieval, LLM API, answer.",
        "Practice: pick an assistant for a college handbook, draw its anatomy from memory, mark the optional "
        "step, and compare with slide {anatomy}. In the capstone project you will use all of these concepts "
        "together to build one real system. Next module: we open the LLM itself.",
    ],
)

MODULE = dict(
    folder="Introduction", kicker="INTRODUCTION", title="Introduction to Generative AI",
    subtitle="What GenAI is, where LLMs fail, and how apps use them",
    guide_intro="Mostly slides, with two board segments (the nested rings and the anatomy of a GenAI app) and "
                "one short screen recording of a single API call.",
    script_intro="The screen segment runs a short Python script (the code slide in Video 1.3) in a terminal, with "
                 "the API key already set as an environment variable.",
    videos=[V11, V12, V13],
    guide_md="""
## Goal of this module

The learner can place Generative AI inside AI, machine learning and deep learning; explain the difference
between discriminative and generative models; name the three ways a plain LLM fails in a company
(hallucination, knowledge cutoff, no private data) and how Retrieval-Augmented Generation (RAG) addresses
them; name the main model families and the difference between closed API models and open-weight models;
explain how an app calls a model and pays per token; and draw the anatomy of a GenAI app.

## Where this sits in the course

- Before: nothing. This is the first module; no AI background is assumed, only basic Python reading.
- Now: the big picture and the vocabulary every later module uses.
- Next: Module 2 opens the LLM (tokens, attention, sampling). Modules 3–7 cover prompting, embeddings,
  vector databases, ingestion and RAG; Module 8 covers observability and running GenAI in the real world, then hands over to
  the capstone project, DocuMind, which learners build next using every concept from this course.

## What we teach

Video 1.1 – What Generative AI is

- AI ⊃ ML ⊃ DL ⊃ GenAI, with one example per ring
- Discriminative vs generative models; real assistants combine ranking and writing
- Kinds of GenAI output; LLMs are the text branch
- Why now: Transformer (2017), data, GPUs, ChatGPT (late 2022), APIs

Video 1.2 – What LLMs can and cannot do

- Strengths: fluent writing, summarising, following formats, combining passages
- Failures: hallucination, knowledge cutoff, no private data
- Leave carry-over example: plain chatbot guess vs a cited answer from the company's own policy
- RAG named for the first time: retrieve, augment, generate

Video 1.3 – The GenAI landscape and this course

- Model families and providers (snapshot at the time of writing)
- Closed API models vs open-weight models
- One API call (OpenAI Python SDK v1); usage record; pay per token, with a worked cost sum
- Board: anatomy of a GenAI app, with retrieval as the optional step
- Course map, ending with the capstone project videos

## Recording notes

- Board segment (1.1): draw the rings from the outside in; write one example beside each.
- Board segment (1.3): draw the top row first (user, app, LLM API), then add the dashed retrieval box and
  the documents box. Keep the word "optional" visible.
- Screen segment (1.3): set `OPENAI_API_KEY` in the terminal before recording and never show it on
  screen. Run the script on slide {call} once; zoom in on the `usage` line (input and output token counts).
- Prices on slide {cost} are illustrative (a small model, at the time of writing). Say so on camera.
- Model names on slide {families} change often; check them before recording.

## Common learner questions

- "Do I have to pay to follow this course?" Calling a hosted model costs money per token, but the
  amounts in our examples are a few cents. Open-weight models can also be run locally for free on capable
  hardware.
- "Is open-weight the same as open source?" Not exactly: the trained weights are shared, but the training
  data and code often are not, and licences vary.
- "Why not fine-tune the model on our documents instead?" Covered in Module 7: fine-tuning changes style
  and behaviour well but is a poor way to store facts that change; RAG keeps facts in the documents, where
  they can be updated and cited.
- "Is RAG still needed now that models accept very long inputs?" Also Module 7: cost, speed and accuracy
  still favour sending a few relevant passages instead of every page.
""",
    notes_md="""
------------------------------------------------------------------------

## Video 1.1 – What Generative AI is

### Four nested rings

| **Ring** | **Meaning** | **Example** |
|:---|:---|:---|
| Artificial Intelligence | Any machine that behaves in a way we call smart | Chess engine with hand-written rules |
| Machine Learning | Learns patterns from examples instead of rules | Spam filter trained on labelled emails |
| Deep Learning | ML with many-layer neural networks; good with images, sound, language | Face unlock |
| Generative AI | Deep-learning models that create new content | ChatGPT, Claude, Gemini |

Every generative model is also deep learning, machine learning and AI. The reverse is not true.

### Discriminative vs generative

- A discriminative model chooses from a fixed set of answers: spam / not spam, department A / B / C.
  It learns the boundary between groups. Output: a label or a number.
- A generative model learns what the data looks like well enough to produce new examples: a reply, a
  summary, an image. Output: open-ended content.

Real assistants often combine both ideas. A help desk assistant first ranks handbook passages by how well
they match the question and keeps the best few (a ranking job, close to the discriminative idea), then
writes an answer from them (generative).

### Kinds of output

Text, code, images, audio and video. Models that handle several kinds at once are multimodal. Large
language models (LLMs) are the generative models for text, and the focus of this course.

### Why now

- 2017: the Transformer architecture (Module 2).
- Huge amounts of text from the web, books and code.
- Powerful GPUs to train on it.
- Late 2022: ChatGPT brings LLMs to the public.
- Today: models are available to any developer through an API.

------------------------------------------------------------------------

## Video 1.2 – What LLMs can and cannot do

| **Strong at** | **Weak at** |
|:---|:---|
| Fluent, well-structured writing | Facts not in the training data |
| Summarising, rewriting, translating | Anything after the knowledge cutoff |
| Following instructions and output formats | Your private documents |
| Combining several passages into one answer | Admitting it does not know (unless told to) |

### Three failures

1. Hallucination: a confident, fluent answer that is false. An LLM generates the most likely next words;
   likely is not the same as true (Module 2 shows why).
2. Knowledge cutoff: training stops at a date; newer facts do not exist for the model.
3. No private data: the model never read your company's files, so it can only guess what a typical
   company would say.

### Example

Question: "How many leave days can I carry over into next year?"

- Plain chatbot: "Usually 5 to 10 days, depending on the company." Plausible, unverifiable, possibly wrong.
- Assistant that reads the company's documents: "Up to 8 days; they expire on 31 March (Source:
  leave_policy, p.4)." Taken from the company's own policy and checkable. (Illustrative example.)

### The fix: Retrieval-Augmented Generation (RAG)

- Retrieval: find the passages in your documents that relate to the question.
- Augmentation: add those passages to the prompt.
- Generation: the LLM writes the answer from them, with citations.

RAG addresses all three failures: private data and freshness come from the documents; grounding rules and
citations reduce and expose hallucination. It does not remove hallucination completely (Module 7).
Modules 4–6 build retrieval (embeddings, vector databases, ingestion and chunking), Module 3 writes the
grounding rules, and Module 7 assembles the pipeline.

------------------------------------------------------------------------

## Video 1.3 – The GenAI landscape and this course

### Model families (at the time of writing; names change often)

| **Provider** | **Family** | **Access** |
|:---|:---|:---|
| OpenAI | GPT (e.g. gpt-4o-mini) | Closed: API, ChatGPT app |
| Anthropic | Claude | Closed: API, Claude apps |
| Google | Gemini; Gemma (open) | Closed API; Gemma weights downloadable |
| Meta | Llama | Open weights |
| Mistral AI | Mistral | Open-weight models and a paid API |

### Closed API vs open-weight

| | **Closed, through an API** | **Open-weight, run yourself** |
|:---|:---|:---|
| Where it runs | Provider's servers | Your GPUs or a rented cloud machine |
| Cost | Pay per token | Pay for hardware |
| Control and data | Text goes to the provider; read the terms | Data stays with you |
| Your work | Very little | Updates, scaling, safety |

"Open weights" means the trained numbers are published; the training data and code may not be, so it
is not always "open source" in the strict sense.

### One API call

```python
from openai import OpenAI

client = OpenAI()   # reads OPENAI_API_KEY from the environment

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful HR assistant."},
        {"role": "user", "content": "Explain notice period in two lines."},
    ],
    temperature=0.2,
)
print(response.choices[0].message.content)
print(response.usage)   # prompt_tokens, completion_tokens, total_tokens
```

- The API key is a secret: keep it in an environment variable, never in the code.
- `messages` carries roles: `system` sets behaviour, `user` asks (Module 3).
- `temperature` controls randomness; low values keep answers focused (Module 2).
- `usage` reports input (prompt) and output (completion) tokens, which is what you are billed for.
  A token is roughly three quarters of an English word (Module 2).

### Cost sum (illustrative small-model prices)

1,000 questions a day, 2,000 input tokens and 300 output tokens each, at $0.15 per million input tokens
and $0.60 per million output tokens:

- Input: 1,000 × 2,000 = 2,000,000 tokens → 2 × $0.15 = $0.30
- Output: 1,000 × 300 = 300,000 tokens → 0.3 × $0.60 = $0.18
- Total: $0.48 per day

Always check the provider's current price page.

### Anatomy of a GenAI app

1. The user asks a question.
2. Your app builds a prompt.
3. Optional retrieval: the app searches your documents (prepared in advance) and adds the most relevant
   passages to the prompt.
4. The LLM API returns the answer as tokens; you pay for tokens in and out.
5. The app shows the answer to the user.

Retrieval + LLM = RAG.

### Course map

| **Module** | **Topic** | **Key ideas** | **Min** |
|:---|:---|:---|:---|
| 1 | Introduction to Generative AI | GenAI, LLM limits, RAG, the landscape | 25 |
| 2 | How LLMs work | Tokens, next-token prediction, attention, temperature, streaming | 40 |
| 3 | Prompt engineering | Roles, instructions, grounding, citations, refusals | 30 |
| 4 | Embeddings and similarity | Meaning as vectors, dimensions, cosine similarity | 30 |
| 5 | Vector databases | Index, upsert, metadata, filters, top-k search | 30 |
| 6 | Document ingestion and chunking | PDF text, cleaning, token chunks with overlap | 30 |
| 7 | Retrieval-Augmented Generation | The full pipeline and how it fails | 30 |
| 8 | Running GenAI in the real world | Observability, then we start the project | 10 |
| Then | Capstone project: DocuMind | Learners build a PDF question-answering assistant using every concept | – |

Total: 225 minutes.
""",
    quiz=[
        dict(q="Which statement about AI, machine learning, deep learning and generative AI is correct?",
             a="Generative AI contains deep learning", b="Every machine learning model is generative",
             c="Generative AI is a part of deep learning, which is a part of machine learning",
             d="Deep learning and machine learning are unrelated", correct="C",
             why="The rings nest: AI ⊃ ML ⊃ DL ⊃ GenAI. Every generative model is deep learning, but not the "
                 "other way round."),
        dict(q="A model reads a support ticket and outputs one of five department names. What kind of model is it?",
             a="Discriminative", b="Generative", c="Multimodal", d="A vector database", correct="A",
             why="It chooses a label from a fixed set. A generative model would write new content."),
        dict(q="An LLM confidently states a leave carry-over limit that appears nowhere in your policy. What is "
               "this called?",
             a="Knowledge cutoff", b="Hallucination", c="Overfitting", d="Tokenisation", correct="B",
             why="Hallucination is fluent, confident output that is not true. LLMs write likely text, not "
                 "checked text."),
        dict(q="Your company changed its travel policy last month. A plain LLM still describes the old rule. "
               "Which limitation is this?",
             a="Knowledge cutoff", b="Multimodality", c="Low temperature", d="Too many output tokens",
             correct="A", why="Training stopped at a date; anything newer does not exist for the model."),
        dict(q="Why can't a plain LLM answer questions about your company's HR handbook?",
             a="LLMs cannot read English", b="The handbook is too short",
             c="It never saw the handbook in training, so it can only guess",
             d="HR questions are blocked by law", correct="C",
             why="The model knows only its training data. Private documents were never part of it."),
        dict(q="In RAG, what does the 'augmented' part mean?",
             a="The model is retrained on your documents", b="Retrieved passages are added to the prompt",
             c="The answer is made longer", d="The PDF is converted to images", correct="B",
             why="Retrieval finds passages, augmentation adds them to the prompt, generation writes the answer."),
        dict(q="Which best describes an open-weight model such as Llama?",
             a="It can only be used through the provider's paid API",
             b="Its trained weights are published, so you can download and run it yourself",
             c="It has no weights", d="It always comes with its full training data", correct="B",
             why="Open weights means the learned numbers are shared. Training data and code are often not."),
        dict(q="Which statement about using a closed model through an API is true?",
             a="The model runs on your laptop", b="You send a request to the provider and pay for input and "
                                                  "output tokens",
             c="Calls are free and unlimited", d="You must download the weights first", correct="B",
             why="Closed models run on the provider's servers; the usage record lists input and output tokens, "
                 "which are billed."),
        dict(q="An app sends 1,000 requests a day, each with 2,000 input tokens. At $0.15 per million input "
               "tokens, what is the daily input cost?",
             a="$0.03", b="$0.30", c="$3.00", d="$300", correct="B",
             why="1,000 × 2,000 = 2,000,000 tokens = 2 million; 2 × $0.15 = $0.30."),
        dict(q="In the anatomy of a GenAI app, which step is optional and finds relevant passages in your "
               "documents before the model is called?",
             a="The user question", b="Retrieval", c="The LLM API call", d="Showing the answer",
             correct="B", why="Retrieval is the optional middle step. Retrieval plus the LLM is RAG."),
    ],
)
