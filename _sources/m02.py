"""Module 2 – How LLMs work: tokens, next-token prediction, attention, temperature and streaming.

Schema: see m01.py. Token splits and IDs on the slides are real (tiktoken 0.14, cl100k_base / o200k_base).
"""
from style import *

# temperature example: next token after "Refunds are accepted within 14"
CANDS = [" days", " working", " business", " calendar"]
BASE = [0.70, 0.15, 0.10, 0.05]


def at_temp(t):
    w = [p ** (1 / t) for p in BASE]
    s = sum(w)
    return [x / s for x in w]


def nexttoken():   # the autoregressive loop with a probability bar chart
    fig, ax = canvas()
    box(ax, 2.7, 4.65, 4.7, 1.35, "Most companies offer\na refund window of", color=BLUE, fs=17)
    text(ax, 2.7, 5.65, "text so far (the prompt)", fs=13, bold=True, color=BLUE)
    arrow(ax, 5.05, 4.65, 5.65, 4.65, color=INK, lw=2)
    box(ax, 6.45, 4.65, 1.6, 1.0, "LLM", color=PURPLE, fs=20, bold=True)
    arrow(ax, 7.25, 4.65, 8.05, 4.65, color=INK, lw=2)
    text(ax, 11.9, 6.05, "probability of every possible next token", fs=14, bold=True, color=PURPLE)
    toks = [(" 30", 0.55), (" 14", 0.20), (" 7", 0.15), (" 60", 0.06), ("other", 0.04)]
    for i, (t, p) in enumerate(toks):
        y = 5.5 - i * 0.52
        c = CORAL if i == 0 else LINE
        text(ax, 9.05, y, f"“{t.strip()}”" if t != "other" else "all other", fs=14, ha="right")
        ax.add_patch(Rectangle((9.25, y - 0.19), p * 8.6, 0.38, fc=c, ec="none"))
        text(ax, 9.4 + p * 8.6, y, f"{p:.0%}", fs=14, ha="left", bold=i == 0)
    pill(ax, 12.0, 2.15, "① pick one token:  “30”", color=CORAL, fs=16)
    arrow(ax, 10.95, 2.15, 8.75, 1.35, color=CORAL, lw=2)
    box(ax, 4.6, 1.2, 7.9, 0.95, "Most companies offer a refund window of 30", color=GREEN, fs=17)
    arrow(ax, 0.55, 1.68, 0.55, 3.97, color=GREEN, lw=2.2)
    text(ax, 0.85, 2.85, "② append it and repeat,\n    one token at a time", fs=15, bold=True, color=GREEN, ha="left")
    text(ax, 15.6, 0.55, "likely ≠ true: your policy may say 14", fs=15, bold=True, color=RED, ha="right")
    return fig


def attention():   # "bank" attends to different words in two sentences
    fig, ax = canvas()

    def row(words, y, focus, weights, color, caption):
        xs = [1.0 + 1.75 * i for i in range(len(words))]
        for x, w in zip(xs, words):
            hl = w == focus
            box(ax, x, y, 1.45, 0.62, w, color=color if hl else LINE, fill=TINT[color] if hl else "white",
                fs=17, bold=hl)
        xb = xs[words.index(focus)]
        text(ax, xb, y - 0.6, f"{1 - sum(weights.values()):.2f}", fs=15, bold=True, color=MUTED)
        for x, w in zip(xs, words):
            if w == focus or w not in weights:
                continue
            wt = weights[w]
            rad = 0.45 if x < xb else -0.45
            arrow(ax, xb, y + 0.33, x, y + 0.33, color=color, lw=1 + wt * 11, rad=rad, ms=14)
            text(ax, x, y - 0.6, f"{wt:.2f}", fs=15, bold=True, color=color)
        text(ax, xs[-1] + 1.15, y + 0.15, caption, fs=16, color=INK, ha="left", va="center")

    row(["I", "sat", "by", "the", "river", "bank"], 3.9, "bank",
        {"I": 0.03, "sat": 0.10, "by": 0.08, "the": 0.05, "river": 0.62}, TEAL,
        "“bank” looks mostly at “river”\n→ the edge of a river")
    row(["I", "opened", "a", "bank", "account"], 1.05, "bank",
        {"I": 0.04, "opened": 0.20, "a": 0.06, "account": 0.58}, ORANGE,
        "“bank” looks mostly at “account”\n→ a place for money")
    text(ax, 15.6, 6.1, "line thickness = attention weight · numbers under the words add up to 1 · made up for teaching",
         fs=12, color=MUTED, ha="right")
    return fig


def temperature():   # same distribution at three temperatures
    fig, ax = canvas()
    panels = [(0.1, "T = 0.1  (factual Q&A)", GREEN, "almost always “days”: steady, factual"),
              (1.0, "T = 1.0  (default)", BLUE, "the model's own odds: some variety"),
              (1.5, "T = 1.5", CORAL, "flatter: odd words appear more often")]
    for k, (t, title, c, cap) in enumerate(panels):
        x0 = 0.55 + k * 5.2
        probs = at_temp(t)
        text(ax, x0 + 2.25, 6.05, title, fs=17, bold=True, color=c)
        ax.plot([x0, x0 + 4.6], [1.3, 1.3], color=INK, lw=1.2)
        for i, (tok, p) in enumerate(zip(CANDS, probs)):
            bx = x0 + 0.15 + i * 1.12
            ax.add_patch(Rectangle((bx, 1.3), 0.85, p * 3.9, fc=c if i == 0 else TINT[c], ec=c, lw=1.5))
            text(ax, bx + 0.425, 1.3 + p * 3.9 + 0.22, f"{p:.0%}" if p >= 0.005 else "~0%", fs=13, bold=i == 0)
            text(ax, bx + 0.425, 1.0, tok.strip(), fs=12, color=MUTED)
        text(ax, x0 + 2.25, 0.45, cap, fs=13, color=INK)
    return fig


V21 = dict(
    num="2.1", title="Tokens and tokenisation", minutes=9, segments="SR",
    slides=[
        dict(type="bullets", key="tokdef", title="What is a token?",
             bullets=["LLMs do not read letters or whole words: they read tokens",
                      "A token is a common piece of text: a word, part of a word, a space + word, a digit, a symbol",
                      "Each token has an ID number in the model's vocabulary",
                      "The model only ever sees lists of IDs, never the letters",
                      "Text → token IDs is called encoding; IDs → text is decoding"]),
        dict(type="table", key="toks", title="Real tokens (cl100k_base)", widths=[2.4, 0.9, 0.8, 3.3], size=17,
             notes="Real output of tiktoken. · marks a leading space inside a token.",
             header=["Text", "Characters", "Tokens", "How it splits"],
             rows=[["What is the refund policy?", "26", "6", "What | ·is | ·the | ·refund | ·policy | ?"],
                   ["Refunds are accepted within 14 days.", "36", "9",
                    "Ref | unds | ·are | ·accepted | ·within | · | 14 | ·days | ."],
                   ["unbelievably", "12", "3", "un | belie | vably"],
                   ["def count_tokens(text):", "23", "5", "def | ·count | _tokens | (text | ):"]]),
        dict(type="cards", key="bpe", title="How a tokenizer learns its pieces (BPE)",
             cards=[("Start small", "Begin with single bytes: every letter, digit and symbol is its own piece."),
                    ("Merge the most frequent pair", "In a huge pile of text, find the pair that appears most "
                                                     "often, like “t”+“h”, and make it one new piece."),
                    ("Repeat", "Keep merging: “th”+“e” → “the”. Common words end up "
                               "as one token; rare words stay in parts."),
                    ("Stop at a size", "cl100k_base has about 100,000 pieces; the newer o200k_base about 200,000.")]),
        dict(type="code", key="tikcode", title="Counting tokens in Python", label="PYTHON · tiktoken",
             code='''import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

ids = enc.encode("What is the refund policy?")
print(ids)              # [3923, 374, 279, 21639, 4947, 30]
print(len(ids))         # 6  <- the token count
print(enc.decode(ids))  # What is the refund policy?

# the tokenizer a given model uses
enc = tiktoken.encoding_for_model("gpt-4o-mini")
print(enc.name)         # o200k_base''',
             bullets=["encode: text → IDs; decode: IDs → the exact same text",
                      "len(enc.encode(text)) is the token count",
                      "Apps count tokens to estimate cost and check limits",
                      "Cutting text every N tokens is how chunking works (Module 6)"]),
        dict(type="table", key="tokzr", title="Different models, different tokenizers",
             widths=[1.5, 1.2, 3.4, 1.3, 1.6], size=16,
             notes="Model-to-tokenizer mapping at the time of writing (tiktoken 0.14). IDs and Hindi counts are real.",
             header=["Tokenizer", "Pieces", "Used by (at the time of writing)", "ID of “What”", "Hindi question"],
             rows=[["cl100k_base", "≈ 100,000", "GPT-4, GPT-3.5 Turbo, text-embedding-3-small / -large", "3923",
                    "19 tokens"],
                   ["o200k_base", "≈ 200,000", "GPT-4o family, e.g. gpt-4o-mini", "4827", "7 tokens"],
                   ["Others", "varies", "Claude, Gemini, Llama: each has its own tokenizer", "different",
                    "different"]]),
        dict(type="stats", key="rules", title="Rules of thumb (English)",
             stats=[("≈ 4", "characters per token"),
                    ("≈ ¾", "of a word per token: 100 tokens ≈ 75 words"),
                    ("≈ 300", "words in a 400-token piece of text, about half a page")]),
        dict(type="two_col", key="whytok", title="Why tokens matter",
             left=("Cost", ["APIs charge per million tokens", "Input and output are priced separately",
                            "Output tokens usually cost more", "More text in the prompt = higher bill"]),
             right=("Limits", ["Every model has a maximum context, in tokens",
                               "text-embedding-3-small: up to 8,191 tokens per input",
                               "A 400-token piece of a handbook fits easily",
                               "The prompt + the answer must fit the LLM's window"])),
    ],
    recap=dict(points=["Models read token IDs, not letters", "BPE merges frequent pieces into one token",
                       "English ≈ 4 characters per token; Hindi and code use more",
                       "Tokens decide both cost and limits",
                       "Different models use different tokenizers: count with the right one"],
               task="pip install tiktoken, then encode your own name, a Hindi sentence and one line of Python with "
                    "cl100k_base and o200k_base. Which uses the most tokens per character?"),
    shot_plan=[("0:00–0:40", "Slide {v21}, hook", "S"), ("0:40–1:40", "Slide {tokdef} what is a token", "S"),
               ("1:40–2:50", "Slide {toks} real tokens", "S"), ("2:50–3:50", "Slide {bpe} BPE", "S"),
               ("3:50–4:50", "Screen: Python shell, tiktoken encode / decode", "R"),
               ("4:50–5:40", "Slide {tikcode} counting tokens in Python", "S"),
               ("5:40–6:50", "Slide {tokzr} different tokenizers", "S"),
               ("6:50–8:20", "Slides {rules}–{whytok} rules of thumb, why tokens matter", "S"),
               ("8:20–9:00", "Slide {r21} recap", "S")],
    script=[
        "[Slide {v21}]",
        "Welcome to Module 2. In this module we open the large language model and look inside. Not with heavy "
        "maths, but with pictures and real examples. And we start with the smallest unit of all, the one every "
        "LLM bill and every LLM limit is measured in: the token. By the end of this video you will know what "
        "“this handbook is 50 thousand tokens” really means.",
        "[Slide {tokdef}]",
        "Here is the first surprise. An LLM does not read letters, and it does not read whole words either. It "
        "reads tokens. A token is a common piece of text. Sometimes it is a whole word. Sometimes it is part of "
        "a word. Very often it is a space plus a word, because spaces are glued to the word that follows them. "
        "A digit or a full stop can be a token too.",
        "Every token has an ID number in the model's vocabulary, a bit like a roll number in a class. The model "
        "never sees letters at all. It only sees lists of these numbers. Turning text into token IDs is called "
        "encoding. Turning IDs back into text is called decoding.",
        "[Slide {toks}]",
        "Let us look at real tokens. These splits come straight from a real tokenizer called cl100k_base, used "
        "by many OpenAI models. The question “What is the refund policy?” has 26 characters and becomes 6 "
        "tokens: What, space-is, space-the, space-refund, space-policy, and the question mark. Common words are "
        "one token each.",
        "Now “Refunds are accepted within 14 days”. The word Refunds is split into two pieces, Ref and unds. "
        "The space before 14 becomes its own token, and 14 is one more. A long, rarer word like unbelievably is "
        "cut into three pieces: un, belie, vably. And code is split too: def, space-count, underscore-tokens, "
        "and so on. So the rule is: frequent text gets big pieces, rare text gets small pieces.",
        "[Slide {bpe}]",
        "Where do these pieces come from? Most modern tokenizers use a method called byte pair encoding, or "
        "BPE. The idea is simple. Start small: every letter, digit and symbol is its own piece. Then look at a "
        "huge pile of text and find the pair of pieces that appears together most often, for example t and h. "
        "Merge them into one new piece, th. Repeat. Now th and e appear together very often, so they become "
        "the. Keep merging until the vocabulary reaches a target size. cl100k_base has about one hundred "
        "thousand pieces. The newer o200k_base has about two hundred thousand.",
        "[Screen: Python shell: import tiktoken, enc = tiktoken.get_encoding(\"cl100k_base\"), enc.encode(...), "
        "enc.decode(...)]",
        "Let me show you on screen. I import tiktoken, OpenAI's open-source tokenizer library, and load "
        "cl100k_base. I encode “What is the refund policy?” and get six numbers. Those numbers are all the "
        "model sees. If I decode them, I get my sentence back, exactly. Now I try the same question in Hindi. "
        "With cl100k_base, a short Hindi question of 19 characters becomes 19 tokens, one token per character. "
        "With o200k_base it is only 7. The newer tokenizer learned many more pieces for Indian languages.",
        "[Slide {tikcode}]",
        "Here is the same thing as a small program. encode turns text into IDs, and the length of that list is "
        "the token count. decode gives back the same text. The last lines show a handy helper: encoding_for_model takes a model name and returns the "
        "tokenizer that model uses. For gpt-4o-mini, that is o200k_base.",
        "Apps count tokens to estimate the bill, to check limits, and to cut long documents into pieces of, "
        "say, 400 tokens. We do that cutting in Module 6.",
        "[Slide {tokzr}]",
        "This brings us to an important fact: different models use different tokenizers. At the time of "
        "writing, cl100k_base is used by GPT-4, GPT-3.5 Turbo and OpenAI's embedding models such as "
        "text-embedding-3-small. o200k_base is used by the GPT-4o family, including gpt-4o-mini. Even the same "
        "word gets a different ID: “What” is 3923 in one and 4827 in the other. And Claude, Gemini and Llama "
        "each have their own tokenizer again.",
        "So, for an exact count, use the model's own tokenizer. Another one gives close, but not equal, numbers "
        "for English, and very different numbers for Hindi.",
        "[Slide {rules}]",
        "Some rules of thumb. In English, one token is about four characters. Or, one token "
        "is about three quarters of a word, so a hundred tokens is roughly seventy-five words. A 400-token "
        "piece of text is about 300 words, half a page.",
        "[Slide {whytok}]",
        "Why should you care? Two reasons: cost and limits. Cost first. LLM providers charge per million tokens, "
        "with separate prices for the tokens you send in and the tokens the model writes out. Output tokens "
        "usually cost several times more. So every extra paragraph you put in a prompt shows up on the bill.",
        "Then limits. Every model has a maximum number of tokens it can handle at once. The embedding model "
        "text-embedding-3-small, which we meet in Module 4, accepts up to 8,191 tokens per input, so a 400-token "
        "piece of a college handbook is comfortably inside. And for an LLM, the prompt plus the answer must fit "
        "in its context window. We come back to that in video 2.4.",
        "[Slide {r21}]",
        "Recap. Models read token IDs, not letters. BPE builds the vocabulary by merging frequent pieces. "
        "English is about four characters per token; Hindi and code use more. Tokens decide both cost and "
        "limits. And different models use different tokenizers, so count with the right one.",
        "Practice: install tiktoken and encode your own name, a Hindi sentence and one line of Python, with "
        "both encodings. Which one uses the most tokens per character? In the next video: how a model uses "
        "these tokens to write, one token at a time.",
    ],
)

V22 = dict(
    num="2.2", title="Next-token prediction and how models are trained", minutes=10, segments="SW",
    slides=[
        dict(type="diagram", key="loop", title="One token at a time", draw="nexttoken",
             draw_list=["Box: “Most companies offer a refund window of” → box LLM",
                        "Bar chart: 30 = 55%, 14 = 20%, 7 = 15%, 60 = 6%, others 4%",
                        "① pick one token: 30",
                        "② append it to the text, arrow back to the start: repeat",
                        "In red: likely ≠ true (your policy may say 14)"]),
        dict(type="bullets", key="autoreg", title="Autoregressive generation",
             bullets=["Input: all the tokens so far", "Output: a probability for every token in the vocabulary",
                      "Pick one token, add it to the text, run the model again",
                      "Stop at a special end token or at the output limit (max_tokens)",
                      "A 200-token answer = 200 runs of the model"]),
        dict(type="cards", key="train", title="Three stages of training",
             cards=[("Pretraining", "Read a huge amount of public text and learn to predict the next token. "
                                    "Result: knows language and many facts, but just continues text."),
                    ("Instruction tuning", "Train on many examples of instructions and good answers. Result: "
                                           "answers questions instead of rambling on."),
                    ("Preference tuning", "Humans (or AI judges) rank answers; the model is nudged towards the "
                                          "preferred ones (RLHF and similar). Result: helpful, polite, safer.")]),
        dict(type="two_col", key="base", title="Base model vs chat model",
             left=("Base model (pretraining only)", ["Prompt: “What is the hostel curfew?”",
                                                     "Continues: “What is the mess timing? What is the fee …”",
                                                     "Great at language, not at helping"]),
             right=("Chat model (all three stages)", ["Same prompt",
                                                      "Answers the question, politely",
                                                      "But it never read your college's handbook: it may guess"])),
        dict(type="two_col", key="likely", title="Why hallucination happens",
             left=("What the model optimises", ["The most likely next token, given the text so far",
                                                "Learned from public text, up to a cutoff date",
                                                "No built-in “fact check” step",
                                                "Fluent and confident by design"]),
             right=("What you need", ["The true answer for your organisation",
                                      "From documents it never saw", "With a source you can check",
                                      "→ give it the facts in the prompt (RAG, Module 7)"])),
    ],
    recap=dict(points=["An LLM predicts a probability for every possible next token",
                       "It writes one token at a time and feeds each one back in",
                       "Pretraining → instruction tuning → preference tuning",
                       "Likely is not the same as true: that is why we use RAG"],
               task="Type the start of a sentence into any chatbot, e.g. “The capital of India is”, and ask "
                    "it to continue with one word. Then try a sentence about your college's rules. Which answer "
                    "can you trust, and why?", label="TRY THIS"),
    shot_plan=[("0:00–0:40", "Slide {v22}, hook", "S"),
               ("0:40–3:50", "Slide {loop} → board: next-token loop", "W"),
               ("3:50–5:00", "Slide {autoreg} autoregressive", "S"),
               ("5:00–7:00", "Slide {train} three stages of training", "S"),
               ("7:00–8:00", "Slide {base} base vs chat model", "S"),
               ("8:00–9:20", "Slide {likely} why hallucination happens", "S"),
               ("9:20–10:00", "Slide {r22} recap", "S")],
    script=[
        "[Slide {v22}]",
        "When you ask a chatbot a question, the answer appears on the screen word by word. It looks like the "
        "model is thinking and then typing. What is really happening is both simpler and more surprising. The "
        "model is playing one game, again and again: guess the next token.",
        "[Slide {loop} → Board]",
        "[Board: box with “Most companies offer a refund window of”] Let us take the text so far: “Most "
        "companies offer a refund window of”. Imagine this is part of a question to an online shop's support "
        "bot. [Board: arrow into a box labelled LLM] We give these tokens to the model.",
        "[Board: bar chart with 30, 14, 7, 60, others] What comes out is not one word. It is a probability for "
        "every token in the vocabulary, all hundred thousand or more of them. Most are almost zero. A few are "
        "large. Here, the token 30 gets 55 percent, 14 gets 20, 7 gets 15, 60 gets 6, and everything else "
        "shares the last 4 percent. The model learned these odds from all the text it has read: “30 days” "
        "is a very common phrase.",
        "[Board: circle 1, pick one token: 30] Step one: pick one token. Usually one of the likely ones; "
        "exactly how is controlled by the temperature, which we meet in video 2.4. Say it picks 30.",
        "[Board: new box with the sentence plus 30, arrow back to the start] Step two: append that token to "
        "the text and run the whole thing again. Now the text ends in “of 30”, and the most likely next "
        "token is probably “days”. Then a full stop. Then maybe the end. One token at a time.",
        "[Board: write in red: likely ≠ true] And now look at the danger. Thirty was the most likely "
        "answer. But this particular shop's policy might say 14. The model did not check anything. It produced "
        "the most likely continuation, and it said it fluently. That is hallucination, seen from the inside.",
        "[Slide {autoreg}]",
        "This way of writing has a name: autoregressive generation. Auto means self: the model feeds on its "
        "own output. The input is all the tokens so far. The output is a probability for every possible next "
        "token. We pick one, add it, and run the model again. It stops when it produces a special end token, "
        "or when it reaches the output limit we set. In the API that limit is a setting, usually called "
        "max_tokens.",
        "This explains something you may have noticed. A 200-token answer means the model runs 200 times. "
        "That is why long answers take longer, and one reason output tokens are priced higher than input "
        "tokens: the whole prompt is read in one go, but every output token needs its own pass.",
        "[Slide {train}]",
        "So how does a model learn these probabilities? In three stages. Stage one, pretraining. The model reads "
        "an enormous amount of public text: websites, books, code. For every position it tries to predict the "
        "next token, checks the real answer, and adjusts its internal numbers a tiny bit. Billions of times. "
        "After pretraining, the model knows grammar, facts and styles. But it is only a text continuer.",
        "Stage two, instruction tuning. The model is trained further on many examples of an instruction and a "
        "good answer. Now it learns to respond: you ask, it answers.",
        "Stage three, preference tuning. People, or AI judges, compare two answers and pick the better one. The "
        "model is nudged towards the preferred style: helpful, clear, polite, and safer. The best-known "
        "method is called RLHF, reinforcement learning from human feedback. The chat models you use, like "
        "ChatGPT, Claude or gpt-4o-mini, went through all three stages.",
        "[Slide {base}]",
        "Let me make the difference concrete. Give a base model, one that has only been pretrained, the "
        "prompt “What is the hostel curfew?”. It might continue with “What is the mess timing? What is the "
        "fee for the gym?”, because a list of questions on a college FAQ page is very likely text. It is great "
        "at language, but it is not trying to help you.",
        "Give the same prompt to a chat model, and it answers the question, politely and clearly. But notice "
        "the catch. The chat model never read your college's handbook. So if it gives a time, it is guessing "
        "a likely one. Training made it helpful. It did not make it know your private facts.",
        "[Slide {likely}]",
        "Now we can say precisely why hallucination happens. What the model optimises is the most likely next "
        "token, given the text so far. It learned from public text, up to a cutoff date, so anything newer, "
        "or anything private, is simply missing. There is no built-in fact-checking step. And it was trained "
        "to sound fluent and confident.",
        "What you need is different: the true answer for your organisation, from documents the model never saw, "
        "with a source you can check. The fix is not to hope the model knows. The fix is to put the facts "
        "into the prompt, so that the most likely continuation is also the true one. That idea is called "
        "retrieval-augmented generation, RAG, and Module 7 is all about it.",
        "[Slide {r22}]",
        "Recap. An LLM predicts a probability for every possible next token. It writes one token at a time and "
        "feeds each token back in. It is trained in three stages: pretraining, instruction tuning and "
        "preference tuning. And likely is not the same as true, which is why we use RAG.",
        "Try this: ask a chatbot to continue “The capital of India is” with one word, and then a sentence about "
        "your own college's rules. Which answer can you trust, and why? In the next video: the transformer, the "
        "machine that computes those probabilities.",
    ],
)

V23 = dict(
    num="2.3", title="Transformers and attention", minutes=10, segments="SW",
    slides=[
        dict(type="bullets", key="inside", title="Inside the model, from text to probabilities",
             bullets=["1 · Text → token IDs (tokenizer)",
                      "2 · Each ID → a vector of numbers, plus its position in the text",
                      "3 · Many identical layers, each with attention + a feed-forward step",
                      "4 · Every layer refines each token's vector using the others",
                      "5 · Last layer → a probability for every possible next token"]),
        dict(type="diagram", key="attn", title="Attention: which words matter for “bank”?", draw="attention",
             draw_list=["Row 1: I sat by the river bank — lines from “bank” to every word",
                        "Thickest line to “river” (0.62): bank = edge of a river",
                        "Row 2: I opened a bank account — thickest line to “account” (0.58)",
                        "Write: same word, different meaning, decided by context",
                        "Weights are made up for teaching; each row adds up to 1"]),
        dict(type="table", key="weights", title="A tiny attention example", widths=[1.6, 1.1, 3.2], size=16,
             header=["Word that “bank” looks at", "Weight", "What it means"],
             rows=[["river", "0.62", "Most of the new meaning comes from here"],
                   ["bank (itself)", "0.12", "Keeps some of its own original meaning"],
                   ["sat", "0.10", "A little: someone is sitting there"],
                   ["by", "0.08", "A little: a place word"],
                   ["the", "0.05", "Almost nothing"], ["I", "0.03", "Almost nothing"],
                   ["Total", "1.00", "New “bank” = weighted mix of all these"]]),
        dict(type="bullets", key="ctxuse", title="Why text in the prompt shapes the answer",
             bullets=["While writing, each new token attends to every token before it (never to future ones)",
                      "That includes everything you pasted into the prompt",
                      "Help desk: three paragraphs of the leave policy + the employee's question",
                      "Answer tokens give high weight to the policy's words: “12 days”, “casual leave”",
                      "This is the mechanism RAG relies on: right text in, grounded answer out"]),
        dict(type="two_col", key="rnn", title="Why transformers won",
             left=("Before 2017: recurrent networks (RNNs)", ["Read one word at a time, left to right",
                                                             "Earlier words fade as the text gets longer",
                                                             "Hard to run in parallel: slow to train"]),
             right=("Transformers (“Attention Is All You Need”, 2017)",
                    ["Look at all tokens at once", "Any token can attend to any other, near or far",
                     "Parallel on GPUs: train on far more text", "The T in GPT: Generative Pre-trained Transformer"])),
        dict(type="stats", key="scale", title="What “large” means",
             stats=[("10s–100s", "of layers stacked on top of each other"),
                    ("Billions", "of learned numbers (parameters) in a modern LLM"),
                    ("Trillions", "of tokens of text read during pretraining")]),
    ],
    recap=dict(points=["Tokens become vectors, then pass through many layers",
                       "Attention: each token looks at the others and mixes in what matters",
                       "Context decides meaning: river bank vs bank account",
                       "Facts in the prompt shape every answer token",
                       "Transformers read everything at once, so they scale"],
               task="Write two sentences where one word changes meaning (e.g. “apple” the fruit vs Apple the "
                    "company). For each, list the words that the ambiguous word should attend to most."),
    shot_plan=[("0:00–0:40", "Slide {v23}, hook", "S"), ("0:40–2:20", "Slide {inside} inside the model", "S"),
               ("2:20–5:20", "Slide {attn} → board: attention lines", "W"),
               ("5:20–6:10", "Slide {weights} tiny example", "S"),
               ("6:10–7:20", "Slide {ctxuse} why prompt text shapes the answer", "S"),
               ("7:20–8:40", "Slide {rnn} why transformers won", "S"),
               ("8:40–9:20", "Slide {scale} what large means", "S"), ("9:20–10:00", "Slide {r23} recap", "S")],
    script=[
        "[Slide {v23}]",
        "In the last video, the model gave us a probability for every next token. But how? What is inside the "
        "box we drew? The answer is an architecture called the transformer. We will not do the maths. We will "
        "build the intuition, so that when someone says “attention” in an interview, you know exactly what "
        "they mean.",
        "[Slide {inside}]",
        "Here is the journey from text to probabilities in five steps. One: the tokenizer turns the text into "
        "token IDs, as we saw in video 2.1. Two: each ID is turned into a vector, a long list of numbers that "
        "represents that token. The model also adds information about the token's position, because “dog bites "
        "man” and “man bites dog” use the same tokens in a different order.",
        "Three: these vectors pass through many identical layers. Each layer has two parts: attention, where "
        "tokens share information with each other, and a feed-forward step, where each token's vector is "
        "processed on its own, a bit like thinking about what it just heard. Four: in every layer, each token's "
        "vector is refined. By the top of the stack, the vector for a word carries its meaning in this "
        "particular sentence. Five: the last layer turns the final vector into a probability for every "
        "possible next token.",
        "The heart of all this is attention. Let me draw it.",
        "[Slide {attn} → Board]",
        "[Board: write I sat by the river bank, one box per word] Take the sentence “I sat by the river "
        "bank”. Focus on the word bank. On its own, bank is ambiguous. It could be a place for money or the "
        "edge of a river.",
        "[Board: lines from bank to every word, thickest to river] In an attention step, bank looks at every "
        "other word and asks: how relevant are you to me? Each word gets a weight. River gets a big weight, "
        "0.62. Sat and by get a little. The and I get almost nothing. Then the new vector for bank becomes a "
        "mix of all the words, weighted by these numbers. So most of its new meaning comes from river. Bank now "
        "means the edge of a river.",
        "[Board: second row, I opened a bank account, thickest line to account] Now the sentence “I opened a "
        "bank account”. Same word, bank. But this time the biggest weight goes to account, 0.58, and some to "
        "opened. So here bank becomes a place for money. Same token, different meaning, decided by context.",
        "[Board: write: context decides meaning] That is attention. Every token, in every layer, does this with "
        "the other tokens. And real models run many attention patterns in parallel, called heads: one might "
        "track grammar, another who did what, another which noun a pronoun refers to.",
        "[Slide {weights}]",
        "Here are the numbers from the board as a table. River 0.62. Bank itself 0.12, so it keeps some of its "
        "own meaning. Sat 0.10, by 0.08, the 0.05, I 0.03. Together they add up to exactly one. The new bank "
        "is a weighted average of all of them. These numbers are made up for teaching, but the mechanism is "
        "real. The model learns, during training, how to compute good weights.",
        "[Slide {ctxuse}]",
        "One more detail, and it matters a lot for the rest of this course. When a model is writing, each new "
        "token can attend to every token before it, but never to future tokens, because they do not exist yet. "
        "And “every token before it” includes everything you put in the prompt.",
        "Picture a company help desk. The app pastes three paragraphs from the leave policy into the prompt, "
        "and then the employee's question: how many casual leaves do I get? As the model writes its answer, "
        "the answer tokens give high attention weight to the words in those paragraphs, like “12 days” and "
        "“casual leave”. The facts in the prompt directly shape every word the model writes. This is the "
        "mechanism that retrieval-augmented generation relies on: put the right text in, and you get a "
        "grounded answer out.",
        "[Slide {rnn}]",
        "Why did transformers take over? Before 2017, language models were mostly recurrent networks, RNNs. "
        "They read one word at a time, left to right, carrying a memory along. Over long text, early words "
        "faded from that memory. And because each step waited for the previous one, they were slow to train.",
        "In 2017, a Google paper with the famous title “Attention Is All You Need” introduced the "
        "transformer. It looks at all the tokens at once. Any token can attend to any other, whether it is next "
        "door or a thousand tokens away. And because the work runs in parallel on GPUs, transformers can be "
        "trained on far more text. That is the T in GPT: generative pre-trained transformer.",
        "[Slide {scale}]",
        "So what does “large” in large language model mean? Tens or even hundreds of layers stacked up. "
        "Billions of learned numbers, called parameters: the weights adjusted during training. And trillions of "
        "tokens of text read during pretraining. Exact numbers for most commercial models, like gpt-4o-mini, "
        "are not published, but this is the scale we are talking about.",
        "[Slide {r23}]",
        "Recap. Tokens become vectors and pass through many layers. In each layer, attention lets every token "
        "look at the others and mix in what matters. Context decides meaning: river bank versus bank account. "
        "The facts in the prompt shape every answer token. And transformers read everything at once, which is "
        "why they scale.",
        "Practice: write two sentences where one word changes meaning, like apple the fruit and Apple the "
        "company. For each, list the words it should attend to most. In the next video: the context window, "
        "temperature, and streaming, three settings every LLM app has to choose.",
    ],
)

V24 = dict(
    num="2.4", title="Context window, temperature and streaming", minutes=11, segments="SW",
    slides=[
        dict(type="stats", key="ctx", title="The context window",
             stats=[("128K", "tokens: gpt-4o-mini's context window, prompt + answer (at the time of writing)"),
                    ("≈ 16K", "tokens: the most it can write in one answer"),
                    ("≈ 2.3K", "tokens: a prompt with 5 passages × 400 + rules + question")]),
        dict(type="bullets", key="ctxrules", title="What fits, and what does not",
             bullets=["Context window = everything the model sees in one call: system prompt, passages, question, answer",
                      "Anything outside the window does not exist for the model",
                      "No memory between calls: chat apps resend the earlier messages every time",
                      "Bigger prompts cost more and take longer, even when they fit",
                      "So good apps send a few relevant passages, not the whole document"]),
        dict(type="diagram", key="temp", title="Temperature reshapes the odds", draw="temperature",
             notes="Example: next token after “Refunds are accepted within 14”. Base odds 70/15/10/5.",
             draw_list=["Write the prompt: “Refunds are accepted within 14 ...”",
                        "Three bar charts, same 4 tokens: days, working, business, calendar",
                        "T = 0.1: days ≈ 100%, others ≈ 0 (factual Q&A)",
                        "T = 1.0: 70 / 15 / 10 / 5, the model's own odds",
                        "T = 1.5: flatter, about 55 / 20 / 15 / 10"]),
        dict(type="two_col", key="topp", title="Two sampling knobs",
             left=("Temperature", ["Low (0–0.3): sharp, repeatable, factual",
                                  "Around 1: the model's natural variety",
                                  "High (>1): creative, more mistakes",
                                  "Answers from a document: keep it low, e.g. 0.1"]),
             right=("Top-p (nucleus sampling)", ["Keep only the most likely tokens whose odds add up to p",
                                                "top_p = 0.9: drop the unlikely tail",
                                                "Usually tune temperature or top-p, not both",
                                                "Many apps leave top-p at its default"])),
        dict(type="code", key="gencode", title="Temperature and streaming in code",
             label="PYTHON · OpenAI SDK v1",
             code='''from openai import OpenAI
client = OpenAI()  # key from OPENAI_API_KEY

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": RULES},
        {"role": "user", "content": question},
    ],
    temperature=0.1,
    max_tokens=300,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)''',
             bullets=["system = the rules; user = the passages + question (Module 3)",
                      "temperature=0.1: steady, factual answers",
                      "stream=True: each piece is sent as soon as it is generated",
                      "chunk.choices[0].delta.content = the new text"]),
        dict(type="cards", key="stream", title="Why stream?",
             cards=[("Without streaming", "The user stares at a spinner until all 200 tokens are written, then "
                                          "sees everything at once."),
                    ("With streaming", "The first words often appear within a second; the rest follows as the "
                                       "model writes."),
                    ("Same total", "Streaming does not make the model faster or cheaper. It only makes the wait "
                                   "feel shorter.")]),
    ],
    recap=dict(points=["Context window = prompt + answer, in tokens; no memory between calls",
                       "Temperature sharpens (low) or flattens (high) the next-token odds",
                       "Top-p cuts off the unlikely tail",
                       "Factual assistants: low temperature + stream=True"],
               task="Run the streaming snippet with temperature=1.2 and ask the same question three times. "
                    "Compare the answers, then repeat with temperature=0.1."),
    shot_plan=[("0:00–0:40", "Slide {v24}, hook", "S"), ("0:40–2:00", "Slide {ctx} context window numbers", "S"),
               ("2:00–3:30", "Slide {ctxrules} what fits", "S"),
               ("3:30–6:50", "Slide {temp} → board: temperature bar charts", "W"),
               ("6:50–7:50", "Slide {topp} temperature and top-p", "S"),
               ("7:50–9:20", "Slide {gencode} streaming code", "S"), ("9:20–10:20", "Slide {stream} why stream", "S"),
               ("10:20–11:00", "Slide {r24} recap", "S")],
    script=[
        "[Slide {v24}]",
        "Every app that calls an LLM makes three choices: how much text to send, how adventurous the model is "
        "allowed to be, and whether the answer arrives all at once or word by word. In this video: the context "
        "window, temperature, and streaming.",
        "[Slide {ctx}]",
        "First, the context window. This is the total number of tokens the model can handle in one call, and "
        "it includes both the prompt and the answer. At the time of writing, gpt-4o-mini has a context window "
        "of about 128 thousand tokens, and it can write up to roughly 16 thousand tokens in one answer. Other "
        "models have other limits, so always check the provider's model page.",
        "How much does a typical question-answering app use? Say it sends five passages of 400 tokens each. "
        "That is 2,000 tokens. Add the rules, the labels and the question, and a typical prompt is around 2,300 "
        "tokens. A tiny fraction of the window.",
        "[Slide {ctxrules}]",
        "Some rules about the window. It is everything the model can see in one call: the system prompt, the "
        "passages, the question, and the answer it is writing. Anything outside it does not exist for the "
        "model. The model also has no memory between calls. Every request starts fresh. If a customer support "
        "bot seems to remember what you said five messages ago, it is because the app sends the whole "
        "conversation again with every new message. So a long chat slowly fills the window, and costs more "
        "with every turn.",
        "And even when something fits, it is not free. A bigger prompt costs more and takes longer. So if the "
        "window is so large, why not paste the whole 300-page handbook? Cost, speed, and accuracy: a model given "
        "five relevant passages usually answers better than one given hundreds of mostly irrelevant pages. We "
        "come back to this in Module 7.",
        "[Slide {temp} → Board]",
        "Now, temperature. Remember: the model gives a probability for every next token, and then one is "
        "picked. Temperature controls how that pick works. [Board: write “Refunds are accepted within 14”] "
        "Let us take the text “Refunds are accepted within 14”.",
        "[Board: middle chart, T = 1.0: days 70, working 15, business 10, calendar 5] At temperature one, we "
        "use the model's own odds. Days 70 percent, working 15, business 10, calendar 5. If we generate many "
        "times, about seven times in ten we get days, and sometimes one of the others.",
        "[Board: left chart, T = 0.1: days almost 100] Low temperature sharpens the odds. Mathematically, each "
        "probability is raised to the power one over T and then rescaled, so at 0.1 the strongest option wins "
        "almost every time. Days becomes practically 100 percent. The output becomes steady and repeatable.",
        "[Board: right chart, T = 1.5: flatter] High temperature flattens the odds. At 1.5, days drops to about "
        "55 percent and calendar rises to about 10. Unusual words appear more often. That is great for poems "
        "and brainstorming, and bad for policy answers.",
        "[Board: circle the T = 0.1 chart, write factual Q&A] So an assistant that answers from a document "
        "usually uses a low temperature, something like 0.1, because we want the answer to follow the source "
        "closely, the same way every time. A tool that writes marketing slogans might use 1 or a little more. And remember: even at a low "
        "temperature, the model can still be wrong. Temperature makes an answer steady, not true. Only good "
        "text in the prompt makes it true.",
        "[Slide {topp}]",
        "To summarise temperature: low, zero to about 0.3, is sharp and factual. Around one is the model's "
        "natural variety. Above one is creative, with more mistakes. There is a second knob called top-p, or "
        "nucleus sampling. Top-p of 0.9 means: keep only the most likely tokens whose probabilities add up to "
        "90 percent, and drop the long tail of unlikely ones. Providers usually advise tuning one of the two, "
        "not both. Many apps set the temperature and leave top-p at its default.",
        "[Slide {gencode}]",
        "Here is how these settings look in code, with the OpenAI Python SDK. We create a client; it reads the "
        "API key from an environment variable, never from the code. We send messages: a system message with "
        "the rules, and a user message with the question, and in a document assistant also the passages. "
        "Module 3 is all about those messages. Temperature is 0.1. max_tokens caps the answer at 300 tokens. "
        "And stream equals True.",
        "With streaming on, the API does not wait for the full answer. As soon as the model produces a token, "
        "it is sent. We loop over the stream. Each chunk carries a delta, a few new characters, at choices "
        "zero, delta, content. Sometimes the delta is empty, so we check it first, then print it without a new "
        "line. In a web app you do the same thing, but send each piece to the page. Streamlit, for example, "
        "has st.write_stream, which takes these pieces and shows them as they arrive. That is the typing "
        "effect you see in chat apps.",
        "[Slide {stream}]",
        "Why bother? Without streaming, the user stares at a spinner until the whole answer is written. With "
        "streaming, the first words appear quickly and the rest follows while the model writes. But be clear "
        "about one thing: streaming does not make the model faster or cheaper. The total time and the total "
        "tokens are the same. It only makes the wait feel much shorter, and that matters a lot for users. People will happily read "
        "along for five seconds, but they will not stare at a blank box for five seconds. One small cost: "
        "your code must handle the pieces one by one, and if you want the full answer, for example to save it "
        "in a log, you join the pieces together at the end.",
        "[Slide {r24}]",
        "Recap. The context window is prompt plus answer, measured in tokens, and there is no memory between "
        "calls. Temperature sharpens or flattens the next-token odds, and top-p cuts off the unlikely tail. "
        "For factual assistants, a low temperature and streaming are the usual choice.",
        "Practice: run the snippet with temperature 1.2, ask the same question three times and compare. Then "
        "try 0.1. In Module 3, we look at what we actually send in those messages: prompt engineering. And in "
        "the capstone project, DocuMind, you will see all of this inside one real system.",
    ],
)

MODULE = dict(
    folder="How_LLMs_Work", kicker="HOW LLMS WORK", title="How LLMs work",
    subtitle="Tokens, next-token prediction, attention, temperature",
    guide_intro="Slides with three board segments (next-token loop, attention, temperature) and one short "
                "screen recording of tiktoken in a Python shell.",
    script_intro="The screen segment in 2.1 uses a Python shell with `tiktoken` installed "
                 "(`pip install tiktoken`).",
    videos=[V21, V22, V23, V24],
    guide_md="""
## Goal of this module

The learner can explain what a token is and why it drives cost and limits; count tokens with tiktoken and
explain why different models use different tokenizers (`cl100k_base`, `o200k_base`); describe how an LLM
writes one token at a time from a probability distribution; name the three training stages; explain
attention with the river bank / bank account example and why text in the prompt shapes the answer; and
choose sensible values for the context, `temperature`, `max_tokens` and `stream=True` in an API call.

## Where this sits in the course

- Before (Module 1): what Generative AI is and where LLMs fit.
- Now: what happens inside an LLM, and the settings every LLM app chooses.
- Next: Module 3 (prompt engineering) uses the message roles and the context window from this module;
  Module 4 (embeddings) reuses the idea of a token as a vector of numbers; Module 6 cuts documents into
  fixed-size token pieces with tiktoken.

## What we teach

Video 2.1 – Tokens and tokenisation

- Token, token ID, vocabulary, encoding / decoding
- Real splits from `cl100k_base` (slide {toks}); BPE merges
- Counting tokens with tiktoken: `get_encoding`, `encode`, `decode`, `encoding_for_model` (slide {tikcode})
- Different models, different tokenizers (slide {tokzr}): cl100k_base vs o200k_base; Hindi 19 vs 7 tokens
- Rules of thumb: ≈4 characters, ≈¾ word per English token
- Cost per million tokens; limits (embedding input 8,191 tokens)

Video 2.2 – Next-token prediction and training

- Probability for every next token; pick, append, repeat (autoregressive); `max_tokens`
- Why output tokens cost more and long answers take longer
- Pretraining → instruction tuning → preference tuning (RLHF); base model vs chat model
- Hallucination explained: likely ≠ true; knowledge cutoff; private facts are missing

Video 2.3 – Transformers and attention

- Tokens → vectors + position → layers of attention + feed-forward → probabilities
- Attention board: river bank vs bank account; weights add up to 1
- Answer tokens attend to the prompt (help-desk example): the mechanism behind RAG
- RNNs vs transformers; "Attention Is All You Need" (2017); the T in GPT
- What "large" means

Video 2.4 – Context window, temperature and streaming

- Context window = prompt + answer; gpt-4o-mini ≈128K context, ≈16K output (at the time of writing)
- A 5-passage prompt ≈2,300 tokens; no memory between calls (chat apps resend history)
- Temperature board: same odds at T = 0.1 / 1.0 / 1.5; top-p
- OpenAI SDK call: system/user messages, `temperature=0.1`, `max_tokens`, `stream=True`,
  `chunk.choices[0].delta.content`; showing a stream in a web app (e.g. Streamlit `st.write_stream`)

## Recording notes

- Screen segment (2.1): prepare a Python shell with tiktoken installed. Type `import tiktoken`,
  `enc = tiktoken.get_encoding("cl100k_base")`, `enc.encode("What is the refund policy?")` → 6 IDs
  `[3923, 374, 279, 21639, 4947, 30]`, then `enc.decode(...)`. For the Hindi comparison, paste a short
  Hindi question such as "रिफंड नीति क्या है?" and compare `len(...)` with `o200k_base` (19 vs 7).
  If your screen font cannot show Devanagari, only show the counts.
- Board (2.2): draw the bar chart horizontally; leave space at the bottom for the "append and repeat"
  loop. Write "likely ≠ true" in red at the end.
- Board (2.3): write both sentences first, then draw the attention lines from "bank" with thick lines for
  big weights. Say clearly that the numbers are made up for teaching.
- Board (2.4): draw the T = 1.0 chart first, then 0.1 on its left and 1.5 on its right.
- Model limits (128K context, ≈16K output) and the model-to-tokenizer mapping are at the time of
  writing; check the provider's model page and `tiktoken.encoding_for_model` before recording and update
  the slides if they changed.
- The only mention of the capstone is the last sentence of video 2.4. Do not add more: this module teaches
  the concepts in general.

## Common learner questions

- "Is a token the same as a word?" No. Common words are one token, rare words several, and a leading
  space is part of the token.
- "Which tokenizer should I count with?" The model's own: `tiktoken.encoding_for_model(name)`. Another
  tokenizer gives close but not identical counts for English, and very different counts for Hindi.
- "Does temperature 0 give exactly the same answer every time?" Almost always, but providers do not
  guarantee perfect repeatability, because of tiny numerical differences on the servers.
- "Does the model understand?" It computes very good probabilities from patterns. Whether that is
  understanding is a debate; for engineering, what matters is that it can be confidently wrong.
- "Should a factual assistant use temperature 0 or 0.1?" Either is fine; for factual answers they behave
  almost the same. The important part is that it is low.
""",
    notes_md="""
------------------------------------------------------------------------

## Video 2.1 – Tokens and tokenisation

### Vocabulary

| **Term** | **Meaning** |
|:---|:---|
| Token | A common piece of text: a word, part of a word, a space + word, a digit or a symbol |
| Token ID | The number of that piece in the model's vocabulary |
| Vocabulary | The full list of pieces a tokenizer knows (≈100,000 in cl100k_base, ≈200,000 in o200k_base) |
| Encoding | Text → list of token IDs |
| Decoding | List of token IDs → text |

The model never sees letters, only token IDs.

### Real examples (tiktoken, cl100k_base)

| **Text** | **Characters** | **Tokens** | **Pieces** (· = leading space) |
|:---|:---|:---|:---|
| What is the refund policy? | 26 | 6 | What · ·is · ·the · ·refund · ·policy · ? |
| Refunds are accepted within 14 days. | 36 | 9 | Ref · unds · ·are · ·accepted · ·within · · · 14 · ·days · . |
| unbelievably | 12 | 3 | un · belie · vably |
| def count_tokens(text): | 23 | 5 | def · ·count · _tokens · (text · ): |

`enc.encode("What is the refund policy?")` → `[3923, 374, 279, 21639, 4947, 30]`.

### Byte pair encoding (BPE)

1. Start with single bytes (letters, digits, symbols).
2. In a large body of text, find the most frequent adjacent pair and merge it into a new piece.
3. Repeat until the vocabulary reaches its target size.

Result: frequent words become one token; rare words are split into several pieces; nothing is ever
"unknown", because any text can fall back to bytes.

### Counting tokens in Python

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("What is the refund policy?")
print(len(ids))          # 6
print(enc.decode(ids))   # What is the refund policy?

enc = tiktoken.encoding_for_model("gpt-4o-mini")
print(enc.name)          # o200k_base
```

Apps count tokens to estimate cost, to check limits, and to cut long documents into pieces of a fixed
number of tokens (Module 6): encode the text, take N IDs at a time, decode each slice back to text.

### Different models, different tokenizers (at the time of writing)

| **Tokenizer** | **Pieces** | **Used by** | **ID of "What"** | **Short Hindi question (19 characters)** |
|:---|:---|:---|:---|:---|
| cl100k_base | ≈100,000 | GPT-4, GPT-3.5 Turbo, text-embedding-3-small / -large | 3923 | 19 tokens |
| o200k_base | ≈200,000 | GPT-4o family (e.g. gpt-4o-mini) | 4827 | 7 tokens |
| Others | varies | Claude, Gemini, Llama each have their own | different | different |

For an exact count, use the model's own tokenizer. Counting English with a different tokenizer gives close
but not identical numbers; for other languages the difference can be large.

### Rules of thumb (English)

- 1 token ≈ 4 characters ≈ ¾ of a word; 100 tokens ≈ 75 words.
- A 400-token piece of text ≈ 300 words ≈ half a page.
- Other languages and code usually need more tokens for the same content.

### Why tokens matter

- Cost: APIs charge per million tokens, with separate prices for input and output; output is usually
  several times more expensive.
- Limits: `text-embedding-3-small` accepts up to 8,191 tokens per input; an LLM's prompt plus its answer
  must fit its context window.

------------------------------------------------------------------------

## Video 2.2 – Next-token prediction and how models are trained

### The loop

1. Input: all tokens so far.
2. Output: a probability for every token in the vocabulary.
3. Pick one token (how: temperature, video 2.4).
4. Append it and repeat, until an end token or the output limit (`max_tokens`).

This is autoregressive generation: the model feeds on its own output.

Example: "Most companies offer a refund window of" → 30 (55%), 14 (20%), 7 (15%), 60 (6%), others (4%).
The most likely continuation is "30", even if this shop's policy says 14.

Consequences:

- A 200-token answer needs 200 passes of the model, so long answers are slower.
- The prompt is processed in one pass, which is one reason input tokens are cheaper than output tokens.

### Three training stages

| **Stage** | **Data** | **Result** |
|:---|:---|:---|
| Pretraining | Huge amounts of public text and code; predict the next token | Knows language and many facts; only continues text |
| Instruction tuning | Examples of instructions with good answers | Responds to requests instead of continuing them |
| Preference tuning (RLHF and similar) | Humans or AI judges rank answers | Helpful, clear, polite, safer behaviour |

Base model vs chat model: given "What is the hostel curfew?", a base model may continue with more
questions ("What is the mess timing? ..."); a chat model answers, but it never read your college's
handbook, so any time it gives is a guess.

### Why hallucination happens

The model optimises "most likely next token given the text so far", learned from public text up to a
cutoff date, with no built-in fact check, and it is trained to sound fluent. What an organisation needs is
the true answer from private documents with a source. RAG (Module 7) puts the facts in the prompt so that
the likely continuation is also the true one.

------------------------------------------------------------------------

## Video 2.3 – Transformers and attention

### From text to probabilities

1. Tokenizer: text → token IDs.
2. Embedding: each ID → a vector; position information is added (word order matters).
3. A stack of identical layers, each with attention (tokens share information) and a feed-forward step
   (each token processed on its own).
4. Each layer refines every token's vector using the other tokens.
5. The final layer turns the last vector into a probability for every possible next token.

### Attention

Each token computes a weight for every other token ("how relevant are you to me?"). The weights add up to
1, and the token's new vector is a weighted mix of the others.

| **"bank" in "I sat by the river bank" looks at** | **Weight** |
|:---|:---|
| river | 0.62 |
| bank (itself) | 0.12 |
| sat | 0.10 |
| by | 0.08 |
| the | 0.05 |
| I | 0.03 |

(Weights made up for teaching; they add up to 1.00.) In "I opened a bank account", most weight goes to
"account", so "bank" ends up meaning a place for money. Real models run several attention patterns in
parallel ("heads") in every layer.

### Why text in the prompt shapes the answer

While generating, each new token attends to every earlier token (never to future ones), and that includes
everything in the prompt. Example: a help desk pastes three paragraphs of the leave policy before the
employee's question; the answer tokens give high weight to "12 days" and "casual leave" in those
paragraphs. This is the mechanism retrieval-augmented generation relies on.

### Why transformers replaced RNNs

| **RNN (before 2017)** | **Transformer (2017 →)** |
|:---|:---|
| Reads one token at a time | Looks at all tokens at once |
| Early words fade over long text | Any token can attend to any other |
| Sequential, slow to train | Parallel on GPUs, trains on far more data |

Paper: "Attention Is All You Need" (Vaswani et al., Google, 2017). GPT = Generative Pre-trained Transformer.

### Scale

Tens to hundreds of layers, billions of parameters (learned numbers), trillions of training tokens. Exact
figures for most commercial models, such as gpt-4o-mini, are not published.

------------------------------------------------------------------------

## Video 2.4 – Context window, temperature and streaming

### Context window

- The maximum number of tokens in one call: prompt (system + user) and answer together.
- gpt-4o-mini: about 128K tokens of context and up to about 16K output tokens (at the time of writing).
- A prompt with 5 passages × 400 tokens = 2,000 tokens + rules + labels + question ≈ 2,300 tokens.
- No memory between calls: chat apps resend earlier messages each time, so long chats grow the prompt.
- Bigger prompts cost more and are slower, and irrelevant text can hurt accuracy; so send a few relevant
  passages, not the whole document.

### Temperature

Each probability is raised to the power 1/T and the results are rescaled to add up to 1.

Next token after "Refunds are accepted within 14":

| **Token** | **T = 0.1** | **T = 1.0** | **T = 1.5** |
|:---|:---|:---|:---|
| days | ≈100% | 70% | ≈55% |
| working | ≈0% | 15% | ≈20% |
| business | ≈0% | 10% | ≈15% |
| calendar | ≈0% | 5% | ≈10% |

- Low (0–0.3): sharp, repeatable; good for answers from a document (e.g. 0.1).
- Around 1: the model's natural variety.
- High (> 1): flatter, more creative, more errors.

Top-p (nucleus sampling): keep only the most likely tokens whose probabilities add up to p (e.g. 0.9), and
sample among them. Usually tune temperature or top-p, not both.

### Streaming with the OpenAI Python SDK (v1)

```python
from openai import OpenAI
client = OpenAI()   # reads OPENAI_API_KEY from the environment
RULES = "Answer only from the policy text below. If it is not there, say you don't know."
question = "How many days do I have to return an item?"

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": RULES},
              {"role": "user", "content": question}],
    temperature=0.1, max_tokens=300, stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

- `system` carries the rules, `user` the question (and, in a document assistant, the passages); see Module 3.
- Each chunk holds a small delta of new text at `chunk.choices[0].delta.content`; it can be empty.
- In a web app, send each delta to the page as it arrives; Streamlit's `st.write_stream` accepts a
  generator of text pieces and displays them as they come.
- Streaming shortens the time to the first words; it does not reduce the total time or the number of tokens.
""",
    quiz=[
        dict(q="What does an LLM actually receive as input?", a="The letters of the text",
             b="A list of token IDs", c="A picture of the text", d="Whole sentences, one at a time", correct="B",
             why="The tokenizer turns text into token IDs; the model only ever sees those numbers."),
        dict(q="Roughly how many tokens is a 300-word English passage?", a="About 75", b="About 100",
             c="About 400", d="About 1,200", correct="C",
             why="One token is about ¾ of a word, so 300 words ÷ 0.75 ≈ 400 tokens."),
        dict(q="In byte pair encoding, how is the vocabulary built?",
             a="By merging the most frequent pairs of pieces again and again",
             b="By listing every word in the dictionary", c="By splitting text at every space only",
             d="By assigning one token to every letter only", correct="A",
             why="BPE starts from bytes and repeatedly merges the most frequent adjacent pair until it reaches "
                 "the target vocabulary size."),
        dict(q="You count an English prompt's tokens with cl100k_base and send it to gpt-4o-mini. What is true?",
             a="gpt-4o-mini does not use tokens", b="The count is exactly right for every model",
             c="The count is close but not exact, because the GPT-4o family uses o200k_base",
             d="tiktoken can only count characters", correct="C",
             why="Different models use different tokenizers. For an exact count, use the model's own tokenizer, "
                 "e.g. tiktoken.encoding_for_model(\"gpt-4o-mini\")."),
        dict(q="What does 'autoregressive' mean for an LLM?",
             a="It corrects its own mistakes automatically",
             b="It writes one token at a time, feeding each new token back in as input",
             c="It reads the whole internet at every question", d="It runs without a GPU", correct="B",
             why="The model predicts a token, appends it, and runs again on the longer text."),
        dict(q="Which training stage turns a text-continuer into a model that answers instructions?",
             a="Tokenisation", b="Pretraining", c="Instruction tuning", d="Streaming", correct="C",
             why="Pretraining teaches language; instruction tuning on instruction–answer examples teaches it to "
                 "respond; preference tuning polishes the style."),
        dict(q="In 'I sat by the river bank', why does the model treat 'bank' as the edge of a river?",
             a="Attention gives 'river' a high weight when updating 'bank'",
             b="The word 'bank' has only one meaning in the vocabulary",
             c="The model looks up a dictionary", d="Because 'bank' is the last word", correct="A",
             why="Attention mixes information from relevant tokens into each token's vector; context decides meaning."),
        dict(q="A help-desk app sends 5 passages of 400 tokens plus rules and a question to gpt-4o-mini "
               "(≈128K context). Which statement is true?",
             a="It is close to the model's limit", b="It uses a small part of the context window",
             c="The model remembers the previous question's passages too", d="The answer tokens do not count",
             correct="B", why="5 × 400 = 2,000 tokens, ≈2,300 with rules and question: far below ≈128K. Each call "
                              "starts fresh, and the answer also counts towards the window."),
        dict(q="Base odds for the next token are days 70%, working 15%, business 10%, calendar 5%. What does "
               "temperature 0.1 do?", a="Makes all four equally likely",
             b="Makes 'calendar' the most likely", c="Makes 'days' chosen almost every time",
             d="Has no effect on the odds", correct="C",
             why="Low temperature sharpens the distribution, so the strongest option wins nearly always."),
        dict(q="What does stream=True change in a chat completion call?",
             a="The answer is cheaper", b="The model writes faster",
             c="Each new piece of text is sent as soon as it is generated, so the first words appear sooner",
             d="The answer is shorter", correct="C",
             why="Streaming improves the time to the first words; total time and tokens stay the same."),
    ],
)
