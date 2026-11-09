"""Module 4 – Embeddings and similarity: meaning as numbers, embedding models in practice, cosine similarity. Schema: see m01.py."""
from matplotlib.patches import Arc

from style import *


def meaning_map():   # 2-D toy map of meaning
    fig, ax = canvas()
    ox, oy = 0.8, 0.55
    arrow(ax, ox, oy, 10.6, oy, color=MUTED, lw=1.8)
    arrow(ax, ox, oy, ox, 6.15, color=MUTED, lw=1.8)
    text(ax, 10.6, oy + 0.28, "dimension 1", fs=12, color=MUTED, ha="right")
    text(ax, ox + 0.15, 6.05, "dimension 2", fs=12, color=MUTED, ha="left")
    clusters = [
        (CORAL, "money back", (7.75, 3.3), 3.9, 2.3,
         [("refund", 7.0, 3.9), ("return policy", 8.5, 3.85), ("reimbursement", 7.05, 2.95), ("money back", 8.5, 2.95)]),
        (TEAL, "time off", (3.05, 1.85), 3.5, 1.9,
         [("holiday", 2.35, 2.3), ("leave", 3.75, 2.3), ("vacation", 2.4, 1.6), ("sick day", 3.75, 1.6)]),
        (PURPLE, "IT help", (3.25, 4.75), 3.9, 1.8,
         [("laptop", 2.45, 5.15), ("password reset", 3.95, 5.1), ("VPN", 3.0, 4.45)]),
    ]
    for c, name, (cx, cy), w, h, words in clusters:
        ax.add_patch(Ellipse((cx, cy), w, h, fc=TINT[c], ec=c, lw=1.8, ls=(0, (5, 3))))
        text(ax, cx, cy + h / 2 + 0.2, name, fs=13, bold=True, color=c)
        for word, x, y in words:
            ax.plot(x, y, "o", color=c, ms=8)
            text(ax, x, y - 0.27, word, fs=13)
    qx, qy = 8.3, 5.4
    ax.plot(qx, qy, "*", color=BLUE, ms=20)
    text(ax, qx + 0.25, qy + 0.05, "Q: “how do refunds work?”", fs=14, bold=True, color=BLUE, ha="left")
    box(ax, 13.45, 3.35, 4.6, 4.7, "", color=BLUE)
    lines = ["Close together = similar meaning", "Far apart = unrelated",
             "The question lands next to\n“return policy” with no\nshared keyword",
             "Real models: 384 to\n3,072 dimensions, not 2"]
    ys = [5.15, 4.45, 3.4, 2.0]
    for y, s in zip(ys, lines):
        text(ax, 11.4, y, "• " + s, fs=14, ha="left", va="center")
    return fig


def pipeline():   # text → same model → 1,536 numbers
    fig, ax = canvas()
    y1, y2 = 4.75, 1.75
    box(ax, 2.35, y1, 4.1, 1.55, "Chunk text\n“Refunds are accepted within\n14 days of delivery…”",
        color=BLUE, fs=14)
    text(ax, 2.35, y1 - 1.0, "one piece of a shop's FAQ page", fs=12, color=MUTED)
    box(ax, 2.35, y2, 4.1, 1.3, "Question\n“how do refunds work?”", color=TEAL, fs=14)
    text(ax, 2.35, y2 - 0.88, "typed by a customer", fs=12, color=MUTED)
    box(ax, 7.45, 3.25, 3.3, 4.1, "", color=PURPLE)
    text(ax, 7.45, 4.45, "Embedding\nmodel", fs=17, bold=True, color=PURPLE)
    text(ax, 7.45, 3.5, "e.g. text-embedding\n-3-small", fs=13, color=MUTED)
    text(ax, 7.45, 2.2, "the SAME model\nfor chunks and\nfor questions", fs=14)
    for y in (y1, y2):
        arrow(ax, 4.4, y, 5.8, y, color=INK, lw=2)
        arrow(ax, 9.1, y, 10.4, y, color=INK, lw=2)
    box(ax, 12.95, y1, 4.9, 1.55, "[0.012, −0.044, 0.031, …, 0.007]\n1,536 numbers", color=CORAL, fs=15)
    text(ax, 12.95, y1 - 1.0, "stored in a vector database, with the text", fs=12, color=MUTED)
    box(ax, 12.95, y2, 4.9, 1.3, "[0.018, −0.051, 0.027, …, 0.004]\n1,536 numbers", color=CORAL, fs=15)
    text(ax, 12.95, y2 - 0.88, "used to search the database", fs=12, color=MUTED)
    text(ax, 8.0, 0.25, "numbers are illustrative: real values come from the model", fs=12, color=MUTED)
    return fig


def angle():   # cosine = the angle between two arrows
    fig, ax = canvas()
    s, oy = 0.42, 3.5
    q = (3, 4)
    panels = [(2.9, "Similar", (4, 3), "A (4, 3)", GREEN, "16°", 36.87, 53.13, "cos = 0.96",
               "q·A = 3×4 + 4×3 = 24\n24 ÷ (5 × 5) = 0.96"),
              (8.0, "Unrelated", (4, -3), "B (4, −3)", ORANGE, "90°", -36.87, 53.13, "cos = 0",
               "q·B = 3×4 + 4×(−3) = 0\n0 ÷ 25 = 0"),
              (13.1, "Opposite", (-3, -4), "C (−3, −4)", RED, "180°", 53.13, 233.13, "cos = −1",
               "q·C = −9 + (−16) = −25\n−25 ÷ 25 = −1")]
    for cx, title, v, vlab, c, ang, t1, t2, cos, calc in panels:
        text(ax, cx, 6.0, title, fs=18, bold=True, color=c)
        ax.plot([cx - 2.3, cx + 2.3], [oy, oy], color=LINE, lw=1)
        ax.plot([cx, cx], [oy - 1.9, oy + 1.9], color=LINE, lw=1)
        arrow(ax, cx, oy, cx + q[0] * s, oy + q[1] * s, color=BLUE, lw=3)
        text(ax, cx + q[0] * s - 0.1, oy + q[1] * s + 0.25, "q (3, 4)", fs=13, bold=True, color=BLUE, ha="right")
        arrow(ax, cx, oy, cx + v[0] * s, oy + v[1] * s, color=c, lw=3)
        tx, ty = cx + v[0] * s, oy + v[1] * s
        if v[0] > 0:
            text(ax, tx + 0.12, ty + (0.15 if v[1] > 0 else -0.2), vlab, fs=13, bold=True, color=c, ha="left")
        else:
            text(ax, tx - 0.12, ty, vlab, fs=13, bold=True, color=c, ha="right")
        ax.add_patch(Arc((cx, oy), 1.1, 1.1, theta1=t1, theta2=t2, color=INK, lw=1.8))
        mid = (t1 + t2) / 2
        import math
        r = 0.85 if ang != "16°" else 0.95
        text(ax, cx + r * math.cos(math.radians(mid)), oy + r * math.sin(math.radians(mid)), ang, fs=13, bold=True)
        text(ax, cx, 1.3, cos, fs=17, bold=True, color=c)
        text(ax, cx, 0.6, calc, fs=14, color=INK)
    return fig


V41 = dict(
    num="4.1", title="From words to vectors", minutes=9, segments="SW",
    slides=[
        dict(type="two_col", key="kw", title="Keyword search vs meaning search",
             left=("Keyword search", ["Q: “how do refunds work?”", "Looks for the letters r-e-f-u-n-d",
                                      "FAQ page says “return policy”", "No shared word → no result"]),
             right=("Meaning search (embeddings)", ["Q: “how do refunds work?”",
                                                    "Turns the question into a position",
                                                    "“return policy” sits right next to it",
                                                    "Found, even with zero shared words"])),
        dict(type="diagram", key="map", title="A map of meaning", draw="meaning_map",
             notes="Draw the axes first, then one cluster at a time, then drop the question star last.",
             draw_list=["Two axes: dimension 1, dimension 2",
                        "Cluster 'money back': refund, return policy, reimbursement, money back",
                        "Cluster 'time off': holiday, leave, vacation, sick day",
                        "Cluster 'IT help': laptop, password reset, VPN",
                        "Star for the question 'how do refunds work?' next to return policy",
                        "Write: close = similar, far = unrelated, real models have 384 to 3,072 dimensions"]),
        dict(type="bullets", key="vec", title="A vector is just a list of numbers",
             bullets=["A point on our map = two numbers, e.g. (7.0, 3.9)",
                      "A real embedding = hundreds or thousands of numbers, e.g. 1,536",
                      "A whole passage gets ONE vector, not one per word",
                      "No single number means “money” or “time”: meaning is spread across all of them",
                      "The model learned where to place text by reading huge amounts of it"]),
        dict(type="cards", key="uses", title="Where embeddings are used",
             cards=[("Search by meaning", "A shop search for “sneakers” also finds “running shoes”."),
                    ("Suggestions", "“Students who read this notice also read…”: nearby items."),
                    ("Grouping", "Cluster 5,000 support tickets by topic; spot duplicate questions."),
                    ("Feeding an LLM", "Find the few passages an LLM should read before it answers (RAG).")]),
    ],
    recap=dict(points=["An embedding turns text into a position: a list of numbers",
                       "Similar meaning → close positions, even with no shared words",
                       "One vector per whole passage (or question)",
                       "Keyword search matches letters; embeddings match meaning"],
               task="Write five sentences from your college or office rules. Sketch them on a 2-D map: which ones "
                    "would sit close together? Which question would land near each group?", label="TRY THIS"),
    shot_plan=[("0:00–0:40", "Slide {cover}, Slide {v41}, hook", "S"),
               ("0:40–2:10", "Slide {kw} keyword vs meaning", "S"),
               ("2:10–5:40", "Slide {map} → board: map of meaning", "W"),
               ("5:40–7:10", "Slide {vec} a vector is a list of numbers", "S"),
               ("7:10–8:20", "Slide {uses} where embeddings are used", "S"),
               ("8:20–9:00", "Slide {r41} recap", "S")],
    script=[
        "[Slide {cover}, Slide {v41}]",
        "Welcome to Module 4. So far we have talked to language models: tokens, temperature, prompts. Now we "
        "meet a different kind of model, one that does not write anything at all. It turns text into a list of "
        "numbers. Those lists are called embeddings, and they quietly power search boxes, recommendations and "
        "almost every document assistant you will ever build. In this module we answer three questions. What "
        "is an embedding? How do we create embeddings in real code? And how do we measure whether two of them "
        "are similar?",
        "[Slide {kw}]",
        "Let us start with the problem embeddings solve. A customer on a shopping website types: how do refunds "
        "work? A keyword search looks for the letters r, e, f, u, n, d in the help pages. But the FAQ never "
        "uses the word refund. It has a heading called return policy. No shared word, so keyword search finds "
        "nothing. The customer gives up and calls the support line.",
        "Meaning search works differently. It turns the question into a position, a place on a map of meaning. "
        "Every piece of the FAQ also has a position on the same map. And the return policy section sits right "
        "next to the question, because the two mean nearly the same thing. Found, with zero words in common.",
        "Keyword search has a second problem too. It matches words that look the same but mean different "
        "things. In an office handbook, search for leave, and you get the leave policy, but also a line that "
        "says: please leave your laptop at reception. Same letters, completely different meaning. An embedding "
        "looks at the whole sentence, so these two land in different places on the map.",
        "[Slide {map} → Board]",
        "Let me draw this map, because once you see it, you will never forget it.",
        "[Board: draw two axes, label them dimension 1 and dimension 2] Two lines, like the graphs you drew "
        "in school. Every piece of text gets a point on this graph: an x value and a y value.",
        "[Board: cluster 'money back' with refund, return policy, reimbursement, money back] Now I place some "
        "phrases. Refund. Return policy. Reimbursement. Money back. I put them close together, in one group, "
        "because they are all about getting money back.",
        "[Board: cluster 'time off' with holiday, leave, vacation, sick day] Far away in another corner: "
        "holiday, leave, vacation, sick day. Time off work. Close to each other, far from the money group.",
        "[Board: cluster 'IT help' with laptop, password reset, VPN] And a third corner: laptop, password "
        "reset, VPN. IT help.",
        "[Board: draw a star next to return policy, label it 'how do refunds work?'] Now the question arrives. "
        "How do refunds work? Where does it land? Right here, in the money-back group, next to return policy. "
        "So when we ask which stored text is closest to the question, the answer is the return policy. Not the "
        "holiday text, not the VPN text.",
        "[Board: write 'close = similar, far = unrelated'] That is the whole idea in one line. Close together "
        "means similar meaning. Far apart means unrelated.",
        "[Board: write '384 to 3,072 dimensions, not 2'] One honest warning. I drew two dimensions because that "
        "is all a whiteboard can hold. Real embedding models use a few hundred to a few thousand dimensions. "
        "A popular one from OpenAI uses 1,536. We cannot draw that, but the rule is exactly the same: close "
        "means similar.",
        "[Slide {vec}]",
        "So what is a vector? Do not let the word scare you. A point on our map is just two numbers, like 7.0 "
        "and 3.9. A vector is a list of numbers. A real embedding is simply a much longer list, for example "
        "1,536 numbers, one after another.",
        "Three things to remember. First, a whole passage gets one vector. Not one per word: one list for the "
        "entire paragraph, and one for the entire question. Long documents are first cut into smaller pieces "
        "called chunks, and each chunk gets its own vector; Module 6 shows how to cut them. Second, you cannot "
        "read the numbers like labels. No single number means money or time off. The meaning is spread across "
        "all of them together. Third, nobody typed these positions by hand. The model learned where to place "
        "text by reading enormous amounts of it.",
        "Here is a comparison that helps. GPS coordinates on your phone: two numbers place any shop in India "
        "on a map, and two shops with close coordinates are close in real life, even if their names have "
        "nothing in common. An embedding is GPS for meaning, with many more coordinates.",
        "[Slide {uses}]",
        "Where do we use this? Everywhere. Search by meaning: a shop search for sneakers also finds running "
        "shoes. Suggestions: students who read this notice also read the notices closest to it. Grouping: "
        "cluster five thousand support tickets by topic, or spot the same question asked twice. And feeding an "
        "LLM: find the few passages it should read before answering. That last one is called RAG, and it is "
        "where this course is heading.",
        "[Slide {r41}]",
        "Recap. An embedding turns text into a position, a list of numbers. Similar meaning gives close "
        "positions, even with no shared words. There is one vector per whole passage or question. Keyword "
        "search matches letters; embeddings match meaning.",
        "Try this: write five sentences from your college or office rules and sketch them on a two-dimensional "
        "map. Which would sit close together? In the next video: real embedding models, and the code that "
        "calls them.",
    ],
)

V42 = dict(
    num="4.2", title="Embedding models in practice", minutes=9, segments="SW",
    slides=[
        dict(type="diagram", key="pipe", title="Text in, a fixed-length list of numbers out", draw="pipeline",
             notes="Stress the middle box: one model for both rows.",
             draw_list=["Left top: chunk text 'Refunds are accepted within 14 days...'",
                        "Left bottom: question 'how do refunds work?'",
                        "Middle: one tall box 'Embedding model', write 'SAME model'",
                        "Right: two lists [0.012, -0.044, ..., 0.007], each '1,536 numbers'",
                        "Under the top list: stored in a vector database; under the bottom: used to search"]),
        dict(type="table", key="models", title="Choosing an embedding model",
             notes="Prices change: check the provider's pricing page before recording.",
             header=["Model", "Dimensions", "Runs where", "Cost (at the time of writing)"],
             widths=[2.2, 1.1, 1.8, 2.3],
             rows=[["OpenAI text-embedding-3-small", "1,536", "OpenAI API", "~$0.02 per million tokens"],
                   ["OpenAI text-embedding-3-large", "3,072", "OpenAI API", "~$0.13 per million tokens"],
                   ["all-MiniLM-L6-v2 (sentence-transformers)", "384", "Your own computer", "Free, uses your CPU/GPU"],
                   ["Open models such as BGE or E5", "768 – 1,024", "Own computer or a host", "Free or hosting cost"]]),
        dict(type="code", key="call", title="Your first embedding call", label="PYTHON",
             code='''from openai import OpenAI
client = OpenAI()        # reads OPENAI_API_KEY from env

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input="How many days of annual leave do I get?")
vec = resp.data[0].embedding
print(len(vec))                 # 1536
print(resp.usage.total_tokens)  # tokens you pay for

# an open model, running on your own machine
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode(["text one", "text two"])
print(vecs.shape)               # (2, 384)''',
             bullets=["The key lives in the environment, never in the code",
                      "One input → one vector",
                      "usage tells you what the call cost",
                      "Open models: no API bill, smaller vectors"]),
        dict(type="code", key="batch", title="Embedding many texts: batches of 100", label="PYTHON",
             code='''from openai import OpenAI

client = OpenAI(max_retries=5)  # waits, then retries
MODEL, BATCH = "text-embedding-3-small", 100

def embed_texts(texts):
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = client.embeddings.create(
            model=MODEL, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend(d.embedding for d in ordered)
    return vectors

def embed_query(question):
    return embed_texts([question])[0]''',
             bullets=["250 texts → calls of 100, 100 and 50",
                      "Sort by index: vector n must match text n",
                      "Rate limit hit? The SDK backs off and retries",
                      "Questions use the SAME function and model"]),
        dict(type="stats", key="cost", title="What embedding costs: a 300-chunk handbook",
             notes="Prices change: re-check OpenAI's pricing page on the recording day.",
             stats=[("3", "API calls at 100 chunks per call"),
                    ("120K", "tokens: 300 chunks × 400 tokens"),
                    ("$0.0024", "with -small at ~$0.02 per million tokens (-large: ~$0.016)")]),
        dict(type="cards", key="rules", title="Four rules for embeddings",
             cards=[("Same model both sides", "Stored texts and questions must use the same model, or their "
                                              "positions cannot be compared."),
                    ("Re-embed on change", "Switching models means embedding every stored text again."),
                    ("Dimensions must match", "An index built for 1,536 numbers cannot store a 3,072-number "
                                              "vector."),
                    ("Keep the text", "A vector cannot be turned back into words. Store the text next to it.")]),
    ],
    recap=dict(points=["An embedding model turns any text into a fixed-length vector",
                       "Pick a model on quality, dimensions, cost and where it runs",
                       "Send texts in batches (e.g. 100 per call) and keep them in input order",
                       "Same model for texts and questions; change the model → re-embed everything"],
               task="A help centre has 250 articles, each cut into 2 chunks of about 400 tokens. At 100 chunks per "
                    "call, how many API calls do you make? How many tokens, and what does it cost at $0.02 per "
                    "million?",
               label="PRACTICE TASK"),
    shot_plan=[("0:00–0:30", "Slide {v42}, hook", "S"),
               ("0:30–2:30", "Slide {pipe} → board: text in, numbers out", "W"),
               ("2:30–3:50", "Slide {models} choosing a model", "S"),
               ("3:50–4:50", "Slide {call} first embedding call", "S"),
               ("4:50–6:30", "Slide {batch} batches of 100", "S"),
               ("6:30–7:20", "Slide {cost} what it costs", "S"),
               ("7:20–8:30", "Slide {rules} four rules", "S"),
               ("8:30–9:00", "Slide {r42} recap", "S")],
    script=[
        "[Slide {v42}]",
        "In the last video we drew a map of meaning. But who decides where each sentence goes on that map? An "
        "embedding model. In this video we choose one, call it from Python, and learn the few rules that keep "
        "a real system from quietly going wrong.",
        "[Slide {pipe} → Board]",
        "[Board: left top box, chunk text 'Refunds are accepted within 14 days of delivery'] On the left, a "
        "piece of a shop's FAQ page: refunds are accepted within 14 days of delivery.",
        "[Board: middle tall box, 'Embedding model'] We send that text to an embedding model. Think of it as a "
        "machine with one job: text goes in, numbers come out.",
        "[Board: right box, a list of numbers, write '1,536 numbers'] Out comes a list: 0.012, minus 0.044, "
        "and so on. With OpenAI's text-embedding-3-small, always 1,536 numbers, whether the text is one word or "
        "a full page. These exact numbers are made up for the drawing. The list is saved in a vector database, "
        "together with the original text.",
        "[Board: left bottom box, question 'how do refunds work?'] Now a customer asks: how do refunds work? "
        "[Board: arrow into the SAME middle box] And here is the most important arrow on this board. The "
        "question goes into the same model. Not a different one, not a newer one. The same. [Board: right "
        "bottom list] Out comes another list of 1,536 numbers, and we use it to search.",
        "[Board: underline 'SAME model'] Why does this matter so much? Because every model builds its own map. "
        "House number 42 in Pune and house number 42 in Jaipur are not neighbours. Positions only mean "
        "something inside the same map.",
        "[Slide {models}]",
        "Which model should you pick? Here are four common choices. OpenAI's text-embedding-3-small gives 1,536 "
        "numbers and, at the time of writing, costs about two cents per million tokens. Its big brother, "
        "text-embedding-3-large, gives 3,072 numbers, retrieves a little better, and costs about six and a half times "
        "more. Or you can run an open model on your own computer with a library called sentence-transformers. "
        "all-MiniLM-L6-v2 is small and fast, with 384 numbers. Models like BGE or E5 sit in between. Choose on "
        "four things: quality on your own documents, vector size, cost, and where your data is allowed to go.",
        "[Slide {call}]",
        "Calling the API takes a few lines. Create a client; it reads your key from the environment variable "
        "OPENAI_API_KEY, so the key never sits in your code or on GitHub. Call client.embeddings.create with the "
        "model name and the text. The vector is in resp.data, item zero, dot embedding. Its length is 1,536. "
        "resp.usage tells you how many tokens you paid for. Below, the open-model version: load the model "
        "once, call encode, get one 384-number vector per text.",
        "[Slide {batch}]",
        "Real systems embed thousands of texts, so we send them in batches. The loop runs from zero in steps of "
        "100. With 250 texts, i is 0, then 100, then 200: three calls, of 100, 100 and 50 texts. At the time of "
        "writing the API accepts up to 2,048 inputs in one call, but 100 is a comfortable size. One call for "
        "100 texts is far faster than 100 separate calls.",
        "Every item in the response carries an index, its position in the batch. We sort by that index before "
        "saving, so text number 7 gets vector number 7. If the order ever got mixed up, the holiday text would "
        "be stored with the meaning of the refund text, and nobody would notice until answers went wrong.",
        "Next, rate limits. Providers cap how many requests and how many tokens you may send per minute. Go "
        "over, and the API replies with a rate-limit error. The OpenAI library waits a little and retries "
        "by itself; here we raise max_retries to five. And notice embed_query: one question is just a list of "
        "one, through the same function, so it always uses the same model.",
        "[Slide {cost}]",
        "What does this cost? Take a college handbook cut into 300 chunks of about 400 tokens. That is three "
        "API calls and 120,000 tokens. With the small model, about a quarter of a US cent, roughly twenty "
        "paise. With the large model, about 1.6 cents. Embedding is rarely the expensive part of an AI app.",
        "[Slide {rules}]",
        "Let me turn all this into four rules. One: same model for stored texts and questions. Two: if you "
        "change the model, you must re-embed every stored text, because the old vectors live on the old map. "
        "Three: dimensions must match. A database index built for 1,536 numbers cannot hold a 3,072-number "
        "vector. Four: keep the text. A vector cannot be turned back into words, so always store the original "
        "text next to it.",
        "[Slide {r42}]",
        "Recap. An embedding model turns any text into a fixed-length vector. Pick it on quality, size, cost "
        "and where it runs. Send texts in batches and keep them in order. Use the same model for texts and "
        "questions, and re-embed everything when the model changes.",
        "Practice task: a help centre has 250 articles, each cut into two chunks. How many API calls, how many "
        "tokens, and what does it cost? In the next video: how we measure whether two vectors are close.",
    ],
)

V43 = dict(
    num="4.3", title="Measuring similarity", minutes=12, segments="SW",
    slides=[
        dict(type="table", key="ways", title="Three ways to compare two vectors",
             header=["Measure", "How to compute it", "Similar vectors give"], widths=[1.3, 3.0, 1.9],
             rows=[["Dot product", "Multiply matching numbers, add them up", "A large positive number"],
                   ["Cosine similarity", "Dot product ÷ (length of a × length of b)", "Close to 1"],
                   ["Euclidean distance", "Straight-line distance between the two points", "Close to 0"]]),
        dict(type="diagram", key="angle", title="Cosine similarity is the angle", draw="angle",
             notes="Work the first panel fully on the board; the other two follow the same steps.",
             draw_list=["Three small axes side by side: Similar, Unrelated, Opposite",
                        "In each, the question arrow q = (3, 4)",
                        "Similar: A = (4, 3), angle 16°, dot 24, lengths 5 and 5, cos 0.96",
                        "Unrelated: B = (4, -3), angle 90°, dot 0, cos 0",
                        "Opposite: C = (-3, -4), angle 180°, dot -25, cos -1"]),
        dict(type="table", key="worked", title="The worked example in one table",
             header=["Pair", "Dot product", "Lengths", "Cosine", "Euclidean distance"],
             widths=[1.3, 1.7, 1.1, 1.0, 1.6],
             rows=[["q (3, 4) and A (4, 3)", "12 + 12 = 24", "5 and 5", "0.96", "√2 ≈ 1.41"],
                   ["q (3, 4) and B (4, −3)", "12 − 12 = 0", "5 and 5", "0", "√50 ≈ 7.07"],
                   ["q (3, 4) and C (−3, −4)", "−9 − 16 = −25", "5 and 5", "−1",
                    "√100 = 10"]]),
        dict(type="code", key="norm", title="Cosine similarity in Python", label="PYTHON · numpy",
             code='''import numpy as np

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

q = [3, 4]
print(cosine(q, [4, 3]))     # 0.96
print(cosine(q, [4, -3]))    # 0.0
print(cosine(q, [-3, -4]))   # -1.0

# real vectors from OpenAI already have length 1,
# so a plain dot product IS the cosine:
scores = np.array(text_vecs) @ np.array(q_vec)
top5 = np.argsort(-scores)[:5]   # 5 closest texts''',
             bullets=["@ is the dot product; norm is the length",
                      "Length 1 → cosine = dot product",
                      "All three measures then rank alike",
                      "A vector database does this for you, fast (Module 5)"]),
        dict(type="bullets", key="show", title="How apps show similarity scores",
             bullets=["Raw score: 0.87, often rounded to 2–4 decimals",
                      "As a percentage: score × 100 → “87% match”",
                      "As a bar, stars or High / Medium / Low label",
                      "Hidden: only the top few results are shown",
                      "Threshold: below a cut-off you chose by testing → “I could not find this”"]),
        dict(type="two_col", key="score", title="What “87% match” means",
             left=("It means", ["Cosine similarity 0.87, shown × 100",
                                "This text is closer to the question than the others",
                                "Useful for ranking and for spotting weak matches",
                                "Compare scores within one question"]),
             right=("It does NOT mean", ["87% chance the answer is right",
                                         "87% of the question is answered",
                                         "A fixed pass mark: 0.5 is not always bad",
                                         "Comparable across models or questions"])),
    ],
    recap=dict(points=["Dot product: multiply and add; length: √(sum of squares)",
                       "Cosine = dot ÷ (length × length) = the angle: 1 same, 0 unrelated, −1 opposite",
                       "OpenAI vectors have length 1, so cosine and dot product rank the same",
                       "“% match” is a similarity score for ranking, not a probability"],
               task="Compute the cosine similarity of q = (1, 2) with D = (2, 1) and with E = (2, 4). Which is more "
                    "similar? (Answers: 0.8 and 1.0: E points in exactly the same direction.)"),
    shot_plan=[("0:00–0:40", "Slide {v43}, hook", "S"),
               ("0:40–2:10", "Slide {ways} three measures", "S"),
               ("2:10–6:40", "Slide {angle} → board: worked cosine example", "W"),
               ("6:40–7:50", "Slide {worked} the example in one table", "S"),
               ("7:50–9:00", "Slide {norm} cosine in Python", "S"),
               ("9:00–10:00", "Slide {show} how apps show scores", "S"),
               ("10:00–11:20", "Slide {score} what % match means", "S"),
               ("11:20–12:00", "Slide {r43} recap", "S")],
    script=[
        "[Slide {v43}]",
        "We now have a vector for every stored text and a vector for the question. The system has to answer "
        "one question, thousands of times a second: how close are these two lists of numbers? In this video we "
        "calculate closeness by hand, with small numbers, so that the score you see in any search app stops "
        "being a mystery.",
        "[Slide {ways}]",
        "There are three common ways to compare two vectors. The first is the dot product: multiply the "
        "matching numbers and add everything up. Similar vectors give a large positive number. The second is "
        "cosine similarity: take the dot product and divide it by the lengths of the two vectors. Similar "
        "vectors give a number close to 1. The third is Euclidean distance: the straight-line distance between "
        "the two points, like measuring with a ruler. Similar vectors give a number close to 0. For text, "
        "cosine is the most common choice, so let us calculate one.",
        "[Slide {angle} → Board]",
        "[Board: draw small axes, title 'Similar'] I will use two numbers per vector instead of 1,536, so we can "
        "do it by hand. The rules are identical for long vectors; there are just more numbers to add.",
        "[Board: arrow q = (3, 4)] Our question vector is q equals 3, 4. Three steps to the right, four steps "
        "up. [Board: arrow A = (4, 3)] And a stored text, the return policy, A equals 4, 3.",
        "[Board: write q·A = 3×4 + 4×3] Step one, the dot product. Multiply the first numbers: 3 times 4 is 12. "
        "Multiply the second numbers: 4 times 3 is 12. Add them: 24.",
        "[Board: write |q| = √(9 + 16) = 5] Step two, the lengths. The length of a vector is the square root "
        "of the sum of its squares. For q: 3 squared is 9, 4 squared is 16, together 25, and the square root "
        "of 25 is 5. This is just Pythagoras from school. [Board: write |A| = 5] For A: 16 plus 9, again 25, "
        "length 5.",
        "[Board: write cos = 24 ÷ (5 × 5) = 0.96] Step three: divide the dot product by the two lengths "
        "multiplied together. 24 divided by 25 is 0.96. Very close to 1. [Board: draw the small arc, write 16°] "
        "And look at the drawing: the two arrows point in almost the same direction. The angle between them is "
        "only about 16 degrees. That is what cosine similarity really measures: the angle. Small angle, score "
        "near 1, similar meaning.",
        "[Board: second axes, 'Unrelated', B = (4, −3)] Now a text about something unrelated, B equals 4, "
        "minus 3. Dot product: 3 times 4 is 12, 4 times minus 3 is minus 12. Total: zero. The lengths are still "
        "5 and 5, so cosine is 0 divided by 25: zero. [Board: arc, 90°] The arrows are at a right angle. "
        "Cosine zero means: no relationship at all.",
        "[Board: third axes, 'Opposite', C = (−3, −4)] And the extreme case, C equals minus 3, minus 4. Dot "
        "product: minus 9 plus minus 16, that is minus 25. Divided by 25: minus 1. [Board: arc, 180°] The "
        "arrows point in exactly opposite directions.",
        "[Board: write the scale 1 same · 0 unrelated · −1 opposite] So cosine similarity always lies between "
        "minus 1 and plus 1. One: same direction. Zero: unrelated. Minus one: opposite. One note for real "
        "life: with text embeddings, you will rarely see negative scores. Two unrelated sentences usually "
        "still score a little above zero, because all text shares some basic features. What matters is which "
        "texts score highest.",
        "[Slide {worked}]",
        "Here is the whole example in one table. For A: dot product 24, cosine 0.96. For B: zero and zero. For "
        "C: minus 25 and minus 1. I also added the Euclidean distance, the ruler distance between the tips of "
        "the arrows. For A it is about 1.41, for B about 7.07, for C exactly 10. Every measure puts the three "
        "texts in the same order: A closest, then B, then C.",
        "One property of cosine worth noticing. Cosine only cares about direction, not length. A short text "
        "and a long text about exactly the same topic can have arrows of different lengths that point the same "
        "way, and cosine still calls them a perfect match. A plain dot product would give extra points to the "
        "longer arrow. For comparing meaning, direction is what we want.",
        "[Slide {norm}]",
        "In Python, cosine is one line with numpy. The at sign is the dot product, and norm gives the length. "
        "Run it on our three examples and you get 0.96, zero and minus one, exactly what we did by hand.",
        "Now a useful fact. OpenAI's embedding models return vectors that are already normalised: their length "
        "is exactly 1. Look at step three: we divided by length times length. If both lengths are 1, we divide "
        "by 1, so cosine equals the plain dot product. The last two lines use that: one multiplication scores "
        "every stored text, and argsort picks the five highest. For vectors of length 1, Euclidean distance "
        "gives the same ranking too, in reverse. In practice a vector database runs this search for you, very "
        "fast, and you choose the metric when you create the index. That is Module 5.",
        "[Slide {show}]",
        "Apps show these scores in different ways. Some show the raw number, 0.87. Many multiply by 100 and "
        "write 87 percent match. Some draw a bar, or a label like high, medium, low. Many hide the scores "
        "and just show the top few results. And careful apps use a threshold: if even the best score is below "
        "a cut-off you chose by testing on real questions, the app says, I could not find this, instead of "
        "guessing.",
        "[Slide {score}]",
        "That percentage confuses many people, so let us be precise. 87 percent match means: the cosine "
        "similarity was 0.87, and this text is closer to the question than texts with lower scores. It is "
        "great for ranking, and for spotting weak searches, when even the best match scores low.",
        "It is not a probability. It does not mean an 87 percent chance the answer is right, or that 87 "
        "percent of the question is answered. The scores of the top five results do not add up to 100, and "
        "cosine can even be negative, which no probability can be. There is no fixed pass mark either: with "
        "some older models almost everything scored above 0.7, while with newer ones 0.5 can be a perfectly "
        "good match. So compare scores within one question and one model. And when a result looks wrong, read "
        "the retrieved text. The text tells you far more than the number.",
        "[Slide {r43}]",
        "Recap. The dot product multiplies and adds; the length is the square root of the sum of squares. Cosine "
        "similarity is dot product divided by length times length, and it measures the angle: one for same, "
        "zero for unrelated, minus one for opposite. OpenAI vectors have length one, so cosine and dot product "
        "rank the same. And percent match is a score for ranking, not a probability.",
        "Practice task: compute the cosine similarity of q equals 1, 2 with D equals 2, 1, and with E equals 2, "
        "4. You should get 0.8 and exactly 1: E is twice as long but points the same way. Next module: the "
        "databases that store and search these vectors. And in the capstone project, DocuMind, you will see "
        "these embeddings and scores at work inside one real system.",
    ],
)

MODULE = dict(
    folder="Embeddings", kicker="EMBEDDINGS", title="Embeddings and similarity",
    subtitle="How text becomes numbers, and how we measure closeness",
    guide_intro="Two board segments carry the module (the map of meaning and the worked cosine example); the "
                "rest is slides with short, generic code (OpenAI Python SDK, sentence-transformers, numpy).",
    videos=[V41, V42, V43],
    guide_md="""
## Goal of this module

The learner can explain what an embedding is (a position on a map of meaning, stored as a list of
numbers), why meaning search finds "return policy" for "how do refunds work?" when keyword search does
not, choose an embedding model, call it from Python in batches while keeping the input order, estimate the
cost, compute cosine similarity by hand for small vectors, and read a "% match" score correctly.

## Where this sits in the course

- Before: Module 2 (tokens, how LLMs work) and Module 3 (prompts). Learners know what a token is and that
  an LLM only sees what we put in the prompt.
- Now: how a system decides which texts are relevant to a question: every stored text and every question
  becomes a vector, and closeness is measured with cosine similarity.
- Next: Module 5 (vector databases) stores these vectors and searches them fast. Module 6 explains how
  documents are cut into chunks; Module 7 joins everything into Retrieval-Augmented Generation (RAG).

## What we teach

Video 4.1 – From words to vectors

- Keyword search vs meaning search ("refund" vs "return policy"; "leave" the policy vs "leave your laptop")
- Board: 2-D map of meaning with three clusters and the question star
- A vector is a list of numbers; one vector per whole passage or question; meaning is spread across all numbers
- Where embeddings are used: meaning search, recommendations, grouping/duplicates, feeding an LLM (RAG)

Video 4.2 – Embedding models in practice

- Board: stored text and question go through the same model; a fixed-length list of numbers comes out
- Choosing a model: text-embedding-3-small (1,536), text-embedding-3-large (3,072), all-MiniLM-L6-v2 (384),
  open models such as BGE / E5 (768–1,024); quality, size, cost, where data may go
- First call: `client.embeddings.create`, `resp.data[0].embedding`, `resp.usage`; open-model `encode`
- Batches of 100, sort by `index`, rate limits and `max_retries`; `embed_query` = list of one
- Cost: 300 chunks × 400 tokens = 120,000 tokens → about $0.0024 (small) or $0.016 (large)
- Four rules: same model, re-embed on change, dimensions match the index, keep the text

Video 4.3 – Measuring similarity

- Dot product, cosine similarity, Euclidean distance
- Board: worked example q = (3, 4) with A = (4, 3) → 0.96, B = (4, −3) → 0, C = (−3, −4) → −1
- numpy version; OpenAI vectors are normalised (length 1), so cosine = dot product and all three measures rank alike
- How apps display scores (raw, %, bars, hidden, thresholds) and why a score is not a probability

## Recording notes

- Board 4.1: draw the axes first, then one cluster at a time; drop the question star last, so the
  "landing" is the moment of the video.
- Board 4.2: make the middle model box tall and write SAME in capitals; that arrow is the main lesson.
- Board 4.3: work the first panel fully (dot, lengths, divide, angle). Panels 2 and 3 can go faster.
  Keep a calculator off-screen to check, but do the arithmetic aloud.
- The numbers in the board vectors ([0.012, −0.044, …]) are illustrative. Say so once.
- Prices, batch limits and SDK defaults (slides {models}, {batch}, {cost}) change. Check the provider's
  pricing and API reference on the recording day.

## Practice task answers

- 4.2: 250 articles × 2 = 500 chunks → 5 API calls of 100. 500 × 400 = 200,000 tokens. At $0.02 per million:
  0.2 × $0.02 = $0.004.
- 4.3: q = (1, 2), D = (2, 1): (2 + 2) ÷ (√5 × √5) = 4 ÷ 5 = 0.8. q and E = (2, 4): 10 ÷ (√5 × √20) =
  10 ÷ 10 = 1.0.

## Common learner questions

- "Can I see what each of the 1,536 numbers means?" No. The dimensions are learned, not designed; meaning
  is spread across all of them. Only distances between vectors are meaningful.
- "Can I turn a vector back into the text?" Not reliably. That is why we store the original text next to
  the vector in the database.
- "Why not always use text-embedding-3-large?" It returns 3,072 numbers and can retrieve a little better,
  but costs about 6.5 times more, needs an index with dimension 3,072 and a full re-embed. Small is a
  sensible default for many English documents. (The text-embedding-3 models also accept a `dimensions`
  parameter to return shorter vectors.)
- "Is 0.45 a bad score?" Not necessarily. Scores depend on the model and the wording. Compare scores for
  the same question; look at the retrieved text, not just the number.
- "Does it work for Hindi?" The text-embedding-3 models are multilingual and handle many languages,
  including Hindi, though quality is usually best in English. Test with your own documents. Some open
  models are English-only; check the model card.
""",
    notes_md="""
------------------------------------------------------------------------

## Video 4.1 – From words to vectors

### The problem embeddings solve

A customer asks "how do refunds work?". The shop's FAQ has a section called "Return policy" and never
uses the word "refund".

| **Approach** | **What it compares** | **Result** |
|:---|:---|:---|
| Keyword search | The letters of the words | No shared word, nothing found |
| Meaning search (embeddings) | Positions on a map of meaning | "Return policy" is next to the question, found |

Keyword search also confuses words that look alike: "leave" (time off) and "please leave your laptop at
reception". An embedding looks at the whole sentence, so the two land far apart.

### A map of meaning

An embedding model places every piece of text at a position. Texts with similar meaning get nearby
positions; unrelated texts are far apart.

Toy example in two dimensions:

- Money back: refund, return policy, reimbursement, money back
- Time off: holiday, leave, vacation, sick day
- IT help: laptop, password reset, VPN

The question "how do refunds work?" lands in the money-back group, next to "return policy".

### A vector is a list of numbers

- A point on a 2-D map is two numbers, e.g. (7.0, 3.9).
- A real embedding has a few hundred to a few thousand numbers (384 to 3,072 for common models;
  1,536 for OpenAI's `text-embedding-3-small`).
- A whole passage (a chunk of a document, see Module 6) or a whole question gets one vector, not one per word.
- No single number has a human meaning such as "money". Meaning is spread across all dimensions; only the
  distances between vectors are meaningful.
- The model learned its positions by training on very large amounts of text, so texts used in similar
  contexts end up close together.

### Where embeddings are used

| **Use** | **Example** |
|:---|:---|
| Search by meaning | A shop search for "sneakers" also finds "running shoes" |
| Recommendations | "Students who read this notice also read…" = the nearest notices |
| Grouping and duplicates | Cluster support tickets by topic; detect the same question asked twice |
| Feeding an LLM (RAG) | Find the passages an LLM should read before it answers (Module 7) |

------------------------------------------------------------------------

## Video 4.2 – Embedding models in practice

### Choosing a model

| **Model** | **Dimensions** | **Runs where** | **Cost (at the time of writing)** |
|:---|:---|:---|:---|
| OpenAI `text-embedding-3-small` | 1,536 | OpenAI API | About $0.02 per million tokens |
| OpenAI `text-embedding-3-large` | 3,072 | OpenAI API | About $0.13 per million tokens |
| `all-MiniLM-L6-v2` (sentence-transformers) | 384 | Your own computer | Free; uses your CPU/GPU |
| Open models such as BGE or E5 | 768 – 1,024 | Own computer or a hosting service | Free or hosting cost |

Choose on: quality on your own documents and questions, vector size (storage and speed), cost, and where
your data is allowed to go (an API sends text to the provider; a local model keeps it on your machine).
The OpenAI text-embedding-3 models accept up to 8,191 tokens per input, use the `cl100k_base` tokenizer,
and return vectors normalised to length 1.

### A first call

```python
from openai import OpenAI
client = OpenAI()                         # reads OPENAI_API_KEY from the environment

resp = client.embeddings.create(model="text-embedding-3-small",
                                input="How many days of annual leave do I get?")
vec = resp.data[0].embedding
print(len(vec))                           # 1536
print(resp.usage.total_tokens)            # tokens billed for this call

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode(["text one", "text two"])
print(vecs.shape)                         # (2, 384)
```

### Many texts: batches, order, rate limits

```python
from openai import OpenAI

client = OpenAI(max_retries=5)            # SDK waits and retries on rate-limit errors
MODEL, BATCH = "text-embedding-3-small", 100

def embed_texts(texts):
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = client.embeddings.create(model=MODEL, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend(d.embedding for d in ordered)
    return vectors

def embed_query(question):
    return embed_texts([question])[0]
```

| **Line(s)** | **What it does** | **Why** |
|:---|:---|:---|
| `OpenAI(max_retries=5)` | Retries failed requests with increasing waits | Rate limits cap requests and tokens per minute |
| `range(0, len(texts), BATCH)` | i = 0, 100, 200, … | Up to 100 texts per call: far fewer round trips |
| `client.embeddings.create(...)` | One API call for the batch | Returns one embedding per input, each with an `index` |
| `sorted(..., key=lambda d: d.index)` | Puts vectors in input order | Vector n must belong to text n |
| `embed_query` | One question → one vector (`[0]`) | Same function, so always the same model |

At the time of writing the API accepts up to 2,048 inputs per call; 100 is a comfortable batch. The SDK
retries some errors by default (twice, at the time of writing); production code may add its own retries
with back-off (Module 8).

### Cost

| **Example** | **Calculation** | **Result** |
|:---|:---|:---|
| Handbook, 300 chunks of ~400 tokens | 300 ÷ 100 | 3 API calls |
| Tokens | 300 × 400 | 120,000 tokens |
| Cost with -small | 0.12 million × $0.02 | about $0.0024 |
| Cost with -large | 0.12 million × $0.13 | about $0.016 |

Embedding is rarely the expensive part of an AI application; generation with the LLM usually is.

### Four rules

1. Same model for stored texts and questions. Each model builds its own map; positions from two maps cannot
   be compared (house number 42 in Pune is not next to house number 42 in Jaipur).
2. Changing the model means re-embedding every stored text.
3. The vector size must match the index: an index created with dimension 1,536 cannot store 3,072 numbers.
4. Keep the text: a vector cannot be turned back into words, so store the original text (and where it came
   from) next to the vector.

------------------------------------------------------------------------

## Video 4.3 – Measuring similarity

### Three measures

| **Measure** | **Formula (2-D)** | **Similar vectors give** |
|:---|:---|:---|
| Dot product | a·b = a₁b₁ + a₂b₂ | Large positive number |
| Length | \\|a\\| = √(a₁² + a₂²) | (used by cosine) |
| Cosine similarity | a·b ÷ (\\|a\\| × \\|b\\|) | Close to 1 (range −1 to 1) |
| Euclidean distance | √((a₁−b₁)² + (a₂−b₂)²) | Close to 0 |

For 1,536 dimensions the formulas are the same, with 1,536 terms instead of 2.

### Worked example

Question q = (3, 4). Three stored texts: A = (4, 3), B = (4, −3), C = (−3, −4).

Lengths: \\|q\\| = √(9 + 16) = √25 = 5. A, B and C also have length 5.

| **Pair** | **Dot product** | **Cosine** | **Angle** | **Euclidean distance** |
|:---|:---|:---|:---|:---|
| q, A | 3×4 + 4×3 = 24 | 24 ÷ 25 = 0.96 | about 16° | √((3−4)² + (4−3)²) = √2 ≈ 1.41 |
| q, B | 3×4 + 4×(−3) = 0 | 0 ÷ 25 = 0 | 90° | √(1 + 49) = √50 ≈ 7.07 |
| q, C | 3×(−3) + 4×(−4) = −25 | −25 ÷ 25 = −1 | 180° | √(36 + 64) = 10 |

Reading cosine: 1 = same direction, 0 = unrelated (right angle), −1 = opposite. With real text embeddings
negative scores are rare; unrelated texts usually score a little above zero. What matters is the ranking.

Cosine ignores length. q = (1, 2) and E = (2, 4) have cosine exactly 1 (same direction, different length);
q = (1, 2) and D = (2, 1) have cosine (2 + 2) ÷ (√5 × √5) = 4 ÷ 5 = 0.8.

### In Python

```python
import numpy as np

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

q = [3, 4]
print(cosine(q, [4, 3]))      # 0.96
print(cosine(q, [4, -3]))     # 0.0
print(cosine(q, [-3, -4]))    # -1.0

# normalised vectors (length 1): the dot product is the cosine
scores = np.array(text_vecs) @ np.array(q_vec)   # one score per stored text
top5 = np.argsort(-scores)[:5]                    # indexes of the 5 closest texts
```

### Why the metric choice matters little for normalised vectors

OpenAI embeddings are normalised to length 1. Then:

- cosine = dot product ÷ (1 × 1) = dot product;
- Euclidean distance² = 2 − 2 × cosine, so a smaller distance always means a higher cosine.

All three measures produce the same ranking. Cosine is the usual default for text because it also works for
models whose vectors are not normalised. A vector database (Module 5) computes it for you; you pick the
metric when you create the index.

### How apps show scores

- The raw score (0.87), the score × 100 ("87% match"), a bar or star rating, or a High / Medium / Low label.
- Often hidden: only the top few results are shown.
- With a threshold: if even the best score is below a cut-off chosen by testing on real questions, the app
  says "I could not find this" instead of guessing.

### Why a similarity score is not a probability

| **"87% match" means** | **It does not mean** |
|:---|:---|
| Cosine similarity of 0.87 | An 87% chance the answer is correct |
| Ranked above texts with lower scores | 87% of the question is answered |
| Useful to spot weak retrieval (best score is low) | A universal pass mark (0.5 can be fine) |
| Comparable within one question and one model | Comparable across models or very different questions |

The scores of the top results do not add up to 1, cosine can be negative, and the typical range depends on
the model (with some older models nearly everything scored above 0.7). When a result looks wrong, read the
retrieved text, not just the number.
""",
    quiz=[
        dict(q="A customer asks 'how do refunds work?' and the FAQ only says 'return policy'. Why does a "
               "meaning search still find the right section?",
             a="The search engine corrects spelling mistakes", b="The question and the section have nearby "
             "embeddings because they mean similar things", c="The LLM reads every page of the website",
             d="The word 'refund' is hidden in the page metadata", correct="B",
             why="Embeddings place text by meaning, so the question lands next to 'return policy' even with no "
                 "shared keyword."),
        dict(q="How many vectors does an embedding model create for one 400-token chunk?",
             a="One per token (400)", b="One per word", c="One for the whole chunk", d="1,536", correct="C",
             why="The whole chunk becomes a single vector (for example 1,536 numbers with text-embedding-3-small)."),
        dict(q="You embed 250 texts with a batch size of 100. How many API calls are made?",
             a="2", b="3", c="25", d="250", correct="B",
             why="The batches hold 100, 100 and 50 texts: three calls."),
        dict(q="Why does the batching code sort the response by item.index before saving the vectors?",
             a="To make the API call cheaper", b="So that vector number n belongs to text number n",
             c="To remove duplicate texts", d="Vector databases only accept sorted vectors", correct="B",
             why="If the order were mixed up, texts would be stored with another text's meaning."),
        dict(q="An app switches from text-embedding-3-small (1,536 numbers) to a model that returns 3,072 "
               "numbers. What must happen?",
             a="Nothing: only new questions use the new model",
             b="Only the question embeddings change",
             c="Create an index with dimension 3,072 and re-embed every stored text",
             d="Restart the web server", correct="C",
             why="Vectors from different models are not comparable, and the index dimension must match."),
        dict(q="A handbook gives 300 chunks of about 400 tokens. At about $0.02 per million tokens, roughly "
               "what does embedding all of it cost?",
             a="$0.0024", b="$0.24", c="$2.40", d="$24", correct="A",
             why="300 × 400 = 120,000 tokens = 0.12 million; 0.12 × $0.02 = $0.0024."),
        dict(q="Vectors q = (3, 4) and A = (4, 3) both have length 5. What is their cosine similarity?",
             a="0.96", b="0.75", c="24", d="1.41", correct="A", why="Dot product 3×4 + 4×3 = 24; "
             "24 ÷ (5 × 5) = 24 ÷ 25 = 0.96."),
        dict(q="Two vectors meet at a right angle (90°). What is their cosine similarity?",
             a="1", b="0", c="−1", d="0.5", correct="B", why="cos 90° = 0: the vectors are unrelated."),
        dict(q="Why do cosine similarity and dot product rank OpenAI embeddings the same way?",
             a="The API converts one into the other",
             b="OpenAI embeddings have length 1, so dividing by the lengths changes nothing",
             c="Both always equal 1", d="They do not; they give opposite rankings", correct="B",
             why="Cosine = dot ÷ (length × length). With length 1 for every vector, cosine equals the dot product."),
        dict(q="A search app shows a result with '62% match'. What is the correct reading?",
             a="There is a 62% chance the answer is correct",
             b="62% of the question is answered by this result",
             c="Its cosine similarity to the question is 0.62; compare it with the other results' scores",
             d="The result failed, because anything under 70% is wrong", correct="C",
             why="The score is cosine similarity × 100. It ranks results for one question; it is not a probability "
                 "or a pass mark."),
    ],
)
