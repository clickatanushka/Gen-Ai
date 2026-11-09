"""Module 5 – Vector databases: why they exist, indexes and records, querying and housekeeping.

Schema: see m01.py.
"""
import numpy as np

from style import *


def ann():   # brute force vs HNSW layers
    fig, ax = canvas()
    # left: brute force
    text(ax, 3.8, 6.0, "Brute force: compare with every vector", fs=17, bold=True, color=CORAL)
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0.6, 7.0, 42), rng.uniform(1.15, 5.2, 42)])
    qx, qy = 3.8, 3.15
    for x, y in pts:
        ax.plot([qx, x], [qy, y], color=CORAL, lw=0.8, alpha=0.45)
    ax.scatter(pts[:, 0], pts[:, 1], s=55, color=BLUE, zorder=3)
    ax.scatter([qx], [qy], s=420, marker="*", color=CORAL, zorder=4)
    text(ax, qx + 0.35, qy - 0.35, "question", fs=13, bold=True, color=CORAL, ha="left")
    text(ax, 3.8, 0.55, "1 question = 1 comparison per stored vector\n10 million vectors = 10 million comparisons",
         fs=13, color=MUTED)
    ax.plot([7.75, 7.75], [0.3, 6.2], color=LINE, lw=1.5, ls=(0, (4, 4)))
    # right: HNSW
    text(ax, 12.0, 6.0, "HNSW: highways first, then local roads", fs=17, bold=True, color=GREEN)
    layers = [(4.75, [9.4, 12.1, 14.9], "layer 2 · few nodes, long links"),
              (3.2, [8.9, 10.2, 11.5, 12.8, 14.1, 15.4], "layer 1"),
              (1.65, list(np.linspace(8.6, 15.6, 12)), "layer 0 · every vector")]
    for y, xs, lab in layers:
        ax.add_patch(FancyBboxPatch((8.25, y - 0.45), 7.7, 0.9, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=TINT[GREEN], ec=LINE, lw=1))
        text(ax, 15.9, y + 0.62, lab, fs=12, color=MUTED, ha="right")
        ax.plot(xs, [y] * len(xs), color=LINE, lw=1.5, zorder=2)
        ax.scatter(xs, [y] * len(xs), s=70, color=BLUE, zorder=3)
    l0 = layers[2][1]
    path = [(9.4, 4.75), (12.1, 4.75), (12.1, 3.2), (12.8, 3.2), (14.1, 3.2), (14.1, 1.65), (l0[9], 1.65)]
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        arrow(ax, x1, y1, x2, y2, color=CORAL, lw=2.5, ms=14, ls="-" if y1 == y2 else (0, (3, 2)))
    ax.scatter([l0[9]], [1.65], s=230, color=GREEN, zorder=4, edgecolors="white", linewidths=1.5)
    text(ax, 12.0, 0.55, "a few hops find a very close neighbour\nnot always the exact closest: approximate",
         fs=13, color=MUTED)
    return fig


def record():   # anatomy of one vector-database record
    fig, ax = canvas()
    y, h = 4.25, 2.3
    boxes = [(1.85, 3.1, BLUE, "id"), (6.05, 4.9, PURPLE, "values"), (12.3, 6.9, TEAL, "metadata")]
    for x, w, c, lab in boxes:
        box(ax, x, y, w, h, "", color=c)
        text(ax, x, y + h / 2 + 0.3, lab, fs=18, bold=True, color=c)
    text(ax, 1.85, 4.55, "refund_policy_7", fs=16, bold=True)
    text(ax, 1.85, 3.85, "doc name + _ +\nchunk index", fs=13, color=MUTED)
    text(ax, 6.05, 4.65, "[0.0123, −0.0451,\n0.0087, 0.0310, …]", fs=16, bold=True)
    text(ax, 6.05, 3.6, "1,536 numbers from the\nembedding model", fs=13, color=MUTED)
    rows = ["text: “Refunds are accepted within 14 days…”", "doc_name: refund_policy", "page_num: 3",
            "chunk_index: 7", "token_count: 400"]
    for i, r in enumerate(rows):
        text(ax, 9.1, 5.05 - i * 0.4, r, fs=14, ha="left")
    uses = [(1.85, BLUE, "find and overwrite\nthis exact record"), (6.05, PURPLE, "find similar meaning\n(the search)"),
            (12.3, TEAL, "filter, cite the page,\nbuild the prompt")]
    for x, c, s in uses:
        arrow(ax, x, y - h / 2 - 0.05, x, 2.05, color=c, lw=2)
        box(ax, x, 1.35, 3.0 if x < 3 else 3.6, 1.2, s, color=c, fs=14)
    return fig


def qflow():   # query → optional filter → cosine scores → top-5
    fig, ax = canvas()
    box(ax, 1.6, 3.3, 2.7, 1.2, "Question:\n“What is the\nrefund window?”", color=TEAL, fs=14)
    arrow(ax, 2.95, 3.3, 3.75, 3.3, color=TEAL, lw=2)
    box(ax, 5.0, 3.3, 2.4, 1.2, "query vector\n1,536 numbers", color=PURPLE, fs=14)
    funnel = [(6.85, 4.55), (9.65, 4.55), (8.75, 2.05), (7.75, 2.05)]
    ax.add_patch(Polygon(funnel, closed=True, fc=TINT[ORANGE], ec=ORANGE, lw=2.5))
    arrow(ax, 6.2, 3.3, 7.3, 3.3, color=PURPLE, lw=2)
    text(ax, 8.25, 4.9, "optional metadata filter", fs=13, bold=True, color=ORANGE)
    text(ax, 8.25, 3.75, "filter", fs=16, bold=True, color=ORANGE)
    text(ax, 8.25, 3.2, "then score\nby cosine", fs=13, color=INK)
    text(ax, 8.25, 1.6, r"{doc_name: {\$eq: refund_policy}}", fs=12, color=ORANGE)
    text(ax, 8.25, 1.15, "only chunks of that document", fs=12, color=MUTED)
    arrow(ax, 9.2, 3.3, 10.3, 3.3, color=ORANGE, lw=2)
    rows = [[1, "refund_policy_7", "0.8123"], [2, "refund_policy_8", "0.7741"], [3, "refund_policy_2", "0.6915"],
            [4, "refund_policy_12", "0.6402"], [5, "refund_policy_3", "0.6188"]]
    table(ax, 10.45, 5.05, ["#", "id", "score"], rows, [0.6, 3.3, 1.45], rh=0.55, color=BLUE, fs=14,
          highlight={(0, None): TINT[GREEN]})
    text(ax, 13.13, 1.45, "top_k = 5 · highest score first", fs=13, bold=True, color=BLUE)
    text(ax, 13.13, 1.0, "scores are illustrative", fs=12, color=MUTED)
    return fig


V51 = dict(
    num="5.1", title="Why a vector database", minutes=10, segments="SW",
    slides=[
        dict(type="bullets", key="why", title="The job: find the closest chunks, fast",
             bullets=["Every question becomes a vector, e.g. 1,536 numbers (Module 4)",
                      "We need the few stored chunk vectors closest to it, say the top 5",
                      "Simplest way: compare the question with every chunk (brute force)",
                      "1,000 chunks: instant. 10 million chunks: far too slow per question",
                      "A vector database makes this search fast, and keeps the data around it"]),
        dict(type="stats", key="scale", title="Why brute force does not scale",
             stats=[("1,536", "multiplications to compare the question with one chunk"),
                    ("10 M", "chunks in a large company's document collection"),
                    ("15 billion", "multiplications for one question, by brute force")],
             notes="1,536 × 10,000,000 = 15.36 billion."),
        dict(type="diagram", key="ann", title="Exact search vs approximate search", draw="ann",
             draw_list=["Left: a star (question) with lines to every dot: brute force, one comparison per vector",
                        "Right: three stacked layers, few dots on top, every dot at the bottom",
                        "Path: hop along the top layer, drop down, hop, drop down, land next to the answer",
                        "Write: approximate = very close, much faster, not always the exact closest"]),
        dict(type="cards", key="vdb", title="What a vector database adds",
             cards=[("ANN index", "Approximate nearest-neighbour search: a few hops instead of millions of "
                                  "comparisons."),
                    ("Storage + metadata", "Keeps the text, document name and page next to each vector."),
                    ("Filtering", "Search only inside one document, one department or one customer."),
                    ("Updates", "Add, overwrite and delete records without rebuilding everything.")]),
        dict(type="table", key="options", title="Vector search options you will hear about",
             header=["Option", "What it is", "A good fit when"], widths=[1.2, 2.6, 2.4], size=14,
             rows=[["Pinecone", "Managed cloud vector database (serverless)", "You do not want to run servers"],
                   ["pgvector", "Vector search extension for PostgreSQL", "Your data already lives in Postgres"],
                   ["Chroma", "Lightweight open-source store, runs locally", "Prototypes and notebooks"],
                   ["FAISS", "Search library from Meta, not a full database", "Fast search inside your own program"],
                   ["Weaviate, Qdrant", "Open-source vector databases, self-host or cloud",
                    "You want open source and control"]]),
    ],
    recap=dict(points=["Brute force compares the question with every vector: exact but slow at scale",
                       "ANN indexes such as HNSW (graph) and IVF (clusters) trade a tiny bit of accuracy for speed",
                       "A vector database = ANN index + storage + metadata + filters + updates",
                       "Pinecone, pgvector, Chroma, FAISS, Weaviate, Qdrant: same ideas, different code"],
               task="A college has 2,000 PDFs of 50 pages each, about 4 chunks per page. Estimate the number "
                    "of chunk vectors and the multiplications for one brute-force search (1,536 dimensions).",
               label="TRY THIS"),
    shot_plan=[("0:00–0:40", "Slide {cover}, Slide {v51}, hook", "S"),
               ("0:40–2:00", "Slide {why} the job", "S"),
               ("2:00–3:00", "Slide {scale} why brute force does not scale", "S"),
               ("3:00–6:30", "Slide {ann} → board: brute force vs HNSW", "W"),
               ("6:30–7:50", "Slide {vdb} what a vector database adds", "S"),
               ("7:50–9:20", "Slide {options} other options", "S"),
               ("9:20–10:00", "Slide {r51} recap", "S")],
    script=[
        "[Slide {cover}, Slide {v51}]",
        "Welcome to Module 5. In Module 4 we turned text into vectors: lists of numbers, for example 1,536 "
        "numbers, that capture meaning. We also learned that two pieces of text with similar meaning have "
        "vectors that point in similar directions. Now a very practical question. Imagine a help desk that "
        "answers questions from a company's policy documents. Every chunk of every document has been turned "
        "into a vector. Where do we keep all these vectors, and how do we find the right ones quickly? That is "
        "the job of a vector database.",
        "[Slide {why}]",
        "Let us state the job clearly. When a user asks a question, the system turns the question into a "
        "vector, using the same embedding model as the chunks. Then it needs the few stored chunk vectors that "
        "are closest to it, say the top five. Closest means most similar in meaning.",
        "The simplest way to do this is called brute force. You take the question vector and compare it with "
        "every single chunk vector, one by one. You compute the similarity score for each, sort them, and keep "
        "the top five. This always gives the exact answer.",
        "For a small project, brute force is completely fine. If you have three PDFs and a thousand chunks, "
        "the computer finishes in a blink. But think of a big company. Thousands of policies, manuals and "
        "contracts. Millions of chunks. Now brute force becomes a problem.",
        "[Slide {scale}]",
        "Here are the numbers. Comparing the question with one chunk takes 1,536 multiplications, one for each "
        "number in the vector, plus some additions. A large company collection can easily hold ten million "
        "chunks. Multiply the two, and one question needs about fifteen billion multiplications. And that is "
        "for one user, asking one question. If a hundred employees ask at the same time, the server is in "
        "trouble. We need a smarter way to search.",
        "[Slide {ann} → Board]",
        "Let me draw the difference.",
        "[Board: on the left, many dots, a star in the middle, lines from the star to every dot] On the left is "
        "brute force. The star is the question. Every dot is a stored chunk. Brute force draws a line from the "
        "star to every dot and measures each one. Exact, but the work grows with every dot you add.",
        "[Board: on the right, three stacked layers: few dots on top, more in the middle, every dot at the "
        "bottom] On the right is the idea behind most vector databases. It is called approximate nearest "
        "neighbour search, or ANN for short. The most popular method is called HNSW. Do not worry about the "
        "long name. Think of it as a road map.",
        "[Board: path along the top layer] The top layer is like the highway. It has very few points, and long "
        "links between them. You start there and jump quickly towards the part of the map where your question "
        "lives. [Board: dashed arrow down, then hops in the middle layer] Then you take the exit and drop to the "
        "next layer. It has more points, like city roads. You hop a few more times, getting closer. [Board: "
        "dashed arrow down, land next to the green dot] Finally you reach the bottom layer, which has every "
        "chunk, like the small lanes near a house. A few hops there, and you are standing next to your answer.",
        "[Board: write approximate = very close, much faster] Here is the catch, and it is in the name: "
        "approximate. HNSW finds a neighbour that is very close, but not always the exact closest one. In "
        "practice it finds the true best match almost every time, and it does it in a few hops instead of "
        "millions of comparisons. For document search, that trade is excellent.",
        "There is another family of ANN methods you may read about, called IVF. It first groups all vectors into "
        "clusters, like sorting library books onto shelves by topic. At question time it only searches the few "
        "shelves that look most promising. Different idea, same goal: look at a small part of the data instead "
        "of all of it.",
        "[Slide {vdb}]",
        "So what exactly is a vector database? It is more than a fast search. It adds four things. First, the "
        "ANN index we just drew. Second, storage with metadata. Next to each vector it keeps information about "
        "it, such as the chunk's text, the document name and the page number. Third, filtering. You can say: "
        "search only inside this one document, or only this department's files. Fourth, updates. You can add "
        "new records, overwrite old ones and delete them, without rebuilding the whole index.",
        "[Slide {options}]",
        "There are many vector databases, and you will hear these names in interviews. Pinecone is a managed "
        "cloud service. You send vectors over the internet and you never run a server yourself. It is the worked "
        "example in this module. pgvector is an extension that adds vector search to PostgreSQL, the popular "
        "database. If your company already uses Postgres, it is a natural fit. Chroma is a lightweight "
        "open-source store that runs on your own laptop, great for prototypes and notebooks. FAISS is a search "
        "library from Meta. It is very fast, but it is a library, not a database: you manage saving, updating "
        "and metadata yourself. And Weaviate and Qdrant are open-source vector databases you can host yourself "
        "or use in the cloud.",
        "The ideas in this module apply to all of them: vectors, metadata, top-k search, filters. Only the code "
        "changes.",
        "[Slide {r51}]",
        "Recap. Brute force compares the question with every vector. It is exact, but too slow at scale. ANN "
        "indexes like HNSW and IVF trade a tiny bit of accuracy for a huge gain in speed. A vector database is "
        "an ANN index plus storage, metadata, filters and updates. And the big names all share these ideas.",
        "Try this: a college has 2,000 PDFs of 50 pages each, with about four chunks per page. How many chunk "
        "vectors is that, and how many multiplications would one brute-force search need? In the next video we "
        "create an index in Pinecone and look closely at what one record contains.",
    ],
)

V52 = dict(
    num="5.2", title="Indexes and records", minutes=10, segments="SWR",
    slides=[
        dict(type="code", key="getindex", title="Creating an index (Pinecone)", label="PYTHON · PINECONE SDK",
             code='''import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX = "help-desk"

if INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX,
        dimension=1536,        # = embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX)''',
             bullets=["API key from the environment, never in code", "Create only if missing, then connect",
                      "Every later call goes through index"]),
        dict(type="two_col", key="settings", title="Four settings fixed at creation",
             left=("The settings", ["name: help-desk", "dimension: 1536", "metric: cosine",
                                    "serverless: aws, us-east-1"]),
             right=("Why they matter", ["Dimension must equal the embedding size",
                                        "Cosine suits OpenAI embeddings",
                                        "Changing them means a new index",
                                        "Serverless: pay for use, no servers to size"])),
        dict(type="diagram", key="record", title="Anatomy of one record", draw="record",
             draw_list=["Three boxes side by side: id | values | metadata",
                        "id: refund_policy_7 = doc name + _ + chunk index",
                        "values: [0.0123, −0.0451, …] = 1,536 numbers",
                        "metadata: text, doc_name, page_num, chunk_index, token_count",
                        "Arrows down: id → overwrite, values → search, metadata → filter, cite, prompt"]),
        dict(type="code", key="upsert", title="Saving chunks: upsert in batches", label="PYTHON · PINECONE SDK",
             code='''records = []
for chunk, vec in zip(chunks, vectors):
    records.append({
        "id": f"{chunk['doc_name']}_{chunk['chunk_index']}",
        "values": vec,                     # 1,536 floats
        "metadata": {"text": chunk["text"],
                     "doc_name": chunk["doc_name"],
                     "page_num": chunk["page_num"]},
    })

BATCH = 100
for i in range(0, len(records), BATCH):
    index.upsert(vectors=records[i : i + BATCH])''',
             bullets=["Upsert = insert, or overwrite if the id exists", "100 records per request",
                      "The text travels with the vector"]),
        dict(type="cards", key="stale", title="The re-ingest trap",
             cards=[("First ingest", "Chunk size 400 gives 30 chunks: ids refund_policy_0 to refund_policy_29."),
                    ("Re-ingest", "Chunk size 800 gives 15 chunks. Upsert overwrites ids _0 to _14 only."),
                    ("Stale vectors", "Ids _15 to _29 still hold the old text. Search can return them."),
                    ("The fix", "Delete all of the document's records first, then ingest it again.")]),
    ],
    recap=dict(points=["An index has a name, a dimension, a metric and a cloud region",
                       "Dimension must match the embedding model, e.g. 1536 for text-embedding-3-small",
                       "A record = id + values + metadata; the text lives in the metadata",
                       "Upsert overwrites by id, so delete before re-ingesting with new settings"],
               task="A PDF ingested with chunk size 400 made 48 chunks. You re-ingest it with chunk size 600 "
                    "and get 32. Which ids are stale, and what would you do?"),
    shot_plan=[("0:00–0:40", "Slide {v52}, hook", "S"), ("0:40–2:10", "Slide {getindex} creating an index", "S"),
               ("2:10–3:20", "Slide {settings} four settings", "S"),
               ("3:20–5:40", "Slide {record} → board: anatomy of a record", "W"),
               ("5:40–6:30", "Screen: Pinecone console, demo index help-desk, browse one record", "R"),
               ("6:30–8:00", "Slide {upsert} upsert in batches", "S"),
               ("8:00–9:20", "Slide {stale} the re-ingest trap", "S"), ("9:20–10:00", "Slide {r52} recap", "S")],
    script=[
        "[Slide {v52}]",
        "In the last video we learned why vector databases exist. Now let us use one. Our worked example is "
        "Pinecone, but every idea here, the index, the record, the upsert, exists in the other databases too. "
        "Our running example is an online shop's help desk, with documents like the refund policy and the "
        "shipping FAQ. By the end of this video you will know what an index is, what one record looks like, "
        "and one trap that catches many beginners.",
        "[Slide {getindex}]",
        "Everything starts with the index. First we read the Pinecone API key from the environment. That is a "
        "secret key, so it lives in a .env file or a secrets manager, never inside your code. Then we create a "
        "Pinecone client and ask: which indexes already exist in my account?",
        "An index is like a table in a normal database. It is the container that holds all our vectors. If an "
        "index called help-desk does not exist yet, the code creates it. The first time you run this, it takes "
        "a little while. After that, the index is already there, and the code simply connects to it. Every "
        "later step, saving, searching, deleting, goes through this index object.",
        "[Slide {settings}]",
        "When we create the index we fix four settings. The name: help-desk. The dimension: 1,536. The metric: "
        "cosine. And the spec: serverless, on AWS in the us-east-1 region.",
        "Why do these matter? The dimension must exactly match the size of our embeddings. "
        "text-embedding-3-small returns 1,536 numbers, so the index expects 1,536. If you switched to a model "
        "that returns 3,072 numbers, the database would reject every upsert. The metric is how closeness is "
        "measured. We use cosine similarity, which is what OpenAI recommends for its embeddings, and what we "
        "studied in Module 4.",
        "These settings are fixed when the index is born. If you want a different dimension or metric, you "
        "create a new index and load your documents again. And serverless means you do not choose or pay for "
        "machines. You pay for how much you store and how much you search. Perfect for a project that starts "
        "small.",
        "[Slide {record} → Board]",
        "Now the most important picture in this module: one record. Every chunk becomes one record.",
        "[Board: three boxes side by side, labelled id, values, metadata] A record has three parts.",
        "[Board: in the id box write refund_policy_7] First, the id. A unique name for this record. A good "
        "habit is a deterministic id: build it from the document name, an underscore and the chunk number. So "
        "chunk number seven of the refund policy is always refund_policy_7. Not a random number. Because the id "
        "is predictable, you can find, overwrite or delete that chunk later.",
        "[Board: in the values box write a few numbers and dots] Second, the values. These are the 1,536 numbers "
        "of the embedding. This is the part the database searches.",
        "[Board: in the metadata box list text, doc_name, page_num, chunk_index, token_count] Third, the "
        "metadata. Extra information stored next to the vector: the chunk's actual text, the document name, the "
        "page number, the chunk index and the token count.",
        "[Board: arrows down from each box] Each part has a different job. The id lets us find and overwrite "
        "this exact record. The values let us find chunks with similar meaning. And the metadata lets us filter, "
        "cite the page, and build the prompt.",
        "Pause on one thing. Why store the full text in the metadata? Because a vector is just numbers. You "
        "cannot turn 1,536 numbers back into the sentence. When the search returns the five best matches, we "
        "need their text to put into the prompt, and their document and page to show the user as sources. "
        "Storing the text right next to the vector means one search gives us everything. One limit to know: "
        "Pinecone caps how much metadata one record can carry, around 40 kilobytes at the time of writing. A "
        "400-token chunk is only a couple of kilobytes, far below that.",
        "[Screen: Pinecone console, demo index help-desk, browse one record]",
        "Here is our demo index in the Pinecone console. You can see the dimension, 1,536, and the metric, "
        "cosine. I open one record. There is the id, there are the values, a long list of small numbers, and "
        "there is the metadata, with the text and the page number we just drew.",
        "[Slide {upsert}]",
        "Now the code that saves records. It walks through the chunks and their embeddings together. For each "
        "pair it builds a dictionary with three keys: id, values and metadata. Exactly the three boxes from "
        "the board.",
        "Then it sends them in batches of 100. Why batches? Sending one record per request would mean hundreds "
        "of slow network calls. Sending everything at once could hit the request size limit. A hundred at a "
        "time is a safe middle.",
        "And the word upsert is a mix of update and insert. If the id is new, the database inserts the record. "
        "If a record with that id already exists, it overwrites it.",
        "[Slide {stale}]",
        "That overwrite behaviour creates a trap. Say you load the refund policy with chunk size 400. You get "
        "30 chunks, ids zero to twenty-nine. Now you change the chunk size to 800 and load the same file "
        "again. Bigger chunks, so only 15 of them. Upsert overwrites ids zero to fourteen. But ids fifteen to "
        "twenty-nine are never touched. They still hold the old, smaller chunks. Your index now mixes two "
        "versions of the same document, and a search can happily return the stale ones.",
        "The fix is simple: delete every record of that document first, and then load it again. In a "
        "production system you make this automatic, so nobody has to remember.",
        "[Slide {r52}]",
        "Recap. An index has a name, a dimension, a metric and a cloud region, all fixed at creation. The "
        "dimension must match the embedding model. A record is an id, the values and the metadata, and the "
        "text lives in the metadata so one search returns everything. Upsert overwrites by id, so delete "
        "before you re-ingest with new settings.",
        "Practice task: a PDF loaded with chunk size 400 made 48 chunks. You re-ingest it with chunk size 600 "
        "and get 32. Which ids are stale? In the next video: searching, filtering and cleaning up.",
    ],
)

V53 = dict(
    num="5.3", title="Querying, filters and housekeeping", minutes=10, segments="SWR",
    slides=[
        dict(type="code", key="query", title="Searching: query the index", label="PYTHON · PINECONE SDK",
             code='''q_vec = embed("What is the refund window?")  # same model!

res = index.query(
    vector=q_vec,
    top_k=5,
    include_metadata=True,
    filter={"doc_name": {"$eq": "refund_policy"}},  # optional
)
for m in res.matches:
    print(m.id, round(m.score, 4), m.metadata["page_num"])
    context = m.metadata["text"]''',
             bullets=["top_k: how many matches to return", "include_metadata: bring back the text",
                      "filter: optional condition on metadata"]),
        dict(type="diagram", key="qflow", title="One search, step by step", draw="qflow",
             draw_list=["Question → query vector (1,536 numbers)",
                        "Funnel: optional filter doc_name $eq refund_policy",
                        "Inside the funnel: score candidates by cosine",
                        "Table: top 5 ids with scores, highest first"]),
        dict(type="table", key="result", title="What to keep from each match",
             header=["Field", "Comes from", "Used for"], widths=[1.3, 2.0, 2.9],
             rows=[["id", "match.id", "Unique key of the chunk"],
                   ["score", "match.score", "Ranking; often shown as “% match”"],
                   ["text", "metadata", "Goes into the prompt; preview for the user"],
                   ["doc_name", "metadata", "Citation: which document"],
                   ["page_num", "metadata", "Citation: which page"],
                   ["chunk_index", "metadata", "Position inside the document"]],
             notes="A cosine score is a similarity, not a probability: 0.81 does not mean 81% correct."),
        dict(type="code", key="house", title="Housekeeping: delete and stats", label="PYTHON · PINECONE SDK",
             code='''# remove every chunk of one document
index.delete(filter={"doc_name": {"$eq": "refund_policy"}})
# or, by id (works on every index type)
index.delete(ids=["refund_policy_0", "refund_policy_1"])

stats = index.describe_index_stats()
print(stats.total_vector_count, stats.dimension)

# keep tenants apart: one namespace each
index.upsert(vectors=batch, namespace="college_a")''',
             bullets=["Delete is how you fix stale vectors", "Stats: how many vectors, what dimension",
                      "Check delete-by-filter support for your index type"]),
        dict(type="two_col", key="list", title="Listing what is stored: do not fake it",
             left=("The tempting shortcut", ["Query with an all-zero dummy vector", "top_k = 100: sees at most 100 chunks",
                                             "One big PDF can fill all 100", "Other documents vanish from the list"]),
             right=("Better options", ["Keep a document registry (small table)", "Update it on ingest and on delete",
                                       "Or list record ids by prefix", "Or one namespace per tenant"])),
        dict(type="bullets", key="ns", title="Namespaces: rooms inside one index",
             bullets=["A namespace is a separate partition inside the same index",
                      "A query searches one namespace only",
                      "Typical use: one namespace per customer (tenant) or per department",
                      "Delete a whole namespace in one call",
                      "Small apps: one default namespace + a metadata filter is enough"]),
    ],
    recap=dict(points=["query = vector + top_k + include_metadata (+ optional filter)",
                       "Scores rank matches; a score is similarity, not probability",
                       "Metadata filters and namespaces narrow the search",
                       "To list contents, keep a registry or list ids; never a dummy-vector query"],
               task="Write the filter dictionary that limits a search to the document “hr_handbook”. Then "
                    "explain why a dummy-vector query with top_k = 100 might not list all 300 uploaded PDFs."),
    shot_plan=[("0:00–0:30", "Slide {v53}, hook", "S"), ("0:30–2:00", "Slide {query} querying the index", "S"),
               ("2:00–4:30", "Slide {qflow} → board: one search, step by step", "W"),
               ("4:30–5:40", "Slide {result} the result fields", "S"),
               ("5:40–6:40", "Screen: notebook, run a query and print matches; then describe_index_stats()", "R"),
               ("6:40–7:30", "Slide {house} delete and stats", "S"),
               ("7:30–8:50", "Slide {list} listing what is stored", "S"),
               ("8:50–9:30", "Slide {ns} namespaces", "S"), ("9:30–10:00", "Slide {r53} recap", "S")],
    script=[
        "[Slide {v53}]",
        "We know how chunks get into the index. Now the other direction: how we search, how we filter, and how "
        "we clean up. We will also look at one tempting shortcut that many projects take, because spotting "
        "shortcuts is a real engineering skill.",
        "[Slide {query}]",
        "Searching starts with the question. We embed it with the same model we used for the chunks. This is "
        "important: vectors from two different models live in different spaces, and comparing them gives "
        "nonsense. Then we call query with a few instructions. The vector: the embedded question. top_k: how "
        "many matches we want, here five. And include_metadata set to True, which says: please send back the "
        "text, document name and page, not just the ids. Without it we would get numbers and ids, and no text "
        "to put in the prompt.",
        "Then an optional filter. Here the rule says: doc_name must equal refund_policy. The dollar e q means "
        "equals. Most vector databases support similar operators, such as not equal, in a list, or greater "
        "than. A filter is useful when the user picks one document, or when each department should only see "
        "its own files.",
        "The answer is a list of matches. For each match we read the id, the score, and the text and page from "
        "the metadata.",
        "[Slide {qflow} → Board]",
        "Let me draw one search from start to end.",
        "[Board: box Question, arrow, box query vector] It starts with the question: what is the refund window? "
        "It has already been embedded into 1,536 numbers, using the same model as the chunks.",
        "[Board: a funnel, write filter on top] If there is a filter, the database first narrows the "
        "candidates. Only chunks whose doc_name is refund_policy are allowed in. Everything else is ignored, no "
        "matter how similar it is.",
        "[Board: inside the funnel write score by cosine] Then, using the ANN index from video 5.1, it finds "
        "the candidates whose vectors are closest to the question by cosine similarity.",
        "[Board: a table with five rows, id and score] And it returns the top k, here five, sorted with the "
        "highest score first. refund_policy_7 with 0.81, refund_policy_8 with 0.77, and so on. These scores are "
        "illustrative, but this is the shape of every answer.",
        "[Slide {result}]",
        "Here is what an application usually keeps from each match. The id. The score, used for ranking. The "
        "text, which goes into the prompt. The document name and page number, which become the citation the "
        "user sees. And the chunk index, the position inside the document.",
        "One warning about the score. Many apps show it as a percentage, like 81 percent match. But a cosine "
        "score is a similarity, not a probability. 0.81 does not mean the chunk is 81 percent likely to be "
        "correct. It only means this chunk is closer to the question than one with 0.70. Use scores to compare "
        "chunks within one search, not as a promise of truth.",
        "[Screen: notebook, run a query and print matches; then describe_index_stats()]",
        "In a notebook, I embed a question and run the query. Five matches print out, each with its id, score "
        "and page. Then I call describe_index_stats: total vectors, and the dimension, 1,536.",
        "[Slide {house}]",
        "Now housekeeping. To delete every chunk of one document, you can delete with a filter: every record "
        "whose doc_name equals refund_policy. This is the fix for the re-ingest trap from the last video. A word "
        "of caution: support for delete-by-filter has differed between Pinecone index types and versions. "
        "Deleting by ids works everywhere, and this is where deterministic ids pay off: you can rebuild the "
        "list refund_policy_0, refund_policy_1 and so on. Always test it on your own index.",
        "describe_index_stats tells you how many vectors are stored and their dimension. A quick health check. "
        "And the last line shows a namespace, which we will meet in a minute.",
        "[Slide {list}]",
        "Now the shortcut. Suppose your app wants to show a list of every document in the index. A tempting "
        "trick is to send a query with a dummy vector, all zeros, with top_k equal to 100, and collect the "
        "document names from the matches. It may seem to work in a small demo. Some databases even reject an all-zero vector.",
        "But think about it. A query returns at most top_k matches, so this trick can only ever see 100 "
        "chunks. A single 120-page PDF can easily have more than 100 chunks. If the 100 chunks that come back "
        "all belong to that one big PDF, every other document disappears from the list, even though it is still "
        "stored. And a vector of all zeros has no direction, so a cosine score against it means nothing.",
        "The rule: a similarity search is for finding similar meaning, not for listing contents. Better "
        "options: keep a document registry, a small table with each document's name, number of chunks and "
        "date, and update it whenever a document is loaded or deleted. Or list record ids by their prefix, "
        "since our ids start with the document name. Or give each tenant its own namespace. The bigger lesson: "
        "a shortcut that works at demo size can quietly break at real size.",
        "[Slide {ns}]",
        "A quick word on namespaces. A namespace is like a separate room inside one index. A query only "
        "searches the room you name. Companies often give each customer, or each college, or each department, "
        "its own namespace, so nobody can ever retrieve someone else's chunks. You can also delete a whole "
        "namespace in one call. A small app can keep everything in the default namespace and use a metadata "
        "filter instead.",
        "[Slide {r53}]",
        "Recap. A query is a vector, a top k and include metadata, plus an optional filter. Scores rank the "
        "matches, but they are similarity, not probability. Metadata filters and namespaces narrow the search. "
        "And to list what is stored, keep a registry or list ids, never a dummy-vector query.",
        "Practice task: write the filter dictionary that limits a search to the document hr_handbook. Then "
        "explain why a dummy-vector query might not list all 300 uploaded PDFs. In Module 6 we go back to the "
        "start of the pipeline: getting text out of PDFs and cutting it into chunks. In the capstone project, "
        "DocuMind, you will see an index, records and top-5 search working inside one real system.",
    ],
)

MODULE = dict(
    folder="Vector_Databases", kicker="VECTOR DATABASES", title="Vector databases",
    subtitle="Why they exist, indexes and records, searching and housekeeping",
    guide_intro="Slides and short generic Pinecone code, with three board segments and short screen "
                "recordings of the Pinecone console and a notebook.",
    script_intro="Screen segments show the Pinecone console with a small demo index `help-desk` (dimension "
                 "1536, cosine) already loaded with two or three policy PDFs, and a Python notebook that "
                 "queries it.",
    videos=[V51, V52, V53],
    guide_md="""
## Goal of this module

The learner can explain why brute-force similarity search does not scale and what approximate
nearest-neighbour (ANN) search trades for speed; describe what a vector database adds on top of an
index; create an index with the right dimension and metric; draw one record (id, values, metadata)
and say what each part is for; upsert in batches with deterministic ids; query with top-k, metadata
and filters; and avoid two real pitfalls: stale vectors after re-ingesting with new chunk settings,
and using a dummy-vector query to list what is stored.

## Where this sits in the course

- Before (Module 4): chunks and questions become vectors (e.g. 1,536 numbers); cosine similarity
  measures closeness of meaning.
- Now: where those vectors are stored and how the closest ones are found, fast.
- Next: Module 6 goes back to the start of the pipeline (PDF text and chunking); Module 7 puts the
  whole RAG flow together and shows how retrieval failures look.
- After the course: the capstone project use exactly these ideas (one serverless index,
  1536 dimensions, cosine, top-5); no new theory is needed there.

## What we teach

Video 5.1 – Why a vector database

- Brute force: one comparison per stored vector; 1,536 × 10 million ≈ 15 billion multiplications
- ANN; HNSW as highways → city roads → lanes; IVF as clusters ("shelves") in one sentence
- Vector database = ANN index + storage + metadata + filtering + updates
- Options: Pinecone, pgvector, Chroma, FAISS (library), Weaviate, Qdrant

Video 5.2 – Indexes and records (Pinecone as the worked example)

- Create-if-missing; name, dimension 1536, metric cosine, serverless aws / us-east-1
- Dimension must match the embedding model; changing model or metric means a new index
- A record = id + values + metadata; deterministic ids `f"{doc_name}_{chunk_index}"`
- Why the text is stored in metadata (prompt + sources); metadata size limit
- Upsert in batches of 100; upsert = insert or overwrite
- The re-ingest trap: fewer chunks after re-chunking leaves stale records

Video 5.3 – Querying, filters and housekeeping

- `index.query(vector, top_k, include_metadata, filter)`; `$eq` and other operators
- Result fields; similarity is not probability
- Delete by filter or by ids; `describe_index_stats()`
- Listing contents: never a dummy-vector query; document registry, id prefixes, namespaces
- Namespaces for tenants

## Practice task answers

- 5.1: 2,000 × 50 × 4 = 400,000 chunk vectors. One brute-force search = 400,000 × 1,536 ≈ 614
  million multiplications. With numpy on a laptop that is still well under a second for one user,
  so brute force is fine at this size; it starts to hurt with many users or tens of millions of
  vectors.
- 5.2: the new ids are `_0` to `_31`; ids `_32` to `_47` (16 records) are stale. Delete the
  document's records first (by filter, or by ids `_0` … `_47`), then ingest again.
- 5.3: `{"doc_name": {"$eq": "hr_handbook"}}`. A dummy-vector query returns at most `top_k` = 100
  chunks; 300 PDFs have far more than 100 chunks, so many documents never appear. Keep a registry.

## Recording notes

- Board segments: 5.1 draw brute force on the left half first, then the three HNSW layers on the
  right; 5.2 three boxes for the record, arrows down to their uses; 5.3 left to right: question,
  vector, funnel, top-5 table.
- Screen 5.2: Pinecone console → Indexes → `help-desk` → Browser. Pick a record whose text is
  readable. Blur or crop the API key if the console shows it.
- Screen 5.3: a notebook with the query code from slide {query}; print id, score, page; then
  `describe_index_stats()`.
- Console layout and SDK method names change between versions; check the current Pinecone docs
  before recording.
- Scores on slide {qflow} are illustrative. If you read real scores aloud, read them from your screen.

## Common learner questions

- "Is approximate search risky for a company?" ANN usually returns the true nearest neighbours or
  ones almost as close. For document Q&A the bigger risks are bad chunking and vague questions
  (Modules 6 and 7), not ANN.
- "Why not store vectors in a normal SQL table?" You can (pgvector does exactly that inside
  Postgres). Without a vector index, though, every search is brute force.
- "Does the vector database see my documents?" With a managed cloud service, yes: the chunk text is
  stored as metadata in the provider's cloud. Companies with strict data rules choose a self-hosted
  option or check the provider's data terms.
- "Why does an app show 81% match if it is not a probability?" It is a friendly display of the
  cosine score × 100. Treat it as a ranking signal only.
- "What if I change the embedding model?" New model, new vector space, often a new dimension:
  create a new index and re-embed every document.
""",
    notes_md="""
------------------------------------------------------------------------

## Video 5.1 – Why a vector database

### The job

At question time a document assistant needs the k stored chunk vectors (often k = 5) that are most
similar to the question vector. Similarity is measured with cosine similarity (Module 4).

### Brute force (exact search)

Compare the question with every stored vector, sort by score, keep the top k.

| **Index size** | **Multiplications per question (1,536 per comparison)** | **Feels** |
|:---|:---|:---|
| 1,000 chunks | about 1.5 million | Instant |
| 1 million chunks | about 1.5 billion | Slow for many users |
| 10 million chunks | about 15 billion | Too slow per question |

Brute force is always exact. Its cost grows in direct proportion to the number of vectors.

### Approximate nearest neighbour (ANN)

ANN methods build an index in advance so that a search only looks at a small part of the data. They
return neighbours that are very close to the best, usually the true best, in a tiny fraction of the time.
"Approximate" means: not guaranteed to be the exact closest.

- HNSW (Hierarchical Navigable Small World): a multi-layer graph. The top layer has few points with long
  links (highways), lower layers have more points (city roads), the bottom layer has every vector
  (lanes). A search starts at the top, moves greedily towards the question, drops a layer, and repeats.
- IVF (inverted file): vectors are grouped into clusters in advance. A search compares the question with
  the cluster centres and only searches the few most promising clusters ("shelves").

### What a vector database adds

| **Feature** | **Meaning** | **Example** |
|:---|:---|:---|
| ANN index | Fast similarity search | Managed for you by the database |
| Storage + metadata | Data stored next to each vector | text, doc_name, page_num, chunk_index, token_count |
| Filtering | Search only records that match a condition | only the HR handbook; only one department |
| Updates | Add, overwrite, delete without rebuilding | `upsert`, `delete` |

### Options

| **Option** | **What it is** | **Good fit** |
|:---|:---|:---|
| Pinecone | Managed cloud vector database (serverless or pod-based) | No servers to run |
| pgvector | Extension adding vector columns and indexes to PostgreSQL | Data already in Postgres |
| Chroma | Lightweight open-source vector store, runs locally | Prototypes, notebooks |
| FAISS | Similarity-search library from Meta | Fast search inside your own program; you handle storage and metadata |
| Weaviate, Qdrant | Open-source vector databases, self-hosted or cloud | Open source, full control |

------------------------------------------------------------------------

## Video 5.2 – Indexes and records

### The index

An index is the container for vectors, like a table in a normal database. The usual pattern: read the
API key from the environment, check whether the index exists, create it only if it is missing, then
connect.

```python
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
if "help-desk" not in pc.list_indexes().names():
    pc.create_index(name="help-desk", dimension=1536, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index("help-desk")
```

| **Setting** | **Example value** | **Why it matters** |
|:---|:---|:---|
| name | `help-desk` | Which container to use |
| dimension | 1536 | Must equal the embedding size (text-embedding-3-small returns 1,536 numbers) |
| metric | cosine | How closeness is measured; recommended for OpenAI embeddings |
| spec | serverless, aws, us-east-1 | Pay per use; no machines to size |

Dimension and metric are fixed when the index is created. Changing either (or changing the embedding
model) means creating a new index and re-embedding every document. SDK method names can change between
versions; check the current docs.

### A record

| **Part** | **Example** | **Used for** |
|:---|:---|:---|
| id | `refund_policy_7` (doc name + `_` + chunk index) | Finding, overwriting or deleting this exact record |
| values | 1,536 floats | The similarity search |
| metadata | text, doc_name, page_num, chunk_index, token_count | Filtering, citations, building the prompt |

Deterministic ids (built from the document name and chunk index, not random) let you overwrite or
delete a document's chunks later. The text is stored in the metadata because an embedding cannot be
turned back into words; one query then returns everything the prompt and the source list need.
Pinecone limits the metadata size per record (about 40 KB at the time of writing; check the current
docs). A 400-token chunk is only a couple of kilobytes.

### Upsert

Build one dictionary per chunk (`id`, `values`, `metadata`) and send them in batches, for example 100
per request. Upsert means: insert if the id is new, overwrite if it exists. Batching avoids hundreds of
small network calls and stays under the request size limit.

### The re-ingest trap

| **Step** | **Chunks** | **Ids written** |
|:---|:---|:---|
| Ingest with chunk size 400 | 30 | `_0` … `_29` |
| Re-ingest with chunk size 800 | 15 | `_0` … `_14` overwritten |
| Result | | `_15` … `_29` still hold old 400-token text (stale) |

Fix: delete all records of the document, then ingest again. Production systems do this automatically,
or version their ids.

------------------------------------------------------------------------

## Video 5.3 – Querying, filters and housekeeping

### Querying

```python
q_vec = embed("What is the refund window?")   # same model as the chunks
res = index.query(vector=q_vec, top_k=5, include_metadata=True,
                  filter={"doc_name": {"$eq": "refund_policy"}})   # filter is optional
for m in res.matches:
    print(m.id, round(m.score, 4), m.metadata["page_num"])
```

- The question must be embedded with the same model as the chunks.
- `top_k`: number of matches to return.
- `include_metadata=True`: return the stored text and fields, not only ids and scores.
- `filter`: optional metadata condition. `$eq` means "equals"; Pinecone also supports operators such as
  `$ne`, `$in`, `$gt`. Other databases have similar filters with their own syntax.

### Result fields

| **Field** | **From** | **Used for** |
|:---|:---|:---|
| id | `match.id` | Unique key |
| score | `match.score` | Ranking; often shown as "% match" |
| text | metadata | Prompt context; preview for the user |
| doc_name, page_num | metadata | Citation, e.g. (Source: refund_policy, p.3) |
| chunk_index | metadata | Position in the document |

A cosine score measures similarity, not the probability of being correct. Compare scores within one
search; do not read 0.81 as "81% right".

### Housekeeping

- Delete by filter: `index.delete(filter={"doc_name": {"$eq": "refund_policy"}})` removes every chunk
  of one document. Support for deleting by metadata filter has differed between Pinecone index types
  and SDK versions; if unsupported, delete by ids (easy with deterministic ids).
- `describe_index_stats()`: total vector count and dimension (plus per-namespace counts).

### Listing what is stored: do not use a dummy-vector query

A tempting shortcut is to query with an all-zero vector and `top_k=100`, then collect distinct
`doc_name` values.

- A query returns at most `top_k` matches, so it sees at most 100 chunks. If one large PDF supplies
  all 100, other documents disappear from the list.
- A zero vector has no direction, so cosine scores against it are meaningless.
- Better: keep a document registry (name, chunk count, ingest date, chunk settings) updated on every
  ingest and delete; list record ids by prefix (ids start with the document name; Pinecone serverless
  offers id listing by prefix at the time of writing); or separate tenants with namespaces.

### Namespaces

A namespace is a partition inside one index. Queries, upserts and deletes target one namespace. Common
use: one namespace per customer, college or department, so data cannot mix. A small app can use the
default namespace and a metadata filter.
""",
    quiz=[
        dict(q="Why does brute-force vector search become a problem for a large company's documents?",
             a="It returns wrong answers", b="Its work grows with every stored vector: millions of comparisons per question",
             c="It cannot use cosine similarity", d="It only works with PDFs", correct="B",
             why="Brute force is exact but compares the question with every vector, so cost grows with index size."),
        dict(q="What does the word 'approximate' in approximate nearest neighbour (ANN) search mean?",
             a="Scores are rounded to 4 places", b="Only half the documents are searched at random",
             c="It finds very close neighbours fast, but not always the exact closest one",
             d="It ignores the metadata", correct="C",
             why="ANN trades a tiny bit of accuracy for a huge gain in speed."),
        dict(q="In the HNSW picture, what is the top layer like?",
             a="Every vector, densely connected", b="A highway: few points with long links",
             c="The metadata table", d="A list of document names", correct="B",
             why="Search starts on the sparse top layer, jumps close to the target, then drops to denser layers."),
        dict(q="Which option is a search library rather than a full vector database?",
             a="FAISS", b="Pinecone", c="Qdrant", d="Weaviate", correct="A",
             why="FAISS gives fast similarity search inside your program; storage, updates and metadata are up to you."),
        dict(q="An index was created with dimension 1536. What happens if you switch to an embedding model that returns 3,072 numbers?",
             a="The database pads the vectors automatically", b="Nothing changes",
             c="Upserts are rejected; you need a new index and must re-embed every document",
             d="Only the metric changes", correct="C",
             why="Dimension is fixed at creation and must match the embedding size."),
        dict(q="Why is each chunk's text stored in the record's metadata?",
             a="The database requires it", b="An embedding cannot be turned back into text, and the prompt and sources need the text",
             c="It makes the vectors smaller", d="To translate the text", correct="B",
             why="One query then returns the text, document and page needed for the prompt and the citations."),
        dict(q="Ids are built as f\"{doc_name}_{chunk_index}\". What is the id of the 13th chunk (chunk_index 12) of the document hr_handbook?",
             a="hr_handbook_13", b="hr_handbook_12", c="12_hr_handbook", d="a random UUID", correct="B",
             why="Chunk indexes start at 0, so the 13th chunk has index 12. Deterministic ids make records easy to find and delete."),
        dict(q="A PDF first gave 30 chunks. You re-ingest it with bigger chunks and get 15. What is in the index now?",
             a="Exactly the 15 new chunks", b="The 30 old chunks only",
             c="15 new chunks plus 15 stale old chunks (ids 15–29)", d="Nothing; upsert deletes first",
             correct="C", why="Upsert only overwrites ids it writes. Delete the document's records before re-ingesting."),
        dict(q="Which filter limits a search to the document hr_handbook?",
             a='{"doc_name": {"$eq": "hr_handbook"}}', b='{"id": "hr_handbook"}',
             c='{"text": {"$eq": "hr_handbook"}}', d='{"top_k": "hr_handbook"}', correct="A",
             why="Filters are conditions on metadata fields; $eq means equals."),
        dict(q="An app lists its documents by querying with an all-zero vector and top_k = 100. Why is this a bad idea?",
             a="It only reads the first PDF page", b="It sees at most 100 chunks, so documents can go missing; keep a registry or list ids instead",
             c="The database hides old documents", d="Zero vectors are rejected by every database", correct="B",
             why="If one large PDF fills all 100 matches, other documents never appear. A search is for similarity, not listing."),
    ],
)
