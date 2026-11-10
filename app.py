import ipaddress
import os
import re
import socket
from urllib.parse import urljoin, urlsplit

import streamlit as st
import requests
import feedparser
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS


# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Exam Content AI Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Smart AI Assistant for Exam Content Aggregation")
st.caption(
    "Collect current-affairs content, create concise exam notes, "
    "and practise with AI-generated MCQs."
)


# -----------------------------------------------------------------------------
# API key and source inputs
# -----------------------------------------------------------------------------
api_key = st.sidebar.text_input(
    "Google Gemini API key",
    type="password",
    value=os.getenv("GOOGLE_API_KEY", ""),
)
api_key = api_key.strip()
st.sidebar.caption(
    "Your key is used for this session. Do not share it or upload it to a public repository."
)

DEFAULT_RSS = "https://feeds.bbci.co.uk/news/world/rss.xml"
rss_url = st.text_input("News RSS feed URL", value=DEFAULT_RSS)
article_urls = st.text_area(
    "Optional: paste article URLs (one per line)",
    height=90,
    placeholder="https://example.com/news-article",
)
extra_text = st.text_area(
    "Optional: paste syllabus, class notes, or article text",
    height=120,
)


# -----------------------------------------------------------------------------
# Text collection and cleaning
# -----------------------------------------------------------------------------
MAX_SOURCE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5


def clean_text(text):
    """Normalize whitespace in scraped or user-provided text."""
    return re.sub(r"\s+", " ", text or "").strip()


def validate_public_url(url):
    """Allow only HTTP(S) URLs whose resolved addresses are publicly routable."""
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP or HTTPS URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")

    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        addresses = socket.getaddrinfo(
            parsed.hostname, port, type=socket.SOCK_STREAM
        )
    except (OSError, ValueError) as exc:
        raise ValueError("The URL host could not be resolved safely.") from exc

    if not addresses or any(
        not ipaddress.ip_address(address[4][0]).is_global
        for address in addresses
    ):
        raise ValueError("URLs resolving to private or reserved addresses are not allowed.")

    return parsed.geturl()


def fetch_url_content(url):
    """Fetch a public URL with bounded redirects, time, and response size."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ExamStudyAssistant/1.0)"
    }
    current_url = url

    for redirect_count in range(MAX_REDIRECTS + 1):
        current_url = validate_public_url(current_url)
        response = requests.get(
            current_url,
            headers=headers,
            timeout=20,
            allow_redirects=False,
            stream=True,
        )
        try:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location or redirect_count == MAX_REDIRECTS:
                    raise ValueError("The source URL redirected too many times.")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    content.extend(chunk)
                    if len(content) > MAX_SOURCE_BYTES:
                        raise ValueError("The source exceeds the 5 MB download limit.")
            return bytes(content)
        finally:
            response.close()

    raise ValueError("The source URL redirected too many times.")


def fetch_article(url):
    """Download an article page and extract its main readable text."""
    content = fetch_url_content(url)
    soup = BeautifulSoup(content, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.body or soup
    title = clean_text(
        soup.title.get_text(" ", strip=True) if soup.title else url
    )

    paragraphs = [
        element.get_text(" ", strip=True)
        for element in main.find_all(["p", "h1", "h2", "h3"])
    ]
    body = clean_text(" ".join(paragraphs))

    if len(body) < 250:
        body = clean_text(main.get_text(" ", strip=True))

    return title, body[:18000]


def collect_sources():
    """Collect documents from RSS, article URLs, and pasted study material."""
    docs = []
    errors = []

    if rss_url.strip():
        try:
            feed_content = fetch_url_content(rss_url.strip())
            feed = feedparser.parse(feed_content)

            if getattr(feed, "bozo", False) and not feed.entries:
                errors.append("Could not read the RSS feed. Check the URL.")

            for entry in feed.entries[:8]:
                link = entry.get("link", "")
                title = clean_text(entry.get("title", "News item"))
                summary_html = entry.get("summary", "")
                summary = clean_text(
                    BeautifulSoup(summary_html, "html.parser").get_text(" ")
                )
                content = summary

                if link:
                    try:
                        scraped_title, scraped_body = fetch_article(link)
                        title = scraped_title or title
                        content = scraped_body or summary
                    except Exception:
                        # Keep the RSS summary when the publisher blocks scraping.
                        pass

                if content:
                    docs.append(
                        Document(
                            page_content=(
                                f"Title: {title}\nSource: {link}\n{content}"
                            ),
                            metadata={
                                "title": title,
                                "source": link or rss_url,
                            },
                        )
                    )

        except Exception as exc:
            errors.append(f"RSS error: {exc}")

    for url in [u.strip() for u in article_urls.splitlines() if u.strip()]:
        try:
            title, body = fetch_article(url)
            if body:
                docs.append(
                    Document(
                        page_content=f"Title: {title}\nSource: {url}\n{body}",
                        metadata={"title": title, "source": url},
                    )
                )
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if extra_text.strip():
        docs.append(
            Document(
                page_content=extra_text.strip(),
                metadata={
                    "title": "User-provided study material",
                    "source": "Pasted text",
                },
            )
        )

    return docs, errors


# -----------------------------------------------------------------------------
# Gemini response handling
# -----------------------------------------------------------------------------
def extract_response_text(content):
    """Extract readable text from string or block-based LangChain responses.

    Gemini/LangChain responses can contain either a plain string or a list of
    content blocks. Converting a list directly with str() can expose Python's
    internal list/dictionary representation in the Streamlit page.
    """
    if isinstance(content, str):
        text = content.strip()

    elif isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                if block.strip():
                    text_parts.append(block.strip())

            elif isinstance(block, dict):
                # Standard LangChain text blocks commonly use the "text" key.
                block_type = block.get("type")
                block_text = block.get("text")

                if block_type in (None, "text") and isinstance(block_text, str):
                    if block_text.strip():
                        text_parts.append(block_text.strip())

            else:
                # Some content block implementations expose a .text attribute.
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str) and block_text.strip():
                    text_parts.append(block_text.strip())

        text = "\n\n".join(text_parts).strip()

    elif isinstance(content, dict):
        block_text = content.get("text")
        text = block_text.strip() if isinstance(block_text, str) else ""

    else:
        text = ""

    # Remove an outer Markdown code fence if the model mistakenly adds one.
    text = re.sub(
        r"^\s*```(?:markdown|md)?\s*\n(.*?)\n```\s*$",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return text.strip()


def extract_stream_chunk_text(content):
    """Extract a streaming chunk without stripping meaningful whitespace."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str):
                    parts.append(block_text)
            else:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    parts.append(block_text)
        return "".join(parts)

    if isinstance(content, dict):
        block_text = content.get("text")
        return block_text if isinstance(block_text, str) else ""

    return ""


@st.cache_resource(show_spinner=False)
def get_gemini_model(api_key, max_output_tokens=900, temperature=0.2):
    """Reuse Gemini with a bounded request timeout and no hidden retry cycle.

    Multiple automatic retries can make a single Streamlit click appear frozen
    for several minutes. A clear timeout lets the user retry deliberately.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=45,
        max_retries=0,
    )


@st.cache_resource(show_spinner=False)
def get_embedding_model(api_key):
    """Reuse the embedding client and enforce a 30-second HTTP timeout."""
    kwargs = {
        "model": "models/gemini-embedding-001",
        "google_api_key": api_key,
    }

    fields = getattr(GoogleGenerativeAIEmbeddings, "model_fields", None)
    if fields is None:
        fields = getattr(GoogleGenerativeAIEmbeddings, "__fields__", {})

    if "client_args" in fields:
        # langchain-google-genai v4+: arguments passed to the underlying HTTPX client.
        kwargs["client_args"] = {"timeout": 30}
    elif "request_options" in fields:
        # Compatibility with releases exposing Google API request options.
        kwargs["request_options"] = {"timeout": 30}
    else:
        raise RuntimeError(
            "Your langchain-google-genai version does not expose a supported "
            "embedding timeout setting. Upgrade it with: "
            "pip install -U langchain-google-genai google-genai"
        )

    return GoogleGenerativeAIEmbeddings(**kwargs)


def is_timeout_error(exc):
    """Recognize common timeout messages from HTTP and Google API clients."""
    message = str(exc).lower()
    return any(term in message for term in (
        "timed out", "timeout", "deadline exceeded", "read operation"
    ))


def retrieve_documents_with_retry(vectorstore, query, k=3):
    """Embed the query once (with an explicit timeout), then search FAISS.

    Avoid automatic retries here: retrying a slow network request can double
    the time the Streamlit page appears blocked. The user can retry after a
    visible error instead.
    """
    embeddings = st.session_state.get("embeddings")
    if embeddings is None:
        embeddings = getattr(vectorstore, "embedding_function", None)
    if embeddings is None:
        raise RuntimeError(
            "The embedding model is not available. Rebuild the knowledge base."
        )

    cache = st.session_state.setdefault("query_embedding_cache", {})
    cache_key = query.strip().casefold()

    if cache_key not in cache:
        cache[cache_key] = embeddings.embed_query(query)

        # Keep the in-session cache small and prevent unbounded growth.
        if len(cache) > 100:
            oldest_key = next(iter(cache))
            if oldest_key != cache_key:
                cache.pop(oldest_key, None)

    return vectorstore.similarity_search_by_vector(cache[cache_key], k=k)


def stream_markdown_response(llm, prompt, placeholder, formatter=None):
    """Stream Gemini text and render as soon as the first chunk arrives."""
    pieces = []
    chunk_count = 0

    for chunk in llm.stream(prompt):
        piece = extract_stream_chunk_text(getattr(chunk, "content", ""))
        if not piece:
            continue

        pieces.append(piece)
        chunk_count += 1

        # Render the first chunk immediately; then batch small updates to keep
        # Streamlit responsive without re-rendering on every single token.
        if chunk_count == 1 or chunk_count % 3 == 0:
            placeholder.markdown("".join(pieces))

    final_text = "".join(pieces).strip()
    if formatter is not None:
        final_text = formatter(final_text)

    placeholder.markdown(final_text or "Gemini returned no readable text.")
    return final_text


def format_mcq_option_lines(markdown_text):
    """Make sure MCQ options A-D render as separate Markdown paragraphs.

    Streamlit's Markdown renderer can collapse a single newline into a space.
    This helper also handles Gemini returning multiple options on one line.
    It only adjusts the MCQ section, leaving revision notes unchanged.
    """
    if not markdown_text:
        return markdown_text

    heading_match = re.search(
        r"(?im)^#{1,6}\s*(?:B\.\s*)?MULTIPLE[- ]CHOICE QUESTIONS\b[^\n]*",
        markdown_text,
    )
    if not heading_match:
        return markdown_text

    notes_section = markdown_text[:heading_match.end()]
    mcq_section = markdown_text[heading_match.end():]
    formatted_lines = []

    for line in mcq_section.splitlines():
        stripped = line.strip()

        if not stripped:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            continue

        # Keep answer and explanation lines intact. In particular, do not treat
        # the letter in "Correct Answer: B." as another option marker.
        if re.match(
            r"^(?:\*\*)?(?:Correct Answer|Explanation)\s*:",
            stripped,
            flags=re.IGNORECASE,
        ):
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            formatted_lines.append(stripped)
            continue

        option_markers = list(
            re.finditer(r"(?<![A-Za-z0-9_])([A-D])[.)]\s+", line)
        )

        if len(option_markers) >= 2:
            # Split inline sequences such as: A. One B. Two C. Three D. Four
            question_or_prefix = line[:option_markers[0].start()].strip()
            if question_or_prefix:
                formatted_lines.append(question_or_prefix)
                formatted_lines.append("")

            for index, marker in enumerate(option_markers):
                option_end = (
                    option_markers[index + 1].start()
                    if index + 1 < len(option_markers)
                    else len(line)
                )
                option_text = line[marker.start():option_end].strip()
                if option_text:
                    formatted_lines.append(option_text)
                    formatted_lines.append("")

        elif re.match(r"^[A-D][.)]\s+", stripped):
            # Separate options that Gemini returned on individual lines with
            # only a single newline; blank lines make Markdown preserve them.
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            formatted_lines.append(stripped)
            formatted_lines.append("")

        else:
            formatted_lines.append(stripped)

    return notes_section + "\n\n" + "\n".join(formatted_lines).strip()


# -----------------------------------------------------------------------------
# Question-answering prompt
# -----------------------------------------------------------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful current-affairs and competitive-exam study assistant.

Answer the user's question using the retrieved source material below.

Important instructions:
- Do NOT require the source material to contain the exact words used in the question.
- When the user asks what is relevant to a government/competitive exam, identify relevant facts by their subject matter (for example: polity, governance, economy, science and technology, environment, geography, history, international relations, reports, awards, or important events).
- You may use your general understanding of competitive-exam syllabi ONLY to judge relevance and organize the supplied facts.
- Do NOT add outside factual claims, dates, figures, names, or events that are not supported by the retrieved source material.
- If the context contains useful related facts, explain them clearly even if it does not explicitly mention an exam.
- If the context is genuinely unrelated or does not contain enough information to answer, say what is missing and suggest a more suitable question or source.
- Distinguish clearly between facts stated in the source and your assessment of why they may be exam-relevant.
- Be concise, specific, and readable. Use headings and bullet points where helpful.

Retrieved source material:
{context}

User question:
{question}

Answer:""",
)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "source_count" not in st.session_state:
    st.session_state.source_count = 0
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "query_embedding_cache" not in st.session_state:
    st.session_state.query_embedding_cache = {}


# -----------------------------------------------------------------------------
# Build the searchable knowledge base
# -----------------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    build = st.button(
        "1. Collect content & build knowledge base",
        type="primary",
        use_container_width=True,
    )

with right:
    clear = st.button("Clear session", use_container_width=True)

if clear:
    st.session_state.vectorstore = None
    st.session_state.source_count = 0
    st.session_state.embeddings = None
    st.session_state.query_embedding_cache = {}
    st.rerun()

if build:
    if not api_key:
        st.error("Enter a Google Gemini API key in the sidebar first.")
    else:
        try:
            with st.spinner(
                "Collecting articles and preparing the searchable knowledge base..."
            ):
                docs, errors = collect_sources()

                if errors:
                    with st.expander("Source warnings"):
                        for error in errors:
                            st.warning(error)

                if not docs:
                    st.error(
                        "No content was collected. Add a working RSS URL, "
                        "article URL, or pasted text."
                    )
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=900,
                        chunk_overlap=150,
                    )
                    chunks = splitter.split_documents(docs)

                    embeddings = get_embedding_model(api_key)

                    st.session_state.vectorstore = FAISS.from_documents(
                        chunks,
                        embeddings,
                    )
                    st.session_state.embeddings = embeddings
                    st.session_state.query_embedding_cache = {}
                    st.session_state.source_count = len(docs)

                    st.success(
                        f"Knowledge base ready: {len(docs)} source(s), "
                        f"{len(chunks)} searchable chunk(s)."
                    )

        except Exception as exc:
            error_message = str(exc)
            if "API_KEY_INVALID" in error_message or "API key not valid" in error_message:
                st.error(
                    "Google rejected this API key. Copy a valid Gemini API key "
                    "from Google AI Studio, make sure Gemini API access is enabled "
                    "for its project, and paste it here without quotes."
                )
            else:
                st.error(f"Could not build the knowledge base: {error_message}")


# -----------------------------------------------------------------------------
# Ask questions from the collected content
# -----------------------------------------------------------------------------
if st.session_state.vectorstore is not None:
    st.divider()
    st.subheader("2. Ask questions from collected content")

    question = st.text_input(
        "Example: Summarize the key facts and explain their possible relevance to competitive exams."
    )

    if st.button("Get answer"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif not api_key:
            st.error("Enter your Gemini API key.")
        else:
            stage = "starting the search"
            stage_message = st.empty()
            answer_placeholder = st.empty()
            try:
                # Stage 1: Embedding request + local FAISS similarity search.
                stage = "embedding your question and searching the knowledge base"
                stage_message.info(
                    "**Step 1 of 2 — Searching your knowledge base.** "
                    "This normally takes a few seconds."
                )
                source_docs = retrieve_documents_with_retry(
                    st.session_state.vectorstore, question, k=3
                )

                if not source_docs:
                    stage_message.empty()
                    st.warning(
                        "No relevant material was found. Try another question "
                        "or add more source content."
                    )
                else:
                    context = "\n\n".join(
                        doc.page_content[:1800] for doc in source_docs
                    )
                    prompt = QA_PROMPT.format(
                        context=context, question=question
                    )
                    llm = get_gemini_model(
                        api_key, max_output_tokens=900, temperature=0.2
                    )

                    # Stage 2: Gemini generation. Show feedback before the
                    # blocking network stream begins, then render each first
                    # token as soon as it arrives.
                    stage = "waiting for Gemini to generate the answer"
                    stage_message.info(
                        "**Step 2 of 2 — Generating your answer with Gemini.** "
                        "The answer will appear below as text arrives. "
                        "The request has a 45-second timeout."
                    )
                    answer_placeholder.caption(
                        "Connected to the generation step; waiting for the first text..."
                    )
                    answer_text = stream_markdown_response(
                        llm, prompt, answer_placeholder
                    )
                    stage_message.empty()

                    if not answer_text:
                        st.warning(
                            "Gemini completed the request but returned no readable text. "
                            "Please try a shorter question."
                        )

                    with st.expander("Sources used"):
                        for doc in source_docs:
                            st.write(doc.metadata.get("title", "Source"))
                            st.caption(doc.metadata.get("source", ""))

            except Exception as exc:
                stage_message.empty()
                answer_placeholder.empty()
                if is_timeout_error(exc):
                    st.error(
                        f"The request timed out while {stage}. "
                        "The app now stops waiting after a bounded timeout instead "
                        "of retrying silently. Check your internet connection, "
                        "Gemini API quota/status, then click Get answer again."
                    )
                else:
                    st.error(
                        f"Could not generate an answer while {stage}: {exc}"
                    )


# -----------------------------------------------------------------------------
# Generate revision notes and MCQs
# -----------------------------------------------------------------------------
if st.session_state.vectorstore is not None:
    st.divider()
    st.subheader("3. Generate revision notes and practice MCQs")

    topic = st.text_input(
        "Topic for revision material",
        placeholder="e.g. International relations, science and technology",
    )
    count = st.slider("Number of MCQs", min_value=3, max_value=10, value=5)

    if st.button("Generate notes + MCQs", type="primary") and topic.strip():
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar.")
        else:
            try:
                with st.spinner(
                    "Retrieving relevant content and generating study material..."
                ):
                    # Four chunks provide useful context without sending an
                    # unnecessarily large prompt to Gemini.
                    relevant_docs = retrieve_documents_with_retry(
                        st.session_state.vectorstore, topic, k=4
                    )

                    if not relevant_docs:
                        st.warning(
                            "No relevant material was found for this topic. "
                            "Try another topic or add more source material."
                        )
                    else:
                        context = "\n\n".join(
                            doc.page_content for doc in relevant_docs
                        )

                        llm = get_gemini_model(
                            api_key, max_output_tokens=2000, temperature=0.2
                        )

                        prompt = f"""
You are an expert competitive-exam preparation tutor.

Use ONLY the information provided in the CONTEXT below. Do not invent facts,
answers, dates, names, statistics, or explanations. Do not use outside knowledge.
If the context does not contain enough information for the requested topic,
clearly say that the source material is insufficient.

TOPIC: {topic}

Create the following two sections.

## A. REVISION NOTES

- Use clear headings and subheadings.
- Use concise bullet points suitable for competitive examinations.
- Highlight important terms, names, dates, definitions, and concepts using
  Markdown bold where appropriate.
- Include only facts supported by the supplied context.
- Avoid repetitive introductions and filler text.

## B. MULTIPLE-CHOICE QUESTIONS

Generate EXACTLY {count} different MCQs based only on the supplied context.
Each question must have exactly four options and one unambiguously correct answer.

Use this exact Markdown structure for every question:

### Question 1. [Write the question here]

A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]

**Correct Answer:** B. [Correct option text]

**Explanation:** [One concise sentence explaining why this answer is correct.]

Continue numbering sequentially up to Question {count}.

FORMATTING RULES:
1. Return readable Markdown, not JSON, Python, XML, or a Python list/dictionary.
2. Do not put the complete answer inside a code block.
3. Keep a blank line between each question, its options, answer, and explanation.
4. Do not add options E or F.
5. Do not repeat questions or options.
6. Do not provide an answer that is unsupported by the context.
7. Put options A, B, C, and D on four separate lines; never place two options on the same line.

CONTEXT:
{context}
"""

                        output_placeholder = st.empty()
                        generated_text = stream_markdown_response(
                            llm,
                            prompt,
                            output_placeholder,
                            formatter=format_mcq_option_lines,
                        )

                        if not generated_text:
                            st.error(
                                "Gemini returned no readable text. "
                                "Please try again or check the model/API response."
                            )

                        with st.expander("Source material used"):
                            for doc in relevant_docs:
                                st.write(doc.metadata.get("title", "Source"))
                                st.caption(doc.metadata.get("source", ""))

                        st.caption(
                            "Generated from retrieved material. Verify important "
                            "facts with official sources before using them for exam preparation."
                        )

            except Exception as exc:
                if is_timeout_error(exc):
                    st.error(
                        "The request timed out while retrieving material or generating notes. "
                        "The embedding and Gemini requests use bounded timeouts. "
                        "Please check your connection and try again."
                    )
                else:
                    st.error(f"Could not generate notes and MCQs: {exc}")


# -----------------------------------------------------------------------------
# Sidebar help
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Demo flow**")
st.sidebar.markdown(
    "1. Add an RSS feed, article URL, or study text.\n"
    "2. Build the knowledge base.\n"
    "3. Ask a question (answers stream onto the page).\n"
    "4. Generate revision notes and MCQs (also streamed)."
)
st.sidebar.caption(
    "Use only sources you are allowed to access. Some websites block automated scraping."
)
