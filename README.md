# Smart AI Assistant for Exam Content Aggregation

A student-friendly prototype for the AVIT STAR case study. It collects articles from an RSS feed
and user-provided URLs/text, indexes the content with LangChain + FAISS, answers questions using
retrieval-augmented generation (RAG), and generates revision notes and MCQs with Google Gemini.

## Requirements
- Python 3.10 or newer
- Internet connection
- Google Gemini API key

## Setup (Windows)
1. Extract this ZIP file.
2. Open Command Prompt in the extracted folder.
3. Create and activate a virtual environment:

   ```bat
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install packages:

   ```bat
   pip install -r requirements.txt
   ```

5. Run the app:

   ```bat
   streamlit run app.py
   ```

6. The browser opens the app. Paste your Gemini API key in the left sidebar.
7. Keep the RSS URL, add article URLs or paste text, then click **Collect content & build knowledge base**.
8. Ask a question or generate revision notes + MCQs.

## What to demonstrate
- **Input:** RSS feed / article URLs / pasted study material.
- **Processing:** scrape and clean text → split into chunks → create embeddings → store in FAISS.
- **RAG:** retrieve relevant chunks for the user's question → Gemini generates a grounded answer.
- **Output:** concise revision notes and MCQs with answers and explanations.

## Important notes
- This is a working prototype, not a production service.
- The default RSS feed is a BBC World News feed; replace it with an appropriate source for your exam.
- Website scraping depends on each site's terms, robots rules, and technical restrictions. RSS feeds and
  pasted content are more reliable for a classroom demo.
- Do not commit API keys to GitHub or share them in screenshots.
- AI-generated content can contain errors; verify facts against official exam sources.
