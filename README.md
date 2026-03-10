# AI & Technology News RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot built for AI and technology news.  
The system collects and processes news articles, chunks them into citation-friendly units, stores them in a persistent Chroma vector database, and answers user questions with grounded responses, source citations, conversation memory, and multi-query retrieval.

---

## Project Overview

This project builds a domain-specific RAG chatbot for **AI and technology news**.

The chatbot is designed to:

- answer questions based on an indexed news corpus
- retrieve relevant document chunks from a vector database
- support follow-up questions with conversation memory
- improve retrieval coverage through multi-query expansion
- display source citations for transparency and verification

The domain was chosen because AI and technology news contains rapidly evolving topics, frequent entity references, and many cross-article themes, making it a good fit for RAG-based question answering.

---

## Domain

**Domain:** AI and Technology News

**Corpus sources include:**

- BBC News
- TechCrunch
- WIRED
- The Verge
- other collected AI / technology articles used during corpus building

The corpus is processed into structured documents with metadata such as title, source, publication date, URL, and document ID.

---

## Key Features

### 1. Domain-Specific RAG
The chatbot answers questions only from the indexed AI and technology news corpus rather than relying on unrestricted open-domain generation.

### 2. Persistent Vector Database
Document chunks are embedded and stored in a **persistent Chroma vector database**, allowing the app to reload the knowledge base without recomputing embeddings every time.

### 3. Metadata-Aware Citations
Each retrieved chunk keeps metadata including:

- title
- source name
- publication date
- URL
- chunk ID

This allows the app to display source citations alongside generated answers.

### 4. Conversation Memory
The chatbot retains the most recent conversation turns and rewrites follow-up questions into standalone retrieval queries so that multi-turn interaction works more naturally.

### 5. Multi-Query Retrieval
Instead of retrieving from only one user query, the system expands the retrieval query into multiple variants, merges the retrieved results, and improves recall.

### 6. Testing and Evaluation Support
The app includes:

- demo question bank
- evaluation panel
- raw retrieved chunk inspection
- downloadable test log CSV

These features help validate retrieval quality, answer grounding, citation alignment, and hallucination behavior.

---

## Architecture Summary

The system follows a standard RAG workflow:

1. **Data Collection**
   - news articles are collected from feeds / source pages
   - articles are saved into a structured corpus

2. **Document Processing**
   - text cleaning
   - metadata validation
   - conversion into LangChain `Document` objects

3. **Chunking**
   - recursive chunking with overlap
   - chunk metadata preserved for citations

4. **Embeddings**
   - local Hugging Face embeddings using `sentence-transformers/all-mpnet-base-v2`

5. **Vector Database**
   - chunk embeddings stored in persistent **Chroma**

6. **Retrieval**
   - top-K retrieval
   - multi-query expansion
   - merged ranking across query variants

7. **Generation**
   - final grounded answer generated locally through **Ollama**

8. **Conversation Memory**
   - recent message history retained
   - follow-up question rewriting into standalone retrieval queries

9. **Citations**
   - retrieved chunk metadata shown in the UI

10. **Testing Interface**
   - evaluation tools for manual validation

---

## Repository Structure

```text
rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── TECHNICAL_WRITEUP.md
├── demo/
├── data/
│   ├── raw_articles/
│   ├── processed/
│   └── corpus.csv
├── scripts/
│   ├── collect_news.py
│   ├── build_index.py
│   └── utils.py
├── chroma_db/
└── screenshots/
```

---

## Tech Stack

* **Frontend / UI:** Streamlit
* **Embeddings:** Hugging Face Sentence Transformers
* **Vector Database:** Chroma
* **LLM Generation:** Ollama
* **Document Framework:** LangChain
* **Data Processing:** pandas
* **Article Extraction:** trafilatura / related parsing utilities

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd rag-chatbot
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install Ollama

Download and install Ollama on your machine.

Then pull a local generation model, for example:

```bash
ollama pull gemma3:12b
```

Other Ollama models can also be used, but `gemma3:12b` was selected for stronger grounded answer quality compared with smaller local models.

---

## How to Build the Knowledge Base

### Step 1. Collect articles

```bash
python scripts/collect_news.py
```

This creates the raw processed corpus used for indexing.

### Step 2. Process, chunk, and build the vector index

```bash
python scripts/build_index.py --rebuild
```

This pipeline performs:

* document cleaning
* metadata validation
* domain filtering
* recursive chunking
* embedding creation
* Chroma vector database rebuild

---

## How to Run Locally

```bash
python -m streamlit run app.py
```

Then open the local app in your browser, usually at:

```text
http://localhost:8501
```

---

## Application Workflow

Once the app is running:

1. enter or confirm the Ollama generation model
2. ask a question about the indexed news corpus
3. the system rewrites follow-up questions if needed
4. the system generates multiple retrieval query variants
5. relevant chunks are retrieved from Chroma
6. the model generates a grounded answer using retrieved context only
7. citations are displayed in expandable source panels

---

## Retrieval and Memory Design

### Top-K Retrieval

The app uses **Top-K = 4** final retrieved chunks.

### Multi-Query Retrieval

A single user question is expanded into several retrieval-friendly variants.
The retrieved results from these variants are merged and ranked, which helps reduce missed context.

### Conversation Memory

The chatbot retains the most recent messages and uses them to rewrite follow-up questions into standalone retrieval queries.

This improves interaction for questions like:

* “Which one focuses more on product launches?”
* “What about the article from BBC?”
* “Summarize that one.”

---

## Example Questions

### Factual

* Which companies launched new AI products?
* What company is planning an open-source AI agent platform?
* What concerns are mentioned about AI privacy or surveillance?

### Synthesis

* What are the major AI trends in the indexed news corpus?
* What are the common themes across these articles?

### Comparison

* How do two articles differ in their framing of AI regulation?
* How do different companies position AI products differently?

### Follow-Up / Multi-Turn

* Which of those focuses on startups?
* What about the article from BBC?
* Summarize that one.
* What is the main concern mentioned there?

---

## Testing and Evaluation

The app includes built-in tools for Phase 10 testing:

* **Demo question bank**
* **Evaluation panel**
* **Raw retrieved chunk inspection**
* **Downloadable test log CSV**

Evaluation criteria include:

* retrieval reasonableness
* answer grounding
* citation alignment
* honest handling of insufficient context

---

## Screenshots

Add project screenshots in the `screenshots/` folder and reference them here.

Suggested screenshots:

1. main chat interface
2. sidebar settings and model configuration
3. standalone retrieval query display
4. multi-query variants display
5. source citations expander
6. evaluation panel
7. test log preview

Example placeholder section:

```markdown
![Main UI](screenshots/main_ui.png)
![Citations](screenshots/citations.png)
![Evaluation Panel](screenshots/evaluation_panel.png)
```

---

## Deployment

**Public URL:** `TBD`

This project is designed to be deployed as a Streamlit application.
Before deployment, make sure:

* `requirements.txt` is complete
* no local absolute paths are hardcoded
* the app entry point is `app.py`
* Ollama or the deployment environment is configured appropriately
* the vector database handling strategy is clear

---

## Known Limitations

* local model quality depends on the selected Ollama model
* smaller local models may produce weaker summaries or noisier answers
* article extraction quality may vary by source page structure
* retrieval quality depends on corpus coverage and document relevance
* public deployment with fully local model behavior may require environment-specific adjustments

---

## Future Improvements

* stronger reranking or hybrid retrieval
* better source filtering and deduplication
* cleaner answer formatting rules
* automated retrieval evaluation
* public deployment optimization
* improved citation rendering

---

## Files of Interest

* `app.py`
  Main Streamlit application.

* `scripts/collect_news.py`
  Builds the news corpus from source feeds / articles.

* `scripts/build_index.py`
  Handles processing, chunking, embeddings, and Chroma index creation.

* `scripts/utils.py`
  Shared utilities for document processing and pipeline support.

* `ARCHITECTURE.md`
  Detailed system architecture and design decisions.

* `TECHNICAL_WRITEUP.md`
  Two-page technical write-up covering architecture, challenges, and results.