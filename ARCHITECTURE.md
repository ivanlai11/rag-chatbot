```markdown
# ARCHITECTURE.md

## AI & Technology News RAG Chatbot

This document explains the architecture, design choices, and data flow for the AI & Technology News RAG Chatbot.

---

## 1. System Overview

This project implements a domain-specific Retrieval-Augmented Generation (RAG) chatbot for **AI and technology news**.

The system is designed to:

- collect and process news articles
- preserve useful metadata for citation and traceability
- chunk documents into retrieval-friendly units
- embed and store chunks in a persistent vector database
- retrieve relevant chunks for user questions
- support follow-up questions with conversation memory
- improve recall with multi-query retrieval
- generate grounded answers with explicit source citations

The architecture combines local retrieval with local generation:

- **Embeddings:** Hugging Face Sentence Transformers
- **Vector Store:** Chroma
- **Generation:** Ollama
- **UI:** Streamlit
- **Framework:** LangChain-compatible document handling

---

## 2. End-to-End Data Flow

The full pipeline can be summarized as:

1. **News collection**
2. **Document cleaning and metadata normalization**
3. **Document-to-LangChain conversion**
4. **Recursive chunking**
5. **Embedding generation**
6. **Persistent Chroma vector database creation**
7. **Query rewriting for follow-up questions**
8. **Multi-query expansion**
9. **Top-K retrieval and result merging**
10. **Grounded answer generation with citations**
11. **Testing and evaluation logging**

---

## 3. Ingestion Layer

### 3.1 Data Sources

The chatbot operates on a curated corpus of AI and technology news articles collected from sources such as:

- BBC News
- TechCrunch
- WIRED
- The Verge
- other selected AI / technology articles used during experimentation

### 3.2 Collection Script

The ingestion pipeline begins with:

- `scripts/collect_news.py`

This script is responsible for:

- reading from configured news feeds or source pages
- extracting article content
- storing articles in a structured corpus
- preserving article-level metadata

### 3.3 Why Ingestion Matters

The quality of a RAG system depends heavily on document quality.  
For a news-based chatbot, ingestion must balance:

- enough corpus coverage
- consistent article formatting
- reliable metadata
- minimal noise from ads, boilerplate, and malformed pages

---

## 4. Document Processing and Metadata Extraction

### 4.1 Cleaning Goals

After collection, article text is cleaned to remove common noise such as:

- excessive whitespace
- repeated line breaks
- malformed characters
- empty or extremely short text
- simple boilerplate fragments

### 4.2 Metadata Fields

Each processed article preserves important metadata for retrieval and citation:

- `doc_id`
- `title`
- `source_name`
- `published_date`
- `url`
- `domain_tag`
- `extraction_method`
- `text_length`

### 4.3 Validation

The processing pipeline checks that important fields are present and usable, especially:

- title
- URL
- cleaned article text

This is important because missing metadata makes citation rendering unreliable and weakens traceability in final answers.

### 4.4 LangChain Document Conversion

After cleaning and validation, each article is converted into a LangChain-compatible `Document`:

- `page_content` = cleaned article text
- `metadata` = citation-friendly article metadata

This conversion makes the downstream chunking and vector indexing stages easier to manage.

---

## 5. Chunking Strategy

### 5.1 Selected Strategy

The system uses:

- **RecursiveCharacterTextSplitter**
- `chunk_size = 1000`
- `chunk_overlap = 200`

### 5.2 Why This Strategy Was Chosen

This project uses recursive chunking because it provides a practical balance between:

- semantic coherence
- implementation simplicity
- chunk size control
- retrieval effectiveness

The recursive splitter attempts to preserve natural boundaries where possible, instead of cutting text completely arbitrarily.

### 5.3 Rationale

This design was chosen for three main reasons:

1. **Preserve local semantic coherence**  
   Chunks should contain enough surrounding context to remain interpretable.

2. **Improve retrieval granularity**  
   Smaller chunks help retrieval return focused evidence rather than entire articles.

3. **Reduce boundary information loss**  
   The 200-character overlap helps preserve continuity across chunk boundaries.

### 5.4 Chunk Metadata

Each chunk inherits document-level metadata and adds chunk-specific fields such as:

- `parent_doc_id`
- `chunk_id`
- `chunk_char_length`

This makes the system citation-friendly and allows retrieved results to map back to the original document.

---

## 6. Embedding Layer

### 6.1 Embedding Model

The project uses local Hugging Face embeddings:

- `sentence-transformers/all-mpnet-base-v2`

### 6.2 Why This Embedding Model

This model was chosen because it provides strong semantic retrieval performance while remaining practical for local use.

Benefits include:

- reliable sentence/document embeddings
- local inference without paid API dependency
- compatibility with Chroma
- good balance between retrieval quality and implementation simplicity

### 6.3 Embedding Scope

Embeddings are created at the **chunk level**, not the full-document level.

This improves retrieval precision because the system compares user queries against smaller, more focused semantic units.

---

## 7. Vector Database

### 7.1 Vector Store Choice

The system uses:

- **Chroma**

### 7.2 Why Chroma

Chroma was selected because it satisfies the assignment requirements:

- vector database support
- metadata storage
- persistent local storage
- easy integration with LangChain-style workflows

### 7.3 Persistent Storage

The vector database is stored locally in:

- `./chroma_db`

This allows the system to:

- reload an existing knowledge base without recomputing everything
- support reproducible testing
- make app startup more practical after indexing has already been done

### 7.4 Rebuild Support

A rebuild workflow is included so the knowledge base can be regenerated cleanly when the corpus changes.

This is important because it prevents stale vector data from mixing with newer processed documents.

---

## 8. Retrieval Pipeline

### 8.1 Standard Retrieval

The app retrieves:

- **Top-K = 4** final chunks

This falls within the assignment requirement of retrieving 3 to 5 chunks.

### 8.2 Query Rewriting for Follow-Up Questions

To support multi-turn conversation, the system rewrites follow-up questions into standalone retrieval queries.

For example:

- “Which one focuses more on product launches?”
- “What about the article from BBC?”
- “Summarize that one.”

These are often ambiguous without prior context.  
The rewrite step uses recent chat history so retrieval is based on a more explicit question.

### 8.3 Multi-Query Retrieval

The project implements **Multi-Query Retrieval** as its advanced feature.

Instead of retrieving from a single query only, the system:

1. generates a standalone retrieval query
2. creates several alternative query variants
3. retrieves results for each variant
4. merges and ranks the combined results

### 8.4 Why Multi-Query Was Chosen

A user question may be phrased in only one way.  
Important documents may still be relevant even when they use slightly different terminology.

Multi-query retrieval helps by:

- improving recall
- reducing missed context
- retrieving results from multiple semantic angles
- making retrieval more robust for entity-heavy and theme-based questions

### 8.5 Merge and Ranking Logic

The system merges retrieved results across multiple query variants and uses a simple ranking logic based on:

- whether a chunk appears in multiple query result sets
- how highly it ranked in each set

This helps prioritize chunks that are consistently relevant across different formulations of the same information need.

---

## 9. Conversation Memory

### 9.1 Memory Scope

The app keeps the most recent:

- **10 messages**
- approximately **5 exchanges**

### 9.2 Why This Memory Window

The memory window is intentionally limited to keep the system:

- responsive
- easy to reason about
- less likely to overfit to distant conversation context

### 9.3 Memory Use

Conversation memory is primarily used for:

- rewriting follow-up questions into standalone retrieval queries
- maintaining continuity across user turns
- supporting natural conversational interaction

This design keeps memory useful without requiring long-context storage or heavy conversation state management.

---

## 10. Generation Layer

### 10.1 Generation Model

The final grounded answer is generated through:

- **Ollama**

Typical tested models include:

- `gemma3:12b`
- other local Ollama-compatible instruction/chat models

### 10.2 Why Ollama

Ollama was chosen for answer generation because it provides:

- local inference
- straightforward model switching
- better local chat-style generation than smaller lightweight pipeline models
- improved control over answer generation without external API cost

### 10.3 Prompt Design

The answer generation prompt is designed to enforce grounding.

It includes:

- recent conversation history
- original user question
- standalone retrieval query
- multi-query variants
- retrieved chunk context
- instructions to avoid outside knowledge
- instructions to say “I do not know” when context is insufficient
- a requirement to include a short sources section

### 10.4 Grounding Goal

The prompt is structured to reduce hallucination by forcing the model to rely only on retrieved context.

This is especially important in a news domain, where users may ask about current events, company behavior, or article-specific claims.

---

## 11. Citation Design

### 11.1 Citation Metadata

Each retrieved chunk is displayed with metadata such as:

- title
- source name
- published date
- URL
- chunk ID
- snippet

### 11.2 Why Citations Matter

Citations are critical because they:

- increase transparency
- help verify whether the answer is grounded
- make debugging easier
- provide evidence to support the generated response

### 11.3 UI Rendering

The chatbot displays citations in expandable source panels so that users can inspect:

- where information came from
- which chunks were retrieved
- whether the cited snippet matches the answer

This is especially useful for evaluation and grading.

---

## 12. Streamlit UI Layer

### 12.1 Main Interface

The app uses Streamlit with a conversational interface built around:

- `st.chat_input`
- `st.chat_message`

The main page presents:

- user questions
- assistant answers
- standalone retrieval query
- multi-query variants
- retrieved source citations

### 12.2 Sidebar Features

The sidebar includes:

- domain description
- generation model selection
- Ollama configuration
- memory settings
- index status
- corpus statistics
- retrieval settings
- demo question bank
- rebuild index button
- clear chat button

### 12.3 Why This UI Design

The UI was designed not only for end-user interaction, but also for debugging, testing, and demonstration.

This supports:

- assignment demo needs
- retrieval transparency
- evaluation workflow
- easier debugging during development

---

## 13. Testing and Evaluation Layer

### 13.1 Evaluation Panel

The app includes an evaluation interface for manual quality validation.

It allows the tester to label each response on dimensions such as:

- retrieval reasonableness
- answer grounding
- citation alignment
- honest handling of insufficient context

### 13.2 Test Log

The app can export test results as CSV, making it easier to:

- document testing outcomes
- support write-up claims
- preserve a record of manual validation

### 13.3 Raw Retrieval Inspection

The system also exposes raw retrieved chunk data for the latest response, which helps inspect:

- retrieved titles
- source names
- chunk IDs
- snippets
- matched multi-query variants

This is useful when validating whether the system is retrieving the right evidence.

---

## 14. Why This Architecture Fits the Assignment

This architecture directly addresses the assignment requirements:

### Domain Selection
- Domain-specific corpus: AI and technology news

### Minimum Document Collection
- corpus built from multiple news sources

### Document Processing
- article extraction
- cleaning
- metadata preservation

### Chunking Strategy
- recursive chunking with overlap

### Vector Database
- Chroma with persistent storage

### RAG Implementation
- embeddings
- top-K retrieval
- grounded answer generation
- citations

### Conversation Memory
- recent history retention
- follow-up query rewriting

### Advanced Feature
- multi-query retrieval

### UI
- conversational interface
- transparent citation display
- testing tools

---

## 15. Current Limitations

Despite the functional architecture, several limitations remain:

- retrieval quality depends on corpus quality and relevance
- article extraction can vary across websites
- citation usefulness depends on chunk cleanliness
- local generation quality depends heavily on the selected Ollama model
- long or noisy context can still hurt answer quality
- public deployment with fully local generation may require environment-specific adjustments

---

## 16. Future Improvements

Potential future improvements include:

- better reranking after multi-query retrieval
- hybrid search combining semantic and keyword retrieval
- improved deduplication of similar articles
- source-level diversity control in retrieval
- stronger automatic evaluation
- cleaner answer formatting
- deployment optimization for public hosting

---

## 17. Summary

This chatbot uses a practical, production-style RAG architecture tailored to AI and technology news.

Its main design strengths are:

- persistent vector retrieval
- metadata-aware citations
- multi-turn question handling
- multi-query retrieval
- local answer generation
- built-in testing support

Together, these components create a domain-specific chatbot that is transparent, inspectable, and aligned with the technical goals of the assignment.
```