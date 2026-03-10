# TECHNICAL_WRITEUP.md

## AI & Technology News RAG Chatbot

### 1. Project Overview

This project builds a domain-specific Retrieval-Augmented Generation (RAG) chatbot for **AI and technology news**. The goal was to create a production-ready system that can answer user questions from a curated news corpus using vector retrieval, conversation memory, source citations, and an advanced retrieval feature.

The selected domain is well suited for a RAG system because AI and technology news changes rapidly, includes many named entities such as companies and products, and often requires synthesizing information across multiple articles. A general-purpose chatbot may answer such questions fluently, but without grounding it may hallucinate or fail to cite evidence. This project addresses that problem by restricting answer generation to retrieved context from the indexed corpus.

The system supports both direct factual questions and more open-ended synthesis questions. It also handles follow-up questions by preserving recent conversation history and reformulating ambiguous user queries into standalone retrieval queries. In addition, it implements **multi-query retrieval** to improve recall by retrieving from multiple query variations rather than a single phrasing.

Overall, the project aims to balance three goals: accuracy, transparency, and usability.

---

### 2. Architecture

The system follows a standard RAG workflow with several enhancements.

#### 2.1 Data Collection

The corpus was built from AI and technology news articles collected from sources such as BBC News, TechCrunch, WIRED, and The Verge. A collection script gathers the raw article content and stores it in a structured corpus for downstream processing.

The news domain required collecting enough documents to support retrieval across multiple subtopics such as AI products, regulation, privacy, robotics, hardware, and startups.

#### 2.2 Document Processing

After collection, the corpus is cleaned and validated. The processing stage removes obvious noise such as extra whitespace, repeated line breaks, malformed characters, and empty or extremely short text. It also validates that important metadata fields are present.

Each article preserves metadata including:

- document ID
- title
- source name
- publication date
- URL
- extraction method
- text length

This metadata is important because the chatbot must display meaningful citations alongside answers.

Each cleaned article is then converted into a LangChain-compatible `Document` object, with the cleaned article body stored in `page_content` and the article metadata stored in `metadata`.

#### 2.3 Chunking

The project uses **recursive chunking** with the following parameters:

- chunk size: 1000 characters
- chunk overlap: 200 characters

This chunking strategy was chosen because it provides a good tradeoff between semantic coherence and retrieval precision. Larger chunks can dilute relevance by mixing too many topics, while smaller chunks may remove too much context. The overlap helps preserve continuity across chunk boundaries and reduces information loss when key ideas span multiple segments.

Each chunk inherits document-level metadata and adds chunk-specific fields such as `chunk_id` and `chunk_char_length`.

#### 2.4 Embeddings

Chunk embeddings are generated locally using the Hugging Face sentence-transformer model:

- `sentence-transformers/all-mpnet-base-v2`

This model was selected because it offers strong semantic retrieval performance while remaining practical for local use. By computing embeddings at the chunk level, the retriever can compare user questions against focused semantic units instead of full articles.

#### 2.5 Vector Database

The embedded chunks are stored in a persistent **Chroma** vector database. Chroma was chosen because it supports vector search, metadata storage, and persistent local indexing with straightforward integration into the RAG workflow.

Persistent storage allows the app to reload an existing knowledge base without recomputing embeddings every time. This makes the workflow more practical during development, testing, and demonstration.

#### 2.6 Retrieval Top-K

The chatbot retrieves **Top-K = 4** final chunks, which fits the assignment requirement of retrieving 3 to 5 relevant pieces of evidence.

The retrieval layer does not rely only on the raw user query. Instead, it incorporates both conversation-aware rewriting and multi-query expansion.

#### 2.7 Conversation Memory

The system retains the most recent **10 messages**, which corresponds to approximately **5 exchanges**. This limited memory window was chosen to preserve conversational continuity without overly increasing prompt size.

Recent history is mainly used to rewrite follow-up questions into standalone retrieval queries. This is important for questions such as:

- “Which one focuses more on product launches?”
- “What about the article from BBC?”
- “Summarize that one.”

Without history-aware rewriting, such questions would be too ambiguous for reliable retrieval.

#### 2.8 Multi-Query Retrieval

The advanced feature implemented in this project is **multi-query retrieval**.

A user question may be phrased in only one way, but relevant evidence may use different wording. To address this, the system generates several query variants from the standalone retrieval query. Each query variant retrieves chunks independently, and the results are merged and ranked.

This improves retrieval recall by reducing the chance that relevant evidence is missed simply because the original wording did not match the most semantically similar representation in the vector space.

#### 2.9 Generation

For final answer generation, the project uses **Ollama** with a local chat/instruction model. This design avoids dependence on paid API usage for answer generation and makes the system more practical for local experimentation.

The answer generation prompt includes:

- recent conversation history
- the original user question
- the standalone retrieval query
- multi-query variants
- retrieved chunk context
- instructions to stay grounded in retrieved evidence only
- a requirement to admit uncertainty when context is insufficient
- a short sources section at the end

This design aims to reduce hallucination and make the answer traceable to the underlying sources.

#### 2.10 UI and Deployment Readiness

The app is built with Streamlit and includes:

- chat-based interaction
- source citation display
- clear chat control
- index rebuild button
- corpus statistics
- evaluation panel
- demo question bank

This makes the system not only usable as a chatbot, but also inspectable during testing and grading.

---

### 3. Challenges

Several practical challenges emerged during implementation.

#### 3.1 Article Extraction Inconsistency

Different news websites format article pages differently. Some pages contain clean article bodies, while others include navigation text, promotional material, or repeated boilerplate. This required additional text cleaning and metadata handling to make the corpus usable.

#### 3.2 Short or Noisy Content

Some collected items were too short or too noisy to be useful. Short summaries often performed poorly in retrieval and could degrade answer quality if indexed directly. To address this, the pipeline filters out empty or overly short articles and applies domain filtering so that clearly irrelevant content is removed.

#### 3.3 Metadata Formatting

Accurate citations depend on metadata consistency. In practice, publication dates, titles, or URLs sometimes required normalization. Missing or malformed metadata made it harder to display trustworthy citations, so the pipeline had to validate and normalize key fields before indexing.

#### 3.4 Citation Quality

Even when retrieval worked, citation usefulness depended on chunk quality. If a retrieved chunk contained too much noise or duplicated text, the citation was technically correct but less useful for human inspection. Preserving the right metadata and displaying chunk snippets helped improve transparency.

#### 3.5 Follow-Up Handling

Follow-up questions were a major challenge because user language in multi-turn conversation is often underspecified. Query rewriting based on recent conversation history significantly improved retrieval for follow-up questions, but it also introduced a dependency on the quality of the rewriting step.

#### 3.6 Multi-Query Noise

Multi-query retrieval improved recall, but local generation models sometimes produced noisy query variants, such as labels, repeated lines, or invented years. Lightweight post-processing was added to clean these variants before retrieval.

#### 3.7 Local Generation Tradeoffs

Local answer generation avoids API cost and external quota issues, but it introduces tradeoffs in speed and model quality. Smaller local models were faster but produced lower-quality grounded answers, while stronger local models improved answer quality at the cost of more compute.

#### 3.8 Deployment Considerations

The project was designed to be deployment-ready, but deployment with fully local generation introduces additional environment-specific considerations. Local model dependencies and persistent vector storage require careful handling when moving from development to public hosting.

---

### 4. Results

The final system successfully supports the main assignment goals.

#### 4.1 Successful Query Types

The chatbot can answer multiple categories of questions, including:

- factual questions  
  Example: “Which companies launched new AI products?”

- synthesis questions  
  Example: “What are the common themes across these articles?”

- comparison questions  
  Example: “How do two articles differ in their framing of AI regulation?”

- follow-up questions  
  Example: “Which of those focuses on startups?”  
  Example: “What about the article from BBC?”  
  Example: “Summarize that one.”

This demonstrates that the pipeline is not limited to one-shot lookup, but can also support multi-turn retrieval and answer generation.

#### 4.2 Follow-Up Question Handling

Conversation memory and query rewriting allow the chatbot to interpret follow-up questions more naturally. This is one of the most important improvements over a single-turn semantic search system. Instead of treating each new question in isolation, the chatbot uses recent context to construct better retrieval queries.

#### 4.3 Citation Display

The app displays source citations for retrieved chunks, including title, source name, publication date, URL, chunk ID, and snippet. This makes the system more transparent and supports manual validation of answer grounding.

#### 4.4 Evaluation Support

The inclusion of an evaluation panel, raw retrieval inspection, and downloadable test logs makes it easier to validate:

- whether the retrieved chunks are reasonable
- whether the answer is grounded
- whether citations align with the generated response
- whether the bot admits uncertainty when context is insufficient

These features improve the usability of the system for grading, debugging, and demonstration.

---

### 5. Limitations

Although the system is functional and aligns well with the assignment requirements, several limitations remain.

First, answer quality still depends strongly on the selected local generation model. Better local models improve performance, but also require more computational resources.

Second, retrieval quality is limited by corpus quality. If the corpus lacks relevant documents or contains noisy articles, retrieval performance suffers even if the pipeline itself is correct.

Third, citation quality depends on chunk cleanliness. A correct retrieved chunk may still be difficult to interpret if the underlying article text is noisy.

Fourth, the system currently relies on relatively simple merged ranking for multi-query retrieval. More advanced reranking or hybrid retrieval could improve precision further.

---

### 6. Conclusion

This project successfully implements a production-style RAG chatbot for AI and technology news with:

- persistent vector storage
- semantic retrieval
- grounded answer generation
- metadata-aware citations
- conversation memory
- multi-query retrieval
- built-in testing and evaluation support

The final system demonstrates how a domain-specific chatbot can be made more accurate and transparent by grounding generation in retrieved evidence. It also shows the practical challenges involved in building a local, end-to-end RAG system, including extraction noise, metadata consistency, retrieval quality, and local generation tradeoffs.

Overall, the project meets the technical objectives of the assignment while also providing a strong foundation for future improvement.