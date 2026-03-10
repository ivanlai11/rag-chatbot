import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.documents import Document


# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="RAG Domain Expert Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI & Technology News RAG Chatbot")
st.markdown(
    "Ask questions about the indexed AI and technology news corpus. "
    "The system retrieves relevant chunks from Chroma and generates grounded answers with citations."
)


# -------------------------
# Constants
# -------------------------
PERSIST_DIR = "chroma_db"
INDEX_REPORT_PATH = "data/processed/index_report.json"
BUILD_INDEX_SCRIPT = os.path.join("scripts", "build_index.py")

TOP_K = 4
MAX_HISTORY_MESSAGES = 10
REWRITE_MAX_NEW_TOKENS = 96
MULTI_QUERY_MAX_NEW_TOKENS = 128
MULTI_QUERY_VARIANTS = 3
PER_QUERY_RETRIEVAL_K = 4

DEMO_QUESTIONS = [
    "Which companies launched new AI products?",
    "What are the common themes across these articles?",
    "How do two articles differ in their framing of AI regulation?",
    "Which of those focuses on startups?",
    "Summarize that one.",
    "What is the main concern mentioned there?",
    "What are the major AI trends?",
    "What are the main themes in recent AI news?",
    "Which one focuses more on product launches?",
    "What about the article from BBC?",
]


# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


@st.cache_resource
def load_vector_store():
    if not os.path.exists(PERSIST_DIR):
        return None

    embedding_function = load_embedding_function()
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function,
    )
    return vector_store


@st.cache_resource
def load_generation_llm(model_name: str, max_new_tokens: int):
    text_gen_pipeline = pipeline(
        task="text-generation",
        model=model_name,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=text_gen_pipeline)


def load_index_report():
    if not os.path.exists(INDEX_REPORT_PATH):
        return None
    try:
        with open(INDEX_REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def format_snippet(text: str, max_chars: int = 250) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def get_recent_history(history: List[Dict], max_messages: int = MAX_HISTORY_MESSAGES) -> List[Dict]:
    return history[-max_messages:]


def format_history_for_prompt(history: List[Dict]) -> str:
    if not history:
        return "No prior conversation."

    lines = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# -------------------------
# Query rewriting (memory-aware)
# -------------------------
def build_rewrite_prompt(user_query: str, recent_history: List[Dict]) -> str:
    history_text = format_history_for_prompt(recent_history)

    prompt = f"""
You are helping reformulate a follow-up question for retrieval in a RAG system.

Given the recent conversation history and the latest user question, rewrite the latest question
as a standalone search query that can be understood without the chat history.

Rules:
- Do NOT answer the question.
- Only output the rewritten standalone query.
- If the latest user question is already standalone, return it unchanged.
- Be concise and specific.

Recent Conversation History:
{history_text}

Latest User Question:
{user_query}
"""
    return prompt.strip()


def rewrite_query_with_history(user_query: str, recent_history: List[Dict], hf_model_name: str) -> str:
    if not recent_history:
        return user_query

    try:
        llm = load_generation_llm(hf_model_name, REWRITE_MAX_NEW_TOKENS)
        rewrite_prompt = build_rewrite_prompt(user_query, recent_history)
        rewritten = llm.invoke(rewrite_prompt)

        if isinstance(rewritten, str):
            rewritten = rewritten.strip()
        else:
            rewritten = str(rewritten).strip()

        if not rewritten:
            return user_query

        rewritten = rewritten.splitlines()[0].strip()

        if len(rewritten) < 3:
            return user_query

        return rewritten

    except Exception:
        return user_query


# -------------------------
# Multi-query advanced feature
# -------------------------
def clean_query_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\d+[\).\-\s]+", "", line)
    line = re.sub(r"^[-•]\s*", "", line)
    return line.strip()


def looks_like_bad_query(line: str) -> bool:
    lower = line.lower().strip()

    bad_prefixes = [
        "alternative queries",
        "query variants",
        "queries",
        "variants",
    ]
    if any(lower.startswith(x) for x in bad_prefixes):
        return True

    if "2021" in lower or "2022" in lower or "2023" in lower or "2024" in lower or "2025" in lower:
        return True

    if len(lower) < 4:
        return True

    return False


def build_multi_query_prompt(standalone_query: str, num_variants: int = 3) -> str:
    prompt = f"""
You are helping query expansion for a RAG retrieval system.

Given one search query, generate {num_variants} alternative retrieval queries
that express the same information need from different angles.

Rules:
- Do NOT answer the question.
- Output exactly {num_variants} lines.
- One query per line.
- No numbering.
- No bullet points.
- No labels like "Alternative Queries".
- Keep each query concise and retrieval-friendly.
- Keep the time frame consistent with the original query.
- Do not invent a year or date unless the original query includes one.

Original Query:
{standalone_query}
"""
    return prompt.strip()


def generate_multi_queries(
    standalone_query: str,
    hf_model_name: str,
    num_variants: int = MULTI_QUERY_VARIANTS,
) -> List[str]:
    try:
        llm = load_generation_llm(hf_model_name, MULTI_QUERY_MAX_NEW_TOKENS)
        prompt = build_multi_query_prompt(standalone_query, num_variants=num_variants)
        raw_output = llm.invoke(prompt)

        if isinstance(raw_output, str):
            raw_output = raw_output.strip()
        else:
            raw_output = str(raw_output).strip()

        lines = [clean_query_line(x) for x in raw_output.splitlines()]
        lines = [x for x in lines if x and not looks_like_bad_query(x)]

        queries = [standalone_query]
        seen = {standalone_query.lower()}

        for line in lines:
            if line.lower() not in seen and len(line) > 3:
                queries.append(line)
                seen.add(line.lower())

        return queries[: num_variants + 1]

    except Exception:
        return [standalone_query]


def doc_unique_key(doc: Document) -> str:
    url = doc.metadata.get("url", "unknown_url")
    chunk_id = doc.metadata.get("chunk_id", "unknown_chunk")
    return f"{url}|{chunk_id}"


def retrieve_documents_multi_query(
    multi_queries: List[str],
    vector_store,
    per_query_k: int = PER_QUERY_RETRIEVAL_K,
    final_top_k: int = TOP_K,
) -> List[Document]:
    merged_results = {}

    for query in multi_queries:
        docs = vector_store.similarity_search(query, k=per_query_k)

        for rank, doc in enumerate(docs, start=1):
            key = doc_unique_key(doc)
            score_gain = per_query_k - rank + 1

            if key not in merged_results:
                merged_results[key] = {
                    "doc": doc,
                    "score": 0,
                    "matched_queries": [],
                }

            merged_results[key]["score"] += score_gain
            merged_results[key]["matched_queries"].append(query)

    ranked = sorted(
        merged_results.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    final_docs = []
    for item in ranked[:final_top_k]:
        doc = item["doc"]
        doc.metadata["multi_query_score"] = item["score"]
        doc.metadata["matched_queries"] = item["matched_queries"]
        final_docs.append(doc)

    return final_docs


# -------------------------
# Answer generation
# -------------------------
def build_generation_prompt(
    original_user_query: str,
    standalone_query: str,
    multi_queries: List[str],
    retrieved_docs: List[Document],
    recent_history: List[Dict],
) -> str:
    history_text = format_history_for_prompt(recent_history)

    context_blocks = []

    for i, doc in enumerate(retrieved_docs, start=1):
        title = doc.metadata.get("title", "Unknown Title")
        source_name = doc.metadata.get("source_name", "Unknown Source")
        published_date = doc.metadata.get("published_date", "Unknown Date")
        url = doc.metadata.get("url", "Unknown URL")
        chunk_id = doc.metadata.get("chunk_id", "N/A")

        block = f"""
[Source {i}]
Title: {title}
Source: {source_name}
Date: {published_date}
URL: {url}
Chunk ID: {chunk_id}
Content:
{doc.page_content}
"""
        context_blocks.append(block.strip())

    joined_context = "\n\n".join(context_blocks)
    multi_query_text = "\n".join(multi_queries)

    prompt = f"""
You are a helpful domain expert assistant for AI and technology news.

Use the retrieved context below to answer the user's question.
The recent conversation history is provided only to help interpret follow-up questions.
Your answer must be grounded ONLY in the retrieved context.

Rules:
- Do not use outside knowledge.
- If the retrieved context is insufficient, say you do not know based on the indexed documents.
- Be accurate, concise, and grounded.
- If the user asks for bullets, use bullets.
- Prefer 2 to 4 short paragraphs or concise bullet points.
- At the end, provide a short "Sources" section using the retrieved source titles or source names.

Recent Conversation History:
{history_text}

Original User Question:
{original_user_query}

Standalone Retrieval Query Used:
{standalone_query}

Multi-Query Variants Used:
{multi_query_text}

Retrieved Context:
{joined_context}
"""
    return prompt.strip()


def docs_to_source_cards(retrieved_docs: List[Document]) -> List[Dict]:
    source_cards = []

    for doc in retrieved_docs:
        source_cards.append({
            "title": doc.metadata.get("title", "Unknown Title"),
            "source_name": doc.metadata.get("source_name", "Unknown Source"),
            "published_date": doc.metadata.get("published_date", "Unknown Date"),
            "url": doc.metadata.get("url", "Unknown URL"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            "snippet": format_snippet(doc.page_content, max_chars=350),
            "matched_queries": doc.metadata.get("matched_queries", []),
            "multi_query_score": doc.metadata.get("multi_query_score", None),
        })

    return source_cards


def fallback_answer_from_docs(
    query: str,
    standalone_query: str,
    multi_queries: List[str],
    retrieved_docs: List[Document],
    error_message: str = "",
) -> str:
    lines = [
        "Local Hugging Face answer generation is currently unavailable.",
    ]

    if error_message:
        lines.extend(["", f"Error: {error_message}"])

    lines.extend([
        "",
        f"Original question: {query}",
        f"Standalone retrieval query used: {standalone_query}",
        "",
        "Multi-query variants used:",
    ])

    for q in multi_queries:
        lines.append(f"- {q}")

    lines.extend([
        "",
        "Here are the most relevant retrieved sources:",
        "",
    ])

    for i, doc in enumerate(retrieved_docs, start=1):
        title = doc.metadata.get("title", "Unknown Title")
        source_name = doc.metadata.get("source_name", "Unknown Source")
        published_date = doc.metadata.get("published_date", "Unknown Date")
        snippet = format_snippet(doc.page_content, max_chars=220)

        lines.append(
            f"{i}. {title} | {source_name} | {published_date}\n"
            f"   Snippet: {snippet}"
        )

    lines.append("")
    lines.append("Sources:")
    for i, doc in enumerate(retrieved_docs, start=1):
        title = doc.metadata.get("title", "Unknown Title")
        source_name = doc.metadata.get("source_name", "Unknown Source")
        lines.append(f"- Source {i}: {title} ({source_name})")

    return "\n".join(lines)


def generate_answer(
    original_user_query: str,
    standalone_query: str,
    multi_queries: List[str],
    retrieved_docs: List[Document],
    recent_history: List[Dict],
    hf_model_name: str,
    max_new_tokens: int,
) -> str:
    prompt = build_generation_prompt(
        original_user_query=original_user_query,
        standalone_query=standalone_query,
        multi_queries=multi_queries,
        retrieved_docs=retrieved_docs,
        recent_history=recent_history,
    )

    try:
        llm = load_generation_llm(hf_model_name, max_new_tokens)
        response = llm.invoke(prompt)

        if isinstance(response, str):
            return response

        return str(response)

    except Exception as e:
        return fallback_answer_from_docs(
            query=original_user_query,
            standalone_query=standalone_query,
            multi_queries=multi_queries,
            retrieved_docs=retrieved_docs,
            error_message=str(e),
        )


# -------------------------
# Index rebuild
# -------------------------
def run_rebuild_index() -> Dict:
    if not os.path.exists(BUILD_INDEX_SCRIPT):
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Could not find build script: {BUILD_INDEX_SCRIPT}",
        }

    cmd = [sys.executable, BUILD_INDEX_SCRIPT, "--rebuild"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )

    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# -------------------------
# testing helpers
# -------------------------
def get_latest_assistant_message(chat_history: List[Dict]) -> Optional[Dict]:
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            return msg
    return None


def build_retrieval_debug_rows(sources: List[Dict]) -> pd.DataFrame:
    rows = []
    for i, src in enumerate(sources, start=1):
        rows.append({
            "rank": i,
            "title": src.get("title"),
            "source_name": src.get("source_name"),
            "published_date": src.get("published_date"),
            "chunk_id": src.get("chunk_id"),
            "multi_query_score": src.get("multi_query_score"),
            "matched_queries": " | ".join(src.get("matched_queries", [])),
            "snippet": src.get("snippet"),
            "url": src.get("url"),
        })
    return pd.DataFrame(rows)


def upsert_test_result(result_row: Dict):
    turn_id = result_row["turn_id"]

    existing_idx = None
    for i, row in enumerate(st.session_state.test_results):
        if row.get("turn_id") == turn_id:
            existing_idx = i
            break

    if existing_idx is None:
        st.session_state.test_results.append(result_row)
    else:
        st.session_state.test_results[existing_idx] = result_row


def test_results_df() -> pd.DataFrame:
    if not st.session_state.test_results:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.test_results)


def summarize_sources_for_log(sources: List[Dict]) -> str:
    parts = []
    for src in sources:
        title = src.get("title", "Unknown Title")
        source_name = src.get("source_name", "Unknown Source")
        parts.append(f"{title} ({source_name})")
    return " | ".join(parts)


def render_assistant_metadata(message: Dict):
    rewritten_query = message.get("rewritten_query")
    if rewritten_query:
        with st.expander("Standalone retrieval query used"):
            st.code(rewritten_query)

    multi_queries = message.get("multi_queries", [])
    if multi_queries:
        with st.expander("Multi-query variants used"):
            for i, q in enumerate(multi_queries, start=1):
                st.write(f"{i}. {q}")

    sources = message.get("sources", [])
    if sources:
        with st.expander("Retrieved Source Citations"):
            for i, src in enumerate(sources, start=1):
                st.write(f"**Source {i}: {src['title']}**")
                st.write(f"**Source Name:** {src['source_name']}")
                st.write(f"**Published Date:** {src['published_date']}")
                st.write(f"**URL:** {src['url']}")
                st.write(f"**Chunk ID:** {src['chunk_id']}")
                if src.get("multi_query_score") is not None:
                    st.write(f"**Merged Retrieval Score:** {src['multi_query_score']}")
                if src.get("matched_queries"):
                    st.write("**Matched Queries:**")
                    for q in src["matched_queries"]:
                        st.write(f"- {q}")
                st.write(f"**Snippet:** {src['snippet']}")
                st.divider()


# -------------------------
# Session state
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rebuild_status" not in st.session_state:
    st.session_state.rebuild_status = "Not run in this session."

if "rebuild_log" not in st.session_state:
    st.session_state.rebuild_log = ""

if "test_results" not in st.session_state:
    st.session_state.test_results = []

if "pending_demo_query" not in st.session_state:
    st.session_state.pending_demo_query = None

if "assistant_turn_counter" not in st.session_state:
    st.session_state.assistant_turn_counter = 0


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Settings")

    st.subheader("Domain")
    st.write("**Corpus:** AI and Technology News")
    st.caption(
        "This app uses local Hugging Face embeddings, local Chroma retrieval, and local Hugging Face generation."
    )

    hf_model_name = st.text_input(
        "Hugging Face generation model",
        value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=64,
        max_value=512,
        value=256,
        step=32,
    )

    st.divider()
    st.subheader("Advanced Feature")
    st.write("**Multi-Query Retrieval:** Enabled")
    st.caption(
        "A single user question is expanded into multiple retrieval variants. "
        "Results from these variants are merged to improve recall."
    )

    st.divider()
    st.subheader("Memory Settings")
    st.write(f"**Recent messages retained:** {MAX_HISTORY_MESSAGES}")
    st.caption("10 messages ≈ 5 exchanges. Follow-up questions are rewritten into standalone retrieval queries.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with col_b:
        rebuild_clicked = st.button("Rebuild index", use_container_width=True)

    if rebuild_clicked:
        with st.spinner("Rebuilding index... this may take a while."):
            result = run_rebuild_index()
            st.session_state.rebuild_status = "Success" if result["ok"] else "Failed"
            combined_log = (result["stdout"] or "") + "\n" + (result["stderr"] or "")
            st.session_state.rebuild_log = combined_log[-6000:]

        st.cache_resource.clear()
        st.rerun()

    st.write(f"**Last rebuild status:** {st.session_state.rebuild_status}")
    if st.session_state.rebuild_log:
        with st.expander("Rebuild log"):
            st.code(st.session_state.rebuild_log)

    st.divider()
    st.subheader("Index Status")

    index_report = load_index_report()
    if index_report:
        st.write(f"**Documents indexed:** {index_report.get('documents_indexed', 'N/A')}")
        st.write(f"**Chunks indexed:** {index_report.get('chunks_created', 'N/A')}")
        st.write(f"**Persist directory:** `{index_report.get('persist_directory', 'N/A')}`")
        st.write(f"**Embedding model:** {index_report.get('embedding_model', 'N/A')}")
        st.write(f"**Last build:** {index_report.get('indexed_at', 'N/A')}")
    else:
        st.warning("No index report found yet.")

    st.divider()
    st.subheader("Corpus Stats")
    if index_report:
        st.write(f"**Corpus documents:** {index_report.get('documents_indexed', 'N/A')}")
        st.write(f"**Vectorized chunks:** {index_report.get('chunks_created', 'N/A')}")
    else:
        st.write("Corpus stats unavailable until an index report is present.")

    st.divider()
    st.subheader("Retrieval Settings")
    st.write(f"**Final Top-K retrieval:** {TOP_K}")
    st.write(f"**Per-query retrieval K:** {PER_QUERY_RETRIEVAL_K}")
    st.write(f"**Multi-query variants generated:** {MULTI_QUERY_VARIANTS}")

    st.divider()
    st.subheader("Testing")

    selected_demo_question = st.selectbox(
        "Demo question bank",
        DEMO_QUESTIONS,
        index=0,
    )

    if st.button("Run selected demo question", use_container_width=True):
        st.session_state.pending_demo_query = selected_demo_question
        st.rerun()

    df_test = test_results_df()
    if not df_test.empty:
        csv_bytes = df_test.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download test log CSV",
            data=csv_bytes,
            file_name="test_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Test log preview"):
            st.dataframe(df_test, use_container_width=True)


# -------------------------
# Load vector store
# -------------------------
vector_store = load_vector_store()

if vector_store is None:
    st.error(
        "No persistent Chroma DB found. Please build the index first "
        "using `python scripts/build_index.py --rebuild`."
    )
    st.stop()


# -------------------------
# Render existing chat history
# -------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_assistant_metadata(message)


# -------------------------
# evaluation panel
# -------------------------
latest_assistant = get_latest_assistant_message(st.session_state.chat_history)

if latest_assistant:
    st.divider()
    st.subheader("Evaluation Panel")

    with st.expander("Evaluate latest response", expanded=False):
        turn_id = latest_assistant.get("turn_id", -1)

        retrieval_reasonable = st.selectbox(
            "Were the top retrieved chunks reasonable?",
            ["Yes", "Partly", "No"],
            key=f"retrieval_reasonable_{turn_id}",
        )

        answer_grounded = st.selectbox(
            "Was the answer grounded in retrieved context?",
            ["Yes", "Partly", "No"],
            key=f"answer_grounded_{turn_id}",
        )

        citations_aligned = st.selectbox(
            "Did the citations match the answer content?",
            ["Yes", "Partly", "No"],
            key=f"citations_aligned_{turn_id}",
        )

        honest_unknown = st.selectbox(
            "If context was insufficient, was the bot honest about not knowing?",
            ["Yes", "Partly", "No", "Not applicable"],
            key=f"honest_unknown_{turn_id}",
        )

        tester_notes = st.text_area(
            "Tester notes",
            key=f"tester_notes_{turn_id}",
            height=120,
        )

        if st.button("Save evaluation", key=f"save_evaluation_{turn_id}"):
            row = {
                "turn_id": turn_id,
                "logged_at": latest_assistant.get("logged_at"),
                "original_user_query": latest_assistant.get("original_user_query"),
                "standalone_query": latest_assistant.get("rewritten_query"),
                "multi_queries": " | ".join(latest_assistant.get("multi_queries", [])),
                "retrieval_reasonable": retrieval_reasonable,
                "answer_grounded": answer_grounded,
                "citations_aligned": citations_aligned,
                "honest_unknown": honest_unknown,
                "num_sources": len(latest_assistant.get("sources", [])),
                "source_summary": summarize_sources_for_log(latest_assistant.get("sources", [])),
                "tester_notes": tester_notes,
            }
            upsert_test_result(row)
            st.success("Evaluation saved.")

    with st.expander("Inspect raw retrieved chunks for latest response", expanded=False):
        latest_sources = latest_assistant.get("sources", [])
        if latest_sources:
            st.dataframe(build_retrieval_debug_rows(latest_sources), use_container_width=True)
        else:
            st.info("No source data found for the latest assistant response.")


# -------------------------
# New user input
# -------------------------
pending_demo_query = st.session_state.pending_demo_query
chat_input_query = st.chat_input("Ask a follow-up or a new question about the indexed news corpus...")

user_query = pending_demo_query if pending_demo_query else chat_input_query

if pending_demo_query:
    st.session_state.pending_demo_query = None

if user_query:
    if not hf_model_name.strip():
        st.warning("Please enter a Hugging Face generation model name in the sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_query)

    recent_history = get_recent_history(st.session_state.chat_history, MAX_HISTORY_MESSAGES)

    with st.spinner("Rewriting follow-up question for retrieval..."):
        standalone_query = rewrite_query_with_history(
            user_query=user_query,
            recent_history=recent_history,
            hf_model_name=hf_model_name,
        )

    with st.spinner("Generating multi-query retrieval variants..."):
        multi_queries = generate_multi_queries(
            standalone_query=standalone_query,
            hf_model_name=hf_model_name,
            num_variants=MULTI_QUERY_VARIANTS,
        )

    with st.spinner("Retrieving and merging results from multi-query variants..."):
        retrieved_docs = retrieve_documents_multi_query(
            multi_queries=multi_queries,
            vector_store=vector_store,
            per_query_k=PER_QUERY_RETRIEVAL_K,
            final_top_k=TOP_K,
        )

    if not retrieved_docs:
        assistant_answer = "I could not retrieve relevant documents for this question."
        assistant_sources = []
    else:
        with st.spinner("Generating grounded answer..."):
            assistant_answer = generate_answer(
                original_user_query=user_query,
                standalone_query=standalone_query,
                multi_queries=multi_queries,
                retrieved_docs=retrieved_docs,
                recent_history=recent_history,
                hf_model_name=hf_model_name,
                max_new_tokens=max_new_tokens,
            )
        assistant_sources = docs_to_source_cards(retrieved_docs)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
    })

    st.session_state.assistant_turn_counter += 1
    assistant_turn_id = st.session_state.assistant_turn_counter

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_answer,
        "rewritten_query": standalone_query,
        "multi_queries": multi_queries,
        "sources": assistant_sources,
        "turn_id": assistant_turn_id,
        "original_user_query": user_query,
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    })

    st.session_state.chat_history = get_recent_history(
        st.session_state.chat_history,
        MAX_HISTORY_MESSAGES,
    )

    st.rerun()