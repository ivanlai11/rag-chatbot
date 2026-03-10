import argparse
import json
import os
import shutil
from datetime import datetime

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils import (
    chunk_langchain_documents,
    dataframe_to_langchain_documents,
    process_corpus_dataframe,
    save_chunk_outputs,
    save_processing_outputs,
)

AI_INCLUDE_KEYWORDS = [
    "ai",
    "artificial intelligence",
    "openai",
    "anthropic",
    "google",
    "nvidia",
    "llm",
    "large language model",
    "model",
    "agent",
    "agents",
    "robotics",
    "robot",
    "semiconductor",
    "chip",
    "chips",
    "gpu",
    "cybersecurity",
    "security",
    "cloud",
    "startup",
    "startups",
    "product launch",
    "launches",
    "generative ai",
    "machine learning",
    "deep learning",
    "inference",
    "foundation model",
]

AI_EXCLUDE_KEYWORDS = [
    "coffee",
    "espresso",
    "mattress",
    "sleep",
    "bed",
    "beds",
    "headphones",
    "alarm clock",
    "deals",
    "coupon",
    "sale",
    "speaker",
    "speakers",
    "brew",
    "tax prep",
    "dating app",
    "wireless headphones",
    "base layers",
]


def filter_domain_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep articles relevant to the AI / technology news domain.
    """
    keep_rows = []

    for _, row in df.iterrows():
        title = str(row.get("title", "")).lower()
        text = str(row.get("raw_text", "")).lower()
        combined = f"{title} {text}"

        if any(keyword in title for keyword in AI_EXCLUDE_KEYWORDS):
            continue

        title_hits = sum(1 for keyword in AI_INCLUDE_KEYWORDS if keyword in title)
        combined_hits = sum(1 for keyword in AI_INCLUDE_KEYWORDS if keyword in combined)

        if title_hits >= 1 or combined_hits >= 1:
            keep_rows.append(row.to_dict())

    if not keep_rows:
        return df.iloc[0:0].copy()

    return pd.DataFrame(keep_rows).reset_index(drop=True)


def clear_persist_directory(persist_dir: str) -> None:
    """
    Delete old Chroma DB folder for clean rebuild.
    """
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)


def get_chunk_count_from_vector_store(vector_store) -> int:
    """
    Try to read Chroma collection count safely.
    """
    try:
        return vector_store._collection.count()
    except Exception:
        return -1


def save_index_report(
    output_dir: str,
    persist_dir: str,
    document_count: int,
    chunk_count: int,
    vector_count: int,
    rebuild_used: bool,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "indexed_at": datetime.utcnow().isoformat() + "Z",
        "persist_directory": persist_dir,
        "documents_indexed": document_count,
        "chunks_created": chunk_count,
        "vectors_in_collection": vector_count,
        "rebuild_used": rebuild_used,
        "embedding_model": "HuggingFaceEmbeddings(sentence-transformers/all-mpnet-base-v2)",
        "vector_store": "Chroma",
    }

    report_path = os.path.join(output_dir, "index_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_path


def build_chroma_vector_store(
    chunked_docs,
    persist_dir: str,
):
    """
    Build Chroma vector DB from chunked documents using local Hugging Face embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    vector_store = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vector_store


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3 + 4 + 5: Process corpus, filter domain, chunk documents, and build Chroma vector DB."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/processed/corpus.csv",
        help="Path to raw corpus CSV collected from RSS/news ingestion.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save cleaned corpus outputs and reports.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory for persistent Chroma vector store.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Minimum article length required to keep a document.",
    )
    parser.add_argument(
        "--preview-docs",
        type=int,
        default=3,
        help="How many LangChain document previews to print.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for recursive chunking.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for recursive chunking.",
    )
    parser.add_argument(
        "--preview-chunks",
        type=int,
        default=5,
        help="How many chunk previews to print.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing chroma_db and rebuild index from scratch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    print(f"[INFO] Loading corpus from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    print(f"[INFO] Raw rows loaded: {len(df)}")
    print(f"[INFO] Running document processing with min_chars={args.min_chars}")

    processed_df, dropped_df, report = process_corpus_dataframe(
        df=df,
        min_chars=args.min_chars,
    )

    print(f"[INFO] Rows after basic processing: {len(processed_df)}")

    domain_filtered_df = filter_domain_articles(processed_df)

    print(f"[INFO] Rows after domain filtering: {len(domain_filtered_df)}")

    report["rows_after_basic_processing"] = int(len(processed_df))
    report["rows_after_domain_filter"] = int(len(domain_filtered_df))
    report["domain_filter_removed_rows"] = int(len(processed_df) - len(domain_filtered_df))

    processed_df = domain_filtered_df

    output_paths = save_processing_outputs(
        processed_df=processed_df,
        dropped_df=dropped_df,
        report=report,
        output_dir=args.output_dir,
    )

    print("\n[INFO] Processing complete")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n[INFO] Saved processing outputs:")
    for name, path in output_paths.items():
        print(f"  - {name}: {path}")

    documents = dataframe_to_langchain_documents(processed_df)

    print(f"\n[INFO] LangChain Documents created: {len(documents)}")

    preview_n = min(args.preview_docs, len(documents))
    if preview_n > 0:
        print(f"\n[INFO] Previewing first {preview_n} LangChain Documents:\n")
        for i in range(preview_n):
            doc = documents[i]
            print(f"===== Document {i+1} =====")
            print("Metadata:")
            print(json.dumps(doc.metadata, indent=2, ensure_ascii=False))
            print("\nPage content preview:")
            print(doc.page_content[:500])
            print("\n")

    if len(documents) < 50:
        print(
            "[WARN] After domain filtering, fewer than 50 documents remain. "
            "You may want to relax filtering rules or collect more relevant articles."
        )

    # -------------------------
    # Phase 4: Recursive chunking
    # -------------------------
    print("\n[INFO] Starting Phase 4: recursive chunking")
    print(
        f"[INFO] Chunking strategy = RecursiveCharacterTextSplitter | "
        f"chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}"
    )

    chunked_docs = chunk_langchain_documents(
        documents=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    chunk_paths = save_chunk_outputs(
        chunked_docs=chunked_docs,
        output_dir=args.output_dir,
    )

    print(f"[INFO] Total chunked documents created: {len(chunked_docs)}")
    print("[INFO] Saved chunk outputs:")
    for name, path in chunk_paths.items():
        print(f"  - {name}: {path}")

    preview_chunk_n = min(args.preview_chunks, len(chunked_docs))
    if preview_chunk_n > 0:
        print(f"\n[INFO] Previewing first {preview_chunk_n} chunks:\n")
        for i in range(preview_chunk_n):
            chunk = chunked_docs[i]
            print(f"===== Chunk {i+1} =====")
            print("Metadata:")
            print(json.dumps(chunk.metadata, indent=2, ensure_ascii=False))
            print("\nChunk content preview:")
            print(chunk.page_content[:400])
            print("\n")

    print("[INFO] Phase 4 finished successfully.")

    # -------------------------
    # Phase 5: Build Chroma vector DB
    # -------------------------
    print("\n[INFO] Starting Phase 5: building Chroma vector database")

    if args.rebuild:
        print(f"[INFO] Rebuild requested. Clearing existing vector store at: {args.persist_dir}")
        clear_persist_directory(args.persist_dir)

    vector_store = build_chroma_vector_store(
        chunked_docs=chunked_docs,
        persist_dir=args.persist_dir,
    )

    vector_count = get_chunk_count_from_vector_store(vector_store)

    print("[INFO] Vector database build complete.")
    print(f"[INFO] Persist directory: {args.persist_dir}")
    print(f"[INFO] Documents indexed: {len(documents)}")
    print(f"[INFO] Chunks indexed: {len(chunked_docs)}")
    print(f"[INFO] Vectors stored in Chroma: {vector_count}")

    index_report_path = save_index_report(
        output_dir=args.output_dir,
        persist_dir=args.persist_dir,
        document_count=len(documents),
        chunk_count=len(chunked_docs),
        vector_count=vector_count,
        rebuild_used=args.rebuild,
    )

    print(f"[INFO] Index report saved: {index_report_path}")
    print("[INFO] Phase 5 finished successfully.")


if __name__ == "__main__":
    main()