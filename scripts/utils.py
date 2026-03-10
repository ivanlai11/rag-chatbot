import html
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


REQUIRED_COLUMNS = [
    "doc_id",
    "title",
    "published_date",
    "source_name",
    "url",
    "raw_text",
]


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in corpus.csv: {missing}")


def clean_metadata_text(value: Optional[str]) -> str:
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_article_text(text: Optional[str]) -> str:
    """
    Basic text cleaning for article body:
    - decode HTML entities
    - remove zero-width / BOM / non-breaking spaces
    - normalize newlines
    - remove excessive spaces
    - remove simple boilerplate lines
    """
    if text is None or pd.isna(text):
        return ""

    text = html.unescape(str(text))
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    raw_lines = text.split("\n")
    cleaned_lines = []

    boilerplate_patterns = [
        r"^advertisement$",
        r"^subscribe$",
        r"^sign up$",
        r"^read more$",
        r"^skip to content$",
        r"^newsletter$",
        r"^follow us$",
    ]

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        lower_line = line.lower()
        if any(re.match(pattern, lower_line) for pattern in boilerplate_patterns):
            continue

        line = re.sub(r"[ \t]+", " ", line)
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def normalize_published_date(value: Optional[str]) -> Optional[str]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None

    dt = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.isoformat()


def normalize_doc_id(doc_id: Optional[str], title: str, url: str) -> str:
    if doc_id is not None and not pd.isna(doc_id) and str(doc_id).strip():
        return str(doc_id).strip()

    fallback = f"{title}|{url}"
    return str(abs(hash(fallback)))


def validate_and_normalize_row(row: pd.Series, min_chars: int = 500) -> Tuple[Optional[Dict], List[str]]:
    """
    Returns:
        normalized_record or None,
        list of issues / drop reasons
    """
    issues = []

    title = clean_metadata_text(row.get("title"))
    url = clean_metadata_text(row.get("url"))
    source_name = clean_metadata_text(row.get("source_name")) or "unknown_source"
    published_date = normalize_published_date(row.get("published_date"))
    raw_text = clean_article_text(row.get("raw_text"))
    rss_summary = clean_article_text(row.get("rss_summary")) if "rss_summary" in row else ""
    domain_tag = clean_metadata_text(row.get("domain_tag")) or "AI_Tech_News"
    extraction_method = clean_metadata_text(row.get("extraction_method")) or "unknown"

    if not raw_text and rss_summary:
        raw_text = rss_summary

    doc_id = normalize_doc_id(row.get("doc_id"), title, url)

    if not title:
        issues.append("missing_title")
    if not url:
        issues.append("missing_url")
    elif not is_valid_url(url):
        issues.append("invalid_url")
    if not raw_text:
        issues.append("missing_text")
    elif len(raw_text) < min_chars:
        issues.append("text_too_short")

    if not published_date:
        issues.append("missing_or_invalid_date")

    if any(x in issues for x in ["missing_title", "missing_url", "invalid_url", "missing_text", "text_too_short"]):
        return None, issues

    normalized = {
        "doc_id": doc_id,
        "title": title,
        "published_date": published_date,
        "source_name": source_name,
        "url": url,
        "rss_summary": rss_summary,
        "raw_text": raw_text,
        "domain_tag": domain_tag,
        "extraction_method": extraction_method,
        "text_length": len(raw_text),
    }
    return normalized, issues


def process_corpus_dataframe(
    df: pd.DataFrame,
    min_chars: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Clean corpus and validate metadata.
    Returns:
        processed_df,
        dropped_df,
        report_dict
    """
    ensure_required_columns(df)

    kept_records = []
    dropped_records = []
    issue_counter = Counter()

    for _, row in df.iterrows():
        normalized, issues = validate_and_normalize_row(row, min_chars=min_chars)

        for issue in issues:
            issue_counter[issue] += 1

        if normalized is None:
            dropped = row.to_dict()
            dropped["drop_reasons"] = "|".join(issues) if issues else "unknown"
            dropped_records.append(dropped)
        else:
            normalized["quality_flags"] = "|".join(issues) if issues else ""
            kept_records.append(normalized)

    processed_df = pd.DataFrame(kept_records)
    dropped_df = pd.DataFrame(dropped_records)

    if not processed_df.empty:
        processed_df = (
            processed_df.sort_values(by="text_length", ascending=False)
            .drop_duplicates(subset=["url"], keep="first")
            .reset_index(drop=True)
        )

    report = {
        "input_rows": int(len(df)),
        "kept_rows": int(len(processed_df)),
        "dropped_rows": int(len(dropped_df)),
        "issue_counts": dict(issue_counter),
        "source_distribution": (
            processed_df["source_name"].value_counts().to_dict()
            if not processed_df.empty
            else {}
        ),
        "avg_text_length": (
            float(processed_df["text_length"].mean())
            if not processed_df.empty
            else 0.0
        ),
        "min_text_length_threshold": min_chars,
    }

    return processed_df, dropped_df, report


def dataframe_to_langchain_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert each processed row into one LangChain Document.
    page_content = cleaned article text
    metadata = citation-friendly fields
    """
    documents: List[Document] = []

    for _, row in df.iterrows():
        page_content = row["raw_text"]
        metadata = {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "url": row["url"],
            "published_date": row["published_date"],
            "source_name": row["source_name"],
            "source_file": row["source_name"],
            "domain_tag": row.get("domain_tag", "AI_Tech_News"),
            "extraction_method": row.get("extraction_method", "unknown"),
            "text_length": int(row.get("text_length", len(page_content))),
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    return documents


def chunk_langchain_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Apply recursive chunking to LangChain Documents.
    Each output chunk preserves parent metadata and adds:
    - parent_doc_id
    - chunk_id
    - chunk_char_length
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunked_docs: List[Document] = []

    for doc in documents:
        parent_doc_id = doc.metadata.get("doc_id", "unknown_doc")
        split_docs = text_splitter.split_documents([doc])

        for idx, chunk_doc in enumerate(split_docs):
            chunk_doc.metadata["parent_doc_id"] = parent_doc_id
            chunk_doc.metadata["chunk_id"] = idx
            chunk_doc.metadata["chunk_char_length"] = len(chunk_doc.page_content)
            chunked_docs.append(chunk_doc)

    return chunked_docs


def save_processing_outputs(
    processed_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
    report: Dict,
    output_dir: str,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    processed_csv_path = os.path.join(output_dir, "corpus_clean.csv")
    dropped_csv_path = os.path.join(output_dir, "corpus_dropped.csv")
    report_json_path = os.path.join(output_dir, "processing_report.json")

    processed_df.to_csv(processed_csv_path, index=False, encoding="utf-8-sig")

    if not dropped_df.empty:
        dropped_df.to_csv(dropped_csv_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["message"]).to_csv(dropped_csv_path, index=False, encoding="utf-8-sig")

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        "processed_csv": processed_csv_path,
        "dropped_csv": dropped_csv_path,
        "report_json": report_json_path,
    }


def save_chunk_outputs(
    chunked_docs: List[Document],
    output_dir: str,
) -> Dict[str, str]:
    """
    Save chunk-level outputs for inspection and debugging.
    Creates:
    - chunk_preview.csv
    - chunk_report.json
    """
    os.makedirs(output_dir, exist_ok=True)

    preview_rows = []
    per_doc_counts = Counter()

    for chunk in chunked_docs:
        parent_doc_id = chunk.metadata.get("parent_doc_id", "unknown_doc")
        per_doc_counts[parent_doc_id] += 1

        preview_rows.append({
            "parent_doc_id": parent_doc_id,
            "chunk_id": chunk.metadata.get("chunk_id"),
            "title": chunk.metadata.get("title"),
            "url": chunk.metadata.get("url"),
            "published_date": chunk.metadata.get("published_date"),
            "source_name": chunk.metadata.get("source_name"),
            "chunk_char_length": chunk.metadata.get("chunk_char_length"),
            "chunk_preview": chunk.page_content[:300],
        })

    preview_df = pd.DataFrame(preview_rows)
    chunk_preview_csv = os.path.join(output_dir, "chunk_preview.csv")
    preview_df.to_csv(chunk_preview_csv, index=False, encoding="utf-8-sig")

    report = {
        "total_chunks": len(chunked_docs),
        "avg_chunks_per_document": (
            len(chunked_docs) / len(per_doc_counts) if per_doc_counts else 0
        ),
        "max_chunks_for_one_document": max(per_doc_counts.values()) if per_doc_counts else 0,
        "min_chunks_for_one_document": min(per_doc_counts.values()) if per_doc_counts else 0,
        "documents_chunked": len(per_doc_counts),
    }

    chunk_report_json = os.path.join(output_dir, "chunk_report.json")
    with open(chunk_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        "chunk_preview_csv": chunk_preview_csv,
        "chunk_report_json": chunk_report_json,
    }