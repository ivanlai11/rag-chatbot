import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import feedparser
import pandas as pd
import requests
import trafilatura


DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://www.theverge.com/rss/index.xml",
    "https://techcrunch.com/feed/",
    "https://www.wired.com/feed/rss",
]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text.strip().lower())
    return text[:max_len] if text else "untitled"


def make_doc_id(url: str, title: str) -> str:
    raw = f"{url}|{title}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_published(entry) -> Optional[str]:
    """
    Return ISO 8601 datetime string if possible, else None.
    """
    candidates = [
        getattr(entry, "published", None),
        getattr(entry, "updated", None),
        entry.get("published"),
        entry.get("updated"),
    ]

    for value in candidates:
        if not value:
            continue
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

    return None


def extract_source_name(feed_url: str, feed_title: Optional[str] = None) -> str:
    if feed_title:
        return feed_title.strip()
    parsed = urlparse(feed_url)
    hostname = parsed.netloc.replace("www.", "")
    return hostname or "unknown_source"


def fetch_rss_entries(feed_url: str, timeout: int = 20) -> Tuple[str, List[dict]]:
    """
    Returns:
        source_name, entries
    """
    resp = requests.get(feed_url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    parsed = feedparser.parse(resp.content)
    source_name = extract_source_name(feed_url, parsed.feed.get("title"))

    entries = []
    for entry in parsed.entries:
        entries.append(entry)

    return source_name, entries


def extract_full_text(url: str, timeout: int = 20, sleep_seconds: float = 0.5) -> Optional[str]:
    """
    Fetch article HTML and extract main text using trafilatura.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()

        downloaded = resp.text
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )

        time.sleep(sleep_seconds)

        if not text:
            return None

        text = normalize_whitespace(text)
        return text if text else None
    except Exception:
        return None


def build_record(
    entry,
    source_name: str,
    domain_tag: str,
    min_chars: int,
    use_full_text: bool = True,
) -> Optional[Dict]:
    title = normalize_whitespace(entry.get("title", "") or "")
    link = entry.get("link", "") or ""
    rss_summary = normalize_whitespace(entry.get("summary", "") or entry.get("description", "") or "")
    published_date = parse_published(entry)

    if not title or not link:
        return None

    raw_text = None
    extraction_method = "rss_summary"

    if use_full_text:
        raw_text = extract_full_text(link)
        if raw_text:
            extraction_method = "full_text"

    if not raw_text:
        raw_text = rss_summary

    raw_text = normalize_whitespace(raw_text or "")

    if len(raw_text) < min_chars:
        return None

    doc_id = make_doc_id(link, title)

    return {
        "doc_id": doc_id,
        "title": title,
        "published_date": published_date,
        "source_name": source_name,
        "url": link,
        "rss_summary": rss_summary,
        "raw_text": raw_text,
        "domain_tag": domain_tag,
        "extraction_method": extraction_method,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "text_length": len(raw_text),
    }


def deduplicate_records(records: List[Dict]) -> List[Dict]:
    """
    Deduplicate by URL first, then by title+date fallback.
    Keep the longer text version if duplicates occur.
    """
    by_key: Dict[str, Dict] = {}

    for rec in records:
        url_key = rec["url"].strip().lower()
        fallback_key = f'{rec["title"].strip().lower()}|{rec.get("published_date") or ""}'
        key = url_key if url_key else fallback_key

        existing = by_key.get(key)
        if existing is None or rec["text_length"] > existing["text_length"]:
            by_key[key] = rec

    return list(by_key.values())


def save_outputs(records: List[Dict], output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(records)

    csv_path = os.path.join(output_dir, "corpus.csv")
    jsonl_path = os.path.join(output_dir, "corpus.jsonl")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def load_feeds_from_file(path: str) -> List[str]:
    """
    Supports:
    - .txt  : one RSS URL per line
    - .json : list of RSS URLs
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feed file not found: {path}")

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON feed file must be a list of RSS URLs.")
        return [str(x).strip() for x in data if str(x).strip()]

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    return [line for line in lines if line and not line.startswith("#")]


def collect_news(
    feeds: List[str],
    output_dir: str,
    target_docs: int,
    per_feed_limit: int,
    min_chars: int,
    domain_tag: str,
    use_full_text: bool,
) -> List[Dict]:
    all_records: List[Dict] = []

    print(f"[INFO] Starting collection from {len(feeds)} feeds")
    print(f"[INFO] Target documents: {target_docs}")
    print(f"[INFO] Minimum characters per article: {min_chars}")
    print(f"[INFO] Full-text extraction enabled: {use_full_text}")

    for i, feed_url in enumerate(feeds, start=1):
        print(f"\n[INFO] Fetching feed {i}/{len(feeds)}: {feed_url}")

        try:
            source_name, entries = fetch_rss_entries(feed_url)
            print(f"[INFO] Source: {source_name} | Entries found: {len(entries)}")
        except Exception as e:
            print(f"[WARN] Failed to fetch feed: {feed_url} | Error: {e}")
            continue

        feed_count = 0
        for entry in entries:
            if feed_count >= per_feed_limit:
                break

            record = build_record(
                entry=entry,
                source_name=source_name,
                domain_tag=domain_tag,
                min_chars=min_chars,
                use_full_text=use_full_text,
            )

            if record:
                all_records.append(record)
                feed_count += 1
                print(
                    f"[OK] {record['title'][:80]} "
                    f"| method={record['extraction_method']} "
                    f"| chars={record['text_length']}"
                )

        print(f"[INFO] Accepted from this feed: {feed_count}")

        deduped_so_far = deduplicate_records(all_records)
        if len(deduped_so_far) >= target_docs:
            print(f"[INFO] Target reached with {len(deduped_so_far)} unique documents.")
            all_records = deduped_so_far
            break

    all_records = deduplicate_records(all_records)
    all_records.sort(
        key=lambda x: (
            x["published_date"] is None,
            x["published_date"] or "",
        ),
        reverse=True,
    )

    if len(all_records) > target_docs:
        all_records = all_records[:target_docs]

    csv_path, jsonl_path = save_outputs(all_records, output_dir)

    print("\n[INFO] Collection complete")
    print(f"[INFO] Unique documents saved: {len(all_records)}")
    print(f"[INFO] CSV saved to: {csv_path}")
    print(f"[INFO] JSONL saved to: {jsonl_path}")

    source_counts = (
        pd.DataFrame(all_records)["source_name"].value_counts().to_dict()
        if all_records else {}
    )
    print(f"[INFO] Source distribution: {source_counts}")

    return all_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect AI/Technology news articles from RSS feeds and save a clean corpus."
    )
    parser.add_argument(
        "--feeds-file",
        type=str,
        default="",
        help="Path to .txt or .json file containing RSS URLs. If omitted, built-in default feeds are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save corpus outputs.",
    )
    parser.add_argument(
        "--target-docs",
        type=int,
        default=80,
        help="Maximum number of unique documents to save.",
    )
    parser.add_argument(
        "--per-feed-limit",
        type=int,
        default=25,
        help="Maximum accepted articles per feed.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=700,
        help="Minimum character count required to keep an article.",
    )
    parser.add_argument(
        "--domain-tag",
        type=str,
        default="AI_Tech_News",
        help="Domain tag stored in metadata.",
    )
    parser.add_argument(
        "--no-full-text",
        action="store_true",
        help="Disable article full-text extraction and use RSS summaries only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    feeds = DEFAULT_FEEDS
    if args.feeds_file:
        feeds = load_feeds_from_file(args.feeds_file)

    if not feeds:
        raise ValueError("No RSS feeds provided.")

    records = collect_news(
        feeds=feeds,
        output_dir=args.output_dir,
        target_docs=args.target_docs,
        per_feed_limit=args.per_feed_limit,
        min_chars=args.min_chars,
        domain_tag=args.domain_tag,
        use_full_text=not args.no_full_text,
    )

    if len(records) < 50:
        print(
            "\n[WARN] Fewer than 50 documents were collected. "
            "You may want to add more feeds or lower the min_chars threshold."
        )


if __name__ == "__main__":
    main()