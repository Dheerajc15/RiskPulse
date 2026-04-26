"""
RiskPulse Document Loader
=========================
Scrapes, cleans, validates, and persists FOMC meeting minutes from
federalreserve.gov for use as the RAG corpus.

"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

FED_BASE = "https://www.federalreserve.gov"
FOMC_CALENDAR_URL = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
FOMC_HISTORICAL_URL = f"{FED_BASE}/monetarypolicy/fomchistorical{{year}}.htm"

REQUEST_TIMEOUT = 20
REQUEST_DELAY_SECONDS = 1.0  # be polite to fed servers
USER_AGENT = "RiskPulse-Research/1.0 (academic; contact: github.com/Dheerajc15)"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FOMCDocument:
    """A single scraped FOMC document with metadata."""
    meeting_date: str          # 'YYYY-MM-DD'
    document_type: str         # 'minutes'
    source_url: str
    text: str
    char_count: int
    fetched_at: str            # ISO timestamp


@dataclass
class IngestionReport:
    """Summary stats from a corpus build run."""
    total_meetings_attempted: int
    documents_fetched: int
    documents_rejected: int
    rejection_reasons: dict
    output_dir: str
    manifest_path: str


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

def _http_get(url: str, max_retries: int = 3) -> Optional[str]:
    """GET a URL with retry + polite delay. Returns None on persistent failure."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.text
            if response.status_code == 404:
                logger.debug(f"404 (expected for some dates): {url}")
                return None
            logger.warning(f"HTTP {response.status_code} on {url} (attempt {attempt})")
        except requests.RequestException as e:
            logger.warning(f"Request failed for {url}: {e} (attempt {attempt})")
        time.sleep(REQUEST_DELAY_SECONDS * attempt)
    return None


def discover_meeting_dates(years_back: int = 5) -> List[str]:
    """
    Find FOMC meeting dates by scraping the calendar pages.
    Returns dates as 'YYYYMMDD' strings (the format used in FOMC URLs).

    The current-year calendar is at fomccalendars.htm; older years are at
    fomchistorical{YEAR}.htm.
    """
    current_year = datetime.now().year
    target_years = list(range(current_year - years_back, current_year + 1))
    discovered: List[str] = []

    # Current calendar covers current + recent past
    html = _http_get(FOMC_CALENDAR_URL)
    if html:
        discovered.extend(_extract_minutes_dates_from_html(html))

    # Historical pages for older years
    for year in target_years:
        if year >= current_year - 1:
            continue  # already covered by current calendar
        url = FOMC_HISTORICAL_URL.format(year=year)
        html = _http_get(url)
        if html:
            discovered.extend(_extract_minutes_dates_from_html(html))
        time.sleep(REQUEST_DELAY_SECONDS)

    # Filter to target years and dedupe
    valid = []
    for d in set(discovered):
        try:
            year = int(d[:4])
            if year in target_years:
                valid.append(d)
        except (ValueError, IndexError):
            continue

    return sorted(valid)


def _extract_minutes_dates_from_html(html: str) -> List[str]:
    """
    Parse calendar HTML for hyperlinks to fomcminutesYYYYMMDD.htm.
    Returns YYYYMMDD strings.
    """
    soup = BeautifulSoup(html, "lxml")
    pattern = re.compile(r"fomcminutes(\d{8})\.htm", re.IGNORECASE)
    dates = set()
    for link in soup.find_all("a", href=True):
        match = pattern.search(link["href"])
        if match:
            dates.add(match.group(1))
    return sorted(dates)


# ---------------------------------------------------------------------------
# Fetch + clean a single document
# ---------------------------------------------------------------------------

def fetch_minutes_for_date(date_str: str) -> Optional[FOMCDocument]:
    """
    date_str: 'YYYYMMDD' (FOMC URL format)
    Returns a FOMCDocument, or None if fetch/parse failed.
    """
    url = f"{FED_BASE}/monetarypolicy/fomcminutes{date_str}.htm"
    html = _http_get(url)
    if not html:
        logger.info(f"No minutes available at {url}")
        return None

    text = _clean_html_to_text(html)
    if not text:
        return None

    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return FOMCDocument(
        meeting_date=formatted_date,
        document_type="minutes",
        source_url=url,
        text=text,
        char_count=len(text),
        fetched_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


def _clean_html_to_text(html: str) -> str:
    """
    Extract the article body from FOMC minutes HTML.
    Strips nav, footer, scripts, styles, and other boilerplate.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove obvious noise
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    # FOMC minutes typically live inside <div id="article"> or <div id="content">
    article = soup.find(id="article") or soup.find(id="content") or soup.find("main")
    if not article:
        article = soup.body or soup

    # Drop social/share/breadcrumb fragments by class hint
    for el in article.find_all(class_=re.compile(r"(share|social|breadcrumb|skip)", re.I)):
        el.decompose()

    raw = article.get_text(separator="\n")
    return _normalize_whitespace(raw)


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace, strip, drop very short lines (likely junk)."""
    lines = [ln.strip() for ln in text.splitlines()]
    # Keep lines with substantive content; drop one-word navigation residue
    lines = [ln for ln in lines if len(ln) >= 3]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------------

MIN_CHAR_COUNT = 5_000        # FOMC minutes are always >> this
MAX_BOILERPLATE_RATIO = 0.40  # boilerplate phrases / total length
BOILERPLATE_PHRASES = [
    "skip to main content",
    "federal reserve system",
    "back to top",
    "last update",
    "javascript is disabled",
]


def validate_document(doc: FOMCDocument) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_valid, rejection_reason).
    """
    if doc.char_count < MIN_CHAR_COUNT:
        return False, f"too_short ({doc.char_count} < {MIN_CHAR_COUNT})"

    lower = doc.text.lower()
    boilerplate_chars = sum(lower.count(p) * len(p) for p in BOILERPLATE_PHRASES)
    if boilerplate_chars / max(doc.char_count, 1) > MAX_BOILERPLATE_RATIO:
        return False, "high_boilerplate_ratio"

    if "fomc" not in lower and "federal open market committee" not in lower:
        return False, "missing_fomc_marker"

    return True, None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_document(doc: FOMCDocument, output_dir: Path) -> Path:
    """Write document text to {output_dir}/{date}_minutes.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{doc.meeting_date}_minutes.txt"
    path = output_dir / filename
    path.write_text(doc.text, encoding="utf-8")
    return path


def write_manifest(docs: List[FOMCDocument], manifest_path: Path) -> None:
    """Write a JSON manifest of all corpus docs for the README and downstream tooling."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "document_count": len(docs),
        "total_char_count": sum(d.char_count for d in docs),
        "date_range": {
            "earliest": min((d.meeting_date for d in docs), default=None),
            "latest": max((d.meeting_date for d in docs), default=None),
        },
        "documents": [asdict(d) | {"text": None} for d in docs],  # exclude full text
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_fomc_corpus(
    years_back: int = 5,
    output_dir: Path = Path("data/documents/fomc"),
    manifest_path: Path = Path("data/documents/corpus_manifest.json"),
) -> IngestionReport:
    """
    End-to-end: discover meeting dates, fetch each minutes page, clean,
    validate, save to disk, and write a manifest.
    """
    logger.info(f"Discovering FOMC meeting dates for last {years_back} years...")
    dates = discover_meeting_dates(years_back=years_back)
    logger.info(f"Discovered {len(dates)} candidate meeting dates")

    successes: List[FOMCDocument] = []
    rejections: dict = {}

    for date_str in dates:
        doc = fetch_minutes_for_date(date_str)
        time.sleep(REQUEST_DELAY_SECONDS)

        if doc is None:
            rejections["fetch_failed"] = rejections.get("fetch_failed", 0) + 1
            continue

        is_valid, reason = validate_document(doc)
        if not is_valid:
            rejections[reason] = rejections.get(reason, 0) + 1
            logger.info(f"Rejected {doc.meeting_date}: {reason}")
            continue

        save_document(doc, output_dir)
        successes.append(doc)
        logger.info(f"Saved {doc.meeting_date} ({doc.char_count:,} chars)")

    write_manifest(successes, manifest_path)

    return IngestionReport(
        total_meetings_attempted=len(dates),
        documents_fetched=len(successes),
        documents_rejected=sum(rejections.values()),
        rejection_reasons=rejections,
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
    )