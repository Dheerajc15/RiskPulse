"""
Sanity tests for the FOMC document loader.
These do NOT hit the network — they exercise parsing logic on fixture HTML.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.document_loader import (
    _clean_html_to_text,
    _extract_minutes_dates_from_html,
    _normalize_whitespace,
    validate_document,
    FOMCDocument,
)


def test_extracts_minutes_dates_from_calendar_html():
    sample = """
    <html><body>
      <a href="/monetarypolicy/fomcminutes20240131.htm">Jan 2024 minutes</a>
      <a href="/monetarypolicy/fomcminutes20240320.htm">Mar 2024 minutes</a>
      <a href="/some/other/page.htm">unrelated</a>
    </body></html>
    """
    dates = _extract_minutes_dates_from_html(sample)
    assert dates == ["20240131", "20240320"]
    print("✓ extract_minutes_dates_from_html")


def test_clean_html_strips_nav_and_scripts():
    html = """
    <html><body>
      <nav>NAV STUFF</nav>
      <script>console.log('x')</script>
      <div id="article">
        <p>The Federal Open Market Committee met on January 31.</p>
        <p>Members agreed that the federal funds rate target range...</p>
      </div>
      <footer>FOOTER STUFF</footer>
    </body></html>
    """
    text = _clean_html_to_text(html)
    assert "NAV STUFF" not in text
    assert "FOOTER STUFF" not in text
    assert "console.log" not in text
    assert "Federal Open Market Committee" in text
    print("✓ clean_html_strips_nav_and_scripts")


def test_normalize_whitespace_drops_short_lines():
    raw = "Real sentence here.\n  \nx\n\nAnother real sentence."
    result = _normalize_whitespace(raw)
    assert "Real sentence here." in result
    assert "Another real sentence." in result
    assert "\nx\n" not in result
    print("✓ normalize_whitespace_drops_short_lines")


def test_validate_rejects_short_doc():
    doc = FOMCDocument(
        meeting_date="2024-01-31",
        document_type="minutes",
        source_url="http://example.com",
        text="too short",
        char_count=9,
        fetched_at="2024-01-01T00:00:00Z",
    )
    is_valid, reason = validate_document(doc)
    assert not is_valid
    assert "too_short" in reason
    print("✓ validate_rejects_short_doc")


def test_validate_rejects_missing_fomc_marker():
    doc = FOMCDocument(
        meeting_date="2024-01-31",
        document_type="minutes",
        source_url="http://example.com",
        text="lorem ipsum " * 1000,  # long enough but no FOMC marker
        char_count=12_000,
        fetched_at="2024-01-01T00:00:00Z",
    )
    is_valid, reason = validate_document(doc)
    assert not is_valid
    assert reason == "missing_fomc_marker"
    print("✓ validate_rejects_missing_fomc_marker")


if __name__ == "__main__":
    test_extracts_minutes_dates_from_calendar_html()
    test_clean_html_strips_nav_and_scripts()
    test_normalize_whitespace_drops_short_lines()
    test_validate_rejects_short_doc()
    test_validate_rejects_missing_fomc_marker()
    print("\n✅ All document loader tests passed")