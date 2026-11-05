"""
Tests for Presentation HTML & Asset Integrity
Ensures that all slides, stylesheets, snapshot images, and Reveal.js structures
remain valid and non-broken during local editing and development.
"""

import os
import re
from pathlib import Path
import pytest

PRESENTATION_DIR = Path(__file__).resolve().parent.parent / "presentation"
INDEX_HTML = PRESENTATION_DIR / "index.html"
THEME_CSS = PRESENTATION_DIR / "css" / "theme.css"


def test_presentation_core_files_exist():
    """Verify that essential presentation files are present on disk."""
    assert PRESENTATION_DIR.exists(), f"Presentation directory not found: {PRESENTATION_DIR}"
    assert INDEX_HTML.exists(), f"index.html not found: {INDEX_HTML}"
    assert THEME_CSS.exists(), f"theme.css not found: {THEME_CSS}"
    assert os.path.getsize(INDEX_HTML) > 1000, "index.html appears truncated or empty"
    assert os.path.getsize(THEME_CSS) > 100, "theme.css appears truncated or empty"


def test_presentation_reveal_structure():
    """Verify Reveal.js DOM hierarchy and slide sections exist."""
    content = INDEX_HTML.read_text(encoding="utf-8")
    assert '<div class="reveal">' in content, "Missing <div class='reveal'> root container"
    assert '<div class="slides">' in content, "Missing <div class='slides'> slide container"
    
    sections = re.findall(r'<section\b[^>]*>', content)
    assert len(sections) >= 15, f"Expected at least 15 slide sections, found {len(sections)}"

    # Check for speaker notes support
    notes = re.findall(r'<aside\s+class=[\"\']notes[\"\']>', content)
    assert len(notes) > 0, "No speaker notes (<aside class='notes'>) found in presentation"


def test_all_local_images_exist_and_non_empty():
    """Verify that every local image referenced in index.html exists and is readable."""
    content = INDEX_HTML.read_text(encoding="utf-8")
    img_matches = re.findall(r'<img[^>]+src=[\"\']([^\"\']+)[\"\']', content)
    
    assert len(img_matches) > 0, "No images found in index.html"
    
    for src in img_matches:
        if src.startswith(("http://", "https://", "data:")):
            continue
        
        img_path = PRESENTATION_DIR / src
        assert img_path.exists(), f"Broken image path in index.html: '{src}' does not exist at {img_path}"
        assert img_path.is_file(), f"Image path is not a file: {img_path}"
        size = img_path.stat().st_size
        assert size > 0, f"Image file is 0 bytes: {img_path}"


def test_all_local_stylesheets_exist():
    """Verify that every local stylesheet referenced in index.html exists."""
    content = INDEX_HTML.read_text(encoding="utf-8")
    link_matches = re.findall(r'<link[^>]+href=[\"\']([^\"\']+)[\"\']', content)
    
    local_css_found = False
    for href in link_matches:
        if href.startswith(("http://", "https://")):
            continue
        if href.endswith(".css"):
            css_path = PRESENTATION_DIR / href
            assert css_path.exists(), f"Broken stylesheet path in index.html: '{href}' does not exist at {css_path}"
            assert css_path.stat().st_size > 0, f"Stylesheet is 0 bytes: {css_path}"
            local_css_found = True

    assert local_css_found, "No local stylesheet was found referenced in index.html"


def test_snapshots_png_format_validity():
    """Verify that referenced PNG snapshots contain valid PNG magic headers."""
    content = INDEX_HTML.read_text(encoding="utf-8")
    img_matches = re.findall(r'<img[^>]+src=[\"\']([^\"\']+)[\"\']', content)
    
    png_magic = b"\x89PNG\r\n\x1a\n"
    tested_snapshots = 0

    for src in img_matches:
        if src.startswith("images/snapshots/") and src.endswith(".png"):
            img_path = PRESENTATION_DIR / src
            assert img_path.exists(), f"Snapshot not found: {img_path}"
            with open(img_path, "rb") as f:
                header = f.read(8)
            assert header == png_magic, f"Snapshot {src} has invalid PNG header bytes: {header!r}"
            tested_snapshots += 1

    assert tested_snapshots >= 4, f"Expected at least 4 profiler snapshots tested, found {tested_snapshots}"


def test_theme_css_braces_balance():
    """Verify that presentation/css/theme.css has balanced curly braces (no syntax cut-offs)."""
    css_content = THEME_CSS.read_text(encoding="utf-8")
    open_braces = css_content.count("{")
    close_braces = css_content.count("}")
    assert open_braces == close_braces, (
        f"Unbalanced braces in theme.css: {open_braces} open '{{' vs {close_braces} close '}}'"
    )
