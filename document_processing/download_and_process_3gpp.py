#!/usr/bin/env python3
"""
3GPP Document Download and Processing Pipeline

Downloads 3GPP Release 18 specifications, processes them to JSON,
and initializes the Neo4j Knowledge Graph.

Usage:
    python download_and_process_3gpp.py download     # Download documents only
    python download_and_process_3gpp.py extract      # Extract ZIP files
    python download_and_process_3gpp.py process      # Process downloaded docs to JSON
    python download_and_process_3gpp.py process-local --input /path/to/docs  # Process local docs
    python download_and_process_3gpp.py init-kg      # Initialize Neo4j from JSON
    python download_and_process_3gpp.py all          # Do everything
    python download_and_process_3gpp.py status       # Check current status
    python document_processing/download_and_process_3gpp.py process-local \
  --input  document_processing/data/rel18_extracted \
  --output 3GPP_JSON_DOC/processed_json_v6
"""

import os
import sys
import re
import json
import zipfile
import subprocess
import argparse
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_status(message: str, status: str = "info"):
    """Print colored status message"""
    colors = {
        "ok": Colors.GREEN,
        "error": Colors.RED,
        "warn": Colors.YELLOW,
        "info": Colors.BLUE,
        "progress": Colors.CYAN
    }
    symbols = {
        "ok": "✓",
        "error": "✗",
        "warn": "!",
        "info": "→",
        "progress": "◐"
    }
    color = colors.get(status, Colors.RESET)
    symbol = symbols.get(status, "→")
    print(f"{color}{symbol}{Colors.RESET} {message}")


def print_header(title: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Pipeline configuration"""
    # Directories
    download_dir: Path = None
    data_dir: Path = None
    output_dir: Path = None

    # Download settings
    release: int = 18
    series: List[str] = None  # None = all series, or ['23', '29', '38'] for specific

    # Neo4j settings
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    def __post_init__(self):
        # Set default directories
        if self.download_dir is None:
            self.download_dir = SCRIPT_DIR / "data" / f"rel{self.release}_download"
        if self.data_dir is None:
            self.data_dir = SCRIPT_DIR / "data" / f"rel{self.release}_extracted"
        if self.output_dir is None:
            self.output_dir = PROJECT_ROOT / "3GPP_JSON_DOC" / "processed_json_v6"

        if self.series is None:
            # 5G Core series: 23.xxx (System Architecture), 29.xxx (APIs),
            # 32.xxx (Charging), 38.xxx (Radio)
            self.series = ['23', '24', '26', '27', '28', '29', '32', '33', '36', '37', '38']


# =============================================================================
# Downloader: Download 3GPP documents
# =============================================================================

class ThreeGPPDownloader:
    """Download 3GPP specifications using download_3gpp package"""

    def __init__(self, config: Config):
        self.config = config
        self.download_dir = config.download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def check_download_3gpp_installed(self) -> bool:
        """Check if download_3gpp package is installed"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import download_3gpp"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False

    def install_download_3gpp(self) -> bool:
        """Install download_3gpp package"""
        print_status("Installing download_3gpp package...", "info")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "download_3gpp"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print_status("download_3gpp installed successfully", "ok")
                return True
            else:
                print_status(f"Failed to install: {result.stderr}", "error")
                return False
        except Exception as e:
            print_status(f"Error installing package: {e}", "error")
            return False

    def download_release(self, release: int = None, series: List[str] = None) -> bool:
        """
        Download 3GPP release using download_3gpp package.

        Args:
            release: Release number (e.g., 18 for Rel-18)
            series: List of series to download (e.g., ['23', '29'])
        """
        release = release or self.config.release
        series = series or self.config.series

        print_header(f"Downloading 3GPP Release {release}")

        # Check/install download_3gpp
        if not self.check_download_3gpp_installed():
            if not self.install_download_3gpp():
                print_status("Failed to install download_3gpp, trying alternative method...", "warn")
                return self.download_via_requests(release, series)

        # Create release directory
        release_dir = self.download_dir / f"Rel-{release}"
        release_dir.mkdir(parents=True, exist_ok=True)

        try:
            if series:
                # Download specific series
                for s in series:
                    print_status(f"Downloading series {s}...", "progress")

                    # Use download_3gpp with series filter
                    cmd = ["download_3gpp", f"--rel={release}", f"--series={s}"]

                    result = subprocess.run(
                        cmd,
                        cwd=str(release_dir),
                        capture_output=True,
                        text=True,
                        timeout=3600  # 1 hour timeout per series
                    )

                    if result.returncode != 0 and result.stderr:
                        print_status(f"Warning: Series {s}: {result.stderr[:200]}", "warn")
                    else:
                        print_status(f"Series {s} downloaded", "ok")
            else:
                # Download all
                print_status(f"Downloading all series for Release {release}...", "progress")
                print_status("This may take a long time (several GB of data)...", "info")

                cmd = ["download_3gpp", f"--rel={release}"]

                result = subprocess.run(
                    cmd,
                    cwd=str(release_dir),
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )

                if result.returncode != 0 and result.stderr:
                    print_status(f"Warning: {result.stderr[:500]}", "warn")

            # Count downloaded files
            zip_files = list(release_dir.rglob("*.zip"))
            docx_files = list(release_dir.rglob("*.docx"))

            print_status(f"Downloaded {len(zip_files)} ZIP files, {len(docx_files)} DOCX files", "ok")
            return len(zip_files) > 0 or len(docx_files) > 0

        except subprocess.TimeoutExpired:
            print_status("Download timed out. Try downloading specific series.", "error")
            return False
        except Exception as e:
            print_status(f"Download error: {e}", "error")
            return False

    def download_via_requests(self, release: int = 18, series: List[str] = None) -> bool:
        """
        Alternative download method using direct HTTP requests.
        Fallback if download_3gpp doesn't work.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            print_status("requests/beautifulsoup4 not installed", "error")
            return False

        print_header(f"Downloading 3GPP Release {release} (Direct HTTP)")

        # 3GPP FTP archive URL
        FTP_BASE = "https://www.3gpp.org/ftp/Specs/archive"

        release_dir = self.download_dir / f"Rel-{release}"
        release_dir.mkdir(parents=True, exist_ok=True)

        series_to_download = series or self.config.series

        downloaded = 0
        failed = 0

        # Release letter mapping: 18 -> 'i', 17 -> 'h', etc.
        release_letter = chr(ord('a') + release - 10)  # Rel-10='a', Rel-18='i'

        for s in series_to_download:
            series_url = f"{FTP_BASE}/{s}_series/"
            series_dir = release_dir / f"{s}_series"
            series_dir.mkdir(exist_ok=True)

            print_status(f"Fetching series {s} from {series_url}...", "progress")

            try:
                response = requests.get(series_url, timeout=30)
                if response.status_code != 200:
                    print_status(f"Cannot access series {s} (HTTP {response.status_code})", "warn")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')

                # Find ZIP files for this release
                for link in soup.find_all('a'):
                    href = link.get('href', '')

                    # Match pattern: 23501-i80.zip (release letter + version)
                    if href.endswith('.zip') and f'-{release_letter}' in href.lower():
                        file_url = f"{series_url}{href}"
                        file_path = series_dir / href

                        if file_path.exists():
                            print_status(f"  Skipping {href} (exists)", "info")
                            continue

                        try:
                            print_status(f"  Downloading {href}...", "info")
                            file_response = requests.get(file_url, timeout=120, stream=True)

                            if file_response.status_code == 200:
                                with open(file_path, 'wb') as f:
                                    for chunk in file_response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                downloaded += 1
                            else:
                                print_status(f"  Failed: HTTP {file_response.status_code}", "warn")
                                failed += 1

                        except requests.exceptions.Timeout:
                            print_status(f"  Timeout: {href}", "warn")
                            failed += 1
                        except Exception as e:
                            print_status(f"  Failed: {href} - {e}", "warn")
                            failed += 1

            except Exception as e:
                print_status(f"Error fetching series {s}: {e}", "warn")

        print_status(f"Downloaded {downloaded} files, {failed} failed", "ok" if failed == 0 else "warn")
        return downloaded > 0


# =============================================================================
# Extractor: Extract ZIP files
# =============================================================================

class ThreeGPPExtractor:
    """Extract and organize downloaded 3GPP ZIP files"""

    def __init__(self, config: Config):
        self.config = config
        self.data_dir = config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Pattern to identify main body files
        self.main_body_pattern = re.compile(r'main.*body|_1_|s\d+_s\d+', re.IGNORECASE)
        self.exclude_pattern = re.compile(r'cover|annex|history|amendment', re.IGNORECASE)

    def extract_all(self, source_dir: Path = None) -> int:
        """
        Extract all ZIP files to organized directory structure.

        Returns:
            Number of DOCX files extracted
        """
        print_header("Extracting ZIP Files")

        source = source_dir or self.config.download_dir

        if not source.exists():
            print_status(f"Source directory not found: {source}", "error")
            return 0

        # Find all ZIP files
        zip_files = list(source.rglob("*.zip"))
        print_status(f"Found {len(zip_files)} ZIP files", "info")

        extracted_count = 0

        for zip_path in zip_files:
            try:
                # Determine series from path or filename
                series = self._get_series_from_path(zip_path)
                if not series:
                    continue

                # Create series directory
                series_dir = self.data_dir / f"{series}_series"
                series_dir.mkdir(exist_ok=True)

                # Extract to spec folder
                spec_name = zip_path.stem
                extract_dir = series_dir / spec_name
                extract_dir.mkdir(exist_ok=True)

                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for file_info in zf.filelist:
                        filename = file_info.filename

                        # Skip directories and temp files
                        if filename.endswith('/') or filename.startswith('~') or '/' in filename:
                            # Extract nested file
                            base_name = Path(filename).name
                            if base_name.startswith('~'):
                                continue
                        else:
                            base_name = filename

                        # Extract DOCX files (main body only)
                        if base_name.endswith('.docx'):
                            if self._is_main_body(base_name):
                                # Extract to flat structure
                                target_path = extract_dir / base_name
                                with zf.open(file_info) as src:
                                    with open(target_path, 'wb') as dst:
                                        dst.write(src.read())
                                extracted_count += 1

            except zipfile.BadZipFile:
                print_status(f"Bad ZIP file: {zip_path.name}", "warn")
            except Exception as e:
                print_status(f"Error extracting {zip_path.name}: {e}", "warn")

        # Also copy direct DOCX files
        for docx_path in source.rglob("*.docx"):
            if docx_path.name.startswith('~'):
                continue

            if self._is_main_body(docx_path.name):
                series = self._get_series_from_path(docx_path)
                if series:
                    series_dir = self.data_dir / f"{series}_series"
                    series_dir.mkdir(exist_ok=True)

                    spec_name = docx_path.stem
                    target_dir = series_dir / spec_name
                    target_dir.mkdir(exist_ok=True)

                    target_path = target_dir / docx_path.name
                    if not target_path.exists():
                        shutil.copy(docx_path, target_path)
                        extracted_count += 1

        print_status(f"Extracted {extracted_count} main body DOCX files", "ok")
        return extracted_count

    def _get_series_from_path(self, path: Path) -> Optional[str]:
        """Extract series number from path"""
        # Try folder name first
        for part in path.parts:
            match = re.search(r'(\d{2})_series', part)
            if match:
                return match.group(1)

        # Try filename
        match = re.search(r'^(\d{2})\d{3}', path.name)
        if match:
            return match.group(1)

        return None

    def _is_main_body(self, filename: str) -> bool:
        """Check if file is a main body document"""
        # Exclude known non-content files
        if self.exclude_pattern.search(filename):
            return False

        # Exclude revision mark files (-rm) and macOS hidden files
        if '-rm.' in filename.lower() or filename.startswith('._'):
            return False

        # Main body file or matches pattern
        if self.main_body_pattern.search(filename):
            return True

        # Match various 3GPP spec filename formats:
        # - 23501-i80.docx (standard: 5 digits + letter + 2 digits)
        # - 23501-ib0.docx (with 'b' suffix)
        # - 38101-1-ib1.docx (multi-part specs like 38.101-1)
        # - 38101-2-ib0.docx (multi-part specs like 38.101-2)
        # - 24283-010.docx (3-digit version number)
        # - 31121-i40_s01-s03_5.docx (section splits)
        # - 38101-1-ib1_s07-XX.docx (multi-part section splits)
        # - 36718-02-01-i00.docx (multi-part with subpart like 36.718-02-01)
        # - 26927-070-cl.docx (clean version suffix)
        # Pattern: 5 digits + optional parts + version + optional suffixes
        if re.match(r'^\d{5}(-\d+)*-[a-z]{0,2}\d{1,3}(-cl)?(_s[\w\-\.]+)?\.docx$', filename, re.IGNORECASE):
            return True

        return False


# =============================================================================
# Processor: Process DOCX to JSON
# =============================================================================

# --- 3GPP heading grammar -----------------------------------------------------
# Module-level (not ThreeGPPParser attributes) because kg_builder/split_tail_chunks.py
# repairs an already-built KG with the SAME grammar — one source of truth.
#
# Why the alternate form exists: the original `SECTION_HEADING_RE` only accepts a
# purely numeric clause id, which silently drops ~10-37% of every document's
# headings. Measured over 6 specs: 258/708 headings in TS 33.501, 117/1314 in
# TS 23.502, 118/1318 in TS 24.501. Two families are missed —
#   (a) letter-suffixed clauses, which 3GPP uses for anything inserted between two
#       existing clauses: 5.8a, 7A, 7A.2.1, 13.4.1.1A, 4.11.0a.2a;
#   (b) annex sub-clauses: A.1, A.7.0, AA.2.
# An unrecognised heading is not skipped — `parse_sections` appends it (and all
# its body text) to whatever section is open, so the preceding clause absorbs it.
# At the end of a document that cascades: every Annex plus the change history
# lands in the last numeric clause, producing chunks up to 1.5 MB whose title has
# nothing to do with most of their content (ts_33_501_16.6.3 = 262 KB titled
# "Subscription/unsubscription of NSACF notification service").
#
# The alternate form REQUIRES a tab between id and title (3GPP heading styles
# always emit one) so prose headings like "5G System Architecture" cannot be
# mistaken for clause "5G". The legacy pattern is tried first so every existing
# numeric chunk_id stays byte-identical.
SECTION_HEADING_RE = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
SECTION_HEADING_ALT_RE = re.compile(
    r'^(\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*|[A-Z]{1,2}(?:\.\d+[A-Za-z]?)*)[ ]*\t\s*(.+)$')

# Annex roots are NOT heading elements in the mammoth HTML — they come through as
# <p> ("Annex A (normative):\tKey derivation functions"). The same lines also
# appear in the table of contents, where they end with a tab + page number, hence
# TOC_LINE_RE.
# Two forms occur: "Annex A (normative): Title" (most specs) and a bare
# "Annex A: Title" (e.g. TS 22.858). The colon is mandatory when the
# normative/informative marker is absent, so prose like "Annex A contains the key
# derivation functions" is not mistaken for a heading.
ANNEX_HEADING_RE = re.compile(
    r'^Annex\s+([A-Z]{1,2})\s*(?:\((?:normative|informative)\)\s*:?|:)\s*(.*)$',
    re.IGNORECASE)
TOC_LINE_RE = re.compile(r'\t\s*\d+\s*$')

_HEADING_TAGS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')

# Everything from the change-history annex on is CR bookkeeping (date / meeting /
# TDoc / CR number tables) — 66 KB of it in TS 33.501 alone, and it answers no
# question. Parsing stops there.
CHANGE_HISTORY_RE = re.compile(r'^change\s+history$', re.IGNORECASE)

# Hard ceiling on a single chunk. A clause longer than this is split into
# `<id>p2`, `<id>p3`… parts at paragraph boundaries: the cross-encoder only reads
# the first 512 tokens of a chunk and the answer prompt only fits ~4 KB of one, so
# an unsplit 40 KB+ clause is unrankable and mostly unreadable regardless of the
# heading fixes above. Depth of the section_id is preserved (suffix on the last
# component) so sibling/parent prefix logic downstream is unaffected.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "40000"))

# Worker processes for the chunking pool. Capped well below cpu_count because the
# limit is RAM, not cores: a typical worker peaks ~1.2 GB, but a single huge DOCX
# (38871-i20.docx is 42 MB) blown up into HTML + a BeautifulSoup tree was measured
# at 5.0 GB RSS. find_docx_files() returns a SORTED list, so the few giant 38-series
# files arrive as neighbours and can land in several workers at once — 8 × 5 GB
# would exceed a 31 GB box. On OOM the pool dies and process_all() restarts it with
# half the workers. Measured on 33_series (102 files): 1 worker 124 s / 100% CPU,
# 8 workers 31.6 s / 450% CPU (3.9x).
DEFAULT_WORKERS = int(os.getenv("CHUNK_WORKERS", "0")) or min(6, max(1, (os.cpu_count() or 4) - 2))

# "3GPP TS 38.331", "TR 21.905". The leading \b matters under IGNORECASE — see the
# note on external_patterns. `\d+\.\d+` deliberately stops before a "-1" part
# suffix; spec_id normalisation handles multi-part specs elsewhere.
SPEC_MENTION_RE = re.compile(r'(?:3GPP\s+)?\b(?:TS|TR)\s+(\d{2}\.\d+)(?![\dxX])', re.IGNORECASE)


def match_section_heading(text: str):
    """(section_id, title) for a 3GPP clause heading, else None. Legacy numeric
    form first so existing chunk_ids never shift."""
    m = SECTION_HEADING_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    m = SECTION_HEADING_ALT_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    return None


def match_annex_heading(text: str):
    """(annex_id, title) for an annex root paragraph, else None. TOC copies of
    the same line are rejected."""
    if TOC_LINE_RE.search(text):
        return None
    m = ANNEX_HEADING_RE.match(text)
    if not m:
        return None
    title = m.group(2).strip() or f"Annex {m.group(1)}"
    return m.group(1), title


def is_change_history(title: str) -> bool:
    return bool(CHANGE_HISTORY_RE.match((title or "").strip()))


@dataclass
class DocumentMetaData:
    specification_id: str
    version: str
    title: str
    file_path: str
    release: str = ""


@dataclass
class SectionStructure:
    section_id: str
    title: str
    level: int
    parent_section: Optional[str] = None
    content: str = ""
    tables: List[Dict] = None


@dataclass
class ProcessedChunk:
    chunk_id: str
    section_id: str
    section_title: str
    content: str
    chunk_type: str
    cross_references: Dict
    content_metadata: Dict = None
    tables: List[Dict] = None
    # Titles of the ancestor clauses, outermost first. A heading-only parent
    # ("6.43 Third-party provided policies", whose body starts at 6.43.1) has no
    # content of its own, so it never becomes a chunk and its title — the only
    # place that clause's topic is named — used to be lost entirely. Measured on
    # the live KG: 28,115 chunks (12%) carry a generic title ("Requirements",
    # "General", "Description") and 83% of those have no parent chunk to
    # disambiguate them, which is why TS 22.261 alone has dozens of retrievable
    # clauses all called "Requirements".
    section_path: List[str] = None


class ThreeGPPParser:
    """Parse 3GPP DOCX documents into structured JSON"""

    def __init__(self):
        # Kept for backward compat; parse_sections now goes through
        # match_section_heading() so letter-suffixed and annex clauses are caught.
        self.section_pattern = SECTION_HEADING_RE

        # Reference patterns
        self.clause_pattern = re.compile(r'(?:clause|section)\s+(\d+(?:\.\d+)*)', re.IGNORECASE)
        self.table_pattern = re.compile(r'table\s+(\d+(?:\.\d+)*(?:-\d+)?)', re.IGNORECASE)
        self.figure_pattern = re.compile(r'figure\s+(\d+(?:\.\d+)*(?:-\d+)?)', re.IGNORECASE)

        # External reference patterns — use named groups so spec_num is unambiguous
        # regardless of token order (TS-first vs clause-first phrasing).
        # \b after \d+\.\d+ prevents greedy backtracking from truncating spec numbers
        # (e.g., "38.331," → "38.33" + leftover "1," to satisfy negative lookahead).
        # NOTE the leading \b on every (?:TS|TR): without it, and with IGNORECASE,
        # the "ts" ENDING OF AN ORDINARY WORD matches — "No requiremenTS\n5.2.2.10"
        # yielded target ts_5_2 and "independent tesTS 8.1" yielded ts_8_1. That was
        # ~250 clause numbers mis-read as spec numbers on v5.
        self.external_patterns = [
            re.compile(r'(?:3GPP\s+)?\b(?:TS|TR)\s+(?P<spec>\d{2}\.\d+)(?![\dxX])\b(?:,?\s+(?:clause|section)\s+(?P<clause>\d+(?:\.\d+)*))', re.IGNORECASE),
            re.compile(r'(?:clause|section)\s+(?P<clause>\d+(?:\.\d+)*)\s+of\s+(?:3GPP\s+)?\b(?:TS|TR)\s+(?P<spec>\d{2}\.\d+)(?![\dxX])\b', re.IGNORECASE),
            re.compile(r'(?:3GPP\s+)?\b(?:TS|TR)\s+(?P<spec>\d{2}\.\d+)(?![\dxX])\b(?!\s*(?:,?\s*(?:clause|section|table|figure)))', re.IGNORECASE)
        ]

    def extract_metadata(self, document_path: str) -> DocumentMetaData:
        """Extract metadata from document"""
        import mammoth
        from bs4 import BeautifulSoup

        with open(document_path, "rb") as docx_file:
            # Skip images: pipeline is text-only. Avoids crash when a DOCX has a
            # broken image reference (e.g. "word/NULL" in 26142-i00.docx).
            result = mammoth.convert_to_html(
                docx_file,
                convert_image=mammoth.images.img_element(lambda image: {}),
            )

        soup = BeautifulSoup(result.value, 'html.parser')
        text = soup.get_text()[:5000]  # First 5000 chars

        # Find TS number and version - handle multi-part specs like TS 38.101-1
        ts_version_match = re.search(r'3GPP\s+TS\s+(\d+\.\d+(?:-\d+)?)\s+V(\d+\.\d+\.\d+)', text)

        if ts_version_match:
            ts_number = ts_version_match.group(1)
            version = ts_version_match.group(2)
            # Replace dots with underscores for spec_id, keep hyphens
            specification_id = f"ts_{ts_number.replace('.', '_')}"
            title = f"3GPP TS {ts_number}"

            # Extract release from version (e.g., 18.9.0 -> Rel-18)
            release = f"Rel-{version.split('.')[0]}"
        else:
            # Fallback: extract from filename
            # Handle formats: 23501-i80, 38101-1-ib1
            filename = Path(document_path).stem
            # Match: 5 digits + optional part number
            filename_match = re.search(r'(\d{2})(\d{3})(?:-(\d+))?', filename)
            if filename_match:
                series = filename_match.group(1)
                spec_num = filename_match.group(2)
                part = filename_match.group(3)
                if part:
                    ts_number = f"{series}.{spec_num}-{part}"
                    specification_id = f"ts_{series}_{spec_num}-{part}"
                else:
                    ts_number = f"{series}.{spec_num}"
                    # Use underscore between series and number for consistency with Path A
                    specification_id = f"ts_{series}_{spec_num}"
                title = f"3GPP TS {ts_number}"
            else:
                specification_id = f"ts_{filename}"
                title = f"3GPP {filename}"
            version = "Unknown"
            release = "Unknown"

        return DocumentMetaData(
            specification_id=specification_id,
            version=version,
            title=title,
            file_path=str(document_path),
            release=release
        )

    def parse_sections(self, document_path: str) -> List[SectionStructure]:
        """Parse document sections with content"""
        import mammoth
        from bs4 import BeautifulSoup

        with open(document_path, "rb") as docx_file:
            # Skip image data URI generation — pure text pipeline; avoids crash
            # when a DOCX contains a broken image reference.
            result = mammoth.convert_to_html(
                docx_file,
                convert_image=mammoth.images.img_element(lambda image: {}),
            )

        soup = BeautifulSoup(result.value, 'html.parser')
        sections = []
        current_section = None
        in_content = False

        for element in soup.find_all(list(_HEADING_TAGS) + ['p', 'table']):
            text = element.get_text().strip()

            # Handle tables
            if element.name == 'table':
                if current_section:
                    table_data = self._extract_table(element)
                    if current_section.tables is None:
                        current_section.tables = []
                    current_section.tables.append(table_data)
                continue

            if not text:
                continue

            # Content starts at the first real clause heading. Keying this on h1
            # alone (the original rule) yields ZERO sections for a section-split
            # DOCX whose body opens at h2/h3 — `_process_single` then reports
            # "fail" and writes no JSON at all. Any hN that parses as a clause
            # heading opens the body; a TOC copy of that same line (trailing tab +
            # page number) must not, or the whole table of contents is parsed as
            # sections.
            if not in_content and element.name in _HEADING_TAGS:
                if element.name == 'h1' or (
                    not TOC_LINE_RE.search(text) and match_section_heading(text)
                ):
                    in_content = True
            if not in_content:
                continue

            # Check if it's a section heading
            if element.name in _HEADING_TAGS:
                match = match_section_heading(text)
                if match:
                    section_number, section_title = match
                    if is_change_history(section_title):
                        break
                    level = len(section_number.split('.'))
                    parent = '.'.join(section_number.split('.')[:-1]) if level > 1 else None

                    current_section = SectionStructure(
                        section_id=section_number,
                        title=section_title,
                        level=level,
                        parent_section=parent,
                        content="",
                        tables=[]
                    )
                    sections.append(current_section)
                    continue

            # Annex roots arrive as <p>, not as a heading element (see
            # ANNEX_HEADING_RE). Without this the whole annex block — and every
            # annex after it — is appended to the last numbered clause.
            if element.name == 'p':
                annex = match_annex_heading(text)
                if annex:
                    annex_id, annex_title = annex
                    if is_change_history(annex_title):
                        break  # change history and everything after it is CR bookkeeping
                    current_section = SectionStructure(
                        section_id=annex_id,
                        title=annex_title,
                        level=1,
                        parent_section=None,
                        content="",
                        tables=[]
                    )
                    sections.append(current_section)
                    continue
                if is_change_history(text):
                    break  # specs that title the block without the "Annex X" wrapper

            # Add content to current section
            if current_section:
                if current_section.content:
                    current_section.content += '\n' + text
                else:
                    current_section.content = text

        return sections

    def _extract_table(self, table_element) -> Dict:
        """Extract table data"""
        table_data = {"headers": [], "rows": []}
        rows = table_element.find_all('tr')

        if rows:
            headers = [th.get_text().strip() for th in rows[0].find_all(['th', 'td'])]
            table_data["headers"] = headers

            for row in rows[1:]:
                row_data = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                if row_data:
                    table_data["rows"].append(row_data)

        return table_data

    def extract_cross_references(self, content: str, spec_id: str) -> Dict:
        """Extract cross-references from content"""
        internal_refs = []
        external_refs = []

        # Normalize spec_id for comparison
        spec_num = spec_id.replace('ts_', '')

        # Spec mentions are located over the WHOLE content first, then matched to a
        # nearby reference by span. Slicing a ±50-char window and running the regex
        # inside it truncated any spec number the window cut in half — measured on
        # v5: "Derivation Path: TS 38.33|1" → ts_38_33, "TS 38.50|8-1" → ts_38_50,
        # ~300 bogus targets.
        spec_mentions = [
            (m.start(), m.end(), m.group(1))
            for m in SPEC_MENTION_RE.finditer(content)
        ]

        # Internal references
        for pattern, ref_type in [(self.clause_pattern, 'clause'),
                                  (self.table_pattern, 'table'),
                                  (self.figure_pattern, 'figure')]:
            for match in pattern.finditer(content):
                ref_num = match.group(1)

                # Check if it's an external reference: first spec mention overlapping
                # the ±50-char neighbourhood, taken whole.
                lo, hi = match.start() - 50, match.end() + 50
                target_norm = next(
                    (num.replace('.', '_') for s, e, num in spec_mentions if s < hi and e > lo),
                    None,
                )

                if target_norm and target_norm != spec_num:
                    external_refs.append({
                        "target_spec": f"ts_{target_norm}",
                        "ref_type": ref_type,
                        "ref_id": ref_num,
                        "confidence": 0.9
                    })
                else:
                    internal_refs.append({
                        "ref_type": ref_type,
                        "ref_id": ref_num
                    })

        # Standalone external spec references
        for pattern in self.external_patterns:
            for match in pattern.finditer(content):
                # Pull spec by name — group order varies across patterns
                target_num = match.group('spec')
                # Normalize to underscore form to match Document.spec_id
                target_norm = target_num.replace('.', '_')
                target_spec = f"ts_{target_norm}"
                if target_norm != spec_num:
                    external_refs.append({
                        "target_spec": target_spec,
                        "ref_type": "spec",
                        "ref_id": "",
                        "confidence": 0.8
                    })

        # Deduplicate external refs
        seen = set()
        unique_external = []
        for ref in external_refs:
            key = f"{ref['target_spec']}_{ref['ref_type']}_{ref['ref_id']}"
            if key not in seen:
                seen.add(key)
                unique_external.append(ref)

        return {
            "internal": internal_refs,
            "external": unique_external
        }

    def classify_content_type(self, title: str, content: str) -> str:
        """Classify content type"""
        title_lower = title.lower()
        content_lower = content.lower()[:500]

        if any(term in title_lower for term in ['definition', 'overview', 'general', 'scope']):
            return 'definition'
        elif any(term in title_lower for term in ['abbreviation']):
            return 'abbreviation'
        elif any(term in title_lower for term in ['procedure', 'flow', 'process', 'registration', 'handover']):
            return 'procedure'
        elif any(term in title_lower for term in ['parameter', 'identifier', 'ie', 'element']):
            return 'parameter'
        elif any(term in title_lower for term in ['reference', 'normative']):
            return 'reference'
        elif 'shall' in content_lower:
            return 'requirement'
        elif any(term in title_lower for term in ['interface', 'api', 'service']):
            return 'interface'
        else:
            return 'general'

    def compute_complexity(self, content: str) -> float:
        """Compute content complexity score"""
        word_count = len(content.split())

        # Count technical patterns
        tech_patterns = [
            r'\b[A-Z]{2,5}\b',  # Abbreviations
            r'clause\s+\d+',
            r'table\s+\d+',
            r'TS\s+\d+\.\d+',
        ]

        tech_count = sum(len(re.findall(p, content)) for p in tech_patterns)

        # Normalize to 0-1 scale
        complexity = min(1.0, (word_count / 1000) * 0.5 + (tech_count / 50) * 0.5)
        return round(complexity, 3)

    def extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content"""
        # Find abbreviations
        abbrs = re.findall(r'\b([A-Z]{2,6})\b', content)

        # Count and get most common
        from collections import Counter
        counter = Counter(abbrs)

        return [term for term, count in counter.most_common(10)]

    @staticmethod
    def _split_oversized(content: str, limit: int) -> List[str]:
        """Split content into <= limit-char parts at line boundaries (a 3GPP
        clause's natural unit — one procedure step / one paragraph per line).
        A single line longer than the limit is emitted whole rather than cut
        mid-sentence."""
        if len(content) <= limit:
            return [content]
        parts, buf, size = [], [], 0
        for line in content.split('\n'):
            add = len(line) + 1
            if buf and size + add > limit:
                parts.append('\n'.join(buf))
                buf, size = [], 0
            buf.append(line)
            size += add
        if buf:
            parts.append('\n'.join(buf))
        return parts

    def create_chunks(self, sections: List[SectionStructure],
                      metadata: DocumentMetaData) -> List[ProcessedChunk]:
        """Create content chunks"""
        chunks = []
        # 3GPP documents occasionally number two different clauses identically
        # (TS 22.858 has two "A.3.3"). Both were emitted with the same chunk_id,
        # and `_create_chunks` MERGEs on chunk_id — so one silently overwrote the
        # other in the KG. Suffix the later ones instead of losing them.
        seen_ids: Dict[str, int] = {}

        # Built from ALL sections, including the content-less ones skipped below —
        # that is the whole point: their titles survive as an ancestor path on
        # their descendants instead of vanishing with the empty section.
        title_by_id: Dict[str, str] = {
            sec.section_id: sec.title for sec in sections if sec.title
        }

        def ancestor_titles(sid: str) -> List[str]:
            parts = sid.split('.')
            out = []
            for k in range(1, len(parts)):
                t = title_by_id.get('.'.join(parts[:k]))
                if t:
                    out.append(t)
            return out

        for section in sections:
            if not section.content.strip():
                continue

            # Oversized clause → parts (see MAX_CHUNK_CHARS). Part 1 keeps the
            # original id so clause citations ("TS 23.502 clause 4.2.2.2.2") and
            # any stored chunk_id still resolve.
            parts = self._split_oversized(section.content, MAX_CHUNK_CHARS)
            for idx, content in enumerate(parts, start=1):
                if not content.strip():
                    continue
                section_id = section.section_id if idx == 1 else f"{section.section_id}p{idx}"
                title = section.title if idx == 1 else f"{section.title} (part {idx})"

                n_dup = seen_ids.get(section_id, 0) + 1
                seen_ids[section_id] = n_dup
                if n_dup > 1:  # duplicate clause number in the source document
                    section_id = f"{section_id}~{n_dup}"

                word_count = len(content.split())

                content_metadata = {
                    "word_count": word_count,
                    "complexity_score": self.compute_complexity(content),
                    "key_terms": self.extract_key_terms(content)
                }

                chunk = ProcessedChunk(
                    chunk_id=f"{metadata.specification_id}_{section_id}",
                    section_id=section_id,
                    section_title=title,
                    content=content,
                    chunk_type=self.classify_content_type(title, content),
                    cross_references=self.extract_cross_references(content, metadata.specification_id),
                    content_metadata=content_metadata,
                    # Tables belong to the clause as a whole; attach them to part 1
                    # so they are not duplicated across every part.
                    tables=section.tables if idx == 1 else None,
                    # Derived from section.section_id, not the suffixed id: a
                    # "p2"/"~2" suffix is a chunker artefact, not a clause level.
                    section_path=ancestor_titles(section.section_id) or None,
                )
                chunks.append(chunk)

        return chunks

    def process_document(self, document_path: str) -> Tuple[List[ProcessedChunk], DocumentMetaData]:
        """Process a single document"""
        metadata = self.extract_metadata(document_path)
        sections = self.parse_sections(document_path)
        chunks = self.create_chunks(sections, metadata)
        return chunks, metadata

    def save_to_json(self, chunks: List[ProcessedChunk],
                     metadata: DocumentMetaData,
                     output_dir: Path) -> str:
        """Save chunks to JSON file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{metadata.specification_id}.json"

        data = {
            "metadata": asdict(metadata),
            "export_info": {
                "export_date": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "pipeline_version": "v3"
            },
            "chunks": [asdict(chunk) for chunk in chunks]
        }

        # Write to a temp file then os.replace() — atomic on the same filesystem.
        # Writing straight to output_path leaves a TRUNCATED but existing JSON when
        # the run is killed mid-dump, and the skip-if-exists resume then skips that
        # file forever (measured: ts_23_501.json after a killed run).
        tmp_path = output_path.with_suffix(".json.tmp")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_path)

        return str(output_path)


# ── Process-pool worker ─────────────────────────────────────────────────────
# Chunking is pure-CPU Python (mammoth → BeautifulSoup → regex), so a thread pool
# gets serialised by the GIL: measured 4 workers = 102% CPU on a 32-core box, the
# 4 threads merely taking turns (~17 min CPU each). Processes give real parallelism.
# These two must live at module level so they pickle by name.
_WORKER: "ThreeGPPProcessor | None" = None


def _worker_init(config: "Config", output_dir: Path) -> None:
    """Build ONE processor per worker process, not one per task."""
    global _WORKER
    _WORKER = ThreeGPPProcessor(config)
    _WORKER.output_dir = output_dir


def _worker_task(task: Union[Path, List[Path]], force: bool) -> str:
    return _WORKER._do_task(task, force)


class ThreeGPPProcessor:
    """Process all downloaded 3GPP documents"""

    def __init__(self, config: Config):
        self.config = config
        self.parser = ThreeGPPParser()
        self.output_dir = config.output_dir

        self.main_body_pattern = re.compile(r'main.*body|_1_|s\d+_s\d+', re.IGNORECASE)
        self.exclude_pattern = re.compile(r'cover|annex|history|amendment', re.IGNORECASE)

    def find_docx_files(self, data_dir: Path = None) -> List[Path]:
        """Find all main body DOCX files"""
        source = data_dir or self.config.data_dir
        docx_files = []

        for docx_path in source.rglob("*.docx"):
            filename = docx_path.name

            if filename.startswith('~'):
                continue

            if self._is_main_body(filename):
                docx_files.append(docx_path)

        return sorted(docx_files)

    def _is_main_body(self, filename: str) -> bool:
        """Check if file is main body document"""
        # Exclude known non-content files
        if self.exclude_pattern.search(filename):
            return False

        # Exclude revision mark files (-rm) and macOS hidden files
        if '-rm.' in filename.lower() or filename.startswith('._'):
            return False

        if self.main_body_pattern.search(filename):
            return True

        # Match various 3GPP spec filename formats:
        # - 23501-i80.docx (standard: 5 digits + letter + 2 digits)
        # - 23501-ib0.docx (with 'b' suffix)
        # - 38101-1-ib1.docx (multi-part specs like 38.101-1)
        # - 38101-2-ib0.docx (multi-part specs like 38.101-2)
        # - 24283-010.docx (3-digit version number)
        # - 31121-i40_s01-s03_5.docx (section splits)
        # - 38101-1-ib1_s07-XX.docx (multi-part section splits)
        # - 36718-02-01-i00.docx (multi-part with subpart like 36.718-02-01)
        # - 26927-070-cl.docx (clean version suffix)
        # Pattern: 5 digits + optional parts + version + optional suffixes
        if re.match(r'^\d{5}(-\d+)*-[a-z]{0,2}\d{1,3}(-cl)?(_s[\w\-\.]+)?\.docx$', filename, re.IGNORECASE):
            return True

        return False

    def process_all(self, data_dir: Path = None, parallel: bool = True,
                    max_workers: int | None = None, force: bool = False) -> Tuple[int, int, int]:
        """Process all documents.

        Returns (success_count, fail_count, skipped_count). A skip happens when
        the output JSON for the corresponding spec_id already exists and force=False.
        """
        print_header("Processing Documents to JSON")

        docx_files = self.find_docx_files(data_dir)
        print_status(f"Found {len(docx_files)} main body DOCX files", "info")

        tasks, collisions = self._build_tasks(docx_files)
        if collisions:
            sections_n = sum(1 for _, k, _ in collisions if k == "sections")
            version_n = sum(1 for _, k, _ in collisions if k == "version")
            print_status(
                f"Detected {len(collisions)} spec_id collision(s): "
                f"{sections_n} section-split (will merge), "
                f"{version_n} version-variant (keep newest)",
                "warn",
            )
            for json_name, kind, filenames in collisions:
                if kind == "sections":
                    print(f"     [merge] {json_name}: {len(filenames)} splits → 1 JSON")
                else:
                    kept, dropped = filenames[-1], filenames[:-1]
                    print(f"     [version] {json_name}: drop {dropped} → keep {kept}")
            print_status(f"Total tasks: {len(tasks)} (after dedup/merge)", "info")

        # Pre-scan output dir to report skip count up front — user visibility
        if force:
            print_status("Force mode: ignore skip-if-exists, reprocess all", "info")
        else:
            existing = (
                list(self.output_dir.glob("*.json"))
                if self.output_dir.exists()
                else []
            )
            if existing:
                print_status(
                    f"Output dir already has {len(existing)} JSON — corresponding files will be skipped "
                    f"(use --force to reprocess all)",
                    "info",
                )
            else:
                print_status("Output dir empty — all DOCX will be processed", "info")

        if not tasks:
            print_status("No documents found to process", "warn")
            return 0, 0, 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        fail_count = 0
        skipped_count = 0
        # `_process_single` swallows both the exception path and the "parsed 0
        # sections" path into the same silent "fail". Without the names there is
        # no way to tell a corrupt DOCX from a parser gap after a 2000-file run.
        failed_labels: List[str] = []

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        # Counter status: "ok" = processed, "skipped" = JSON already existed, "fail" = error
        def tally(status: str, label: str = "") -> None:
            nonlocal success_count, fail_count, skipped_count
            if status == "ok":
                success_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                fail_count += 1
                if label:
                    failed_labels.append(label)

        # Update tqdm postfix so the user sees the 3 live counters
        def update_pbar(pbar) -> None:
            if pbar is not None and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    ok=success_count,
                    skip=skipped_count,
                    fail=fail_count,
                    refresh=False,
                )

        # Helper: display name for a task (single Path or list of Path)
        def task_label(t: Union[Path, List[Path]]) -> str:
            if isinstance(t, list):
                return f"[merge {len(t)}] {t[0].name}..."
            return t.name

        if parallel and len(tasks) > 1:
            workers = max_workers or DEFAULT_WORKERS
            print_status(f"Parallel: {workers} worker processes", "info")
            pending = list(tasks)
            pbar = tqdm(total=len(pending), desc="Processing") if use_tqdm else None
            # A worker dying hard (segfault in lxml, OOM on a 42 MB DOCX) raises
            # BrokenProcessPool and takes the WHOLE pool with it — unlike a thread
            # pool, where only that one future fails. Restart the pool on what is
            # left instead of losing the rest of the run; skip-if-exists makes the
            # already-written files free to re-visit.
            while pending:
                done_now: List[Union[Path, List[Path]]] = []
                try:
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_worker_init,
                        initargs=(self.config, self.output_dir),
                    ) as executor:
                        futures = {
                            executor.submit(_worker_task, t, force): t for t in pending
                        }
                        for future in as_completed(futures):
                            task = futures[future]
                            try:
                                tally(future.result(), task_label(task))
                            except Exception as e:
                                print_status(f"Error: {task_label(task)}: {e}", "warn")
                                tally("fail", task_label(task))
                            done_now.append(task)
                            if pbar is not None:
                                pbar.update(1)
                                update_pbar(pbar)
                    pending = []
                except BrokenProcessPool:
                    pending = [t for t in pending if t not in done_now]
                    print_status(
                        f"Worker pool died — restarting for the remaining {len(pending)} task(s)",
                        "warn",
                    )
                    if workers > 1:
                        workers = max(1, workers // 2)  # likely memory pressure
            if pbar is not None:
                pbar.close()
        else:
            pbar = tqdm(total=len(tasks), desc="Processing") if use_tqdm else None
            for task in tasks:
                try:
                    tally(self._do_task(task, force), task_label(task))
                except Exception as e:
                    print_status(f"Error: {task_label(task)}: {e}", "warn")
                    tally("fail", task_label(task))
                if pbar is not None:
                    pbar.update(1)
                    update_pbar(pbar)
            if pbar is not None:
                pbar.close()

        msg = f"Processed {success_count}, skipped {skipped_count} (already had JSON), {fail_count} failed"
        print_status(msg, "ok" if fail_count == 0 else "warn")

        if failed_labels:
            failed_labels.sort()
            fail_log = self.output_dir / "_failed_docx.txt"
            try:
                fail_log.write_text("\n".join(failed_labels) + "\n", encoding="utf-8")
                print_status(f"Failed file list → {fail_log}", "warn")
            except OSError as e:
                print_status(f"Could not write {fail_log}: {e}", "warn")
            for name in failed_labels[:20]:
                print(f"     [fail] {name}")
            if len(failed_labels) > 20:
                print(f"     ... and {len(failed_labels) - 20} more (see {fail_log.name})")

        return success_count, fail_count, skipped_count

    def _build_tasks(
        self, docx_files: List[Path]
    ) -> Tuple[List[Union[Path, List[Path]]], List[Tuple[str, str, List[str]]]]:
        """Build worker tasks, classifying collisions into 2 groups:

        - **version variants** (e.g. `23501-ib0` vs `23501-ic0`): different version
          of the same spec → keep the alphabetically-last file (newest).
        - **section splits** (e.g. `38133-ib0_s0-7`, `38133-ib0_s8-8`, ...): same
          version, different sections → merge all into 1 JSON.

        Classification heuristic: strip everything after the first `_` in the filename.
          - single base in the group → section splits
          - ≥2 distinct bases → version variants

        Returns:
          tasks: list. Each element is a Path (single file) or List[Path] (section
                 merge — all splits of the same spec).
          summary: list (json_name, kind, file_list_for_log).
        """
        groups: Dict[str, List[Path]] = defaultdict(list)
        for d in docx_files:
            # First candidate = Path B output; a collision is detected when 2 files
            # derive the same candidate.
            primary = self._expected_json_names(d)[0]
            groups[primary].append(d)

        tasks: List[Union[Path, List[Path]]] = []
        summary: List[Tuple[str, str, List[str]]] = []
        for json_name, paths in groups.items():
            if len(paths) == 1:
                tasks.append(paths[0])
                continue

            paths_sorted = sorted(paths, key=lambda p: p.name)
            bases = {p.stem.split("_", 1)[0] for p in paths_sorted}

            if len(bases) == 1:
                # All same prefix → section splits → MERGE
                tasks.append(paths_sorted)
                summary.append((json_name, "sections", [p.name for p in paths_sorted]))
            else:
                # Different prefixes → version variants → keep newest
                tasks.append(paths_sorted[-1])
                summary.append((
                    json_name, "version",
                    [p.name for p in paths_sorted],
                ))
        return tasks, summary

    def _expected_json_names(self, docx_path: Path) -> List[str]:
        """Derive the possible JSON filenames from the DOCX name, WITHOUT opening the file.

        Returns a candidate list because the same DOCX name may map to 2 spec_ids
        depending on Path A (DOCX header) or Path B (filename regex):
            23501-i90.docx       → [ts_23_501.json]            # i90 is not a part
            38101-1-ib1.docx     → [ts_38_101-1.json, ts_38_101.json]
            24283-010.docx       → [ts_24_283-010.json, ts_24_283.json]
                                   ↑ 010 is version, Path A drops it → real file = ts_24_283
            <no match>.docx      → [ts_<stem>.json]
        """
        filename = docx_path.stem
        m = re.search(r'(\d{2})(\d{3})(?:-(\d+))?', filename)
        if not m:
            return [f"ts_{filename}.json"]
        series, spec_num, part = m.group(1), m.group(2), m.group(3)
        candidates = []
        if part:
            # Path B style: keep -part
            candidates.append(f"ts_{series}_{spec_num}-{part}.json")
        # Path A style: no part (DOCX header usually omits the version suffix)
        candidates.append(f"ts_{series}_{spec_num}.json")
        return candidates

    def _do_task(self, task: Union[Path, List[Path]], force: bool = False) -> str:
        """Dispatch task to single-file or section-merge processor."""
        if isinstance(task, list):
            return self._process_section_splits(task, force)
        return self._process_single(task, force)

    def _process_section_splits(self, paths: List[Path], force: bool = False) -> str:
        """Process N section-split DOCX of the same spec, merging chunks → 1 JSON.

        Each split contains different sections of the same document (e.g. 38133-ib0_s0-7
        and 38133-ib0_s8-8). Concat chunks, dedup by chunk_id, save 1 file.
        """
        try:
            if not force:
                # Skip-if-exists: use the candidate name of the first file
                for name in self._expected_json_names(paths[0]):
                    if (self.output_dir / name).exists():
                        return "skipped"

            all_chunks = []
            canonical_meta = None
            seen_chunk_ids = set()
            for p in paths:
                chunks, meta = self.parser.process_document(str(p))
                # Use the first split's metadata as canonical
                if canonical_meta is None:
                    canonical_meta = meta
                for c in chunks:
                    # Dedup chunk_id in case sections overlap between splits
                    if c.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(c.chunk_id)
                    all_chunks.append(c)

            if all_chunks and canonical_meta:
                self.parser.save_to_json(all_chunks, canonical_meta, self.output_dir)
                return "ok"
            return "fail"

        except Exception:
            return "fail"

    def _process_single(self, docx_path: Path, force: bool = False) -> str:
        """Process a single document.

        Returns: 'ok' | 'skipped' | 'fail'.
        When force=False, check whether the JSON already exists in output (compared
        by candidate names derived from the DOCX name). Cheap O(1), no DOCX read.
        """
        try:
            if not force:
                for name in self._expected_json_names(docx_path):
                    if (self.output_dir / name).exists():
                        return "skipped"

            chunks, metadata = self.parser.process_document(str(docx_path))

            if chunks:
                self.parser.save_to_json(chunks, metadata, self.output_dir)
                return "ok"
            return "fail"

        except Exception:
            return "fail"


# =============================================================================
# Knowledge Graph Initializer
# =============================================================================

class KnowledgeGraphInitializer:
    """Initialize Neo4j Knowledge Graph from JSON files"""

    def __init__(self, config: Config):
        self.config = config
        self.json_dir = config.output_dir

    def initialize(self, clear_first: bool = True) -> bool:
        """Initialize knowledge graph from JSON files"""
        try:
            from neo4j import GraphDatabase
            from tqdm import tqdm
        except ImportError as e:
            print_status(f"Missing dependency: {e}", "error")
            return False

        print_header("Initializing Knowledge Graph")

        if not self.json_dir.exists():
            print_status(f"JSON directory not found: {self.json_dir}", "error")
            return False

        json_files = list(self.json_dir.glob("*.json"))
        if not json_files:
            print_status("No JSON files found", "error")
            return False

        print_status(f"Found {len(json_files)} JSON files", "info")

        print_status("Connecting to Neo4j...", "info")
        driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )

        try:
            with driver.session() as session:
                session.run("RETURN 1")
            print_status("Neo4j connection established", "ok")

            if clear_first:
                print_status("Clearing existing database...", "info")
                with driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
                print_status("Database cleared", "ok")

            # Load JSON files
            print_status("Loading JSON files...", "info")
            documents = {}
            chunks = []

            for json_file in tqdm(json_files, desc="Loading JSON"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    spec_id = data["metadata"]["specification_id"]
                    documents[spec_id] = data
                    chunks.extend(data["chunks"])

            print_status(f"Loaded {len(documents)} documents with {len(chunks)} chunks", "ok")

            # Create constraints
            print_status("Creating database constraints...", "info")
            with driver.session() as session:
                try:
                    session.run("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.spec_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT term_abbr IF NOT EXISTS FOR (t:Term) REQUIRE t.abbreviation IS UNIQUE")
                except:
                    pass

            # Create Document nodes
            print_status("Creating Document nodes...", "info")
            with driver.session() as session:
                for spec_id, data in tqdm(documents.items(), desc="Documents"):
                    session.run("""
                        MERGE (d:Document {spec_id: $spec_id})
                        SET d.version = $version,
                            d.title = $title,
                            d.release = $release,
                            d.total_chunks = $total_chunks
                    """,
                        spec_id=spec_id,
                        version=data["metadata"]["version"],
                        title=data["metadata"]["title"],
                        release=data["metadata"].get("release", ""),
                        total_chunks=data["export_info"]["total_chunks"]
                    )
            print_status(f"Created {len(documents)} Document nodes", "ok")

            # Create Chunk nodes
            print_status("Creating Chunk nodes...", "info")
            with driver.session() as session:
                for chunk in tqdm(chunks, desc="Chunks"):
                    chunk_id = chunk["chunk_id"]

                    if "_" in chunk_id:
                        parts = chunk_id.rsplit("_", 1)
                        spec_id = parts[0]
                    else:
                        spec_id = chunk_id

                    content_meta = chunk.get("content_metadata", {})

                    session.run("""
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.section_id = $section_id,
                            c.section_title = $section_title,
                            c.content = $content,
                            c.chunk_type = $chunk_type,
                            c.spec_id = $spec_id,
                            c.word_count = $word_count,
                            c.complexity_score = $complexity_score,
                            c.key_terms = $key_terms
                    """,
                        chunk_id=chunk["chunk_id"],
                        section_id=chunk["section_id"],
                        section_title=chunk["section_title"],
                        content=chunk["content"],
                        chunk_type=chunk["chunk_type"],
                        spec_id=spec_id,
                        word_count=content_meta.get("word_count", 0),
                        complexity_score=content_meta.get("complexity_score", 0.0),
                        key_terms=content_meta.get("key_terms", [])
                    )
            print_status(f"Created {len(chunks)} Chunk nodes", "ok")

            # Create CONTAINS relationships
            print_status("Creating CONTAINS relationships...", "info")
            with driver.session() as session:
                session.run("""
                    MATCH (d:Document), (c:Chunk)
                    WHERE d.spec_id = c.spec_id
                    MERGE (d)-[:CONTAINS]->(c)
                """)

            # Create REFERENCES_SPEC relationships
            print_status("Creating REFERENCES_SPEC relationships...", "info")
            ref_count = 0
            with driver.session() as session:
                for chunk in tqdm(chunks, desc="References"):
                    cross_refs = chunk.get("cross_references", {})

                    for ref in cross_refs.get("external", []):
                        ref_uid = hashlib.md5(
                            f"{chunk['chunk_id']}_{ref['target_spec']}_{ref.get('ref_id', '')}".encode()
                        ).hexdigest()[:10]

                        result = session.run("""
                            MATCH (source:Chunk {chunk_id: $source_id})
                            MATCH (target_doc:Document {spec_id: $target_spec})
                            MERGE (source)-[r:REFERENCES_SPEC {ref_uid: $ref_uid}]->(target_doc)
                            SET r.ref_type = $ref_type,
                                r.ref_id = $ref_id,
                                r.confidence = $confidence
                            RETURN count(*) as created
                        """,
                            source_id=chunk["chunk_id"],
                            target_spec=ref["target_spec"],
                            ref_uid=ref_uid,
                            ref_type=ref.get("ref_type", ""),
                            ref_id=ref.get("ref_id", ""),
                            confidence=ref.get("confidence", 0.0)
                        )
                        rec = result.single()
                        if rec:
                            ref_count += rec["created"]

            print_status(f"Created {ref_count} REFERENCES_SPEC relationships", "ok")

            # Create Term nodes
            print_status("Creating Term nodes from abbreviations...", "info")
            term_count = self._create_term_nodes(driver, chunks)
            print_status(f"Created {term_count} Term nodes", "ok")

            print_status("Knowledge Graph initialized successfully!", "ok")
            return True

        except Exception as e:
            print_status(f"Error initializing graph: {e}", "error")
            import traceback
            traceback.print_exc()
            return False
        finally:
            driver.close()

    def _create_term_nodes(self, driver, chunks: list) -> int:
        """Create Term nodes from abbreviation sections"""
        try:
            from term_extractor import TermExtractor
        except ImportError:
            print_status("term_extractor not found, skipping Term nodes", "warn")
            return 0

        extractor = TermExtractor()
        term_dict = {}

        for chunk in chunks:
            section_title = chunk.get("section_title", "").lower()
            content = chunk.get("content", "")
            chunk_id = chunk["chunk_id"]

            if "_" in chunk_id:
                parts = chunk_id.rsplit("_", 1)
                spec_id = parts[0]
            else:
                spec_id = chunk_id

            if 'abbreviation' in section_title:
                terms = extractor.extract_abbreviations(content, spec_id)
                self._merge_terms(term_dict, terms)
            elif 'definition' in section_title:
                terms = extractor.extract_definitions(content, spec_id)
                self._merge_terms(term_dict, terms)

        term_count = 0
        with driver.session() as session:
            for abbr, term_info in term_dict.items():
                try:
                    session.run("""
                        MERGE (t:Term {abbreviation: $abbreviation})
                        SET t.full_name = $full_name,
                            t.term_type = $term_type,
                            t.source_specs = $source_specs,
                            t.primary_spec = $primary_spec
                    """,
                        abbreviation=abbr,
                        full_name=term_info['full_name'],
                        term_type=term_info['term_type'],
                        source_specs=term_info['source_specs'],
                        primary_spec=term_info['primary_spec']
                    )

                    for source_spec in term_info['source_specs']:
                        session.run("""
                            MATCH (t:Term {abbreviation: $abbreviation})
                            MATCH (d:Document {spec_id: $spec_id})
                            MERGE (t)-[:DEFINED_IN]->(d)
                        """,
                            abbreviation=abbr,
                            spec_id=source_spec
                        )

                    term_count += 1
                except:
                    pass

        return term_count

    def _merge_terms(self, term_dict: Dict, terms: list):
        """Merge extracted terms into dictionary"""
        for term in terms:
            abbr = term.abbreviation
            if abbr not in term_dict:
                term_dict[abbr] = {
                    'abbreviation': abbr,
                    'full_name': term.full_name,
                    'term_type': term.term_type,
                    'source_specs': [term.source_spec],
                    'primary_spec': term.source_spec
                }
            else:
                if term.source_spec not in term_dict[abbr]['source_specs']:
                    term_dict[abbr]['source_specs'].append(term.source_spec)


# =============================================================================
# Main Pipeline
# =============================================================================

class Pipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.downloader = ThreeGPPDownloader(self.config)
        self.extractor = ThreeGPPExtractor(self.config)
        self.processor = ThreeGPPProcessor(self.config)
        self.kg_initializer = KnowledgeGraphInitializer(self.config)

    def status(self):
        """Show current status"""
        print_header("Pipeline Status")

        # Download status
        download_dir = self.config.download_dir
        if download_dir.exists():
            zip_count = len(list(download_dir.rglob("*.zip")))
            print_status(f"Downloaded: {zip_count} ZIP files in {download_dir}", "ok" if zip_count > 0 else "warn")
        else:
            print_status("Downloaded: No files yet", "warn")

        # Extracted status
        data_dir = self.config.data_dir
        if data_dir.exists():
            docx_count = len(list(data_dir.rglob("*.docx")))
            print_status(f"Extracted: {docx_count} DOCX files in {data_dir}", "ok" if docx_count > 0 else "warn")
        else:
            print_status("Extracted: No files yet", "warn")

        # Processed status
        output_dir = self.config.output_dir
        if output_dir.exists():
            json_count = len(list(output_dir.glob("*.json")))
            print_status(f"Processed: {json_count} JSON files in {output_dir}", "ok" if json_count > 0 else "warn")
        else:
            print_status("Processed: No JSON files yet", "warn")

        # Neo4j status
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            with driver.session() as session:
                result = session.run("""
                    MATCH (d:Document) WITH count(d) as docs
                    MATCH (c:Chunk) WITH docs, count(c) as chunks
                    MATCH (t:Term) WITH docs, chunks, count(t) as terms
                    RETURN docs, chunks, terms
                """)
                record = result.single()
                print_status(f"Neo4j: {record['docs']} Documents, {record['chunks']} Chunks, {record['terms']} Terms", "ok")
            driver.close()
        except Exception as e:
            print_status(f"Neo4j: Not connected ({e})", "warn")

    def download(self):
        """Download 3GPP documents"""
        return self.downloader.download_release()

    def extract(self):
        """Extract downloaded ZIP files"""
        return self.extractor.extract_all()

    def process(self, force: bool = False):
        """Process documents to JSON"""
        return self.processor.process_all(force=force)

    def process_local(self, input_dir: Path, force: bool = False):
        """
        Process documents from a local directory.
        Supports both ZIP files and DOCX files.
        If ZIP files found, extracts them first then processes.
        """
        if not input_dir.exists():
            print_status(f"Input directory not found: {input_dir}", "error")
            return 0, 0, 0

        # Check for ZIP files
        zip_files = list(input_dir.rglob("*.zip"))
        docx_files = list(input_dir.rglob("*.docx"))

        print_header(f"Processing Local Documents: {input_dir}")
        print_status(f"Found {len(zip_files)} ZIP files, {len(docx_files)} DOCX files", "info")

        # If ZIP files exist, extract them first
        if zip_files:
            print_status("Extracting ZIP files...", "progress")
            extracted_count = self.extractor.extract_all(source_dir=input_dir)
            print_status(f"Extracted {extracted_count} DOCX files", "ok")
            # Process from extracted directory
            return self.processor.process_all(data_dir=self.config.data_dir, force=force)
        elif docx_files:
            # Process DOCX files directly
            return self.processor.process_all(data_dir=input_dir, force=force)
        else:
            print_status("No ZIP or DOCX files found in input directory", "error")
            return 0, 0, 0

    def init_kg(self):
        """Initialize Knowledge Graph"""
        return self.kg_initializer.initialize()

    def run_all(self, force: bool = False):
        """Run complete pipeline"""
        print_header("3GPP Document Processing Pipeline")
        print_status(f"Release: {self.config.release}", "info")
        print_status(f"Series: {self.config.series}", "info")
        print_status(f"Output: {self.config.output_dir}", "info")
        print()

        # Step 1: Download
        if not self.download():
            print_status("Download failed or no new files", "warn")

        # Step 2: Extract
        extracted = self.extract()
        if extracted == 0:
            # Try processing existing data_dir
            existing_docx = len(list(self.config.data_dir.rglob("*.docx"))) if self.config.data_dir.exists() else 0
            if existing_docx == 0:
                print_status("No files to process. Check download.", "error")
                return False

        # Step 3: Process
        success, fail, skipped = self.process(force=force)
        if success == 0 and skipped == 0:
            print_status("No documents processed.", "error")
            return False

        # Step 4: Initialize KG
        if not self.init_kg():
            print_status("Knowledge Graph initialization failed.", "error")
            return False

        print_header("Pipeline Complete!")
        self.status()
        return True


def main():
    parser = argparse.ArgumentParser(
        description="3GPP Document Download and Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_and_process_3gpp.py status
  python download_and_process_3gpp.py download --release 18 --series 23,29
  python download_and_process_3gpp.py process
  python download_and_process_3gpp.py process-local --input document_processing/data/rel18_extracted/
  python download_and_process_3gpp.py all
        """
    )
    parser.add_argument(
        "command",
        choices=["download", "extract", "process", "process-local", "init-kg", "all", "status"],
        help="Command to run"
    )
    parser.add_argument(
        "--release", "-r",
        type=int,
        default=18,
        help="3GPP Release number (default: 18)"
    )
    parser.add_argument(
        "--series", "-s",
        type=str,
        default=None,
        help="Comma-separated series numbers (e.g., '23,29,38')"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input directory for process-local command (contains DOCX files)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-process JSON verification step"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all DOCX even if output JSON already exists (default: skip-if-exists)"
    )

    args = parser.parse_args()

    # Create config
    config = Config(release=args.release)

    if args.series:
        config.series = [s.strip() for s in args.series.split(',')]

    if args.output:
        config.output_dir = Path(args.output)

    # Create pipeline
    pipeline = Pipeline(config)

    # Run command
    if args.command == "process-local":
        # Validate input directory is provided
        if not args.input:
            print_status("Error: --input/-i is required for process-local command", "error")
            print_status("Usage: python download_and_process_3gpp.py process-local --input /path/to/docs", "info")
            sys.exit(1)
        input_path = Path(args.input)
        pipeline.process_local(input_path, force=args.force)
    elif args.command == "process":
        pipeline.process(force=args.force)
    elif args.command == "all":
        pipeline.run_all(force=args.force)
    else:
        commands = {
            "download": pipeline.download,
            "extract": pipeline.extract,
            "init-kg": pipeline.init_kg,
            "status": pipeline.status
        }
        commands[args.command]()

    # Auto-verify JSON output for commands that produce JSON
    # (process / process-local / all). Skipped with --no-verify or non-producing commands.
    PRODUCES_JSON = {"process", "process-local", "all"}
    if args.command in PRODUCES_JSON and not args.no_verify:
        run_post_verify(config.output_dir)


def run_post_verify(output_dir: Path) -> None:
    """Run the verify suite on output JSON. Non-fatal — only prints results."""
    # Load verify.py via importlib to avoid clashing with the `tests/` package at repo
    # root (both PROJECT_ROOT/tests/ and document_processing/tests/ exist).
    verify_path = SCRIPT_DIR / "tests" / "verify.py"
    if not verify_path.exists():
        print_status(f"Verify skipped: not found at {verify_path}", "warn")
        return

    import importlib.util
    spec = importlib.util.spec_from_file_location("dp_tests_verify", verify_path)
    verify_mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(verify_mod)
    except Exception as e:
        print_status(f"Verify skipped: load error ({e})", "warn")
        return

    load_all = verify_mod.load_all
    run_checks = verify_mod.run_checks
    print_summary = verify_mod.print_summary

    print_header("Verifying Output JSON")
    if not output_dir.is_dir():
        print_status(f"Output dir does not exist: {output_dir}", "warn")
        return

    docs = load_all(output_dir)
    if not docs:
        print_status(f"No JSON files in {output_dir}", "warn")
        return

    print_status(f"Loaded {len(docs)} files", "info")
    print()
    passed, failed = run_checks(docs)
    print_summary(docs)
    print()
    if failed:
        print_status(f"{failed} check fail / {passed + failed} total", "warn")
    else:
        print_status(f"All {passed} checks passed", "ok")


if __name__ == "__main__":
    main()
