"""
KG Builder — ported from kg_initializer.py + term_extractor.py + subject_classifier.py
of the parent project, merged into a self-contained file for the demo.

Pipeline when load_json_dir(path) is called:
  1. Load JSON files → documents + chunks (inject _spec_id from metadata)
  2. Create constraints (Document.spec_id, Chunk.chunk_id, Term.abbreviation, Subject.name,
     ServiceOperation.name, Parameter.param_id, Step.step_id, Message.name UNIQUE)
  3. Create Document nodes
  4. Create Chunk nodes (key_terms, word_count, complexity_score, ...)
  5. CONTAINS edges (Document → Chunk) — matched on spec_id
  6. REFERENCES_SPEC edges (Chunk → Document) — from cross_references.external (doc-level)
  7. REFERENCES_CHUNK edges (Chunk → Chunk) — 3-tier matching on section_id;
     covers both internal (same spec) and external (cross-spec, filter clause + conf≥0.7).
     Property `is_external` distinguishes the two groups.
  8. Term nodes + DEFINED_IN edges (Term → Document) — from section abbreviation/definition
  9. Subject nodes + HAS_SUBJECT edges (Chunk → Subject)
  10. PARENT_SECTION edges (Chunk → Chunk) + Chunk.is_parent_section property
  11. MENTIONS edges (Chunk → Term) — from key_terms ∩ Term.abbreviation, filtered by df band
  12. ServiceOperation nodes + PROVIDED_BY (→Term) + DESCRIBES_OPERATION (←Chunk) — from
      SBI pattern `N<nf>_<Service>_<Operation>` in content, NF-prefix must match a Term
  13. Term.semantic_type='network_function' (from PROVIDED_BY) + CO_OCCURS_WITH (Term↔Term,
      from MENTIONS co-occurrence, weight = number of shared chunks)
  14. Parameter nodes + DEFINED_IN_TABLE (←Chunk) — from Chunk.tables (name/description cols)
  15. Step nodes + HAS_STEP (←Chunk) + NEXT (Step→Step) + INVOLVES (→Term) — from chunks with
      chunk_type='procedure', split on the "N.\\t"/"Na.\\t" line-start marker
  16. Message nodes + DESCRIBES_MESSAGE (←Chunk) — ALL-CAPS message names (REGISTRATION
      REQUEST, PDU SESSION ESTABLISHMENT REQUEST...), filtered by document-frequency band

Schema (9 nodes + 13 edges):
  Nodes:
    Document         (spec_id, version, title, total_chunks)
    Chunk            (chunk_id, spec_id, section_id, section_title, content,
                      chunk_type, word_count, complexity_score, key_terms,
                      subject, subject_confidence, is_parent_section)
    Term             (abbreviation, full_name, term_type, source_specs, primary_spec,
                      semantic_type)                                          # NEW property
    Subject          (name, priority, description)
    ServiceOperation (name, nf_prefix, service, operation, df)                # NEW
    Parameter        (param_id, name, description, spec_id)                   # NEW
    Step             (step_id, chunk_id, order, text)                        # NEW
    Message          (name, df)                                              # NEW

    REFERENCES_CHUNK edge properties:
      is_external (bool), ref_type (str), ref_id (str), confidence (float)
    CO_OCCURS_WITH / MENTIONS edge property: weight / df

  Edges:
    (Document)-[:CONTAINS]->(Chunk)
    (Chunk)-[:REFERENCES_SPEC]->(Document)
    (Chunk)-[:REFERENCES_CHUNK]->(Chunk)
    (Term)-[:DEFINED_IN]->(Document)
    (Chunk)-[:HAS_SUBJECT]->(Subject)
    (Chunk)-[:PARENT_SECTION]->(Chunk)
    (Chunk)-[:MENTIONS]->(Term)
    (ServiceOperation)-[:PROVIDED_BY]->(Term)                                 # NEW
    (Chunk)-[:DESCRIBES_OPERATION]->(ServiceOperation)                       # NEW
    (Term)-[:CO_OCCURS_WITH]->(Term)                                         # NEW — first Term↔Term edge
    (Parameter)-[:DEFINED_IN_TABLE]->(Chunk)                                 # NEW
    (Chunk)-[:HAS_STEP]->(Step)                                              # NEW
    (Step)-[:NEXT]->(Step)                                                   # NEW
    (Step)-[:INVOLVES]->(Term)                                               # NEW
    (Chunk)-[:DESCRIBES_MESSAGE]->(Message)                                  # NEW

  Intentionally out of scope (see docs/de_xuat_cai_thien_kg_quan_he.md): do NOT extract
  actor_from/actor_to for Step — measured on the corpus only ~1.9% of steps have the
  telegraphic "Actor to Actor: message" form at the start; the rest is free prose,
  so coverage is too low and it easily creates wrong actor nodes (variants like
  "Target I-SMF", "UPF (PSA)", "old AMF"... don't map cleanly to a single Term).
  INVOLVES reuses the validated chunk-level key_terms (like MENTIONS) instead of free NLP.
"""
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# 5G-related spec series (used for conflict resolution when merging Terms)
_5G_SPEC_PREFIXES = (
    'ts_23_5', 'ts_29_5', 'ts_23_4', 'ts_29_2',
    'ts_33_5', 'ts_38_', 'ts_24_5', 'ts_26_5'
)

# Rows per UNWIND query when bulk-writing to Neo4j. Batching cuts round-trips
# (vs 1 s.run() per item) — speeds up graph build several-fold.
WRITE_BATCH_SIZE = 5000

# Document-frequency band for filtering noise when creating (Chunk)-[:MENTIONS]->(Term)
# edges from key_terms: drop ubiquitous terms (IDF≈0: TS/UE/NOTE...) and overly rare ones.
# Kept in sync with kg_builder/enrich_mentions.py (standalone backfill).
MENTIONS_MIN_DF = int(os.getenv("MENTIONS_MIN_DF", "5"))
MENTIONS_MAX_DF = int(os.getenv("MENTIONS_MAX_DF", "5000"))

# Minimum weight (shared chunks) to keep a (Term)-[:CO_OCCURS_WITH]->(Term) pair.
# Measured on the real KG: weight>=3 → ~62k pairs, weight>=10 → ~22k. 3 balances
# keeping real domain signal without being too sparse.
COOCCUR_MIN_WEIGHT = int(os.getenv("COOCCUR_MIN_WEIGHT", "3"))

# Service-Based Interface operation name: N<nf-lowercase>_<Service-PascalCase>_<Operation-PascalCase>
# e.g. Nudm_SDM_Get, Nausf_UEAuthentication_Authenticate. Service/Operation must start
# uppercase — cuts false positives from ordinary phrases. NF-prefix is also re-validated
# (must match an existing Term.abbreviation) at build time, see _create_service_operations.
SVC_OP_RE = re.compile(r"\bN[a-z][a-z0-9]*_[A-Z][A-Za-z0-9]*_[A-Z][A-Za-z0-9]*\b")

# "name" / "description" columns in 3GPP parameter tables — matched by column name
# (lowercase) to catch many variants (ASN.1-style IE tables, OpenAPI attribute tables,
# TTCN-3 parameter tables...). Measured on the real KG: 17,457/131,880 tables match both.
PARAM_NAME_HEADERS = {
    "name", "attribute name", "ie/group name", "information element",
    "information element/group name", "parameter", "iei", "field",
}
PARAM_DESC_HEADERS = {"description", "definition", "semantics description"}
PARAM_MAX_NAME_LEN = 80  # drop rows where the "name" column is actually a sentence (misaligned table)

# Step-start marker in procedure chunks: "1.\t", "3a.\t", "12b.\t" at line start —
# very consistent format in 3GPP (verified on TS 23.502/23.503 samples).
STEP_RE = re.compile(r"(?:\r?\n|^)(\d+[a-z]?)\.\t")

# Some specs number steps WITHOUT the period — TS 23.502 clause 4.17.6.2 writes
# "1<TAB>The SMF issues a Nnrf_NFManagement_..." — and STEP_RE's mandatory dot made
# those clauses invisible: that one, the case that started this whole investigation,
# had zero Steps. The dotless form cannot be enabled unconditionally, because it also
# matches AT-command tables (TS 27.005/27.007) and IMEI worked examples where the
# leading number is a table row, not a step. It is therefore accepted ONLY when the
# flow wording is present as well — 53 chunks, hand-checked at ~8/10 genuine.
STEP_RE_LOOSE = re.compile(r"(?:\r?\n|^)(\d+[a-z]?)\.?\t")

# Step extraction used to trust `chunk_type == 'procedure'` alone, and that label is a
# LOSSY PROXY: measured on the live KG, 10,080 chunks carry step markers with no Step
# node at all, against 3,592 that have them — 74% of the corpus's step structure was
# invisible. The misses are real procedures the classifier called something else:
# ts_23_502_4.17.6.2 "SMF provisioning of UPF instances using NRF" is chunk_type
# 'general' with ZERO steps, so a question about SMF↔NRF interaction could only ever be
# answered from study reports (TR 23.700-*), whose solution flows the classifier does
# label 'procedure'. Any procedure benchmark built on that gold would systematically
# reward study reports and penalise the normative spec.
#
# Two positive signals replace the proxy, and BOTH are required:
#   1. the markers form a real sequence — >=3 of them, starting at 0 or 1, >=70%
#      consecutive. This alone still admits ~50% noise (numbered requirement lists,
#      "Initial conditions" blocks in RF test specs).
#   2. the text says so — 3GPP procedure clauses announce themselves ("Figure X
#      illustrates the procedure", "signalling flow", "depicts"). Adding this took a
#      12-sample hand check from 6/12 correct to 8/8.
# Net: +1,797 chunks (a 50% coverage gain) instead of +7,878 at half precision.
# RF/RRC conformance specs are excluded outright: their "Initial conditions" blocks are
# test setup, not signalling, and ts_36_521-3 alone already contributes 4,987 of the
# 26,832 existing Steps.
# "flow" alone is unusable — 5G prose is full of QoS flow / traffic flow / SDF, and
# admitting it costs 121 chunks of noise. The qualified forms below add 14 chunks,
# all verified genuine ("Figure 7.4.2.2.3-1 provides an example flow for a call").
_STEP_FLOW_RE = re.compile(
    r"(?i)\b(procedure|signalling flow|call flow|message flow|example flow|"
    r"flow for a|flow diagram|illustrat|depict|shows the|following steps)\b")
_STEP_TEST_SPEC_RE = re.compile(r"^ts_(3[4-9]|2[5-9])_(5\d\d|571)")
STEP_MIN_MARKERS = int(os.getenv("STEP_MIN_MARKERS", "3"))
STEP_MIN_CONSECUTIVE = float(os.getenv("STEP_MIN_CONSECUTIVE", "0.7"))


def has_step_flow(chunk: dict) -> bool:
    """True if this chunk carries a real numbered procedure flow.

    Independent of chunk_type on purpose — see the note above. A chunk already typed
    'procedure' passes on the marker count alone, so existing extraction never regresses.
    """
    content = chunk.get("content") or ""
    head = (chunk.get("section_title") or "") + " " + content[:400]
    marks = [m.group(1) for m in STEP_RE.finditer(content)]
    dotless = False
    if len(marks) < STEP_MIN_MARKERS:
        # Fall back to the dotless marker, but only for text that announces itself as
        # a flow — see STEP_RE_LOOSE. Without that condition this admits AT-command
        # syntax tables wholesale.
        if not _STEP_FLOW_RE.search(head):
            return False
        marks = [m.group(1) for m in STEP_RE_LOOSE.finditer(content)]
        if len(marks) < STEP_MIN_MARKERS:
            return False
        dotless = True
    if chunk.get("chunk_type") == "procedure" and not dotless:
        return True
    if _STEP_TEST_SPEC_RE.match(chunk.get("chunk_id") or ""):
        return False
    nums = [int(re.match(r"(\d+)", m).group(1)) for m in marks if re.match(r"(\d+)", m)]
    if not nums or nums[0] not in (0, 1):
        return False
    consecutive = sum(1 for a, b in zip(nums, nums[1:]) if b in (a, a + 1))
    if consecutive / max(1, len(nums) - 1) < STEP_MIN_CONSECUTIVE:
        return False
    return bool(_STEP_FLOW_RE.search(head))

# 3GPP message name: 1-4 ALL-CAPS words + 1 suffix word from the standard message vocabulary.
# Measured on the real KG: 2,361 distinct candidates, df>=3 keeps ~53% (REGISTRATION
# REQUEST df=491, ATTACH REQUEST df=464... all real NAS/RRC messages).
_MESSAGE_SUFFIXES = (
    "REQUEST|RESPONSE|ACCEPT|REJECT|COMPLETE|COMMAND|NOTIFICATION|FAILURE|"
    "INDICATION|CONFIRM|SETUP|RELEASE|MODIFICATION|TRANSFER|REPORT"
)
MESSAGE_RE = re.compile(r"\b((?:[A-Z][A-Z0-9\-]* ){1,4}(?:" + _MESSAGE_SUFFIXES + r"))\b")
MESSAGE_MIN_DF = int(os.getenv("MESSAGE_MIN_DF", "3"))

# ── Standardized-value tables → Concept / StandardizedValue (Layer C, 2026-08-09) ──
# Three deterministic table shapes (checked in this order; a table matches at most one):
#   (a) enumeration  — headers "Enumeration value | Description [| Applicability]"
#       (stage-3 OpenAPI enums, ts_29.xxx); Concept name comes from the section title
#       ("Enumeration: SmContextStatus" / "XxxType enumeration").
#   (b) value_set    — a "<X> value"-style header + a description-ish column
#       (SST table 23.501 §5.15.2.2: "Slice/Service type | SST value | Characteristics").
#   (c) attribute_map— a "<X>Value" first column + >=2 named attribute columns
#       (5QI table 23.501 §5.7.4: 5QIValue | Resource Type | Priority | PDB | ...).
# Conformance-test specs are excluded: their thousands of tables are RF/protocol test
# parameter grids, not standardized value definitions (ts_38_533 alone: 968 candidates).
VALUE_SPEC_BLOCKLIST_PREFIXES = (
    "ts_34_", "ts_36_508", "ts_36_521", "ts_36_523", "ts_36_579", "ts_36_133",
    "ts_37_571", "ts_38_508", "ts_38_521", "ts_38_522", "ts_38_533",
    "ts_25_123", "ts_31_121", "ts_31_124",
    # RF-requirement measurement-grid family + study/test-analysis docs — the
    # 2026-08-09 adversarial audit measured these as the top garbage sources
    # (ts_38_133/ts_25_133 were the #2/#3 sources corpus-wide; ts_36_905 rows
    # are meeting-admin metadata, ts_38_869/ts_25_95x are simulation assumptions).
    "ts_25_133", "ts_38_133", "ts_38_174", "ts_25_951", "ts_25_956",
    "ts_36_905", "ts_38_869", "ts_38_523", "ts_37_579",
)
VALUE_DESC_HEADERS = {"characteristics", "description", "meaning", "semantics", "definition"}
# "Enumeration: SmContextStatus" | "SmContextStatus enumeration" | "Standardised SST values"
CONCEPT_TITLE_RES = [
    re.compile(r"^Enumeration:\s*(?P<name>[A-Za-z0-9_\-]+)", re.IGNORECASE),
    re.compile(r"^(?P<name>[A-Za-z0-9_\-]+)\s+enumeration\b", re.IGNORECASE),
    re.compile(r"^Standardi[sz]ed\s+(?P<name>[A-Za-z0-9_\-/ ]+?)\s+values?\b", re.IGNORECASE),
]
# Qualifier words before "value(s)" in a header are column adjectives, not
# concept names ("Reported value", "Valid values", "Default Value"…) — promoting
# them mints one global Concept merging dozens of unrelated tables. Verdicts
# from the 3-lens audit (2026-08-09): all majority-drop concepts land here.
VALUE_CONCEPT_STOPLIST = {
    "REPORTED", "VALID", "LEGAL", "DEFAULT", "BIT", "BINARY", "ASSUMED",
    "SAMPLE", "REQUIRED", "REQUIREDBIT", "ENUMERATED", "SUPPORTED", "ALLOWED",
    "MEASURED", "PERMITTED", "ATTRIBUTE", "FIELD", "PROTOCOL", "TAG", "NS",
    "ONPOWER", "HARQ-ACK",
}
# Audit rename verdicts — split-off header variants of the same registry.
VALUE_CONCEPT_RENAMES = {"CQIORCQIS": "CQI", "STATUS": "TDoc Status"}
# Enumeration identifiers (ALL_CAPS_UNDERSCORE stage-3 enums) run long —
# 12 chars cuts 52% of real enum values (TLS_WITH_AKMA, EAS_NOT_AVAILABLE…),
# and a PARTIAL enum in the KG is worse than an absent one.
VALUE_CODE_MAX_LEN = 12        # value_set / attribute_map codes ("2", "URLLC", "5QI-82")
VALUE_ENUM_CODE_MAX_LEN = 40   # enumeration identifiers
VALUE_MIN_PROSE_ROWS = 0.5  # >=50% of description cells must look like prose (space + len>=8)
_NOTE_FRAG_RE = re.compile(r"\(\s*NOTE[^)]*\)", re.IGNORECASE)
# Transposed-matrix detector: a table whose non-first headers are data-like
# (pure numbers / decimals / {..} sets) was read sideways by the extractor.
_DATAISH_HDR_RE = re.compile(r"^[\d\s.,%(){}\[\]\-–+]*$")

CYPHER_CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.spec_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Term) REQUIRE t.abbreviation IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subject) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (so:ServiceOperation) REQUIRE so.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Parameter) REQUIRE p.param_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (st:Step) REQUIRE st.step_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (co:Concept) REQUIRE co.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (v:StandardizedValue) REQUIRE v.value_id IS UNIQUE",
]

CYPHER_INDEXES = [
    "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.spec_id)",
    "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_type)",
    "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.section_id)",
    # BM25 (sparse) retrieval. Neo4j full-text indexes are Lucene-backed and score
    # with BM25 — this powers the `bm25` / `bm25_dense` retrieval modes without a
    # separate index store (rag-engine/retrieval/bm25_search.py).
    "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.content, c.section_title]",
    # Title-only sibling of the above. Titles average 4.1 tokens, so ranking them
    # separately is a different retriever, not a duplicate — it is what lets the
    # graph branch reach section_title by rank instead of Pattern B's LLM-written
    # regex (rag-engine/retrieval/title_search.py).
    "CREATE FULLTEXT INDEX chunk_title_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.section_title]",
]


# ─────────────────────────────────────────────────────────────────────────────
# Term Extraction (ported from term_extractor.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedTerm:
    """An extracted term (abbreviation or definition)."""
    abbreviation: str
    full_name: str
    term_type: str  # 'abbreviation' or 'definition'
    source_spec: str


class TermExtractor:
    """Extracts abbreviations + definitions from chunk content."""

    def __init__(self):
        # Pattern: ABBR<spaces>Full Name (alternative when there is no tab)
        self._abbr_space_pattern = re.compile(
            r'^([A-Z][A-Z0-9/-]{1,15})\s{2,}([A-Z][A-Za-z0-9\s\-/()]+)$',
            re.MULTILINE
        )

    def extract_abbreviations(self, content: str, spec_id: str) -> List[ExtractedTerm]:
        """Pattern: ABBR<tab>Full Name or ABBR<spaces>Full Name."""
        terms: List[ExtractedTerm] = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Skip introductory sentences
            lower = line.lower()
            if lower.startswith('for the purposes') or 'apply' in lower[:50]:
                continue
            if 'tr 21.905' in lower or 'precedence' in lower:
                continue

            # Tab-separated format
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    abbr, full_name = parts[0].strip(), parts[1].strip()
                    if self._is_valid_abbreviation(abbr) and full_name:
                        terms.append(ExtractedTerm(abbr, full_name, 'abbreviation', spec_id))
                        continue

            # Space-separated format
            m = self._abbr_space_pattern.match(line)
            if m:
                abbr, full_name = m.group(1).strip(), m.group(2).strip()
                if self._is_valid_abbreviation(abbr) and full_name:
                    terms.append(ExtractedTerm(abbr, full_name, 'abbreviation', spec_id))
        return terms

    def extract_definitions(self, content: str, spec_id: str) -> List[ExtractedTerm]:
        """Pattern: 'Term: definition text'."""
        terms: List[ExtractedTerm] = []
        for line in content.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            lower = line.lower()
            if lower.startswith('for the purposes') or 'apply' in lower[:50]:
                continue
            if 'tr 21.905' in lower or 'ts 23.501' in lower:
                continue
            term, definition = line.split(':', 1)
            term, definition = term.strip(), definition.strip()
            if self._is_valid_definition_term(term) and len(definition) > 10:
                terms.append(ExtractedTerm(term, definition, 'definition', spec_id))
        return terms

    @staticmethod
    def _is_valid_abbreviation(abbr: str) -> bool:
        if not abbr or len(abbr) < 2:
            return False
        if not (abbr[0].isupper() or abbr[0].isdigit()):
            return False
        valid_chars = sum(1 for c in abbr if c.isupper() or c.isdigit() or c in '-/')
        return valid_chars >= len(abbr) * 0.6

    # Junk-shaped definition heads. `extract_definitions` splits on the FIRST ':',
    # so any line that merely CONTAINS a colon becomes a term — which swallowed
    # ABNF grammar, OpenAPI/JSON keys and bulleted prose out of 29-series specs
    # (measured 2026-08-28: 470 of 29,934 Term nodes, all with 0 MENTIONS edges).
    # Each pattern below was checked against the live KG for false positives:
    # legitimate heads like '(S)Gi-LAN', 'MSK = MBMS Service Key' and
    # 'Universal Subscriber Identity Module (USIM)' must survive, so length,
    # '=' and '(' are deliberately NOT filtered on.
    _JUNK_TERM_PATTERNS = (
        re.compile(r'^["\u2018\u2019\u201c\u201d]'),  # '"basePath"', '"$ref"' — JSON/OpenAPI keys
        re.compile(r'[<>\\]'),                   # '<element ref="ntfIRPData' — XML fragments
        re.compile(r'^[-;/*]'),                  # '-\tAT commands' — bullet lines, ABNF alternatives
        re.compile(r'\t'),                       # a tab never occurs inside a real term head
        # ABNF rule head: 'Sbi-Lci-Header = "3gpp-Sbi-Lci'. Neither '=' nor '"'
        # can be banned alone ('MSK = MBMS Service Key' is a real term, and a
        # quote mid-string is harmless), but a quote AFTER an '=' only ever
        # appears in a grammar production.
        re.compile(r'=[^"]*"'),
    )

    @staticmethod
    def _is_valid_definition_term(term: str) -> bool:
        if not term or len(term) < 2 or len(term) > 100:
            return False
        has_letters = any(c.isalpha() for c in term)
        is_reference = term.startswith('[') and term.endswith(']')
        if not (has_letters and not is_reference):
            return False
        return not any(p.search(term) for p in TermExtractor._JUNK_TERM_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# Subject Classification (ported from subject_classifier.py)
# ─────────────────────────────────────────────────────────────────────────────

class Subject(Enum):
    """5 subjects matching the TeleQnA benchmark."""
    STANDARDS_SPECIFICATIONS = "Standards specifications"
    STANDARDS_OVERVIEW = "Standards overview"
    LEXICON = "Lexicon"
    RESEARCH_PUBLICATIONS = "Research publications"
    RESEARCH_OVERVIEW = "Research overview"


@dataclass
class SubjectClassification:
    subject: Subject
    confidence: float
    reason: str


class SubjectClassifier:
    """Classifies a chunk into one of the 5 Subjects."""

    STANDARDS_SPEC_KEYWORDS = [
        'procedure', 'ie ', 'information element', 'message', 'timer',
        'state machine', 'nas ', 'rrc ', 'ngap ', 'xnap ', 'f1ap ',
        'pdcp', 'rlc', 'mac ', 'phy ', 'harq', 'drb', 'srb',
        'service operation', 'qos flow', 'pdu session'
    ]
    STANDARDS_OVERVIEW_KEYWORDS = [
        'overview', 'architecture', 'introduction', 'general',
        'reference model', 'functional', 'deployment', 'use case',
        'service', 'feature', 'capability', 'scenario'
    ]
    LEXICON_KEYWORDS = [
        'abbreviation', 'definition', 'terminology', 'acronym',
        'vocabulary', 'glossary'
    ]
    RESEARCH_PUB_KEYWORDS = [
        'algorithm', 'optimization', 'machine learning', 'deep learning',
        'neural network', 'theorem', 'proof', 'simulation', 'experimental',
        'performance analysis', 'complexity', 'convergence'
    ]
    RESEARCH_OVERVIEW_KEYWORDS = [
        'survey', 'review', 'state of the art', 'trend', 'challenge',
        'future', 'evolution', 'comparison', 'taxonomy'
    ]
    SPEC_PATTERN = re.compile(
        r'TS[_\s]*\d+[\._]\d+|TR[_\s]*\d+[\._]\d+|3GPP\s+Release\s+\d+',
        re.IGNORECASE
    )

    def classify_chunk(self, chunk: dict) -> SubjectClassification:
        section_title = chunk.get('section_title', '').lower()
        content = chunk.get('content', '').lower()
        chunk_type = chunk.get('chunk_type', '').lower()
        spec_id = chunk.get('spec_id', '') or chunk.get('_spec_id', '')

        # Lexicon (highest confidence for abbreviation/definition)
        if chunk_type in ['abbreviation', 'definition']:
            return SubjectClassification(Subject.LEXICON, 0.95, f"chunk_type={chunk_type}")
        if any(kw in section_title for kw in self.LEXICON_KEYWORDS):
            return SubjectClassification(Subject.LEXICON, 0.9, "section title contains lexicon keyword")

        spec_score = sum(
            1 for kw in self.STANDARDS_SPEC_KEYWORDS
            if kw in section_title or kw in content[:500]
        )
        if spec_score >= 2 or any(kw in section_title for kw in ['procedure', 'ie ', 'message']):
            return SubjectClassification(
                Subject.STANDARDS_SPECIFICATIONS,
                min(0.7 + spec_score * 0.05, 0.95),
                f"matched {spec_score} standards spec keywords"
            )

        overview_score = sum(1 for kw in self.STANDARDS_OVERVIEW_KEYWORDS if kw in section_title)
        if overview_score >= 1:
            return SubjectClassification(
                Subject.STANDARDS_OVERVIEW,
                0.7 + overview_score * 0.1,
                "section title contains overview keyword"
            )

        research_score = sum(1 for kw in self.RESEARCH_PUB_KEYWORDS if kw in content[:1000])
        if research_score >= 2:
            return SubjectClassification(
                Subject.RESEARCH_PUBLICATIONS,
                min(0.6 + research_score * 0.1, 0.9),
                f"matched {research_score} research keywords"
            )

        if any(kw in section_title or kw in content[:500] for kw in self.RESEARCH_OVERVIEW_KEYWORDS):
            return SubjectClassification(Subject.RESEARCH_OVERVIEW, 0.7, "matched research overview keyword")

        if self.SPEC_PATTERN.search(spec_id):
            return SubjectClassification(Subject.STANDARDS_SPECIFICATIONS, 0.6, "default for 3GPP spec")
        return SubjectClassification(Subject.RESEARCH_OVERVIEW, 0.5, "default classification")


# 5 Subject taxonomy (constant)
_SUBJECT_TAXONOMY = [
    ('Standards specifications', 1, 'Specific 3GPP procedures, IEs, messages'),
    ('Standards overview',       2, 'Architecture, overview, introduction'),
    ('Lexicon',                  3, 'Abbreviations, definitions, terminology'),
    ('Research publications',    4, 'Algorithms, techniques, methods'),
    ('Research overview',        5, 'General concepts, surveys'),
]


# ─────────────────────────────────────────────────────────────────────────────
# KG Builder — main entry
# ─────────────────────────────────────────────────────────────────────────────

class KGBuilder:
    """Builds the Knowledge Graph from processed JSON files into Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self._term_extractor = TermExtractor()
        self._subject_classifier = SubjectClassifier()

    def close(self) -> None:
        self._driver.close()

    def verify_connection(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"[kg] Neo4j connection failed: {e}")
            return False

    def setup_schema(self) -> None:
        """Create constraints + indexes (idempotent)."""
        with self._driver.session() as s:
            for cypher in CYPHER_CONSTRAINTS + CYPHER_INDEXES:
                try:
                    s.run(cypher)
                except Exception:
                    # Ignore conflict with an existing constraint
                    pass
        print("[kg] Schema ready.")

    def clear(self, batch_size: int = 10000) -> None:
        """Delete all nodes + relationships in batches to avoid OOM/timeout on a
        large KG (a single-transaction DETACH DELETE usually fails >100k nodes)."""
        total_deleted = 0
        with self._driver.session() as s:
            while True:
                result = s.run(
                    f"""
                    MATCH (n)
                    WITH n LIMIT {batch_size}
                    DETACH DELETE n
                    RETURN count(*) AS deleted
                    """
                )
                deleted = result.single()["deleted"]
                if deleted == 0:
                    break
                total_deleted += deleted
                print(f"[kg] Deleted {total_deleted:,} nodes...", end="\r")

            # Verify — must reach 0 before printing "cleared"
            remaining = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]

        if remaining == 0:
            print(f"\n[kg] Graph cleared ({total_deleted:,} nodes deleted).")
        else:
            raise RuntimeError(
                f"[kg] Clear failed: {remaining:,} nodes remain after deleting {total_deleted:,}"
            )

    def drop_all_schema(self) -> None:
        """Drop all constraints + indexes (including vector + legacy).
        A later setup_schema() recreates exactly the ones needed."""
        dropped_c, dropped_i = 0, 0
        with self._driver.session() as s:
            constraints = list(s.run("SHOW CONSTRAINTS YIELD name RETURN name"))
            for c in constraints:
                try:
                    s.run(f"DROP CONSTRAINT {c['name']} IF EXISTS")
                    dropped_c += 1
                except Exception as e:
                    print(f"[kg] ⚠ Cannot drop constraint {c['name']}: {e}")

            # Drop ALL indexes (including vector index + legacy)
            indexes = list(s.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name"))
            for i in indexes:
                try:
                    s.run(f"DROP INDEX {i['name']} IF EXISTS")
                    dropped_i += 1
                except Exception as e:
                    print(f"[kg] ⚠ Cannot drop index {i['name']}: {e}")
        print(f"[kg] Dropped {dropped_c} constraints, {dropped_i} indexes.")

    def clean_all(self, batch_size: int = 10000) -> None:
        """Full wipe: drop schema + delete data.
        Does NOT clear zombie label/type tokens (that needs a container restart).
        Use restart_neo4j_container() for a 100% clean."""
        print("[kg] === FULL CLEAN ===")
        self.drop_all_schema()
        self.clear(batch_size=batch_size)

    def load_json_dir(self, json_dir: Path) -> int:
        """Full pipeline: load JSON → create nodes + edges → classify subject → parent_section.
        Returns the total number of chunks."""
        json_dir = Path(json_dir)
        files = sorted(json_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {json_dir}")

        documents, chunks = self._load_json_files(files)
        print(f"[kg] Loaded {len(documents)} documents, {len(chunks)} chunks")

        self._create_documents(documents)
        self._create_chunks(chunks)
        self._create_contains_edges()
        self._create_references_spec_edges(chunks)
        n_ref_chunk = self._create_references_chunk_edges(chunks)
        print(f"[kg] Created {n_ref_chunk} REFERENCES_CHUNK edges")
        n_terms = self._create_terms(chunks)
        print(f"[kg] Created {n_terms} Term nodes")
        self._create_subjects(chunks)
        n_parent = self._create_parent_section_edges()
        print(f"[kg] Created {n_parent} PARENT_SECTION edges")
        n_mentions = self._create_mentions_edges(chunks)
        print(f"[kg] Created {n_mentions} MENTIONS edges")

        # ServiceOperation, Term relations, Parameter, Step, Message
        # (see docs/de_xuat_cai_thien_kg_quan_he.md — P0-P3). Order is mandatory:
        # service_operations BEFORE term_relations (semantic_type needs PROVIDED_BY).
        n_svc_op = self._create_service_operations(chunks)
        print(f"[kg] Created {n_svc_op} DESCRIBES_OPERATION edges")
        n_cooccur = self._create_term_relations()
        print(f"[kg] Created {n_cooccur} CO_OCCURS_WITH edges")
        n_params = self._create_parameters(chunks)
        print(f"[kg] Created {n_params} Parameter nodes")
        n_steps = self._create_procedure_steps(chunks)
        print(f"[kg] Created {n_steps} Step nodes")
        n_messages = self._create_messages(chunks)
        print(f"[kg] Created {n_messages} Message nodes")
        n_values = self._create_standardized_values(chunks)
        print(f"[kg] Created {n_values} StandardizedValue rows")

        return len(chunks)

    def _load_json_files(self, files: List[Path]) -> Tuple[Dict[str, dict], List[dict]]:
        documents: Dict[str, dict] = {}
        chunks: List[dict] = []
        for f in tqdm(files, desc="[kg] Loading JSON"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[kg] Skip {f.name}: {e}")
                continue
            spec_id = data["metadata"]["specification_id"]
            documents[spec_id] = data
            for chunk in data.get("chunks", []):
                chunk["_spec_id"] = spec_id
                chunks.append(chunk)
        return documents, chunks

    def _create_documents(self, documents: Dict[str, dict]) -> None:
        with self._driver.session() as s:
            for spec_id, data in tqdm(documents.items(), desc="[kg] Documents"):
                meta = data.get("metadata", {})
                export = data.get("export_info", {})
                s.run(
                    """
                    MERGE (d:Document {spec_id: $spec_id})
                    SET d.title = $title,
                        d.version = $version,
                        d.total_chunks = $total_chunks
                    """,
                    spec_id=spec_id,
                    title=meta.get("title", spec_id),
                    version=meta.get("version", ""),
                    total_chunks=export.get("total_chunks", 0),
                )

    @staticmethod
    def _batched(rows: List[dict], size: int = WRITE_BATCH_SIZE):
        """Yield rows in batches of `size` elements (for UNWIND)."""
        for i in range(0, len(rows), size):
            yield rows[i : i + size]

    def _create_chunks(self, chunks: List[dict]) -> None:
        rows = []
        for chunk in chunks:
            content_meta = chunk.get("content_metadata", {})
            rows.append({
                "chunk_id": chunk["chunk_id"],
                "spec_id": chunk["_spec_id"],
                "section_id": chunk.get("section_id", ""),
                "section_title": chunk.get("section_title", ""),
                "content": chunk.get("content", ""),
                "chunk_type": chunk.get("chunk_type", "general"),
                "word_count": content_meta.get("word_count", 0),
                "complexity_score": content_meta.get("complexity_score", 0.0),
                "key_terms": content_meta.get("key_terms", []),
            })

        with self._driver.session() as s:
            for batch in tqdm(
                list(self._batched(rows)), desc="[kg] Chunks", unit="batch"
            ):
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (c:Chunk {chunk_id: row.chunk_id})
                    SET c.spec_id = row.spec_id,
                        c.section_id = row.section_id,
                        c.section_title = row.section_title,
                        c.content = row.content,
                        c.chunk_type = row.chunk_type,
                        c.word_count = row.word_count,
                        c.complexity_score = row.complexity_score,
                        c.key_terms = row.key_terms
                    """,
                    rows=batch,
                )

    def _create_contains_edges(self) -> None:
        with self._driver.session() as s:
            s.run(
                """
                MATCH (d:Document), (c:Chunk)
                WHERE d.spec_id = c.spec_id
                MERGE (d)-[:CONTAINS]->(c)
                """
            )

    def _create_references_spec_edges(self, chunks: List[dict]) -> None:
        # MATCH Document inside the UNWIND ensures an edge is created only when the
        # target document exists.
        rows = []
        for chunk in chunks:
            source_id = chunk["chunk_id"]
            for ref in chunk.get("cross_references", {}).get("external", []):
                target_spec = ref.get("target_spec", "")
                if not target_spec:
                    continue
                ref_uid = hashlib.md5(
                    f"{source_id}_{target_spec}_{ref.get('ref_id', '')}".encode()
                ).hexdigest()[:10]
                rows.append({
                    "source_id": source_id,
                    "target_spec": target_spec,
                    "ref_uid": ref_uid,
                    "ref_id": ref.get("ref_id", ""),
                    "ref_type": ref.get("ref_type", ""),
                    "confidence": ref.get("confidence", 0.0),
                })

        with self._driver.session() as s:
            for batch in tqdm(
                list(self._batched(rows)), desc="[kg] REFERENCES_SPEC", unit="batch"
            ):
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (src:Chunk {chunk_id: row.source_id})
                    MATCH (dst:Document {spec_id: row.target_spec})
                    MERGE (src)-[r:REFERENCES_SPEC {ref_uid: row.ref_uid}]->(dst)
                    SET r.ref_id = row.ref_id,
                        r.ref_type = row.ref_type,
                        r.confidence = row.confidence
                    """,
                    rows=batch,
                )

    # Filter for external refs: keep only ref_type='clause' (per the Apr 2026
    # feasibility test — 15.7% gross match rate falls into the conservative bucket).
    _EXT_ALLOWED_REF_TYPES = {"clause"}
    _EXT_CONFIDENCE_THRESHOLD = 0.7

    def _create_references_chunk_edges(self, chunks: List[dict]) -> int:
        """Create Chunk→Chunk edges from cross_references.internal + .external.

        - Internal: match within the same spec, set r.is_external = false.
        - External: match cross-spec with filter ref_type ∈ _EXT_ALLOWED_REF_TYPES
          and confidence ≥ _EXT_CONFIDENCE_THRESHOLD, set r.is_external = true.

        3-tier matching: exact section_id → prefix → parent (strip suffix '-N')."""
        # Build a section index for ALL specs once — serves both internal + external
        section_index_per_spec: Dict[str, Dict[str, str]] = defaultdict(dict)
        for c in chunks:
            section_index_per_spec[c["_spec_id"]][c.get("section_id", "")] = c["chunk_id"]

        # Group chunks by spec to keep per-spec log structure
        spec_chunks: Dict[str, List[dict]] = defaultdict(list)
        for chunk in chunks:
            spec_chunks[chunk["_spec_id"]].append(chunk)

        total_created = 0
        for spec_id, spec_chunk_list in tqdm(spec_chunks.items(), desc="[kg] REFERENCES_CHUNK"):
            same_spec_index = section_index_per_spec[spec_id]
            refs_to_create: List[dict] = []

            for chunk in spec_chunk_list:
                source_id = chunk["chunk_id"]
                cross_refs = chunk.get("cross_references", {})

                # Internal refs (same spec)
                for ref in cross_refs.get("internal", []):
                    ref_id = ref.get("ref_id", "")
                    if not ref_id:
                        continue
                    target_id = self._match_ref_to_chunk(ref_id, source_id, same_spec_index)
                    if target_id:
                        refs_to_create.append({
                            "source": source_id,
                            "target": target_id,
                            "is_external": False,
                            "ref_type": ref.get("ref_type", "clause"),
                            "ref_id": ref_id,
                            "confidence": float(ref.get("confidence", 1.0)),
                        })

                # External refs (cross-spec, filtered)
                for ref in cross_refs.get("external", []):
                    ref_id = ref.get("ref_id", "")
                    target_spec = ref.get("target_spec", "")
                    ref_type = ref.get("ref_type", "")
                    conf = float(ref.get("confidence", 0.0))

                    if not ref_id or not target_spec:
                        continue
                    if ref_type not in self._EXT_ALLOWED_REF_TYPES:
                        continue
                    if conf < self._EXT_CONFIDENCE_THRESHOLD:
                        continue

                    target_index = section_index_per_spec.get(target_spec)
                    if not target_index:
                        # Target spec not loaded into the KG → skip (REFERENCES_SPEC still covers doc-level)
                        continue

                    target_id = self._match_ref_to_chunk(ref_id, source_id, target_index)
                    if target_id:
                        refs_to_create.append({
                            "source": source_id,
                            "target": target_id,
                            "is_external": True,
                            "ref_type": ref_type,
                            "ref_id": ref_id,
                            "confidence": conf,
                        })

            if refs_to_create:
                with self._driver.session() as s:
                    s.run(
                        """
                        UNWIND $refs AS ref
                        MATCH (src:Chunk {chunk_id: ref.source})
                        MATCH (tgt:Chunk {chunk_id: ref.target})
                        MERGE (src)-[r:REFERENCES_CHUNK]->(tgt)
                        SET r.is_external = ref.is_external,
                            r.ref_type = ref.ref_type,
                            r.ref_id = ref.ref_id,
                            r.confidence = ref.confidence
                        """,
                        refs=refs_to_create,
                    )
                total_created += len(refs_to_create)
        return total_created

    @staticmethod
    def _match_ref_to_chunk(
        ref_id: str, source_chunk_id: str, section_index: Dict[str, str]
    ) -> Optional[str]:
        """3-tier matching: exact → prefix → parent (strip '-N' suffix)."""
        # Tier 1: exact match
        if ref_id in section_index:
            target = section_index[ref_id]
            if target != source_chunk_id:
                return target
        # Tier 2: prefix match — ref "5.2" matches section "5.2.1"
        for sid, cid in section_index.items():
            if sid.startswith(ref_id + ".") and cid != source_chunk_id:
                return cid
        # Tier 3: parent match — ref "5.2.3-1" → strip "-1" → "5.2.3"
        if "-" in ref_id:
            parent_ref = ref_id.split("-")[0]
            if parent_ref in section_index:
                target = section_index[parent_ref]
                if target != source_chunk_id:
                    return target
        return None

    def _create_terms(self, chunks: List[dict]) -> int:
        """Extract Terms from abbreviation/definition sections, dedupe + merge across specs."""
        term_dict: Dict[str, dict] = {}

        for chunk in chunks:
            section_title = chunk.get("section_title", "").lower()
            content = chunk.get("content", "")
            spec_id = chunk["_spec_id"]

            if 'abbreviation' in section_title:
                terms = self._term_extractor.extract_abbreviations(content, spec_id)
                self._merge_terms(term_dict, terms)
            elif 'definition' in section_title:
                terms = self._term_extractor.extract_definitions(content, spec_id)
                self._merge_terms(term_dict, terms)

        # term_rows for Term nodes, defined_in_rows for DEFINED_IN edges
        # (flattened over source_specs).
        term_rows = []
        defined_in_rows = []
        for abbr, td in term_dict.items():
            term_rows.append({
                "abbr": abbr,
                "full_name": td['full_name'],
                "term_type": td['term_type'],
                "source_specs": td['source_specs'],
                "primary_spec": td['primary_spec'],
            })
            for spec_id in td['source_specs']:
                defined_in_rows.append({"abbr": abbr, "spec_id": spec_id})

        # Write to Neo4j in batches. If a batch fails, DON'T drop all 5000 terms in
        # it (that's what once made AMF/SMF/NRF silently disappear — they shared a
        # batch with one bad row and the whole UNWIND was lost). Instead fall back
        # to writing that batch one term at a time, so only the genuinely-bad row
        # is skipped and everything else in the batch still lands.
        _TERM_MERGE = """
            UNWIND $rows AS row
            MERGE (t:Term {abbreviation: row.abbr})
            SET t.full_name = row.full_name,
                t.term_type = row.term_type,
                t.source_specs = row.source_specs,
                t.primary_spec = row.primary_spec
        """
        with self._driver.session() as s:
            for batch in tqdm(
                list(self._batched(term_rows)), desc="[kg] Term nodes", unit="batch"
            ):
                try:
                    s.run(_TERM_MERGE, rows=batch)
                except Exception as e:
                    print(
                        f"[kg] Term batch write failed ({len(batch)} terms) — "
                        f"retrying per-term: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    for row in batch:
                        try:
                            s.run(_TERM_MERGE, rows=[row])
                        except Exception as e2:
                            print(
                                f"[kg]   dropped Term '{row['abbr']}': "
                                f"{type(e2).__name__}: {e2}",
                                file=sys.stderr,
                            )

            for batch in tqdm(
                list(self._batched(defined_in_rows)),
                desc="[kg] DEFINED_IN", unit="batch",
            ):
                try:
                    s.run(
                        """
                        UNWIND $rows AS row
                        MATCH (t:Term {abbreviation: row.abbr})
                        MATCH (d:Document {spec_id: row.spec_id})
                        MERGE (t)-[:DEFINED_IN]->(d)
                        """,
                        rows=batch,
                    )
                except Exception as e:
                    print(
                        f"[kg] DEFINED_IN batch write failed ({len(batch)} edges): "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
        return len(term_rows)

    @staticmethod
    def _merge_terms(term_dict: Dict[str, dict], terms: List[ExtractedTerm]) -> None:
        """Merge terms with 5G priority resolution.
        Conflict order:
          1. 5G beats legacy
          2. Both 5G & differ → most 5G citations (tie: lexically smallest spec)
          3. Otherwise first-write-wins
        """
        def is_5g(spec: str) -> bool:
            return any(spec.startswith(p) for p in _5G_SPEC_PREFIXES)

        for term in terms:
            abbr = term.abbreviation
            if abbr not in term_dict:
                term_dict[abbr] = {
                    'abbreviation': abbr,
                    'full_name': term.full_name,
                    'term_type': term.term_type,
                    'source_specs': [term.source_spec],
                    'primary_spec': term.source_spec,
                    'definitions': {term.full_name: [term.source_spec]},
                }
                continue

            entry = term_dict[abbr]
            if term.source_spec not in entry['source_specs']:
                entry['source_specs'].append(term.source_spec)

            defs = entry.setdefault(
                'definitions', {entry['full_name']: [entry['primary_spec']]}
            )
            defs.setdefault(term.full_name, []).append(term.source_spec)

            existing_5g = is_5g(entry['primary_spec'])
            new_5g = is_5g(term.source_spec)

            if new_5g and not existing_5g:
                entry['full_name'] = term.full_name
                entry['primary_spec'] = term.source_spec
            elif new_5g and existing_5g and term.full_name != entry['full_name']:
                new_5g_cites = sum(1 for s in defs[term.full_name] if is_5g(s))
                exist_5g_cites = sum(1 for s in defs[entry['full_name']] if is_5g(s))
                if (new_5g_cites > exist_5g_cites or
                        (new_5g_cites == exist_5g_cites
                         and term.source_spec < entry['primary_spec'])):
                    entry['full_name'] = term.full_name
                    entry['primary_spec'] = term.source_spec

    def _create_subjects(self, chunks: List[dict]) -> int:
        """Create the 5 Subject nodes + classify every chunk + create HAS_SUBJECT edges."""
        with self._driver.session() as s:
            for name, priority, description in _SUBJECT_TAXONOMY:
                s.run(
                    """
                    MERGE (s:Subject {name: $name})
                    SET s.priority = $priority, s.description = $description
                    """,
                    name=name, priority=priority, description=description,
                )

            rows = []
            for chunk in chunks:
                cls = self._subject_classifier.classify_chunk(chunk)
                rows.append({
                    "chunk_id": chunk["chunk_id"],
                    "subject": cls.subject.value,
                    "confidence": cls.confidence,
                })

            for batch in tqdm(
                list(self._batched(rows)), desc="[kg] HAS_SUBJECT", unit="batch"
            ):
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Chunk {chunk_id: row.chunk_id})
                    SET c.subject = row.subject,
                        c.subject_confidence = row.confidence
                    """,
                    rows=batch,
                )
            classified = len(rows)

            s.run(
                """
                MATCH (c:Chunk), (s:Subject)
                WHERE c.subject = s.name
                MERGE (c)-[:HAS_SUBJECT]->(s)
                """
            )
        return classified

    def _create_parent_section_edges(self) -> int:
        """Mark chunks that are parent sections + create child→nearest_parent edges.
        Relationship is based on section_id prefix (e.g. '6.3.1.1' → parent '6.3.1')."""
        with self._driver.session() as s:
            # Mark chunks that have at least one child
            s.run(
                """
                MATCH (parent:Chunk)
                WHERE EXISTS {
                    MATCH (child:Chunk)
                    WHERE child.spec_id = parent.spec_id
                      AND child.section_id STARTS WITH parent.section_id + '.'
                }
                SET parent.is_parent_section = true
                """
            )

            # Create an edge to the nearest parent (no intermediate chunk between)
            result = s.run(
                """
                MATCH (child:Chunk), (parent:Chunk)
                WHERE child.spec_id = parent.spec_id
                  AND child.section_id <> parent.section_id
                  AND child.section_id STARTS WITH parent.section_id + '.'
                  AND NOT EXISTS {
                    MATCH (mid:Chunk)
                    WHERE mid.spec_id = child.spec_id
                      AND child.section_id STARTS WITH mid.section_id + '.'
                      AND mid.section_id STARTS WITH parent.section_id + '.'
                      AND mid.section_id <> child.section_id
                      AND mid.section_id <> parent.section_id
                  }
                MERGE (child)-[:PARENT_SECTION]->(parent)
                RETURN count(*) AS created
                """
            )
            return result.single()["created"]

    def _create_mentions_edges(self, chunks: List[dict]) -> int:
        """Create (Chunk)-[:MENTIONS]->(Term) edges from `content_metadata.key_terms`
        intersected with Term.abbreviation. Filter noise by document-frequency band
        [MENTIONS_MIN_DF, MENTIONS_MAX_DF] — drop ubiquitous terms (IDF≈0) and overly
        rare ones. Property `r.df` = document-frequency (used as an IDF weight).
        Must run AFTER _create_terms (needs Terms to exist) and _create_chunks."""
        df: Dict[str, int] = defaultdict(int)
        for c in chunks:
            for kt in (c.get("content_metadata", {}).get("key_terms") or []):
                df[kt] += 1
        # Set of Term.abbreviation already in the KG (only create edges to existing Terms)
        with self._driver.session() as s:
            term_set = {r["a"] for r in s.run("MATCH (t:Term) RETURN t.abbreviation AS a")}
        keep = {
            kt: n for kt, n in df.items()
            if kt in term_set and MENTIONS_MIN_DF <= n <= MENTIONS_MAX_DF
        }
        rows = [
            {"id": c["chunk_id"], "abbr": kt, "df": keep[kt]}
            for c in chunks
            for kt in (c.get("content_metadata", {}).get("key_terms") or [])
            if kt in keep
        ]
        with self._driver.session() as s:
            for batch in tqdm(
                list(self._batched(rows)), desc="[kg] MENTIONS", unit="batch"
            ):
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Chunk {chunk_id: row.id})
                    MATCH (t:Term {abbreviation: row.abbr})
                    MERGE (c)-[r:MENTIONS]->(t)
                    SET r.df = row.df
                    """,
                    rows=batch,
                )
        return len(rows)

    def _create_service_operations(self, chunks: List[dict]) -> int:
        """Create ServiceOperation nodes from the SBI pattern `N<nf>_<Service>_<Operation>`
        in Chunk.content, + edges (so)-[:PROVIDED_BY]->(t:Term) + (c)-[:DESCRIBES_OPERATION]->(so).
        Keep only operations whose NF-prefix matches an EXISTING Term.abbreviation (drops
        regex noise from random CamelCase_Case phrases). Must run AFTER _create_terms.
        Returns the number of DESCRIBES_OPERATION edges created."""
        with self._driver.session() as s:
            term_set = {r["a"] for r in s.run("MATCH (t:Term) RETURN t.abbreviation AS a")}

        op_meta: Dict[str, dict] = {}
        op_df: Dict[str, int] = defaultdict(int)
        chunk_edges: List[dict] = []
        for c in chunks:
            content = c.get("content") or ""
            found = set(SVC_OP_RE.findall(content))
            if not found:
                continue
            for name in found:
                nf, service, operation = name.split("_", 2)
                nf = nf[1:].upper()
                if nf not in term_set:
                    continue
                op_meta.setdefault(name, {"nf": nf, "service": service, "operation": operation})
                op_df[name] += 1
                chunk_edges.append({"chunk_id": c["chunk_id"], "name": name})

        op_rows = [{"name": name, **meta, "df": op_df[name]} for name, meta in op_meta.items()]
        with self._driver.session() as s:
            if op_rows:
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (so:ServiceOperation {name: row.name})
                    SET so.nf_prefix = row.nf, so.service = row.service,
                        so.operation = row.operation, so.df = row.df
                    WITH so, row
                    MATCH (t:Term {abbreviation: row.nf})
                    MERGE (so)-[:PROVIDED_BY]->(t)
                    """,
                    rows=op_rows,
                )
            for batch in tqdm(
                list(self._batched(chunk_edges)), desc="[kg] ServiceOperation", unit="batch"
            ):
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Chunk {chunk_id: row.chunk_id})
                    MATCH (so:ServiceOperation {name: row.name})
                    MERGE (c)-[:DESCRIBES_OPERATION]->(so)
                    """,
                    rows=batch,
                )
        return len(chunk_edges)

    def _create_term_relations(self, min_weight: int = COOCCUR_MIN_WEIGHT) -> int:
        """Term.semantic_type='network_function' (from PROVIDED_BY — direct evidence,
        not a heuristic guess) + (Term)-[:CO_OCCURS_WITH {weight}]->(Term) derived from
        MENTIONS co-occurrence (2 Terms MENTIONS the same Chunk). Pure server-side Cypher,
        no JSON data needed. Must run AFTER _create_service_operations (needs PROVIDED_BY)
        and _create_mentions_edges (needs MENTIONS). Returns the number of CO_OCCURS_WITH
        edges created."""
        with self._driver.session() as s:
            s.run(
                "MATCH (t:Term)<-[:PROVIDED_BY]-(:ServiceOperation) "
                "SET t.semantic_type = 'network_function'"
            )
            result = s.run(
                """
                MATCH (t1:Term)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(t2:Term)
                WHERE t1.abbreviation < t2.abbreviation
                WITH t1, t2, count(DISTINCT c) AS weight
                WHERE weight >= $min_weight
                MERGE (t1)-[r:CO_OCCURS_WITH]->(t2)
                SET r.weight = weight
                RETURN count(*) AS n
                """,
                min_weight=min_weight,
            )
            return result.single()["n"]

    @staticmethod
    def _find_col(headers: List[str], keywords: set) -> Optional[int]:
        headers_l = [h.strip().lower() for h in headers]
        for i, h in enumerate(headers_l):
            if h in keywords:
                return i
        for i, h in enumerate(headers_l):
            if any(k in h for k in keywords):
                return i
        return None

    def _create_parameters(self, chunks: List[dict]) -> int:
        """Create Parameter nodes from Chunk.tables (IE/attribute/parameter tables —
        detected via 'name'/'description' column names and variants). param_id is scoped
        per chunk (NOT globally unique by name — column names like 'type'/'period' are too
        generic and collide with different meanings across tables). Returns the number of
        Parameter nodes created."""
        rows = []
        for c in chunks:
            for t_idx, table in enumerate(c.get("tables") or []):
                headers = table.get("headers") or []
                ni = self._find_col(headers, PARAM_NAME_HEADERS)
                di = self._find_col(headers, PARAM_DESC_HEADERS)
                if ni is None or di is None or ni == di:
                    continue
                for r_idx, row in enumerate(table.get("rows") or []):
                    if len(row) <= max(ni, di):
                        continue
                    name = (row[ni] or "").strip()
                    desc = (row[di] or "").strip()
                    if not name or not desc:
                        continue
                    if name.upper().startswith("NOTE") or len(name) > PARAM_MAX_NAME_LEN:
                        continue
                    rows.append({
                        "param_id": f"{c['chunk_id']}::{t_idx}::{r_idx}",
                        "name": name,
                        "description": desc,
                        "spec_id": c["_spec_id"],
                        "chunk_id": c["chunk_id"],
                    })

        with self._driver.session() as s:
            for batch in tqdm(list(self._batched(rows)), desc="[kg] Parameters", unit="batch"):
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (p:Parameter {param_id: row.param_id})
                    SET p.name = row.name, p.description = row.description, p.spec_id = row.spec_id
                    WITH p, row
                    MATCH (c:Chunk {chunk_id: row.chunk_id})
                    MERGE (p)-[:DEFINED_IN_TABLE]->(c)
                    """,
                    rows=batch,
                )
        return len(rows)

    # ── Layer C: Concept / StandardizedValue ─────────────────────────────────

    @staticmethod
    def _clean_cell(s: str) -> str:
        s = (s or "").replace("\xa0", " ")
        s = _NOTE_FRAG_RE.sub("", s)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _concept_from_title(section_title: str) -> Optional[str]:
        for rx in CONCEPT_TITLE_RES:
            m = rx.match((section_title or "").strip())
            if m:
                return re.sub(r"\s+", " ", m.group("name")).strip()
        return None

    @classmethod
    def _accept_concept(cls, concept: Optional[str]) -> Optional[str]:
        """Apply the audit-derived concept hygiene: stoplist (header adjectives),
        rename map (split-off header variants), slashed-compound rejection.
        Returns the canonical concept name or None to skip the table."""
        if not concept:
            return None
        concept = re.sub(r"\s+", " ", concept).strip()
        if not concept or "/" in concept or len(concept) > 40:
            return None
        canon = VALUE_CONCEPT_RENAMES.get(concept.upper().replace(" ", ""), concept)
        if canon.upper() in VALUE_CONCEPT_STOPLIST:
            return None
        return canon

    @classmethod
    def classify_value_table(cls, section_title: str, headers: List[str]) -> Optional[dict]:
        """Classify one table against the three standardized-value shapes.
        Returns {'kind', 'concept', 'code_col', 'name_col', 'desc_col', 'attr_cols'}
        or None. Pure function of (title, headers) so it is unit-testable and the
        dry-run can reuse it without touching Neo4j."""
        hdrs = [cls._clean_cell(h) for h in (headers or [])]
        low = [h.lower() for h in hdrs]
        if len(hdrs) < 2:
            return None
        # Sections titled "Examples of usage" hold illustrative sample values,
        # not standardized assignments (audit: the PROTOCOL false concept).
        if "example" in (section_title or "").lower():
            return None
        # Transposed-matrix guard: data-like column headers mean the extractor
        # read a value grid sideways (audit: HARQ-ACK/FIELD/ONPOWER artifacts).
        tail = [h for h in hdrs[1:] if h]
        if tail and sum(1 for h in tail if _DATAISH_HDR_RE.match(h)) * 2 >= len(tail):
            return None

        desc_col = next((i for i, h in enumerate(low) if h in VALUE_DESC_HEADERS), None)

        # (a) enumeration: "Enumeration value | Description [| Applicability]"
        if "enumeration value" in low and desc_col is not None:
            concept = cls._accept_concept(cls._concept_from_title(section_title))
            if concept:
                return {"kind": "enumeration", "concept": concept,
                        "code_col": low.index("enumeration value"),
                        "name_col": None, "desc_col": desc_col, "attr_cols": []}
            # Known recall gap (audit finding 6): ~63 genuine enum tables whose
            # section titles fit no whitelist pattern are skipped — precision-first.
            return None

        # (b) value_set: a "<X> value" header + a description-ish column.
        for i, h in enumerate(low):
            m = re.match(r"^([a-z0-9/\- ]{2,30}?)\s+values?$", h)
            if m and desc_col is not None and i != desc_col:
                # Concept from the header minus the trailing "value(s)" — original
                # case, upper-cased when it looks like an abbreviation ("SST").
                concept = re.sub(r"\s+values?$", "", hdrs[i], flags=re.IGNORECASE).strip()
                if len(concept) <= 6:
                    concept = concept.upper()
                concept = cls._accept_concept(concept or cls._concept_from_title(section_title))
                if not concept:
                    return None
                # 3GPP convention places the name column LEFT of its value column
                # ("Slice/Service type | SST value"); a column to the right is an
                # auxiliary attribute (measured: TDoc "Used for" polluting names).
                name_col = next((j for j in range(i)
                                 if j != desc_col and hdrs[j]), None)
                return {"kind": "value_set", "concept": concept,
                        "code_col": i, "name_col": name_col,
                        "desc_col": desc_col, "attr_cols": []}

        # (c) attribute_map: first column "<X>Value" + >=2 named attribute columns.
        m = re.match(r"^([a-z0-9\-]{2,12})values?$", low[0].replace(" ", "")) if low else None
        if m and len(hdrs) >= 3:
            concept = cls._accept_concept(m.group(1).upper())
            if not concept:
                return None
            attr_cols = [j for j in range(1, len(hdrs)) if 0 < len(hdrs[j]) <= 40]
            if len(attr_cols) >= 2:
                return {"kind": "attribute_map", "concept": concept,
                        "code_col": 0, "name_col": None, "desc_col": None,
                        "attr_cols": attr_cols}
        return None

    @classmethod
    def collect_standardized_value_rows(
        cls, chunks: List[dict]
    ) -> Tuple[List[dict], Dict[str, Dict]]:
        """Classify + row-filter every table WITHOUT touching Neo4j. Single
        implementation shared by the write path and the dry-run audit — the
        2026-08-09 review caught the dry-run counting raw rows while the write
        path filtered, so the audit signed off on a different dataset than the
        one ingested. Returns (rows, concept_meta)."""
        rows: List[dict] = []
        concept_meta: Dict[str, Dict] = {}
        for c in chunks:
            spec = c["_spec_id"]
            if any(spec.startswith(p) for p in VALUE_SPEC_BLOCKLIST_PREFIXES):
                continue
            for t_idx, table in enumerate(c.get("tables") or []):
                spec_tbl = cls.classify_value_table(
                    c.get("section_title") or "", table.get("headers") or [])
                if not spec_tbl:
                    continue
                headers = [cls._clean_cell(h) for h in (table.get("headers") or [])]
                # Duplicate header names ("Cause value|Cause|Cause|Diag-") would
                # silently overwrite each other in the attrs dict — disambiguate.
                seen_hdr: Dict[str, int] = {}
                uniq_headers: List[str] = []
                for h in headers:
                    n = seen_hdr.get(h, 0)
                    seen_hdr[h] = n + 1
                    uniq_headers.append(h if n == 0 else f"{h} ({n + 1})")
                t_rows = table.get("rows") or []
                # Enumeration identifiers (ALL_CAPS_UNDERSCORE) run long; only
                # value_set/attribute_map codes are short-identifier shaped.
                code_max = VALUE_ENUM_CODE_MAX_LEN \
                    if spec_tbl["kind"] == "enumeration" else VALUE_CODE_MAX_LEN
                cand: List[dict] = []
                prose_hits = 0
                for r_idx, row in enumerate(t_rows):
                    cells = [cls._clean_cell(x) for x in row]
                    # Group sub-header rows inside enum tables carry one cell.
                    if sum(1 for x in cells if x) <= 1:
                        continue
                    ci = spec_tbl["code_col"]
                    if len(cells) <= ci:
                        continue
                    # Strip wrapping quotes + ASN.1 ordinal suffix ("cellID(1)").
                    code = cells[ci].strip('"').strip("'")
                    code = re.sub(r"\(\d+\)$", "", code).strip()
                    if not code or code.upper().startswith("NOTE") \
                            or len(code) > code_max:
                        continue
                    name = code
                    if spec_tbl["name_col"] is not None and len(cells) > spec_tbl["name_col"]:
                        name = cells[spec_tbl["name_col"]] or code
                    if name.upper().startswith("NOTE") or len(name) > PARAM_MAX_NAME_LEN:
                        continue
                    desc = ""
                    if spec_tbl["desc_col"] is not None and len(cells) > spec_tbl["desc_col"]:
                        desc = cells[spec_tbl["desc_col"]]
                    attrs = {uniq_headers[j]: cells[j] for j in spec_tbl["attr_cols"]
                             if len(cells) > j and cells[j]}
                    if spec_tbl["kind"] == "attribute_map":
                        name = f"{spec_tbl['concept']} {code}"
                        desc = desc or "; ".join(f"{k}: {v}" for k, v in attrs.items())
                    if desc and " " in desc and len(desc) >= 8:
                        prose_hits += 1
                    cand.append({
                        "value_id": f"{c['chunk_id']}::{t_idx}::{r_idx}",
                        "concept": spec_tbl["concept"], "name": name, "code": code,
                        "description": desc,
                        "attrs_json": json.dumps(attrs, ensure_ascii=False) if attrs else "",
                        "spec_id": spec, "chunk_id": c["chunk_id"],
                    })
                if not cand:
                    continue
                # Prose gate applies to value_set ONLY: its description column is
                # definitional prose by construction. Enumerations legitimately
                # carry empty/short descriptions (ReservPriority PRIO_1..15), and
                # attribute_map rows have no desc column — the audit measured the
                # old any-kind gate killing real enums while the RF grids it
                # targeted entered via attribute_map and bypassed it anyway
                # (those are now stopped by stoplist/blocklist/transposed guard).
                # Boundary: exactly-50% prose PASSES (2*hits >= n).
                if spec_tbl["kind"] == "value_set" \
                        and prose_hits * 2 < len(cand):
                    continue
                rows.extend(cand)
                meta = concept_meta.setdefault(
                    spec_tbl["concept"], {"kind": spec_tbl["kind"], "specs": set()})
                meta["specs"].add(spec)
        return rows, concept_meta

    def _create_standardized_values(self, chunks: List[dict]) -> int:
        """Layer C: (Concept)-[:HAS_VALUE]->(StandardizedValue)-[:DEFINED_IN_TABLE]->(Chunk),
        plus (StandardizedValue)-[:DENOTES]->(Term) / (Concept)-[:ABOUT]->(Term) where the
        name matches an existing Term.abbreviation. Fully deterministic — table-header
        classification only, no LLM, no link prediction: every node carries provenance to
        the exact defining chunk. StandardizedValue is per-row scoped (value_id) and NEVER
        merged globally by name (the Parameter lesson: short names collide across tables);
        only Concept merges globally by name. Known limits (2026-08-09 audit): Concept.kind
        is first-write-wins on kind collision (deterministic — files are sorted), and a
        cross-protocol registry name like CAUSE groups per-protocol tables under one
        Concept; values stay chunk-scoped so retrieval precision is unaffected.
        Returns the number of value rows written."""
        rows, concept_meta = self.collect_standardized_value_rows(chunks)

        with self._driver.session() as s:
            concepts = [{"name": n, "kind": m["kind"], "source_specs": sorted(m["specs"])}
                        for n, m in concept_meta.items()]
            for batch in self._batched(concepts):
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (co:Concept {name: row.name})
                    SET co.kind = row.kind, co.source_specs = row.source_specs
                    """,
                    rows=batch,
                )
            for batch in tqdm(list(self._batched(rows)), desc="[kg] StandardizedValues",
                              unit="batch"):
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (co:Concept {name: row.concept})
                    MERGE (v:StandardizedValue {value_id: row.value_id})
                    SET v.name = row.name, v.code = row.code,
                        v.description = row.description, v.attrs_json = row.attrs_json,
                        v.spec_id = row.spec_id
                    MERGE (co)-[:HAS_VALUE]->(v)
                    WITH v, row
                    MATCH (c:Chunk {chunk_id: row.chunk_id})
                    MERGE (v)-[:DEFINED_IN_TABLE]->(c)
                    """,
                    rows=batch,
                )
            # Bridge Layer C to Layer B where names are closed-set Term abbreviations.
            s.run(
                """
                MATCH (v:StandardizedValue)
                MATCH (t:Term {abbreviation: v.name})
                MERGE (v)-[:DENOTES]->(t)
                """
            )
            s.run(
                """
                MATCH (co:Concept)
                MATCH (t:Term {abbreviation: co.name})
                MERGE (co)-[:ABOUT]->(t)
                """
            )
        return len(rows)

    def _create_procedure_steps(self, chunks: List[dict]) -> int:
        """Split chunk_type='procedure' chunks into Steps on the "N.\\t"/"Na.\\t"
        line-start marker. (Chunk)-[:HAS_STEP]->(Step), (Step)-[:NEXT]->(Step) links
        them sequentially within a chunk, (Step)-[:INVOLVES]->(Term) REUSES the existing
        chunk-level key_terms (∩ Term.abbreviation, like MENTIONS) — NO free NLP on the
        sentence, only checks whether that term appears (word-boundary) in the step's text.
        Does NOT extract actor_from/actor_to (see module docstring — real coverage is too
        low ~1.9% to be reliable). Must run AFTER _create_mentions_edges — reuses exactly
        the Term set that MENTIONS already df-filtered for that chunk (do NOT recompute
        document-frequency here, which would let in ubiquitous tokens like TS/NOTE/ID/SS
        that MENTIONS_MIN_DF/MAX_DF already excluded). Returns the number of Step nodes created."""
        with self._driver.session() as s:
            # No chunk_type filter here either: has_step_flow admits chunks the
            # classifier typed otherwise, and they need their MENTIONS terms too or
            # their Steps land with no INVOLVES edge — the very thing Pattern F reads.
            chunk_terms: Dict[str, List[str]] = {
                r["chunk_id"]: r["terms"]
                for r in s.run(
                    "MATCH (c:Chunk)-[:MENTIONS]->(t:Term) "
                    "RETURN c.chunk_id AS chunk_id, collect(t.abbreviation) AS terms"
                )
            }

        step_rows = []
        chain_rows = []
        involves_rows = []
        for c in chunks:
            if not has_step_flow(c):
                continue
            content = c.get("content") or ""
            # Split with whichever marker qualified the chunk — splitting a dotless
            # flow with STEP_RE yields one part and silently drops it.
            rx = STEP_RE if len(STEP_RE.findall(content)) >= STEP_MIN_MARKERS else STEP_RE_LOOSE
            parts = rx.split(content)
            if len(parts) < 3:
                continue  # no step could be split out — skip this chunk
            chunk_key_terms = chunk_terms.get(c["chunk_id"], [])
            step_ids_in_order = []
            for i in range(1, len(parts), 2):
                order = parts[i]
                text = (parts[i + 1] if i + 1 < len(parts) else "").strip()
                if not text:
                    continue
                step_id = f"{c['chunk_id']}::{order}"
                step_rows.append({
                    "step_id": step_id, "chunk_id": c["chunk_id"], "order": order,
                    "text": text[:2000],
                })
                step_ids_in_order.append(step_id)
                for kt in chunk_key_terms:
                    if re.search(r"\b" + re.escape(kt) + r"\b", text):
                        involves_rows.append({"step_id": step_id, "abbr": kt})
            for a, b in zip(step_ids_in_order, step_ids_in_order[1:]):
                chain_rows.append({"a": a, "b": b})

        with self._driver.session() as s:
            for batch in tqdm(list(self._batched(step_rows)), desc="[kg] Steps", unit="batch"):
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (st:Step {step_id: row.step_id})
                    SET st.chunk_id = row.chunk_id, st.order = row.order, st.text = row.text
                    WITH st, row
                    MATCH (c:Chunk {chunk_id: row.chunk_id})
                    MERGE (c)-[:HAS_STEP]->(st)
                    """,
                    rows=batch,
                )
            for batch in self._batched(chain_rows):
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Step {step_id: row.a}), (b:Step {step_id: row.b})
                    MERGE (a)-[:NEXT]->(b)
                    """,
                    rows=batch,
                )
            for batch in self._batched(involves_rows):
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (st:Step {step_id: row.step_id})
                    MATCH (t:Term {abbreviation: row.abbr})
                    MERGE (st)-[:INVOLVES]->(t)
                    """,
                    rows=batch,
                )
        return len(step_rows)

    def _create_messages(self, chunks: List[dict]) -> int:
        """Create Message nodes from ALL-CAPS message names (e.g. 'REGISTRATION REQUEST',
        'PDU SESSION ESTABLISHMENT REQUEST') appearing in content. Message.name is
        GLOBALLY unique (like Term — message names have consistent meaning across specs,
        unlike Parameter which follows generic table columns). Filter by document-frequency
        band [MESSAGE_MIN_DF, +inf) to drop rare/noisy matches. Returns the number of
        distinct Message nodes created."""
        # Normalize whitespace before regex — avoid "\n" being swallowed by \s and gluing two lines.
        chunk_matches: List[Tuple[str, set]] = []
        df: Dict[str, int] = defaultdict(int)
        for c in chunks:
            content = re.sub(r"\s+", " ", c.get("content") or "")
            found = {m.strip() for m in MESSAGE_RE.findall(content)}
            if not found:
                continue
            chunk_matches.append((c["chunk_id"], found))
            for name in found:
                df[name] += 1

        keep = {name for name, n in df.items() if n >= MESSAGE_MIN_DF}
        edge_rows = [
            {"chunk_id": chunk_id, "name": name}
            for chunk_id, names in chunk_matches
            for name in names
            if name in keep
        ]
        msg_rows = [{"name": name, "df": df[name]} for name in keep]

        with self._driver.session() as s:
            if msg_rows:
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (m:Message {name: row.name})
                    SET m.df = row.df
                    """,
                    rows=msg_rows,
                )
            for batch in tqdm(list(self._batched(edge_rows)), desc="[kg] Messages", unit="batch"):
                s.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Chunk {chunk_id: row.chunk_id})
                    MATCH (m:Message {name: row.name})
                    MERGE (c)-[:DESCRIBES_MESSAGE]->(m)
                    """,
                    rows=batch,
                )
        return len(msg_rows)

    # ── Reporting ────────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        with self._driver.session() as s:
            print("[kg] Schema stats:")
            for label in ["Document", "Chunk", "Term", "Subject",
                          "ServiceOperation", "Parameter", "Step", "Message",
                          "Concept", "StandardizedValue"]:
                count = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                print(f"[kg]   :{label:<16} {count:>10}")
            for rel in ["CONTAINS", "REFERENCES_SPEC", "REFERENCES_CHUNK",
                        "DEFINED_IN", "HAS_SUBJECT", "PARENT_SECTION", "MENTIONS",
                        "PROVIDED_BY", "DESCRIBES_OPERATION", "CO_OCCURS_WITH",
                        "DEFINED_IN_TABLE", "HAS_STEP", "NEXT", "INVOLVES", "DESCRIBES_MESSAGE"]:
                count = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
                print(f"[kg]   :{rel:<20} {count:>10}")

    def validate(self) -> Dict[str, int]:
        """Verify every relationship type is populated. Returns a dict of counts."""
        rels = ["CONTAINS", "REFERENCES_SPEC", "REFERENCES_CHUNK",
                "DEFINED_IN", "HAS_SUBJECT", "PARENT_SECTION", "MENTIONS",
                "PROVIDED_BY", "DESCRIBES_OPERATION", "CO_OCCURS_WITH",
                "DEFINED_IN_TABLE", "HAS_STEP", "NEXT", "INVOLVES", "DESCRIBES_MESSAGE",
                "HAS_VALUE", "DENOTES", "ABOUT"]
        counts: Dict[str, int] = {}
        with self._driver.session() as s:
            for rel in rels:
                counts[rel] = s.run(
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
                ).single()["c"]

        missing = [r for r, c in counts.items() if c == 0]
        if missing:
            print(f"[kg] ⚠ Empty relationships: {', '.join(missing)}")
        else:
            print("[kg] ✓ All relationship types populated")

        # Sanity check: the core 5G network functions MUST resolve as Term nodes.
        # AMF/SMF/NRF went silently missing in earlier builds (a Term write issue),
        # which breaks Pattern A/F/G anchoring for the most common questions. Flag
        # it loudly at build time so it's caught here, not at query time — and run
        # `python -m kg_builder.enrich_terms --json-dir <dir>` to backfill.
        with self._driver.session() as s:
            present = {
                r["a"] for r in s.run(
                    "MATCH (t:Term) WHERE t.abbreviation IN $nfs RETURN t.abbreviation AS a",
                    nfs=["AMF", "SMF", "UPF", "UDM", "AUSF", "NRF", "PCF", "NEF"],
                )
            }
        core_missing = [nf for nf in ("AMF", "SMF", "UPF", "UDM", "AUSF", "NRF", "PCF", "NEF")
                        if nf not in present]
        if core_missing:
            print(f"[kg] ⚠ Core NF Terms MISSING: {', '.join(core_missing)} — "
                  f"run `python -m kg_builder.enrich_terms --json-dir <json_dir>` to backfill")
        else:
            print("[kg] ✓ Core NF Terms present (AMF/SMF/UPF/UDM/AUSF/NRF/PCF/NEF)")

        # Layer C sanity: the SST value table (23.501 §5.15.2.2) is the canonical
        # standardized-value case — URLLC must resolve as SST value 2. If it does
        # not, the value extraction silently regressed (the exact failure mode that
        # motivated Layer C: the whole standardized-value class invisible to the KG).
        with self._driver.session() as s:
            urllc = s.run(
                "MATCH (:Concept {name:'SST'})-[:HAS_VALUE]->(v:StandardizedValue {name:'URLLC'}) "
                "RETURN v.code AS code LIMIT 1"
            ).single()
        if urllc and urllc["code"] == "2":
            print("[kg] ✓ Layer C sanity: SST -> URLLC = 2 resolves")
        else:
            print("[kg] ⚠ Layer C sanity FAILED: SST -> URLLC(2) not found — "
                  "run `python -m kg_builder.enrich_standardized_values` to backfill")
        return counts
