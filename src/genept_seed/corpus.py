"""Auditable extensions to the frozen GenePT NCBI + UniProt text corpus."""

from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json, digest_file, digest_text, utc_now

NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
HUMAN_TAXON_ID = 9606


def _normalized_symbol(value: str) -> str:
    symbol = value.strip()
    if not symbol or any(character.isspace() for character in symbol):
        raise ValueError(f"invalid gene symbol: {value!r}")
    return symbol.upper()


def _display_symbol(value: str) -> str:
    symbol = value.strip()
    _normalized_symbol(symbol)
    return symbol


class _HTTPClient:
    def __init__(self, *, request_interval: float = 0.5, max_retries: int = 5) -> None:
        if request_interval < 0:
            raise ValueError("request_interval must be non-negative")
        self.request_interval = request_interval
        self.max_retries = max_retries
        self._last_request_at = 0.0

    def get_text(self, url: str) -> str:
        for attempt in range(self.max_retries + 1):
            delay = max(0.0, self._last_request_at + self.request_interval - time.monotonic())
            if delay:
                time.sleep(delay)
            request = urllib.request.Request(url, headers={"User-Agent": "GenePT-Seed/0.1"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = response.read().decode("utf-8")
                self._last_request_at = time.monotonic()
                return body
            except urllib.error.HTTPError as error:
                self._last_request_at = time.monotonic()
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    raise RuntimeError(f"metadata request failed with HTTP {error.code}") from error
            except urllib.error.URLError as error:
                self._last_request_at = time.monotonic()
                if attempt == self.max_retries:
                    raise RuntimeError("metadata request failed after retries") from error
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    def get_json(self, url: str) -> dict[str, Any]:
        payload = json.loads(self.get_text(url))
        if not isinstance(payload, dict):
            raise ValueError("metadata endpoint returned a non-object JSON payload")
        return payload


@dataclass(frozen=True)
class NCBIGeneRecord:
    requested_symbol: str
    current_symbol: str
    gene_id: str
    description: str
    summary: str


@dataclass(frozen=True)
class UniProtRecord:
    accession: str
    reviewed: bool
    primary_symbol: str
    protein_name: str
    function: str


class NCBIGeneClient:
    """Resolve human symbols and aliases against NCBI Gene E-utilities."""

    def __init__(self, http: _HTTPClient | None = None) -> None:
        self.http = http or _HTTPClient()

    def resolve(self, symbol: str, expected_gene_id: str | None = None) -> NCBIGeneRecord:
        requested_symbol = _display_symbol(symbol)
        requested = _normalized_symbol(symbol)
        if expected_gene_id is None:
            search_url = f"{NCBI_EUTILS_BASE}/esearch.fcgi?" + urllib.parse.urlencode(
                {
                    "db": "gene",
                    "term": f"{requested}[sym] AND {HUMAN_TAXON_ID}[taxid]",
                    "retmode": "json",
                    "retmax": 20,
                    "tool": "genept_seed",
                }
            )
            search = self.http.get_json(search_url)
            identifiers = search.get("esearchresult", {}).get("idlist", [])
            if not identifiers:
                raise ValueError(f"NCBI Gene has no human symbol or alias match for {requested}")
        else:
            if not expected_gene_id.isdigit():
                raise ValueError(f"invalid NCBI GeneID for {requested}: {expected_gene_id!r}")
            identifiers = [expected_gene_id]
        summary_url = f"{NCBI_EUTILS_BASE}/esummary.fcgi?" + urllib.parse.urlencode(
            {
                "db": "gene",
                "id": ",".join(map(str, identifiers)),
                "retmode": "json",
                "tool": "genept_seed",
            }
        )
        result = self.http.get_json(summary_url).get("result", {})
        candidates = [result[str(identifier)] for identifier in identifiers if str(identifier) in result]
        exact = [row for row in candidates if str(row.get("name", "")).upper() == requested]
        aliases = [
            row
            for row in candidates
            if requested
            in {
                alias.strip().upper()
                for alias in str(row.get("otheraliases", "")).split(",")
                if alias.strip()
            }
        ]
        selected = exact or aliases
        if len(selected) != 1:
            names = sorted(str(row.get("name", "")) for row in candidates)
            if expected_gene_id is not None:
                raise ValueError(
                    f"NCBI GeneID {expected_gene_id} does not identify symbol or alias {requested}: {names}"
                )
            raise ValueError(f"ambiguous NCBI Gene match for {requested}: {names}")
        row = selected[0]
        return NCBIGeneRecord(
            requested_symbol=requested_symbol,
            current_symbol=_normalized_symbol(str(row["name"])),
            gene_id=str(row["uid"]),
            description=str(row.get("description", "")).strip(),
            summary=str(row.get("summary", "")).strip(),
        )


_ECO_BLOCK = re.compile(r"\s*\{ECO:[^{}]*\}\.?")


def clean_uniprot_function(value: str) -> str:
    """Convert UniProt's FUNCTION comment field to GenePT-like plain text."""

    cleaned = value.strip()
    cleaned = re.sub(r"(?:^|;\s*)FUNCTION:\s*", " ", cleaned)
    cleaned = _ECO_BLOCK.sub("", cleaned)
    return " ".join(cleaned.split()).strip(" ;")


class UniProtClient:
    """Fetch the best human UniProtKB function comment for an exact current symbol."""

    def __init__(self, http: _HTTPClient | None = None) -> None:
        self.http = http or _HTTPClient()

    def find(self, current_symbol: str) -> UniProtRecord | None:
        symbol = _normalized_symbol(current_symbol)
        url = UNIPROT_SEARCH_URL + "?" + urllib.parse.urlencode(
            {
                "query": f"(gene_exact:{symbol}) AND (organism_id:{HUMAN_TAXON_ID})",
                "format": "tsv",
                "fields": "accession,reviewed,gene_primary,protein_name,cc_function",
                "size": 100,
            }
        )
        rows = list(csv.DictReader(io.StringIO(self.http.get_text(url)), delimiter="\t"))
        rows = [row for row in rows if str(row.get("Gene Names (primary)", "")).upper() == symbol]
        if not rows:
            return None
        rows.sort(
            key=lambda row: (
                row.get("Reviewed") != "reviewed",
                not bool(clean_uniprot_function(row.get("Function [CC]", ""))),
                row.get("Entry", ""),
            )
        )
        row = rows[0]
        return UniProtRecord(
            accession=row.get("Entry", ""),
            reviewed=row.get("Reviewed") == "reviewed",
            primary_symbol=symbol,
            protein_name=row.get("Protein names", "").strip(),
            function=clean_uniprot_function(row.get("Function [CC]", "")),
        )


def compose_genept_text(ncbi: NCBIGeneRecord, uniprot: UniProtRecord | None) -> str:
    pieces = [f"Gene Symbol {ncbi.requested_symbol}"]
    if ncbi.current_symbol.upper() != ncbi.requested_symbol.upper():
        pieces.append(f"Current NCBI symbol: {ncbi.current_symbol}.")
    if ncbi.summary:
        pieces.append(ncbi.summary)
    elif ncbi.description:
        pieces.append(ncbi.description)
    if uniprot is not None and uniprot.function:
        pieces.append(f"Protein summary: {uniprot.function}")
    return " ".join(pieces)


def extend_genept_texts(
    *,
    base_path: Path,
    genes_path: Path,
    output_path: Path,
    manifest_path: Path,
    ncbi_client: NCBIGeneClient | None = None,
    uniprot_client: UniProtClient | None = None,
) -> dict[str, Any]:
    raw = json.loads(base_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError("base GenePT text corpus must be a JSON object keyed by gene symbol")
    base = {str(key): value for key, value in raw.items()}
    existing = {_normalized_symbol(key) for key in base}
    requested_specs: list[tuple[str, str | None]] = []
    for line_number, line in enumerate(
        genes_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) not in {1, 2}:
            raise ValueError(f"gene allowlist line {line_number} must contain SYMBOL [NCBI_GENE_ID]")
        requested_specs.append((_display_symbol(columns[0]), columns[1] if len(columns) == 2 else None))
    if not requested_specs:
        raise ValueError("gene allowlist is empty")
    requested = [symbol for symbol, _ in requested_specs]
    requested_normalized = [_normalized_symbol(symbol) for symbol in requested]
    if len(requested_normalized) != len(set(requested_normalized)):
        raise ValueError("gene allowlist contains duplicate symbols")
    overlap = sorted(set(requested_normalized) & existing)
    if overlap:
        raise ValueError(f"refusing to overwrite existing GenePT texts: {overlap}")

    ncbi_source = ncbi_client or NCBIGeneClient()
    uniprot_source = uniprot_client or UniProtClient()
    additions: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for symbol, expected_gene_id in requested_specs:
        ncbi = ncbi_source.resolve(symbol, expected_gene_id)
        uniprot = uniprot_source.find(ncbi.current_symbol)
        text = compose_genept_text(ncbi, uniprot)
        additions[symbol] = text
        records.append(
            {
                "requested_symbol": symbol,
                "current_ncbi_symbol": ncbi.current_symbol,
                "ncbi_gene_id": ncbi.gene_id,
                "ncbi_description": ncbi.description,
                "ncbi_summary_present": bool(ncbi.summary),
                "uniprot_accession": uniprot.accession if uniprot else None,
                "uniprot_reviewed": uniprot.reviewed if uniprot else None,
                "uniprot_function_present": bool(uniprot and uniprot.function),
                "text_characters": len(text),
                "text_sha256": digest_text(text),
            }
        )

    merged = {**base, **additions}
    atomic_write_json(output_path, merged)
    manifest: dict[str, Any] = {
        "schema_version": "genept-seed-corpus-extension-v1",
        "created_at": utc_now(),
        "base_file": base_path.name,
        "base_sha256": digest_file(base_path),
        "base_entries": len(base),
        "added_entries": len(additions),
        "output_entries": len(merged),
        "output_file": output_path.name,
        "output_sha256": digest_file(output_path),
        "ncbi_source": f"{NCBI_EUTILS_BASE}/ (Gene, human taxon {HUMAN_TAXON_ID})",
        "uniprot_source": f"{UNIPROT_SEARCH_URL} (UniProtKB, human taxon {HUMAN_TAXON_ID})",
        "records": records,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
