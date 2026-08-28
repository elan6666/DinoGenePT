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
ENSEMBL_LOOKUP_URL = "https://rest.ensembl.org/lookup/id"
ENSEMBL_ARCHIVE_URL = "https://rest.ensembl.org/archive/id"
HGNC_REST_URL = "https://rest.genenames.org/fetch"
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

    def get_text(self, url: str, *, headers: dict[str, str] | None = None) -> str:
        for attempt in range(self.max_retries + 1):
            delay = max(0.0, self._last_request_at + self.request_interval - time.monotonic())
            if delay:
                time.sleep(delay)
            request_headers = {"User-Agent": "GenePT-Seed/0.1", **(headers or {})}
            request = urllib.request.Request(url, headers=request_headers)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = response.read().decode("utf-8")
                self._last_request_at = time.monotonic()
                return body
            except urllib.error.HTTPError as error:
                self._last_request_at = time.monotonic()
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    raise RuntimeError(f"metadata request failed with HTTP {error.code}") from error
            except (TimeoutError, urllib.error.URLError) as error:
                self._last_request_at = time.monotonic()
                if attempt == self.max_retries:
                    raise RuntimeError("metadata request failed after retries") from error
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    def get_json(self, url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
        payload = json.loads(self.get_text(url, headers=headers))
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


@dataclass(frozen=True)
class EnsemblGeneRecord:
    stable_id: str
    resolved_stable_id: str
    replacement_score: float | None
    resolution_status: str
    archive_release: str | None
    display_name: str
    description: str
    biotype: str
    assembly_name: str


@dataclass(frozen=True)
class HGNCGeneRecord:
    hgnc_id: str
    requested_symbol: str
    approved_symbol: str
    approved_name: str
    match_field: str
    entrez_id: str | None
    ensembl_gene_id: str | None


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


class EnsemblGeneClient:
    """Fetch a human Ensembl gene annotation by a stable ID."""

    def __init__(self, http: _HTTPClient | None = None) -> None:
        self.http = http or _HTTPClient()

    def lookup(self, stable_id: str) -> EnsemblGeneRecord:
        identifier = stable_id.strip()
        if not re.fullmatch(r"ENSG\d+", identifier):
            raise ValueError(f"invalid human Ensembl gene ID: {stable_id!r}")
        try:
            row = self._lookup_current(identifier)
            resolved_identifier = identifier
            replacement_score = None
            resolution_status = "current"
            archive_release = None
        except RuntimeError as error:
            if "HTTP 400" not in str(error) and "HTTP 404" not in str(error):
                raise
            archive_url = f"{ENSEMBL_ARCHIVE_URL}/{identifier}?" + urllib.parse.urlencode(
                {"content-type": "application/json"}
            )
            archive = self.http.get_json(archive_url)
            replacements = sorted(
                archive.get("possible_replacement", []),
                key=lambda value: (-float(value.get("score", 0.0)), str(value.get("stable_id", ""))),
            )
            archive_release = str(archive.get("release", "")) or None
            if replacements and float(replacements[0].get("score", 0.0)) >= 0.9:
                top_score = float(replacements[0]["score"])
                top = [row for row in replacements if float(row.get("score", 0.0)) == top_score]
            else:
                top = []
            if len(top) == 1:
                resolved_identifier = str(top[0]["stable_id"])
                replacement_score = top_score
                resolution_status = "high_confidence_replacement"
                row = self._lookup_current(resolved_identifier)
            else:
                resolved_identifier = identifier
                replacement_score = None
                resolution_status = "archived_unresolved"
                row = {
                    "display_name": "",
                    "description": "",
                    "biotype": "",
                    "assembly_name": archive.get("assembly", ""),
                }
        return EnsemblGeneRecord(
            stable_id=identifier,
            resolved_stable_id=resolved_identifier,
            replacement_score=replacement_score,
            resolution_status=resolution_status,
            archive_release=archive_release,
            display_name=str(row.get("display_name", "")).strip(),
            description=str(row.get("description", "")).strip(),
            biotype=str(row.get("biotype", "")).strip(),
            assembly_name=str(row.get("assembly_name", "")).strip(),
        )

    def _lookup_current(self, identifier: str) -> dict[str, Any]:
        url = f"{ENSEMBL_LOOKUP_URL}/{identifier}?" + urllib.parse.urlencode(
            {"content-type": "application/json", "species": "homo_sapiens"}
        )
        row = self.http.get_json(url)
        if row.get("object_type") != "Gene" or row.get("species") != "homo_sapiens":
            raise ValueError(f"Ensembl ID is not a human gene: {identifier}")
        return row


class HGNCGeneClient:
    """Resolve an approved, previous, or unambiguous alias human gene symbol."""

    def __init__(self, http: _HTTPClient | None = None) -> None:
        self.http = http or _HTTPClient()

    def find(self, symbol: str) -> HGNCGeneRecord | None:
        requested = _display_symbol(symbol)
        for field in ("symbol", "prev_symbol", "alias_symbol"):
            url = f"{HGNC_REST_URL}/{field}/{urllib.parse.quote(requested, safe='')}"
            payload = self.http.get_json(url, headers={"Accept": "application/json"})
            docs = payload.get("response", {}).get("docs", [])
            exact = [row for row in docs if self._field_contains(row.get(field), requested)]
            if len(exact) == 1:
                row = exact[0]
                entrez = row.get("entrez_id")
                ensembl = row.get("ensembl_gene_id")
                return HGNCGeneRecord(
                    hgnc_id=str(row.get("hgnc_id", "")),
                    requested_symbol=requested,
                    approved_symbol=str(row.get("symbol", "")),
                    approved_name=str(row.get("name", "")),
                    match_field=field,
                    entrez_id=str(entrez) if entrez else None,
                    ensembl_gene_id=str(ensembl) if ensembl else None,
                )
            if len(exact) > 1:
                return None
        return None

    @staticmethod
    def _field_contains(value: Any, requested: str) -> bool:
        values = value if isinstance(value, list) else [value]
        return any(str(item).upper() == requested.upper() for item in values if item)


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


def compose_ensembl_text(requested_symbol: str, ensembl: EnsemblGeneRecord) -> str:
    pieces = [f"Gene Symbol {requested_symbol}", f"Ensembl stable gene ID: {ensembl.stable_id}."]
    if ensembl.resolved_stable_id != ensembl.stable_id:
        pieces.append(f"Current Ensembl replacement: {ensembl.resolved_stable_id}.")
    elif ensembl.resolution_status == "archived_unresolved":
        release = f"; last recorded in release {ensembl.archive_release}" if ensembl.archive_release else ""
        assembly = f" on {ensembl.assembly_name}" if ensembl.assembly_name else ""
        pieces.append(f"This is an archived Ensembl gene{assembly}{release}.")
    if ensembl.display_name and ensembl.display_name.upper() != requested_symbol.upper():
        pieces.append(f"Current Ensembl display name: {ensembl.display_name}.")
    if ensembl.description:
        pieces.append(ensembl.description.rstrip(".") + ".")
    if ensembl.biotype:
        pieces.append(f"Gene biotype: {ensembl.biotype}.")
    return " ".join(pieces)


def extend_genept_texts(
    *,
    base_path: Path,
    genes_path: Path,
    output_path: Path,
    manifest_path: Path,
    ncbi_client: NCBIGeneClient | None = None,
    uniprot_client: UniProtClient | None = None,
    ensembl_client: EnsemblGeneClient | None = None,
    hgnc_client: HGNCGeneClient | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    raw = json.loads(base_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError("base GenePT text corpus must be a JSON object keyed by gene symbol")
    base = {str(key): value for key, value in raw.items()}
    existing = {_normalized_symbol(key) for key in base}
    requested_specs: list[tuple[str, str | None, str | None]] = []
    for line_number, line in enumerate(
        genes_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) not in {1, 2, 3}:
            raise ValueError(
                f"gene allowlist line {line_number} must contain "
                "SYMBOL [NCBI_GENE_ID|-] [ENSEMBL_GENE_ID]"
            )
        expected_gene_id = columns[1] if len(columns) >= 2 and columns[1] != "-" else None
        ensembl_id = columns[2] if len(columns) == 3 else None
        requested_specs.append((_display_symbol(columns[0]), expected_gene_id, ensembl_id))
    if not requested_specs:
        raise ValueError("gene allowlist is empty")
    requested = [symbol for symbol, _, _ in requested_specs]
    requested_normalized = [_normalized_symbol(symbol) for symbol in requested]
    if len(requested_normalized) != len(set(requested_normalized)):
        raise ValueError("gene allowlist contains duplicate symbols")
    overlap = sorted(set(requested_normalized) & existing)
    if overlap:
        raise ValueError(f"refusing to overwrite existing GenePT texts: {overlap}")

    ncbi_source = ncbi_client or NCBIGeneClient()
    uniprot_source = uniprot_client or UniProtClient()
    ensembl_source = ensembl_client or EnsemblGeneClient()
    hgnc_source = hgnc_client or HGNCGeneClient()
    checkpoint_header = {
        "schema_version": "genept-seed-corpus-extension-checkpoint-v1",
        "base_sha256": digest_file(base_path),
        "genes_sha256": digest_file(genes_path),
    }
    additions: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for key, expected in checkpoint_header.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"corpus extension checkpoint {key} mismatch")
        additions = {str(key): str(value) for key, value in checkpoint.get("additions", {}).items()}
        records = list(checkpoint.get("records", []))
    for symbol, expected_gene_id, ensembl_id in requested_specs:
        if symbol in additions:
            continue
        hgnc: HGNCGeneRecord | None = None
        try:
            ncbi = ncbi_source.resolve(symbol, expected_gene_id)
        except ValueError as ncbi_error:
            hgnc = hgnc_source.find(symbol)
            if hgnc is not None and hgnc.entrez_id is not None:
                ncbi = ncbi_source.resolve(symbol, hgnc.entrez_id)
            elif ensembl_id is None or not ensembl_id.startswith("ENSG"):
                raise ncbi_error
            else:
                ncbi = None
        if ncbi is None:
            assert ensembl_id is not None
            ensembl = ensembl_source.lookup(ensembl_id)
            text = compose_ensembl_text(symbol, ensembl)
            additions[symbol] = text
            records.append(
                {
                    "requested_symbol": symbol,
                    "source": "Ensembl",
                    "ensembl_gene_id": ensembl.stable_id,
                    "ensembl_resolved_gene_id": ensembl.resolved_stable_id,
                    "ensembl_replacement_score": ensembl.replacement_score,
                    "ensembl_resolution_status": ensembl.resolution_status,
                    "ensembl_archive_release": ensembl.archive_release,
                    "ensembl_display_name": ensembl.display_name,
                    "ensembl_description_present": bool(ensembl.description),
                    "ensembl_biotype": ensembl.biotype,
                    "ensembl_assembly_name": ensembl.assembly_name,
                    "text_characters": len(text),
                    "text_sha256": digest_text(text),
                }
            )
            if checkpoint_path is not None:
                atomic_write_json(
                    checkpoint_path,
                    {**checkpoint_header, "additions": additions, "records": records},
                )
            continue
        uniprot = uniprot_source.find(ncbi.current_symbol)
        text = compose_genept_text(ncbi, uniprot)
        additions[symbol] = text
        records.append(
            {
                "requested_symbol": symbol,
                "source": (
                    "HGNC mapping + NCBI Gene + UniProtKB" if hgnc else "NCBI Gene + UniProtKB"
                ),
                "hgnc_id": hgnc.hgnc_id if hgnc else None,
                "hgnc_match_field": hgnc.match_field if hgnc else None,
                "hgnc_approved_symbol": hgnc.approved_symbol if hgnc else None,
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
        if checkpoint_path is not None:
            atomic_write_json(
                checkpoint_path,
                {**checkpoint_header, "additions": additions, "records": records},
            )

    merged = {**base, **additions}
    atomic_write_json(output_path, merged)
    manifest: dict[str, Any] = {
        "schema_version": "genept-seed-corpus-extension-v2",
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
        "ensembl_source": f"{ENSEMBL_LOOKUP_URL} (human gene fallback when NCBI has no match)",
        "ensembl_archive_source": f"{ENSEMBL_ARCHIVE_URL} (archived stable-ID replacement)",
        "hgnc_source": f"{HGNC_REST_URL} (approved/previous/alias symbol disambiguation)",
        "records": records,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
