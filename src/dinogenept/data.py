"""Pinned official data acquisition. Downloads are intended for the server."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .provenance import atomic_write_json, digest_file, utc_now

GENEPT_ZIP_URL = "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1"
GENEPT_ZIP_MD5 = "3f6ce4317e3a0091978ae5cb8fbf05a3"
GENEPT_MEMBERS = {
    "NCBI_summary_of_genes.json",
    "NCBI_UniProt_summary_of_genes.json",
    "GenePT_gene_embedding_ada_text.pickle",
    "GenePT_gene_protein_embedding_model_3_text.pickle",
}
GENE2VEC_COMMIT = "6236e0b21fbc367bf8ff5695ed0cc1443861bd1e"
GGI_BASE_URL = "https://api.github.com/repos/jingcheng-du/Gene2vec/contents/predictionData"
GGI_FILES = ("train_text.txt", "train_label.txt", "test_text.txt", "test_label.txt")
GGI_SHA256 = {
    "train_text.txt": "5d7808130c0b337c15b927545cbff9552b754610a29c89dfe2a6c54c91299e96",
    "train_label.txt": "d8f5631b3e4ac8eb60bca80ef5f73a9d521c017d2bafee3a6fc65659789b3434",
    "test_text.txt": "c03f4b276bb9ddbb60b3e82eba32391578c95b4c353479a369e16066a04d5360",
    "test_label.txt": "1e9ffeba8c927e3b12565c37ff5bc8fd9300d6500f67152687eeee198c7ccdb1",
}
GO_RELEASE = "2026-08-05"
GO_RELEASE_BASE_URL = f"https://release.geneontology.org/{GO_RELEASE}"
GOEXP_FILES = {
    "HUMAN-uniprot.gaf.gz": {
        "url": f"{GO_RELEASE_BASE_URL}/annotations/gaf/HUMAN-uniprot.gaf.gz",
        "sha256": "a0afba19dfb1f8fa996bc1bdcd61fd0c9bd4cf0d2bf2509d09ac86993c0e70a2",
    },
    "go-basic.obo": {
        "url": f"{GO_RELEASE_BASE_URL}/ontology/go-basic.obo",
        "sha256": "b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc",
    },
}
KNOWLEDGE_SOURCE_URLS = {
    "uniprot-human-reviewed.tsv": (
        "https://rest.uniprot.org/uniprotkb/stream?format=tsv&query=%28organism_id%3A9606%29%20"
        "AND%20%28reviewed%3Atrue%29&fields=accession%2Cgene_primary%2Cprotein_name%2C"
        "cc_subcellular_location%2Ccc_catalytic_activity%2Ccc_cofactor%2Ccc_ptm%2C"
        "cc_activity_regulation%2Cft_domain%2Cft_act_site%2Cft_binding%2Cxref_interpro"
    ),
    "interpro.entry.list": "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list",
    "reactome.UniProt2Reactome.txt": "https://reactome.org/download/current/UniProt2Reactome.txt",
    "signor.human.tsv": "https://signor.uniroma2.it/API/getHumanData.php",
    "hpa.proteinatlas.tsv.zip": "https://www.proteinatlas.org/download/proteinatlas.tsv.zip",
}


def download(
    url: str,
    destination: Path,
    *,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 4,
) -> Path:
    """Resume each retry explicitly; never let curl or fallback truncate progress."""
    if max_retries < 0 or chunk_size < 1:
        raise ValueError("Invalid download retry/chunk configuration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    curl = shutil.which("curl")
    if curl:
        for attempt in range(max_retries + 1):
            command = [
                curl,
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                "0",
                "--connect-timeout",
                "20",
                "--max-time",
                "1800",
                "--continue-at",
                "-",
                "--output",
                str(temporary),
            ]
            if url.startswith("https://api.github.com/"):
                command.extend(["--header", "Accept: application/vnd.github.raw+json"])
            command.append(url)
            try:
                subprocess.run(command, check=True)  # noqa: S603 - vector, no shell
                temporary.replace(destination)
                return destination
            except subprocess.CalledProcessError:
                if attempt == max_retries:
                    raise  # Preserve partial bytes; no truncating fallback.
                time.sleep(min(2**attempt, 30))
    else:
        for attempt in range(max_retries + 1):
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"User-Agent": "DinoGenePT/0.1"}
            if url.startswith("https://api.github.com/"):
                headers["Accept"] = "application/vnd.github.raw+json"
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    status = getattr(response, "status", 200)
                    response_headers = getattr(response, "headers", {})
                    if offset and (
                        status != 206 or not response_headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                    ):
                        raise RuntimeError("Server did not honor resume offset; partial download preserved")
                    expected = response_headers.get("Content-Length")
                    written = 0
                    with temporary.open("ab" if offset else "wb") as handle:
                        while data := response.read(chunk_size):
                            handle.write(data)
                            written += len(data)
                    if expected is not None and written != int(expected):
                        raise TimeoutError("Response ended before declared Content-Length")
                temporary.replace(destination)
                return destination
            except (TimeoutError, urllib.error.URLError, ConnectionError):
                if attempt == max_retries:
                    raise
                time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify the publisher checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path, expected: str) -> None:
    actual = file_md5(path)
    if actual != expected.lower():
        raise ValueError(f"checksum mismatch for {path.name}: expected {expected}, got {actual}")


def verify_sha256(path: Path, expected: str) -> None:
    actual = digest_file(path)
    if actual != expected.lower():
        raise ValueError(f"checksum mismatch for {path.name}: expected {expected}, got {actual}")


def extract_allowlisted(zip_path: Path, destination: Path, members: set[str]) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        by_basename = {Path(info.filename.rstrip(".")).name: info for info in archive.infolist()}
        missing = members - set(by_basename)
        if missing:
            raise ValueError(f"archive missing expected files: {sorted(missing)}")
        for name in sorted(members):
            info = by_basename[name]
            output = destination / name
            with archive.open(info) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(output)
    return extracted


def prepare_genept(destination: Path, *, keep_archive: bool = False) -> dict[str, object]:
    archive = destination / "GenePT_embedding_v2.zip"
    download(GENEPT_ZIP_URL, archive)
    verify_md5(archive, GENEPT_ZIP_MD5)
    files = extract_allowlisted(archive, destination, GENEPT_MEMBERS)
    if not keep_archive:
        archive.unlink()
    manifest = {
        "prepared_at": utc_now(),
        "source_url": GENEPT_ZIP_URL,
        "source_md5": GENEPT_ZIP_MD5,
        "files": [path.name for path in files],
    }
    atomic_write_json(destination / "genept_manifest.json", manifest)
    return manifest


def prepare_ggi(destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name in GGI_FILES:
        path = destination / name
        if not path.exists() or path.stat().st_size == 0:
            download(f"{GGI_BASE_URL}/{name}?ref={GENE2VEC_COMMIT}", path)
        verify_sha256(path, GGI_SHA256[name])
        downloaded.append(path)
    manifest = {
        "prepared_at": utc_now(),
        "source_base_url": GGI_BASE_URL,
        "source_commit": GENE2VEC_COMMIT,
        "files": {path.name: digest_file(path) for path in downloaded},
    }
    atomic_write_json(destination / "ggi_manifest.json", manifest)
    return manifest


def prepare_go_exp(destination: Path) -> dict[str, object]:
    """Download and verify the pinned human GO annotation and ontology release."""

    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name, descriptor in GOEXP_FILES.items():
        path = destination / name
        if not path.exists() or path.stat().st_size == 0:
            download(str(descriptor["url"]), path)
        verify_sha256(path, str(descriptor["sha256"]))
        files[name] = digest_file(path)
    manifest = {
        "prepared_at": utc_now(),
        "release": GO_RELEASE,
        "release_base_url": GO_RELEASE_BASE_URL,
        "files": files,
    }
    atomic_write_json(destination / "go_exp_source_manifest.json", manifest)
    return manifest


def prepare_knowledge_sources(destination: Path) -> dict[str, object]:
    """Materialize official current releases and freeze their content hashes."""

    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, str]] = {}
    for name, url in KNOWLEDGE_SOURCE_URLS.items():
        path = destination / name
        if not path.exists() or path.stat().st_size == 0:
            download(url, path)
        files[name] = {"url": url, "sha256": digest_file(path)}
    manifest = {
        "schema_version": "genept-seed-knowledge-sources-v1",
        "prepared_at": utc_now(),
        "release_policy": "publisher-current-snapshot-frozen-by-sha256",
        "files": files,
    }
    atomic_write_json(destination / "knowledge_source_manifest.json", manifest)
    return manifest
