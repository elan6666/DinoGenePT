"""Server-only, additive audit of current vocabulary, master text and CellFM axes."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from dinogenept.gene_identity import GeneIdentityIndex, build_identity_report, read_identity_json
from dinogenept.provenance import atomic_write_json, digest_file
from dinogenept.source_corpus import build_source_only_corpus

HGNC_SHA = "0615a070f1628e6727953f67ad9248dd0f0ddbb16d41a7b40e06aa852fc3f448"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    hgnc = Path("data/hgnc/2026-08-28/hgnc_complete_set.txt")
    identity = GeneIdentityIndex(hgnc, expected_sha256=HGNC_SHA)
    inputs = {str(hgnc): HGNC_SHA}

    def read(path, expected=None):
        path = Path(path)
        checksum = digest_file(path)
        if expected and checksum != expected:
            raise ValueError(f"Frozen source changed: {path}")
        inputs[str(path)] = checksum
        return read_identity_json(path)

    stages = [Path("data/corpora") / (stem + ".json") for stem in (
        "seed-master", "seed-go-master", "seed-go-protein-master",
        "seed-go-protein-pathway-master", "seed-go-protein-pathway-hpa-master",
    )]
    for path in stages:
        manifest = read(path.with_suffix(".manifest.json"))
        read(path, manifest["output_sha256"])
    corpora = {"TextBase": read(stages[0])}
    for name, previous, enriched in zip(["GO", "Protein", "Pathway", "HPA"], stages[:-1], stages[1:], strict=True):
        target = output / f"{name}.json"
        build_source_only_corpus(base_path=previous, enriched_path=enriched,
                                 output_path=target, manifest_path=output / f"{name}.source.json", source=name)
        corpora[name] = read(target)

    # Downstream uses its already extended 1777-gene bundle, not master-only text.
    cellfm_root = Path("data/knowledge/cellfm-1777-v1")
    cellfm_receipt = read(cellfm_root / "receipt.json")
    cellfm_corpora = {
        name: read(cellfm_root / (name + ".json"), cellfm_receipt["files_sha256"][name + ".json"])
        for name in corpora
    }

    vocab_root = Path("data/pretraining/genecompass-human500k-mmap-v1")
    manifest = read(vocab_root / "manifest.json")
    vocab = read(vocab_root / manifest["vocabulary"]["path"], manifest["vocabulary"]["sha256"])
    axes = {"genecompass_vocabulary": vocab, "master_labels": list(corpora["TextBase"])}
    for dataset in ["adamson", "norman"]:
        root = Path("data/perturbation/cellfm-v1") / dataset
        audit = read(root / "audit.json")
        symbols = read(root / "axis_symbols.json", audit["files_sha256"]["axis_symbols.json"])
        targets = read(root / "target_symbols.json", audit["files_sha256"]["target_symbols.json"])
        mapping = read(root / "gene_mapping.json")
        if mapping["symbols"] != symbols or mapping["downstream_audit_sha256"] != digest_file(root / "audit.json"):
            raise ValueError("Downstream identity mapping does not match its frozen axis/audit")
        rows = [dict(label=symbol, ensembl_id=ensg)
                for symbol, ensg in zip(symbols, mapping["ensembl_ids"], strict=True)]
        axes[dataset + "_axis"] = rows
        by_symbol = {row["label"]: row for row in rows}
        axes[dataset + "_targets"] = [by_symbol[target] for target in targets]

    vocab_identity = identity.axis(vocab)
    tokens = defaultdict(list)
    for row in vocab_identity["rows"]:
        if row["status"] == "resolved":
            tokens[row["hgnc_id"]].append(row["position"] + 1)  # Current native PAD0 contract.
    summaries = {}
    for name, rows in axes.items():
        selected = cellfm_corpora if name.startswith(("adamson_", "norman_")) else corpora
        report = build_identity_report(identity, rows, selected)
        report["source_bundle"] = "cellfm-1777-v1" if selected is cellfm_corpora else "master-source-only"
        for row in report["rows"]:
            candidates = tokens.get(row["hgnc_id"], [])
            row["pretraining_token_candidates"] = candidates
            row["pretraining_token_status"] = (
                "identity_unresolved" if row["status"] != "resolved" else
                "unique" if len(candidates) == 1 else "missing" if not candidates else "ambiguous"
            )
        destination = output / (name + ".json")
        atomic_write_json(destination, report)
        summaries[name] = {
            "rows": len(rows), "identities": report["counts"],
            "source_bundle": report["source_bundle"],
            "duplicate_identities": len(report["duplicate_identities"]),
            "safe_unique_axis": report["safe_unique_axis"],
            "pretraining_tokens": dict(Counter(r["pretraining_token_status"] for r in report["rows"])),
            "sources": {s: v["counts"] for s, v in report["sources"].items()},
            "report": str(destination), "report_sha256": digest_file(destination),
            "unresolved_source_keys": {s: len(v["unresolved_source_keys"]) for s, v in report["sources"].items()},
        }
    if any(digest_file(path) != checksum for path, checksum in inputs.items()):
        raise ValueError("An input changed during the identity audit")
    receipt = {
        "schema": "dinogenept.identity-campaign-audit.v1", "hgnc_sha256": HGNC_SHA,
        "inputs": inputs, "audits": summaries,
        "scope": "Current vocabulary/all master labels/released CellFM Adamson+Norman; not all future datasets",
        "note": "No data/token/text/vector rewritten; availability is not functional coverage or vector readiness.",
        "inputs_unchanged_after_audit": True,
    }
    atomic_write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
