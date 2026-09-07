"""Extend the frozen GenePT text universe, derive independent knowledge views.

No API call. Existing master text is retained (encoder whitespace trimming only);
new axis genes use exact original GenePT summaries. Unknown identities fail.
"""

import argparse
import fcntl
import json
from pathlib import Path

from dinogenept.embedding import DEFAULT_MODEL, audit_embedding_checkpoint
from dinogenept.go_corpus import build_go_exp_corpus
from dinogenept.knowledge_corpus import build_knowledge_corpus
from dinogenept.provenance import atomic_write_json, digest_file
from dinogenept.source_corpus import build_source_only_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellfm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master", type=Path, default=Path("data/corpora/seed-master.json"))
    parser.add_argument("--genept", type=Path, default=Path("data/genept/NCBI_UniProt_summary_of_genes.json"))
    parser.add_argument("--go-dir", type=Path, default=Path("data/go-exp/2026-08-05"))
    parser.add_argument("--knowledge-dir", type=Path, default=Path("data/knowledge/current"))
    args = parser.parse_args()
    master_manifest = args.master.with_name(args.master.stem + ".manifest.json")
    proof = json.loads(master_manifest.read_text())
    if digest_file(args.master) != proof["output_sha256"]:
        raise ValueError("Frozen master text changed")
    sources = {"master": args.master, "genept": args.genept, "master_manifest": master_manifest}
    for directory, manifest_name in (
        (args.go_dir, "go_exp_source_manifest.json"),
        (args.knowledge_dir, "knowledge_source_manifest.json"),
    ):
        manifest_path = directory / manifest_name
        sources[manifest_name] = manifest_path
        for name, entry in json.loads(manifest_path.read_text())["files"].items():
            source = directory / name
            if digest_file(source) != (entry["sha256"] if isinstance(entry, dict) else entry):
                raise ValueError(f"Frozen public knowledge source changed: {name}")
            sources[name] = source
    genes, targets = set(), set()
    for name in ("adamson", "norman"):
        directory = args.cellfm / name
        audit = json.loads((directory / "audit.json").read_text())
        for filename, destination in (("axis_symbols.json", genes), ("target_symbols.json", targets)):
            path = directory / filename
            if digest_file(path) != audit["files_sha256"][filename]:
                raise ValueError("Frozen downstream gene identity changed")
            destination.update(json.loads(path.read_text()))
            sources[f"{name}/{filename}"] = path
    if not targets <= genes:
        raise ValueError("Perturbation target outside complete downstream axis union")
    identity = {key: digest_file(path) for key, path in sources.items()}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".preparation.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = output / "preparation.json"
        if progress.exists() and json.loads(progress.read_text())["identity"] != identity:
            raise ValueError("Refusing to mix different source snapshots in one corpus directory")
        if (output / "receipt.json").exists():
            receipt = json.loads((output / "receipt.json").read_text())
            if receipt["identity"] != identity:
                raise ValueError("Existing corpus identity differs")
            for name, sha in receipt["files_sha256"].items():
                if digest_file(output / name) != sha:
                    raise ValueError("Existing prepared corpus file differs")
            print(json.dumps({"status": "already_verified", "genes": len(genes)}))
            return
        atomic_write_json(progress, {"identity": identity, "status": "preparing"})
        master, original = (json.loads(path.read_text()) for path in (args.master, args.genept))
        corpus, added = {}, []
        for gene in sorted(genes):
            text = master.get(gene) or original.get(gene)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing exact source-grounded base text: {gene}")
            corpus[gene] = text.strip()  # Same whitespace policy as load_gene_texts.
            if gene not in master:
                added.append(gene)
        genes_path = output / "genes.txt"
        genes_path.write_text("\n".join(sorted(genes)) + "\n")
        atomic_write_json(output / "TextBase.json", corpus)
        stages = {"TextBase": output / "TextBase.json", "GO": output / "cumulative-GO.json"}
        build_go_exp_corpus(
            base_path=stages["TextBase"],
            genes_path=genes_path,
            gaf_path=args.go_dir / "HUMAN-uniprot.gaf.gz",
            obo_path=args.go_dir / "go-basic.obo",
            output_path=stages["GO"],
            manifest_path=output / "cumulative-GO.manifest.json",
        )
        for name, profile in (("Protein", "protein"), ("Pathway", "protein-pathway"), ("HPA", "protein-pathway-hpa")):
            stages[name] = output / f"cumulative-{name}.json"
            build_knowledge_corpus(
                base_path=stages["GO"],
                genes_path=genes_path,
                uniprot_path=args.knowledge_dir / "uniprot-human-reviewed.tsv",
                interpro_path=args.knowledge_dir / "interpro.entry.list",
                profile=profile,
                reactome_path=args.knowledge_dir / "reactome.UniProt2Reactome.txt" if name != "Protein" else None,
                signor_path=args.knowledge_dir / "signor.human.tsv" if name != "Protein" else None,
                hpa_path=args.knowledge_dir / "hpa.proteinatlas.tsv.zip" if name == "HPA" else None,
                output_path=stages[name],
                manifest_path=output / f"cumulative-{name}.manifest.json",
            )
        available = {"TextBase": len(genes)}
        previous = "TextBase"
        for name in ("GO", "Protein", "Pathway", "HPA"):
            manifest = build_source_only_corpus(
                base_path=stages[previous],
                enriched_path=stages[name],
                output_path=output / f"{name}.json",
                manifest_path=output / f"{name}.source.json",
                source=name,
                genes_path=genes_path,
            )
            available[name] = manifest["available_genes"]
            previous = name
        cached = {}
        # Audit only, never merge conflicting caches or regenerate automatically.
        candidates = {
            "TextBase": [
                "dinogenept-base-targets",
                "doubao",
                "gradpert-extension-case-doubao",
                "gradpert-jurkat-axis-doubao",
            ],
            "GO": ["dinogenept-go-exp-only-targets"],
            "Protein": ["dinogenept-protein-only-targets"],
            "Pathway": ["dinogenept-pathway-only-targets"],
            "HPA": ["dinogenept-hpa-only-targets"],
        }
        for name, choices in candidates.items():
            texts = json.loads((output / f"{name}.json").read_text())
            cached[name] = {
                candidate: audit_embedding_checkpoint(
                    texts,
                    checkpoint_path=Path("checkpoints") / f"{candidate}.sqlite3",
                    model=DEFAULT_MODEL,
                )
                for candidate in choices
            }
        receipt = {
            "status": "corpora_ready_vectors_not_generated",
            "identity": identity,
            "genes": len(genes),
            "perturbation_targets": len(targets),
            "added_from_exact_official_genept": added,
            "available": available,
            "checkpoint_audits": cached,
            "source_note": "Master text retained; additional exact original GenePT text; no fabricated function",
            "files_sha256": {
                p.name: digest_file(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "preparation.json" and not p.name.startswith(".")
            },
        }
        atomic_write_json(output / "receipt.json", receipt)
        atomic_write_json(progress, {"identity": identity, "status": "completed"})
        print(
            json.dumps(
                {
                    "genes": len(genes),
                    "targets": len(targets),
                    "new_base_genes": len(added),
                    "available": available,
                    "exact_cached": {s: {c: r["exact_cached"] for c, r in rows.items()} for s, rows in cached.items()},
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
