"""Command-line interface for server-side data, embedding, and benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn

from . import __version__
from .axis_corpus import build_axis_corpus
from .axis_vectors import align_npz_to_axis, materialize_axis_vectors
from .benchmarks import evaluate_ggi, evaluate_property_task
from .corpus import extend_genept_texts
from .data import prepare_genept, prepare_ggi, prepare_go_exp, prepare_knowledge_sources
from .embedding import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ArkEmbeddingClient,
    audit_embedding_checkpoint,
    generate_embeddings,
    load_gene_texts,
    select_gene_texts,
    text_statistics,
)
from .ggi_comparison import build_ggi_comparison
from .go_corpus import build_go_exp_corpus
from .gradpert_union import build_gradpert_union
from .knowledge_corpus import PROFILES, audit_knowledge_corpora, build_knowledge_corpus
from .knowledge_vectors import audit_knowledge_vectors
from .provenance import credential_present, digest_file
from .reports import write_results
from .tasks import ggi_genes, load_ggi, load_property_tasks
from .vectors import coverage, l2_normalize, load_npz, load_official_pickle, select_universe_vectors


def _embedding(path: Path, *, trusted_pickle: bool, normalize: bool):
    if path.suffix in {".pickle", ".pkl"}:
        loaded = load_official_pickle(path, trusted=trusted_pickle)
    else:
        loaded = load_npz(path)
    return l2_normalize(loaded) if normalize else loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genept-seed")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="check runtime and credential presence")

    data = subparsers.add_parser("data", help="prepare pinned datasets on the server")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    genept = data_sub.add_parser("prepare-genept")
    genept.add_argument("--output", type=Path, required=True)
    genept.add_argument("--keep-archive", action="store_true")
    ggi = data_sub.add_parser("prepare-ggi")
    ggi.add_argument("--output", type=Path, required=True)
    genes = data_sub.add_parser("ggi-genes", help="write the unique benchmark gene set")
    genes.add_argument("--data", type=Path, required=True)
    genes.add_argument("--output", type=Path, required=True)
    extend = data_sub.add_parser(
        "extend-genept-texts",
        help="append missing human genes from NCBI Gene and UniProtKB with provenance",
    )
    extend.add_argument("--base", type=Path, required=True)
    extend.add_argument("--genes", type=Path, required=True)
    extend.add_argument("--output", type=Path, required=True)
    extend.add_argument("--manifest", type=Path, required=True)
    extend.add_argument("--checkpoint", type=Path)
    axis_text = data_sub.add_parser(
        "build-axis-texts",
        help="materialize an exact graph-axis corpus with HGNC identity fallbacks",
    )
    axis_text.add_argument("--source", type=Path, required=True)
    axis_text.add_argument("--genes", type=Path, required=True)
    axis_text.add_argument("--hgnc", type=Path, required=True)
    axis_text.add_argument("--axis-mapping", type=Path)
    axis_text.add_argument("--ensembl-archive", type=Path)
    axis_text.add_argument("--output", type=Path, required=True)
    axis_text.add_argument("--manifest", type=Path, required=True)
    axis_text.add_argument("--allow-case-duplicates", action="store_true")
    axis_text.add_argument("--allow-identity-only", action="store_true")
    axis_vectors = data_sub.add_parser(
        "build-axis-vectors",
        help="align a trusted official embedding pickle to a graph axis",
    )
    axis_vectors.add_argument("--source", type=Path, required=True)
    axis_vectors.add_argument("--genes", type=Path, required=True)
    axis_vectors.add_argument("--hgnc", type=Path, required=True)
    axis_vectors.add_argument("--axis-mapping", type=Path)
    axis_vectors.add_argument("--output", type=Path, required=True)
    axis_vectors.add_argument("--manifest", type=Path, required=True)
    axis_vectors.add_argument("--model", required=True)
    axis_vectors.add_argument("--trusted-pickle", action="store_true")
    align_vectors = data_sub.add_parser(
        "align-axis-vectors",
        help="reorder a complete NPZ embedding set to an exact graph axis",
    )
    align_vectors.add_argument("--source", type=Path, required=True)
    align_vectors.add_argument("--genes", type=Path, required=True)
    align_vectors.add_argument("--output", type=Path, required=True)
    align_vectors.add_argument("--manifest", type=Path, required=True)
    go_source = data_sub.add_parser("prepare-go-exp", help="prepare the pinned human GO release")
    go_source.add_argument("--output", type=Path, required=True)
    knowledge_source = data_sub.add_parser(
        "prepare-knowledge-sources", help="freeze official protein/pathway/HPA snapshots"
    )
    knowledge_source.add_argument("--output", type=Path, required=True)
    go_text = data_sub.add_parser("build-go-exp-texts", help="append bounded experimental GO text")
    go_text.add_argument("--base", type=Path, required=True)
    go_text.add_argument("--genes", type=Path, required=True)
    go_text.add_argument("--gaf", type=Path, required=True)
    go_text.add_argument("--obo", type=Path, required=True)
    go_text.add_argument("--output", type=Path, required=True)
    go_text.add_argument("--manifest", type=Path, required=True)
    go_text.add_argument("--max-terms-per-aspect", type=int, default=8)
    go_text.add_argument("--include-interaction-evidence", action="store_true")
    go_text.add_argument("--allow-case-duplicates", action="store_true")
    union = data_sub.add_parser("build-gradpert-union", help="freeze all five GraD-Pert graph axes")
    union.add_argument("--gradpert-root", type=Path, required=True)
    union.add_argument("--extra-genes", type=Path)
    union.add_argument("--output", type=Path, required=True)
    union.add_argument("--manifest", type=Path, required=True)
    knowledge = data_sub.add_parser("build-knowledge-texts", help="append sparse progressive protein knowledge")
    knowledge.add_argument("--base", type=Path, required=True)
    knowledge.add_argument("--genes", type=Path, required=True)
    knowledge.add_argument("--uniprot", type=Path, required=True)
    knowledge.add_argument("--interpro", type=Path, required=True)
    knowledge.add_argument("--reactome", type=Path)
    knowledge.add_argument("--signor", type=Path)
    knowledge.add_argument("--hpa", type=Path)
    knowledge.add_argument("--profile", choices=PROFILES, required=True)
    knowledge.add_argument("--max-items-per-source", type=int, default=8)
    knowledge.add_argument("--max-characters-per-field", type=int, default=2000)
    knowledge.add_argument("--output", type=Path, required=True)
    knowledge.add_argument("--manifest", type=Path, required=True)
    corpus_audit = data_sub.add_parser(
        "audit-knowledge-corpora", help="prove complete and append-only progressive corpora"
    )
    corpus_audit.add_argument("--genes", type=Path, required=True)
    corpus_audit.add_argument("--gradpert-root", type=Path, required=True)
    corpus_audit.add_argument("--base", type=Path, required=True)
    corpus_audit.add_argument("--protein", type=Path, required=True)
    corpus_audit.add_argument("--pathway", type=Path, required=True)
    corpus_audit.add_argument("--hpa", type=Path, required=True)
    corpus_audit.add_argument("--output", type=Path, required=True)

    embed = subparsers.add_parser("embed", help="generate resumable Doubao gene embeddings")
    embed.add_argument("--texts", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--checkpoint", type=Path, required=True)
    embed.add_argument("--model", default=DEFAULT_MODEL)
    embed.add_argument("--base-url", default=DEFAULT_BASE_URL)
    embed.add_argument("--dimensions", type=int)
    embed.add_argument("--expected-dimension", type=int, default=2048)
    embed.add_argument("--batch-size", type=int, default=10)
    embed.add_argument("--max-workers", type=int, default=1)
    embed.add_argument("--request-interval", type=float, default=0.0)
    embed.add_argument("--limit", type=int)
    embed.add_argument("--genes", type=Path, help="optional newline-delimited gene allowlist")
    embed.add_argument(
        "--preserve-gene-case",
        action="store_true",
        help="preserve exact gene-symbol case for strict downstream matching",
    )

    checkpoint_audit = subparsers.add_parser(
        "audit-checkpoint", help="count exact text-hash hits in an embedding checkpoint"
    )
    checkpoint_audit.add_argument("--texts", type=Path, required=True)
    checkpoint_audit.add_argument("--checkpoint", type=Path, required=True)
    checkpoint_audit.add_argument("--model", default=DEFAULT_MODEL)
    checkpoint_audit.add_argument("--genes", type=Path)
    checkpoint_audit.add_argument("--preserve-gene-case", action="store_true")

    texts = subparsers.add_parser("audit-texts", help="inspect source text size and coverage")
    texts.add_argument("--texts", type=Path, required=True)
    texts.add_argument("--genes", type=Path)
    texts.add_argument("--output-selected", type=Path)
    texts.add_argument("--preserve-gene-case", action="store_true")

    audit = subparsers.add_parser("audit-vectors", help="inspect vector shape and optional coverage")
    audit.add_argument("--vectors", type=Path, required=True)
    audit.add_argument("--genes", type=Path)
    audit.add_argument("--trusted-pickle", action="store_true")

    comparison = subparsers.add_parser(
        "audit-ggi-comparison", help="enforce matched GGI receipts and compute deltas"
    )
    comparison.add_argument("--result", type=Path, action="append", required=True)
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--output", type=Path, required=True)

    knowledge_vector_audit = subparsers.add_parser(
        "audit-knowledge-vectors", help="prove exact graph, target, and GGI vector coverage"
    )
    knowledge_vector_audit.add_argument("--genes", type=Path, required=True)
    knowledge_vector_audit.add_argument("--ggi-genes", type=Path, required=True)
    knowledge_vector_audit.add_argument("--gradpert-root", type=Path, required=True)
    knowledge_vector_audit.add_argument("--protein", type=Path, required=True)
    knowledge_vector_audit.add_argument("--pathway", type=Path, required=True)
    knowledge_vector_audit.add_argument("--hpa", type=Path, required=True)
    knowledge_vector_audit.add_argument("--expected-dimension", type=int, default=2048)
    knowledge_vector_audit.add_argument("--output", type=Path, required=True)

    benchmark = subparsers.add_parser("benchmark", help="run selected GenePT paper benchmarks")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    for name in ("ggi", "properties"):
        command = benchmark_sub.add_parser(name)
        command.add_argument("--vectors", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--name", required=True, help="embedding condition label")
        command.add_argument("--trusted-pickle", action="store_true")
        command.add_argument(
            "--normalize",
            action="store_true",
            help="L2-normalize each gene vector before evaluation",
        )
        if name == "ggi":
            command.add_argument("--data", type=Path, required=True)
            command.add_argument(
                "--genes",
                type=Path,
                help="fixed newline-delimited gene universe shared by all embeddings",
            )
        else:
            command.add_argument("--tasks", type=Path, required=True)
            command.add_argument("--folds", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = {
            "python": platform.python_version(),
            "ark_api_key_present": credential_present("ARK_API_KEY"),
        }
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "data":
        if args.data_command == "prepare-genept":
            result = prepare_genept(args.output, keep_archive=args.keep_archive)
        elif args.data_command == "prepare-ggi":
            result = prepare_ggi(args.output)
        elif args.data_command == "prepare-go-exp":
            result = prepare_go_exp(args.output)
        elif args.data_command == "prepare-knowledge-sources":
            result = prepare_knowledge_sources(args.output)
        elif args.data_command == "extend-genept-texts":
            result = extend_genept_texts(
                base_path=args.base,
                genes_path=args.genes,
                output_path=args.output,
                manifest_path=args.manifest,
                checkpoint_path=args.checkpoint,
            )
        elif args.data_command == "build-axis-texts":
            result = build_axis_corpus(
                source_path=args.source,
                genes_path=args.genes,
                hgnc_path=args.hgnc,
                axis_mapping_path=args.axis_mapping,
                ensembl_archive_path=args.ensembl_archive,
                output_path=args.output,
                manifest_path=args.manifest,
                allow_case_duplicates=args.allow_case_duplicates,
                allow_identity_only=args.allow_identity_only,
            )
        elif args.data_command == "build-axis-vectors":
            result = materialize_axis_vectors(
                source_path=args.source,
                genes_path=args.genes,
                hgnc_path=args.hgnc,
                axis_mapping_path=args.axis_mapping,
                output_path=args.output,
                manifest_path=args.manifest,
                model=args.model,
                trusted_pickle=args.trusted_pickle,
            )
        elif args.data_command == "align-axis-vectors":
            result = align_npz_to_axis(
                source_path=args.source,
                genes_path=args.genes,
                output_path=args.output,
                manifest_path=args.manifest,
            )
        elif args.data_command == "build-go-exp-texts":
            result = build_go_exp_corpus(
                base_path=args.base,
                genes_path=args.genes,
                gaf_path=args.gaf,
                obo_path=args.obo,
                output_path=args.output,
                manifest_path=args.manifest,
                max_terms_per_aspect=args.max_terms_per_aspect,
                include_interaction_evidence=args.include_interaction_evidence,
                allow_case_duplicates=args.allow_case_duplicates,
            )
        elif args.data_command == "build-gradpert-union":
            result = build_gradpert_union(
                gradpert_root=args.gradpert_root,
                extra_genes_path=args.extra_genes,
                output_path=args.output,
                manifest_path=args.manifest,
            )
        elif args.data_command == "build-knowledge-texts":
            result = build_knowledge_corpus(
                base_path=args.base,
                genes_path=args.genes,
                uniprot_path=args.uniprot,
                interpro_path=args.interpro,
                reactome_path=args.reactome,
                signor_path=args.signor,
                hpa_path=args.hpa,
                profile=args.profile,
                max_items_per_source=args.max_items_per_source,
                max_characters_per_field=args.max_characters_per_field,
                output_path=args.output,
                manifest_path=args.manifest,
            )
        elif args.data_command == "audit-knowledge-corpora":
            result = audit_knowledge_corpora(
                genes_path=args.genes,
                gradpert_root=args.gradpert_root,
                base_path=args.base,
                protein_path=args.protein,
                pathway_path=args.pathway,
                hpa_path=args.hpa,
                output_path=args.output,
            )
        else:
            selected = sorted(ggi_genes(load_ggi(args.data)))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("\n".join(selected) + "\n", encoding="utf-8")
            result = {"genes": len(selected), "output": str(args.output)}
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "audit-texts":
        uppercase_genes = not args.preserve_gene_case
        gene_texts = load_gene_texts(args.texts, uppercase_genes=uppercase_genes)
        report = text_statistics(gene_texts)
        if args.genes:
            requested = set(args.genes.read_text(encoding="utf-8").splitlines())
            selected = select_gene_texts(
                gene_texts,
                requested,
                uppercase_genes=uppercase_genes,
            )
            report["requested_genes"] = len(
                {
                    gene.upper() if uppercase_genes else gene
                    for gene in requested
                    if gene
                }
            )
            report["selected_genes"] = len(selected)
            report["selected_coverage"] = len(selected) / report["requested_genes"]
            if args.output_selected:
                args.output_selected.parent.mkdir(parents=True, exist_ok=True)
                args.output_selected.write_text(
                    "\n".join(sorted(selected)) + "\n",
                    encoding="utf-8",
                )
                report["selected_output"] = str(args.output_selected)
        elif args.output_selected:
            raise ValueError("--output-selected requires --genes")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "embed":
        client = ArkEmbeddingClient(base_url=args.base_url, model=args.model, dimensions=args.dimensions)
        uppercase_genes = not args.preserve_gene_case
        gene_texts = load_gene_texts(args.texts, uppercase_genes=uppercase_genes)
        if args.genes:
            requested = set(args.genes.read_text(encoding="utf-8").splitlines())
            gene_texts = select_gene_texts(
                gene_texts,
                requested,
                uppercase_genes=uppercase_genes,
            )
        vectors = generate_embeddings(
            gene_texts, embed=client.embed, model=args.model,
            checkpoint_path=args.checkpoint, output_path=args.output,
            batch_size=args.batch_size, max_workers=args.max_workers,
            request_interval=args.request_interval, limit=args.limit,
            expected_dimension=args.expected_dimension,
            uppercase_genes=uppercase_genes,
        )
        print(json.dumps({"model": args.model, "genes": len(vectors), "output": str(args.output)}, indent=2))
        return 0
    if args.command == "audit-checkpoint":
        uppercase_genes = not args.preserve_gene_case
        gene_texts = load_gene_texts(args.texts, uppercase_genes=uppercase_genes)
        if args.genes:
            requested = set(args.genes.read_text(encoding="utf-8").splitlines())
            gene_texts = select_gene_texts(
                gene_texts, requested, uppercase_genes=uppercase_genes
            )
        print(
            json.dumps(
                audit_embedding_checkpoint(
                    gene_texts, checkpoint_path=args.checkpoint, model=args.model
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "audit-ggi-comparison":
        result = build_ggi_comparison(
            result_paths=args.result, baseline=args.baseline, output_path=args.output
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "audit-knowledge-vectors":
        result = audit_knowledge_vectors(
            genes_path=args.genes,
            ggi_genes_path=args.ggi_genes,
            gradpert_root=args.gradpert_root,
            protein_path=args.protein,
            pathway_path=args.pathway,
            hpa_path=args.hpa,
            output_path=args.output,
            expected_dimension=args.expected_dimension,
        )
        print(json.dumps(result, indent=2))
        return 0
    loaded = _embedding(
        args.vectors,
        trusted_pickle=args.trusted_pickle,
        normalize=getattr(args, "normalize", False),
    )
    if args.command == "audit-vectors":
        report: dict[str, object] = {
            "model": loaded.model,
            "genes": len(loaded.genes),
            "dimension": int(loaded.vectors.shape[1]),
        }
        if args.genes:
            report["coverage"] = coverage(loaded, set(args.genes.read_text().splitlines()))
        print(json.dumps(report, indent=2))
        return 0
    if args.benchmark_command == "ggi":
        if args.genes:
            universe = {
                gene.strip()
                for gene in args.genes.read_text(encoding="utf-8").splitlines()
                if gene.strip()
            }
            vectors = select_universe_vectors(loaded, universe)
        else:
            vectors = loaded.as_dict()
        rows = [evaluate_ggi(load_ggi(args.data), vectors)]
    else:
        vectors = loaded.as_dict()
        rows = []
        for task in load_property_tasks(args.tasks).values():
            rows.extend(evaluate_property_task(task, vectors, folds=args.folds))
    data_receipt = (
        args.data / "ggi_manifest.json"
        if args.benchmark_command == "ggi"
        else args.tasks
    )
    rows = [
        {
            "embedding": args.name,
            "normalization": "l2" if args.normalize else "none",
            "gene_universe": str(args.genes) if getattr(args, "genes", None) else "embedding-native",
            "gene_universe_sha256": (
                digest_file(args.genes) if getattr(args, "genes", None) else None
            ),
            "vectors_sha256": digest_file(args.vectors),
            "data_receipt_sha256": digest_file(data_receipt),
            "random_state": 42,
            "genept_seed_version": __version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            **row,
        }
        for row in rows
    ]
    write_results(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))
    return 0
