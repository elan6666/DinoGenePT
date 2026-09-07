"""One foreground, checkpointed Agent Plan job; never print or persist keys."""

import argparse
import fcntl
import json
import time
from pathlib import Path

from dinogenept.datasets.knowledge import KNOWLEDGE_SOURCES, KnowledgeBank
from dinogenept.embedding import (
    DEFAULT_MODEL,
    ArkEmbeddingClient,
    audit_embedding_checkpoint,
    generate_embeddings,
    merge_exact_checkpoint_hits,
)
from dinogenept.provenance import atomic_write_json, digest_file

SEED_CACHES = {
    "TextBase": "doubao.sqlite3",
    "GO": "dinogenept-go-exp-only-targets.sqlite3",
    "Protein": "dinogenept-protein-only-targets.sqlite3",
    "Pathway": "dinogenept-pathway-only-targets.sqlite3",
    "HPA": "dinogenept-hpa-only-targets.sqlite3",
}


def pinned(path):
    return {"path": str(path.resolve()), "sha256": digest_file(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.corpora / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for name, sha in receipt["files_sha256"].items():
        if digest_file(args.corpora / name) != sha:
            raise ValueError("Prepared corpus file changed")
    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    with (args.output / ".embedding.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {
            "corpus_receipt_sha256": digest_file(receipt_path),
            "model": DEFAULT_MODEL,
            "width": 2048,
            "batch_size": 10,
            "workers": 1,
            "request_interval": 8,
            "seed_cache_policy": "one_frozen_source_per_profile_no_ambiguous_multi_cache_merge",
            "seed_caches": {s: pinned(Path("checkpoints") / f) for s, f in SEED_CACHES.items()},
        }
        identity_path = args.output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Refusing to resume with different corpus/model/cache lineage")
        else:
            if any(args.output.glob("*.npz")) or any(args.checkpoints.glob("*.sqlite3")):
                raise FileExistsError("Existing unbound artifacts require inspection, not adoption")
            atomic_write_json(identity_path, identity)
        entries = {}
        client = None
        for source in KNOWLEDGE_SOURCES:
            corpus_path = args.corpora / f"{source}.json"
            texts = json.loads(corpus_path.read_text())
            entry = {"corpus": pinned(corpus_path)}
            if source != "TextBase":
                entry["source_manifest"] = pinned(args.corpora / f"{source}.source.json")
            entries[source] = entry
            if not texts:
                continue
            checkpoint = args.checkpoints / f"{source}.sqlite3"
            vector_path = args.output / f"{source}.npz"
            manifest_path = vector_path.with_suffix(".npz.manifest.json")
            if vector_path.exists() and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if (
                    manifest["output_sha256"] != digest_file(vector_path)
                    or manifest["corpus_sha256"] != digest_file(corpus_path)
                    or manifest["model"] != DEFAULT_MODEL
                    or manifest["dimension"] != 2048
                ):
                    raise ValueError("Existing completed source artifact differs")
                print(json.dumps({"source": source, "status": "existing_artifact_not_regenerated"}), flush=True)
            else:
                merge_exact_checkpoint_hits(
                    texts,
                    source_paths=[Path(identity["seed_caches"][source]["path"])],
                    destination_path=checkpoint,
                    model=DEFAULT_MODEL,
                )
                audit = audit_embedding_checkpoint(texts, checkpoint_path=checkpoint, model=DEFAULT_MODEL)
                if audit["dimensions"] not in ([], [2048]):
                    raise ValueError("Seed cache dimension differs")
                print(json.dumps({"source": source, "status": "generating", **audit}), flush=True)
                if audit["pending"]:
                    if client is None:
                        client = ArkEmbeddingClient(max_retries=0)  # stop on 429/401; never accelerate retries
                    time.sleep(8)  # also rate-limit transitions between source profiles

                def embed(text_batch, api_client=client, source_name=source):
                    if api_client is None:
                        raise RuntimeError("Unexpected cache miss after a complete checkpoint audit")
                    vectors = api_client.embed(text_batch)
                    print(
                        json.dumps({"source": source_name, "status": "api_batch_returned", "texts": len(text_batch)}),
                        flush=True,
                    )
                    return vectors

                generate_embeddings(
                    texts,
                    embed=embed,
                    model=DEFAULT_MODEL,
                    checkpoint_path=checkpoint,
                    output_path=vector_path,
                    batch_size=10,
                    max_workers=1,
                    request_interval=8,
                    expected_dimension=2048,
                    uppercase_genes=False,
                    profile=f"cellfm-{source}-only",
                    corpus_path=corpus_path,
                    universe_path=args.corpora / "genes.txt",
                )
                final = audit_embedding_checkpoint(texts, checkpoint_path=checkpoint, model=DEFAULT_MODEL)
                if final["pending"] or final["dimensions"] != [2048]:
                    raise ValueError("Completed source does not pass exact checkpoint audit")
                print(json.dumps({"source": source, "status": "checkpoint_verified", **final}), flush=True)
            entry.update(vectors=pinned(vector_path), embedding_manifest=pinned(manifest_path))
        genes = (args.corpora / "genes.txt").read_text().splitlines()
        bank = KnowledgeBank(entries, genes, model=DEFAULT_MODEL, width=2048)
        atomic_write_json(
            args.output / "bundle.json",
            {
                "status": "verified",
                "identity": identity,
                "coverage": bank.coverage,
                "knowledge": {"sources": entries, "model": DEFAULT_MODEL, "width": 2048},
            },
        )
        print(json.dumps({"status": "all_sources_verified", "genes": len(genes)}), flush=True)


if __name__ == "__main__":
    main()
