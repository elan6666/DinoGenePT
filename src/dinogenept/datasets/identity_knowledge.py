"""Opt-in native adapter: unchanged verified vectors, shared HGNC identities."""

import numpy as np

from dinogenept.gene_identity import GeneIdentityIndex, build_identity_report, read_identity_json

from .knowledge import KnowledgeBank, _read_pinned


class IdentityKnowledgeBank:
    """Keep axis order; missing optional sources yield None, conflicts fail closed.

    This adapter does not bypass KnowledgeBank's corpus/vector/model/hash checks.
    It is not automatically injected into existing fine-tuning configurations.
    """

    def __init__(self, *, hgnc, hgnc_sha256, axis_records, sources, model, width=2048):
        identity = GeneIdentityIndex(hgnc, expected_sha256=hgnc_sha256)
        corpora = {name: read_identity_json(_read_pinned(entry["corpus"])) for name, entry in sources.items()}
        report = build_identity_report(identity, axis_records, corpora)
        if not report["safe_unique_axis"]:
            raise ValueError("Axis has unresolved/conflicting/duplicate identities; inspect identity audit")
        # Validate all original vector artifacts before selecting/rekeying any row.
        original = KnowledgeBank(sources, list(corpora["TextBase"]), model=model, width=width)
        self.tables, self.coverage = {}, {}
        self.genes = tuple(row["original_label"] for row in report["rows"])
        self.report = report
        for name, data in report["sources"].items():
            table, missing = {}, []
            for gene, join in zip(self.genes, data["joins"], strict=True):
                if join["status"] == "source_conflict":
                    raise ValueError(f"Conflicting source texts for {name}/{gene}; no automatic winner")
                if join["status"] == "source_identity_ambiguous":
                    raise ValueError(f"Unresolved source identity for {name}/{gene}; not missing annotation")
                if join["status"] != "available":
                    if name == "TextBase":
                        raise ValueError(f"Required TextBase missing resolved identity: {gene}")
                    missing.append(gene)
                    continue
                # Identical text keys must also have identical verified vectors.
                vector = original.tables[name][join["source_key"]]
                if any(not np.array_equal(vector, original.tables[name][key]) for key in join["equivalent_keys"]):
                    raise ValueError(f"Conflicting cached vectors for identical text: {name}/{gene}")
                table[gene] = vector
            self.tables[name] = table
            self.coverage[name] = {"required_genes": len(self.genes), "available": len(table), "missing": missing}
        self.model, self.width = model, width

    def for_targets(self, targets):
        # Reuse our existing strict combination/missing-source behavior.
        return KnowledgeBank.for_targets(self, targets)

    def for_axis(self, source="TextBase"):
        """Return available rows and their axis positions, never fabricate zeros."""
        table = self.tables[source]
        positions = np.asarray([i for i, gene in enumerate(self.genes) if gene in table], dtype=np.int64)
        vectors = (np.stack([table[self.genes[i]] for i in positions]) if len(positions)
                   else np.empty((0, self.width), dtype=np.float32))
        return positions, vectors
