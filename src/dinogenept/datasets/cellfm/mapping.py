"""Exact symbol-to-Census-ID mapping; no guessing, aliases or row removal."""


def map_axis(symbols, targets, vocabulary, feature_records):
    if len(set(symbols)) != len(symbols) or not symbols or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Gene axes must be nonempty/unique")
    positions = {gene: i + 1 for i, gene in enumerate(vocabulary)}
    names = {}
    for record in feature_records:
        names.setdefault(record["feature_name"], set()).add(record["feature_id"])
    ambiguous = [symbol for symbol in symbols if len(names.get(symbol, set())) != 1]
    if ambiguous:
        raise ValueError(f"Missing/ambiguous exact Census gene symbols: {ambiguous}")
    ensembl = [next(iter(names[symbol])) for symbol in symbols]
    if set(ensembl) - set(positions) or len(set(ensembl)) != len(ensembl):
        raise ValueError("Census mapping is outside or duplicates the frozen vocabulary")
    if set(targets) - set(symbols):
        raise ValueError("Perturbation target outside prediction axis")
    native = [positions[gene] for gene in ensembl]
    lookup = dict(zip(symbols, native, strict=True))
    return {
        "symbols": list(symbols),
        "ensembl_ids": ensembl,
        "model_gene_ids": native,
        "target_model_ids": {symbol: lookup[symbol] for symbol in targets},
        "policy": "exact_unique_feature_name_to_feature_id_no_alias_no_drop",
        "vocabulary_size": len(vocabulary),
    }
