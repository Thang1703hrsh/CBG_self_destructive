"""Scenario datasets share the same immutable split manifest."""
from src.data_processing.cbg_datasets import CbgPairedSafetyDataset, CbgBenignDataset
from src.experiments.protocol import read_json, validate_manifest


def defense_datasets(manifest_path, tokenizer, args):
    manifest = read_json(manifest_path)
    loaded = validate_manifest(manifest)
    indices = manifest["defense_indices"]
    if len(indices) != args.defense_size:
        raise ValueError("defense_size differs from frozen scenario manifest")
    if len(manifest["benign_indices"]) != args.defense_size:
        raise ValueError("scenario defense benign pool must match defense_size")
    safety = CbgPairedSafetyDataset([loaded["safety"][i] for i in indices], tokenizer,
        source_indices=indices, max_length=args.safety_max_length)
    benign = CbgBenignDataset([loaded["benign"][i] for i in manifest["benign_indices"]], tokenizer,
        max_length=args.benign_max_length)
    # Preserve compatibility with legacy attack manifest readers while retaining
    # all three pools and both source hashes in checkpoint fingerprints.
    metadata = dict(manifest, safety_source_sha256=manifest["sources"]["safety"]["sha256"],
                    benign_source=manifest["sources"]["benign"]["path"], benign_examples=len(benign))
    return safety, benign, metadata
