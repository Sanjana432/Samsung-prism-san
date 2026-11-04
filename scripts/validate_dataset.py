"""CPU-only dataset gate before expensive model runs. Never modifies images or labels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism.common import P, load_config, read_jsonl
from prism.dataset import assert_dataset_ready, dataset_fingerprint, validate_pairs, validate_samples
from prism.validation import validate_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--allow-empty-splits", action="store_true", help="CPU pilot only; training still requires nonempty train/val")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert_dataset_ready(cfg)
    samples = read_jsonl(P(cfg, "metadata_dir", "ladder.jsonl"))
    engine = Path(P(cfg, "metadata_dir", "ladder_engine.jsonl"))
    if engine.exists():
        samples += read_jsonl(str(engine))
    summary = validate_samples(samples, root=cfg["_root"], verify_images=True)
    pairs = {split: read_jsonl(P(cfg, "pairs_dir", f"{split}.jsonl")) for split in ("train", "val", "test")}
    validate_pairs(pairs, cfg["splits"].get("holdout_templates", []), allow_empty=args.allow_empty_splits)
    expected = {f"{s['sample_id']}_{r['release_id']}": (s, r) for s in samples for r in s["releases"]}
    actual = {row["id"]: row for rows in pairs.values() for row in rows}
    if set(expected) != set(actual):
        raise ValueError("split membership does not cover exactly the generated metadata pairs")
    for pair_id, row in actual.items():
        sample, release = expected[pair_id]
        if (row["scores"] != release["scores"] or row["original_image_path"] != sample["original_image_path"]
                or row["release_image_path"] != release["release_image_path"]):
            raise ValueError(f"split disagrees with measured metadata: {pair_id}")
        scores, _ = validate_output(row.get("target", ""))
        if scores != row["scores"]:
            raise ValueError(f"training target disagrees with measured scores: {pair_id}")
    summary.update(split_pairs={s: len(rows) for s, rows in pairs.items()}, fingerprint=dataset_fingerprint(cfg))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
