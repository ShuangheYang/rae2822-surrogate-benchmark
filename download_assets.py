"""Download immutable public artifacts with SHA-256 verification; never require credentials."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import ROOT, PROJECT_ROOT, read_json, file_hash, write_json


def fetch(entry, offline_release=None):
    target = ROOT / entry["destination"]
    if not target.resolve().is_relative_to(ROOT):
        raise ValueError(
            "Artifact destination must stay inside the experiment directory."
        )
    if target.exists():
        if file_hash(target) != entry["sha256"]:
            raise RuntimeError(
                f"Existing file differs from the pinned release: {target}. Use a new run directory."
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    try:
        if offline_release:
            source = offline_release / entry["local_repository"] / entry["path"]
            shutil.copyfile(source, temporary)
        else:
            request = urllib.request.Request(
                entry["url"], headers={"User-Agent": "rae2822-surrogate-benchmark"}
            )
            with urllib.request.urlopen(
                request, timeout=120
            ) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if file_hash(temporary) != entry["sha256"]:
            raise RuntimeError(f'Download checksum mismatch: {entry["path"]}')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", choices=["inputs", "reference", "models"], required=True
    )
    parser.add_argument(
        "--offline-release",
        type=Path,
        help="Optional local dataset/mlp/kriging repository root; verified against the same hashes",
    )
    args = parser.parse_args()
    manifest = read_json(PROJECT_ROOT / "configs/assets.lock.json")
    groups = {args.group}
    if args.group == "reference":
        if (ROOT / "models/kriging").exists() or (
            ROOT / "results/mlp_learning_complete.json"
        ).exists():
            raise RuntimeError(
                "Do not overlay published results onto a trained experiment."
            )
        groups.add("inputs")
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "REFERENCE_ONLY").write_text(
            "Published evidence; do not train in this directory.\n"
        )
    items = [e for e in manifest["artifacts"] if e["group"] in groups]
    for i, entry in enumerate(items, 1):
        fetch(entry, args.offline_release)
        print(f'[{i}/{len(items)}] verified {entry["destination"]}', flush=True)
    if args.group in ("inputs", "reference"):
        protocol = PROJECT_ROOT / "configs/protocol.json"
        dest = ROOT / "protocol.json"
        if dest.exists() and file_hash(dest) != file_hash(protocol):
            raise RuntimeError("Protocol differs: use a fresh experiment directory.")
        shutil.copyfile(protocol, dest)
        validate_data()
    write_json(
        ROOT / f"assets_{args.group}.json",
        {
            "lock_sha256": file_hash(PROJECT_ROOT / "configs/assets.lock.json"),
            "artifacts": [
                {"file": e["destination"], "sha256": e["sha256"]} for e in items
            ],
        },
    )


def validate_data():
    """Check source pool and split identities, including geometry overlap across independent LHS pools."""
    import numpy as np

    split = read_json(ROOT / "data/split.json")
    with np.load(ROOT / "data/development.npz", allow_pickle=False) as d, np.load(
        ROOT / "data/sealed_test.npz", allow_pickle=False
    ) as t:
        assert d["X"].shape == (10000, 20) and d["Y"].shape == (10000, 2)
        assert d["X_val"].shape == (500, 20) and t["X"].shape == (1500, 20)
        assert d["ids"].tolist() == split["train_ids"]
        assert d["ids_val"].tolist() == split["validation_ids"]
        assert t["ids"].tolist() == split["test_ids"]
        assert not set(d["ids_val"]) & set(t["ids"])
        train = {row.tobytes() for row in d["X"]}
        other = np.concatenate([d["X_val"], t["X"]])
        assert len(train) == 10000 and len({row.tobytes() for row in other}) == 2000
        assert not train & {row.tobytes() for row in other}
        for arr in [d["X"], d["Y"], d["X_val"], d["Y_val"], t["X"], t["Y"]]:
            assert np.isfinite(arr).all()
        for arr in [d["Y"], d["Y_val"], t["Y"]]:
            assert (arr[:, 1] > 0).all()
    print(
        "Verified 10,000 training / 500 validation / 1,500 test rows; no identical cross-pool geometries."
    )


if __name__ == "__main__":
    main()
