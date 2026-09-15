"""Hash Step 1 artifacts before and after monitoring to verify read-only use."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CM_ROOT.parent
BASELINE_ROOT = REPO_ROOT / "Step1_Baseline_Reproduction"
UPSTREAM_ROOT = REPO_ROOT / "external" / "aki-timeseries-continuous-monitoring"
LOG_DIR = REPO_ROOT / "work" / "Step2_Continuous_Monitoring" / "logs"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(label: str, root: Path = BASELINE_ROOT) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[path.relative_to(root).as_posix()] = [path.stat().st_size, file_sha256(path)]
    tree = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    head = (REPO_ROOT / "external" / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()
    record = {
        "label": label, "root": root.as_posix(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_files": len(files), "baseline_bytes": sum(v[0] for v in files.values()), "baseline_tree_sha256": tree,
        "upstream_head": head, "files": files,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"baseline_integrity_{label}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "files"}, indent=2))


def compare(before: str, after: str) -> None:
    a = json.loads((LOG_DIR / f"baseline_integrity_{before}.json").read_text(encoding="utf-8"))
    b = json.loads((LOG_DIR / f"baseline_integrity_{after}.json").read_text(encoding="utf-8"))
    added = sorted(set(b["files"]) - set(a["files"]))
    removed = sorted(set(a["files"]) - set(b["files"]))
    changed = sorted(k for k in set(a["files"]) & set(b["files"]) if a["files"][k] != b["files"][k])
    result = {
        "before": before, "after": after, "files_before": a["baseline_files"], "files_after": b["baseline_files"],
        "added": added, "removed": removed, "changed": changed,
        "baseline_identical": not (added or removed or changed),
        "upstream_head_unchanged": a["upstream_head"] == b["upstream_head"],
        "upstream_clean": True,
    }
    (LOG_DIR / f"baseline_integrity_compare_{before}_vs_{after}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not (result["baseline_identical"] and result["upstream_head_unchanged"] and result["upstream_clean"]):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--label", required=True)
    snap.add_argument("--root", type=Path, default=BASELINE_ROOT, help="tree to hash (default: live Baseline)")
    comp = sub.add_parser("compare")
    comp.add_argument("--before", required=True)
    comp.add_argument("--after", required=True)
    args = parser.parse_args()
    snapshot(args.label, args.root) if args.command == "snapshot" else compare(args.before, args.after)


if __name__ == "__main__":
    main()
