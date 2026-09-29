#!/usr/bin/env python3
"""Fetch or verify the public-data inputs. The core simulation needs none.

Only the optional public-data comparison (`python -m emd4_simulation.public_data
complete --offline`) reads third-party files. This deposit bundles none of
them: they total about 322 MiB and remain with their original repositories.

The pins are the 24 records in
`code/emd4_simulation/public_data_results/complete_source_manifest.json`, which
is the manifest the deposited public-data results were generated from. Each
file is downloaded to a staging name, checked against its pinned byte count and
SHA-256, and only then installed together with its provenance record, so the
workflow's own offline check (`public_data.sources.fetch_source`) accepts it.
The record is written exactly as pinned, including the original `retrieved_utc`:
it describes the snapshot, and matching SHA-256 means this is the same one.

Run from any directory. `--verify` downloads nothing and exits non-zero only on
a checksum mismatch; absent inputs are optional and are reported, not failed.
`--from DIR` copies from an existing local cache instead of downloading.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "code" / "emd4_simulation"
PINS = PACKAGE / "public_data_results" / "complete_source_manifest.json"
DEFAULT_DEST = PACKAGE / "data" / "public"


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records():
    manifest = json.loads(PINS.read_text())
    return manifest["sources"] + manifest["extension_sources"]


def install(stream, rec: dict, raw: Path) -> None:
    """Never install a partial or mismatched file as an analysis input."""
    target = raw / rec["filename"]
    with tempfile.NamedTemporaryFile(dir=raw, delete=False) as fh:
        staging = Path(fh.name)
        shutil.copyfileobj(stream, fh)
    size, got = staging.stat().st_size, sha256(staging)
    if size != rec["bytes"] or got != rec["sha256"]:
        staging.unlink(missing_ok=True)
        raise ValueError(
            f"checksum mismatch: expected {rec['bytes']} bytes, "
            f"sha256 {rec['sha256']}; got {size} bytes, sha256 {got}. The "
            "source has changed since this deposit was made. Nothing was "
            "installed.")
    staging.replace(target)
    (raw / (rec["filename"] + ".provenance.json")).write_text(
        json.dumps(rec, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check existing files without downloading")
    ap.add_argument("--from", dest="source_dir", default=None,
                    help="copy from this local directory instead of downloading")
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help="data directory (default: the package's data/public)")
    args = ap.parse_args()
    raw = Path(args.dest).resolve() / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    problems, installed, ok, absent = [], 0, 0, 0
    for rec in records():
        name = rec["filename"]
        target = raw / name
        if target.exists():
            if sha256(target) == rec["sha256"]:
                print(f"  ok        {name}")
                ok += 1
            else:
                print(f"  MISMATCH  {name}")
                problems.append(f"{name}: checksum mismatch")
            continue

        if args.verify:
            print(f"  absent    {name}  (optional; {rec['bytes']:,} bytes)")
            absent += 1
            continue

        origin = (Path(args.source_dir) / name) if args.source_dir else rec["url"]
        print(f"  {'copying' if args.source_dir else 'fetching'}  {name}",
              flush=True)
        try:
            if args.source_dir:
                with Path(origin).open("rb") as stream:
                    install(stream, rec, raw)
            else:
                req = urllib.request.Request(
                    origin, headers={"User-Agent": "EMD4-public-data/1.0"})
                with urllib.request.urlopen(req, timeout=90) as stream:
                    install(stream, rec, raw)
        except Exception as exc:              # network, 404, timeout, mismatch
            print(f"  FAILED    {name}: {exc}")
            problems.append(f"{name}: {exc}")
            continue
        installed += 1

    print(f"\n  {ok} verified, {installed} installed, {absent} absent, "
          f"{len(problems)} problem(s)")
    for p in problems:
        print(f"    - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
