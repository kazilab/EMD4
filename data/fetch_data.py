#!/usr/bin/env python3
"""Prepare the public-data cache from downloads or existing files using pinned sizes and SHA-256 hashes."""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = HERE / "input_sources.json"
DEFAULT_DEST = HERE / "public"


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records():
    return json.loads(PINS.read_text())


def install(stream, rec: dict, raw: Path) -> None:
    """Check the size and hash before installing the input and its pinned provenance record."""
    target = raw / rec["filename"]
    with tempfile.TemporaryDirectory(dir=raw) as tmp:
        staging = Path(tmp) / "input"
        with staging.open("wb") as fh:
            shutil.copyfileobj(stream, fh)
        size, got = staging.stat().st_size, sha256(staging)
        if size != rec["bytes"] or got != rec["sha256"]:
            raise ValueError(
                f"expected {rec['bytes']} bytes and SHA-256 {rec['sha256']}, "
                f"received {size} bytes and SHA-256 {got}")
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
    if not args.verify:
        raw.mkdir(parents=True, exist_ok=True)

    problems, installed, ok, absent = [], 0, 0, 0
    for rec in records():
        name = rec["filename"]
        target = raw / name
        if target.exists():
            if target.stat().st_size == rec["bytes"] and sha256(target) == rec["sha256"]:
                provenance = raw / (name + ".provenance.json")
                expected = json.dumps(rec, indent=2) + "\n"
                if args.verify:
                    try:
                        matches = json.loads(provenance.read_text()) == rec
                    except (FileNotFoundError, ValueError):
                        matches = False
                    if not matches:
                        problems.append(f"{name}: missing or inconsistent provenance")
                        continue
                else:
                    provenance.write_text(expected)
                print(f"  ok        {name}")
                ok += 1
            else:
                print(f"  MISMATCH  {name}")
                problems.append(f"{name}: checksum mismatch")
            continue

        if args.verify:
            print(f"  absent    {name}  ({rec['bytes']:,} bytes)")
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
        except Exception as exc:
            print(f"  FAILED    {name}: {exc}")
            problems.append(f"{name}: {exc}")
            continue
        installed += 1

    print(f"\n  {ok} verified, {installed} installed, {absent} absent, "
          f"{len(problems)} problem(s)")
    for p in problems:
        print(f"    - {p}")
    return 1 if problems or (args.verify and absent) else 0


if __name__ == "__main__":
    raise SystemExit(main())
