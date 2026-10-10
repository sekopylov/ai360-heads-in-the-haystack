#!/usr/bin/env python
"""Add or refresh a `configs/models.json` entry, with its SHA-256 pins.

The registry is not documentation: `--verify-hashes` re-checks every pinned digest
before a job spends money, and since the resume fingerprint includes the entry, a
wrong pin is a wrong run.  Pinning by hand is also how a 60 GiB model ends up
needing a 60 GiB download just to learn its hashes.

    .venv/bin/python scripts/pin_registry.py --key qwen3-4b --repo Qwen/Qwen3-4B
    .venv/bin/python scripts/pin_registry.py --key qwen3-8b --repo Qwen/Qwen3-8B --dry-run

How each digest is obtained:

* a **weight shard** that is already on disk (complete, same size) is hashed
  locally -- authoritative;
* a shard that is *not* on disk takes its digest from the Hub's own LFS metadata
  (`lfs.sha256`), so a model can be pinned without downloading it.  The job
  re-verifies it with `--verify-hashes`, so a wrong pin fails loudly there;
* every **small file** (`config.json`, the tokenizer, `chat_template.jinja`, ...)
  is downloaded into the model directory and hashed locally: they are a few MB,
  they are what the tokenizer-dependent tests need, and a hash from a HEAD
  request is not the same guarantee.

The file set is "everything the repo ships except the weights, the README, the
licence and `.gitattributes`", which is what the existing entries list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "configs" / "models.json"
#: Files that are never needed to load a model or render its prompt.
SKIP = {".gitattributes", "LICENSE", "README.md", "USE_POLICY.md", "LICENSE.txt"}
API = "https://huggingface.co/api/models/{repo}?blobs=true"


def hub_files(repo: str) -> list[dict]:
    """The repo's file list with sizes and LFS digests, from the Hub API."""
    request = urllib.request.Request(API.format(repo=repo),
                                     headers={"User-Agent": "retrieval-heads-pin"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.load(response)["siblings"]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path, expect_size: int | None) -> None:
    """Download ``url`` to ``target`` unless a same-sized copy is already there."""
    if target.exists() and expect_size is not None and target.stat().st_size == expect_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"    downloading {target.name} ({expect_size or 0} B)")
    request = urllib.request.Request(url, headers={"User-Agent": "retrieval-heads-pin"})
    with urllib.request.urlopen(request, timeout=300) as response, \
            target.open("wb") as handle:  # noqa: S310
        while block := response.read(1 << 22):
            handle.write(block)


def build_entry(repo: str, directory: Path, *, notes: str, chat_template: bool,
                dry_run: bool) -> dict:
    files: dict[str, str] = {}
    shards: dict[str, str] = {}
    from_hub = 0
    entries = [entry for entry in hub_files(repo) if entry["rfilename"] not in SKIP]

    # A repo can ship the same weights twice: Mistral publishes `consolidated.safetensors`
    # *and* `model-*-of-*.safetensors` plus an index.  Pinning both would make the fetch
    # job download the model twice, so when an index exists only the shards it names are
    # pinned -- which also guarantees the registry matches what the loader will read.
    index_name = "model.safetensors.index.json"
    wanted_shards: set[str] | None = None
    if any(entry["rfilename"] == index_name for entry in entries):
        if dry_run:
            wanted_shards = {entry["rfilename"] for entry in entries
                             if entry["rfilename"].endswith(".safetensors")
                             and "consolidated" not in entry["rfilename"]}
        else:
            target = directory / index_name
            if not target.exists():
                size = next(e.get("size") for e in entries if e["rfilename"] == index_name)
                fetch(f"https://huggingface.co/{repo}/resolve/main/{index_name}", target, size)
            weight_map = json.loads(target.read_text(encoding="utf-8"))["weight_map"]
            wanted_shards = set(weight_map.values())

    for entry in entries:
        name = entry["rfilename"]
        size = entry.get("size")
        target = directory / name
        if name.endswith(".safetensors"):
            if wanted_shards is not None and name not in wanted_shards:
                print(f"    skipping {name} (not named by {index_name})")
                continue
            if target.exists() and size and target.stat().st_size == size:
                shards[name] = sha256_file(target)
            else:
                digest = (entry.get("lfs") or {}).get("sha256")
                if not digest:
                    raise SystemExit(f"{repo}: {name} is not LFS and is not on disk, so "
                                     f"there is no digest to pin; download it first")
                shards[name] = digest
                from_hub += 1
            continue
        if dry_run:
            files[name] = "<dry-run>"
            continue
        fetch(f"https://huggingface.co/{repo}/resolve/main/{name}", target, size)
        if size and target.stat().st_size != size:
            raise SystemExit(f"{repo}: {name} downloaded {target.stat().st_size} B, "
                             f"expected {size}")
        files[name] = sha256_file(target)
    if from_hub:
        print(f"    {from_hub} shard digest(s) taken from the Hub's LFS metadata "
              f"(not present locally); --verify-hashes re-checks them in the job")
    config = directory / "config.json"
    architectures = []
    if config.exists():
        architectures = json.loads(config.read_text(encoding="utf-8")).get("architectures", [])
    return {
        "path": f"models/{directory.name}",
        "dtype": "float32",
        "repo": repo,
        "source": f"https://huggingface.co/{repo}",
        "architectures": architectures,
        "chat_template": chat_template,
        "notes": notes,
        "files": dict(sorted(files.items())),
        "shards": dict(sorted(shards.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key", required=True, help="registry key, e.g. qwen3-4b")
    parser.add_argument("--repo", required=True, help="HuggingFace repo, e.g. Qwen/Qwen3-4B")
    parser.add_argument("--dir", default=None,
                        help="model directory (default: models/<repo basename>)")
    parser.add_argument("--notes", default="", help="the registry's notes field")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="the checkpoint has no chat template of its own")
    parser.add_argument("--dry-run", action="store_true",
                        help="download nothing and write nothing; print the pins")
    args = parser.parse_args(argv)

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    directory = Path(args.dir) if args.dir else REPO_ROOT / "models" / args.repo.split("/")[-1]
    print(f"{args.key}: {args.repo} -> {directory}")
    entry = build_entry(args.repo, directory, notes=args.notes,
                        chat_template=not args.no_chat_template, dry_run=args.dry_run)

    if args.dry_run:
        print(json.dumps(entry, indent=2, ensure_ascii=False)[:2000])
        return 0
    if args.key in registry["models"]:
        print(f"    updating the existing entry (was {registry['models'][args.key]['path']})")
    registry["models"][args.key] = entry
    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"    pinned {len(entry['files'])} file(s) and {len(entry['shards'])} shard(s) "
          f"-> {REGISTRY.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
