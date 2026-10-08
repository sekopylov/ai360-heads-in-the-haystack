#!/usr/bin/env bash
# Fetch the two models used in this reproduction and verify **every** file.
#
#   ./scripts/download_models.sh            # into ./models
#   MODELS_DIR=/data/models ./scripts/download_models.sh
#
# Notes
#   * `configs/models.json` is the single source of truth: it lists the HF repo,
#     every file name and its sha256.  Nothing is hard-coded here, so the registry
#     and the downloader cannot drift apart.
#   * Every file is checked, not just the weights.  `tokenizer.json`,
#     `tokenizer_config.json` and `chat_template.jinja` decide how the needle and
#     the prompt are rendered, so a silent change in any of them moves every
#     number in the results -- exactly the kind of drift the checksums exist to
#     catch.  The files are still fetched from `/resolve/main/`; the pinned hash
#     is what turns an upstream change into a loud failure instead of a quiet
#     difference.
#   * Downloads are atomic and resumable-by-hash: a file whose sha256 already
#     matches is skipped, and a fresh download lands in `<file>.part` and is only
#     moved into place after the checksum passes, so an interrupted run cannot
#     leave a truncated checkpoint behind.
#   * Qwen3.5-0.8B ships a sharded-looking index whose single shard is named
#     `model.safetensors-00001-of-00001.safetensors`; there is no
#     `model.safetensors`.  The shard names live in the registry, so this script
#     never has to guess them from the index.
#   * A failure while reading the registry aborts the script: an empty file list
#     must not be mistaken for "nothing to do".
set -euo pipefail

cd "$(dirname "$0")/.."
MODELS_DIR="${MODELS_DIR:-models}"
REGISTRY="${REGISTRY:-configs/models.json}"

mkdir -p "$MODELS_DIR"

model_keys() { # every model key in the registry, so nothing is hard-coded here
  python3 - "$REGISTRY" <<'PY'
import json
import sys

for key in json.load(open(sys.argv[1], encoding="utf-8"))["models"]:
    print(key)
PY
}

sha256() { # sha256 <path>; GNU coreutils or the BSD/macOS shasum
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "no sha256 tool found (need sha256sum or shasum)" >&2
    exit 1
  fi
}

check() { # check <path> <expected-sha256>
  local path="$1" want="$2" got
  got="$(sha256 "$path")"
  if [[ "$got" != "$want" ]]; then
    echo "CHECKSUM MISMATCH for $path" >&2
    echo "  expected $want" >&2
    echo "  got      $got" >&2
    return 1
  fi
}

fetch_verified() { # fetch_verified <repo> <file> <destdir> <sha256>
  local repo="$1" file="$2" dest="$3" want="$4"
  mkdir -p "$dest"
  if [[ -f "$dest/$file" ]] && [[ "$(sha256 "$dest/$file")" == "$want" ]]; then
    echo "  cached  $file"
    return 0
  fi
  echo "  -> $file"
  curl -fsSL --retry 3 --retry-delay 2 -o "$dest/$file.part" \
    "https://huggingface.co/${repo}/resolve/main/${file}"
  check "$dest/$file.part" "$want"
  mv "$dest/$file.part" "$dest/$file"
  echo "  sha256 ok  $file"
}

manifest() { # manifest <key> -> "<repo>\t<file>\t<sha256>" lines
  python3 - "$REGISTRY" "$1" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
try:
    entry = json.load(open(path, encoding="utf-8"))["models"][key]
except Exception as exc:  # noqa: BLE001 - the message matters more than the type
    sys.exit(f"cannot read {key!r} from {path}: {exc}")
files = dict(entry.get("files") or {})
files.update(entry.get("shards") or {})
repo = entry.get("repo")
if not repo or not files:
    sys.exit(f"{key!r} needs 'repo' and a non-empty files/shards map in {path}")
for name, sha in sorted(files.items()):
    if not sha:
        sys.exit(f"{key!r}: {name} has no pinned sha256 in {path}")
    print(f"{repo}\t{name}\t{sha}")
PY
}

dest_dir() { # dest_dir <key> -> the leaf directory name from the registry
  python3 - "$REGISTRY" "$1" <<'PY'
import json
import os
import sys

entry = json.load(open(sys.argv[1], encoding="utf-8"))["models"][sys.argv[2]]
print(os.path.basename(entry["path"]))
PY
}

if ! keys="$(model_keys)" || [[ -z "$keys" ]]; then
  echo "ERROR: no models listed in $REGISTRY" >&2
  exit 1
fi
while IFS= read -r key; do
  [[ -n "$key" ]] || continue
  echo "== $key =="
  dest="$MODELS_DIR/$(dest_dir "$key")"
  if ! lines="$(manifest "$key")" || [[ -z "$lines" ]]; then
    echo "ERROR: no files listed for $key in $REGISTRY" >&2
    exit 1
  fi
  while IFS=$'\t' read -r repo file sha; do
    [[ -n "$file" && -n "$sha" ]] || continue
    fetch_verified "$repo" "$file" "$dest" "$sha"
  done <<< "$lines"
done <<< "$keys"

echo
echo "done. models are in $MODELS_DIR"
