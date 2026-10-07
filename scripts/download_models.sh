#!/usr/bin/env bash
# Fetch the two models used in this reproduction and verify their checksums.
#
#   ./scripts/download_models.sh            # into ./models
#   MODELS_DIR=/data/models ./scripts/download_models.sh
#
# Notes
#   * Qwen3.5-0.8B ships a *sharded-looking* index whose single shard is named
#     `model.safetensors-00001-of-00001.safetensors`; there is no
#     `model.safetensors`.  Downloading the obvious filename yields a 15-byte
#     "Entry not found" body, so we read the real name out of the index.
#   * `set -euo pipefail` plus explicit sha256 checks keep a truncated download
#     from silently poisoning a multi-hour experiment.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-models}"
mkdir -p "$MODELS_DIR"

fetch() { # fetch <repo> <file> <destdir>
  local repo="$1" file="$2" dest="$3"
  mkdir -p "$dest"
  echo "  -> $file"
  curl -fsSL --retry 3 --retry-delay 2 -o "$dest/$file" \
    "https://huggingface.co/${repo}/resolve/main/${file}"
}

check() { # check <path> <expected-sha256>
  local path="$1" want="$2"
  local got
  got="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$got" != "$want" ]]; then
    echo "CHECKSUM MISMATCH for $path" >&2
    echo "  expected $want" >&2
    echo "  got      $got" >&2
    return 1
  fi
  echo "  sha256 ok  $path"
}

echo "== Qwen3.5-0.8B =="
Q35="$MODELS_DIR/Qwen3.5-0.8B"
for f in config.json tokenizer.json tokenizer_config.json merges.txt vocab.json \
         chat_template.jinja preprocessor_config.json video_preprocessor_config.json \
         model.safetensors.index.json; do
  fetch Qwen/Qwen3.5-0.8B "$f" "$Q35"
done
shard="$(python3 -c "import json,sys; print(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))[0])" \
  "$Q35/model.safetensors.index.json")"
echo "  shard name from index: $shard"
fetch Qwen/Qwen3.5-0.8B "$shard" "$Q35"
check "$Q35/$shard" "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"

echo "== Qwen3-0.6B =="
Q3="$MODELS_DIR/Qwen3-0.6B"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         merges.txt vocab.json model.safetensors; do
  fetch Qwen/Qwen3-0.6B "$f" "$Q3"
done
check "$Q3/model.safetensors" "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"

echo
echo "done. models are in $MODELS_DIR"
