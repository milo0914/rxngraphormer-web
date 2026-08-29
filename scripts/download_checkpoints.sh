#!/usr/bin/env bash
# US-004: Download & verify fine-tuned RXNGraphormer checkpoints from figshare.
# Resume-safe (curl -C -) and MD5-verified against the figshare manifest.
# Run from the repo root. Does NOT commit anything (models/ and *.7z are gitignored).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARTICLE=30498368
API="https://api.figshare.com/v2/articles/${ARTICLE}"

mkdir -p models data

echo "== Fetching figshare manifest =="
curl -sS "$API" -o /tmp/fig.json
python3 - <<'PY'
import json
d=json.load(open('/tmp/fig.json'))
for f in d['files']:
    print(f['name'], f['size'], f['download_url'], f.get('computed_md5'))
PY

# name -> figshare file id (download_url uses the numeric id)
declare -A ID=(
  [seq-v2-USPTO_STEREO-20250423_044122_ft.7z]=59201303
  [seq-v2-USPTO_STEREO-20250509_070206_ft.7z]=59201306
  [Test_Dataset.7z]=59201294
)
# name -> expected md5
declare -A MD5=(
  [seq-v2-USPTO_STEREO-20250423_044122_ft.7z]=163e5a48e4f845eec5e5afd07b00026d
  [seq-v2-USPTO_STEREO-20250509_070206_ft.7z]=52d506a2ecee0c77cad7de03c692f653
  [Test_Dataset.7z]=1d5fcebb2ae10e657180eed52188c9d9
)

for name in "${!ID[@]}"; do
  url="https://ndownloader.figshare.com/files/${ID[$name]}"
  if [[ "$name" == Test_Dataset.7z ]]; then
    out="data/$name"
  else
    out="models/$name"
  fi
  echo "== Downloading $name (resume) =="
  curl -L -C - -o "$out" "$url"
  got=$(md5sum "$out" | awk '{print $1}')
  if [[ "$got" == "${MD5[$name]}" ]]; then
    echo "MD5 OK  $name  ($got)"
  else
    echo "MD5 MISMATCH $name got=$got expected=${MD5[$name]}" >&2
    exit 1
  fi
done

echo "== Extracting =="
python3 - <<'PY'
import py7zr
py7zr.unpack_7zarchive('models/seq-v2-USPTO_STEREO-20250423_044122_ft.7z','models/')
py7zr.unpack_7zarchive('models/seq-v2-USPTO_STEREO-20250509_070206_ft.7z','models/')
py7zr.unpack_7zarchive('data/Test_Dataset.7z','data/')
print('extraction done')
PY
echo "== Done. models/ and *.7z are gitignored and NOT committed. =="
