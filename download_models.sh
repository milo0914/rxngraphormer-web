#!/usr/bin/env bash
# download_models.sh — Fetch RXNGraphormer checkpoints from GitHub Releases (primary) with figshare fallback.
# Verifies MD5, extracts archives, writes manifest.
#
# Usage:
#   bash download_models.sh              # Download all missing
#   bash download_models.sh --forward-only
#   bash download_models.sh --retro-only
#   bash download_models.sh --verify     # Check existing only
#   bash download_models.sh --force      # Re-download even if present
#
# Environment variables:
#   GITHUB_TOKEN        - Optional GitHub token for private repos / rate limits
#   GITHUB_RELEASES_REPO - GitHub repo (default: milo0914/rxngraphormer-web)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODELS_DIR="$REPO_ROOT/models"
MANIFEST_PATH="$MODELS_DIR/.manifest.json"

# Release configuration
GITHUB_REPO="${GITHUB_RELEASES_REPO:-milo0914/rxngraphormer-web}"
RELEASE_TAG="v1.0-weights"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# Checkpoint definitions
# Format: name|dir_name|asset_name|figshare_file_id|figshare_md5|figshare_size|archive_type|required_files
FORWARD_DEF="forward|seq-v2-USPTO_STEREO-20250509_070206_ft|forward-stereo-ft.7z|59201306|52d506a2ecee0c77cad7de03c692f653|391622627|7z|parameters.json,model/valid_checkpoint.pt"
RETRO_DEF="retro|USPTO_50k|retro-uspto50k.7z|53998184|1d993b40b8ff38def31788c1ced69de5|197722944|7z|parameters.json,model/valid_checkpoint.pt"

FORWARD_ONLY=false
RETRO_ONLY=false
VERIFY_ONLY=false
FORCE=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --forward-only) FORWARD_ONLY=true ;;
        --retro-only) RETRO_ONLY=true ;;
        --verify) VERIFY_ONLY=true ;;
        --force) FORCE=true ;;
        -h|--help)
            echo "Usage: $0 [--forward-only|--retro-only|--verify|--force]"
            exit 0
            ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

mkdir -p "$MODELS_DIR"

# Helper: compute MD5
compute_md5() {
    md5sum "$1" | awk '{print $1}'
}

# Helper: download with resume
download_file() {
    local url="$1"
    local dest="$2"
    local expected_size="${3:-}"
    local headers=()

    if [[ -n "$GITHUB_TOKEN" && "$url" == https://github.com/* ]]; then
        headers+=(-H "Authorization: token $GITHUB_TOKEN")
    fi

    local mode="-O"
    local resume_pos=0
    if [[ -f "$dest" ]]; then
        resume_pos=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null)
        if [[ $resume_pos -gt 0 ]]; then
            headers+=(-H "Range: bytes=$resume_pos-")
        fi
    fi

    if curl -L -C - "${headers[@]}" -o "$dest" "$url"; then
        return 0
    else
        return 1
    fi
}

# Helper: verify MD5
verify_md5() {
    local file="$1"
    local expected="$2"
    local actual
    actual=$(compute_md5 "$file")
    if [[ "${actual,,}" == "${expected,,}" ]]; then
        return 0
    else
        echo "  ✗ MD5 mismatch: expected $expected, got $actual" >&2
        return 1
    fi
}

# Helper: extract archive
extract_archive() {
    local archive="$1"
    local dest_dir="$2"
    local type="$3"

    mkdir -p "$dest_dir"
    case "$type" in
        7z)
            if ! command -v 7z &>/dev/null && ! command -v 7zz &>/dev/null; then
                echo "  ✗ 7z not installed" >&2
                return 1
            fi
            (cd "$dest_dir" && 7z x "../$(basename "$archive")" -y >/dev/null) || \
            (cd "$dest_dir" && 7zz x "../$(basename "$archive")" -y >/dev/null)
            ;;
        zip)
            unzip -q -o "$archive" -d "$dest_dir" || return 1
            ;;
        *) echo "Unknown archive type: $type" >&2; return 1 ;;
    esac
}

# Helper: load manifest
load_manifest() {
    if [[ -f "$MANIFEST_PATH" ]]; then
        cat "$MANIFEST_PATH"
    else
        echo "{}"
    fi
}

# Helper: save manifest
save_manifest() {
    local manifest="$1"
    echo "$manifest" > "$MANIFEST_PATH.tmp"
    mv "$MANIFEST_PATH.tmp" "$MANIFEST_PATH"
}

# Helper: check if model exists
check_model_exists() {
    local def="$1"
    local dir_name
    dir_name=$(echo "$def" | cut -d'|' -f2)
    local model_dir="$MODELS_DIR/$dir_name"
    local required_files
    required_files=$(echo "$def" | cut -d'|' -f8)

    [[ -d "$model_dir" ]] || return 1

    IFS=',' read -ra files <<< "$required_files"
    for req in "${files[@]}"; do
        [[ -f "$model_dir/$req" ]] || return 1
    done
    return 0
}

# Helper: download and extract a checkpoint
download_checkpoint() {
    local def="$1"
    local force="$2"

    local name dir_name asset_name figshare_id figshare_md5 figshare_size archive_type required_files
    IFS='|' read -r name dir_name asset_name figshare_id figshare_md5 figshare_size archive_type required_files <<< "$def"

    local model_dir="$MODELS_DIR/$dir_name"
    local archive_name="${dir_name}.${archive_type}"
    local archive_path="$MODELS_DIR/$archive_name"

    if check_model_exists "$def" && [[ "$force" != true ]]; then
        echo "✓ $name: already present at $model_dir"
        return 0
    fi

    echo "⬇ $name: downloading..."

    # Try GitHub Releases first
    local github_url="https://github.com/$GITHUB_REPO/releases/download/$RELEASE_TAG/$asset_name"
    local success=false
    local source="github"

    echo "  → Trying GitHub Releases: $github_url"
    if download_file "$github_url" "$archive_path"; then
        if [[ -f "$archive_path" && $(stat -c%s "$archive_path" 2>/dev/null || stat -f%z "$archive_path" 2>/dev/null) -gt 1048576 ]]; then
            success=true
        fi
    fi

    # Fallback to figshare
    if [[ "$success" != true ]]; then
        local figshare_url="https://ndownloader.figshare.com/files/$figshare_id"
        echo "  → GitHub failed/unavailable, trying figshare: $figshare_url"
        rm -f "$archive_path"
        if download_file "$figshare_url" "$archive_path"; then
            if verify_md5 "$archive_path" "$figshare_md5"; then
                success=true
                source="figshare"
            else
                rm -f "$archive_path"
            fi
        fi
    fi

    if [[ "$success" != true ]]; then
        echo "  ✗ $name: all download sources failed" >&2
        return 1
    fi

    # Extract
    echo "  → Extracting to $model_dir..."
    rm -rf "$model_dir"
    if ! extract_archive "$archive_path" "$model_dir" "$archive_type"; then
        return 1
    fi

    # Verify extracted files
    if ! check_model_exists "$def"; then
        echo "  ✗ Extraction incomplete: required files missing" >&2
        return 1
    fi

    # Update manifest
    local manifest
    manifest=$(load_manifest)
    local archive_md5
    archive_md5=$(compute_md5 "$archive_path")
    local extracted_at
    extracted_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local archive_size
    archive_size=$(stat -c%s "$archive_path" 2>/dev/null || stat -f%z "$archive_path" 2>/dev/null)

    # Use jq if available, otherwise Python
    if command -v jq &>/dev/null; then
        manifest=$(echo "$manifest" | jq --arg name "$name" \
            --arg dir_name "$dir_name" \
            --arg source "$source" \
            --arg archive_md5 "$archive_md5" \
            --arg extracted_at "$extracted_at" \
            --argjson archive_size "$archive_size" \
            '.[$name] = {"dir_name": $dir_name, "source": $source, "archive_md5": $archive_md5, "extracted_at": $extracted_at, "archive_size": $archive_size}')
    else
        manifest=$(python3 -c "
import json, sys
manifest = json.loads(sys.argv[1])
manifest[sys.argv[2]] = {
    'dir_name': sys.argv[3],
    'source': sys.argv[4],
    'archive_md5': sys.argv[5],
    'extracted_at': sys.argv[6],
    'archive_size': int(sys.argv[7])
}
print(json.dumps(manifest, indent=2))
" "$manifest" "$name" "$dir_name" "$source" "$archive_md5" "$extracted_at" "$archive_size")
    fi
    save_manifest "$manifest"

    # Clean up archive (keep manifest)
    rm -f "$archive_path"

    echo "  ✓ $name: ready at $model_dir"
    return 0
}

# Helper: verify all checkpoints
verify_all() {
    local manifest
    manifest=$(load_manifest)
    local all_ok=true

    for def in "$FORWARD_DEF" "$RETRO_DEF"; do
        local name dir_name required_files
        IFS='|' read -r name dir_name _ _ _ _ _ required_files <<< "$def"
        local model_dir="$MODELS_DIR/$dir_name"

        if [[ ! -d "$model_dir" ]]; then
            echo "✗ $name: missing directory $model_dir"
            all_ok=false
            continue
        fi

        local missing=()
        IFS=',' read -ra files <<< "$required_files"
        for req in "${files[@]}"; do
            [[ -f "$model_dir/$req" ]] || missing+=("$req")
        done

        if [[ ${#missing[@]} -gt 0 ]]; then
            echo "✗ $name: missing files: ${missing[*]}"
            all_ok=false
        else
            local recorded source
            if command -v jq &>/dev/null; then
                source=$(echo "$manifest" | jq -r --arg name "$name" '.[$name].source // "unknown"')
            else
                source=$(python3 -c "
import json, sys
m = json.loads(sys.argv[1])
print(m.get(sys.argv[2], {}).get('source', 'unknown'))
" "$manifest" "$name")
            fi
            echo "✓ $name: OK (source: $source)"
        fi
    done

    [[ "$all_ok" == true ]]
}

# Main
if [[ "$VERIFY_ONLY" == true ]]; then
    echo "Verifying models..."
    if verify_all; then
        exit 0
    else
        exit 1
    fi
fi

targets=()
if [[ "$FORWARD_ONLY" == true ]]; then
    targets=("$FORWARD_DEF")
elif [[ "$RETRO_ONLY" == true ]]; then
    targets=("$RETRO_DEF")
else
    targets=("$FORWARD_DEF" "$RETRO_DEF")
fi

all_ok=true
for def in "${targets[@]}"; do
    if ! download_checkpoint "$def" "$FORCE"; then
        all_ok=false
    fi
done

if [[ "$all_ok" == true ]]; then
    echo ""
    echo "✓ All requested models ready"
    exit 0
else
    echo ""
    echo "✗ Some models failed" >&2
    exit 1
fi