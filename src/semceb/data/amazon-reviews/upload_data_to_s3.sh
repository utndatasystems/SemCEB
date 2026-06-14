#!/usr/bin/env bash
set -Eeuo pipefail

# Upload select Amazon Reviews dataset artifacts to S3
# Prerequisites:
# - AWS CLI installed and configured (credentials + default region)
# - zip installed
#
# Target: s3://azimmerer-semceb-datasets/amazon-reviews/

# --- Configuration (adjust only if your layout changes) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/processed/Arts_Crafts_and_Sewing__raw_5core"
IMAGES_SRC_DIR="$DATASET_DIR/images/m.media-amazon.com/images/I"
IMAGES_ZIP="$DATASET_DIR/images.zip"
S3_URI_PREFIX="s3://azimmerer-semceb-datasets/amazon-reviews"
# ---------------------------------------------------------

log() { echo "[upload] $*"; }
fail() { echo "[error] $*" >&2; exit 1; }

# Check dependencies
for cmd in aws zip; do
  command -v "$cmd" >/dev/null 2>&1 || fail "Required command '$cmd' not found in PATH"
done

# Sanity checks
[[ -d "$DATASET_DIR" ]] || fail "Dataset directory not found: $DATASET_DIR"
[[ -d "$IMAGES_SRC_DIR" ]] || fail "Images source directory not found: $IMAGES_SRC_DIR"

# Verify AWS credentials before performing any zipping or uploads
log "Verifying AWS credentials via STS..."
aws sts get-caller-identity --no-cli-pager >/dev/null 2>&1 \
  || fail "AWS CLI is not authenticated or lacks STS permissions"
log "AWS credentials look valid. Proceeding."

# Create images.zip from the contents of the images folder (not the folder itself)
log "Creating zip archive from $IMAGES_SRC_DIR"
rm -f "$IMAGES_ZIP"
(
  cd "$IMAGES_SRC_DIR"
  # Zip all contents inside IMAGES_SRC_DIR into IMAGES_ZIP (relative roots) 
  zip -r -q "$IMAGES_ZIP" .
)
[[ -f "$IMAGES_ZIP" ]] || fail "Failed to create images archive: $IMAGES_ZIP"
log "Created archive: $IMAGES_ZIP"

# Zip the embedding cache before upload
EMBEDDING_DB="$DATASET_DIR/embedding_cache.sqlite3"
EMBEDDING_ZIP="$DATASET_DIR/embedding_cache.sqlite3.zip"
[[ -f "$EMBEDDING_DB" ]] || fail "Embedding DB not found: $EMBEDDING_DB"
log "Creating zip archive of $(basename "$EMBEDDING_DB")"
rm -f "$EMBEDDING_ZIP"
zip -q -j "$EMBEDDING_ZIP" "$EMBEDDING_DB"
[[ -f "$EMBEDDING_ZIP" ]] || fail "Failed to create embedding archive: $EMBEDDING_ZIP"
log "Created archive: $EMBEDDING_ZIP"

# Files to upload
FILES_TO_UPLOAD=(
  "$IMAGES_ZIP"
  "$EMBEDDING_ZIP"
  "$DATASET_DIR/products_filtered.parquet"
  "$DATASET_DIR/reviews_filtered.parquet"
)

# Validate files exist before attempting upload
for f in "${FILES_TO_UPLOAD[@]}"; do
  [[ -f "$f" ]] || fail "Missing required file: $f"
  log "Will upload: $f"
done

# Optional: verify AWS identity (uncomment to enforce before upload)
# aws sts get-caller-identity >/dev/null || fail "AWS CLI is not authenticated"

# Upload files
for f in "${FILES_TO_UPLOAD[@]}"; do
  bn="$(basename "$f")"
  log "Uploading $bn to $S3_URI_PREFIX/"
  aws s3 cp "$f" "$S3_URI_PREFIX/" --only-show-errors
  log "Uploaded: $bn"
done

log "All uploads completed successfully."