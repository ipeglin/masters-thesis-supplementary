#!/bin/bash
BIDS_DIR="/Users/ipeglin/Documents/masters_thesis/bids_processed_consolidated_data"

cd $BIDS_DIR || { echo "Failed to change directory to $BIDS_DIR"; exit 1; }

for f in sub-*/*.h5; do
  echo "repacking $f"
  h5repack "$f" "$f.tmp" && mv "$f.tmp" "$f" || { echo "FAILED: $f"; rm -f "$f.tmp"; }
done