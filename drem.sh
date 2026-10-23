#!/bin/bash
################################################################################
# DREM: Deep Resistance gene Evaluator Model
# BiLSTM-based scoring and prioritization of plant resistance proteins
#
# Usage: bash drem.sh -f proteins.fasta [-o output_dir]
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"

FASTA_FILE=""
OUTPUT_DIR="./drem_output"

print_usage() {
    cat << EOF
DREM - Deep Resistance gene Evaluator Model

Usage:
  bash drem.sh -f proteins.fasta [options]

Required:
  -f, --fasta FILE     Input protein FASTA file (.fasta or .fasta.gz)

Optional:
  -o, --output DIR     Output directory (default: ./drem_output)
  -h, --help           Show this help message

Output:
  DIR/drem_results.tsv          one row per protein
  DIR/drem_results_windows.tsv  one row per sequence window
EOF
}

log_info()  { echo "[DREM] $1" >&2; }
log_error() { echo "[ERROR] $1" >&2; }

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--fasta)  FASTA_FILE="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)   print_usage; exit 0 ;;
        *) log_error "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

if [[ -z "$FASTA_FILE" ]]; then
    log_error "Input FASTA file required (-f)"
    print_usage
    exit 1
fi

if [[ ! -f "$FASTA_FILE" ]]; then
    log_error "FASTA file not found: $FASTA_FILE"
    exit 1
fi

for model in resistance_model.pt domain_model.pt; do
    if [[ ! -f "$MODEL_DIR/$model" ]]; then
        log_error "Model file not found: $MODEL_DIR/$model"
        exit 1
    fi
done

if ! command -v python3 &> /dev/null; then
    log_error "python3 not found"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

log_info "Start: $(date)"
log_info "Input: $FASTA_FILE"
log_info "Output directory: $OUTPUT_DIR"

python3 "$SCRIPT_DIR/drem_inference.py" \
    -f "$FASTA_FILE" \
    -o "$OUTPUT_DIR/drem_results.tsv" \
    -m "$MODEL_DIR"

log_info "Results: $OUTPUT_DIR/drem_results.tsv"
log_info "End: $(date)"
