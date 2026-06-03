#!/usr/bin/env bash
# scripts/train_cfm_3stage.sh — Three-stage CFM training orchestrator (Req 2).
# Usage: train_cfm_3stage.sh --manifest <path> --out-dir <dir> [...]
set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 --manifest <path> --out-dir <dir>
          [--epochs-cfm N] [--epochs-mdp N] [--epochs-depth N]
          [--batch-size N] [--lr F] [--use-gpu true|false]
          [--attention real_scaled_dot|legacy_gated]
          [--mdp-backbone resnet18|mobilenet_v3_small|lightweight_cnn]
          [--mdp-backbone-weights <path>]
          [--seed N]
EOF
    exit 1
}

# --- arg parsing (validation BEFORE mkdir; Req 2.2) ---
MANIFEST=""; OUT_DIR=""
E_CFM=10; E_MDP=10; E_DEPTH=10
BATCH_SIZE=4; LR=0.0001; USE_GPU=true
ATTN=real_scaled_dot
BACKBONE=resnet18
BACKBONE_W=models/pretrained/resnet18_in1k.pt
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --epochs-cfm)   E_CFM="$2";   shift 2 ;;
        --epochs-mdp)   E_MDP="$2";   shift 2 ;;
        --epochs-depth) E_DEPTH="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --use-gpu)      USE_GPU="$2"; shift 2 ;;
        --attention)    ATTN="$2"; shift 2 ;;
        --mdp-backbone) BACKBONE="$2"; shift 2 ;;
        --mdp-backbone-weights) BACKBONE_W="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done
[[ -z "$MANIFEST" || -z "$OUT_DIR" ]] && usage

# Integer-range validation on optional epochs (Req 2.1).
for E in "$E_CFM" "$E_MDP" "$E_DEPTH"; do
    [[ "$E" =~ ^[0-9]+$ ]] || { echo "epoch must be integer: $E" >&2; usage; }
    (( E >= 1 && E <= 1000 )) || { echo "epoch out of range [1,1000]: $E" >&2; usage; }
done

# Req 2.4: mkdir -p, no deletion of existing checkpoints.
mkdir -p "$OUT_DIR"
CFM_S1="$OUT_DIR/cfm_s1.pt"
CFM_S2="$OUT_DIR/cfm_s2.pt"
CFM_FINAL="$OUT_DIR/cfm.pt"
LOG1="$OUT_DIR/train_s1.log"
LOG2="$OUT_DIR/train_s2.log"
LOG3="$OUT_DIR/train_s3.log"

CMS=./build/bin/cms
COMMON=( --manifest "$MANIFEST" --batch-size "$BATCH_SIZE" --lr "$LR"
         --use-gpu "$USE_GPU" --seed "$SEED"
         --attention "$ATTN" --mdp-backbone "$BACKBONE"
         --mdp-backbone-weights "$BACKBONE_W" )

run_stage() {
    local stage="$1"; local epochs="$2"; local out="$3"; local log="$4"
    local init_from="${5:-}"
    local t0; t0=$(date +%s.%N)
    local cmd=( "$CMS" cfm-train --stage "$stage" --epochs "$epochs"
                --output "$out" "${COMMON[@]}" )
    [[ -n "$init_from" ]] && cmd+=( --init-from "$init_from" )

    # Req 2.14: per-stage tee'd log; Req 2.6: propagate non-zero exit.
    "${cmd[@]}" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "[3stage] stage $stage FAILED rc=$rc log=$(realpath "$log")" >&2
        exit "$rc"
    fi
    local t1; t1=$(date +%s.%N)
    printf "[3stage] stage %s took %.1fs\n" "$stage" \
        "$(echo "$t1 - $t0" | bc -l)"
}

run_stage cfm   "$E_CFM"   "$CFM_S1"    "$LOG1"
run_stage mdp   "$E_MDP"   "$CFM_S2"    "$LOG2" "$CFM_S1"
run_stage depth "$E_DEPTH" "$CFM_FINAL" "$LOG3" "$CFM_S2"

# Req 2.14: final absolute-path summary.
echo "[3stage] checkpoints:"
echo "  s1: $(realpath "$CFM_S1")"
echo "  s2: $(realpath "$CFM_S2")"
echo "  final: $(realpath "$CFM_FINAL")"
