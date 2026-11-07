#!/bin/bash
# launch_all_seeds.sh — the Table 1 run of the Infrastructure Security Game
# ("Strategic Decision Focused Learning"): 4 models x 5 seeds = 20 background
# processes, 500 epochs each. This is the paper run's launch script
# (launch_paper.sh, run id paper_20260630_030550) with the non-paper models
# removed; every flag and MTA_EVAL_WORKERS are unchanged.
#
# Models:
#   pto      : predict-then-optimise baseline (Huber loss only)
#   zoc      : Huber blocks with accept-or-revert on the game utility
#   kkt_v3   : KKT LP layer, top_k 20, annealed Huber (5.0 → 3.0)   = "KKT" in the paper
#   hybrid_v3: Huber warm-up → KKT blocks, annealed Huber (5.0 → 3.0) = "Hybrid" in the paper
#
# Before running, activate a Python environment with torch, numpy, pandas,
# scipy, scikit-learn, cvxpy and cvxpylayers (`python` below is that
# environment's interpreter; this script does not activate one). The data file
# is expected at ../data/data_MTA_8H_Manhattan.csv (see train.py --data).
# Needs a many-core machine: each kkt_v3 / hybrid_v3 process uses 6 LP threads
# plus a pool of 6 evaluation workers (about 2.7 h per hybrid_v3 seed with all
# 20 processes running on a 32-core box; pto / zoc seeds take minutes).
#
# Usage:
#   bash launch_all_seeds.sh
#   logs:    logs/paper_<model>_s<seed>.log
#   results: results/<RUN_ID>/  (seed json files, summary.json, checkpoints/)

set -e
cd "$(dirname "$0")"
mkdir -p logs

RUN_ID=paper_$(date +%Y%m%d_%H%M%S)
echo "========================================"
echo "  MTA Paper Run ID: ${RUN_ID}"
echo "  4 models x 5 seeds = 20 processes"
echo "========================================"

for s in 1 2 3 4 5; do

    # PTO — pure NN prediction
    MTA_EVAL_WORKERS=6 nohup python -u train.py \
        --model pto --seed $s --epochs 500 \
        --run_id ${RUN_ID} \
        > logs/paper_pto_s${s}.log 2>&1 &

    # ZOC — zeroth-order strategic
    MTA_EVAL_WORKERS=6 nohup python -u train.py \
        --model zoc --seed $s --epochs 500 \
        --run_id ${RUN_ID} \
        > logs/paper_zoc_s${s}.log 2>&1 &

    # KKT V3 — annealed Huber (5.0 → 3.0)
    MTA_EVAL_WORKERS=6 nohup python -u train.py \
        --model kkt_v3 --seed $s --epochs 500 --top_k 20 \
        --lp_workers 6 --huber_weight_end 3.0 \
        --run_id ${RUN_ID} \
        > logs/paper_kkt_v3_s${s}.log 2>&1 &

    # Hybrid V3 — Huber warm-up + annealed Huber (5.0 → 3.0)
    MTA_EVAL_WORKERS=6 nohup python -u train.py \
        --model hybrid_v3 --seed $s --epochs 500 --top_k 20 \
        --lp_workers 6 --huber_weight_end 3.0 \
        --run_id ${RUN_ID} \
        > logs/paper_hybrid_v3_s${s}.log 2>&1 &

done

echo ""
echo "All 20 processes launched."
echo "Run ID: ${RUN_ID}"
echo ""
echo "Check progress:"
echo "  for m in pto zoc kkt_v3 hybrid_v3; do echo \"=== \$m ===\"; grep -a 'Epoch\|per_window' logs/paper_\${m}_s1.log 2>/dev/null | tail -1; done"
echo ""
echo "Check results when done:"
echo "  cat results/\${RUN_ID}/summary.json"
