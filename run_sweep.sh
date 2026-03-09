#!/bin/bash
# run_sweep.sh
# Generates data for Pareto curves: Safety Loss (ASR) vs. Usability (BRR)

set -e

LIMIT=200 # 100 Safe, 100 Unsafe
RATIOS=(0.1 0.3 0.5 0.7 0.9)
STRATEGIES=("entropy" "sentence" "safety_aware" "hybrid" "abstractive" "two_stage")

echo "======================================================"
echo "NLP 245 Experiment Sweep: Generating Pareto Curve Data"
echo "Model: Llama-3.2-1B | Sample Size: $LIMIT"
echo "======================================================"
echo ""

# Check Hugging Face Auth
if ! huggingface-cli whoami > /dev/null 2>&1 && ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" > /dev/null 2>&1; then
    echo "❌ ERROR: Not logged into Hugging Face!"
    exit 1
fi

mkdir -p results/sweep_logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 1. Run Baseline (0% Compression)
echo "------------------------------------------------------"
echo "[0/5] Running Baseline (No Compression)..."
python3 pipeline_improved.py --strategy hybrid --compression 0.0 --limit $LIMIT --save > "results/sweep_logs/baseline_${TIMESTAMP}.log" 2>&1
echo "✓ Baseline complete."

# 2. Sweep across all parameters
for strategy in "${STRATEGIES[@]}"; do
    echo "------------------------------------------------------"
    echo "Testing Strategy: [ $strategy ]"
    
    for ratio in "${RATIOS[@]}"; do
        echo "  -> Running Compression Ratio: ${ratio}"
        
        # Note: We pipe stdout/stderr to a log, but rely on the python script's --save flag to output the clean JSON
        python3 pipeline_improved.py \
            --strategy $strategy \
            --compression $ratio \
            --limit $LIMIT \
            --save > "results/sweep_logs/run_${strategy}_${ratio}_${TIMESTAMP}.log" 2>&1
            
        echo "     ✓ Done (${ratio})"
    done
done

echo ""
echo "======================================================"
echo "Sweep Complete! All JSON results saved to: results/"
echo "Run a python script to parse the JSONs and plot your graphs!"
echo "======================================================"
