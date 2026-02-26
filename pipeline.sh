#!/bin/bash
# run_sweep_updated.sh
# Generates data for Pareto curves ONLY for the updated adaptive strategies

set -e

LIMIT=200 # 100 Safe, 100 Unsafe
RATIOS=(0.1 0.3 0.5 0.7 0.9)
# TARGETED SWEEP: Only run the strategies impacted by the adaptive threshold update
STRATEGIES=("safety_aware" "hybrid")

echo "======================================================"
echo "NLP 245 Targeted Sweep: Updated Adaptive Strategies"
echo "Model: Llama-3.2-1B | Sample Size: $LIMIT"
echo "======================================================"
echo ""

# Check Hugging Face Auth
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "❌ ERROR: Not logged into Hugging Face!"
    exit 1
fi

mkdir -p results/sweep_logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Skipping Baseline, Entropy, and Sentence strategies (already computed)."
echo "Running updated 'safety_aware' and 'hybrid' strategies..."

# Sweep across targeted parameters
for strategy in "${STRATEGIES[@]}"; do
    echo "------------------------------------------------------"
    echo "Testing Strategy: [ $strategy ]"
    
    for ratio in "${RATIOS[@]}"; do
        echo "  -> Running Compression Ratio: ${ratio}"
        
        # Run pipeline and save JSON to results/ while logging stdout
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
echo "Targeted Sweep Complete! All JSON results saved to: results/"
echo "Run your python plotting script to see the new Pareto curves!"
echo "======================================================"