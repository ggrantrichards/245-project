#!/bin/bash
# Experiment runner for NLP 245 Project
# Run multiple compression ratios to measure safety loss

set -e

echo "======================================"
echo "NLP 245 Safety-Preserving Compression"
echo "Multi-Ratio Experiment Runner"
echo "======================================"
echo ""

# Check if authenticated
echo "Checking Hugging Face authentication..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "❌ ERROR: Not logged into Hugging Face!"
    echo "Please run: huggingface-cli login"
    exit 1
fi

echo "✅ Authenticated"
echo ""

# Create results directory
mkdir -p results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Running baseline (no compression)..."
echo "------------------------------------"
python3 pipeline.py --compression 0.0 --save | tee "results/baseline_${TIMESTAMP}.txt"

echo ""
echo "Running compression experiments..."
echo "------------------------------------"

for ratio in 0.3 0.5 0.7; do
    echo ""
    echo "==> Compression Ratio: ${ratio}"
    python3 pipeline.py --compression ${ratio} --save | tee "results/compression_${ratio}_${TIMESTAMP}.txt"
done

echo ""
echo "======================================"
echo "Experiments Complete!"
echo "Results saved in: results/"
echo "======================================"
