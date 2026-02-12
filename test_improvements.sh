#!/bin/bash
# Quick test script for improved pipeline

echo "=========================================="
echo "Testing Improved Pipeline"
echo "=========================================="
echo ""

echo "This script will run quick tests with the improved pipeline."
echo "Make sure you have:"
echo "  1. Installed dependencies (pip install -r requirements.txt)"
echo "  2. Authenticated with Hugging Face (huggingface-cli login)"
echo "  3. Access to Llama-Guard-3-8B"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

echo ""
echo "=========================================="
echo "Test 1: Improved Pipeline (Hybrid Strategy)"
echo "=========================================="
python3 pipeline_improved.py --strategy hybrid --compression 0.5 --limit 10 --verbose

echo ""
echo "=========================================="
echo "Test 2: Compare All Strategies"
echo "=========================================="
echo "This will run all strategies and compare results..."
echo ""
python3 compare_pipelines.py --compression 0.5 --limit 10

echo ""
echo "=========================================="
echo "Test 3: Safety-Aware Strategy"
echo "=========================================="
python3 pipeline_improved.py --strategy safety_aware --compression 0.5 --limit 10 --verbose --save

echo ""
echo "=========================================="
echo "Tests Complete!"
echo "=========================================="
echo ""
echo "Check the results/ directory for detailed JSON outputs."
echo ""
echo "To run individual tests:"
echo "  python3 pipeline_improved.py --strategy [hybrid|safety_aware|sentence] --compression 0.5 --verbose"
echo ""
echo "To compare with original:"
echo "  python3 compare_pipelines.py --compression 0.5 --limit 10"
echo ""
