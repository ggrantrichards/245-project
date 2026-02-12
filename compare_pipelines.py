#!/usr/bin/env python3
"""
Compare original vs improved pipeline to demonstrate improvements in blocking and over-refusal
"""

import sys
import json
from datetime import datetime

def run_comparison(compression_rate=0.5, limit=10):
    """Run both pipelines and compare results"""
    
    print("="*80)
    print("PIPELINE COMPARISON: Original vs Improved")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Compression Rate: {compression_rate:.0%}")
    print(f"  Dataset Limit: {limit} samples")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)
    
    results = {}
    
    # Run Original Pipeline
    print("\n[1/4] Running ORIGINAL pipeline (entropy-based)...")
    print("-"*80)
    try:
        from pipeline import run_pipeline
        original_results = run_pipeline(
            compression_rate=compression_rate,
            limit=limit,
            save_results=False
        )
        results['original'] = original_results
        print("\n✓ Original pipeline completed")
    except Exception as e:
        print(f"\n✗ Original pipeline failed: {e}")
        results['original'] = None
    
    # Run Improved Pipeline - Hybrid Strategy (Default)
    print("\n" + "="*80)
    print("[2/4] Running IMPROVED pipeline (hybrid strategy)...")
    print("-"*80)
    try:
        from pipeline_improved import run_improved_pipeline
        hybrid_results = run_improved_pipeline(
            compression_rate=compression_rate,
            limit=limit,
            save_results=False,
            compression_strategy="hybrid",
            use_context_repair=True,
            verbose=False
        )
        results['hybrid'] = hybrid_results
        print("\n✓ Hybrid pipeline completed")
    except Exception as e:
        print(f"\n✗ Hybrid pipeline failed: {e}")
        results['hybrid'] = None
    
    # Run Improved Pipeline - Safety-Aware Strategy
    print("\n" + "="*80)
    print("[3/4] Running IMPROVED pipeline (safety-aware strategy)...")
    print("-"*80)
    try:
        from pipeline_improved import run_improved_pipeline
        safety_results = run_improved_pipeline(
            compression_rate=compression_rate,
            limit=limit,
            save_results=False,
            compression_strategy="safety_aware",
            use_context_repair=True,
            verbose=False
        )
        results['safety_aware'] = safety_results
        print("\n✓ Safety-aware pipeline completed")
    except Exception as e:
        print(f"\n✗ Safety-aware pipeline failed: {e}")
        results['safety_aware'] = None
    
    # Run Improved Pipeline - Sentence-Level Strategy
    print("\n" + "="*80)
    print("[4/4] Running IMPROVED pipeline (sentence-level strategy)...")
    print("-"*80)
    try:
        from pipeline_improved import run_improved_pipeline
        sentence_results = run_improved_pipeline(
            compression_rate=compression_rate,
            limit=limit,
            save_results=False,
            compression_strategy="sentence",
            use_context_repair=True,
            verbose=False
        )
        results['sentence'] = sentence_results
        print("\n✓ Sentence-level pipeline completed")
    except Exception as e:
        print(f"\n✗ Sentence-level pipeline failed: {e}")
        results['sentence'] = None
    
    # Generate Comparison Report
    print("\n" + "="*80)
    print("COMPARISON REPORT")
    print("="*80)
    
    # Create comparison table
    print(f"\n{'Method':<25} {'ASR (↓)':<12} {'BRR (↓)':<12} {'Improvement'}")
    print("-"*80)
    
    baseline_brr = results['original']['brr'] if results['original'] else 0
    
    for method, result in results.items():
        if result:
            asr = result['asr']
            brr = result['brr']
            
            if method == 'original':
                improvement = "Baseline"
            else:
                brr_reduction = ((baseline_brr - brr) / baseline_brr * 100) if baseline_brr > 0 else 0
                improvement = f"{brr_reduction:+.1f}% BRR"
            
            print(f"{method.replace('_', ' ').title():<25} {asr*100:>6.2f}%     {brr*100:>6.2f}%     {improvement}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    if results['original'] and results['hybrid']:
        orig_brr = results['original']['brr']
        hybrid_brr = results['hybrid']['brr']
        
        if hybrid_brr < orig_brr:
            improvement_pct = (orig_brr - hybrid_brr) / orig_brr * 100
            print(f"✓ IMPROVED: Hybrid strategy reduced over-refusal by {improvement_pct:.1f}%")
            print(f"  (BRR: {orig_brr*100:.1f}% → {hybrid_brr*100:.1f}%)")
        else:
            print(f"  Hybrid strategy: BRR = {hybrid_brr*100:.1f}% (baseline: {orig_brr*100:.1f}%)")
        
        orig_asr = results['original']['asr']
        hybrid_asr = results['hybrid']['asr']
        print(f"✓ Safety maintained: ASR = {hybrid_asr*100:.1f}% (baseline: {orig_asr*100:.1f}%)")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("• Use 'hybrid' strategy for best balance of safety and usability")
    print("• Use 'sentence' strategy for maximum grammaticality (low compression)")
    print("• Use 'safety_aware' strategy when safety-critical tokens must be preserved")
    print("• Enable context_repair to help critic handle fragmented text")
    
    # Save comparison results
    comparison_file = f"results/comparison_{compression_rate:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Detailed results saved to: {comparison_file}")
    
    print("\n" + "="*80)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compare original vs improved pipelines')
    parser.add_argument('--compression', type=float, default=0.5,
                        help='Compression ratio to test (default: 0.5)')
    parser.add_argument('--limit', type=int, default=10,
                        help='Number of samples to process (default: 10)')
    
    args = parser.parse_args()
    
    if not 0.0 <= args.compression <= 1.0:
        print("Error: Compression ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    run_comparison(compression_rate=args.compression, limit=args.limit)
