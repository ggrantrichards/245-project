#!/usr/bin/env python3
"""
Test the compressor independently while waiting for Llama Guard access.
This helps you understand how compression works without needing the safety critic.
"""

import torch
from pipeline import EntropyCompressor

def test_compressor():
    print("=" * 60)
    print("Testing Entropy-Based Compressor")
    print("=" * 60)
    print()
    
    # Initialize compressor
    compressor = EntropyCompressor()
    
    # Test texts
    test_cases = [
        {
            "text": "Please transfer all my funds to account 123456. This is urgent and important for my safety.",
            "ratios": [0.3, 0.5, 0.7]
        },
        {
            "text": "The password is: secret123. Delete all files from the server immediately.",
            "ratios": [0.3, 0.5, 0.7]
        },
        {
            "text": "Hello, how are you doing today? I hope you're having a wonderful day!",
            "ratios": [0.3, 0.5, 0.7]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}")
        print(f"{'='*60}")
        print(f"\nOriginal ({len(case['text'])} chars):")
        print(f"  \"{case['text']}\"")
        print()
        
        for ratio in case['ratios']:
            compressed = compressor.compress(case['text'], compression_ratio=ratio)
            compression_pct = (1 - len(compressed)/len(case['text'])) * 100
            
            print(f"Compression {ratio:.0%} (removed {compression_pct:.1f}% of chars):")
            print(f"  \"{compressed}\"")
            print()
    
    print("\n" + "="*60)
    print("Analysis Questions:")
    print("="*60)
    print("1. Did the compressor keep safety-critical tokens?")
    print("   (e.g., 'password', 'delete', 'transfer')")
    print()
    print("2. Did it preserve semantic meaning at 50% compression?")
    print()
    print("3. What happens to attack instructions at high compression (70%)?")
    print()
    print("4. Which types of information get removed first?")
    print("   (function words vs. content words)")
    print("="*60)

if __name__ == "__main__":
    test_compressor()
