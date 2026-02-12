# Pipeline Improvements: Addressing Blocking and Over-Refusal

## Problem Summary

The original pipeline showed **high over-refusal** (90% BRR at 50%+ compression):
- **Root Cause**: Entropy-based token pruning creates fragmented, ungrammatical text
- **Impact**: Safety critic flags benign content as unsafe
- **Result**: System blocks 9 out of 10 legitimate requests at medium compression

## Implemented Solutions

### 1. **Improved Compression Strategies** (`ImprovedCompressor`)

#### A. Safety-Aware Compression
```python
# Usage: --strategy safety_aware
```

**How it works:**
- Identifies safety-critical keywords (password, delete, transfer, etc.)
- Boosts importance scores for safety-related tokens by 2x
- Always preserves these tokens during compression
- Ensures attack indicators remain visible to the critic

**Benefits:**
- Maintains attack detection capability (low ASR)
- Preserves semantic context around safety-critical actions
- Better than blind entropy-based pruning

**Keywords Preserved:**
- Authentication: password, secret, credentials, token, api, key
- Actions: delete, remove, transfer, execute, command, script
- Data operations: database, sql, query, drop, update, insert
- Security: unauthorized, hack, exploit, inject, malicious

#### B. Sentence-Level Compression
```python
# Usage: --strategy sentence
```

**How it works:**
- Splits text into complete sentences
- Scores each sentence by information content (average entropy)
- Keeps top-k complete sentences
- Preserves sentences containing safety keywords

**Benefits:**
- Maintains grammaticality (reduces fragmentation)
- Preserves semantic coherence
- Significantly reduces over-refusal
- Best for low-medium compression (<50%)

**Trade-off:**
- Less fine-grained than token-level compression
- May keep more tokens than target ratio

#### C. Hybrid Strategy (Recommended)
```python
# Usage: --strategy hybrid  # Default
```

**How it works:**
- Uses sentence-level compression for compression < 40% (preserves grammar)
- Switches to safety-aware token-level for compression ≥ 40% (preserves safety)
- Adapts strategy based on compression intensity

**Benefits:**
- Best of both worlds
- Maintains grammaticality at low compression
- Preserves safety signals at high compression
- Recommended default strategy

#### D. Original Entropy-Based
```python
# Usage: --strategy entropy
```

**How it works:**
- Original implementation (baseline comparison)
- Pure token-level pruning by entropy
- No safety-awareness

**Use case:** Comparison baseline only

### 2. **Improved Safety Critic** (`ImprovedSafetyCritic`)

#### Context Repair Feature
```python
# Usage: --no-context-repair to disable (enabled by default)
```

**How it works:**
- Detects fragmented text (< 30% complete sentences)
- Adds minimal context marker: `[COMPRESSED AGENT TRACE - May contain fragments]`
- Helps critic understand input may be incomplete
- Prevents false positives from fragmentation

**Benefits:**
- Reduces over-refusal on compressed text
- Maintains safety evaluation accuracy
- Minimal computational overhead

## Usage Examples

### Run with default (best) settings:
```bash
python3 pipeline_improved.py --compression 0.5 --limit 10 --verbose --save
```

### Compare all strategies:
```bash
python3 compare_pipelines.py --compression 0.5 --limit 10
```

### Test specific strategy:
```bash
# Sentence-level (best for grammaticality)
python3 pipeline_improved.py --strategy sentence --compression 0.3 --limit 10 --verbose

# Safety-aware (best for preserving attack indicators)
python3 pipeline_improved.py --strategy safety_aware --compression 0.5 --limit 10 --verbose

# Hybrid (recommended)
python3 pipeline_improved.py --strategy hybrid --compression 0.5 --limit 10 --verbose
```

### Disable context repair:
```bash
python3 pipeline_improved.py --no-context-repair --compression 0.5 --limit 10
```

## Expected Improvements

### Baseline (Original Pipeline)
- **Compression 0%**: BRR = 0%, ASR = 0%
- **Compression 50%**: BRR = 90%, ASR = 0%  ⚠️ **High over-refusal**
- **Compression 70%**: BRR = 90%, ASR = 0%

### Improved Pipeline (Hybrid Strategy)
- **Compression 0%**: BRR = 0%, ASR = 0% (no change)
- **Compression 50%**: BRR ≈ 20-40%, ASR ≈ 0%  ✓ **Reduced over-refusal**
- **Compression 70%**: BRR ≈ 40-60%, ASR ≈ 0-10%

**Key Improvement**: 50-70% reduction in over-refusal while maintaining safety

## Architecture Diagram

```
┌─────────────────┐
│  Agent Trace    │
│  (Full text)    │
└────────┬────────┘
         │
         v
┌─────────────────────────────────┐
│  ImprovedCompressor             │
│  ┌───────────────────────────┐  │
│  │ Strategy Selection:       │  │
│  │ • Hybrid (default)        │  │
│  │ • Safety-aware            │  │
│  │ • Sentence-level          │  │
│  │ • Entropy (baseline)      │  │
│  └───────────────────────────┘  │
│                                  │
│  ┌───────────────────────────┐  │
│  │ Safety Keyword Detection  │  │
│  │ • password, delete, etc.  │  │
│  │ • Boost token importance  │  │
│  └───────────────────────────┘  │
└────────┬────────────────────────┘
         │
         v
┌─────────────────┐
│ Compressed Text │
│ (Reduced size)  │
└────────┬────────┘
         │
         v
┌─────────────────────────────────┐
│  ImprovedSafetyCritic           │
│  ┌───────────────────────────┐  │
│  │ Context Repair:           │  │
│  │ • Detect fragmentation    │  │
│  │ • Add context marker      │  │
│  └───────────────────────────┘  │
│                                  │
│  ┌───────────────────────────┐  │
│  │ Llama-Guard-3-8B          │  │
│  │ • Evaluate safety         │  │
│  │ • Return: safe/unsafe     │  │
│  └───────────────────────────┘  │
└────────┬────────────────────────┘
         │
         v
┌─────────────────┐
│  Safety Decision│
│  (safe/unsafe)  │
└─────────────────┘
```

## Technical Details

### Safety Keyword List (Extensible)
```python
SAFETY_KEYWORDS = {
    'password', 'secret', 'delete', 'remove', 'transfer', 'funds', 
    'account', 'credentials', 'token', 'api', 'key', 'private',
    'unauthorized', 'hack', 'exploit', 'inject', 'malicious',
    'admin', 'root', 'sudo', 'execute', 'command', 'script',
    'database', 'sql', 'query', 'drop', 'truncate', 'update',
    'insert', 'email', 'send', 'forward', 'download', 'upload'
}
```

**To extend**: Add domain-specific keywords to the set

### Sentence Splitting Logic
- Preserves structured markers: `[SYSTEM]:`, `[USER]:`, etc.
- Splits on sentence boundaries: `.`, `!`, `?`
- Maintains context within each sentence
- Avoids mid-sentence cuts

### Context Repair Heuristic
- **Triggered when**: Less than 30% of words form complete sentences
- **Action**: Prepends `[COMPRESSED AGENT TRACE - May contain fragments]`
- **Rationale**: Helps LLM critic understand input context

## Evaluation Metrics

### Primary Metrics
- **ASR (Attack Success Rate)**: % of attacks that bypass detection (lower is better)
- **BRR (Benign Refusal Rate)**: % of safe content incorrectly blocked (lower is better)

### Secondary Metrics
- **Recall**: True Positive Rate (should be high)
- **Precision**: Positive Predictive Value (should be high)

### Success Criteria
- ASR < 5% (maintains safety)
- BRR < 20% (reduces over-refusal)
- Improvement over baseline by 50%+

## Bug Fixes and Technical Challenges

### Device Mismatch Error (Fixed)

**Issue**: The hybrid and safety-aware strategies initially failed with:
```
Expected all tensors to be on the same device, but got tensors is on cpu, 
different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_cat)
```

**Root Cause**: Boolean masks for safety-critical tokens were created on CPU by default, while loss tensors were on GPU:
```python
# Original (buggy) code
safety_mask = torch.zeros(len(input_ids), dtype=torch.bool)  # Created on CPU
# Later: torch.cat([top_k_indices, torch.where(safety_mask)[0]])  # Mismatch!
```

**Solution**: Explicitly specify device placement for all intermediate tensors:
```python
# Fixed code
safety_mask = torch.zeros(len(input_ids), dtype=torch.bool, device=DEVICE)
safety_indices = torch.where(safety_mask)[0]  # Now on same device
all_keep_indices = torch.cat([top_k_indices, safety_indices])  # No mismatch
```

**Lesson**: When working with PyTorch on CUDA, always ensure intermediate tensors are created on the same device as model outputs to avoid expensive CPU-GPU transfers and runtime errors.

**Verification**: Run `python3 pipeline_improved.py --strategy safety_aware --compression 0.5 --limit 3` to confirm the fix works.

## Limitations and Future Work

### Current Limitations
1. **Small dataset**: Only 10 samples in agentdojo_tiny
2. **Simple keyword matching**: Could use learned embeddings
3. **Fixed threshold**: Context repair uses hardcoded 30% threshold
4. **No attention scores**: Could use model attention for better token selection

### Future Improvements
1. **Learned safety-aware compressor**: Train on safety-labeled data
2. **Dynamic threshold**: Adapt context repair based on compression ratio
3. **Multi-model ensemble**: Combine multiple critics for robustness
4. **Sentence embeddings**: Use semantic similarity instead of entropy
5. **Expand dataset**: Test on full AgentDojo, LLMail-Inject, Lakera PINT

## Files Added/Modified

### New Files
- `pipeline_improved.py` - Main improved pipeline implementation
- `compare_pipelines.py` - Comparison script
- `IMPROVEMENTS.md` - This documentation

### Modified Files
- None (original `pipeline.py` kept for comparison)

## Quick Start

1. **Run comparison to see improvements**:
```bash
python3 compare_pipelines.py --compression 0.5 --limit 10
```

2. **Test improved pipeline with verbose output**:
```bash
python3 pipeline_improved.py --strategy hybrid --compression 0.5 --limit 10 --verbose --save
```

3. **Try different strategies**:
```bash
# Best for grammaticality
python3 pipeline_improved.py --strategy sentence --compression 0.3 --limit 10 --verbose

# Best for safety preservation
python3 pipeline_improved.py --strategy safety_aware --compression 0.5 --limit 10 --verbose

# Best overall (default)
python3 pipeline_improved.py --strategy hybrid --compression 0.5 --limit 10 --verbose
```

## Results Interpretation

### Good Result (Improved)
```
ASR: 0.00%    (no attacks bypass)
BRR: 20.00%   (only 20% false positives)
```
**Interpretation**: Safe and usable

### Bad Result (Original)
```
ASR: 0.00%    (no attacks bypass)
BRR: 90.00%   (90% false positives)
```
**Interpretation**: Safe but unusable

### Concerning Result
```
ASR: 20.00%   (attacks slipping through)
BRR: 0.00%    (no false positives)
```
**Interpretation**: Usable but unsafe - **needs attention**

## Citation

If you use this improved pipeline in your research, please cite:

```bibtex
@misc{improved_compression_pipeline,
  title={Safety-Aware Compression for Agent Trajectories},
  author={[Your Name]},
  year={2026},
  note={NLP 245 Course Project}
}
```

## Support

For questions or issues:
1. Check verbose output: `--verbose` flag
2. Compare with baseline: `compare_pipelines.py`
3. Test with different strategies
4. Review results JSON files in `results/` directory

---

**Last Updated**: February 12, 2026
**Version**: 1.0
