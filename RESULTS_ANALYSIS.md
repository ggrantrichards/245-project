# Experimental Results Analysis

## Results Summary

Your first experiments completed successfully! Here's what you found:

| Compression | ASR (Safety Loss) | BRR (Over-refusal) | Interpretation |
|-------------|-------------------|---------------------|----------------|
| **0% (Baseline)** | 0.00% | 0.00% | Perfect - no attacks missed, no false positives |
| **30%** | 0.00% | 0.00% | Light compression maintains perfect performance |
| **50%** | 0.00% | **90.00%** | ⚠️ High over-refusal! Blocking 90% of safe content |
| **70%** | 0.00% | **90.00%** | ⚠️ Also high over-refusal at high compression |

## What Do These Results Mean?

### Good News: No Safety Loss Detected
- **ASR = 0%** across all conditions means the critic never missed an unsafe trace
- Compression did NOT cause attacks to slip through undetected
- This is actually positive from a safety perspective!

### Concerning: High Over-Refusal at 50%+ Compression
- **BRR jumped from 0% to 90%** at 50% compression
- This means the compressed text becomes so garbled that the critic flags safe content as unsafe
- The critic is being overly cautious when reading fragmented text

### Key Insight
**Your hypothesis is partially validated**: Compression DOES affect the critic's decision-making, but in the opposite way you might have expected:
- Instead of missing attacks (false negatives)
- The critic over-refuses benign content (false positives)

This is actually an important finding! It suggests fragmented/compressed text triggers safety false alarms.

---

## 🔍 Potential Issues to Investigate

### 1. Dataset Composition
The AgentDojo "tiny" dataset only has 10 samples. Check:
```bash
python3 -c "
import json
data = [json.loads(line) for line in open('data/agentdojo_tiny.json')]
unsafe_count = sum(1 for d in data if d.get('jailbreak') or d.get('label') == 1 or not d.get('is_safe'))
print(f'Total samples: {len(data)}')
print(f'Unsafe samples: {unsafe_count}')
print(f'Safe samples: {len(data) - unsafe_count}')
"
```

**If the dataset is imbalanced** (mostly safe or mostly unsafe), your metrics might be skewed.

### 2. Ground Truth Labeling
Double-check how you're extracting ground truth in `pipeline.py`:
```python
# In AgentDojoLoader.__iter__()
is_unsafe = False
if 'is_safe' in example:
    is_unsafe = not example['is_safe']
elif 'label' in example:
    is_unsafe = (example['label'] == 1) 
elif 'jailbreak' in example:
    is_unsafe = example['jailbreak']
```

**Action**: Print some examples to verify labels are correct.

### 3. Llama Guard's Sensitivity
Llama Guard might be very conservative on fragmented text. Try:
```bash
# Look at what the compressed text looks like
python3 test_compressor.py
```

At 50-70% compression, text becomes ungrammatical, which might trigger the safety system.

---

## 📝 For Your Milestone (Due Feb 17)

### Results Section

**Present your findings honestly:**

```latex
\subsection{Results}

Table~\ref{tab:results} presents our initial experimental results on the AgentDojo tiny dataset (N=10 samples).

\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Compression & ASR (\%) & BRR (\%) & Tokens Kept \\
\midrule
None (Baseline) & 0.00 & 0.00 & 100\% \\
Light (30\%) & 0.00 & 0.00 & 70\% \\
Medium (50\%) & 0.00 & 90.00 & 50\% \\
High (70\%) & 0.00 & 90.00 & 30\% \\
\bottomrule
\end{tabular}
\caption{Safety metrics across compression ratios. ASR = Attack Success Rate (false negatives), BRR = Benign Refusal Rate (false positives).}
\label{tab:results}
\end{table}

\subsection{Discussion}

Contrary to our initial hypothesis, compression did not cause the safety critic to miss attacks (ASR remained 0\% across all conditions). Instead, we observed a dramatic increase in over-refusal at compression ratios $\geq$ 50\%, with the BRR jumping from 0\% to 90\%.

This suggests that entropy-based token pruning produces fragmented, ungrammatical text that triggers false positives in the safety classifier. The critic appears to treat incomprehensible input as potentially unsafe, adopting a "refuse when uncertain" strategy.

\textbf{Implications:} While this prevents safety loss, it severely degrades usability. A 90\% BRR means the system would block 9 out of 10 legitimate requests.

\textbf{Limitations:} Our dataset (N=10) is too small for robust conclusions. The AgentDojo "tiny" subset may not be representative of real-world attack distributions.
```

### Future Work Section

```latex
\subsection{Future Work}

Based on these preliminary findings, we plan to:

\begin{enumerate}
    \item \textbf{Expand dataset:} Test on full AgentDojo, LLMail-Inject, and Lakera PINT benchmarks (as proposed) to obtain statistically significant results.
    
    \item \textbf{Improve compression method:} Implement sentence-level or semantic compression to preserve grammaticality and reduce false positives.
    
    \item \textbf{Safety-aware compression:} Train a compressor that explicitly preserves safety-critical tokens (e.g., "password", "delete", "transfer") while removing filler words.
    
    \item \textbf{Qualitative analysis:} Manually inspect compressed traces to understand which linguistic features trigger over-refusal.
    
    \item \textbf{Alternative critics:} Test with other safety models to determine if the over-refusal pattern is specific to Llama Guard or general to safety classifiers.
\end{enumerate}
```

---

## Immediate Next Steps

### 1. Understand Your Data Better
```bash
# See what your dataset actually contains
python3 -c "
import json
data = [json.loads(line) for line in open('data/agentdojo_tiny.json')]
for i, d in enumerate(data):
    print(f'\n=== Sample {i} ===')
    print(f'Name: {d.get(\"name\", \"N/A\")}')
    print(f'Prompt: {d.get(\"prompt\", \"N/A\")[:100]}...')
    print(f'System: {d.get(\"system_prompt\", \"N/A\")[:100]}...')
"
```

### 2. Add Logging to See What the Critic Sees
Modify `pipeline.py` to print the compressed text and critic's decision:

```python
# In run_pipeline(), add after compression:
if i < 3:  # Print first 3 examples
    logger.info(f"\n--- Example {i} ---")
    logger.info(f"Original: {original[:200]}...")
    logger.info(f"Compressed: {compressed[:200]}...")
    logger.info(f"Ground truth: {'UNSAFE' if ground_truth else 'SAFE'}")
    logger.info(f"Critic decision: {decision}")
```

### 3. Test with More Samples
```bash
# Try with all 10 samples (if you used fewer)
python3 pipeline.py --compression 0.5 --limit 10 --save
```

### 4. Try a Lower Compression Ratio
```bash
# Test at 20% compression
python3 pipeline.py --compression 0.2 --limit 10 --save
```

---

## 💡 Interesting Research Directions

Your results actually reveal something important:

**Finding**: Compression causes safety systems to become overly conservative rather than miss attacks.

**Why this matters**:
- Shows a trade-off: safety vs. usability
- Suggests need for "interpretability-preserving" compression
- Opens question: Can we compress while maintaining grammaticality?

**This is publishable!** The milestone instructions say papers don't need perfect results - they need novel insights. You have one.

---

## ✅ For Your Milestone Checklist

- [x] Ran baseline experiments
- [x] Ran compression experiments (3 ratios)
- [x] Obtained quantitative results
- [ ] Understand dataset composition
- [ ] Add qualitative analysis (inspect compressed traces)
- [ ] Write up: Abstract, Approach, Experiments, Results, Future Work
- [ ] Create results table
- [ ] Proofread 3+ times

---

**Bottom line**: Your experiments worked! Now focus on understanding and explaining the results clearly in your 2-page milestone. The over-refusal finding is actually interesting and worth exploring further.
