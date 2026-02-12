# NLP 245 Project Milestone Checklist
**Due: February 17th, 2026 11:59PM**

## 📋 Immediate Actions (Before Running Experiments)

- [ ] **Authenticate with Hugging Face**
  - [ ] Get HF token from https://huggingface.co/settings/tokens
  - [ ] Request access to Llama Guard 3: https://huggingface.co/meta-llama/Llama-Guard-3-8B
  - [ ] Run: `huggingface-cli login`
  - [ ] Verify: `huggingface-cli whoami` shows your username

- [ ] **Test the Pipeline**
  ```bash
  cd /home/apinero/nlp-main/NLP\ 245/245-project
  python3 pipeline.py
  ```

---

## 🧪 Experiments to Run (Max 2 pages for milestone)

### 1. Baseline Experiments
- [ ] Run pipeline with NO compression (critic sees full trace)
- [ ] Record: True Positives, False Negatives, False Positives, True Negatives
- [ ] Calculate: Attack Success Rate (ASR), Benign Refusal Rate (BRR)

### 2. Compression Experiments
- [ ] Run with 30% compression ratio
- [ ] Run with 50% compression ratio
- [ ] Run with 70% compression ratio
- [ ] Compare metrics to baseline

### 3. Naive Baseline Comparison
- [ ] Implement simple truncation (keep first N tokens)
- [ ] Implement simple truncation (keep last N tokens)
- [ ] Compare to entropy-based compression

### 4. Error Analysis
- [ ] Identify which attack types are missed after compression
- [ ] Analyze: What tokens were removed in failed cases?
- [ ] Document: Patterns in false negatives vs. false positives

---

## 📝 Milestone Writing (Required Sections)

### 1. Abstract (1 paragraph)
- [ ] Motivate the problem (long agent traces + limited context)
- [ ] State your approach (entropy-based compression between generator & critic)
- [ ] Highlight main results (even if preliminary)
- [ ] **Keep it concise!**

### 2. Approach (≈0.5 pages)
- [ ] Describe your Generator-Compressor-Critic pipeline
- [ ] Explain entropy-based token selection
- [ ] Specify: GPT-2 as compressor, Llama Guard 3 as critic
- [ ] Make clear what you implemented vs. what's from libraries
- [ ] Mention baselines (no compression, naive truncation)

### 3. Experiments (≈0.75 pages)
- [ ] **Data**: AgentDojo (cite, describe attack types)
- [ ] **Evaluation metrics**: ASR (False Negative Rate), BRR (False Positive Rate)
- [ ] **Experimental details**: 
  - Model: Llama-Guard-3-8B
  - Compressor: GPT-2
  - Compression ratios tested: [0.0, 0.3, 0.5, 0.7]
  - Hardware: NVIDIA A100 80GB
- [ ] **Results**: Create table showing:
  ```
  | Compression | ASR (%) | BRR (%) | Tokens Kept |
  |-------------|---------|---------|-------------|
  | None (0.0)  |   X%    |   Y%    |    100%     |
  | Low (0.3)   |   X%    |   Y%    |     70%     |
  | Med (0.5)   |   X%    |   Y%    |     50%     |
  | High (0.7)  |   X%    |   Y%    |     30%     |
  ```
- [ ] **Discussion**:
  - Are results expected? Better or worse than anticipated?
  - Where do errors come from? (e.g., specific attack types)
  - What does this tell you about next steps?

### 4. Future Work (≈0.25 pages)
- [ ] Expand to more datasets (LLMail-Inject, Lakera PINT)
- [ ] Try alternative compression methods (sentence-level, semantic)
- [ ] Train a safety-aware compressor (explicitly preserve safety tokens)
- [ ] Qualitative analysis (show example traces)
- [ ] Test with different critics

### 5. References
- [ ] Use BibTeX format (already set up in template)
- [ ] Include: AgentDojo paper, Llama Guard, HarmBench, LLMLingua papers

---

## 🎯 Key Success Criteria (From Instructions)

### Progress (Most Important!)
- [ ] Have a working code base that runs experiments
- [ ] Obtained initial results (even if performance is poor)
- [ ] Set up data, baselines, and evaluation metrics
- [ ] ≈50% of total project work complete

### Understanding
- [ ] Show comprehensive understanding of:
  - Problem: Safety loss under compression
  - Methods: Entropy-based token selection
  - Metrics: ASR, BRR
  - Research area: Prompt compression + AI safety

### Writing Quality
- [ ] Clear and concise (max 2 pages!)
- [ ] Technical but readable by fellow NLP 245 students
- [ ] No grammatical errors
- [ ] Logical flow
- [ ] **Proofread multiple times!**

---

## ⚠️ Common Pitfalls to Avoid

- ❌ **Don't worry if performance is poor!** The instructors explicitly say they won't take points off for bad results. Just explain *why* and what you'll do next.
- ❌ **Don't spend too much space reintroducing your approach.** Focus on what you've **done since the proposal**.
- ❌ **Don't exceed 2 pages** (excluding figures and references) - this is a hard limit!
- ❌ **Don't use ChatGPT to write the paper** - instructors explicitly prohibit this
- ❌ **Don't forget to cite code you didn't write** (e.g., transformers library, Llama Guard)

---

## 📊 Code Enhancements to Consider

### Priority: High
- [ ] Add command-line arguments to `pipeline.py`:
  ```python
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--compression', type=float, default=0.5)
  parser.add_argument('--limit', type=int, default=10)
  args = parser.parse_args()
  ```

- [ ] Implement no-compression baseline (ratio=0.0)
- [ ] Save results to JSON for easy analysis
- [ ] Add progress bars for long runs

### Priority: Medium
- [ ] Implement simple truncation baselines
- [ ] Add qualitative output (show example compressed traces)
- [ ] Error analysis: categorize which attacks are missed
- [ ] Expand dataset to full AgentDojo (not just tiny subset)

### Priority: Low (for final paper)
- [ ] Integrate LLMail-Inject and Lakera PINT datasets
- [ ] Implement alternative compression methods
- [ ] Train safety-aware compressor
- [ ] Add visualization of compression patterns

---

## 💬 When to Meet with Course Staff

- If ASR is unexpectedly high/low and you don't know why
- If you're stuck on implementation
- If you want feedback on early draft (bring it!)
- If you're unsure about next steps

**Remember**: They want to help you succeed and potentially publish this!

---

## ✅ Final Pre-Submission Checklist

- [ ] All code runs without errors
- [ ] Results table is complete
- [ ] Milestone PDF is exactly 2 pages (excluding figures/refs)
- [ ] References in BibTeX format
- [ ] All group members reviewed and approved
- [ ] No grammatical errors (proofread 3+ times)
- [ ] Submitted on Canvas before 11:59 PM Feb 17

---

**Good luck! You have a strong foundation. Focus on getting solid experimental results and clear writing. 🚀**
