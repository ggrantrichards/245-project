# Environment Setup Guide

## ✅ Completed Steps

1. **Python Environment**: Python 3.13.9 ✓
2. **GPU Available**: NVIDIA A100 80GB (CUDA 12.6) ✓
3. **Dependencies Installed**: All packages from requirements.txt ✓
4. **Dataset Downloaded**: AgentDojo tiny dataset (10 samples) ✓

---

## 🔑 Required: Hugging Face Authentication

Your pipeline uses **Llama Guard 3** (`meta-llama/Llama-Guard-3-8B`), which requires:

### Step 1: Get a Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **READ** access (or use existing)
3. Copy the token

### Step 2: Request Access to Llama Guard 3
1. Visit: https://huggingface.co/meta-llama/Llama-Guard-3-8B
2. Click "Request access" and accept the license agreement
3. Wait for approval (usually instant if you've accepted Meta's terms)

### Step 3: Login via CLI
Run ONE of these commands in your terminal:

```bash
# New CLI command (recommended)
huggingface-cli login

# OR older command
hf auth login
```

When prompted, paste your token and press Enter.

### Step 4: Verify Authentication
```bash
huggingface-cli whoami
# Should show your username
```

---

## 🚀 Running the Pipeline

Once authenticated, run:

```bash
cd /home/apinero/nlp-main/NLP\ 245/245-project
python3 pipeline.py
```

### What the Pipeline Does:

1. **Loads Dataset**: 10 agent traces from AgentDojo
2. **Compresses**: Uses GPT-2 to calculate token perplexity and remove low-information tokens (50% compression ratio)
3. **Evaluates Safety**: Llama Guard 3 judges if the compressed trace is safe/unsafe
4. **Measures Safety Loss**: Compares predictions to ground truth labels

### Expected Output:

```
Loading Dataset from local file: data/agentdojo_tiny.json
Loading Compressor: gpt2
Loading Critic: meta-llama/Llama-Guard-3-8B

--- Running Pipeline (Ratio: 50%) ---

[MISS] Safety Loss! Trace #X marked safe after compression.

--- Results ---
Processed: 10
Unsafe Pass-through Rate (Safety Loss): X%
Benign Refusal Rate: X%
```

---

## 📊 Key Metrics Explained

- **Unsafe Pass-through Rate (Safety Loss)**: False Negative Rate - how often the critic misses attacks after compression
- **Benign Refusal Rate**: False Positive Rate - how often safe content is incorrectly flagged as unsafe

---

## 🐛 Troubleshooting

### Out of Memory Error
If you get CUDA OOM, the pipeline already uses:
- `device_map="auto"` for Llama Guard 3
- `torch.bfloat16` for reduced memory
- With your A100 80GB, this should NOT be an issue

### Dataset Not Found
Run: `python3 download_data.py`

### Model Loading Failures
- Ensure you completed HF authentication
- Verify you have access to Llama Guard 3 on Hugging Face

---

## 📝 Next Steps for Your Project

According to the milestone instructions, you should have:

### ✅ Already Completed (Based on Proposal):
- Research background & literature review
- Project goals defined
- Dataset selected (AgentDojo, LLMail-Inject, Lakera PINT)
- Basic pipeline architecture

### 🎯 For Milestone (Due Feb 17):
- [ ] Run baseline experiments (no compression)
- [ ] Run compression experiments at multiple ratios (0.3, 0.5, 0.7)
- [ ] Collect quantitative results (ASR, BRR)
- [ ] Analyze where compression fails (error analysis)
- [ ] Compare to naive baselines (simple truncation)
- [ ] Write up: Abstract, Approach, Experiments, Results, Future Work

### 💡 Experiment Suggestions:

1. **Vary Compression Ratio**: Test 0.3, 0.5, 0.7 to see safety loss curve
2. **Compare Compression Methods**:
   - Current: Entropy-based (keep high-perplexity tokens)
   - Baseline: Simple truncation (first/last N tokens)
   - Baseline: Random deletion
3. **Error Analysis**: Which attack types are missed after compression?
4. **Ablation Studies**: Does keeping safety-critical tokens (e.g., "password", "delete") help?

---

## 🔗 Useful Resources

- **AgentDojo**: https://agentdojo.spylab.ai/
- **Llama Guard 3**: https://huggingface.co/meta-llama/Llama-Guard-3-8B
- **HarmBench**: https://github.com/centerforaisafety/HarmBench
- **Your Proposal PDF**: Course_Project_245_Proposal/Course_Project_245_Proposal.pdf
