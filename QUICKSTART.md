# Quick Start Guide

## 🚦 Before You Begin

**You must authenticate with Hugging Face first:**

```bash
huggingface-cli login
```

Then paste your token when prompted. Get your token from: https://huggingface.co/settings/tokens

**Request access to Llama Guard 3:** https://huggingface.co/meta-llama/Llama-Guard-3-8B

---

## 🏃 Running Experiments

### Basic Usage

```bash
# Run with default settings (50% compression, 10 samples)
python3 pipeline.py

# Run without compression (baseline)
python3 pipeline.py --compression 0.0

# Run with high compression
python3 pipeline.py --compression 0.7

# Process more samples
python3 pipeline.py --limit 50

# Save results to JSON
python3 pipeline.py --compression 0.5 --save
```

### Run All Experiments at Once

```bash
./run_experiments.sh
```

This will run baseline + 3 compression ratios (0.3, 0.5, 0.7) and save all results.

---

## 📊 Understanding the Output

```
--- Results ---
Processed: 10                                  # Total samples evaluated
Unsafe Pass-through Rate (Safety Loss): 20%    # How often attacks are MISSED
Benign Refusal Rate: 10%                       # How often safe content is BLOCKED
```

**Goal**: Low ASR (safety loss) + Low BRR (over-refusal)

---

## 🎯 For Your Milestone

1. **Run experiments** with multiple compression ratios
2. **Create a table** comparing results
3. **Analyze errors**: Which attacks get through? Why?
4. **Write 2 pages** (see MILESTONE_CHECKLIST.md for structure)

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Not logged in" | Run `huggingface-cli login` |
| "Access denied" | Request access at https://huggingface.co/meta-llama/Llama-Guard-3-8B |
| Out of memory | You have 80GB GPU, this shouldn't happen. Check if other processes are using GPU |
| Dataset not found | Run `python3 download_data.py` |

---

## 📁 Files Overview

- **`pipeline.py`** - Main experiment code
- **`download_data.py`** - Downloads AgentDojo dataset
- **`run_experiments.sh`** - Runs all experiments automatically
- **`SETUP.md`** - Detailed setup instructions
- **`MILESTONE_CHECKLIST.md`** - Milestone writing guide
- **`requirements.txt`** - Python dependencies

---

## 💡 Command Examples

```bash
# Quick test (just 5 samples)
python3 pipeline.py --compression 0.5 --limit 5

# Full baseline run
python3 pipeline.py --compression 0.0 --limit 50 --save

# High compression test
python3 pipeline.py --compression 0.8 --limit 20 --save

# Check available options
python3 pipeline.py --help
```

---

**Need more help?** See `SETUP.md` for detailed instructions or `MILESTONE_CHECKLIST.md` for project tasks.
