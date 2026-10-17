# Safety-Preserving Compression for Agentic Traces

How does compressing an agent's reasoning and tool output change a safety critic's ability to detect attacks?

This NLP 245 research project compares compression strategies and measures the trade-off between attack detection and benign refusal.

**[Final report](NLP_245_Final%20Report/NLP_245_Project_Report.pdf)** · [Implementation](pipeline_improved.py) · [Saved results](results/)

## Approach

1. Load agent traces and labeled evaluation examples.
2. Compress the traces using entropy-based, sentence-level, safety-aware, or hybrid strategies.
3. Evaluate the compressed inputs with a safety critic.
4. Compare attack success rate (ASR) and benign refusal rate (BRR) across compression settings.

The project uses [AgentDojo](https://github.com/ethz-spylab/agentdojo) and related evaluation data. The original pipeline uses GPT-2 perplexity as a compression signal and Llama Guard 3 as its critic.

## My contribution

Glenn's implementation explores safety-aware and hybrid compression, with parameter sweeps for comparing safety and usability. See [pipeline.sh](pipeline.sh) for the targeted sweep and the final report for the project's analysis.

## Setup

```bash
git clone https://github.com/ggrantrichards/245-project.git
cd 245-project
pip install -r requirements.txt
huggingface-cli login
```

Accept access requirements for any gated models used by your chosen configuration. Model downloads and GPU requirements depend on the pipeline and settings.

## Run experiments

```bash
# Original pipeline
python pipeline.py --compression 0.5 --limit 10 --save

# Hybrid compression
python3 pipeline_improved.py --strategy hybrid --compression 0.5 --limit 10 --verbose --save

# Compare pipelines
python3 compare_pipelines.py --compression 0.5 --limit 10

# Targeted safety-aware and hybrid sweep
bash pipeline.sh

# Plot saved results
python3 plot_results.py
```

## Strategies

| Strategy | Purpose |
| --- | --- |
| `entropy` | Perplexity-based baseline |
| `sentence` | Preserve complete sentences |
| `safety_aware` | Preserve safety-relevant content |
| `hybrid` | Combine compression approaches |

## Interpreting results

Lower ASR and lower BRR are desirable, but improving one can worsen the other. Comparisons must use the same dataset, critic, sample size, and compression definition.

Earlier development notes include small-run observations and expected improvements. Those are not universal performance guarantees. Use the [final report](NLP_245_Final%20Report/NLP_245_Project_Report.pdf) and matching saved experiment outputs for quantitative conclusions. A model-based critic is an evaluator, not independent ground truth.

## Further documentation

- [Setup notes](SETUP.md)
- [Quick start](QUICKSTART.md)
- [Implementation changes](IMPROVEMENTS.md)
- [Results analysis](RESULTS_ANALYSIS.md)
