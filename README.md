# Safety-Preserving Compression for Agentic Traces

This project implements a research pipeline to evaluate the **Safety Loss** incurred when compressing agent trajectories (Chain-of-Thought + Tool Output) before passing them to a Safety Critic.

## Architecture

1.  **Dataset:** Loads real attack vectors from [SourceryAI/AgentDojo](https://huggingface.co/datasets/Genteki/agentdojo_tiny / https://github.com/ethz-spylab/agentdojo / https://agentdojo.spylab.ai/).
2.  **Compressor:** An entropy-based token compressor using `gpt2` as a proxy for information density. It discards low-perplexity (redundant) tokens.
3.  **Critic:** Uses `meta-llama/Llama-Guard-3-8B` as the ground-truth evaluator.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install torch transformers datasets accelerate bitsandbytes
    ```

2.  **Authenticate Hugging Face:**
    You must have access to Llama Guard 3.
    ```bash
    huggingface-cli login
    ```

3.  **Run:**
    ```bash
    python pipeline.py
    ```