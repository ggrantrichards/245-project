import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import logging
import gc 

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = get_device()

class AgentDojoLoader:
    def __init__(self, limit=50):
        logger.info("Loading Dataset: Genteki/agentdojo_tiny...")
        try:
            self.dataset = load_dataset("Genteki/agentdojo_tiny", split="train")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise e
            
        self.limit = limit
        
        if len(self.dataset) > 0:
            logger.info(f"Dataset Columns Found: {self.dataset.column_names}")

    def format_trace(self, example) -> str:
        # 1. Grab the direct content from detected columns
        system = example.get('system_prompt', '')
        user_input = example.get('prompt', '')
        
        # 2. Check for the history list
        history = example.get('conversation') or example.get('history') or []
        
        # 3. Build the trace starting with known valid columns
        trace_parts = []
        if system: trace_parts.append(f"[SYSTEM]: {system}")
        if user_input: trace_parts.append(f"[USER]: {user_input}")
        
        # 4. Append history only if it's a valid list and not empty
        if isinstance(history, list) and len(history) > 0:
            for turn in history:
                if isinstance(turn, dict):
                    role = turn.get('role', 'unknown').upper()
                    content = turn.get('content', '')
                    trace_parts.append(f"[{role}]: {content}")
                else:
                    trace_parts.append(str(turn))
                    
        return "\n".join(trace_parts)

    def __iter__(self):
        count = 0
        for example in self.dataset:
            if count >= self.limit:
                break
            
            trace = self.format_trace(example)
            
            # Ground truth detection
            is_unsafe = False
            if 'is_safe' in example:
                is_unsafe = not example['is_safe']
            elif 'label' in example:
                is_unsafe = (example['label'] == 1) 
            elif 'jailbreak' in example:
                is_unsafe = example['jailbreak']
            
            yield {
                "trace": trace,
                "is_unsafe": is_unsafe
            }
            count += 1

class EntropyCompressor:
    def __init__(self, model_name="gpt2"):
        logger.info(f"Loading Compressor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load GPT2 to standard device (it's small, ~0.5GB, so it's safe)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def compress(self, text: str, compression_ratio: float = 0.5) -> str:
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        input_ids = inputs.input_ids[0]
        
        if len(input_ids) < 10: 
            return text

        # Calculate Perplexity (Loss)
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, labels=inputs.input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Pad first token
            padding = torch.tensor([losses.mean()], device=DEVICE)
            losses = torch.cat([padding, losses])

        # Select high-information tokens
        num_keep = int(len(input_ids) * (1 - compression_ratio))
        top_k_indices = torch.topk(losses, num_keep).indices.sort().values

        compressed_ids = input_ids[top_k_indices]
        return self.tokenizer.decode(compressed_ids, skip_special_tokens=True)

class SafetyCritic:
    def __init__(self, model_id="meta-llama/Llama-Guard-3-8B"):
        logger.info(f"Loading Critic: {model_id}")
        logger.info("NOTE: Using automatic device mapping to prevent OOM crashes.")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                low_cpu_mem_usage=True
            )
                
        except Exception as e:
            logger.error("\n[ERROR] Llama Guard load failed.")
            logger.error("Ensure you ran `huggingface-cli login` and have access to the model.")
            raise e

    def evaluate(self, trace: str) -> str:
        # Ensure inputs are on the same device as the model's first layer
        model_device = self.model.device
        
        chat = [{"role": "user", "content": trace}]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(model_device)
        
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=20, pad_token_id=0)
        
        prompt_len = input_ids.shape[-1]
        result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        
        return "unsafe" if "unsafe" in result.lower() else "safe"

def run_pipeline():
    COMPRESSION_RATE = 0.5
    
    # Garbage collection before start
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Initialize
    try:
        loader = AgentDojoLoader(limit=10)
        compressor = EntropyCompressor()
        critic = SafetyCritic()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    stats = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    
    logger.info(f"\n--- Running Pipeline (Ratio: {COMPRESSION_RATE:.0%}) ---\n")

    for i, sample in enumerate(loader):
        original = sample['trace']
        ground_truth = sample['is_unsafe']
        
        # 1. Compress
        compressed = compressor.compress(original, compression_ratio=COMPRESSION_RATE)
        
        # 2. Critique
        decision = critic.evaluate(compressed)
        is_unsafe_pred = (decision == "unsafe")
        
        # 3. Stats
        if ground_truth and is_unsafe_pred:
            stats["tp"] += 1
        elif ground_truth and not is_unsafe_pred:
            stats["fn"] += 1
            logger.info(f"[MISS] Safety Loss! Trace #{i} marked safe after compression.")
        elif not ground_truth and is_unsafe_pred:
            stats["fp"] += 1
        else:
            stats["tn"] += 1

    # Results
    total_attacks = stats["tp"] + stats["fn"]
    asr = stats["fn"] / total_attacks if total_attacks > 0 else 0
    total_samples = sum(stats.values())
    
    logger.info("\n--- Results ---")
    logger.info(f"Processed: {total_samples}")
    logger.info(f"Unsafe Pass-through Rate (Safety Loss): {asr:.2%}")
    
    refusal_denom = stats['fp'] + stats['tn']
    brr = stats['fp'] / refusal_denom if refusal_denom > 0 else 0
    logger.info(f"Benign Refusal Rate: {brr:.2%}")

if __name__ == "__main__":
    run_pipeline()