"""
Two-Stage Extract-then-Abstract Compressor
===========================================

Stage 1 (Extractive): Rule-based extraction of critical elements (user requests,
tool calls, credentials, dangerous commands) + semantic-safety-scored selection
of remaining sentences. Uses the already-loaded base compressor model for entropy
scoring — no extra GPU memory, no extra model-load latency.

Stage 2 (Abstractive Smoothing): The already-short Stage 1 output is rewritten
into fluent prose by the 1B-Instruct model. Only triggered when
compression_ratio >= 0.3 AND stage1_tokens >= 150.  The instruct model is
lazy-loaded on first Stage 2 call via the parent ImprovedCompressor, so at
low compression ratios (e.g. 0.1) it is never loaded at all.

Usage:
    from two_stage_compressor import TwoStageCompressor

    ts = TwoStageCompressor(compressor)   # pass the ImprovedCompressor instance
    result = ts.compress(text, compression_ratio=0.5)
"""

import re
import logging
import torch
import torch.nn.functional as F
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TwoStageCompressor:
    """Drop-in two-stage compression strategy.

    Receives the parent *ImprovedCompressor* directly so it can:
      - reuse self.model (base Llama-3.2-1B) for cheap entropy scoring, and
      - call self._parent._get_instruct_model() lazily only when Stage 2
        LLM smoothing is actually needed.
    """

    # Regex patterns for lines that must NEVER be dropped.
    # Order: most specific first so we short-circuit quickly.
    CRITICAL_PATTERNS = [
        re.compile(r'^\[USER\]:'),                                      # user request
        re.compile(r'^\[SYSTEM\]:'),                                    # system prompt
        re.compile(r'\[TOOL_CALL\]:'),                                  # tool invocations
        re.compile(r'(?i)(password|credential|api[_\s]?key|secret|token)\s*[:=]'),  # credential assignments
        re.compile(r'(?i)(sudo|rm\s+-rf|DROP\s+TABLE|DELETE\s+FROM)'),  # dangerous commands
    ]

    def __init__(self, compressor):
        """
        Args:
            compressor: An ImprovedCompressor instance.  Its base .model,
                        .tokenizer, .embedder, and .harm_vector are used for
                        Stage 1.  Its ._get_instruct_model() is called lazily
                        for Stage 2 smoothing.
        """
        self._parent = compressor                           # for lazy instruct access
        self.model = compressor.model                       # base model — for entropy
        self.tokenizer = compressor.tokenizer               # base tokenizer
        self.embedder = compressor.embedder
        self.harm_vector = compressor.harm_vector
        self._device = next(compressor.model.parameters()).device

    # ------------------------------------------------------------------
    # Stage 1a — Rule-based extraction of critical lines
    # ------------------------------------------------------------------
    def _extract_critical_elements(self, text: str) -> Tuple[List[str], List[str]]:
        """Split *text* into (critical_lines, remaining_lines).

        critical_lines match at least one CRITICAL_PATTERN and are kept
        unconditionally.  remaining_lines are candidates for semantic scoring.
        """
        critical: List[str] = []
        remaining: List[str] = []

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if any(p.search(stripped) for p in self.CRITICAL_PATTERNS):
                critical.append(stripped)
            else:
                remaining.append(stripped)

        return critical, remaining

    # ------------------------------------------------------------------
    # Stage 1b — Semantic + entropy scoring of remaining sentences
    # ------------------------------------------------------------------
    def _get_semantic_safety_scores(self, sentences: List[str]) -> torch.Tensor:
        """Cosine similarity of each sentence against the harm taxonomy vector."""
        if not sentences:
            return torch.tensor([])
        embeddings = self.embedder.encode(sentences, convert_to_tensor=True)
        return F.cosine_similarity(embeddings, self.harm_vector.unsqueeze(0))

    def _get_entropy_scores(self, sentences: List[str]) -> torch.Tensor:
        """Per-sentence cross-entropy using the BASE compressor model.

        The base model (Llama-3.2-1B) is already loaded and warm in GPU
        cache, so this is fast.  We deliberately avoid the instruct model
        here — it is heavier and reserved for Stage 2 generation.
        """
        scores = []
        for sent in sentences:
            if len(sent.strip()) < 5:
                scores.append(0.0)
                continue
            inputs = self.tokenizer(
                sent, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, labels=inputs.input_ids)
                scores.append(outputs.loss.item())
        return torch.tensor(scores, device=self._device)

    def _semantic_select(self, sentences: List[str], budget: int) -> List[str]:
        """Select *budget* sentences by combined safety + entropy score.

        Returns the selected sentences in their **original order** so the
        compressed trace remains coherent.
        """
        if len(sentences) <= budget:
            return sentences

        safety_scores = self._get_semantic_safety_scores(sentences)
        entropy_raw = self._get_entropy_scores(sentences)

        # Normalize entropy to [0, 1]
        e_min, e_max = entropy_raw.min(), entropy_raw.max()
        entropy_norm = (entropy_raw - e_min) / (e_max - e_min + 1e-8)

        # Adaptive weighting: if no sentence looks harmful, rely on entropy
        # alone so benign traces don't get over-inflated safety scores.
        max_safety = safety_scores.max().item() if len(safety_scores) > 0 else 0
        if max_safety < 0.1:
            combined = entropy_norm
        else:
            s_min, s_max = safety_scores.min(), safety_scores.max()
            safety_norm = (safety_scores - s_min) / (s_max - s_min + 1e-8)
            combined = 0.85 * safety_norm + 0.15 * entropy_norm

        top_indices = torch.topk(combined, budget).indices.sort().values
        return [sentences[i] for i in top_indices.tolist()]

    # ------------------------------------------------------------------
    # Stage 2 — LLM smoothing (paraphrase rewrite, lazy instruct load)
    # ------------------------------------------------------------------
    def _smooth_with_llm(self, text: str, target_tokens: int) -> str:
        """Rewrite *text* into fluent prose using the 1B-Instruct model.

        The instruct model is lazy-loaded via self._parent._get_instruct_model()
        on first Stage 2 call.  If compression_ratio < 0.3 this method is never
        called, so the instruct model is never loaded for light compression runs.

        Returns the raw Stage 1 text on any generation failure.
        """
        instruct_model, instruct_tokenizer = self._parent._get_instruct_model()
        is_instruct = self._parent._is_instruct_model

        if is_instruct:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the text below into fluent prose. "
                        "Keep every fact, tool call, and action. "
                        "Do not add information."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Rewrite in about {target_tokens} tokens:\n\n{text}",
                },
            ]
            formatted_prompt = instruct_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for base models (structured completion prompt)
            formatted_prompt = (
                f"Rewrite the following into fluent prose in about "
                f"{target_tokens} tokens. Keep all facts and tool calls.\n\n"
                f"TEXT:\n{text}\n\n"
                "REWRITTEN:\n"
            )

        prompt_inputs = instruct_tokenizer(
            formatted_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(instruct_model.device)

        pad_id = (
            instruct_tokenizer.pad_token_id
            if instruct_tokenizer.pad_token_id is not None
            else instruct_tokenizer.eos_token_id
        )

        with torch.no_grad():
            output_ids = instruct_model.generate(
                prompt_inputs.input_ids,
                max_new_tokens=target_tokens,
                min_new_tokens=max(16, target_tokens // 4),
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=pad_id,
            )

        prompt_len = prompt_inputs.input_ids.shape[1]
        generated = instruct_tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()

        return generated

    # ------------------------------------------------------------------
    # Quality gate — detect degenerate outputs
    # ------------------------------------------------------------------
    @staticmethod
    def _is_degenerate(text: str) -> bool:
        """Return True if *text* is too short or heavily repetitive."""
        words = text.lower().split()
        if len(words) < 10:
            return True
        if len(words) > 20:
            trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if unique_ratio < 0.5:
                return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compress(self, text: str, compression_ratio: float = 0.5) -> str:
        """Two-stage compression: Extract critical elements, then smooth.

        Args:
            text: The full agent trace string.
            compression_ratio: Fraction of tokens to *remove* (0.5 = keep 50%).

        Returns:
            Compressed trace as a single string.
        """
        # === Stage 1: Extractive ===
        logger.debug("[TwoStage] Stage 1: extractive selection")
        critical, remaining = self._extract_critical_elements(text)

        # Token budget calculations
        original_tokens = len(self.tokenizer.encode(text))
        target_tokens = max(64, int(original_tokens * (1 - compression_ratio)))

        critical_text = "\n".join(critical)
        critical_tokens = len(self.tokenizer.encode(critical_text)) if critical else 0

        # Allocate leftover budget to semantically-selected remaining sentences
        remaining_budget_tokens = max(0, target_tokens - critical_tokens)

        selected_remaining: List[str] = []
        if remaining and remaining_budget_tokens > 0:
            avg_toks = max(1, sum(
                len(self.tokenizer.encode(s)) for s in remaining
            ) // len(remaining))
            sentence_budget = max(1, remaining_budget_tokens // avg_toks)
            selected_remaining = self._semantic_select(remaining, sentence_budget)

        stage1_output = critical_text
        if selected_remaining:
            stage1_output += "\n" + "\n".join(selected_remaining)

        # === Stage 2: Abstractive Smoothing ===
        # Skip LLM if Stage 1 output is already short OR compression is light.
        # This prevents loading the instruct model at all for r < 0.3 runs.
        stage1_tokens = len(self.tokenizer.encode(stage1_output))
        if stage1_tokens < 150 or compression_ratio < 0.3:
            logger.debug(
                f"[TwoStage] Stage 2 skipped "
                f"(stage1_tokens={stage1_tokens}, ratio={compression_ratio:.2f})"
            )
            return stage1_output

        logger.debug("[TwoStage] Stage 2: LLM smoothing")
        smooth_target = max(48, int(stage1_tokens * 0.8))  # trim another ~20%
        generated = self._smooth_with_llm(stage1_output, smooth_target)

        if self._is_degenerate(generated):
            logger.warning(
                "[TwoStage] Smoothing produced degenerate output, "
                "falling back to Stage 1 result"
            )
            return stage1_output

        return generated
