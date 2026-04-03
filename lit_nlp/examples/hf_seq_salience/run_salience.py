r"""CLI script: compute sequence salience and print results as text.

No server, no UI — just load a model, generate a response, and print
per-token salience scores for every response token back to every context token.

USAGE
-----

Basic (GPT-2, CPU, built-in sample prompts):

    python -m lit_nlp.examples.hf_seq_salience.run_salience

With a specific model and prompt:

    python -m lit_nlp.examples.hf_seq_salience.run_salience \
        --model_id=Qwen/Qwen3-4B-Instruct-2507 \
        --prompt="Explain gradient descent in one paragraph." \
        --precision=bfloat16

With a pre-written target (skip generation, explain the reference text):

    python -m lit_nlp.examples.hf_seq_salience.run_salience \
        --model_id=gpt2 \
        --prompt="The quick brown fox" \
        --target=" jumps over the lazy dog"

From a JSONL file (one {"prompt": ..., "target": ...} per line):

    python -m lit_nlp.examples.hf_seq_salience.run_salience \
        --model_id=gpt2 \
        --dataset=lit_nlp/examples/hf_seq_salience/sample_prompts.jsonl \
        --max_examples=3

FLAGS
-----
  --model_id        HuggingFace model ID or local path.  Default: gpt2
  --prompt          A single prompt string (overrides --dataset).
  --target          Optional reference target; if omitted the model generates.
  --dataset         Path to a .jsonl or .txt file of prompts.
  --max_examples    Max examples to process from --dataset.
  --max_new_tokens  Max tokens to generate.  Default: 128
  --precision       bfloat16 (default) or float32.
  --device          PyTorch device string.  Auto-detected.
  --trust_remote_code  Pass trust_remote_code=True to HF.  Default: True.
  --method          Salience method: grad_l2 (default) or grad_dot_input.
  --top_k           Show only the top-k context tokens per target token (0=all).
  --output          Path to write results as JSON Lines (optional).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

_MODEL_ID = flags.DEFINE_string("model_id", "gpt2", "HuggingFace model ID.")
_PROMPT = flags.DEFINE_string("prompt", None, "Single prompt string.")
_TARGET = flags.DEFINE_string(
    "target", None,
    "Optional reference target. If omitted the model generates a response."
)
_DATASET = flags.DEFINE_string(
    "dataset", None,
    "Path to a .jsonl or .txt file. Ignored when --prompt is set."
)
_MAX_EXAMPLES = flags.DEFINE_integer("max_examples", 10, "Max examples.")
_MAX_NEW_TOKENS = flags.DEFINE_integer("max_new_tokens", 128, "Max tokens to generate.")
_PRECISION = flags.DEFINE_enum("precision", "bfloat16", ["bfloat16", "float32"], "dtype.")
_DEVICE = flags.DEFINE_string("device", None, "PyTorch device (auto if unset).")
_TRUST_REMOTE_CODE = flags.DEFINE_boolean("trust_remote_code", True, "HF trust_remote_code.")
_METHOD = flags.DEFINE_enum(
    "method", "grad_l2", ["grad_l2", "grad_dot_input"],
    "Salience method. grad_l2 = unsigned gradient norm, "
    "grad_dot_input = signed gradient · embedding."
)
_TOP_K = flags.DEFINE_integer(
    "top_k", 10,
    "Show only top-k context tokens per target token. 0 = show all."
)
_OUTPUT = flags.DEFINE_string(
    "output", None,
    "Optional path to write results as a JSON Lines file."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_examples(dataset_path: str, max_examples: int) -> list[dict]:
    """Load examples from a .jsonl or .txt file."""
    examples = []
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                examples.append({
                    "prompt": obj.get("prompt", ""),
                    "target": obj.get("target", ""),
                })
                if len(examples) >= max_examples:
                    break
    else:
        with open(dataset_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                examples.append({"prompt": line, "target": ""})
                if len(examples) >= max_examples:
                    break
    return examples


def _bar(score: float, width: int = 20, signed: bool = False) -> str:
    """Render a simple ASCII bar for a salience score."""
    if signed:
        # Centre-out bar: positive goes right, negative goes left.
        half = width // 2
        norm = max(-1.0, min(1.0, score))
        if norm >= 0:
            filled = round(norm * half)
            return " " * half + "█" * filled + "░" * (half - filled)
        else:
            filled = round(-norm * half)
            return "░" * (half - filled) + "█" * filled + " " * half
    else:
        norm = max(0.0, min(1.0, score))
        filled = round(norm * width)
        return "█" * filled + "░" * (width - filled)


def _render_token(tok: str) -> str:
    """Make a token printable (replace magic underscore / newlines)."""
    return tok.replace("▁", " ").replace("\n", "↵")


def _print_salience(
    example_idx: int,
    prompt: str,
    target_text: str,
    response: str,
    tokens: list[str],
    prompt_token_count: int,
    salience_scores: list[float],  # per-token scores (len = len(tokens))
    target_span: tuple[int, int],
    method: str,
    top_k: int,
) -> dict:
    """Pretty-print salience for one target span; return a result dict."""
    signed = method == "grad_dot_input"
    target_tokens = tokens[target_span[0]: target_span[1]]
    context_tokens = tokens[: target_span[0]]
    context_scores = salience_scores[: target_span[0]]

    # Normalise scores for display.
    abs_scores = [abs(s) for s in context_scores]
    max_abs = max(abs_scores) if abs_scores else 1.0
    if max_abs == 0:
        max_abs = 1.0
    norm_scores = [s / max_abs for s in context_scores]

    # Sort by absolute value for top-k.
    indexed = sorted(
        enumerate(norm_scores), key=lambda x: abs(x[1]), reverse=True
    )
    if top_k > 0:
        top_indices = {i for i, _ in indexed[:top_k]}
    else:
        top_indices = set(range(len(context_tokens)))

    target_str = "".join(_render_token(t) for t in target_tokens).strip()

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Example {example_idx + 1}  │  target token(s): «{target_str}»")
    print(sep)
    print(f"  {'TOKEN':<22} {'SCORE':>9}  {'BAR'}")
    print(f"  {'─'*22} {'─'*9}  {'─'*20}")

    result_tokens = []
    for i, (tok, raw_score, norm) in enumerate(
        zip(context_tokens, context_scores, norm_scores)
    ):
        if i not in top_indices:
            continue
        display = _render_token(tok)
        bar = _bar(norm, width=20, signed=signed)
        # Is the token in the prompt or in the prior generated tokens?
        region = "prompt" if i < prompt_token_count else "prev_gen"
        print(f"  {display:<22} {raw_score:>9.4f}  {bar}  [{region}]")
        result_tokens.append({"token": tok, "score": raw_score, "region": region})

    result = {
        "example_idx": example_idx,
        "prompt": prompt,
        "target": target_text,
        "response": response,
        "target_tokens": target_tokens,
        "method": method,
        "context_salience": result_tokens,
    }
    return result


# ---------------------------------------------------------------------------
# Core pipeline (pure Python, no server)
# ---------------------------------------------------------------------------

def run_pipeline(
    model_id: str,
    examples: list[dict],
    max_new_tokens: int = 128,
    precision: str = "bfloat16",
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    method: str = "grad_l2",
    top_k: int = 10,
    output_path: Optional[str] = None,
) -> list[dict]:
    """Load the model, run generation + salience, and print/return results.

    Args:
        model_id:          HuggingFace model ID or local path.
        examples:          List of dicts with "prompt" and "target" keys.
        max_new_tokens:    Max tokens to generate per prompt.
        precision:         "bfloat16" or "float32".
        device:            PyTorch device string (auto if None).
        trust_remote_code: Pass trust_remote_code=True to HF.
        method:            "grad_l2" or "grad_dot_input".
        top_k:             Show only top-k context tokens per target token
                           (0 = all).
        output_path:       Optional .jsonl path for saving structured results.

    Returns:
        List of result dicts, one per (example, target_token).
    """
    # Import here so absl flag parsing has already happened.
    from lit_nlp.examples.hf_seq_salience.backend.model import (
        HFGenerativeModel, HFSalienceModel, HFTokenizerModel,
        PROMPT, TARGET, RESPONSE, TOKENS, TARGET_MASK, GRAD_NORM, GRAD_DOT_INPUT,
    )

    print(f"\n{'='*72}")
    print(f"  Model : {model_id}")
    print(f"  Method: {method}  |  top_k: {top_k if top_k > 0 else 'all'}")
    print(f"  Device: {device or 'auto'}  |  dtype: {precision}")
    print(f"{'='*72}\n")

    # ── Load model (shared weights across all three wrappers) ──────────────
    logging.info("Loading model '%s'…", model_id)
    gen_model = HFGenerativeModel(
        model_id=model_id,
        batch_size=1,
        max_new_tokens=max_new_tokens,
        precision=precision,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    sal_model = HFSalienceModel.from_loaded(gen_model, batch_size=1)
    tok_model = HFTokenizerModel.from_loaded(gen_model, batch_size=1)

    all_results: list[dict] = []
    out_file = open(output_path, "w", encoding="utf-8") if output_path else None

    try:
        for ex_idx, ex in enumerate(examples):
            prompt = ex["prompt"]
            ref_target = ex.get("target", "").strip()

            print(f"\n{'━'*72}")
            print(f" Example {ex_idx + 1}/{len(examples)}")
            print(f"{'━'*72}")
            print(f" Prompt: {prompt[:120]}{'…' if len(prompt) > 120 else ''}")

            # ── Step 1: Generate (or use provided target) ──────────────────
            if ref_target:
                response = ref_target
                print(f" Target: {response[:120]}{'…' if len(response) > 120 else ''}")
            else:
                print(" Generating response…")
                gen_preds = list(gen_model.predict([{PROMPT: prompt, TARGET: ""}]))
                response = gen_preds[0].get(RESPONSE, "")
                print(f" Response: {response[:120]}{'…' if len(response) > 120 else ''}")

            if not response.strip():
                print(" (empty response — skipping)")
                continue

            # ── Step 2: Tokenise prompt+response ──────────────────────────
            tok_preds = list(tok_model.predict([{PROMPT: prompt, TARGET: response}]))
            all_tokens: list[str] = tok_preds[0].get(TOKENS, [])

            # Tokenise prompt alone to find the boundary.
            tok_preds_prompt = list(tok_model.predict([{PROMPT: prompt, TARGET: ""}]))
            prompt_token_count = len(tok_preds_prompt[0].get(TOKENS, []))

            response_token_count = len(all_tokens) - prompt_token_count
            print(
                f" Tokens: {len(all_tokens)} total"
                f" ({prompt_token_count} prompt + {response_token_count} response)"
            )

            if response_token_count == 0:
                print(" No response tokens found — skipping.")
                continue

            # ── Step 3: Compute salience for every response token ──────────
            print(f" Computing {method} salience for each response token…\n")

            for resp_tok_idx in range(prompt_token_count, len(all_tokens)):
                target_mask = [
                    1 if i == resp_tok_idx else 0
                    for i in range(len(all_tokens))
                ]

                sal_preds = list(sal_model.predict([{
                    PROMPT: prompt,
                    TARGET: response,
                    TARGET_MASK: target_mask,
                }]))
                pred = sal_preds[0]
                scores: list[float] = list(
                    pred.get(method, pred.get(GRAD_NORM, []))
                )

                # sal_model returns tokens that match all_tokens after masking
                # padding — check lengths match.
                sal_tokens: list[str] = pred.get(TOKENS, all_tokens)
                if len(scores) != len(sal_tokens):
                    logging.warning(
                        "Score/token length mismatch (%d vs %d) at token %d",
                        len(scores), len(sal_tokens), resp_tok_idx,
                    )
                    continue

                result = _print_salience(
                    example_idx=ex_idx,
                    prompt=prompt,
                    target_text=response,
                    response=response,
                    tokens=sal_tokens,
                    prompt_token_count=prompt_token_count,
                    salience_scores=scores,
                    target_span=(resp_tok_idx, resp_tok_idx + 1),
                    method=method,
                    top_k=top_k,
                )
                all_results.append(result)

                if out_file:
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_file.flush()

    finally:
        if out_file:
            out_file.close()
            print(f"\nResults written to: {output_path}")

    return all_results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Unexpected arguments: " + str(argv[1:]))

    # Build example list.
    if _PROMPT.value:
        examples = [{"prompt": _PROMPT.value, "target": _TARGET.value or ""}]
    elif _DATASET.value:
        examples = _load_examples(_DATASET.value, _MAX_EXAMPLES.value)
        if not examples:
            print(f"No examples loaded from {_DATASET.value}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default: a handful of built-in prompts.
        _sample_path = os.path.join(
            os.path.dirname(__file__), "sample_prompts.jsonl"
        )
        examples = _load_examples(_sample_path, _MAX_EXAMPLES.value)

    run_pipeline(
        model_id=_MODEL_ID.value,
        examples=examples,
        max_new_tokens=_MAX_NEW_TOKENS.value,
        precision=_PRECISION.value,
        device=_DEVICE.value,
        trust_remote_code=_TRUST_REMOTE_CODE.value,
        method=_METHOD.value,
        top_k=_TOP_K.value,
        output_path=_OUTPUT.value,
    )


if __name__ == "__main__":
    app.run(main)
