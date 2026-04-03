"""HuggingFace CausalLM wrappers for LIT sequence salience.

Three models are created for each HuggingFace model ID:

  * <name>                 – generative model (runs model.generate)
  * _<name>_salience       – gradient-based token salience via input embeddings
  * _<name>_tokenizer      – tokenizer-only view (no forward pass)

All three share the same underlying HuggingFace model/tokenizer weights so
only one copy is loaded into GPU memory.

Salience method
---------------
Gradient × Input (and L2-norm of gradients) are computed by:
  1. Embedding the input tokens.
  2. Running a forward pass through the causal LM with requires_grad=True on
     the embeddings.
  3. Computing per-token cross-entropy loss weighted by the *target_mask* that
     marks which output tokens the user wants explained.
  4. Back-propagating to obtain ∂loss/∂emb for each input position.
  5. Returning ||∂loss/∂emb||₂  (grad_l2) and  Σ(∂loss/∂emb · emb)  (grad_dot_input).

Usage::

    gen, sal, tok = make_model_group("mymodel", "Qwen/Qwen3-4B-Instruct-2507")
    models = {"mymodel": gen, "_mymodel_salience": sal, "_mymodel_tokenizer": tok}
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, Mapping, Optional

from absl import logging
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
import numpy as np
import torch
import transformers

# ---------------------------------------------------------------------------
# Field name constants
# ---------------------------------------------------------------------------

PROMPT = "prompt"
TARGET = "target"
RESPONSE = "response"
TOKENS = "tokens"
TARGET_MASK = "target_mask"
GRAD_DOT_INPUT = "grad_dot_input"
GRAD_NORM = "grad_l2"
PROMPT_EMBEDDINGS = "prompt_embeddings"
RESPONSE_EMBEDDINGS = "response_embeddings"

# ---------------------------------------------------------------------------
# LIT specs shared across model classes
# ---------------------------------------------------------------------------

INPUT_SPEC: lit_types.Spec = {
    PROMPT: lit_types.TextSegment(),
    TARGET: lit_types.TextSegment(required=False),
}

INPUT_SPEC_SALIENCE: lit_types.Spec = {
    TARGET_MASK: lit_types.TokenScores(align="", required=False),
}

OUTPUT_SPEC_GENERATION: lit_types.Spec = {
    RESPONSE: lit_types.GeneratedText(parent=TARGET),
    PROMPT_EMBEDDINGS: lit_types.Embeddings(required=False),
    RESPONSE_EMBEDDINGS: lit_types.Embeddings(required=False),
}

OUTPUT_SPEC_SALIENCE: lit_types.Spec = {
    GRAD_DOT_INPUT: lit_types.TokenScores(align=TOKENS),
    GRAD_NORM: lit_types.TokenScores(align=TOKENS),
    TOKENS: lit_types.Tokens(parent=""),
}

OUTPUT_SPEC_TOKENIZER: lit_types.Spec = {
    TOKENS: lit_types.Tokens(parent=""),
}


def _select_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFBaseModel(lit_model.BatchedModel):
    """Shared HuggingFace model/tokenizer loader for PyTorch."""

    @classmethod
    def init_spec(cls) -> lit_model.Spec:
        return {
            "model_id": lit_types.String(default="gpt2"),
            "batch_size": lit_types.Integer(default=4, min_val=1, max_val=32),
            "trust_remote_code": lit_types.Boolean(default=True),
        }

    def __init__(
        self,
        model_id: str = "gpt2",
        batch_size: int = 4,
        trust_remote_code: bool = True,
        precision: str = "bfloat16",
        # Accept pre-built objects so that from_loaded() can share weights
        model: Optional[transformers.PreTrainedModel] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.device = device if device is not None else _select_device()
        logging.info("Using device: %s", self.device)

        if model is not None and tokenizer is not None:
            # Reuse pre-loaded objects
            self.model = model
            self.tokenizer = tokenizer
        else:
            _dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
            logging.info(
                "Loading '%s' in %s on %s …", model_id, precision, self.device
            )
            # Left-padding is required for batched generation with decoder-only LMs.
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
                trust_remote_code=trust_remote_code,
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=_dtype,
                output_hidden_states=True,
                output_attentions=False,
                trust_remote_code=trust_remote_code,
            ).to(self.device)
            self.model.eval()

        # Ensure pad token exists (many causal LMs don't define one).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.embedding_table = self.model.get_input_embeddings()
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Class method: share weights across subclass instances
    # ------------------------------------------------------------------

    @classmethod
    def from_loaded(cls, existing: "HFBaseModel", **kw) -> "HFBaseModel":
        """Create a new instance sharing the model/tokenizer of *existing*."""
        return cls(
            model=existing.model,
            tokenizer=existing.tokenizer,
            device=existing.device,
            **kw,
        )

    # ------------------------------------------------------------------
    # BatchedModel interface
    # ------------------------------------------------------------------

    def max_minibatch_size(self) -> int:
        return self._batch_size

    def input_spec(self) -> lit_types.Spec:
        return INPUT_SPEC

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _clean_token(self, tok: str) -> str:
        """Normalise subword tokens to human-readable form."""
        tok = tok.replace("Ċ", "\n")   # GPT-2: newline
        tok = tok.replace("Ġ", "▁")   # GPT-2: word-start space
        tok = tok.replace("<0x0A>", "\n")  # SentencePiece newline
        return tok

    def ids_to_clean_tokens(self, ids: Sequence[int]) -> list[str]:
        raw = self.tokenizer.convert_ids_to_tokens(ids)
        return [self._clean_token(t) for t in raw]


# ---------------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------------


class HFGenerativeModel(HFBaseModel):
    """Runs causal LM generation and returns the generated response + embeddings."""

    @classmethod
    def init_spec(cls) -> lit_model.Spec:
        return super().init_spec() | {
            "max_new_tokens": lit_types.Integer(default=256, min_val=1, max_val=2048),
        }

    def __init__(self, *args, max_new_tokens: int = 256, **kw):
        super().__init__(*args, **kw)
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------

    def _postprocess(self, preds: dict[str, Any]) -> dict[str, Any]:
        ntok_in = preds["ntok_in"]
        ntok_out = preds["ntok_out"]
        embs = preds["embs"]          # (total_tokens, hidden)
        result = {RESPONSE: preds[RESPONSE]}
        if embs.shape[0] >= ntok_in + ntok_out and ntok_out > 0:
            result[PROMPT_EMBEDDINGS] = np.mean(
                embs[-(ntok_in + ntok_out):-ntok_out], axis=0
            )
            result[RESPONSE_EMBEDDINGS] = np.mean(embs[-ntok_out:], axis=0)
        return result

    def predict_minibatch(self, inputs: Sequence[Mapping[str, Any]]):
        prompts = [ex[PROMPT] for ex in inputs]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=True,
        ).to(self.device)

        ntok_in = enc["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        ntok_out = output_ids.shape[1] - ntok_in
        responses = self.tokenizer.batch_decode(
            output_ids[:, ntok_in:], skip_special_tokens=True
        )

        with torch.no_grad():
            embs_tensor = self.embedding_table(output_ids).cpu().float()

        embs_np = embs_tensor.numpy()  # (batch, total_tokens, hidden_dim)
        batch_size = len(inputs)

        batched = {
            "embs": embs_np,
            "ntok_in": ntok_in,
            "ntok_out": np.full((batch_size,), ntok_out),
            RESPONSE: responses,
        }
        unbatched = utils.unbatch_preds(batched)
        return map(self._postprocess, unbatched)

    def output_spec(self) -> lit_types.Spec:
        return OUTPUT_SPEC_GENERATION


# ---------------------------------------------------------------------------
# Salience model
# ---------------------------------------------------------------------------


class HFSalienceModel(HFBaseModel):
    """Computes gradient-based token salience for a causal LM (PyTorch)."""

    def _pad_target_masks(
        self,
        seq_length: int,
        target_masks: list[list[int]],
    ) -> np.ndarray:
        """Left-pad target masks to *seq_length*."""
        # Token 0 is never predicted, so zero its mask entry.
        corrected = [[0] + list(m[1:]) for m in target_masks]
        pad_fn = functools.partial(
            utils.pad1d,
            min_len=seq_length,
            max_len=seq_length,
            pad_val=0,
            pad_left=True,
        )
        return np.stack([pad_fn(m) for m in corrected], axis=0)

    def _forward_with_salience(
        self,
        enc: "transformers.BatchEncoding",
        target_masks: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        enc = enc.to(self.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        seq_len = input_ids.shape[1]

        # Shift input_ids by one to get target ids (next-token prediction).
        target_ids = torch.roll(input_ids, shifts=-1, dims=1)

        padded = self._pad_target_masks(seq_len, target_masks)
        loss_mask = torch.tensor(padded).bool()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).to(self.device)

        # Embeddings with gradient tracking.
        embs = self.embedding_table(input_ids)
        embs.retain_grad()

        outs = self.model(
            input_ids=None,
            inputs_embeds=embs,
            attention_mask=attention_mask,
        )

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        # logits: (batch, seq, vocab) → permute to (batch, vocab, seq)
        per_token_loss = loss_fn(outs.logits.permute(0, 2, 1), target_ids)
        masked_loss = per_token_loss * loss_mask.float()

        grads = torch.autograd.grad(
            masked_loss,
            embs,
            grad_outputs=torch.ones_like(masked_loss),
        )[0]  # (batch, seq, hidden)

        embs_detached = embs.detach()
        grad_l2 = torch.norm(grads, dim=2)                   # (batch, seq)
        grad_dot_input = torch.sum(grads * embs_detached, dim=2)  # (batch, seq)

        return {
            "input_ids": input_ids.cpu().int(),
            "attention_mask": attention_mask.cpu().int(),
            GRAD_NORM: grad_l2.cpu().float(),
            GRAD_DOT_INPUT: grad_dot_input.cpu().float(),
        }

    def _postprocess(self, preds: dict[str, Any]) -> dict[str, Any]:
        mask = preds.pop("attention_mask").astype(bool)
        ids = preds.pop("input_ids")[mask]
        preds[TOKENS] = self.ids_to_clean_tokens(ids)
        for key in utils.find_spec_keys(self.output_spec(), lit_types.TokenScores):
            preds[key] = preds[key][mask]
        return preds

    def predict_minibatch(self, inputs: Sequence[Mapping[str, Any]]):
        texts = [
            ex[PROMPT] + ex.get(TARGET, "") for ex in inputs
        ]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=True,
        )
        target_masks = [ex.get(TARGET_MASK, []) for ex in inputs]

        batched = self._forward_with_salience(enc, target_masks)
        np_batched = {k: v.numpy() for k, v in batched.items()}
        unbatched = utils.unbatch_preds(np_batched)
        return map(self._postprocess, unbatched)

    def input_spec(self) -> lit_types.Spec:
        return super().input_spec() | INPUT_SPEC_SALIENCE

    def output_spec(self) -> lit_types.Spec:
        return OUTPUT_SPEC_SALIENCE


# ---------------------------------------------------------------------------
# Tokenizer-only model
# ---------------------------------------------------------------------------


class HFTokenizerModel(HFBaseModel):
    """Runs only the tokenizer; returns tokens without any forward pass."""

    def _postprocess(self, preds: dict[str, Any]) -> dict[str, Any]:
        mask = preds.pop("attention_mask").astype(bool)
        ids = preds.pop("input_ids")[mask]
        preds[TOKENS] = self.ids_to_clean_tokens(ids)
        return preds

    def predict_minibatch(self, inputs: Sequence[Mapping[str, Any]]):
        texts = [
            ex[PROMPT] + ex.get(TARGET, "") for ex in inputs
        ]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=True,
        )
        batched = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        np_batched = {k: v.numpy() for k, v in batched.items()}
        unbatched = utils.unbatch_preds(np_batched)
        return map(self._postprocess, unbatched)

    def output_spec(self) -> lit_types.Spec:
        return OUTPUT_SPEC_TOKENIZER


# ---------------------------------------------------------------------------
# Factory: create all three model wrappers sharing one HF model
# ---------------------------------------------------------------------------


def make_model_group(
    name: str,
    model_id: str,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    precision: str = "bfloat16",
    trust_remote_code: bool = True,
    device: Optional[str] = None,
) -> tuple[str, str, str, "lit_model.ModelMap"]:
    """Load one HF model and create generative + salience + tokenizer wrappers.

    Args:
        name:              Human-readable model name visible in the LIT UI.
        model_id:          HuggingFace Hub model ID or local path.
        batch_size:        Forward-pass batch size. Reduce if OOM.
        max_new_tokens:    Maximum tokens to generate.
        precision:         "bfloat16" (default) or "float32".
        trust_remote_code: Required for models like Qwen that ship custom code.
        device:            Override device (e.g. "cuda:1"). Auto-detected by default.

    Returns:
        A four-tuple: (gen_name, salience_name, tokenizer_name, model_map).
        model_map is ready to pass to ``dev_server.Server(models=…)``.
    """
    salience_name = f"_{name}_salience"
    tokenizer_name = f"_{name}_tokenizer"

    logging.info("Initialising generative model '%s' …", name)
    gen_model = HFGenerativeModel(
        model_id=model_id,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        precision=precision,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    sal_model = HFSalienceModel.from_loaded(gen_model, batch_size=batch_size)
    tok_model = HFTokenizerModel.from_loaded(gen_model, batch_size=batch_size)

    model_map: lit_model.ModelMap = {
        name: gen_model,
        salience_name: sal_model,
        tokenizer_name: tok_model,
    }
    return name, salience_name, tokenizer_name, model_map
