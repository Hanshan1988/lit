"""Dataset classes for HF sequence salience demo.

Three ways to load data:

    PromptDataset        – hardcoded list of {prompt, target?} dicts.
    JSONLDataset         – load from a .jsonl file (one JSON object per line).
    PlaintextDataset     – load from a plain .txt file (one prompt per line).

Each example has at least a "prompt" field; "target" is optional and is used
as the reference output when computing salience.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types


PROMPT = "prompt"
TARGET = "target"

_SPEC: lit_types.Spec = {
    PROMPT: lit_types.TextSegment(),
    TARGET: lit_types.TextSegment(required=False),
}

# ---------------------------------------------------------------------------
# Default sample prompts bundled with this example
# ---------------------------------------------------------------------------

_SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_prompts.jsonl")

SAMPLE_PROMPTS: list[dict] = [
    {
        "prompt": (
            "Summarize the following article in one sentence.\n\n"
            "Article: A new study published in Nature found that regular exercise "
            "reduces the risk of developing Alzheimer's disease by up to 40%. "
            "Researchers tracked 5,000 participants over ten years and found a "
            "clear correlation between physical activity levels and cognitive health.\n\n"
            "Summary:"
        ),
        "target": " Regular exercise can significantly reduce the risk of Alzheimer's disease.",
    },
    {
        "prompt": (
            "Classify the sentiment of the following review as Positive, Negative, or Neutral.\n\n"
            "Review: The food was excellent but the service was painfully slow.\n\n"
            "Sentiment:"
        ),
        "target": " Neutral",
    },
    {
        "prompt": (
            "Translate the following sentence to French.\n\n"
            "Sentence: The quick brown fox jumps over the lazy dog.\n\n"
            "Translation:"
        ),
        "target": " Le rapide renard brun saute par-dessus le chien paresseux.",
    },
    {
        "prompt": (
            "Answer the following question about basic arithmetic.\n\n"
            "Question: If a train travels at 60 mph for 2.5 hours, how far does it travel?\n\n"
            "Answer:"
        ),
        "target": " The train travels 150 miles.",
    },
    {
        "prompt": (
            "Continue the following story in 2-3 sentences.\n\n"
            "Story: The astronaut floated weightlessly through the capsule, "
            "watching Earth grow smaller through the porthole. She had trained "
            "for this moment for fifteen years.\n\n"
            "Continuation:"
        ),
        "target": "",
    },
    {
        "prompt": "What is the capital of France?",
        "target": " The capital of France is Paris.",
    },
    {
        "prompt": "Explain the concept of gradient descent in one paragraph.",
        "target": "",
    },
]


class PromptDataset(lit_dataset.Dataset):
    """In-memory prompt dataset."""

    def __init__(
        self,
        examples: Optional[list[dict]] = None,
        max_examples: Optional[int] = None,
    ):
        data = examples if examples is not None else SAMPLE_PROMPTS
        self._examples = data[:max_examples] if max_examples else data

    def spec(self) -> lit_types.Spec:
        return _SPEC


class JSONLDataset(lit_dataset.Dataset):
    """Load prompts from a JSON-Lines file.

    Each line must be a JSON object with at least a "prompt" key.
    An optional "target" key provides the reference output.
    """

    def __init__(self, path: str, max_examples: Optional[int] = None):
        self._examples = self._load(path, max_examples)

    @classmethod
    def init_spec(cls) -> lit_types.Spec:
        return {
            "path": lit_types.String(default=""),
            "max_examples": lit_types.Integer(
                default=1000, min_val=1, max_val=100_000, required=False
            ),
        }

    def _load(self, path: str, max_examples: Optional[int]) -> list[dict]:
        examples: list[dict] = []
        defaults = {PROMPT: "", TARGET: ""}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                examples.append({**defaults, **obj})
                if max_examples and len(examples) >= max_examples:
                    break
        return examples

    def load(self, path: str) -> "lit_dataset.Dataset":
        return lit_dataset.Dataset(
            base=self, examples=self._load(path, max_examples=None)
        )

    def spec(self) -> lit_types.Spec:
        return _SPEC


class PlaintextDataset(lit_dataset.Dataset):
    """Load one prompt per line from a plain text file."""

    def __init__(self, path: str, max_examples: Optional[int] = None):
        self._examples = self._load(path, max_examples)

    @classmethod
    def init_spec(cls) -> lit_types.Spec:
        return {
            "path": lit_types.String(default=""),
            "max_examples": lit_types.Integer(
                default=1000, min_val=1, max_val=100_000, required=False
            ),
        }

    def _load(self, path: str, max_examples: Optional[int]) -> list[dict]:
        examples: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                examples.append({PROMPT: line, TARGET: ""})
                if max_examples and len(examples) >= max_examples:
                    break
        return examples

    def load(self, path: str) -> "lit_dataset.Dataset":
        return lit_dataset.Dataset(
            base=self, examples=self._load(path, max_examples=None)
        )

    def spec(self) -> lit_types.Spec:
        return _SPEC
