r"""LIT server for sequence salience with any HuggingFace CausalLM.

Quick-start
-----------

Run with a pre-trained model from the HuggingFace Hub::

    python -m lit_nlp.examples.hf_seq_salience.backend.server \
        --model_id=Qwen/Qwen3-4B-Instruct-2507 \
        --port=5432

Then open http://localhost:5432 in your browser.
Select an example, click any generated/output token to set it as the salience
target, and the preceding tokens will be coloured by their influence on that
prediction.

Flags
-----

  --model_id            HuggingFace model ID or local path.
                        Default: "gpt2"
  --model_name          Display name used in the LIT UI.
                        Default: last component of --model_id.
  --precision           "bfloat16" (default) or "float32".
  --batch_size          Per-forward-pass batch size. Reduce if OOM.
  --max_new_tokens      Maximum tokens the model may generate.
  --layout              UI layout: "salience_only" (default), "simple", "full".
  --dataset             Path to a .jsonl or .txt file, or "sample" for the
                        built-in examples.  Default: "sample".
  --max_examples        Cap on dataset size.
  --port                HTTP port.  Default: 5432.
  --device              PyTorch device string, e.g. "cuda", "cuda:1", "cpu".
                        Auto-detected by default.
"""

import importlib
import pathlib
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.hf_seq_salience.backend import dataset as ds_lib
from lit_nlp.examples.hf_seq_salience.backend import layout as layout_lib
from lit_nlp.examples.hf_seq_salience.backend import model as model_lib

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

_MODEL_ID = flags.DEFINE_string(
    "model_id",
    "gpt2",
    "HuggingFace model ID or local path to load "
    "(e.g. Qwen/Qwen3-4B-Instruct-2507).",
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    None,
    "Display name shown in the LIT UI. Defaults to the last path component of "
    "--model_id.",
)
_PRECISION = flags.DEFINE_enum(
    "precision",
    "bfloat16",
    ["bfloat16", "float32"],
    "Floating-point precision used when loading the model.",
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    4,
    "Number of examples to process per forward pass. Reduce if OOM.",
    lower_bound=1,
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    256,
    "Maximum number of tokens to generate per prompt.",
    lower_bound=1,
)
_DEVICE = flags.DEFINE_string(
    "device",
    None,
    "PyTorch device string (e.g. 'cuda', 'cuda:1', 'cpu'). "
    "Auto-detected by default.",
)
_TRUST_REMOTE_CODE = flags.DEFINE_boolean(
    "trust_remote_code",
    True,
    "Set trust_remote_code=True when loading the HF model. "
    "Required for models like Qwen that ship custom modelling code.",
)
_LAYOUT = flags.DEFINE_enum(
    "layout",
    layout_lib.DEFAULT_LAYOUT,
    list(layout_lib.LAYOUTS.keys()),
    "Initial UI layout.",
)
_DATASET = flags.DEFINE_string(
    "dataset",
    "sample",
    "Dataset to load.  Use 'sample' for the built-in prompts, or supply a "
    "path to a .jsonl or .txt file.",
)
_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples",
    1000,
    "Maximum number of examples to load from the dataset.",
    lower_bound=1,
)

# Apply LIT defaults.
_FLAGS = flags.FLAGS
_FLAGS.set_default("development_demo", True)
_FLAGS.set_default("page_title", "HF Sequence Salience")
_FLAGS.set_default("default_layout", layout_lib.DEFAULT_LAYOUT)

_SPLASH_SCREEN_DOC = """
# Sequence Salience

Select an example from the data table, then click one or more **output tokens**
in the Salience panel to choose your target.  The preceding context tokens will
be highlighted according to how much they influenced the model's prediction of
the selected target token(s).

Darker colours indicate higher gradient salience.
"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _find_client_root() -> Optional[str]:
    """Locate the compiled LIT frontend.

    Checks two locations in order:
      1. The local source-tree build (``lit_nlp/client/build/default``).
      2. The build shipped inside any pip-installed ``lit-nlp`` package.

    Returns the first valid directory found, or ``None`` if neither exists.
    """
    # 1. Local source build (only exists if the user ran `yarn build`).
    # server.py lives at lit_nlp/examples/hf_seq_salience/backend/server.py
    # parents[3] is lit_nlp/
    src_root = (
        pathlib.Path(__file__).resolve().parents[3]
        / "client" / "build" / "default"
    )
    if src_root.is_dir():
        return str(src_root)

    # 2. pip-installed package (ships with a pre-built client).
    try:
        lit_mod = importlib.import_module("lit_nlp")
        pkg_build = pathlib.Path(lit_mod.__file__).parent / "client" / "build" / "default"
        if pkg_build.is_dir():
            logging.info(
                "Using pre-built client from pip package at %s", pkg_build
            )
            return str(pkg_build)
    except Exception:  # pylint: disable=broad-except
        pass

    return None


def _get_model_name(model_id: str, model_name: Optional[str]) -> str:
    if model_name:
        return model_name
    # Use the last path component (handles both Hub IDs and local paths).
    return model_id.rstrip("/").split("/")[-1]


def _load_dataset(spec: str, max_examples: int) -> dict:
    """Load dataset from flag value."""
    if spec == "sample":
        return {"sample_prompts": ds_lib.PromptDataset(max_examples=max_examples)}
    if spec.endswith(".jsonl"):
        return {"prompts": ds_lib.JSONLDataset(spec, max_examples=max_examples)}
    # Fall back to plain-text
    return {"prompts": ds_lib.PlaintextDataset(spec, max_examples=max_examples)}


# ---------------------------------------------------------------------------
# WSGI entry point (for gunicorn / container deployments)
# ---------------------------------------------------------------------------


def get_wsgi_app():
    """Return WSGI app without starting a server (for external WSGI runners)."""
    _FLAGS.set_default("server_type", "external")
    _FLAGS.set_default("demo_mode", True)
    unused = _FLAGS(sys.argv, known_only=True)
    if unused:
        logging.info("get_wsgi_app() called with unused args: %s", unused)
    return main([])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Auto-detect a built frontend if the default client_root doesn't exist.
    server_kw = server_flags.get_flags()
    current_root = server_kw.get("client_root", "")
    if not current_root or not pathlib.Path(current_root).is_dir():
        detected = _find_client_root()
        if detected:
            server_kw["client_root"] = detected
            logging.info("Using client_root: %s", detected)
        else:
            logging.error(
                "Could not find a built LIT frontend client.\n"
                "Run `pip install lit-nlp` for a pre-built client, or build\n"
                "from source: cd lit_nlp && yarn && yarn build"
            )

    model_name = _get_model_name(_MODEL_ID.value, _MODEL_NAME.value)
    logging.info(
        "Loading model '%s' from '%s' …", model_name, _MODEL_ID.value
    )

    _gen_name, _sal_name, _tok_name, model_map = model_lib.make_model_group(
        name=model_name,
        model_id=_MODEL_ID.value,
        batch_size=_BATCH_SIZE.value,
        max_new_tokens=_MAX_NEW_TOKENS.value,
        precision=_PRECISION.value,
        trust_remote_code=_TRUST_REMOTE_CODE.value,
        device=_DEVICE.value,
    )

    datasets = _load_dataset(_DATASET.value, _MAX_EXAMPLES.value)

    lit_demo = dev_server.Server(
        models=model_map,
        datasets=datasets,
        layouts=layout_lib.LAYOUTS,
        onboard_start_doc=_SPLASH_SCREEN_DOC,
        **server_kw,
    )
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
