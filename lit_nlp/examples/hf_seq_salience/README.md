# HuggingFace Sequence Salience

A self-contained LIT example for exploring **gradient-based sequence salience** with any HuggingFace `AutoModelForCausalLM` (e.g. GPT-2, Llama, Mistral, Qwen, Gemma…).

**Sequence salience** attributes each *generated* token back to the tokens that influenced it most in the context.  Two gradient signals are available:

| Method | Description |
|---|---|
| `grad_l2` | L2-norm of ∂loss/∂embedding — unsigned influence score |
| `grad_dot_input` | Gradient · input embedding — signed influence (positive = promotes the target, negative = suppresses it) |

---

## Quick start

### 1 · Install dependencies

```bash
# From the lit_nlp repo root
pip install -e .

# GPU (replace cu121 with your CUDA version, see pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Model dependencies
pip install transformers>=4.40 accelerate>=0.30 sentencepiece
```

### 2 · Start the server

```bash
# With any public HuggingFace model:
python -m lit_nlp.examples.hf_seq_salience.backend.server \
    --model_id=Qwen/Qwen3-4B-Instruct-2507 \
    --port=5432

# Minimal GPU-friendly defaults (bfloat16, batch_size=1):
python -m lit_nlp.examples.hf_seq_salience.backend.server \
    --model_id=Qwen/Qwen3-4B-Instruct-2507 \
    --batch_size=1 \
    --max_new_tokens=256 \
    --precision=bfloat16 \
    --port=5432

# GPT-2 (CPU-safe for testing):
python -m lit_nlp.examples.hf_seq_salience.backend.server \
    --model_id=gpt2 \
    --precision=float32 \
    --port=5432
```

### 3 · Open the UI

**Option A — LIT built-in UI** (full-featured):

Open [http://localhost:5432](http://localhost:5432) in your browser.

1. Select an example from the **Data Table** on the left.
2. The model will generate a response (shown in the **Datapoint Editor**).
3. In the **Sequence Salience** panel, click **"Select sequence"** and choose the model response.
4. Click any output token — context tokens will be coloured by their influence on that target.

**Option B — Standalone viewer** (lightweight, no build step):

```bash
# Serve the standalone HTML from this directory
python -m http.server 8080
```

Then open [http://localhost:8080/frontend/standalone_viewer.html](http://localhost:8080/frontend/standalone_viewer.html).

- Configure the server URL (default `http://localhost:5432`) and click **Connect**.
- Select an example from the sidebar, or type a prompt.
- Click **Generate**, then click any response token to see salience.

**Option C — CLI script** (headless, text output):

No UI needed — compute salience directly from the command line:

```bash
# Run with a single prompt
python -m lit_nlp.examples.hf_seq_salience.run_salience \
    --model_id=gpt2 \
    --prompt="Write a haiku:"

# Run with multiple examples from a dataset
python -m lit_nlp.examples.hf_seq_salience.run_salience \
    --model_id=gpt2 \
    --dataset=/path/to/prompts.jsonl \
    --max_examples=5

# Save results to JSON Lines for further processing
python -m lit_nlp.examples.hf_seq_salience.run_salience \
    --model_id=gpt2 \
    --max_examples=10 \
    --output=results.jsonl
```

Output shows each generated token with ranked context tokens and ASCII salience bars.

---

## CLI flags reference

### Server flags

| Flag | Default | Description |
|---|---|---|
| `--model_id` | `gpt2` | HuggingFace Hub ID or local path |
| `--model_name` | *(auto)* | Display name in the UI |
| `--precision` | `bfloat16` | `bfloat16` or `float32` |
| `--batch_size` | `4` | Forward-pass batch size (reduce if OOM) |
| `--max_new_tokens` | `256` | Max tokens to generate |
| `--device` | *(auto)* | PyTorch device, e.g. `cuda`, `cuda:1`, `cpu` |
| `--trust_remote_code` | `True` | Required for Qwen and similar models |
| `--layout` | `salience_only` | `salience_only`, `simple`, `full` |
| `--dataset` | `sample` | `sample` or path to `.jsonl` / `.txt` |
| `--max_examples` | `1000` | Cap dataset size |
| `--port` | `5432` | HTTP port |

### CLI script flags (`run_salience.py`)

| Flag | Default | Description |
|---|---|---|
| `--model_id` | `gpt2` | HuggingFace Hub ID or local path |
| `--prompt` | *(none)* | Single prompt to analyze (overrides `--dataset`) |
| `--target` | *(none)* | Target text (optional reference output) |
| `--dataset` | `sample` | Path to `.jsonl` / `.txt` file, or `sample` for built-in |
| `--max_examples` | `100` | Cap dataset size |
| `--max_new_tokens` | `256` | Max tokens to generate |
| `--precision` | `bfloat16` | `bfloat16` or `float32` |
| `--device` | *(auto)* | PyTorch device, e.g. `cuda`, `cuda:1`, `cpu` |
| `--trust_remote_code` | `True` | Required for Qwen and similar models |
| `--method` | `grad_l2` | `grad_l2` (unsigned) or `grad_dot_input` (signed) |
| `--top_k` | `10` | Show top K most influential context tokens per response token |
| `--output` | *(none)* | Optional output file path for JSON Lines results |

---

## File structure

```
hf_seq_salience/
├── README.md                          # this file
├── requirements.txt
├── sample_prompts.jsonl               # built-in example prompts
│
├── backend/
│   ├── __init__.py
│   ├── model.py          # HFGenerativeModel, HFSalienceModel, HFTokenizerModel
│   ├── dataset.py        # PromptDataset, JSONLDataset, PlaintextDataset
│   ├── layout.py         # LIT UI layout definitions
│   └── server.py         # Main server entry point (run this)
│
└── frontend/
    ├── sequence_salience_module.ts    # LIT TypeScript module (for LIT build)
    ├── sequence_salience_module.css   # CSS for the LIT module
    └── standalone_viewer.html        # Self-contained HTML+JS viewer
```

---

## Using your own dataset

Pass a `.jsonl` file where each line has at least a `"prompt"` key (and optionally a `"target"` key for reference outputs):

```bash
python -m lit_nlp.examples.hf_seq_salience.backend.server \
    --model_id=meta-llama/Llama-3-8B-Instruct \
    --dataset=/path/to/my/prompts.jsonl \
    --port=5432
```

Plain text files (one prompt per line) are also supported.

---

## Running the CLI script for batch salience computation

The `run_salience.py` script provides a command-line interface for computing salience without starting a server or UI. Perfect for batch processing or integration into pipelines.

### Basic usage

```bash
# Analyze a single prompt
python -m lit_nlp.examples.hf_seq_salience.run_salience \
    --prompt="Translate to French: Hello world"

# Process multiple examples from built-in samples
python -m lit_nlp.examples.hf_seq_salience.run_salience \
    --model_id=gpt2 \
    --max_examples=3

# Use your own dataset
python -m lit_nlp.examples.hf_seq_salience.run_salience \
    --model_id=gpt2 \
    --dataset=/path/to/my/prompts.jsonl \
    --max_examples=10
```

### Output format

The script prints human-readable output for each example:

```
─ Example 1: "Write a haiku about cats:" ─

[Response salience breakdown]

  TARGET TOKEN 1: "A"
  ┌─────────────────────────────────────────┐
  │ Context token: "haiku" (score: 0.89)    │
  │ Context token: "about" (score: 0.78)    │
  │ Context token: ":" (score: 0.65)        │
  └─────────────────────────────────────────┘

  TARGET TOKEN 2: "cat's"
  ┌─────────────────────────────────────────┐
  │ Context token: "cats" (score: 0.92)     │
  │ Context token: "about" (score: 0.81)    │
  └─────────────────────────────────────────┘

  ...
```

### Export to JSON Lines

For downstream processing or visualization:

```bash
python -m lit_nlp.examples.hf_seq_salience.run_salience \
    --model_id=gpt2 \
    --dataset=/path/to/prompts.jsonl \
    --max_examples=100 \
    --output=salience_results.jsonl
```

Each line in the output file is a JSON object containing:
- `prompt`: Input text
- `response`: Generated text
- `tokens`: List of tokens
- `salience`: Salience scores per position (based on `--method`)
- `method`: Method used (`grad_l2` or `grad_dot_input`)

### Using different salience methods

```bash
# Unsigned influence (default)
python -m lit_nlp.examples.hf_seq_salience.run_salience \
    --prompt="Count to 5:" \
    --method=grad_l2

# Signed influence (positive = promotes, negative = suppresses)
python -m lit_nlp.examples.hf_seq_salience.run_salience \
    --prompt="Count to 5:" \
    --method=grad_dot_input
```

---

## How the gradient salience is computed

For each forward pass the model runs through these steps:

1. **Embed** the input token IDs via `get_input_embeddings()`.
2. **Forward pass** with `inputs_embeds` (no `input_ids`) so we can back-propagate through the embedding table.
3. **Masked loss** — cross-entropy per token, masked to the *target span* selected by the user.
4. **Backward pass** → obtain `∂loss/∂emb` for every input position.
5. **Two signals** returned per token:
   - `grad_l2  = ‖∂loss/∂emb‖₂`
   - `grad_dot_input = (∂loss/∂emb) · emb`

These are the same gradient signals used in the original LIT prompt-debugging example (`examples/prompt_debugging/`).

---

## Integrating the frontend TypeScript module into LIT's build

The file `frontend/sequence_salience_module.ts` is a copy of the module that
ships with LIT's built-in client. To use it in a custom LIT build:

1. Copy (or symlink) `sequence_salience_module.ts` and `sequence_salience_module.css`
   into `client/modules/`.
2. Import and register the element in `client/main.ts`:
   ```typescript
   import './modules/sequence_salience_module';
   ```
3. Build the client (see the main LIT README for build instructions).

The standalone `standalone_viewer.html` requires **no build step** and communicates
directly with the LIT Python server's REST API.
