# Sari_theory

Two small experiments that illustrate a learning-theory result about LLM-generated
rubrics, run entirely on a local Ollama server.

## Files

| File | Purpose |
|---|---|
| `config.py` | Model, data attributes, hidden rules `RULES`, handcrafted prompts, experiment sizes |
| `data.py` | Synthetic item generator, labelling rules, pairwise disagreement and Delta |
| `cache.py` | On-disk SQLite cache for every model call (keyed by model, prompt and options) |
| `ollama_client.py` | Start-up checks, label probabilities M(y \| r, x) from logprobs, rubric sampling, risks |
| `rubric.py` | Rubric procedure (propose k from S1, select on S2) and `eps_m(k, delta)` |
| `prior.py` | Prior log(1/pi(r)) via llama-cpp-python on Ollama's GGUF, with a code-length fallback |
| `example1.py` | Separation between handcrafted prompts and the rubric procedure |
| `example2.py` | (a) selection excess vs m; (b) Occam bound under the LLM prior |
| `plotting.py` | Shared matplotlib style |

## Setup

```bash
ollama pull qwen2.5:3b

# Let Ollama serve parallel requests; match NUM_WORKERS in config.py (default 4).
OLLAMA_NUM_PARALLEL=4 ollama serve      # or set it in the Ollama app's environment and restart

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional, for the LLM prior in Example 2(b). Without it a code-length prior is used
# (and the Occam bound is vacuous). On Apple Silicon, build with Metal:
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

Your Ollama version must return `logprobs` from `/api/chat`; the scripts check for
this at start-up. To use another model, set `RUBRIC_MODEL=llama3.2:1b` or edit
`config.MODEL`. For models with a thinking mode, `"think": false` is sent automatically.

## Running

```bash
python example1.py                 # options: --workers N  --seeds 0 1 2
python example2.py                 # options: --workers N  --theta 1  --seeds 0 1 ...
```

For a first look, `QUICK=1 python example1.py && QUICK=1 python example2.py` keeps all
definitions but uses a 200-point test set, 2 seeds in Example 1, 3 seeds in Example 2
and n up to 200. That is about 25k calls (about 40 min for a 3B model), with results in
`results/quick/`. Calls cached by the quick run are reused by a later full run.

Results (CSV, JSON of the generated rubrics, PNG plots) go to `results/`. Model
calls are cached in `cache/calls.sqlite`, so a re-run is fast. Delete that file to
start fresh.

## Number of model calls

Label-probability calls (1 token each) dominate. With the default sizes (test set
1000, k = 8):

| Step | Calls |
|---|---|
| Ex 1, handcrafted prompts on test (shared across thetas) | 3 × 1000 = 3,000 |
| Ex 1, per (theta, seed): 8 rubrics + 8 × 50 (S2) + 8 × 1000 (test) | 8,408 |
| **Ex 1 total** (3 thetas × 5 seeds) | **≈ 129,000** |
| Ex 2(a), per seed: 8 rubrics + 8 × 200 (S2, nested) + 8 × 1000 (test) | 9,608 |
| Ex 2(a), 10 seeds | ≈ 96,000 (≈ 54,000 new if Example 1 ran first) |
| Ex 2(b): pool of ≈ 83 rubrics × 400 (nested S) + prior scoring (83 forward passes) | ≈ 33,000 |
| **Ex 2 total** | **≈ 130,000** (≈ 87,000 after Example 1) |

For reference, `llama3.2:1b` on this machine runs about 20–25 label calls per second
with 4 workers. A 3B model is slower, so expect a few hours for Example 1. The
easiest way to cut runtime is to lower `TEST_SIZE` (most calls go to evaluating
candidates on the test set) or pass fewer `--seeds`.

## Implementation notes

- **Label probabilities.** The first generated token's top-20 logprobs are used.
  Variants of "0"/"1" with a leading space are summed (log-sum-exp) and then
  normalised over {0, 1}. A label missing from the top 20 gets (smallest logprob
  returned − log 2). How often this happened is printed at the end of each run.
  The label prompt contains the shared task description, the rubric and the input.
- **Seeds.** Samples come from `numpy.random.default_rng([seed, theta, tag])`, with tag 1 = S1,
  2 = S2 and 3 = the Occam sample. Rubric j of seed s uses Ollama seed `1000 s + j`.
  Example 2 with theta = 1 and seeds 0–4 therefore reuses Example 1's rubrics and
  cached calls.
- **Nested samples.** Items are drawn one at a time, so a sample of size m is the
  first m points of a sample of size 200. In Example 2(a), S2 for each m is a prefix
  of one draw, and the k candidates of a seed are shared across m. In 2(b), S for
  each n is a prefix of one fresh draw of size 400 that is independent of the pool.
  Each sample is still i.i.d. of its size, so the per-m and per-n statements are
  unchanged. Only the different m (or n) values are correlated with each other.
- **Prior.** The GGUF is found through the manifest under `~/.ollama/models` (or
  `$OLLAMA_MODELS`). The prefix is rendered with the GGUF's own Jinja chat template,
  and the tokens of r are teacher-forced and summed (nats). No end-of-turn token is
  included. For Llama-3 templates, the prefix contains today's date, so the prior
  changes slightly from day to day. Qwen2.5's template does not do this.

## Project layout

| Path | Contents |
|---|---|
| `config.py`, `data.py`, `cache.py`, `ollama_client.py`, `rubric.py`, `prior.py`, `plotting.py` | Experiment code (see Files above) |
| `example1.py`, `example2.py` | The two experiments |
| `notes/` | LaTeX notes on the theory (`main.tex`, `sections/`, `appendix_proofs.tex`, `refs.bib`) |
| `papers/` | Reading list (`papers/README.md`); arXiv PDFs in `papers/pdf/` (gitignored) |
| `talks/` | The two Simons Institute talks that motivate the project |
| `open_questions.md` | Open questions |

## Building the notes

Requires a TeX distribution with `latexmk`.

```bash
cd notes
make          # latexmk -pdf main.tex, produces notes/main.pdf
make clean    # remove build files
```
