"""LLM prior pi(r) for the Occam bound.

log pi(r) = sum of token log-probabilities of r under the same model, conditioned
only on the task description. Ollama cannot score a given text, so we load the
SAME GGUF weights Ollama uses with llama-cpp-python and teacher-force.

Fallback (llama-cpp-python missing, or the model is not a GGUF): a code-length
prior log(1/pi(r)) = (c + 1) log 2 + c log 128, c = number of characters of r.
All quantities are in nats.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np

import config
from cache import DiskCache, make_key


def ollama_models_dir() -> str:
    return os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models"))


def find_gguf(model: str) -> Optional[str]:
    """Path of the GGUF blob for an Ollama model name such as 'qwen2.5:3b'."""
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library"] + parts
    elif len(parts) == 2:
        parts = ["registry.ollama.ai"] + parts
    root = ollama_models_dir()
    manifest = os.path.join(root, "manifests", *parts, tag)
    if not os.path.exists(manifest):
        return None
    with open(manifest) as f:
        layers = json.load(f).get("layers", [])
    for layer in layers:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            path = os.path.join(root, "blobs", layer["digest"].replace(":", "-"))
            with open(path, "rb") as f:
                if f.read(4) == b"GGUF":
                    return path
    return None


def code_length_neg_log_prior(r: str) -> float:
    c = len(r)
    return (c + 1) * math.log(2) + c * math.log(128)


class PriorScorer:
    def __init__(self, model: str = config.MODEL, cache_path: str = config.CACHE_PATH):
        self.model = model
        self.cache = DiskCache(cache_path)
        self.kind = "code_length"
        self.llm = None
        try:
            from llama_cpp import Llama
        except ImportError:
            print("WARNING: llama-cpp-python is not installed; using the code-length prior "
                  "log(1/pi(r)) = (c+1) log 2 + c log 128. This prior makes the Occam bound "
                  "vacuous at these sample sizes.")
            return
        path = find_gguf(model)
        if path is None:
            print(f"WARNING: no GGUF weights found for '{model}' under {ollama_models_dir()}; "
                  "using the code-length prior (the Occam bound will be vacuous).")
            return
        self.gguf = path
        self.llm = Llama(model_path=path, logits_all=True, n_ctx=2048, n_gpu_layers=-1,
                         seed=0, verbose=False)
        self.kind = "llm"
        self.prefix = self._prefix()
        print(f"Prior: llama-cpp-python teacher forcing on {path}")

    def _prefix(self) -> str:
        """Chat-formatted prompt up to the start of the assistant turn."""
        user = config.PRIOR_PROMPT.format(description=config.TASK_DESCRIPTION)
        template = self.llm.metadata.get("tokenizer.chat_template")
        if template:
            try:
                from llama_cpp.llama_chat_format import Jinja2ChatFormatter
                tok = lambda t: self.llm.detokenize([t], special=True).decode("utf-8", "ignore")
                fmt = Jinja2ChatFormatter(template=template, eos_token=tok(self.llm.token_eos()),
                                          bos_token=tok(self.llm.token_bos()),
                                          add_generation_prompt=True)
                return fmt(messages=[{"role": "user", "content": user}]).prompt
            except Exception as e:  # noqa: BLE001 - template rendering is best effort
                print(f"WARNING: could not render chat template ({e}); using ChatML.")
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    def _score_llm(self, r: str) -> float:
        bos = self.llm.detokenize([self.llm.token_bos()], special=True).decode("utf-8", "ignore")
        add_bos = not (bos and self.prefix.startswith(bos))
        prefix_toks = self.llm.tokenize(self.prefix.encode("utf-8"), add_bos=add_bos, special=True)
        r_toks = self.llm.tokenize(r.encode("utf-8"), add_bos=False, special=False)
        toks = prefix_toks + r_toks
        if len(toks) > self.llm.n_ctx():
            raise ValueError("prefix + rubric longer than the context window")
        self.llm.reset()
        self.llm.eval(toks)
        # scores[t] are the logits predicting token t + 1.
        p = len(prefix_toks)
        logits = np.asarray(self.llm.scores[p - 1:p - 1 + len(r_toks)], dtype=np.float64)
        log_z = np.logaddexp.reduce(logits, axis=1)
        return float(-np.sum(logits[np.arange(len(r_toks)), r_toks] - log_z))

    def neg_log_prior(self, r: str) -> float:
        """log(1 / pi(r)) in nats."""
        if self.kind != "llm":
            return code_length_neg_log_prior(r)
        key = make_key("prior", self.gguf, self.prefix, r)
        hit = self.cache.get(key)
        if hit is None:
            hit = self._score_llm(r)
            self.cache.set(key, hit)
        return hit
