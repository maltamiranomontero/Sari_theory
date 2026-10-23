"""Ollama client: start-up checks, label probabilities and rubric sampling.

All model calls go through `OllamaClient.chat`, which caches responses on disk
keyed by (model, payload), so re-running an experiment is cheap.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

import numpy as np
import requests
from tqdm import tqdm

import config
from cache import DiskCache, make_key


def label_prompt(rubric: str, x: str) -> str:
    return (
        f"Task: {config.TASK_DESCRIPTION}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Input: {x}\n\n"
        "Using the rubric, decide the label for this input. "
        "Answer with a single character: 0 or 1."
    )


def rubric_generation_prompt(examples: Sequence[str], ys: Sequence[int]) -> str:
    lines = "\n".join(f"{i + 1}. {x} -> {y}" for i, (x, y) in enumerate(zip(examples, ys)))
    return (
        f"Task: {config.TASK_DESCRIPTION}\n\n"
        f"Here are labelled examples (input -> label):\n{lines}\n\n"
        "Write a short rubric (a few sentences of explicit decision rules) that would let "
        "someone label new inputs for this task correctly. Output only the rubric."
    )


class OllamaClient:
    def __init__(self, model: str = config.MODEL, host: str = config.OLLAMA_HOST,
                 cache_path: str = config.CACHE_PATH, workers: int = config.NUM_WORKERS):
        self.model = model
        self.host = host.rstrip("/")
        self.cache = DiskCache(cache_path)
        self.workers = workers
        self.supports_thinking = False
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self.n_label_evals = 0
        self.n_missing = {"0": 0, "1": 0}

    # ------------------------------------------------------------------ HTTP
    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def _post(self, path: str, payload: dict, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                resp = self._session().post(f"{self.host}{path}", json=payload,
                                            timeout=config.REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------ start-up
    def check(self) -> None:
        """Stop with a clear message if the server, model or logprobs are unavailable."""
        def die(msg: str) -> None:
            sys.exit(f"\n[start-up check failed] {msg}\n")

        try:
            version = requests.get(f"{self.host}/api/version", timeout=10).json().get("version")
        except requests.RequestException:
            die(f"Cannot reach the Ollama server at {self.host}. Start it with `ollama serve`.")
        resp = requests.post(f"{self.host}/api/show", json={"model": self.model}, timeout=30)
        if resp.status_code == 404:
            die(f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}")
        resp.raise_for_status()
        self.supports_thinking = "thinking" in (resp.json().get("capabilities") or [])

        test = self._post("/api/chat", self._chat_payload(
            [{"role": "user", "content": "Answer with a single character 0 or 1: is 2 > 1?"}],
            config.LABEL_OPTIONS, logprobs=True))
        lp = test.get("logprobs")
        if not lp or not lp[0].get("top_logprobs"):
            die(f"Ollama {version} did not return `logprobs`. Update Ollama to a version "
                "that supports logprobs on /api/chat (https://ollama.com/download).")
        print(f"Ollama {version} OK; model={self.model}; logprobs OK; "
              f"thinking mode {'disabled' if self.supports_thinking else 'n/a'}; workers={self.workers}")

    # ---------------------------------------------------------------- chat
    def _chat_payload(self, messages: List[dict], options: dict, logprobs: bool = False) -> dict:
        payload = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = config.TOP_LOGPROBS
        if self.supports_thinking:
            payload["think"] = False
        return payload

    def chat(self, messages: List[dict], options: dict, logprobs: bool = False) -> dict:
        """Cached /api/chat call. Returns {"content": str, "logprobs": list | None}."""
        payload = self._chat_payload(messages, options, logprobs)
        key = make_key("chat", self.model, payload)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        resp = self._post("/api/chat", payload)
        out = {"content": resp["message"]["content"], "logprobs": resp.get("logprobs")}
        self.cache.set(key, out)
        return out

    # ------------------------------------------------------ label probabilities
    def prob1(self, rubric: str, x: str) -> float:
        """M(1 | r, x) from the first-token top_logprobs, normalised over {0, 1}."""
        out = self.chat([{"role": "user", "content": label_prompt(rubric, x)}],
                        config.LABEL_OPTIONS, logprobs=True)
        top = out["logprobs"][0]["top_logprobs"]
        found: Dict[str, List[float]] = {"0": [], "1": []}
        for entry in top:
            tok = entry["token"].strip()
            if tok in found:                     # accepts "1" and " 1" style variants
                found[tok].append(entry["logprob"])
        floor = min(e["logprob"] for e in top) - math.log(2)
        lp = {}
        with self._stats_lock:
            self.n_label_evals += 1
            for lab, vals in found.items():
                if vals:
                    lp[lab] = float(np.logaddexp.reduce(vals))
                else:
                    lp[lab] = floor
                    self.n_missing[lab] += 1
        return float(1.0 / (1.0 + math.exp(lp["0"] - lp["1"])))

    def prob1_matrix(self, rubrics: Sequence[str], xs: Sequence[str],
                     desc: str = "label probs") -> np.ndarray:
        """Matrix P[i, j] = M(1 | rubrics[i], xs[j]), evaluated with a thread pool."""
        jobs = [(i, j) for i in range(len(rubrics)) for j in range(len(xs))]
        out = np.empty((len(rubrics), len(xs)))
        with ThreadPoolExecutor(self.workers) as pool, \
                tqdm(total=len(jobs), desc=desc, leave=False) as bar:
            futures = {pool.submit(self.prob1, rubrics[i], xs[j]): (i, j) for i, j in jobs}
            for fut in as_completed(futures):
                i, j = futures[fut]
                out[i, j] = fut.result()
                bar.update(1)
        return out

    def report_missing(self) -> str:
        n = max(self.n_label_evals, 1)
        return (f"Label evaluations: {self.n_label_evals}. Label absent from top-{config.TOP_LOGPROBS} "
                f"(filled with min logprob - log 2): '0' {self.n_missing['0']} times "
                f"({100 * self.n_missing['0'] / n:.2f}%), '1' {self.n_missing['1']} times "
                f"({100 * self.n_missing['1'] / n:.2f}%).")

    # -------------------------------------------------------- rubric sampling
    def sample_rubric(self, prompt: str, seed: int) -> str:
        options = dict(config.RUBRIC_OPTIONS, seed=int(seed))
        return self.chat([{"role": "user", "content": prompt}], options)["content"].strip()


def risks(client: OllamaClient, rubrics: Sequence[str], xs: Sequence[str], y: np.ndarray,
          desc: str = "risk") -> np.ndarray:
    """Empirical risk R_hat_S(r) = mean_i (1 - M(y_i | r, x_i)) for every rubric."""
    p1 = client.prob1_matrix(rubrics, xs, desc=desc)
    p_correct = np.where(np.asarray(y)[None, :] == 1, p1, 1.0 - p1)
    return 1.0 - p_correct.mean(axis=1)


def startup(workers: Optional[int] = None) -> OllamaClient:
    client = OllamaClient(workers=workers or config.NUM_WORKERS)
    client.check()
    return client
