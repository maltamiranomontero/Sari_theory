"""Shared configuration for both experiments.

Everything that you are likely to want to edit lives here: the model, the data
attributes, the hidden labelling rules, the handcrafted prompts and the
experiment sizes.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Model / server
# --------------------------------------------------------------------------- #
MODEL = os.environ.get("RUBRIC_MODEL", "qwen2.5:3b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
REQUEST_TIMEOUT = 300           # seconds per HTTP request
NUM_WORKERS = 4                 # thread-pool size; match OLLAMA_NUM_PARALLEL
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "calls.sqlite")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Label-probability call: deterministic, one token, top-20 logprobs.
LABEL_OPTIONS = {"temperature": 0, "num_predict": 1, "seed": 0}
TOP_LOGPROBS = 20

# Rubric sampling call.
RUBRIC_OPTIONS = {"temperature": 1.0, "top_p": 0.95, "num_predict": 150}

# --------------------------------------------------------------------------- #
# Data generator
# --------------------------------------------------------------------------- #
SIZES = ["small", "medium", "large"]
COLOURS = ["red", "blue", "green", "black"]
PRICE_RANGE = (5, 100)          # inclusive
ITEMS = ["jacket", "lamp", "mug", "backpack", "chair"]

TASK_DESCRIPTION = (
    "Decide whether the customer will accept this item. "
    "Answer 1 for accept, 0 for reject."
)

# Hidden labelling rules f_theta. Each entry: theta -> (human description, rule).
# A rule receives an `Item` (see data.py) and returns True for label 1.
RULES = {
    1: ("price < 50", lambda it: it.price < 50),
    2: ("colour in {red, black}", lambda it: it.colour in ("red", "black")),
    3: ("size == large or item == chair", lambda it: it.size == "large" or it.item == "chair"),
}

TEST_SIZE = 1000
TEST_SEED = 12345

# --------------------------------------------------------------------------- #
# Handcrafted prompts r_H (written from the task description only)
# --------------------------------------------------------------------------- #
HANDCRAFTED_PROMPTS = [
    "Accept items that seem good value and useful.",
    "Accept the item if a typical customer would be happy to buy it at that price; otherwise reject it.",
    "Accept affordable, practical items in common colours and sizes. Reject items that seem overpriced or unappealing.",
]

# --------------------------------------------------------------------------- #
# Theory constants
# --------------------------------------------------------------------------- #
DELTA = 0.05
EPS0_LIST = [0.05, 0.1, 0.2]

# --------------------------------------------------------------------------- #
# Example 1
# --------------------------------------------------------------------------- #
EX1_N1 = 20
EX1_K = 8
EX1_M = 50
EX1_SEEDS = [0, 1, 2, 3, 4]

# --------------------------------------------------------------------------- #
# Example 2
# --------------------------------------------------------------------------- #
EX2_THETA = 1
EX2_N1 = 20
EX2_K = 8
EX2_M_LIST = [10, 25, 50, 100, 200]
EX2_SEEDS = list(range(10))
EX2_N_LIST = [25, 50, 100, 200, 400]
EX2_BOUND_SEED = 999            # seed of the fresh sample used for the Occam bound

# Prompt used to define the LLM prior pi(r) (formatted with the chat template).
PRIOR_PROMPT = "Write a rubric for the following task: {description}"

# --------------------------------------------------------------------------- #
# Quick mode (QUICK=1): same definitions, smaller sizes (~25k calls, ~40 min for
# a 3B model). The quick test set is a prefix of the full one, so cached calls
# carry over to a full run.
# --------------------------------------------------------------------------- #
QUICK = os.environ.get("QUICK") == "1"
if QUICK:
    TEST_SIZE = 200
    EX1_SEEDS = [0, 1]
    EX2_SEEDS = [0, 1, 2]
    EX2_N_LIST = [25, 50, 100, 200]
    RESULTS_DIR = os.path.join(RESULTS_DIR, "quick")
