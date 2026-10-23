"""The rubric procedure (propose k rubrics from S1, select on S2) and theory helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from data import Item
from ollama_client import OllamaClient, risks, rubric_generation_prompt


def eps_m(k: int, m: int, delta: float) -> float:
    """Selection bound eps_m(k, delta) = sqrt(2 log(2k / delta) / m)."""
    return math.sqrt(2.0 * math.log(2.0 * k / delta) / m)


@dataclass
class RubricResult:
    rubrics: List[str]
    s2_risks: np.ndarray        # R_hat_{S2}(r_j)
    test_risks: np.ndarray      # R(r_j), test-set proxy for the true risk
    hat: int                    # index of r_hat = argmin_j R_hat_{S2}(r_j)

    @property
    def r_hat(self) -> str:
        return self.rubrics[self.hat]

    @property
    def R_hat(self) -> float:
        return float(self.test_risks[self.hat])

    @property
    def R_min(self) -> float:
        return float(self.test_risks.min())

    def proposal_quality(self, eps0: float) -> float:
        """Fraction of candidates with test risk <= eps0 (estimate of p)."""
        return float(np.mean(self.test_risks <= eps0))


def rubric_procedure(client: OllamaClient, S1: Sequence[Item], y1: np.ndarray,
                     S2: Sequence[Item], y2: np.ndarray,
                     test: Sequence[Item], y_test: np.ndarray,
                     k: int, seed: int, desc: str = "") -> RubricResult:
    """1. prompt with S1, 2. sample k rubrics (seeds seed*1000 + j), 3. select on S2."""
    prompt = rubric_generation_prompt([it.text for it in S1], list(y1))
    rubrics = [client.sample_rubric(prompt, seed * 1000 + j) for j in range(k)]
    s2 = risks(client, rubrics, [it.text for it in S2], y2, desc=f"{desc} S2")
    test_r = risks(client, rubrics, [it.text for it in test], y_test, desc=f"{desc} test")
    return RubricResult(rubrics, s2, test_r, int(np.argmin(s2)))
