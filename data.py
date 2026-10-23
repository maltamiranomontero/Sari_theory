"""Synthetic item descriptions and hidden labelling rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

import config


@dataclass(frozen=True)
class Item:
    size: str
    colour: str
    price: int
    item: str
    text: str


TEMPLATES = [
    "A {size} {colour} {item} priced at {price} pounds.",
    "{Size} {colour} {item}, {price} pounds.",
    "This {item} is {colour}, {size}, and costs {price} pounds.",
    "For sale: a {colour} {item} in size {size} for {price} pounds.",
    "A {colour} {item} ({size}) at a price of {price} pounds.",
]


def _render(rng: np.random.Generator, size: str, colour: str, price: int, item: str) -> str:
    template = TEMPLATES[rng.integers(len(TEMPLATES))]
    return template.format(size=size, Size=size.capitalize(), colour=colour, price=price, item=item)


def sample_items(n: int, rng: np.random.Generator) -> List[Item]:
    """Draw n items with independent uniform attributes.

    Items are drawn one at a time, so with the same generator seed the first m
    items of sample_items(n) equal sample_items(m). Nested samples therefore share
    cached model calls.
    """
    lo, hi = config.PRICE_RANGE
    items = []
    for _ in range(n):
        size = config.SIZES[rng.integers(len(config.SIZES))]
        colour = config.COLOURS[rng.integers(len(config.COLOURS))]
        price = int(rng.integers(lo, hi + 1))
        kind = config.ITEMS[rng.integers(len(config.ITEMS))]
        items.append(Item(size, colour, price, kind, _render(rng, size, colour, price, kind)))
    return items


def labels(items: Sequence[Item], theta: int) -> np.ndarray:
    rule = config.RULES[theta][1]
    return np.array([int(bool(rule(it))) for it in items], dtype=int)


def make_sample(n: int, theta: int, seed: Sequence[int]) -> Tuple[List[Item], np.ndarray]:
    """Labelled sample of size n for task theta; `seed` is a numpy seed sequence."""
    items = sample_items(n, np.random.default_rng(list(seed)))
    return items, labels(items, theta)


def test_items() -> List[Item]:
    return sample_items(config.TEST_SIZE, np.random.default_rng(config.TEST_SEED))


def disagreement(items: Sequence[Item]) -> Tuple[dict, float]:
    """Pairwise P(f_theta != f_theta') on `items`, and Delta = 0.5 * min."""
    thetas = sorted(config.RULES)
    ys = {t: labels(items, t) for t in thetas}
    dis = {}
    for i, a in enumerate(thetas):
        for b in thetas[i + 1:]:
            dis[(a, b)] = float(np.mean(ys[a] != ys[b]))
    return dis, 0.5 * min(dis.values())


def print_disagreement(items: Sequence[Item]) -> float:
    dis, delta_sep = disagreement(items)
    print("Pairwise disagreement P(f_theta != f_theta') on the test inputs:")
    for (a, b), d in dis.items():
        print(f"  theta={a} vs theta={b}: {d:.3f}")
    print(f"Delta = 0.5 * min disagreement = {delta_sep:.3f}\n")
    return delta_sep
