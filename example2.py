"""Example 2: selection bound vs m (part a) and the Occam bound under the LLM prior (part b)."""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

import config
import data
from ollama_client import risks, startup
from plotting import BOUND, GENERATED, HANDCRAFTED, plt
from prior import PriorScorer
from rubric import eps_m, rubric_procedure


def part_a(client, theta, test, y_test, seeds):
    """Excess risk R(r_hat) - min_j R(r_j) as a function of m.

    For each seed the k candidates are generated once from S1, and S2 of size m is
    the first m points of one sample of size max(m) (nested samples: each is still
    an i.i.d. sample of size m, and model calls are shared through the cache).
    """
    rows, pool = [], []
    m_max = max(config.EX2_M_LIST)
    for seed in seeds:
        S1, y1 = data.make_sample(config.EX2_N1, theta, [seed, theta, 1])
        S2, y2 = data.make_sample(m_max, theta, [seed, theta, 2])
        for m in config.EX2_M_LIST:
            res = rubric_procedure(client, S1, y1, S2[:m], y2[:m], test, y_test, config.EX2_K,
                                   seed, desc=f"seed={seed} m={m}")
            rows.append({"seed": seed, "m": m, "R_r_hat": res.R_hat, "min_j_R_rj": res.R_min,
                         "excess": res.R_hat - res.R_min,
                         "eps_m": eps_m(config.EX2_K, m, config.DELTA)})
        pool.extend(res.rubrics)
        print(f"seed={seed}: excess by m = "
              + ", ".join(f"{r['m']}:{r['excess']:.3f}" for r in rows if r["seed"] == seed))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULTS_DIR, "example2a_per_seed.csv"), index=False)
    summary = df.groupby("m").agg(excess_mean=("excess", "mean"), excess_std=("excess", "std"),
                                  eps_m=("eps_m", "first"))
    summary.to_csv(os.path.join(config.RESULTS_DIR, "example2a_summary.csv"))
    print("\n=== Example 2(a): excess risk vs m ===")
    print(summary.to_string(float_format="%.3f"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ms = summary.index.values
    ax.errorbar(ms, summary.excess_mean, yerr=summary.excess_std, color=GENERATED, marker="o",
                markersize=6, capsize=4, label="R(r_hat) - min_j R(r_j)  (mean ± std)")
    ax.plot(ms, summary.eps_m, color=BOUND, marker="s", markersize=6, linestyle="--",
            label=f"eps_m(k={config.EX2_K}, delta={config.DELTA})")
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("selection sample size m")
    ax.set_ylabel("risk")
    ax.set_title(f"theta = {theta}: selection excess vs bound ({len(seeds)} seeds)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "example2a_selection.png"))
    plt.close(fig)
    return pool


def part_b(client, theta, test, y_test, generated):
    """Occam bound bound(r) = R_hat_S(r) + sqrt((log 1/pi(r) + log 1/delta) / (2n))."""
    gen = list(dict.fromkeys(generated))                        # drop exact duplicates
    hand = [r for r in config.HANDCRAFTED_PROMPTS if r not in gen]
    pool = gen + hand
    is_hand = np.array([False] * len(gen) + [True] * len(hand))
    print(f"\nPool: {len(gen)} generated rubrics + {len(hand)} handcrafted prompts")

    scorer = PriorScorer()
    neg_log_pi = np.array([scorer.neg_log_prior(r) for r in pool])
    R = risks(client, pool, [it.text for it in test], y_test, desc="pool test")

    # One fresh sample of size max(n), independent of how the pool was built;
    # S of size n is its first n points (nested, each an i.i.d. sample of size n).
    n_max = max(config.EX2_N_LIST)
    S, yS = data.make_sample(n_max, theta, [config.EX2_BOUND_SEED, theta, 3])
    rows = []
    for n in config.EX2_N_LIST:
        R_emp = risks(client, pool, [it.text for it in S[:n]], yS[:n], desc=f"pool n={n}")
        bound = R_emp + np.sqrt((neg_log_pi + math.log(1 / config.DELTA)) / (2 * n))
        i = int(np.argmin(bound))
        rows.append({"n": n, "coverage": float(np.mean(R <= bound)),
                     "r_tilde_handcrafted": bool(is_hand[i]), "R_r_tilde": float(R[i]),
                     "R_hat_r_tilde": float(R_emp[i]), "bound_r_tilde": float(bound[i]),
                     "log_inv_pi_r_tilde": float(neg_log_pi[i]), "r_tilde": pool[i]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULTS_DIR, "example2b_occam.csv"), index=False)
    pd.DataFrame({"rubric": pool, "handcrafted": is_hand, "log_inv_pi": neg_log_pi,
                  "test_risk": R}).to_csv(
        os.path.join(config.RESULTS_DIR, "example2b_pool.csv"), index=False)

    print(f"\n=== Example 2(b): Occam bound, prior = {scorer.kind} ===")
    if scorer.kind != "llm":
        print("NOTE: code-length prior in use; it makes the bound vacuous at these sample sizes.")
    print(f"log(1/pi(r)) in nats: min {neg_log_pi.min():.1f}, median {np.median(neg_log_pi):.1f}, "
          f"max {neg_log_pi.max():.1f}")
    print(df.drop(columns="r_tilde").to_string(index=False, float_format="%.3f"))
    print(f"(theory: coverage >= 1 - delta = {1 - config.DELTA:.2f} jointly over the pool)")
    print(f"r_tilde at n={df.n.iloc[-1]}:\n  {df.r_tilde.iloc[-1]}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(neg_log_pi[~is_hand], R[~is_hand], s=40, color=GENERATED, edgecolor="white",
               linewidth=1, label="generated rubrics")
    ax.scatter(neg_log_pi[is_hand], R[is_hand], s=70, marker="D", color=HANDCRAFTED,
               edgecolor="white", linewidth=1, label="handcrafted prompts")
    ax.set_xlabel("log(1/pi(r))  [nats]")
    ax.set_ylabel("test risk R(r)")
    ax.set_title(f"theta = {theta}: prior code length vs risk ({scorer.kind} prior)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "example2b_prior_vs_risk.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.n, df.bound_r_tilde, color=BOUND, marker="s", markersize=6, linestyle="--",
            label="bound(r_tilde)")
    ax.plot(df.n, df.R_r_tilde, color=GENERATED, marker="o", markersize=6, label="R(r_tilde)")
    ax.set_xscale("log")
    ax.set_xticks(df.n)
    ax.set_xticklabels([str(n) for n in df.n])
    ax.set_xlabel("sample size n")
    ax.set_ylabel("risk")
    ax.set_title(f"theta = {theta}: Occam-selected rubric ({scorer.kind} prior)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "example2b_bound_vs_n.png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--theta", type=int, default=config.EX2_THETA)
    ap.add_argument("--seeds", type=int, nargs="+", default=config.EX2_SEEDS)
    args = ap.parse_args()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    client = startup(args.workers)
    test = data.test_items()
    y_test = data.labels(test, args.theta)
    print(f"theta={args.theta} ({config.RULES[args.theta][0]}), "
          f"test size={len(test)}, k={config.EX2_K}, n1={config.EX2_N1}\n")

    generated = part_a(client, args.theta, test, y_test, args.seeds)
    part_b(client, args.theta, test, y_test, generated)
    print(f"\nSaved CSVs and plots to {config.RESULTS_DIR}")
    print(client.report_missing())


if __name__ == "__main__":
    main()
