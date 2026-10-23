"""Example 1: separation between handcrafted prompts and the rubric procedure.

For each task theta and seed, run the rubric procedure (n1, k, m from config) and
compare the test risk of r_hat with the test risk of each handcrafted prompt r_H.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import config
import data
from ollama_client import risks, startup
from plotting import GENERATED, HANDCRAFTED, REFERENCE, plt
from rubric import eps_m, rubric_procedure


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=config.EX1_SEEDS)
    args = ap.parse_args()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    client = startup(args.workers)
    test = data.test_items()
    test_x = [it.text for it in test]
    Delta = data.print_disagreement(test)
    thetas = sorted(config.RULES)
    H = config.HANDCRAFTED_PROMPTS
    eps = eps_m(config.EX1_K, config.EX1_M, config.DELTA)

    # Handcrafted prompts do not depend on theta: one pass over the test set,
    # then the risk for each theta is computed from the same probabilities.
    print("Evaluating handcrafted prompts on the test set ...")
    RH = {t: risks(client, H, test_x, data.labels(test, t), desc="handcrafted") for t in thetas}

    rows, rubric_log = [], []
    for t in thetas:
        y_test = data.labels(test, t)
        for seed in args.seeds:
            S1, y1 = data.make_sample(config.EX1_N1, t, [seed, t, 1])
            S2, y2 = data.make_sample(config.EX1_M, t, [seed, t, 2])
            res = rubric_procedure(client, S1, y1, S2, y2, test, y_test, config.EX1_K, seed,
                                   desc=f"theta={t} seed={seed}")
            row = {"theta": t, "seed": seed}
            row.update({f"R_H{h + 1}": float(RH[t][h]) for h in range(len(H))})
            row.update({"R_r_hat": res.R_hat, "min_j_R_rj": res.R_min, "eps_m": eps, "Delta": Delta})
            row.update({f"p_eps0={e}": res.proposal_quality(e) for e in config.EPS0_LIST})
            rows.append(row)
            rubric_log.append({"theta": t, "seed": seed, "rubrics": res.rubrics,
                               "s2_risks": res.s2_risks.tolist(),
                               "test_risks": res.test_risks.tolist(), "hat": res.hat})
            print(f"theta={t} seed={seed}: R(r_hat)={res.R_hat:.3f} min_j R(r_j)={res.R_min:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULTS_DIR, "example1_per_seed.csv"), index=False)
    summary = df.drop(columns="seed").groupby("theta").agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(os.path.join(config.RESULTS_DIR, "example1_summary.csv"))
    with open(os.path.join(config.RESULTS_DIR, "example1_rubrics.json"), "w") as f:
        json.dump(rubric_log, f, indent=2)

    pd.set_option("display.width", 200, "display.max_columns", 50, "display.precision", 3)
    print("\n=== Example 1 results (mean and std over seeds) ===")
    show = ["R_H%d" % (h + 1) for h in range(len(H))] + ["R_r_hat", "min_j_R_rj"] + \
           [f"p_eps0={e}" for e in config.EPS0_LIST]
    for t in thetas:
        print(f"\ntheta={t} ({config.RULES[t][0]}), eps_m={eps:.3f}, Delta={Delta:.3f}")
        print(df[df.theta == t][show].agg(["mean", "std"]).to_string())

    # ----------------------------------------------------------------- checks
    mean_hat = df.groupby("theta")["R_r_hat"].mean()
    print("\n=== Lemma check: #thetas with R(r_H) < Delta (theory: at most 1) ===")
    for h, prompt in enumerate(H):
        below = [t for t in thetas if RH[t][h] < Delta]
        print(f"  H{h + 1}: {len(below)} (thetas {below})  \"{prompt}\"")

    print("\n=== Separation check: max_theta [R(r_H) - R(r_hat)] vs Delta - eps0 - eps_m ===")
    print("    (the lower bound applies when a candidate with risk <= eps0 is proposed; "
          "see mean p above)")
    lower = {e: Delta - e - eps for e in config.EPS0_LIST}
    print("  lower bounds: " + ", ".join(f"eps0={e}: {v:.3f}" for e, v in lower.items()))
    for h in range(len(H)):
        gaps = {t: RH[t][h] - mean_hat[t] for t in thetas}
        t_star = max(gaps, key=gaps.get)
        print(f"  H{h + 1}: max gap = {gaps[t_star]:.3f} (theta={t_star})  " +
              ", ".join(f"exceeds eps0={e} bound: {gaps[t_star] >= v}" for e, v in lower.items()))

    # ------------------------------------------------------------------ plots
    for t in thetas:
        sub = df[df.theta == t]
        labels = [f"H{h + 1}" for h in range(len(H))] + ["r_hat", "best cand."]
        means = [RH[t][h] for h in range(len(H))] + [sub.R_r_hat.mean(), sub.min_j_R_rj.mean()]
        errs = [0] * len(H) + [sub.R_r_hat.std(), sub.min_j_R_rj.std()]
        colours = [HANDCRAFTED] * len(H) + [GENERATED, GENERATED]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, yerr=errs, color=colours, width=0.6, capsize=4,
               edgecolor="white", linewidth=2)
        ax.axhline(Delta, color=REFERENCE, linestyle="--", linewidth=1.5)
        ax.text(-0.45, Delta, f"Delta = {Delta:.2f}", va="bottom", ha="left", color=REFERENCE,
                backgroundcolor="white")
        ax.set_ylabel("test risk R(r)")
        ax.set_ylim(0, max(max(means) * 1.15, Delta * 1.2))
        ax.set_title(f"theta = {t}: {config.RULES[t][0]}\n"
                     f"orange: handcrafted, blue: rubric procedure (mean ± std over "
                     f"{len(args.seeds)} seeds)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(config.RESULTS_DIR, f"example1_theta{t}.png"))
        plt.close(fig)

    print(f"\nSaved CSVs, rubrics and plots to {config.RESULTS_DIR}")
    print(client.report_missing())


if __name__ == "__main__":
    main()
