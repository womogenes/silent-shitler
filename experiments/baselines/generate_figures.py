"""Generate figures for NeurIPS paper."""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Output directory
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def generate_meta_heatmaps():
    """Generate Meta Agent parameter sweep heatmaps."""
    # Load sweep results
    sweep_file = Path(__file__).parent.parent / "meta" / "results" / "param_sweep_20251212_043025.json"
    with open(sweep_file) as f:
        data = json.load(f)

    # Convert to arrays for heatmap
    all_results = data["all_results"]

    # Build pivot tables
    prez_vals = sorted(set(r["config"]["fasc_policy_prez_sus"] for r in all_results))
    chanc_vals = sorted(set(r["config"]["fasc_policy_chanc_sus"] for r in all_results))
    conflict_vals = sorted(set(r["config"]["conflict_sus"] for r in all_results))
    vote_vals = sorted(set(r["config"]["vote_threshold_mult"] for r in all_results))

    # Heatmap 1: prez vs chanc (averaged over other params)
    prez_chanc = np.zeros((len(prez_vals), len(chanc_vals)))
    prez_chanc_counts = np.zeros((len(prez_vals), len(chanc_vals)))

    for r in all_results:
        i = prez_vals.index(r["config"]["fasc_policy_prez_sus"])
        j = chanc_vals.index(r["config"]["fasc_policy_chanc_sus"])
        prez_chanc[i, j] += r["combined_score"]
        prez_chanc_counts[i, j] += 1

    prez_chanc /= prez_chanc_counts

    # Heatmap 2: conflict vs vote (averaged over other params)
    conflict_vote = np.zeros((len(conflict_vals), len(vote_vals)))
    conflict_vote_counts = np.zeros((len(conflict_vals), len(vote_vals)))

    for r in all_results:
        i = conflict_vals.index(r["config"]["conflict_sus"])
        j = vote_vals.index(r["config"]["vote_threshold_mult"])
        conflict_vote[i, j] += r["combined_score"]
        conflict_vote_counts[i, j] += 1

    conflict_vote /= conflict_vote_counts

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap 1
    im1 = axes[0].imshow(prez_chanc * 100, aspect="auto", cmap="viridis")
    axes[0].set_xticks(np.arange(len(chanc_vals)))
    axes[0].set_yticks(np.arange(len(prez_vals)))
    axes[0].set_xticklabels(chanc_vals)
    axes[0].set_yticklabels(prez_vals)
    axes[0].set_xlabel("Chancellor Suspicion")
    axes[0].set_ylabel("President Suspicion")
    axes[0].set_title("Combined Win Rate (%)")
    for i in range(len(prez_vals)):
        for j in range(len(chanc_vals)):
            axes[0].text(j, i, f"{prez_chanc[i,j]*100:.1f}", ha="center", va="center", color="white", fontsize=9)
    plt.colorbar(im1, ax=axes[0])

    # Heatmap 2
    im2 = axes[1].imshow(conflict_vote * 100, aspect="auto", cmap="viridis")
    axes[1].set_xticks(np.arange(len(vote_vals)))
    axes[1].set_yticks(np.arange(len(conflict_vals)))
    axes[1].set_xticklabels(vote_vals)
    axes[1].set_yticklabels(conflict_vals)
    axes[1].set_xlabel("Vote Threshold Multiplier")
    axes[1].set_ylabel("Conflict Suspicion")
    axes[1].set_title("Combined Win Rate (%)")
    for i in range(len(conflict_vals)):
        for j in range(len(vote_vals)):
            axes[1].text(j, i, f"{conflict_vote[i,j]*100:.1f}", ha="center", va="center", color="white", fontsize=9)
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "meta_param_heatmap.png", dpi=150)
    plt.savefig(FIGURES_DIR / "meta_param_heatmap.pdf")
    print(f"Saved: {FIGURES_DIR / 'meta_param_heatmap.png'}")
    print(f"Saved: {FIGURES_DIR / 'meta_param_heatmap.pdf'}")


def generate_cfr_convergence():
    """Generate CFR convergence plots."""
    results_dir = Path(__file__).parent.parent / "results"

    # Load all CFR checkpoints
    data_points = []
    for f in results_dir.glob("cfr_liberal_*.json"):
        with open(f) as fp:
            d = json.load(fp)
        data_points.append({
            "iterations": d["iterations"],
            "win_rate": d["liberal_win_rate"],
            "infosets": d["infosets"],
        })

    # Sort by iterations
    data_points.sort(key=lambda x: x["iterations"])

    iterations = [d["iterations"] for d in data_points]
    win_rates = [d["win_rate"] * 100 for d in data_points]
    infosets = [d["infosets"] for d in data_points]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Win rate
    axes[0].plot(iterations, win_rates, "b-o", markersize=4)
    axes[0].axhline(y=50, linestyle="--", color="gray", alpha=0.7, label="Random baseline")
    axes[0].set_xlabel("Training Iterations")
    axes[0].set_ylabel("Liberal Win Rate (%)")
    axes[0].set_title("CFR Learning Curve")
    axes[0].legend()
    axes[0].set_ylim(40, 60)

    # Infosets
    axes[1].plot(iterations, [i / 1e6 for i in infosets], "r-o", markersize=4)
    axes[1].set_xlabel("Training Iterations")
    axes[1].set_ylabel("Information Sets (millions)")
    axes[1].set_title("State Space Exploration")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cfr_convergence.png", dpi=150)
    plt.savefig(FIGURES_DIR / "cfr_convergence.pdf")
    print(f"Saved: {FIGURES_DIR / 'cfr_convergence.png'}")
    print(f"Saved: {FIGURES_DIR / 'cfr_convergence.pdf'}")


def generate_agent_comparison():
    """Generate agent comparison bar chart."""
    # Data from run_full_eval.py output
    agents = ["Random", "Selfish", "CFR", "PPO", "Meta"]
    vs_random = [48.3, 51.2, 50.8, 50.4, 68.6]
    vs_selfish = [30.7, 26.0, 28.1, 33.4, 48.2]

    # CI widths (approximate from output)
    vs_random_err = [3.1, 3.1, 3.1, 3.1, 2.9]
    vs_selfish_err = [2.8, 2.6, 2.7, 3.0, 3.1]

    x = np.arange(len(agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, vs_random, width, yerr=vs_random_err, capsize=4, label="vs Random Fascists")
    bars2 = ax.bar(x + width/2, vs_selfish, width, yerr=vs_selfish_err, capsize=4, label="vs Selfish Fascists")

    ax.axhline(y=50, linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("Liberal Agent")
    ax.set_ylabel("Liberal Win Rate (%)")
    ax.set_title("Agent Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend()
    ax.set_ylim(0, 80)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "agent_comparison.png", dpi=150)
    plt.savefig(FIGURES_DIR / "agent_comparison.pdf")
    print(f"Saved: {FIGURES_DIR / 'agent_comparison.png'}")
    print(f"Saved: {FIGURES_DIR / 'agent_comparison.pdf'}")


if __name__ == "__main__":
    print("Generating figures for NeurIPS paper...")
    print(f"Output directory: {FIGURES_DIR}")
    print()

    generate_meta_heatmaps()
    print()
    generate_cfr_convergence()
    print()
    generate_agent_comparison()

    print()
    print("Done! Figures saved to:", FIGURES_DIR)
