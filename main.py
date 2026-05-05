"""
main.py — Entry point. Wires together all modules and runs the full experiment.

Run with:
    python main.py
"""

from config  import EPOCHS, BATCH_SIZE, LR
from model   import get_dataloaders
from trainer import run_experiment
from plots   import plot_comparison, plot_ablation
from ablation import (
    run_ablation_study,
    print_final_summary,
    ABLATION_CONFIGS,
)


def main():
    # ── Data ──────────────────────────────────
    print("Loading MNIST dataset …")
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # ── Part 3A: Train custom Adam ─────────────
    print("\n" + "═" * 60)
    print("PART 3A — Custom Adam Optimizer (from scratch)")
    print("═" * 60)
    custom_hist = run_experiment(
        label='Custom Adam',
        use_custom=True,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=LR,
        epochs=EPOCHS
    )

    # ── Part 3B: Train built-in Adam ───────────
    print("\n" + "═" * 60)
    print("PART 3B — PyTorch Built-in Adam (for comparison)")
    print("═" * 60)
    builtin_hist = run_experiment(
        label='PyTorch Adam',
        use_custom=False,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=LR,
        epochs=EPOCHS
    )

    # ── Part 4: Comparison plots ──────────────
    plot_comparison(custom_hist, builtin_hist, save_path='comparison_plots.png')

    # ── Part 5: Ablation study ────────────────
    ablation_results = run_ablation_study(train_loader, test_loader, epochs=EPOCHS, lr=LR)
    plot_ablation(ablation_results, ABLATION_CONFIGS, save_path='ablation_plots.png')

    # ── Summary ───────────────────────────────
    print_final_summary(custom_hist, builtin_hist, ablation_results)


if __name__ == '__main__':
    main()