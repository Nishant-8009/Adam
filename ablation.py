"""
ablation.py — Ablation study: varying beta1 and beta2 across Adam configurations.
"""

from config  import LR
from trainer import run_experiment


# ─────────────────────────────────────────────
# Ablation Configurations
# Each entry: (label, beta1, beta2)
# ─────────────────────────────────────────────
ABLATION_CONFIGS = [
    ('Default (β1=0.9, β2=0.999)',  0.9,  0.999),
    ('Case 1  (β1=0.8, β2=0.999)',  0.8,  0.999),
    ('Case 2  (β1=0.9, β2=0.99)',   0.9,  0.99),
    ('Case 3  (β1=0.7, β2=0.95)',   0.7,  0.95),
]


def run_ablation_study(train_loader, test_loader, epochs=10, lr=LR):
    """
    Run the ablation study across different (beta1, beta2) settings.
    Uses the custom Adam implementation throughout.

    Returns:
        results: dict mapping label -> history dict
    """
    results = {}

    print("\n" + "═" * 60)
    print("ABLATION STUDY — varying beta1 and beta2")
    print("═" * 60)

    for label, beta1, beta2 in ABLATION_CONFIGS:
        print(f"\n▶ Config: {label}")
        hist = run_experiment(
            label=label,
            use_custom=True,
            train_loader=train_loader,
            test_loader=test_loader,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epochs=epochs
        )
        results[label] = hist

    return results


def print_final_summary(custom_hist, builtin_hist, ablation_results):
    """Print a concise summary table of final metrics."""
    print("\n" + "═" * 65)
    print("FINAL SUMMARY")
    print("═" * 65)
    print(f"{'Optimizer/Config':<40} {'Test Loss':>10} {'Test Acc':>10}")
    print("-" * 65)

    tl = custom_hist['test_loss'][-1]
    ta = custom_hist['test_acc'][-1]
    print(f"{'Custom Adam (scratch)':<40} {tl:>10.4f} {ta:>9.2f}%")

    tl = builtin_hist['test_loss'][-1]
    ta = builtin_hist['test_acc'][-1]
    print(f"{'PyTorch Built-in Adam':<40} {tl:>10.4f} {ta:>9.2f}%")

    print("-" * 65)
    print("Ablation Study Results:")
    for label, _, _ in ABLATION_CONFIGS:
        h  = ablation_results[label]
        tl = h['test_loss'][-1]
        ta = h['test_acc'][-1]
        print(f"  {label:<38} {tl:>10.4f} {ta:>9.2f}%")
    print("═" * 65)