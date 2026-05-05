"""
plots.py — Visualization: comparison plots and ablation study plots.
"""

import matplotlib.pyplot as plt


def plot_comparison(custom_hist, builtin_hist, save_path='comparison_plots.png'):
    """
    Plot training/test loss and accuracy for custom vs built-in Adam.
    """
    epochs = range(1, len(custom_hist['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Custom Adam vs PyTorch Built-in Adam\n(MNIST, MLP)', fontsize=14, fontweight='bold')

    # ── Train Loss ──
    axes[0, 0].plot(epochs, custom_hist['train_loss'],  label='Custom Adam',  color='steelblue',  linewidth=2)
    axes[0, 0].plot(epochs, builtin_hist['train_loss'], label='PyTorch Adam', color='darkorange', linewidth=2, linestyle='--')
    axes[0, 0].set_title('Training Loss vs Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ── Test Loss ──
    axes[0, 1].plot(epochs, custom_hist['test_loss'],  label='Custom Adam',  color='steelblue',  linewidth=2)
    axes[0, 1].plot(epochs, builtin_hist['test_loss'], label='PyTorch Adam', color='darkorange', linewidth=2, linestyle='--')
    axes[0, 1].set_title('Test Loss vs Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ── Train Accuracy ──
    axes[1, 0].plot(epochs, custom_hist['train_acc'],  label='Custom Adam',  color='steelblue',  linewidth=2)
    axes[1, 0].plot(epochs, builtin_hist['train_acc'], label='PyTorch Adam', color='darkorange', linewidth=2, linestyle='--')
    axes[1, 0].set_title('Training Accuracy vs Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ── Test Accuracy ──
    axes[1, 1].plot(epochs, custom_hist['test_acc'],  label='Custom Adam',  color='steelblue',  linewidth=2)
    axes[1, 1].plot(epochs, builtin_hist['test_acc'], label='PyTorch Adam', color='darkorange', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Test Accuracy vs Epochs')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved to: {save_path}")


def plot_ablation(ablation_results, ablation_configs, save_path='ablation_plots.png'):
    """
    Plot the effect of different (beta1, beta2) settings on convergence and stability.

    Args:
        ablation_results : dict mapping label -> history dict
        ablation_configs : list of (label, beta1, beta2) tuples (from ablation.py)
        save_path        : output file path
    """
    epochs = range(1, len(next(iter(ablation_results.values()))['train_loss']) + 1)
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ablation Study: Effect of β₁ and β₂ on Custom Adam\n(MNIST, MLP)',
                 fontsize=13, fontweight='bold')

    # ── Training Loss ──
    for (label, _, _), color in zip(ablation_configs, colors):
        hist = ablation_results[label]
        axes[0].plot(epochs, hist['train_loss'], label=label, color=color, linewidth=2)
    axes[0].set_title('Training Loss — Effect of β₁, β₂')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Test Accuracy ──
    for (label, _, _), color in zip(ablation_configs, colors):
        hist = ablation_results[label]
        axes[1].plot(epochs, hist['test_acc'], label=label, color=color, linewidth=2)
    axes[1].set_title('Test Accuracy — Effect of β₁, β₂')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Ablation plot saved to: {save_path}")