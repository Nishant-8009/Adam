"""
trainer.py — Training pipeline: per-epoch train loop, evaluation, and full experiment run.
"""

import torch
import torch.nn as nn

from config    import DEVICE, SEED
from model     import MLP
from optimizer import AdamOptimizer


def train_one_epoch(model, optimizer, train_loader, criterion, is_custom):
    """
    Train the model for one epoch.

    Args:
        model       : neural network
        optimizer   : custom AdamOptimizer or torch.optim optimizer
        train_loader: DataLoader for training data
        criterion   : loss function
        is_custom   : bool – True if using our custom optimizer

    Returns:
        avg_loss    : mean training loss over the epoch
        accuracy    : training accuracy (%)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += images.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, test_loader, criterion):
    """
    Evaluate the model on the test set.

    Returns:
        avg_loss : mean test loss
        accuracy : test accuracy (%)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct      += predicted.eq(labels).sum().item()
            total        += images.size(0)

    return total_loss / total, 100.0 * correct / total


def run_experiment(label, use_custom, train_loader, test_loader,
                   lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, epochs=10):
    """
    Full training run for one optimizer configuration.

    Args:
        label      : string label for this run (for logging/plots)
        use_custom : True  → use our custom Adam
                     False → use torch.optim.Adam (for comparison)
        train_loader, test_loader : DataLoaders
        lr, beta1, beta2, eps     : Adam hyperparameters
        epochs                    : number of training epochs

    Returns:
        dict with keys: 'train_loss', 'test_loss', 'train_acc', 'test_acc'
    """
    torch.manual_seed(SEED)   # same init for fair comparison

    model     = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if use_custom:
        optimizer = AdamOptimizer(
            model.parameters(), lr=lr, beta1=beta1, beta2=beta2, eps=eps
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps
        )

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, optimizer, train_loader, criterion, is_custom=use_custom
        )
        te_loss, te_acc = evaluate(model, test_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)

        print(f"[{label}] Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}% | "
              f"Test  Loss: {te_loss:.4f}  Acc: {te_acc:.2f}%")

    return history