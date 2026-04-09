import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import DEVICE


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += outputs.argmax(dim=1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train_model(model, loader, optimizer, criterion, epochs, label=""):
    history = {"epoch": [], "loss": [], "accuracy": []}

    for epoch in range(epochs):
        avg_loss, acc = train_one_epoch(model, loader, optimizer, criterion)
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)
        print(f"[{label}] Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")

    return history


def evaluate_model(model, loader, class_names, label=""):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n{'='*40}")
    print(f"{label} Results:")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1-score : {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return acc, f1
