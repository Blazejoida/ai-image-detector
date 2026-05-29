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


def train_model(model, loader, optimizer, criterion, epochs, label="", val_loader=None):
    history = {"epoch": [], "loss": [], "accuracy": []}
    if val_loader is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []

    for epoch in range(epochs):
        avg_loss, acc = train_one_epoch(model, loader, optimizer, criterion)
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)

        if val_loader is not None:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
                    val_outputs = model(val_images)
                    v_loss = criterion(val_outputs, val_labels)
                    val_loss += v_loss.item()
                    val_correct += val_outputs.argmax(dim=1).eq(val_labels).sum().item()
                    val_total += val_labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(val_acc)

            print(f"[{label}] Epoch {epoch+1}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
        else:
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
