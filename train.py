import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

from config import (
    DEVICE, MODEL_DIR,
    EPOCHS_BASELINE, EPOCHS_FINETUNE,
    LR_BASELINE, LR_FINETUNE,
)
from model.dataset       import get_loaders
from model.architectures import BaselineCNN, build_resnet
from model.trainer       import train_model, evaluate_model
from model.reporting     import save_results, save_plots

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Device: {DEVICE}\n")

train_loader, test_loader, class_names = get_loaders()
criterion = nn.CrossEntropyLoss()

# Step 1: Baseline CNN 
print("=" * 50)
print("Step 1: Baseline CNN (trained from scratch)")
print("=" * 50)

baseline = BaselineCNN().to(DEVICE)
opt_b = optim.Adam(baseline.parameters(), lr=LR_BASELINE)
hist_b = train_model(
    baseline, train_loader, opt_b, criterion,
    EPOCHS_BASELINE, label="Baseline", val_loader=test_loader
)
acc_b, f1_b = evaluate_model(baseline, test_loader, class_names, "Baseline CNN")
torch.save(baseline.state_dict(), os.path.join(MODEL_DIR, "baseline.pth"))

# Save baseline epoch logs to CSV
pd.DataFrame(hist_b).to_csv(os.path.join(MODEL_DIR, "baseline_epoch_logs.csv"), index=False)
print(f"Saved epoch logs to {os.path.join(MODEL_DIR, 'baseline_epoch_logs.csv')}")

# Step 2: ResNet50 (transfer learning + fine-tuning) 
print("\n" + "=" * 50)
print("Step 2: ResNet50 fine-tuning (transfer learning)")
print("=" * 50)

resnet = build_resnet(freeze_backbone=True).to(DEVICE)
trainable = filter(lambda p: p.requires_grad, resnet.parameters())
opt_r = optim.Adam(trainable, lr=LR_FINETUNE)
hist_r = train_model(
    resnet, train_loader, opt_r, criterion,
    EPOCHS_FINETUNE, label="ResNet50", val_loader=test_loader
)
acc_r, f1_r = evaluate_model(resnet, test_loader, class_names, "ResNet50")
torch.save(resnet.state_dict(), os.path.join(MODEL_DIR, "model.pth"))

# Save resnet epoch logs to CSV
pd.DataFrame(hist_r).to_csv(os.path.join(MODEL_DIR, "resnet_epoch_logs.csv"), index=False)
print(f"Saved epoch logs to {os.path.join(MODEL_DIR, 'resnet_epoch_logs.csv')}")

# Step 3: Save results and plots 
print("\n" + "=" * 50)
print("Step 3: Results")
print("=" * 50)
save_results(acc_b, f1_b, acc_r, f1_r)
save_plots(hist_b, hist_r)
print("\nDone. Best model saved to model/model.pth")
