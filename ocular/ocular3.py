"""
ocular_finetune_advanced.py
-----------------------------------
Fine-Tuning en ODIR-5K con técnicas anti-overfitting.

Incluye:
- WeightedRandomSampler (--balanced)
- Early Stopping
- ReduceLROnPlateau LR Scheduler
- Weight Decay
- Opción de Focal Loss (--focal)
- Data Augmentation más fuerte
- FLOPs con ptflops
- Power logging encapsulado
"""

import os
import time
import json
import random
import numpy as np
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from ptflops import get_model_complexity_info
import argparse
from collections import Counter

# ==============================
# 1. Argumentos CLI
# ==============================
parser = argparse.ArgumentParser(description="Fine-Tuning Ocular con mejoras")

parser.add_argument("--model", type=str, default="resnet50",
                    choices=["resnet50", "mobilenet_v2", "densenet121", "inception_v3"])
parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate inicial")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay para regularización")
parser.add_argument("--device", type=str, default="cuda:0", help="'cpu' o 'cuda:0'")
parser.add_argument("--data_dir", type=str, default="data/ODIR-5K", help="Ruta dataset train/val")
parser.add_argument("--seed", type=int, default=42, help="Semilla")
parser.add_argument("--balanced", action="store_true", help="WeightedRandomSampler")
parser.add_argument("--focal", action="store_true", help="Usar Focal Loss en vez de CrossEntropy")
parser.add_argument("--patience", type=int, default=7, help="Paciencia para Early Stopping")

args = parser.parse_args()

# ==============================
# 2. Reproducibilidad
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# ==============================
# 3. Configuración
# ==============================
DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
MODEL_NAME = args.model
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr

print("\n===== Experimento de Fine-Tuning Avanzado =====")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("===============================================\n")

# ==============================
# 4. Transformaciones con augmentación fuerte
# ==============================
input_size = 224 if MODEL_NAME != "inception_v3" else 299

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# ==============================
# 5. Dataset + Sampler
# ==============================
image_datasets = {
    x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}

if args.balanced:
    targets = [y for _, y in image_datasets["train"]]
    class_counts = Counter(targets)
    class_weights = 1. / torch.tensor(list(class_counts.values()), dtype=torch.float)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(image_datasets["train"], batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
else:
    train_loader = DataLoader(image_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_loader = DataLoader(image_datasets["val"], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
dataloaders = {"train": train_loader, "val": val_loader}
class_names = image_datasets["train"].classes
num_classes = len(class_names)

# ==============================
# 6. Modelos
# ==============================
def initialize_model(name, num_classes, pretrained=True):
    if name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Modelo no soportado")
    return model

model = initialize_model(MODEL_NAME, num_classes).to(DEVICE)

# Fine-tuning: desbloquear últimos bloques
for param in model.parameters():
    param.requires_grad = False
if MODEL_NAME == "resnet50":
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True

# ==============================
# 7. Loss (CrossEntropy o Focal)
# ==============================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

criterion = FocalLoss() if args.focal else nn.CrossEntropyLoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.1)

# ==============================
# 8. FLOPs
# ==============================
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (3, input_size, input_size),
                                             as_strings=True, print_per_layer_stat=False, verbose=False)
print(f"🔹 FLOPs: {macs}, Parámetros: {params}")

# ==============================
# 9. Entrenamiento + Early Stopping
# ==============================
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=EPOCHS, patience=args.patience):
    power_log = open(f"power_train_{MODEL_NAME}.csv", "w")
    power_process = subprocess.Popen(
        ["nvidia-smi", "--loop=1", "--query-gpu=timestamp,power.draw", "--format=csv"],
        stdout=power_log
    )

    best_acc, no_improve = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, y_true, y_pred = 0.0, [], []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {acc:.4f} F1: {f1:.4f}")

            if phase == "train":
                history["train_loss"].append(epoch_loss)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(acc)
                scheduler.step(acc)
                if acc > best_acc:
                    best_acc, no_improve = acc, 0
                    torch.save(model.state_dict(), f"best_{MODEL_NAME}_finetuned.pth")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("⏹️ Early stopping activado")
                        total_time = time.time() - start_time
                        power_process.terminate()
                        power_log.close()
                        return model, history, total_time, y_true, y_pred

    total_time = time.time() - start_time
    power_process.terminate()
    power_log.close()
    return model, history, total_time, y_true, y_pred

trained_model, hist, train_time, y_true, y_pred = train_model(model, dataloaders, criterion, optimizer, scheduler)

# ==============================
# 10. Guardar resultados
# ==============================
results = {**vars(args),
           "train_time_sec": train_time,
           "FLOPs": macs,
           "params": params,
           "mode": "Fine-Tuning Avanzado"}

with open(f"results_{MODEL_NAME}_advanced.txt", "w") as f:
    for k, v in results.items():
        f.write(f"{k}: {v}\n")
with open(f"metadata_{MODEL_NAME}_advanced.json", "w") as f:
    json.dump(results, f, indent=4)

# ==============================
# 11. Curvas y Confusion Matrix
# ==============================
plt.figure()
plt.plot(hist["train_loss"], label="Train Loss")
plt.plot(hist["val_loss"], label="Val Loss")
plt.legend(); plt.title(f"{MODEL_NAME} Loss")
plt.savefig(f"loss_curve_{MODEL_NAME}_advanced.png")

plt.figure()
plt.plot(hist["val_acc"], label="Val Accuracy")
plt.legend(); plt.title(f"{MODEL_NAME} Accuracy")
plt.savefig(f"accuracy_curve_{MODEL_NAME}_advanced.png")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel("True"); plt.xlabel("Predicted")
plt.title(f"Confusion Matrix {MODEL_NAME}")
plt.savefig(f"confusion_matrix_{MODEL_NAME}_advanced.png")

pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(f"confusion_matrix_{MODEL_NAME}_advanced.csv")

print("✅ Fine-Tuning Avanzado completado y resultados guardados.")
