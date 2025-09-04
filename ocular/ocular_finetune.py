"""
ocular_finetune_ptflops.py
-----------------------------------
Fine-Tuning en imágenes oculares (ej: ODIR-5K) con PyTorch.

Características:
- Modelos: ResNet50, MobileNetV2, DenseNet121, InceptionV3
- Configuración vía argparse
- Control de GPU/CPU (incluye DataParallel para multi-GPU)
- Fine-tuning real: entrena cabeza + capas profundas
- Cálculo de FLOPs y parámetros con ptflops
- Registro de tiempo y consumo energético

--------------------------------------------------------------------------------------------------------------
CLI:

python ocular_finetune.py --model resnet50 --epochs 30 --batch_size 32 --device cuda:0
python ocular_finetune.py --model mobilenet_v2 --epochs 30 --batch_size 32 --device cuda:0
python ocular_finetune.py --model densenet121 --epochs 30 --batch_size 32 --device cuda:0
python ocular_finetune.py --model inception_v3 --epochs 30 --batch_size 32 --device cuda:0
"""

# --------------------------------------------------------------------------------------------------------------
# 1. Librerías

import os
import time
import json
import subprocess
import threading
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from ptflops import get_model_complexity_info
import argparse

# --------------------------------------------------------------------------------------------------------------
# 2. Argumentos CLI

parser = argparse.ArgumentParser(description="Fine-Tuning en imágenes oculares con PyTorch")

parser.add_argument("--model", type=str, default="resnet50",
                    choices=["resnet50", "mobilenet_v2", "densenet121", "inception_v3"],
                    help="Modelo a usar")
parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate inicial")
parser.add_argument("--device", type=str, default="cuda:0",
                    help="Dispositivo: 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...")
parser.add_argument("--data_dir", type=str, default="data/ODIR-5K",
                    help="Ruta al dataset con carpetas train/val")

args = parser.parse_args()

# --------------------------------------------------------------------------------------------------------------
# 3. Configuración

DATA_DIR = args.data_dir
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
MODEL_NAME = args.model

# Selección de device
if args.device == "cpu":
    DEVICE = torch.device("cpu")
elif args.device.startswith("cuda"):
    if torch.cuda.is_available():
        DEVICE = torch.device(args.device)
    else:
        raise ValueError("CUDA no disponible")
else:
    raise ValueError("Opción de device inválida")

# --------------------------------------------------------------------------------------------------------------
# 4. Transformaciones de datos

input_size = 224 if MODEL_NAME != "inception_v3" else 299

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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

# --------------------------------------------------------------------------------------------------------------
# 5. Dataset

image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                  shuffle=True, num_workers=4)
    for x in ["train", "val"]
}
class_names = image_datasets["train"].classes
num_classes = len(class_names)

# --------------------------------------------------------------------------------------------------------------
# 6. Inicialización de modelos

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

# Crear modelo
model = initialize_model(MODEL_NAME, num_classes)

# Multi-GPU si se solicita
if args.device == "cuda" and torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs con DataParallel")
    model = torch.nn.DataParallel(model)

model = model.to(DEVICE)

# --------------------------------------------------------------------------------------------------------------
# 7. Fine-Tuning setup

# Primero congelamos todo
for param in model.parameters():
    param.requires_grad = False

# Descongelamos última capa
if MODEL_NAME == "resnet50":
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
elif MODEL_NAME == "mobilenet_v2":
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
elif MODEL_NAME == "densenet121":
    for name, param in model.named_parameters():
        if "denseblock4" in name or "classifier" in name:
            param.requires_grad = True
elif MODEL_NAME == "inception_v3":
    for name, param in model.named_parameters():
        if "Mixed_7" in name or "fc" in name:
            param.requires_grad = True

# Optimizer solo para capas entrenables
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss()

# --------------------------------------------------------------------------------------------------------------
# 8. FLOPs y parámetros

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        model,
        (3, input_size, input_size),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )
print(f"🔹 FLOPs: {macs}, Parámetros: {params}")

# --------------------------------------------------------------------------------------------------------------
# 9. Logger Energético

def log_power(log_file="power_log.csv"):
    with open(log_file, "w") as f:
        subprocess.run(
            ["nvidia-smi", "--loop=1", "--query-gpu=timestamp,power.draw", "--format=csv"],
            stdout=f
        )
power_thread = threading.Thread(target=log_power, daemon=True)
power_thread.start()

# --------------------------------------------------------------------------------------------------------------
# 10. Entrenamiento

def train_model(model, dataloaders, criterion, optimizer, num_epochs=EPOCHS):
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            y_true, y_pred = [], []

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
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), f"best_{MODEL_NAME}_finetuned.pth")

    total_time = time.time() - start_time
    return model, history, total_time, y_true, y_pred

trained_model, hist, train_time, y_true, y_pred = train_model(model, dataloaders, criterion, optimizer)

# --------------------------------------------------------------------------------------------------------------
# 11. Guardar resultados

results = {
    "model": MODEL_NAME,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "train_time_sec": train_time,
    "FLOPs": macs,
    "params": params,
    "device": str(DEVICE),
    "mode": "Fine-Tuning"
}

with open(f"results_{MODEL_NAME}_finetuned.txt", "w") as f:
    for k, v in results.items():
        f.write(f"{k}: {v}\n")

with open(f"metadata_{MODEL_NAME}_finetuned.json", "w") as f:
    json.dump(results, f, indent=4)

# --------------------------------------------------------------------------------------------------------------
# 12. Gráficas

plt.figure()
plt.plot(hist["train_loss"], label="Train Loss")
plt.plot(hist["val_loss"], label="Val Loss")
plt.legend()
plt.title(f"{MODEL_NAME} Fine-Tuning Loss")
plt.savefig(f"loss_curve_{MODEL_NAME}_finetuned.png")

plt.figure()
plt.plot(hist["val_acc"], label="Val Accuracy")
plt.legend()
plt.title(f"{MODEL_NAME} Fine-Tuning Accuracy")
plt.savefig(f"accuracy_curve_{MODEL_NAME}_finetuned.png")

# --------------------------------------------------------------------------------------------------------------
# 13. Matriz de confusión

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title(f"Confusion Matrix {MODEL_NAME} Fine-Tuning")
plt.savefig(f"confusion_matrix_{MODEL_NAME}_finetuned.png")

pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(f"confusion_matrix_{MODEL_NAME}_finetuned.csv")

print("Fine-Tuning completado y resultados guardados.")
