"""
pip install kaggle
kaggle datasets download -d andrewmvd/ocular-disease-recognition-odir5k
kaggle datasets download andrewmvd/ocular-disease-recognition-odir5k
unzip ocular-disease-recognition-odir5k.zip -d data/ODIR-5K

python prepare_dataset.py

-------------------
Script para organizar ODIR-5K en carpetas train/ y val/
a partir del CSV de etiquetas.

Estructura de salida:
data/ODIR-5K/
    train/<clase>/*.jpg
    val/<clase>/*.jpg
"""

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


RAW_DIR = "data/ODIR-5K/raw"       # carpeta donde quedaron las imágenes originales
CSV_FILE = os.path.join(RAW_DIR, "labels.csv")  # ajusta según nombre real del CSV
OUTPUT_DIR = "data/ODIR-5K"        # carpeta de salida organizada
VAL_SPLIT = 0.2                    # 20% validación


df = pd.read_csv(CSV_FILE)

# Supongamos que las columnas son: ['Image', 'Label']
# donde 'Label' es la clase (Normal, Cataract, etc.)
# Si las etiquetas están en otra columna, ajusta aquí.
print(df.head())


train_df, val_df = train_test_split(
    df,
    test_size=VAL_SPLIT,
    stratify=df['Label'],
    random_state=42
)

def prepare_split(split_df, split_name):
    for _, row in split_df.iterrows():
        label = str(row['Label'])
        img_name = row['Image']
        src = os.path.join(RAW_DIR, img_name)
        dst_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)
        try:
            shutil.copy(src, dst_dir)
        except FileNotFoundError:
            print(f"Imagen no encontrada: {src}")

prepare_split(train_df, "train")
prepare_split(val_df, "val")

print(" Dataset preparado en carpetas train/ y val/")
"""
tree data/ODIR-5K/train -L 1

"""
