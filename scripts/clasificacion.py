#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inferencia multi-etiqueta con modelo rouberta fine-tuned.

- Carga un modelo entrenado (multi-label) desde un directorio.
- Lee un CSV con textos (columna especificada).
- Limpia texto de forma consistente con el entrenamiento.
- Genera probabilidades y predicciones binarias (umbral configurable).
- Guarda un CSV con las columnas originales + columnas de etiquetas predichas.

Uso típico desde la raíz del repo:

    python inference/predict_multilabel_rouberta.py \
        --model-dir models/rouberta_finetuned_multilabel/final_model \
        --input-file data/processed/test_conjunto.csv \
        --text-column texto \
        --output-file data/processed/clasificacion_final.csv

Requisitos:
    pip install torch transformers datasets pandas numpy tqdm emoji
"""

import os
import re
import sys
from pathlib import Path

import emoji
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =====================================================
#  CONFIG BÁSICA
# =====================================================

LABELS = ["advocacy", "atack", "image", "issue", "call to action", "ceremonial"]


def get_repo_root(current_file: Path) -> Path:
    """
    Intenta inferir la raíz del repo:
      - Si el script está en 'inference/' o 'training/' o 'scripts/', sube un nivel.
      - En cualquier otro caso, usa la carpeta del archivo.
    """
    parent = current_file.resolve().parent
    if parent.name in ("inference", "training", "scripts"):
        return parent.parent
    return parent


# =====================================================
#  LIMPIEZA DE TEXTO (consistente con entrenamiento)
# =====================================================

def clean_text(text: str) -> str:
    """
    Limpieza de texto:
      - convierte a string
      - elimina emojis
      - borra '#', saltos de línea y tabs
      - colapsa espacios
      - pasa a minúsculas
      - normaliza a ASCII
    """
    text = str(text)
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"#", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.lower()
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text.strip()


# =====================================================
#  PARÁMETROS CLI
# =====================================================

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia multi-etiqueta con rouberta fine-tuned."
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/rouberta_finetuned_multilabel/final_model",
        help="Directorio del modelo fine-tuned (por defecto: models/rouberta_finetuned_multilabel/final_model).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/test_conjunto.csv",
        help="CSV con textos a clasificar (por defecto: data/processed/test_conjunto.csv).",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="texto",
        help="Nombre de la columna de texto en el CSV de entrada (default: 'texto').",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/clasificacion_multilabel.csv",
        help="Ruta del CSV de salida con las predicciones (default: data/processed/clasificacion_multilabel.csv).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=120,
        help="Longitud máxima de tokens para el tokenizer (default: 120).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.36,
        help="Umbral para binarizar probabilidades (default: 0.36).",
    )

    return parser.parse_args()


# =====================================================
#  MAIN
# =====================================================

def main():
    current_file = Path(__file__)
    repo_root = get_repo_root(current_file)
    args = parse_args()

    model_dir = (repo_root / args.model_dir).resolve()
    input_file = (repo_root / args.input_file).resolve()
    output_file = (repo_root / args.output_file).resolve()

    print("\n==============================")
    print("   INFERENCIA MULTI-ETIQUETA  ")
    print("==============================\n")
    print(f"Repo root:   {repo_root}")
    print(f"Modelo dir:  {model_dir}")
    print(f"Input CSV:   {input_file}")
    print(f"Output CSV:  {output_file}")
    print(f"Columna tex: {args.text_column}")
    print(f"MAX_LENGTH:  {args.max_length}")
    print(f"Threshold:   {args.threshold}")
    print(f"Labels:      {LABELS}\n")

    # --- Verificaciones básicas ---
    if not model_dir.exists():
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {model_dir}")
    if not input_file.exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo CSV de entrada en: {input_file}")

    # --- Cargar modelo y tokenizer ---
    print("🔄 Cargando modelo y tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    except Exception as e:
        raise RuntimeError(f"❌ Error al cargar el modelo/tokenizer desde {model_dir}: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✅ Modelo cargado en dispositivo: {device}\n")

    # --- Leer CSV y preparar textos ---
    print("📄 Cargando CSV de entrada...")
    df = pd.read_csv(input_file)

    if args.text_column not in df.columns:
        raise ValueError(f"❌ La columna '{args.text_column}' no existe en el archivo CSV.")

    # Creamos una columna interna 'text' limpia
    texts = df[args.text_column].fillna("").astype(str).tolist()
    if len(texts) == 0:
        raise ValueError("❌ El archivo CSV no contiene textos para clasificar.")

    print(f"🔢 {len(texts)} textos detectados. Limpiando textos...\n")
    texts_clean = [clean_text(t) for t in texts]

    # --- Inferencia uno a uno (modo seguro) ---
    print("🤖 Clasificando texto por texto (modo seguro)...\n")
    all_probs = []

    with torch.no_grad():
        for idx, text in tqdm(
            list(enumerate(texts_clean)),
            total=len(texts_clean),
            desc="🧠 Inferencia"
        ):
            try:
                encoding = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

                output = model(input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(output.logits).cpu().numpy()[0]
                all_probs.append(probs)
            except Exception as e:
                print(f"❌ Error en la fila {idx}: {e}")
                # mantengo la forma con NaNs
                all_probs.append(np.full(len(LABELS), np.nan))

    if len(all_probs) != len(texts_clean):
        raise RuntimeError(
            f"❌ Cantidad de predicciones ({len(all_probs)}) "
            f"no coincide con cantidad de textos ({len(texts_clean)})."
        )

    # --- Convertir a DataFrame y aplicar umbral ---
    print("\n💾 Construyendo DataFrame de resultados...")
    df_probs = pd.DataFrame(all_probs, columns=[f"prob_{l}" for l in LABELS])
    df_bin = (df_probs.values >= args.threshold).astype(int)
    df_bin = pd.DataFrame(df_bin, columns=LABELS)

    # --- Combinar con el DF original ---
    df_out = pd.concat(
        [df.reset_index(drop=True), df_probs.reset_index(drop=True), df_bin.reset_index(drop=True)],
        axis=1,
    )

    # Crear carpeta de salida si no existe
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(output_file, index=False)
    print(f"✅ Clasificación terminada. Resultados guardados en:\n   {output_file}\n")


if __name__ == "__main__":
    main()
