#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrenamiento de un modelo multi-etiqueta (rouberta) para tipología de anuncios.

- Toma un CSV de train y uno de validación con columnas:
    * 'texto'
    * columnas de labels binarias (0/1): advocacy, atack, image, issue, call to action, ceremonial

- Limpia texto
- Tokeniza con pln-udelar/rouberta-base-uy22-cased (por defecto)
- Entrena modelo multi-label con Hugging Face Trainer
- Guarda:
    * modelo final
    * tokenizer
    * métricas de evaluación (JSON)
    * log de entrenamiento (txt)

Uso típico desde la raíz del repo:

    python training/train_multilabel_rouberta.py \
        --train-file data/processed/train_conjunto.csv \
        --val-file   data/processed/val_conjunto.csv \
        --output-dir models/rouberta_multilabel_v1

"""

import os
import re
import sys
import json
import emoji
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    set_seed,
    TrainingArguments,
    Trainer,
)


# =====================================================
#  UTILIDADES DE RUTAS
# =====================================================

def get_repo_root(current_file: Path) -> Path:
    """
    Intenta inferir la raíz del repo:
      - Si el script está en 'training/' o 'scripts/', sube un nivel.
      - En cualquier otro caso, usa la carpeta del archivo.
    """
    parent = current_file.resolve().parent
    if parent.name in ("training", "scripts"):
        return parent.parent
    return parent


# =====================================================
#  LIMPIEZA DE TEXTO
# =====================================================

def clean_text(text):
    text = str(text)
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"#", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.lower()
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text.strip()


# =====================================================
#  MÉTRICAS
# =====================================================

def compute_metrics_builder(label_names):
    """
    Devuelve una función compute_metrics que usa label_names
    para armar métricas por etiqueta.
    """

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        labels_int = labels.astype(int)

        f1_macro = f1_score(labels_int, preds, average="macro")
        accs = {
            f"accuracy_{label}": accuracy_score(labels_int[:, i], preds[:, i])
            for i, label in enumerate(label_names)
        }

        return {"f1_macro": f1_macro, **accs}

    return compute_metrics


# =====================================================
#  LOGGER A ARCHIVO + CONSOLA
# =====================================================

class Logger(object):
    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, message):
        self.stream.write(message)
        self.logfile.write(message)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()


# =====================================================
#  MAIN
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning multi-etiqueta con rouberta para tipología de anuncios."
    )

    # Paths de datos
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/processed/train_conjunto.csv",
        help="Ruta al CSV de entrenamiento (por defecto: data/processed/train_conjunto.csv).",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/processed/val_conjunto.csv",
        help="Ruta al CSV de validación (por defecto: data/processed/val_conjunto.csv).",
    )

    # Salida
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/rouberta_finetuned_multilabel",
        help="Directorio base de salida para el modelo y logs.",
    )

    # Modelo
    parser.add_argument(
        "--model-name",
        type=str,
        default="pln-udelar/rouberta-base-uy22-cased",
        help="Nombre o path del modelo base de Hugging Face.",
    )

    # Hiperparámetros
    parser.add_argument("--num-epochs", type=int, default=4, help="Número de epochs (default: 4).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size por dispositivo (default: 16).")
    parser.add_argument("--max-length", type=int, default=125, help="Longitud máxima de tokens (default: 125).")
    parser.add_argument("--seed", type=int, default=1891, help="Semilla de aleatoriedad (default: 1891).")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps (default: 100).")

    return parser.parse_args()


def main():
    current_file = Path(__file__)
    repo_root = get_repo_root(current_file)
    args = parse_args()

    # Labels de la tarea (en el mismo orden que columnas del CSV)
    LABELS = ["advocacy", "atack", "image", "issue", "call to action", "ceremonial"]
    num_labels = len(LABELS)

    # Resolver rutas relativas al repo
    train_path = (repo_root / args.train_file).resolve()
    val_path   = (repo_root / args.val_file).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    os.makedirs(output_dir, exist_ok=True)

    # Logging
    log_path = output_dir / "training_log.txt"
    metrics_path = output_dir / "eval_metrics.json"

    # Redirigir stdout/stderr a archivo + consola
    with open(log_path, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Logger(sys.stdout, log_file)
        sys.stderr = Logger(sys.stderr, log_file)

        print("\n==============================")
        print("  CONFIGURACIÓN DE ENTRENAMIENTO")
        print("==============================\n")
        print(f"Repo root:      {repo_root}")
        print(f"Train file:     {train_path}")
        print(f"Val file:       {val_path}")
        print(f"Output dir:     {output_dir}")
        print(f"Modelo base:    {args.model_name}")
        print(f"Labels:         {LABELS}")
        print(f"Num epochs:     {args.num_epochs}")
        print(f"Batch size:     {args.batch_size}")
        print(f"Max length:     {args.max_length}")
        print(f"Seed:           {args.seed}")
        print(f"Learning rate:  {args.learning_rate}")
        print(f"Warmup steps:   {args.warmup_steps}\n")

        # =====================================================
        #  CARGA DE DATOS
        # =====================================================
        print("📥 Cargando datos...")

        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(val_path)

        # Limpieza de texto
        train_df["text"] = train_df["texto"].astype(str).apply(clean_text)
        valid_df["text"] = valid_df["texto"].astype(str).apply(clean_text)

        # Construir columna 'labels' como lista de floats por fila
        train_df["labels"] = train_df[LABELS].values.tolist()
        valid_df["labels"] = valid_df[LABELS].values.tolist()

        train_df = train_df.drop(columns=LABELS + ["texto"])
        valid_df = valid_df.drop(columns=LABELS + ["texto"])

        train_df["labels"] = train_df["labels"].apply(lambda x: [float(i) for i in x])
        valid_df["labels"] = valid_df["labels"].apply(lambda x: [float(i) for i in x])

        train_dataset = Dataset.from_pandas(train_df)
        valid_dataset = Dataset.from_pandas(valid_df)

        if "__index_level_0__" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns("__index_level_0__")
            valid_dataset = valid_dataset.remove_columns("__index_level_0__")

        # =====================================================
        #  TOKENIZACIÓN Y MODELO
        # =====================================================
        print("\n🔧 Preparando modelo y tokenizador...\n")

        set_seed(args.seed)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            )

        train_dataset = train_dataset.map(tokenize, batched=True)
        valid_dataset = valid_dataset.map(tokenize, batched=True)

        train_dataset = train_dataset.remove_columns("text")
        valid_dataset = valid_dataset.remove_columns("text")

        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        valid_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # =====================================================
        #  TRAINING ARGUMENTS
        # =====================================================
        print("⚙️ Configurando Trainer...\n")

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=50,
            seed=args.seed,
            save_total_limit=3,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
        )

        compute_metrics = compute_metrics_builder(LABELS)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
        )

        # =====================================================
        #  ENTRENAMIENTO
        # =====================================================
        print("\n🚀 INICIANDO ENTRENAMIENTO...\n")
        trainer.train()
        print("\n🏁 ENTRENAMIENTO COMPLETADO.\n")

        # =====================================================
        #  EVALUACIÓN
        # =====================================================
        print("\n📊 EVALUANDO MODELO FINAL...\n")
        eval_metrics = trainer.evaluate()
        print("\n📊 MÉTRICAS FINALES:")
        print(eval_metrics)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=4)

        # =====================================================
        #  GUARDAR MODELO
        # =====================================================
        final_model_dir = output_dir / "final_model"
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        print("\n✅ Modelo guardado en:", final_model_dir)
        print("📁 Logs guardados en:", log_path)
        print("📁 Métricas guardadas en:", metrics_path)

        # Restaurar stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    main()
