# -*- coding: utf-8 -*-
"""
Transcripción de audios de anuncios (Meta) usando Whisper "base".

- Escanea una carpeta con audios (media/audio/<label>/ por defecto).
- Extrae ad_id del nombre del archivo.
- Transcribe con Whisper.

- Guarda:
    * Un .txt por anuncio: <ad_id>.txt
    * Un JSON y un Excel consolidados:
        - transcripciones.json
        - transcripciones.xlsx

Requisitos:
    pip install openai-whisper pandas openpyxl

Además, Whisper requiere ffmpeg instalado y disponible en el PATH.

Uso típico desde la raíz del repo:

    # Balotaje 2024
    python src/transcripcion_whisper.py --label balotaje_2024

    # Especificando carpetas
    python src/transcripcion_whisper.py \\
        --label balotaje_2024 \\
        --input-dir media/audio/balotaje_2024 \\
        --output-dir ocr_transcripts/whisper/balotaje_2024

Estructura esperada del repo (por defecto):

    repo_root/
      src/transcripcion_whisper.py
      media/audio/<label>/        # audios .mp3, .wav, etc.
      ocr_transcripts/whisper/<label>/   # salidas de texto
"""

import os
import sys
import time
import json
import re
import logging
import argparse
from pathlib import Path

import pandas as pd
import whisper
import torch

# Extensiones de audio válidas
ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".aac", ".flac", ".ogg"}


# ==================== HELPERS DE RUTAS / LOG ====================

def get_repo_root(current_file: Path) -> Path:
    """
    Intenta inferir la raíz del repo:
    - Si el script está en src/, sube un nivel.
    - En caso contrario, usa la carpeta del archivo.
    """
    parent = current_file.resolve().parent
    if parent.name == "src":
        return parent.parent
    return parent


def setup_logging(output_dir: Path) -> Path:
    """
    Configura logging a archivo + consola.
    Crea un log con timestamp en la carpeta de salida.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"log_whisper_{int(time.time())}.txt"

    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S")
    )
    logging.getLogger().addHandler(console)

    logging.info(f"Logging inicializado. Archivo: {log_path}")
    return log_path


def log(msg, level=logging.INFO):
    logging.log(level, msg)


# ==================== FUNCIONES AUXILIARES ====================

def extract_ad_id_from_filename(path: Path):
    """
    Extrae el ad_id del nombre del archivo buscando secuencias numéricas.
    - Si el nombre es puramente numérico → lo toma completo.
    - Si hay varias secuencias → toma la numérica más larga.
    """
    stem = path.stem.strip()
    if re.fullmatch(r"\d+", stem):
        return stem

    cands = re.findall(r"\d+", stem)
    if cands:
        cands.sort(key=len, reverse=True)
        return cands[0]

    return None


def load_existing(out_dir: Path):
    """
    Carga JSON existente de transcripciones para continuar sin repetir trabajo.
    Devuelve dict {ad_id: transcripcion}.
    """
    file = out_dir / "transcripciones.json"
    if not file.exists():
        return {}

    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        return {str(r["ad_id"]): r["transcripcion"] for r in data if "ad_id" in r}
    except Exception as e:
        log(f"[WARN] No se pudo leer JSON existente: {e}")
        return {}


def save_consolidated(rows, out_dir: Path):
    """
    Guarda un JSON y un Excel con las transcripciones.
    """
    json_path = out_dir / "transcripciones.json"
    xlsx_path = out_dir / "transcripciones.xlsx"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.drop_duplicates(subset=["ad_id"], keep="last", inplace=True)
        df.to_excel(xlsx_path, index=False)
        log(f"[SAVE] Actualizados archivos -> {json_path.name} | {xlsx_path.name}")


# ==================== PARÁMETROS CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audios de anuncios usando Whisper base."
    )
    parser.add_argument(
        "--label",
        type=str,
        default="balotaje_2024",
        help="Etiqueta de la corrida (ej.: internas_2024, balotaje_2024, elecciones_2024).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Directorio con los audios a transcribir. "
            "Por defecto: media/audio/<label>/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directorio de salida para transcripciones. "
            "Por defecto: ocr_transcripts/whisper/<label>/"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="large-v3",
        help="Nombre del modelo Whisper a usar (default: large-v3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Dispositivo: 'cpu' o 'cuda'. "
            "Por defecto: autodetección (cuda si hay GPU, sino cpu)."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        help="Idioma a forzar en la transcripción (default: es).",
    )
    return parser.parse_args()


# ==================== MAIN ====================

def main():
    current_file = Path(__file__)
    repo_root = get_repo_root(current_file)
    args = parse_args()

    # Resolución de rutas por defecto
    if args.input_dir is None:
        input_dir = repo_root / "media" / "audio" / args.label
    else:
        input_dir = Path(args.input_dir)

    if args.output_dir is None:
        output_dir = repo_root / "ocr_transcripts" / "whisper" / args.label
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(output_dir)

    log("=== INICIO TRANSCRIPCIÓN WHISPER ===")
    log(f"Label: {args.label}")
    log(f"Directorio entrada audios: {input_dir}")
    log(f"Directorio salida transcripciones: {output_dir}")
    log(f"Log: {log_path}")

    if not input_dir.exists() or not input_dir.is_dir():
        log(f"[ERROR] Carpeta de entrada no válida: {input_dir}", logging.ERROR)
        sys.exit(1)

    # Detección de dispositivo
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"[INIT] Cargando modelo Whisper '{args.model_name}' en device={device}...")
    try:
        model = whisper.load_model(args.model_name, device=device)
    except Exception as e:
        log(f"[FATAL] No se pudo cargar el modelo Whisper: {e}", logging.ERROR)
        sys.exit(1)

    # Escaneo de archivos de audio
    files = [
        p for p in input_dir.glob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    ]
    log(f"[INFO] Se encontraron {len(files)} archivos de audio.")

    if not files:
        log("[ERROR] No hay archivos válidos en la carpeta de entrada.", logging.ERROR)
        sys.exit(0)

    # Cargar estado previo
    prev_data = load_existing(output_dir)
    done_ids = set(prev_data.keys())
    rows = [{"ad_id": k, "transcripcion": v} for k, v in prev_data.items()]
    log(f"[INFO] Transcripciones previas detectadas: {len(done_ids)}")

    ok, fail, skip = 0, 0, 0
    total = len(files)

    for i, audio_file in enumerate(files, start=1):
        log(f"[STEP {i}/{total}] Procesando archivo: {audio_file.name}")

        ad_id = extract_ad_id_from_filename(audio_file)
        if not ad_id:
            log(f"[SKIP] No se pudo extraer ID del archivo: {audio_file.name}", logging.WARNING)
            skip += 1
            continue

        if ad_id in done_ids:
            log(f"[SKIP] ID {ad_id} ya procesado previamente.")
            skip += 1
            continue

        try:
            t0 = time.time()
            # Transcripción con Whisper
            result = model.transcribe(
                str(audio_file),
                language=args.language,
                task="transcribe",
            )
            text = (result.get("text") or "").strip()

            # Guardar .txt individual
            (output_dir / f"{ad_id}.txt").write_text(text, encoding="utf-8")

            elapsed = time.time() - t0
            log(f"[OK] {ad_id}: {len(text)} caracteres | {elapsed:.1f}s")

            rows.append({"ad_id": ad_id, "transcripcion": text})
            done_ids.add(ad_id)

            # Guardar consolidado (incremental)
            save_consolidated(rows, output_dir)
            ok += 1

        except Exception as e:
            fail += 1
            log(f"[ERROR] Falló {audio_file.name}: {e}", logging.ERROR)

    # Resumen final
    log("=" * 60)
    log(f"[RESUMEN] TOTAL={total} | OK={ok} | FAIL={fail} | SKIP={skip}")
    log(f"[INFO] Archivos guardados en: {output_dir}")
    log(f"[INFO] Log: {log_path}")
    log("=== FIN TRANSCRIPCIÓN WHISPER ===")


if __name__ == "__main__":
    main()
