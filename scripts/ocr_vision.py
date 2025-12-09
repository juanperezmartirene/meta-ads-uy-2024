#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR por lotes con Google Cloud Vision API (REST, API key)

Objetivo:
- Recorrer una carpeta con imágenes nombradas como "{id}_imagen.*"
- Enviar las imágenes a Google Vision (DOCUMENT_TEXT_DETECTION por defecto)
- Recuperar el texto por imagen y una confianza promedio (cuando esté disponible)
- Exportar resultados a:
    * JSON (con meta + resultados + errores)
    * Excel (id, filename, text, avg_confidence)


Estructura esperada (por defecto):

    repo_root/
      src/ocr_vision.py
      media/images/<label>/         # {ad_id}_imagen.jpg
      ocr_transcripts/ocr/<label>/  # vision_ocr_results.json/.xlsx

Autenticación:
- Definir la variable de entorno VISION_API_KEY con tu API key de Google Cloud Vision
  (recomendado), por ejemplo:

    En PowerShell (Windows):
        $env:VISION_API_KEY = "TU_API_KEY"

    En bash (Linux/macOS):
        export VISION_API_KEY="TU_API_KEY"

- Opcionalmente, se puede pasar por parámetro: --api-key "TU_API_KEY"
"""

import os
import sys
import re
import json
import time
import base64
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

import requests
import pandas as pd

# -----------------------------
# Constantes
# -----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Extrae el ID del nombre "{id}_imagen.*" (ej.: "123456_imagen.jpg" → id="123456")
ID_PATTERN = re.compile(r"^(?P<id>.+?)_imagen(?:\.[A-Za-z0-9]+)?$", re.IGNORECASE)

# Endpoint REST de Vision API (v1)
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

# Tamaño máximo de lote por request
MAX_PER_REQUEST = 16


# -----------------------------
# Helpers de rutas y logging
# -----------------------------

def get_repo_root(current_file: Path) -> Path:
    """
    Intenta inferir la raíz del repo:
    - Si el script está en src/, sube un nivel.
    - Si no, usa el directorio del archivo.
    """
    parent = current_file.resolve().parent
    if parent.name == "src":
        return parent.parent
    return parent


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vision_ocr_{int(time.time())}.log"

    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logging.getLogger().addHandler(console)

    logging.info(f"Logging inicializado. Archivo: {log_path}")
    return log_path


# -----------------------------
# Utilitarios
# -----------------------------

def list_images(input_dir: Path, recursive: bool = False) -> List[Path]:
    """
    Lista las rutas de imágenes bajo input_dir.
    Si recursive=True, recorre subcarpetas; si no, solo el nivel actual.
    """
    files: List[Path] = []
    if recursive:
        for root, _, names in os.walk(input_dir):
            for n in names:
                ext = os.path.splitext(n)[1].lower()
                if ext in IMG_EXTS:
                    files.append(Path(root) / n)
    else:
        for n in os.listdir(input_dir):
            p = input_dir / n
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
    files.sort()
    return files


def extract_id_from_filename(path: Path) -> str:
    """
    Dado el path de una imagen, devuelve el id extraído del patrón "{id}_imagen.*".
    Si no cumple el patrón, devuelve cadena vacía.
    """
    stem = path.stem
    m = ID_PATTERN.match(stem)
    return m.group("id") if m else ""


def chunked(seq, size):
    """Generador que parte la secuencia en chunks de tamaño size."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def encode_image_b64(path: Path) -> str:
    """Lee una imagen binaria y la codifica en base64 (string)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_request_payload(
    batch_paths: List[Path],
    feature: str,
    language_hints: List[str]
) -> Dict[str, Any]:
    """
    Construye el payload JSON para Vision API (images:annotate).
    - feature: 'DOCUMENT_TEXT_DETECTION' o 'TEXT_DETECTION'
    - language_hints: lista de hints (ej.: ['es','en'])
    """
    feats = [{"type": feature}]
    requests_list = []
    for p in batch_paths:
        img_b64 = encode_image_b64(p)
        req = {
            "image": {"content": img_b64},
            "features": feats,
        }
        if language_hints:
            req["imageContext"] = {"languageHints": language_hints}
        requests_list.append(req)
    return {"requests": requests_list}


def vision_request(
    payload: Dict[str, Any],
    api_key: str,
    timeout: int = 60,
    max_retries: int = 5,
    backoff_base: float = 1.5
) -> Dict[str, Any]:
    """
    Envía POST a Vision API con reintentos exponenciales ante 429/5xx.
    - timeout: segundos por request
    - max_retries: reintentos máximos
    - backoff_base: base para exponenciación del backoff (1.5 → 1.5^n)
    Lanza excepción si falla definitivamente.
    """
    params = {"key": api_key}
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(VISION_ENDPOINT, params=params, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt >= max_retries:
                    resp.raise_for_status()
                sleep_s = backoff_base ** attempt + (0.05 * attempt)
                logging.warning(
                    f"HTTP {resp.status_code} en attempt {attempt}, reintentando en {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)
            else:
                resp.raise_for_status()

        except requests.RequestException as e:
            if attempt >= max_retries:
                logging.error(f"Error de red definitivo: {e}")
                raise
            sleep_s = backoff_base ** attempt + (0.05 * attempt)
            logging.warning(
                f"Error de red {e} en attempt {attempt}, reintentando en {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)


def parse_text_and_conf(result_obj: Dict[str, Any]) -> Tuple[str, float]:
    """
    Extrae el texto completo y una confianza promedio cuando sea posible.
    Regresa (texto, avg_confidence) con avg_confidence en [0,1] o -1.0 si no disponible.

    - Preferimos `fullTextAnnotation` (DOCUMENT_TEXT_DETECTION).
    - Fallback: `textAnnotations[0].description` (TEXT_DETECTION).
    """
    text = ""
    avg_conf = -1.0

    fta = result_obj.get("fullTextAnnotation")
    if fta and "text" in fta:
        text = fta.get("text", "")

        confidences = []
        for page in fta.get("pages", []):
            for block in page.get("blocks", []):
                if "confidence" in block:
                    confidences.append(block["confidence"])
                for par in block.get("paragraphs", []):
                    if "confidence" in par:
                        confidences.append(par["confidence"])
        if confidences:
            avg_conf = sum(confidences) / len(confidences)

    else:
        tanns = result_obj.get("textAnnotations", [])
        if tanns:
            text = tanns[0].get("description", "")

    return text.strip(), avg_conf


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_excel(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["id", "filename", "text", "avg_confidence"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.to_excel(path, index=False, engine="openpyxl")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR por lotes (Vision API) para imágenes '{id}_imagen.*'"
    )

    parser.add_argument(
        "--label",
        type=str,
        default="balotaje_2024",
        help="Etiqueta para la corrida (ej.: internas_2024, nacionales_2024, balotaje_2024).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Carpeta con las imágenes. "
            "Por defecto: media/images/<label>/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Carpeta base de salida. "
            "Por defecto: ocr_transcripts/ocr/<label>/"
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key de Google Cloud Vision. "
            "Por defecto se toma de la variable de entorno VISION_API_KEY."
        ),
    )
    parser.add_argument(
        "--feature",
        type=str,
        choices=["TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION"],
        default="DOCUMENT_TEXT_DETECTION",
        help="Tipo de detección a usar (default: DOCUMENT_TEXT_DETECTION).",
    )
    parser.add_argument(
        "--language-hints",
        nargs="*",
        default=["es"],
        help="Hints de idioma (ej.: es en). Default: ['es']",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Si se indica, busca también en subcarpetas del input-dir.",
    )
    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    current_file = Path(__file__)
    repo_root = get_repo_root(current_file)
    args = parse_args()

    # Resolver rutas por defecto
    if args.input_dir is None:
        input_dir = repo_root / "media" / "images" / args.label
    else:
        input_dir = Path(args.input_dir)

    if args.output_dir is None:
        output_dir = repo_root / "ocr_transcripts" / "ocr" / args.label
    else:
        output_dir = Path(args.output_dir)

    json_path = output_dir / "vision_ocr_results.json"
    xlsx_path = output_dir / "vision_ocr_results.xlsx"
    log_dir = repo_root / "logs"
    log_path = setup_logging(log_dir)

    logging.info("=== INICIO OCR VISION API ===")
    logging.info(f"Label: {args.label}")
    logging.info(f"Carpeta de entrada: {input_dir}")
    logging.info(f"Carpeta de salida:  {output_dir}")
    logging.info(f"Log file: {log_path}")

    # API key
    api_key = args.api_key or os.getenv("VISION_API_KEY")
    if not api_key:
        logging.error("ERROR: Proveé --api-key o definí la variable de entorno VISION_API_KEY.")
        sys.exit(1)

    # Validar carpeta de entrada
    if not input_dir.exists() or not input_dir.is_dir():
        logging.error(f"La carpeta de entrada no existe o no es un directorio: {input_dir}")
        sys.exit(2)

    # Listado de imágenes con patrón correcto
    images = list_images(input_dir, recursive=args.recursive)
    images_filtered = [p for p in images if extract_id_from_filename(p)]

    if not images_filtered:
        logging.error(f"No se encontraron imágenes con patrón '{{id}}_imagen.*' en {input_dir}")
        sys.exit(3)

    logging.info(f"Se encontraron {len(images_filtered)} imágenes válidas para OCR.")

    all_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for batch in chunked(images_filtered, MAX_PER_REQUEST):
        payload = build_request_payload(batch, args.feature, args.language_hints)
        try:
            resp_json = vision_request(payload, api_key=api_key)
        except Exception as e:
            logging.exception(
                "Fallo el request para el lote que inicia con: %s",
                os.path.basename(batch[0])
            )
            for p in batch:
                errors.append({"filename": str(p), "error": str(e)})
            continue

        responses = resp_json.get("responses", [])
        for p, r in zip(batch, responses):
            r = r if isinstance(r, dict) else {}
            _id = extract_id_from_filename(p)
            text, avg_conf = parse_text_and_conf(r)

            if not text and "error" in r:
                err = r.get("error", {})
                errors.append(
                    {
                        "filename": str(p),
                        "error": f"{err.get('code')}: {err.get('message')}",
                    }
                )

            all_rows.append(
                {
                    "id": _id,
                    "filename": str(p),
                    "text": text,
                    "avg_confidence": (
                        round(avg_conf, 4)
                        if isinstance(avg_conf, (int, float)) and avg_conf >= 0
                        else None
                    ),
                }
            )

    outputs = {
        "meta": {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "input_dir": str(input_dir),
            "recursive": args.recursive,
            "feature": args.feature,
            "language_hints": args.language_hints,
            "n_images": len(images_filtered),
            "endpoint": VISION_ENDPOINT,
        },
        "results": all_rows,
        "errors": errors,
    }

    write_json(json_path, outputs)
    write_excel(xlsx_path, all_rows)

    logging.info(f"JSON guardado en:  {json_path}")
    logging.info(f"Excel guardado en: {xlsx_path}")
    if errors:
        logging.warning(f"Hubo {len(errors)} errores. Revisá la sección 'errors' en el JSON.")
    logging.info("=== FIN OCR VISION API ===")


if __name__ == "__main__":
    main()
