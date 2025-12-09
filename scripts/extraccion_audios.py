# -*- coding: utf-8 -*-
"""
Descarga audios a partir de las video_url detectadas en la extracción de media
y extrae el audio con ffmpeg, dejando el ad_id en el título del archivo de audio.

Entrada:
    - CSV/Excel con columnas:
        * ad_id
        * video_url

Salida:
    - Carpeta con archivos MP3:
        * <OUTPUT_DIR>/<ad_id>.mp3

    - Usa una carpeta ALREADY_DIR para NO repetir audios que ya existen,
      leyendo el título (metadata) con mutagen.

Requisitos:
    pip install requests pandas mutagen

    ffmpeg instalado y disponible en PATH
    (o pasar la ruta con --ffmpeg-path)

Estructura esperada:
    repo_root/
      src/extraccion_audios.py
      data/intermediate/media_urls_balotaje_2024.csv
      media/audio/balotaje_2024/
      media/audio_existing/    (opcional)
"""

import os
import re
import requests
import subprocess
import pandas as pd
from pathlib import Path
import argparse


AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg"}


def get_repo_root(current_file: Path) -> Path:
    parent = current_file.resolve().parent
    if parent.name == "src":
        return parent.parent
    return parent


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ffmpeg_ok(ffmpeg_path: str) -> bool:
    if os.path.isabs(ffmpeg_path):
        return os.path.isfile(ffmpeg_path)
    # Si es "ffmpeg", confiamos en que está en PATH
    return True


def load_input_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def read_audio_title(filepath: str):
    """
    Devuelve el 'title' de los metadatos del audio si existe; sino None.
    Usa mutagen para múltiples formatos.
    """
    try:
        from mutagen import File
        audio = File(filepath, easy=True)
        if not audio or not audio.tags:
            return None
        title = audio.tags.get("title")
        if title:
            if isinstance(title, list) and title:
                return str(title[0])
            return str(title)
        title2 = audio.tags.get("\xa9nam")
        if title2:
            if isinstance(title2, list) and title2:
                return str(title2[0])
            return str(title2)
    except Exception:
        pass
    return None


def extract_ad_id_from_title(title: str):
    """
    Intenta obtener el ad_id del título:
      1) Si es un número grande puro, lo usamos.
      2) Si contiene secuencias largas de dígitos (>= 6), tomamos la más larga.
      3) Último recurso: devolvemos el título entero.
    """
    if not title:
        return None
    t = title.strip()

    if re.fullmatch(r"\d+", t):
        return t

    candidates = re.findall(r"\d{6,}", t)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    return t


def build_existing_ids_from_folder(folder: Path):
    """
    Escanea audios en 'folder', lee sus títulos y extrae ad_id para formar un set.
    """
    existing = set()
    if not folder.is_dir():
        return existing

    for root, _, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in AUDIO_EXTS:
                continue
            path = os.path.join(root, fn)
            title = read_audio_title(path)
            ad_id = extract_ad_id_from_title(title)
            if ad_id:
                existing.add(ad_id)
    return existing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Descarga videos desde video_url y extrae audio con ffmpeg."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Archivo de entrada (CSV/Excel) con columnas 'ad_id' y 'video_url'. "
            "Por defecto: data/intermediate/media_urls_balotaje_2024.csv"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default="balotaje_2024",
        help="Etiqueta para esta corrida (ej.: internas_2024, nacionales_2024, balotaje_2024).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directorio donde guardar los audios (MP3). "
            "Por defecto: media/audio/<label>/"
        ),
    )
    parser.add_argument(
        "--already-dir",
        type=str,
        default=None,
        help=(
            "Carpeta con audios YA descargados para no repetir. "
            "Se usan los metadatos de título. "
            "Por defecto: media/audio_existing/"
        ),
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=str,
        default="ffmpeg",
        help=(
            "Ruta a ffmpeg (si no está en PATH). Por defecto: 'ffmpeg', "
            "asumiendo que está disponible en PATH."
        ),
    )
    return parser.parse_args()


def main():
    current_file = Path(__file__)
    repo_root = get_repo_root(current_file)
    args = parse_args()

    label = args.label

    # Paths por defecto
    if args.input is None:
        input_path = repo_root / "data" / "intermediate" / f"media_urls_{label}.csv"
    else:
        input_path = Path(args.input)

    if args.output_dir is None:
        output_dir = repo_root / "media" / "audio" / label
    else:
        output_dir = Path(args.output_dir)

    if args.already_dir is None:
        already_dir = repo_root / "media" / "audio_existing"
    else:
        already_dir = Path(args.already_dir)

    ffmpeg_path = args.ffmpeg_path

    ensure_dir(output_dir)
    ensure_dir(already_dir)

    if not input_path.exists():
        print(f"❌ Archivo de entrada no encontrado: {input_path}")
        return

    if not ffmpeg_ok(ffmpeg_path):
        print(f"❌ ffmpeg no encontrado en: {ffmpeg_path}")
        return

    # Cargar input
    try:
        df = load_input_df(input_path)
    except Exception as e:
        print(f"❌ Error al abrir archivo de entrada: {e}")
        return

    if "video_url" not in df.columns or "ad_id" not in df.columns:
        print("❌ Faltan columnas 'video_url' o 'ad_id' en el archivo de entrada.")
        return

    print(f"⏳ Escaneando audios existentes en: {already_dir}")
    existing_ids = build_existing_ids_from_folder(already_dir)
    print(f"✅ IDs ya presentes (según título en metadata): {len(existing_ids)}")

    total = len(df)
    processed = 0
    skipped = 0
    errors = 0

    for i, row in df.iterrows():
        url = str(row["video_url"]).strip() if pd.notna(row["video_url"]) else ""
        ad_id = str(row["ad_id"]).strip() if pd.notna(row["ad_id"]) else ""

        if not url or url.lower() == "nan" or not ad_id:
            continue

        if ad_id in existing_ids:
            skipped += 1
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{total}] ⏭️  Saltado ad_id={ad_id} (ya existe en {already_dir})")
            continue

        try:
            print(f"[{i+1}/{total}] 🔽 Descargando ad_id={ad_id} → {url}")
            video_path = output_dir / f"{ad_id}.mp4"
            audio_path = output_dir / f"{ad_id}.mp3"

            # Descargar video
            with requests.get(url, stream=True, timeout=90) as r:
                r.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            # Extraer audio con ffmpeg y setear metadatos (title=ad_id)
            command = [
                ffmpeg_path,
                "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "libmp3lame",
                "-q:a", "2",
                "-metadata", f"title={ad_id}",
                str(audio_path),
            ]
            subprocess.run(command, check=True)

            try:
                os.remove(video_path)
            except OSError:
                pass

            processed += 1
            if (i + 1) % 50 == 0:
                print(f"   Progreso: {processed} nuevos, {skipped} saltados, {errors} errores.")

        except requests.exceptions.RequestException as e:
            errors += 1
            print(f"[{i+1}] ❌ Error al descargar: {e}")
        except subprocess.CalledProcessError as e:
            errors += 1
            print(f"[{i+1}] ❌ Error en ffmpeg: {e}")
        except Exception as e:
            errors += 1
            print(f"[{i+1}] ❌ Error general: {e}")

    print("\nResumen:")
    print(f"  Nuevos audios: {processed}")
    print(f"  Saltados (ya estaban): {skipped}")
    print(f"  Errores: {errors}")
    print("✅ Listo.")


if __name__ == "__main__":
    main()
