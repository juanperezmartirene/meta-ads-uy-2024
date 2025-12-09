# -*- coding: utf-8 -*-
"""
Descarga de anuncios políticos (Meta Ad Library) para Uruguay.

Requisitos (instalar una vez):
    pip install requests pandas openpyxl tqdm python-dateutil

Autenticación:
    - Definir la variable de entorno META_ADLIBRARY_TOKEN con tu token de acceso:
      En Windows (PowerShell):
          $env:META_ADLIBRARY_TOKEN = "TU_TOKEN_AQUI"
      En Linux/macOS:
          export META_ADLIBRARY_TOKEN="TU_TOKEN_AQUI"

Salida (por defecto):
    - Carpeta ./data/raw/<etiqueta_eleccion>/ con:
        * archivos diarios CSV: ads_YYYY-MM-DD.csv
    - Carpeta ./data/processed/ con:
        * BD_totales_<etiqueta_eleccion>.xlsx
        * BD_totales_<etiqueta_eleccion>.csv
        * BD_totales_<etiqueta_eleccion>.pkl

Ejemplos de uso:
    # Internas 2024
    python download_ads_meta.py --date-min 2024-06-01 --date-max 2024-06-30 --label internas_2024

    # Nacionales 2024
    python download_ads_meta.py --date-min 2024-10-01 --date-max 2024-10-31 --label nacionales_2024

    # Balotaje 2024
    python download_ads_meta.py --date-min 2024-11-01 --date-max 2024-11-24 --label balotaje_2024

    # Todo el ciclo 2024
    python download_ads_meta.py --date-min 2024-01-01 --date-max 2024-11-24 --label elecciones_2024
"""

import os
import time
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from pathlib import Path
import argparse

import requests
import pandas as pd
from tqdm import tqdm

# ========= CONFIG GLOBAL =========

API_VERSION = "v20.0"
ENDPOINT_URL = f"https://graph.facebook.com/{API_VERSION}/ads_archive"

# Campos a recuperar (equivalente a tu vector FIELDS en R)
FIELDS = [
    "id",
    "ad_creation_time",
    "ad_creative_bodies",
    "ad_creative_link_captions",
    "ad_creative_link_descriptions",
    "ad_creative_link_titles",
    "ad_delivery_start_time",
    "ad_delivery_stop_time",
    "ad_snapshot_url",
    "bylines",
    "currency",
    "estimated_audience_size",
    "impressions",
    "languages",
    "page_id",
    "page_name",
    "publisher_platforms",
    "spend",
    "demographic_distribution",
    "delivery_by_region",
]

AD_REACHED_COUNTRIES = "UY"
AD_ACTIVE_STATUS = "ALL"
AD_TYPE = "POLITICAL_AND_ISSUE_ADS"
PUBLISHER_PLATFORMS = "facebook,instagram,messenger,whatsapp"
PAGE_LIMIT = 35  # podés subir a 100

MAX_RETRIES = 6
BACKOFF_BASE = 2.0  # segundos


def get_access_token() -> str:
    """
    Recupera el token de la variable de entorno META_ADLIBRARY_TOKEN.
    Esto evita subir el token al repositorio.
    """
    token = os.getenv("META_ADLIBRARY_TOKEN")
    if not token:
        raise RuntimeError(
            "No se encontró la variable de entorno META_ADLIBRARY_TOKEN.\n"
            "Definila antes de ejecutar el script."
        )
    return token


def daterange(date_min: str, date_max: str):
    """Genera fechas (YYYY-MM-DD) día a día entre date_min y date_max (incluido)."""
    start = datetime.strptime(date_min, "%Y-%m-%d").date()
    end = datetime.strptime(date_max, "%Y-%m-%d").date()
    d = start
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def build_params(day_str: str, access_token: str) -> dict:
    """Arma los parámetros para una fecha concreta (min=max=day_str)."""
    return {
        "access_token": access_token,
        "ad_reached_countries": AD_REACHED_COUNTRIES,
        "ad_active_status": AD_ACTIVE_STATUS,
        "ad_type": AD_TYPE,
        "ad_delivery_date_min": day_str,
        "ad_delivery_date_max": day_str,
        "search_terms": "NULL",
        "publisher_platforms": PUBLISHER_PLATFORMS,
        "fields": ",".join(FIELDS),
        "limit": PAGE_LIMIT,
    }


def get_with_retries(url: str, params: dict):
    """GET con reintentos y backoff exponencial simple."""
    attempt = 0
    while True:
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()

            # Manejo de rate limit o errores temporales
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt >= MAX_RETRIES:
                    resp.raise_for_status()
                sleep_s = BACKOFF_BASE * (2 ** attempt)
                print(f"[WARN] HTTP {resp.status_code}. Reintentando en {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                attempt += 1
            else:
                # Errores no recuperables
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
        except requests.RequestException as e:
            if attempt >= MAX_RETRIES:
                raise
            sleep_s = BACKOFF_BASE * (2 ** attempt)
            print(f"[WARN] Error de red: {e}. Reintentando en {sleep_s:.1f}s...")
            time.sleep(sleep_s)
            attempt += 1


def fetch_ads_for_day(day_str: str, access_token: str) -> pd.DataFrame:
    """Descarga y pagina resultados para una fecha dada (day_str)."""
    params = build_params(day_str, access_token)
    data_all = []

    # Primera página
    payload = get_with_retries(ENDPOINT_URL, params)

    while True:
        batch = payload.get("data", [])
        data_all.extend(batch)

        # Siguiente página si existe
        paging = payload.get("paging", {})
        next_url = paging.get("next")
        if not next_url:
            break

        # El 'next' ya trae querystring, hacemos GET directo
        payload = get_with_retries(next_url, params={})

    # A DataFrame
    if not data_all:
        return pd.DataFrame(columns=FIELDS)

    df = pd.json_normalize(data_all, max_level=1)

    # Filtro: ad_delivery_start_time == day_str (igual que en tu R original)
    if "ad_delivery_start_time" in df.columns:
        def normalize_date(x):
            if pd.isna(x):
                return None
            try:
                return dateparser.parse(str(x)).date().strftime("%Y-%m-%d")
            except Exception:
                return None

        df["_start_date"] = df["ad_delivery_start_time"].apply(normalize_date)
        df = df[df["_start_date"] == day_str].drop(columns=["_start_date"])

    return df


def parse_args():
    """Parámetros de línea de comandos para hacerlo reutilizable."""
    parser = argparse.ArgumentParser(
        description="Descarga anuncios políticos de la Meta Ad Library para Uruguay."
    )
    parser.add_argument(
        "--date-min",
        required=True,
        help="Fecha mínima (YYYY-MM-DD), inclusive. Ej: 2024-06-01",
    )
    parser.add_argument(
        "--date-max",
        required=True,
        help="Fecha máxima (YYYY-MM-DD), inclusive. Ej: 2024-06-30",
    )
    parser.add_argument(
        "--label",
        default="elecciones_uy",
        help="Etiqueta de la elección / periodo (ej.: internas_2024, nacionales_2024, balotaje_2024).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    date_min = args.date_min
    date_max = args.date_max
    label = args.label

    # Directorios relativos al repo
    repo_root = Path(__file__).resolve().parent.parent if (Path(__file__).parent.name == "src") else Path(__file__).resolve().parent
    raw_dir = repo_root / "data" / "raw" / label
    processed_dir = repo_root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    access_token = get_access_token()

    frames = []
    diarios_guardados = []

    print(f"Descargando anuncios UY de {date_min} a {date_max} para '{label}'...")
    for day in tqdm(list(daterange(date_min, date_max))):
        try:
            # Pequeño delay para no saturar la API
            time.sleep(1.0)
            df_day = fetch_ads_for_day(day, access_token)

            # Guardado diario (CSV)
            out_csv = raw_dir / f"ads_{day}.csv"
            df_day.to_csv(out_csv, index=False, encoding="utf-8-sig")
            diarios_guardados.append(out_csv)

            frames.append(df_day)

        except Exception as e:
            print(f"[ERROR] {day}: {e}")

    # Consolidado
    if frames:
        total = pd.concat(frames, ignore_index=True)
    else:
        total = pd.DataFrame(columns=FIELDS)

    # Archivos de salida consolidados
    out_xlsx = processed_dir / f"BD_totales_{label}.xlsx"
    out_csv = processed_dir / f"BD_totales_{label}.csv"
    out_pkl = processed_dir / f"BD_totales_{label}.pkl"

    # Excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        total.to_excel(writer, index=False, sheet_name="anuncios_totales")

    # CSV y Pickle
    total.to_csv(out_csv, index=False, encoding="utf-8-sig")
    total.to_pickle(out_pkl)

    print("\nListo ✅")
    print(f"- Archivos diarios guardados: {len(diarios_guardados)} (ej.: {diarios_guardados[:2]} ...)")
    print(f"- Consolidado Excel: {out_xlsx}")
    print(f"- Consolidado CSV:   {out_csv}")
    print(f"- Pickle pandas:     {out_pkl}")
    print(f"- Filtrado aplicado: ad_delivery_start_time == fecha del día")


if __name__ == "__main__":
    main()
