# -*- coding: utf-8 -*-
"""
Extracción de imagen principal o video de anuncios de la Meta Ad Library,
a partir de un Excel/CSV con columnas:
    - adlib_id
    - ad_snapshot_url

Usa Selenium para abrir cada snapshot, detectar si hay video o imagen,
descargar la portada / imagen principal y guardar un CSV con metadatos.

Requisitos:
    pip install selenium webdriver-manager pandas requests

Token:
    Usar la variable de entorno META_ADLIBRARY_TOKEN (por defecto), por ejemplo:

    En PowerShell:
        $env:META_ADLIBRARY_TOKEN = "TU_TOKEN"

    En bash:
        export META_ADLIBRARY_TOKEN="TU_TOKEN"

Estructura esperada del repo:
    repo_root/
      src/extraccion_imagenes.py
      data/raw/BD_balotaje.xlsx          (u otro archivo de entrada)
      media/images/<label>/
      media/videos/<label>/
      data/intermediate/media_urls_<label>.csv
"""

import os
import time
import random
import logging
import urllib.request
import subprocess  # (no se usa aquí pero lo dejamos por si amplías luego)
import pandas as pd
from pathlib import Path
import re
import argparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait

# ======================= COMÚN / HELPERS =====================================

MIN_SLEEP = 1.0
MAX_SLEEP = 3.0
LONG_COOLDOWN_EVERY = 5000
LONG_COOLDOWN_SECONDS = 30
BACKOFF_BASE_SECONDS = 30

CUSTOM_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)


def get_repo_root(current_file: Path) -> Path:
    """
    Intenta inferir la raíz del repo:
      - Si el script está en src/, se va un nivel arriba.
      - Si no, usa el directorio del propio archivo.
    """
    parent = current_file.resolve().parent
    if parent.name == "src":
        return parent.parent
    return parent


def get_token_from_env(env_var: str = "META_ADLIBRARY_TOKEN") -> str:
    token = os.getenv(env_var)
    if not token:
        raise RuntimeError(
            f"No se encontró la variable de entorno {env_var}. "
            f"Definila antes de ejecutar el script."
        )
    return token


def setup_logging(log_dir: Path, label: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"scraper_imagenes_{label}.log"

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logging.info(f"Logging inicializado. Archivo: {log_path}")
    return log_path


def page_ready(driver, max_checks=12, pause=0.5):
    try:
        for _ in range(max_checks):
            state = driver.execute_script("return document.readyState")
            if state == "complete":
                return True
            time.sleep(pause)
    except Exception:
        pass
    return False


def detect_block_page(driver):
    """
    Detecta pantalla de bloqueo de Meta/Facebook.
    """
    try:
        src = driver.page_source
        if "Se te bloqueó temporalmente" in src or "Se te bloqueó" in src:
            return True
        if "temporarily blocked" in src or "blocked temporarily" in src:
            return True
    except Exception:
        pass
    return False


def exponential_backoff_sleep(attempts):
    """
    Backoff exponencial ante bloqueos.
    """
    wait = BACKOFF_BASE_SECONDS * (2 ** (attempts - 1))
    logging.warning(f"Backoff: bloqueado. Esperando {wait} segundos (intento #{attempts}).")
    try:
        for s in range(int(wait), 0, -1):
            if s % 30 == 0:
                logging.info(f"Backoff remaining: {s} s")
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Interrumpido por usuario durante backoff.")


def ensure_token_in_url(url, token):
    """Reemplaza SOLO el valor de access_token si ya está; si no, lo agrega."""
    if not url:
        return ""
    if "access_token=" in url:
        return re.sub(r"access_token=[^&]+", f"access_token={token}", url)
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}access_token={token}"


def download_file(url, dest):
    url = (url or "").strip()
    if not url:
        logging.warning("URL vacía, no descargo.")
        return False
    try:
        urllib.request.urlretrieve(url, dest)
        logging.info(f"Descargado: {dest}")
        return True
    except Exception as e:
        logging.warning(f"Error al descargar {url}: {e}")
        with open(dest, "wb") as f:
            f.write(b"")
        return False


# ======================= CLI PARÁMETROS ======================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Descarga imagen principal o video de anuncios de la "
            "Meta Ad Library usando Selenium."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Archivo de entrada (Excel/CSV) con columnas 'adlib_id' y 'ad_snapshot_url'. "
            "Por defecto: data/raw/BD_balotaje.xlsx"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default="balotaje_2024",
        help="Etiqueta para esta corrida (ej.: internas_2024, nacionales_2024, balotaje_2024).",
    )
    parser.add_argument(
        "--env-token-var",
        type=str,
        default="META_ADLIBRARY_TOKEN",
        help="Variable de entorno donde está el token. Default: META_ADLIBRARY_TOKEN",
    )
    parser.add_argument(
        "--use-real-profile",
        action="store_true",
        help="Usar un perfil real de Chrome (user-data-dir). Si no, se crea uno aislado en el repo.",
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        default=None,
        help="Ruta al user-data-dir de Chrome (solo si usás --use-real-profile).",
    )
    return parser.parse_args()


# ======================= MAIN ================================================

def main():
    current_file = Path(__file__)
    repo_root = get_repo_root(current_file)
    args = parse_args()

    # Paths por defecto
    if args.input is None:
        input_path = repo_root / "data" / "raw" / "BD_balotaje.xlsx"
    else:
        input_path = Path(args.input)

    label = args.label

    images_dir = repo_root / "media" / "images" / label
    videos_dir = repo_root / "media" / "videos" / label
    meta_dir = repo_root / "data" / "intermediate"
    meta_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    output_csv = meta_dir / f"media_urls_{label}.csv"
    log_dir = repo_root / "logs"
    setup_logging(log_dir, label)

    logging.info("=== INICIO SCRAPER IMÁGENES/VIDEOS ===")
    logging.info(f"Archivo de entrada: {input_path}")
    logging.info(f"Label: {label}")
    logging.info(f"DIR imágenes: {images_dir}")
    logging.info(f"DIR videos:   {videos_dir}")
    logging.info(f"Salida CSV metadatos: {output_csv}")

    if not input_path.exists():
        logging.error(f"Archivo de entrada no encontrado: {input_path}")
        return

    # Token
    token = get_token_from_env(args.env_token_var)
    logging.info(f"Token obtenido desde env var {args.env_token_var}")

    # Cargar BD
    logging.info("Leyendo archivo de entrada...")
    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path, dtype=str)
    else:
        df = pd.read_csv(input_path, dtype=str)

    df = df.dropna(subset=["adlib_id", "ad_snapshot_url"])
    df = df.reset_index(drop=True)
    total = len(df)
    logging.info(f"Total anuncios a procesar: {total}")

    # IDs ya procesados (reanudar)
    processed_ids = set()
    if output_csv.exists():
        prev = None
        try:
            prev = pd.read_csv(output_csv, dtype=str)
        except Exception:
            prev = None
        if prev is not None and "ad_id" in prev.columns:
            processed_ids = set(prev["ad_id"].astype(str).str.strip().tolist())
            logging.info(f"IDs ya procesados detectados: {len(processed_ids)}")

    # Init Selenium
    opts = webdriver.ChromeOptions()
    if args.use_real_profile:
        if not args.profile_path:
            raise RuntimeError(
                "Si usás --use-real-profile, debés especificar --profile-path."
            )
        opts.add_argument(f"--user-data-dir={args.profile_path}")
    else:
        # perfil fresco aislado en el repo
        chrome_profile_dir = repo_root / ".chromescraper_profile"
        opts.add_argument(f"--user-data-dir={chrome_profile_dir}")

    opts.add_argument("--start-maximized")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--remote-debugging-port=9222")
    opts.add_argument(f"--user-agent={CUSTOM_USER_AGENT}")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    wait = WebDriverWait(driver, 20)

    # Login
    logging.info("Abriendo Facebook para login...")
    driver.get("https://www.facebook.com/")
    page_ready(driver, max_checks=12, pause=0.5)

    login_ok = False
    for _ in range(120):
        try:
            cookies = {c.get("name"): c.get("value") for c in driver.get_cookies()}
            if "c_user" in cookies and cookies["c_user"]:
                login_ok = True
                break
        except Exception:
            pass
        time.sleep(2)

    if not login_ok:
        logging.info("Si ya iniciaste sesión en la ventana, presioná Enter para continuar...")
        try:
            input()
            cookies = {c.get("name"): c.get("value") for c in driver.get_cookies()}
            login_ok = "c_user" in cookies and cookies["c_user"]
        except Exception:
            pass

    if not login_ok:
        logging.error("No se detectó sesión. Abortando.")
        driver.quit()
        return

    logging.info("Sesión detectada. Comenzando scraping...")

    videos_meta = []
    block_attempts = 0

    for idx, row in df.iterrows():
        ad_id = str(row["adlib_id"]).strip()
        raw_url = str(row["ad_snapshot_url"]) if pd.notna(row["ad_snapshot_url"]) else ""
        page = ensure_token_in_url(raw_url, token)

        logging.info(f"[{idx+1}/{total}] ad_id={ad_id}")

        if ad_id in processed_ids:
            logging.info(f"Saltando ad_id ya procesado: {ad_id}")
            continue

        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

        try:
            driver.get(page)
            page_ready(driver)
            time.sleep(random.uniform(1.0, 2.5))

            if detect_block_page(driver):
                block_attempts += 1
                logging.warning("Página de bloqueo detectada.")
                exponential_backoff_sleep(block_attempts)
                logging.info("Verificá la ventana del navegador. Presioná Enter para continuar.")
                try:
                    input()
                except Exception:
                    pass
                continue

            # Intentar detectar video
            def try_video_block():
                vids = driver.find_elements(By.TAG_NAME, "video")
                if not vids:
                    return False, None, None
                vid = vids[0]
                video_url = (vid.get_attribute("src") or "").strip()
                poster = (vid.get_attribute("poster") or video_url or "").strip()
                if poster:
                    dst = videos_dir / f"{ad_id}_portada.jpg"
                    download_file(poster, str(dst))
                return True, video_url, poster

            got_video, video_url, poster = try_video_block()
            text = ""
            src = None

            # Si no hay video, buscar imagen grande
            if not got_video:
                imgs = driver.find_elements(By.CSS_SELECTOR, "img[src*='scontent'], img[src*='fbcdn']")
                src = next(
                    (
                        img.get_attribute("src")
                        for img in imgs
                        if img.size.get("width", 0) > 100 and img.size.get("height", 0) > 100
                    ),
                    None,
                )

            # Opcional: expandir y capturar algún texto si querés
            # (dejamos la lógica básica del original)
            if not got_video and not src:
                try:
                    btn = driver.find_element(
                        By.XPATH,
                        "//div[@role='button' and (contains(., 'Ver anuncio') or contains(., 'See ad'))]"
                    )
                    btn.click()
                    time.sleep(random.uniform(1.0, 2.0))
                    spans = driver.find_elements(By.CSS_SELECTOR, "span")
                    textos = [s.text for s in spans if s.text and len(s.text) > 40]
                    if textos:
                        text = textos[0][:120].replace("\n", " ")
                    got_video, video_url, poster = try_video_block()
                    if not got_video:
                        imgs = driver.find_elements(By.CSS_SELECTOR, "img[src*='scontent'], img[src*='fbcdn']")
                        src = next(
                            (
                                img.get_attribute("src")
                                for img in imgs
                                if img.size.get("width", 0) > 100 and img.size.get("height", 0) > 100
                            ),
                            None,
                        )
                except Exception as e:
                    logging.debug(f"No se pudo expandir: {e}")

            # Guardar metadata
            if got_video:
                videos_meta.append(
                    {
                        "ad_id": ad_id,
                        "tipo": "video",
                        "image_url": "",
                        "video_url": (video_url or ""),
                        "text": text,
                        "label": label,
                    }
                )
                logging.info(f"Video registrado para {ad_id}")
            elif src:
                dst = images_dir / f"{ad_id}_imagen.jpg"
                download_file(src, str(dst))
                videos_meta.append(
                    {
                        "ad_id": ad_id,
                        "tipo": "imagen",
                        "image_url": (src or ""),
                        "video_url": "",
                        "text": text,
                        "label": label,
                    }
                )
                logging.info(f"Imagen registrada para {ad_id}")
            else:
                logging.info(f"No se detectó media para {ad_id}")

            # Backup incremental
            if (idx + 1) % 1 == 0:
                df_run = pd.DataFrame(videos_meta)
                prev = None
                if output_csv.exists():
                    try:
                        prev = pd.read_csv(output_csv, dtype=str)
                    except Exception:
                        prev = None
                if prev is not None and not prev.empty:
                    combined = pd.concat([prev, df_run], ignore_index=True)
                else:
                    combined = df_run
                if not combined.empty:
                    combined = (
                        combined
                        .sort_values(by=["ad_id"])
                        .drop_duplicates(subset=["ad_id"], keep="last")
                    )
                    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")
                    logging.info(f"Backup guardado ({len(combined)} registros)")

            if (idx + 1) % LONG_COOLDOWN_EVERY == 0:
                logging.info(f"Pausa larga de enfriamiento: {LONG_COOLDOWN_SECONDS}s")
                time.sleep(LONG_COOLDOWN_SECONDS)

            if block_attempts > 0:
                block_attempts = 0

        except Exception as e:
            logging.exception(f"Error procesando ad_id={ad_id}: {e}")
            try:
                driver.quit()
            except Exception:
                pass
            time.sleep(10)
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=opts)
            continue

    # Cierre
    try:
        driver.quit()
    except Exception:
        pass

    # Consolidado final
    if videos_meta:
        df_v_run = pd.DataFrame(videos_meta)
        prev_final = None
        if output_csv.exists():
            try:
                prev_final = pd.read_csv(output_csv, dtype=str)
            except Exception:
                prev_final = None
        if prev_final is not None and not prev_final.empty:
            df_v = pd.concat([prev_final, df_v_run], ignore_index=True)
        else:
            df_v = df_v_run
        df_v = (
            df_v
            .sort_values(by=["ad_id"])
            .drop_duplicates(subset=["ad_id"], keep="last")
        )
        df_v.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logging.info(f"CSV final guardado en {output_csv} ({len(df_v)} registros)")
    else:
        logging.info("No se detectaron medios; no se generó CSV final.")

    logging.info("=== FIN SCRAPER IMÁGENES/VIDEOS ===")


if __name__ == "__main__":
    main()
