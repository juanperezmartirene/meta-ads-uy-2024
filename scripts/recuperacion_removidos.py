# -*- coding: utf-8 -*-
"""
Recuperación de texto de anuncios REMOVIDOS o SIN DISCLAIMER
a partir de sus ad_snapshot_url (Meta Ad Library) usando Selenium.

Este script:
  - Lee una base Excel/CSV con anuncios (incluye columnas:
      * ad_creative_bodies
      * adlib_id
      * ad_snapshot_url
    )
  - Filtra aquellos anuncios donde el cuerpo contiene mensajes de:
      * "This ad ran without a required disclaimer."
      * "This content was removed because it didn't follow our Advertising Standards."
  - Abre cada snapshot con Selenium, expande el texto del anuncio y extrae el cuerpo.
  - Guarda un CSV con:
      * ad_id
      * text_body

Ejemplo de uso:

    # Usando defaults (data/raw/BD_removidos.xlsx y data/intermediate/texto_removidos)
    python src/recuperacion_removidos.py

    # Especificando archivo de entrada y carpeta de salida
    python src/recuperacion_removidos.py \
        --input data/raw/BD_removidos.xlsx \
        --output-dir data/intermediate/texto_removidos

Autenticación:
    - Token para snapshot: variable de entorno META_ADLIBRARY_TOKEN

    En PowerShell (Windows):
        $env:META_ADLIBRARY_TOKEN = "TU_TOKEN"

    En bash (Linux/macOS):
        export META_ADLIBRARY_TOKEN="TU_TOKEN"
"""

import os
import re
import time
import random
import logging
import tempfile
import argparse
from pathlib import Path

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# ======================= CONSTANTES GENERALES =================================

MIN_SLEEP = 1.0
MAX_SLEEP = 3.0
LONG_COOLDOWN_EVERY = 5000
LONG_COOLDOWN_SECONDS = 30
BACKOFF_BASE_SECONDS = 30

USE_REAL_PROFILE_DEFAULT = False
CUSTOM_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)

# Palabras/labels de interfaz a excluir del cuerpo
BLACKLIST_RE = re.compile(
    r"(?i)\b(Publicidad|Sponsored|Identificador de la biblioteca|Ad Library ID|"
    r"Anuncio eliminado|This ad was removed|This content was removed|Archivo de anuncios|"
    r"Me gusta|Like|Comentar|Comment|Compartir|Share|Seguir|Follow|Enviar mensaje|Message|"
    r"Ver anuncio|See ad|Ver más|See more|Más información|Learn more)\b"
)

# XPATH directo que inspeccionaste (lo mantenemos como opción específica)
XPATH_CUERPO_INSPECCION = (
    "/html/body/div[1]/div[1]/div[1]/div/div/div/div/div/div/div/div[2]/div[1]/span"
)


# ======================= HELPERS DE ENTORNO ==================================

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
    """Obtiene el token desde una variable de entorno, o levanta error si no está."""
    token = os.getenv(env_var)
    if not token:
        raise RuntimeError(
            f"No se encontró la variable de entorno {env_var}. "
            f"Definila antes de ejecutar el script."
        )
    return token


def setup_logging(output_dir: Path) -> None:
    """Configura logging a archivo + consola."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "scraper.log"

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


# ======================= HELPERS DE ESCRITURA =================================

def save_csv_atomic(df, out_path, retries=10, sleep_secs=1.5):
    """Escritura atómica con reintentos para evitar PermissionError en Windows."""
    folder = os.path.dirname(out_path)
    Path(folder).mkdir(parents=True, exist_ok=True)

    attempt = 1
    while True:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                dir=folder,
                suffix=".tmp",
                encoding="utf-8-sig",
                newline=""
            ) as tmp:
                tmp_path = tmp.name
                df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
            os.replace(tmp_path, out_path)
            logging.info(f"Guardado atómico OK -> {out_path}")
            break
        except PermissionError as e:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            if attempt >= retries:
                logging.exception(
                    f"No se pudo escribir {out_path} tras {retries} reintentos."
                )
                raise e
            logging.warning(
                f"Archivo bloqueado: {out_path}. "
                f"Reintento {attempt}/{retries} en {sleep_secs}s."
            )
            time.sleep(sleep_secs)
            attempt += 1
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise e


# ======================= HELPERS SELENIUM =====================================

def page_ready(driver, max_checks=12, pause=0.5):
    try:
        for _ in range(max_checks):
            if driver.execute_script("return document.readyState") == "complete":
                return True
            time.sleep(pause)
    except Exception:
        pass
    return False


def detect_block_page(driver):
    try:
        src = driver.page_source
        if ("Se te bloqueó temporalmente" in src or
                "temporarily blocked" in src or
                "blocked temporarily" in src):
            return True
    except Exception:
        pass
    return False


def exponential_backoff_sleep(attempts):
    wait = BACKOFF_BASE_SECONDS * (2 ** (attempts - 1))
    logging.warning(
        f"Backoff: bloqueado. Esperando {wait} segundos (intento #{attempts})."
    )
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


def click_see_ad(driver, wait):
    """Clic en 'Ver anuncio' / 'See ad'. Devuelve True si clickeó (o ya estaba abierto)."""
    candidates = [
        (By.XPATH, "//div[@role='button' and (contains(., 'Ver anuncio') or contains(., 'See ad'))]"),
        (By.XPATH, "//span[contains(., 'Ver anuncio') or contains(., 'See ad')]/ancestor::div[@role='button']"),
        (By.XPATH, "//a[contains(., 'Ver anuncio') or contains(., 'See ad')]"),
        (By.XPATH, "//button[contains(., 'Ver anuncio') or contains(., 'See ad')]"),
    ]
    for by, sel in candidates:
        try:
            btn = wait.until(EC.element_to_be_clickable((by, sel)))
            ActionChains(driver).move_to_element(btn).pause(0.1).click(btn).perform()
            time.sleep(random.uniform(0.6, 1.2))
            return True
        except Exception:
            continue
    return False


def expand_body_text(driver, wait):
    """
    Expande el texto del cuerpo:
    1) Click en 'Ver más/See more' dentro del article
    2) Si no aparece, click en nodos con '…' o '...'
    3) Último recurso: ver más genérico
    """
    # 1) 'Ver más' dentro del article
    more_xpaths = [
        "//div[@role='article']//div[@role='button'][contains(., 'Ver más') or contains(., 'See more')]",
        "//div[@role='article']//span[contains(., 'Ver más') or contains(., 'See more')]/ancestor::*[@role='button']"
    ]
    for xp in more_xpaths:
        try:
            btn = driver.find_element(By.XPATH, xp)
            if btn.is_displayed() and btn.is_enabled():
                ActionChains(driver).move_to_element(btn).pause(0.1).click(btn).perform()
                time.sleep(0.4)
                return True
        except Exception:
            pass

    # 2) nodos con '…' o '...'
    trunc_xpaths = [
        "//div[@role='article']//span[contains(., '…')]",
        "//div[@role='article']//span[contains(., '...')]",
        "//div[@role='article']//div[contains(., '…')]",
        "//div[@role='article']//div[contains(., '...')]",
    ]
    for xp in trunc_xpaths:
        try:
            nodes = driver.find_elements(By.XPATH, xp)
            for n in nodes:
                t = (n.text or "").strip()
                if not t or BLACKLIST_RE.search(t):
                    continue
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", n
                )
                # ancestro clickable
                try:
                    anc = n.find_element(By.XPATH, "./ancestor::*[@role='button'][1]")
                    if anc and anc.is_displayed() and anc.is_enabled():
                        anc.click()
                        time.sleep(0.4)
                        return True
                except Exception:
                    pass
                # click directo
                try:
                    n.click()
                    time.sleep(0.4)
                    return True
                except Exception:
                    try:
                        driver.execute_script("arguments[0].click();", n)
                        time.sleep(0.4)
                        return True
                    except Exception:
                        pass
        except Exception:
            pass

    # 3) genérico
    generic_more = [
        "//div[@role='button'][contains(., 'Ver más') or contains(., 'See more')]",
        "//span[contains(., 'Ver más') or contains(., 'See more')]/ancestor::*[@role='button']"
    ]
    for xp in generic_more:
        try:
            btns = driver.find_elements(By.XPATH, xp)
            for b in btns:
                if b.is_displayed() and b.is_enabled():
                    b.click()
                    time.sleep(0.4)
                    return True
        except Exception:
            pass

    return False


def extract_ad_body_text_with_xpath(driver):
    """Extrae el cuerpo por el XPATH inspeccionado."""
    try:
        elems = driver.find_elements(By.XPATH, XPATH_CUERPO_INSPECCION)
        textos = [e.text.strip() for e in elems if e.text and e.text.strip()]
        if textos:
            text = textos[0]
            print(f"[DEBUG] XPATH capturado: {text[:200]}...")
            return text
        else:
            print("[DEBUG] XPATH: no se encontró texto en ese nodo.")
            return ""
    except Exception as e:
        print(f"[DEBUG] XPATH error: {e}")
        return ""


def extract_ad_body_text_strict(driver):
    """
    Extrae SOLO el texto del cuerpo:
    1) Contenedores canónicos del preview (más limpios)
    2) Fallback: bloque textual más largo dentro de article (filtrando UI)
    """
    canonical_selectors = [
        (By.CSS_SELECTOR, "[data-ad-preview='message']"),
        (By.CSS_SELECTOR, "[data-testid='ad-preview-message']"),
        (By.CSS_SELECTOR, "[data-ad-preview='message'] span, [data-ad-preview='message'] div"),
        (By.CSS_SELECTOR, "[data-testid='ad-preview-message'] span, [data-testid='ad-preview-message'] div"),
    ]
    for by, sel in canonical_selectors:
        try:
            elems = driver.find_elements(by, sel)
            texts = []
            for e in elems:
                t = (e.text or "").strip()
                if not t or BLACKLIST_RE.search(t):
                    continue
                t = re.sub(r"\s+", " ", t).strip()
                if len(t) >= 5:
                    texts.append(t)
            if texts:
                body = max(texts, key=len)
                return body
        except Exception:
            pass

    # Fallback dentro de article
    try:
        articles = driver.find_elements(By.XPATH, "//div[@role='article']")
        scope = articles[0] if articles else driver
        nodes = scope.find_elements(By.XPATH, ".//div|.//span")
        candidates = []
        for n in nodes:
            try:
                t = (n.text or "").strip()
                if not t or BLACKLIST_RE.search(t):
                    continue
                if len(t) < 15:
                    continue
                letters = sum(ch.isalpha() for ch in t)
                if letters / max(1, len(t)) < 0.45:
                    continue
                t = re.sub(r"\s+", " ", t).strip()
                candidates.append(t)
            except Exception:
                continue
        if candidates:
            body = sorted(candidates, key=len, reverse=True)[0]
            body = re.sub(
                r"\s*(Ver anuncio|See ad|Ver más|See more)\s*$",
                "",
                body,
                flags=re.IGNORECASE,
            ).strip()
            return body
    except Exception:
        pass

    return ""


# ======================= PARÁMETROS CLI ======================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recupera texto de anuncios removidos/sin disclaimer "
            "desde sus ad_snapshot_url (Meta Ad Library)."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Ruta al archivo de entrada (Excel/CSV). "
             "Por defecto: data/raw/BD_removidos.xlsx"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio de salida para texto y log. "
             "Por defecto: data/intermediate/texto_removidos"
    )
    parser.add_argument(
        "--use-real-profile",
        action="store_true",
        help="Usar un perfil real de Chrome (user-data-dir). "
             "Si no se indica, se crea un perfil aislado en la carpeta de salida."
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        default=None,
        help="Ruta al user-data-dir de Chrome (solo si usás --use-real-profile)."
    )
    parser.add_argument(
        "--env-token-var",
        type=str,
        default="META_ADLIBRARY_TOKEN",
        help="Nombre de la variable de entorno donde está el token. "
             "Default: META_ADLIBRARY_TOKEN"
    )
    return parser.parse_args()


# ======================= MAIN =======================================

def main():
    current_file = Path(__file__)
    repo_root = get_repo_root(current_file)

    args = parse_args()

    # Input por defecto
    if args.input is None:
        input_path = repo_root / "data" / "raw" / "BD_removidos.xlsx"
    else:
        input_path = Path(args.input)

    # Output por defecto
    if args.output_dir is None:
        output_dir = repo_root / "data" / "intermediate" / "texto_removidos"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    output_csv = output_dir / "texto_removidos.csv"
    env_token_var = args.env_token_var

    logging.info("=== INICIO: extracción de texto (removidos por disclaimer) ===")
    logging.info(f"Archivo de entrada: {input_path}")
    logging.info(f"Directorio salida : {output_dir}")

    if not input_path.exists():
        logging.error(f"Archivo de entrada no encontrado: {input_path}")
        return

    # Token
    token = get_token_from_env(env_token_var)
    logging.info(f"Token obtenido desde variable de entorno {env_token_var}")

    # Leer base
    logging.info("Leyendo archivo de entrada (Excel/CSV)...")
    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df_all = pd.read_excel(input_path, dtype=str)
    else:
        df_all = pd.read_csv(input_path, dtype=str)

    if "ad_creative_bodies" not in df_all.columns:
        logging.error("La base no tiene columna 'ad_creative_bodies'. Abortando.")
        return

    # Filtro: removidos / sin disclaimer (dos frases posibles)
    pattern = (
        r"(This ad ran without a required disclaimer\.)|"
        r"(This content was removed because it didn't follow our Advertising Standards\.)"
    )
    mask = df_all["ad_creative_bodies"].fillna("").str.contains(
        pattern, case=False, regex=True
    )

    df = df_all.loc[mask].copy()
    df = df.dropna(subset=["adlib_id", "ad_snapshot_url"]).reset_index(drop=True)

    total = len(df)
    logging.info(f"Total anuncios a procesar (texto): {total}")
    if total == 0:
        logging.info("No hay anuncios que cumplan el criterio. Fin.")
        return

    # Ya procesados (para reanudar)
    processed_ids = set()
    if output_csv.exists():
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
        profile_path = args.profile_path
        if not profile_path:
            raise RuntimeError(
                "Si usás --use-real-profile, debés especificar --profile-path."
            )
        opts.add_argument(f"--user-data-dir={profile_path}")
    else:
        # perfil fresco dentro del repo (para no depender de rutas de usuario)
        chrome_profile_dir = output_dir / "chromescraper_profile"
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
        logging.info(
            "No se detectó login automático. "
            "Iniciá sesión en la ventana y luego volvés acá."
        )
        logging.info("Presioná Enter cuando hayas iniciado sesión en el navegador.")
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

    results = []
    block_attempts = 0

    for idx, row in df.iterrows():
        ad_id = str(row["adlib_id"]).strip()
        raw_url = str(row["ad_snapshot_url"])
        page = ensure_token_in_url(raw_url, token) if pd.notna(raw_url) else ""

        logging.info(f"[{idx + 1}/{total}] ad_id={ad_id}")

        if ad_id in processed_ids:
            logging.info(f"Saltando ad_id ya procesado: {ad_id}")
            continue

        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

        try:
            driver.get(page)
            page_ready(driver)
            time.sleep(random.uniform(1.0, 2.2))

            if detect_block_page(driver):
                block_attempts += 1
                logging.warning("Página de bloqueo detectada.")
                exponential_backoff_sleep(block_attempts)
                logging.info(
                    "Revisá la ventana e inicia sesión si es necesario. "
                    "Enter para continuar."
                )
                try:
                    input()
                except Exception:
                    pass
                continue

            # 1) Click en "Ver anuncio" / "See ad"
            click_see_ad(driver, wait)
            time.sleep(random.uniform(0.6, 1.2))

            # 2) Expansión del body (botón y/o texto con '…'/'...')
            expand_body_text(driver, wait)

            # 3) EXTRAER TEXTO DEL CUERPO
            text_body = extract_ad_body_text_with_xpath(driver)  # primero XPATH inspeccionado
            if not text_body:
                text_body = extract_ad_body_text_strict(driver)   # luego canónico/fallback

            # 4) Si quedó truncado, reintentar expansión y re-extraer
            if text_body.endswith("…") or text_body.endswith("...") or "…" in text_body:
                print("[DEBUG] Truncamiento detectado. Reintentando expansión…")
                expand_body_text(driver, wait)
                time.sleep(0.3)
                text_retry = (
                    extract_ad_body_text_with_xpath(driver)
                    or extract_ad_body_text_strict(driver)
                )
                if len(text_retry) > len(text_body):
                    text_body = text_retry

            print(
                f"[DEBUG] Iteración {idx + 1}/{total} | ad_id={ad_id} | "
                f"Texto: {text_body[:200]}..."
            )

            # 5) Guardar resultado (solo texto)
            results.append(
                {
                    "ad_id": ad_id,
                    "text_body": text_body
                }
            )

            # 6) Backup incremental (atómico)
            if (idx + 1) % 1 == 0:  # cada iteración, como en tu original
                df_run = pd.DataFrame(results)
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
                    save_csv_atomic(combined, output_csv)

            if (idx + 1) % LONG_COOLDOWN_EVERY == 0:
                logging.info(
                    f"Pausa larga de enfriamiento: {LONG_COOLDOWN_SECONDS}s"
                )
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
            wait = WebDriverWait(driver, 20)
            continue

    # Cierre y guardado final
    try:
        driver.quit()
    except Exception:
        pass

    if results:
        df_v_run = pd.DataFrame(results)
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
        save_csv_atomic(df_v, output_csv)
        logging.info(f"CSV final guardado en {output_csv} ({len(df_v)} registros)")
    else:
        logging.info("No se generaron resultados.")

    logging.info("=== FIN ===")


if __name__ == "__main__":
    main()
