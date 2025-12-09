# Construye BD_elecciones + BD_texto a partir de todas las fuentes disponibles.

library(dplyr)
library(readxl)
library(readr)
library(stringr)
library(here)

# ---------------------------
# 1. Paths del proyecto
# ---------------------------

raw_dir    <- here("data", "raw")
proc_dir   <- here("data", "processed")

if (!dir.exists(proc_dir)) dir.create(proc_dir, recursive = TRUE)

# ---------------------------
# 2. Cargar bases de anuncios
# ---------------------------

bd_elecciones_path <- file.path(proc_dir, "BD_elecciones.xlsx")

BD_elecciones <- read_excel(bd_elecciones_path)


# ---------------------------
# 3. Flags de disclaimer / removidos
# ---------------------------

BD_elecciones <- BD_elecciones %>%
  mutate(
    disclaimer_faltante = if_else(
      str_detect(ad_creative_bodies %||% "", 
                 "This ad ran without a required disclaimer\\."),
      TRUE, FALSE, missing = FALSE
    ),
    anuncio_removido = if_else(
      str_detect(ad_creative_bodies %||% "", 
                 "This content was removed because it didn't follow our Advertising Standards\\."),
      TRUE, FALSE, missing = FALSE
    ),
    cuenta_borrada = if_else(
      str_detect(ad_creative_bodies %||% "", 
                 "This ad was run by an account or Page we later disabled for not following our Advertising Standards\\."),
      TRUE, FALSE, missing = FALSE
    )
  )

# ---------------------------
# 4. Cargar fuentes de texto extra
# ---------------------------

# 4a) Textos recuperados de anuncios removidos / sin disclaimer (pipeline final)
texto_removidos_path <- file.path(raw_dir, "texto_removidos", "texto_removidos.csv")

texto_removidos <- read_csv(
  texto_removidos_path,
  show_col_types = FALSE
) %>%
  rename(id = ad_id)   # el CSV viene con columna ad_id


# 4c) Transcripciones de audio (1ª vuelta + balotaje)
transcripciones_1_path  <- file.path(raw_dir, "transcripciones",          "transcripciones.xlsx")
transcripciones_2_path  <- file.path(raw_dir, "transcripciones_balotaje", "transcripciones.xlsx")

transcripciones_1 <- read_excel(transcripciones_1_path) %>%
  rename(id = ad_id)

transcripciones_2 <- read_excel(transcripciones_2_path) %>%
  rename(id = ad_id)

transcripciones <- full_join(transcripciones_1, transcripciones_2)

# 4d) OCR (Google Vision) – resultados ya consolidados
ocr_path <- file.path(raw_dir, "vision_results.xlsx")

ocr <- read_excel(raw_dir)

ocr <- ocr %>%
  mutate(
    calidad = case_when(
      !is.na(avg_confidence) & avg_confidence >= 0.85 ~ "alta",
      !is.na(avg_confidence) & avg_confidence >= 0.70 ~ "media",
      !is.na(avg_confidence)                         ~ "baja",
      TRUE                                           ~ "s/d"
    )
  )

# ---------------------------
# 5. Unir todas las fuentes a BD_elecciones
# ---------------------------

BD_elecciones <- BD_elecciones %>%
  # tipo de media (imagen / video)
  left_join(listado_imagenes_videos, by = "id") %>%
  # texto recuperado de anuncios removidos/sin disclaimer
  left_join(texto_removidos,          by = "id") %>%
  # transcripciones de audio (whisper)
  left_join(transcripciones,          by = "id") %>%
  # texto por OCR sólo para calidad alta
  left_join(
    ocr %>%
      filter(calidad == "alta") %>%
      select(id, text),
    by = "id"
  ) %>%
  rename(texto_ocr = text)

# ---------------------------
# 6. Construir BD_texto (texto consolidado + contexto)
# ---------------------------

BD_texto <- BD_elecciones %>%
  select(
    id,
    ad_creative_bodies,
    text_body,
    texto_ocr,
    transcripcion,
    disclaimer_faltante,
    anuncio_removido,
    tipo,
    part_org,
    tipo_eleccion
  ) %>%
  mutate(
    across(c(ad_creative_bodies, text_body, texto_ocr, transcripcion),
           ~ ifelse(is.na(.x), "", .x)),
    
    # Base: si fue removido o sin disclaimer → usar text_body; si no, el texto original
    texto_base = if_else(anuncio_removido | disclaimer_faltante,
                         text_body,
                         ad_creative_bodies),
    
    # Unificar todas las fuentes textuales
    texto_fuente = str_trim(str_squish(
      paste(texto_base, texto_ocr, transcripcion, sep = " ")
    )),
    
    # Contexto mínimo: partido y tipo de elección
    contexto = paste0(
      ifelse(!is.na(part_org), paste0("[", part_org, "]"), ""),
      ifelse(!is.na(tipo_eleccion), paste0(" [ELECCION:", tipo_eleccion, "]"), "")
    ),
    
    texto = str_trim(str_squish(texto_fuente))
  ) %>%
  select(id, texto, contexto) %>%
  filter(texto != "", contexto != "")

# Reagrupar Partido Independiente en "Otros" (como en tu script original)
BD_texto <- BD_texto %>%
  mutate(
    contexto = if_else(contexto == "[Partido Independiente]", "[Otros]", contexto)
  )

# ---------------------------
# 7. Guardar en data/processed
# ---------------------------

bd_texto_path <- file.path(proc_dir, "BD_texto.csv")
write_csv(BD_texto, bd_texto_path)

cat("BD_texto guardado en:", bd_texto_path, "\n")
