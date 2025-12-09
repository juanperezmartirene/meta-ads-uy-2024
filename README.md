# *Estrategias electorales diferenciadas en Meta en las elecciones internas y nacionales en 2024 en Uruguay*

## Repositorio para reproducir la extracción, enriquecimiento, análisis y clasificación de anuncios políticos de Meta relacionados con las elecciones uruguayas de 2024.

Autor: Juan Pérez Martirené (Universidad Católica del Uruguay – UCU)

Licencia: MIT

Presentado en el IX Congreso Uruguayo de Ciencia Política.

Incluye:

Extracción completa de anuncios políticos desde Facebook/Instagram (Meta Ad Library API).

Recuperación de anuncios removidos o sin disclaimer.

Extracción de imágenes y videos.

OCR automatizado (Google Vision API).

Transcripción de audios/videos (Whisper).

Unificación, limpieza y enriquecimiento del dataset electoral.

Construcción de corpus textual consolidado.

Preparación de datos para modelado supervisado.

Entrenamiento de clasificadores (RoberTa fine-tuning multilabel y multiclass).

Exportación en formatos replicables (jsonl, csv, modelos HuggingFace, etc.)


## Reproducibilidad del proyecto

Este repositorio sigue un pipeline modular, donde cada etapa puede ejecutarse por separado pero también fue diseñado para correr de forma secuencial.

### Etapa 1 — Extracción de ads (Meta Ad Library)

📌 scripts/ads_download.py

Descarga anuncios políticos de Uruguay 2024.

Permite filtrar por fechas (internas, nacionales, balotaje).

Guarda outputs limpios en data/raw/.

### Etapa 2 — Recuperación de anuncios removidos o sin disclaimer

📌 scripts/recuperacion_removidos.py

Sigue URLs de snapshot.

Extrae metadatos de anuncios eliminados.

Detecta contenido borrado por violaciones de estándares.

### Etapa 3 — Extracción de imágenes y videos

📌 scripts/extraccion_imagenes.py
📌scripts/ extraccion_audios.py

Descarga portadas de videos, contenido visual y metadatos.

Estandariza nombres: {id}_imagen, {id}_video.

### Etapa 4 — OCR (Google Vision API)

📌 scripts/ocr_vision.py

Procesa imágenes en batches de 16.

Produce JSON completo y Excel con texto + confianza.

Evalúa calidad (alta / media / baja).

### Etapa 5 — Transcripción de audios con Whisper

📌 scripts/transcripcion.py

Usa Whisper base (modelo liviano reproducible).

Genera una transcripción por id, archivo .txt y Excel agrupado.

### Etapa 6 — Unificación del corpus (R)

📌 scripts/construccion_corpus.R

Merge de ads, OCR, transcripciones y metadatos.

Detección de disclaimers, anuncios removidos y cuentas borradas.

Construye el extenso corpus final unificando todas las fuentes textuales.

### Etapa 7 — Construcción de dataset para ML

📌 scripts/splits.R

Muestra estratificada del 10% para entrenamiento.

Exporta train.csv, val.csv, test.csv y .jsonl.

### Etapa 8 — Clasificación supervisada (transformers)

📌 scripts/finetuning.py
📌 scripts/clasificacion.py

Fine-tuning de pln-udelar/rouberta-base-uy22-cased.

Guarda modelo, métricas y logs.

Incluye limpieza homogénea del texto.

## Licencia

Este proyecto se distribuye bajo la licencia MIT, que permite:

Uso comercial

Reutilización

Modificación

Redistribución

Con obligación de mantener el aviso de copyright.

## Contacto

Para consultas

Juan Pérez Martirené
Universidad Católica del Uruguay
juan.perezmartirene@ucu.edu.uy
