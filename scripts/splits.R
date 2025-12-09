# Toma BD_texto.csv y construye splits train/val/test + JSONL (clasificación por partido/contexto)

library(dplyr)
library(readr)
library(rsample)
library(jsonlite)
library(here)

set.seed(1891)  # Reproducibilidad

proc_dir   <- here("data", "processed")
splits_dir <- file.path(proc_dir, "splits")
if (!dir.exists(splits_dir)) dir.create(splits_dir, recursive = TRUE)

bd_texto_path <- file.path(proc_dir, "BD_texto.csv")
BD_texto <- read_csv(bd_texto_path, show_col_types = FALSE)

# ---------------------------
# 1. Mapear contexto → label_id
# ---------------------------

labels <- sort(unique(BD_texto$contexto))
label_map <- tibble::tibble(contexto = labels, label_id = seq_along(labels) - 1L)

BD_texto <- BD_texto %>%
  left_join(label_map, by = "contexto")

# ---------------------------
# 2. Split 10% (muestra anotada) vs 95% (clasificación)
# ---------------------------

muestra_5_split <- BD_texto %>%
  distinct(texto, .keep_all = TRUE) %>%   # evitar duplicados textuales exactos
  initial_split(prop = 0.1, strata = contexto)

muestra_5 <- training(muestra_5_split)
resto_95  <- testing(muestra_5_split)

# ---------------------------
# 3. Dentro del 5%: 80% train / 20% val
# ---------------------------

sp2  <- initial_split(muestra_5, prop = 0.8, strata = contexto)
train <- training(sp2)
val   <- testing(sp2)
test  <- resto_95

# ---------------------------
# 4. Exportar CSV
# ---------------------------

export_cols <- c("id", "texto", "contexto", "label_id")

write_csv(train %>% select(all_of(export_cols)),
          file.path(splits_dir, "train_partidos.csv"))
write_csv(val   %>% select(all_of(export_cols)),
          file.path(splits_dir, "val_partidos.csv"))
write_csv(test  %>% select(all_of(export_cols)),
          file.path(splits_dir, "test_partidos.csv"))

# ---------------------------
# 5. Exportar JSONL (formato Hugging Face)
# ---------------------------

to_jsonl <- function(d, path) {
  lines <- apply(
    d %>% select(id, texto, label_id),
    1,
    function(r) {
      toJSON(
        list(
          id    = unname(r[["id"]]),
          text  = unname(r[["texto"]]),
          label = as.integer(r[["label_id"]])
        ),
        auto_unbox = TRUE
      )
    }
  )
  readr::write_lines(lines, path)
}

to_jsonl(train, file.path(splits_dir, "train_partidos.jsonl"))
to_jsonl(val,   file.path(splits_dir, "val_partidos.jsonl"))
to_jsonl(test,  file.path(splits_dir, "test_partidos.jsonl"))

cat("Splits generados en:", splits_dir, "\n")
