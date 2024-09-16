# Grid search with Montecarlo Cross Validation
# Optimized version with reduced parameter space and incremental saving

# Clear the environment and perform garbage collection
rm(list = ls())
gc()

# Load required libraries
require("data.table")
require("rpart")
require("parallel")
require("primes")
require("lubridate")  # For timestamp generation

# Set up parameters
PARAM <- list()
PARAM$semilla_primigenia <- 102191  # Replace with your seed
PARAM$qsemillas <- 20
PARAM$training_pct <- 70L  # between 1L and 99L

# Choose your dataset
PARAM$dataset_nom <- "~/datasets/vivencial_dataset_pequeno.csv"
# PARAM$dataset_nom <- "~/datasets/conceptual_dataset_pequeno.csv"

# Function to perform stratified partitioning of the dataset
particionar <- function(data, division, agrupa = "", campo = "fold", start = 1, seed = NA) {
  if (!is.na(seed)) set.seed(seed)
  bloque <- unlist(mapply(function(x, y) {
    rep(y, x)
  }, division, seq(from = start, length.out = length(division))))
  data[, (campo) := sample(rep(bloque, ceiling(.N / length(bloque))))[1:.N],
    by = agrupa
  ]
}

# Function to estimate gain for a decision tree model
ArbolEstimarGanancia <- function(semilla, training_pct, param_basicos) {
  particionar(dataset,
    division = c(training_pct, 100L - training_pct), 
    agrupa = "clase_ternaria",
    seed = semilla
  )
  modelo <- rpart("clase_ternaria ~ .",
    data = dataset[fold == 1],
    xval = 0,
    control = param_basicos
  )
  prediccion <- predict(modelo,
    dataset[fold == 2],
    type = "prob"
  )
  ganancia_test <- dataset[
    fold == 2,
    sum(ifelse(prediccion[, "BAJA+2"] > 0.025,
      ifelse(clase_ternaria == "BAJA+2", 117000, -3000),
      0
    ))
  ]
  ganancia_test_normalizada <- ganancia_test / ((100 - PARAM$training_pct) / 100)
  return( 
    c(list("semilla" = semilla),
      param_basicos,
      list("ganancia_test" = ganancia_test_normalizada)
    )
  )
}

# Function to perform Monte Carlo simulations
ArbolesMontecarlo <- function(semillas, param_basicos) {
  salida <- mcmapply(ArbolEstimarGanancia,
    semillas,
    MoreArgs = list(PARAM$training_pct, param_basicos),
    SIMPLIFY = FALSE,
    mc.cores = detectCores()
  )
  return(salida)
}

# Set the working directory
setwd("~/buckets/b1/")

# Generate prime numbers for seeds
primos <- generate_primes(min = 100000, max = 1000000)
set.seed(PARAM$semilla_primigenia)
PARAM$semillas <- sample(primos, PARAM$qsemillas)

# Load the dataset
dataset <- fread(PARAM$dataset_nom)
dataset <- dataset[clase_ternaria != ""]

# Create the experiment folder
dir.create("~/buckets/b1/exp/HT2810/", showWarnings = FALSE)
setwd("~/buckets/b1/exp/HT2810/")

# Generate a unique run identifier
run_id <- format(now(), "%Y%m%d_%H%M%S")

# Perform grid search with optimized parameter ranges
for (cp in c(-1, 0, 1)) {
  for (maxdepth in c(4, 8, 12)) {
    for (minsplit in c(1000, 400, 100, 20)) {
      for (minbucket in c(5, 20, 50)) {
        if (minbucket <= minsplit) {
          param_basicos <- list(
            "cp" = cp,
            "maxdepth" = maxdepth,
            "minsplit" = minsplit,
            "minbucket" = minbucket
          )
          
          print(paste("Current parameters: cp =", cp, 
                      "maxdepth =", maxdepth, 
                      "minsplit =", minsplit, 
                      "minbucket =", minbucket))
          
          ganancias <- ArbolesMontecarlo(PARAM$semillas, param_basicos)
          
          dt_ganancias <- rbindlist(ganancias, fill=TRUE)
          dt_ganancias[, `:=`(
            cp = cp,
            maxdepth = maxdepth,
            minsplit = minsplit,
            minbucket = minbucket
          )]
          
          if (!"ganancia_test" %in% names(dt_ganancias)) {
            stop("ganancia_test not present in results")
          }
          
          # Append current results to file
          fwrite(dt_ganancias,
                 file = paste0("gridsearch_detalle_", run_id, ".txt"),
                 sep = "\t",
                 append = TRUE)
          
          print(paste("Results appended. Current file size:", 
                      file.size(paste0("gridsearch_detalle_", run_id, ".txt")), 
                      "bytes"))
        }
      }
    }
  }
}

# Rename the detailed results file
file.rename(paste0("gridsearch_detalle_", run_id, ".txt"), "gridsearch_detalle.txt")

# Read the full results file
full_results <- fread("gridsearch_detalle.txt", sep="\t")

# Generate and save summary
tb_grid_search <- full_results[,
  list(
    "ganancia_mean" = mean(ganancia_test),
    "qty" = .N
  ),
  by = list(cp, maxdepth, minsplit, minbucket)
]

# Sort descending by ganancia_mean
setorder(tb_grid_search, -ganancia_mean)

# Add an ID column
tb_grid_search[, id := .I]

# Write the summary to file
fwrite(tb_grid_search,
  file = "gridsearch.txt",
  sep = "\t"
)

print("Grid search completed. Results saved in 'gridsearch_detalle.txt' and 'gridsearch.txt'.")