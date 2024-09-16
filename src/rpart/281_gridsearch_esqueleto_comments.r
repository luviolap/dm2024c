# Grid search with Montecarlo Cross Validation
# This script performs a grid search over hyperparameters for a decision tree model
# using Monte Carlo Cross-Validation

# Clear the environment and perform garbage collection
rm(list = ls()) # Remove all objects from the environment
gc() # Run garbage collection to free up memory

# Load required libraries
require("data.table") # For efficient data manipulation
require("rpart") # For decision tree modeling
require("parallel") # For parallel processing
require("primes") # For generating prime numbers

# Set up parameters
PARAM <- list() # Initialize an empty list to store parameters
PARAM$semilla_primigenia <- 102191 # Set the primary seed for reproducibility
PARAM$qsemillas <- 20 # Number of seeds to use for Monte Carlo iterations
PARAM$training_pct <- 70L  # Percentage of data to use for training (70%)

# Choose your dataset by commenting/uncommenting the appropriate line
PARAM$dataset_nom <- "~/datasets/vivencial_dataset_pequeno.csv"
# PARAM$dataset_nom <- "~/datasets/conceptual_dataset_pequeno.csv"

#------------------------------------------------------------------------------
# Function to perform stratified partitioning of the dataset
particionar <- function(data, division, agrupa = "", campo = "fold", start = 1, seed = NA) {
  if (!is.na(seed)) set.seed(seed) # Set seed if provided

  # Create a vector of fold assignments
  bloque <- unlist(mapply(function(x, y) {
    rep(y, x)
  }, division, seq(from = start, length.out = length(division))))

  # Assign folds to data points, stratified by 'agrupa'
  data[, (campo) := sample(rep(bloque, ceiling(.N / length(bloque))))[1:.N],
    by = agrupa
  ]
}
#------------------------------------------------------------------------------

# Function to estimate gain for a decision tree model
ArbolEstimarGanancia <- function(semilla, training_pct, param_basicos) {
  # Partition the dataset
  particionar(dataset,
    division = c(training_pct, 100L - training_pct), 
    agrupa = "clase_ternaria",
    seed = semilla
  )

  # Generate the model
  modelo <- rpart("clase_ternaria ~ .",
    data = dataset[fold == 1], # Use training data
    xval = 0,
    control = param_basicos # Pass tree parameters
  )

  # Apply the model to testing data
  prediccion <- predict(modelo,
    dataset[fold == 2], # Use testing data
    type = "prob" # Return probabilities
  )

  # Calculate gain in testing set
  ganancia_test <- dataset[
    fold == 2,
    sum(ifelse(prediccion[, "BAJA+2"] > 0.025,
      ifelse(clase_ternaria == "BAJA+2", 117000, -3000),
      0
    ))
  ]

  # Normalize the gain
  ganancia_test_normalizada <- ganancia_test / ((100 - PARAM$training_pct) / 100)

  # Return results
  return( 
    c(list("semilla" = semilla),
      param_basicos,
      list("ganancia_test" = ganancia_test_normalizada)
    )
  )
}
#------------------------------------------------------------------------------

# Function to perform Monte Carlo simulations
ArbolesMontecarlo <- function(semillas, param_basicos) {
  # Use mcmapply for parallel processing
  salida <- mcmapply(ArbolEstimarGanancia,
    semillas, # Pass the vector of seeds
    MoreArgs = list(PARAM$training_pct, param_basicos), # Additional arguments
    SIMPLIFY = FALSE,
    mc.cores = detectCores() # Use all available cores
  )

  return(salida)
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Set the working directory
setwd("~/buckets/b1/")

# Generate prime numbers for seeds
primos <- generate_primes(min = 100000, max = 1000000)
set.seed(PARAM$semilla_primigenia) # Initialize random number generator
PARAM$semillas <- sample(primos, PARAM$qsemillas) # Select random prime numbers as seeds

# Load the dataset
dataset <- fread(PARAM$dataset_nom)
# Filter for rows with non-empty clase_ternaria
dataset <- dataset[clase_ternaria != ""]

# Create the experiment folder
dir.create("~/buckets/b1/exp/HT2810/", showWarnings = FALSE)
setwd("~/buckets/b1/exp/HT2810/")

# Initialize the data.table for grid search results
tb_grid_search_detalle <- data.table(
  semilla = integer(),
  cp = numeric(),
  maxdepth = integer(),
  minsplit = integer(),
  minbucket = integer(),
  ganancia_test = numeric()
)

# Perform grid search
for (cp in c(-1.0, -0.5, 0, 0.5, 1.0)) {
  for (maxdepth in c(4, 6, 8, 10, 12, 14)) {
    for (minsplit in c(1000, 800, 600, 400, 200, 100, 50, 20, 10)) {
      for (minbucket in c(5, 10, 20, 30, 40, 50)) {
        # Ensure minbucket <= minsplit
        if (minbucket <= minsplit) {
          # Set up parameters for this iteration
          param_basicos <- list(
            "cp" = cp,
            "maxdepth" = maxdepth,
            "minsplit" = minsplit,
            "minbucket" = minbucket
          )

          # Print current parameters for debugging
          print(paste("Current parameters:", 
                      "cp =", cp, 
                      "maxdepth =", maxdepth, 
                      "minsplit =", minsplit, 
                      "minbucket =", minbucket))

          # Perform Monte Carlo simulations
          ganancias <- ArbolesMontecarlo(PARAM$semillas, param_basicos)

          # Print structure of results for debugging
          print("Structure of ganancias:")
          print(str(ganancias))

          # Convert results to a data.table
          dt_ganancias <- rbindlist(ganancias, fill=TRUE)

          # Print structure of converted results for debugging
          print("Structure of dt_ganancias:")
          print(str(dt_ganancias))

          # Ensure all necessary columns are present
          if (!"cp" %in% names(dt_ganancias)) dt_ganancias[, cp := cp]
          if (!"maxdepth" %in% names(dt_ganancias)) dt_ganancias[, maxdepth := maxdepth]
          if (!"minsplit" %in% names(dt_ganancias)) dt_ganancias[, minsplit := minsplit]
          if (!"minbucket" %in% names(dt_ganancias)) dt_ganancias[, minbucket := minbucket]

          # Check for presence of ganancia_test
          if (!"ganancia_test" %in% names(dt_ganancias)) {
            stop("ganancia_test not present in results")
          }

          # Append results to main table
          tb_grid_search_detalle <- rbindlist(list(tb_grid_search_detalle, dt_ganancias))

          # Print summary of results for debugging
          print("Summary of tb_grid_search_detalle:")
          print(summary(tb_grid_search_detalle))
        }
      }
    }

    # Save results after each maxdepth iteration
    fwrite(tb_grid_search_detalle,
           file = "gridsearch_detalle.txt",
           sep = "\t")
    
    # Verify file was saved correctly
    if (file.exists("gridsearch_detalle.txt")) {
      print(paste("File saved. Size:", file.size("gridsearch_detalle.txt"), "bytes"))
    } else {
      print("Error: File was not saved correctly")
    }
  }
}

# Generate and save summary
tb_grid_search <- tb_grid_search_detalle[,
  list("ganancia_mean" = mean(ganancia_test),
    "qty" = .N),
  by = list(cp, maxdepth, minsplit, minbucket)
]

# Sort by descending gain
setorder(tb_grid_search, -ganancia_mean)

# Add an ID column
tb_grid_search[, id := .I]

# Save the summary
fwrite(tb_grid_search,
  file = "gridsearch.txt",
  sep = "\t"
)