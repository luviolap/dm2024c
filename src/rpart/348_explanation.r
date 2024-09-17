# Bayesian Optimization of hyperparameters for rpart using 5-fold cross-validation

#------------------------------------------------------------------------------
# Section 1: Environment Setup
#------------------------------------------------------------------------------

# Clear the memory to ensure a clean workspace
rm(list = ls()) # Remove all objects from the environment
gc() # Run garbage collection to free up memory

# Load required libraries
require("data.table") # For efficient data manipulation
require("rlist") # For working with lists
require("yaml") # For reading YAML configuration files
require("rpart") # For decision tree modeling
require("parallel") # For parallel processing

# Load packages needed specifically for Bayesian Optimization
require("DiceKriging") # For Gaussian process modeling
require("mlrMBO") # For model-based optimization

#------------------------------------------------------------------------------
# Section 2: Bayesian Optimization Parameter Setup
#------------------------------------------------------------------------------

# Initialize a list to store Bayesian Optimization parameters
PARAM <- list()

# Set the number of iterations for Bayesian Optimization
PARAM$BO_iter <- 100 # 100 iterations will be performed

# Define the hyperparameter search space
PARAM$hs <- makeParamSet(
    # Complexity parameter for pruning (continuous)
    makeNumericParam("cp", lower = -1, upper = 0.1),
    # Minimum number of observations in a node for a split (integer)
    makeIntegerParam("minsplit", lower = 1L, upper = 8000L),
    # Minimum number of observations in any terminal node (integer)
    makeIntegerParam("minbucket", lower = 1L, upper = 4000L),
    # Maximum depth of the tree (integer)
    makeIntegerParam("maxdepth", lower = 3L, upper = 20L),
    # Constraint: minbucket must not be greater than half of minsplit
    forbidden = quote(minbucket > 0.5 * minsplit)
)

#------------------------------------------------------------------------------
# Section 3: Utility Functions
#------------------------------------------------------------------------------

# Function to log results to a file
loguear <- function(reg, arch = NA, folder = "./work/", ext = ".txt",
                    verbose = TRUE) {
    archivo <- arch
    if (is.na(arch)) archivo <- paste0(substitute(reg), ext)

    # Write headers if file doesn't exist
    if (!file.exists(archivo)) {
        linea <- paste0(
            "fecha\t",
            paste(list.names(reg), collapse = "\t"), "\n"
        )
        cat(linea, file = archivo)
    }

    # Prepare log line with timestamp and data
    linea <- paste0(
        format(Sys.time(), "%Y%m%d %H%M%S"), "\t",
        gsub(", ", "\t", toString(reg)), "\n"
    )

    # Append to file
    cat(linea, file = archivo, append = TRUE)

    # Print to console if verbose is TRUE
    if (verbose) cat(linea)
}

# Function to partition the dataset into folds for cross-validation
particionar <- function(data, division, agrupa = "", campo = "fold",
                        start = 1, seed = NA) {
    if (!is.na(seed)) set.seed(seed)

    # Create blocks for partitioning
    bloque <- unlist(mapply(
        function(x, y) {
            rep(y, x)
        }, division,
        seq(from = start, length.out = length(division))
    ))

    # Assign folds to data
    data[, (campo) := sample(rep(bloque, ceiling(.N / length(bloque))))[1:.N],
        by = agrupa
    ]
}

#------------------------------------------------------------------------------
# Section 4: Model Training and Evaluation
#------------------------------------------------------------------------------

# Function to train and evaluate a single decision tree
ArbolSimple <- function(fold_test, param_rpart) {
    # Train the model on all folds except the test fold
    modelo <- rpart("clase_ternaria ~ .",
        data = dataset[fold != fold_test, ],
        xval = 0,
        control = param_rpart
    )

    # Make predictions on the test fold
    prediccion <- predict(modelo,
        dataset[fold == fold_test, ],
        type = "prob"
    )

    # Extract probability of "BAJA+2" class
    prob_baja2 <- prediccion[, "BAJA+2"]

    # Calculate the gain for the test fold
    ganancia_testing <- dataset[fold == fold_test][
        prob_baja2 > 1 / 40,
        sum(ifelse(clase_ternaria == "BAJA+2",
            117000, -3000
        ))
    ]

    return(ganancia_testing)
}

# Function to perform cross-validation
ArbolesCrossValidation <- function(param_rpart, qfolds, pagrupa, semilla) {
    # Create partition for cross-validation
    divi <- rep(1, qfolds)
    particionar(dataset, divi, seed = semilla, agrupa = pagrupa)

    # Apply ArbolSimple function to each fold in parallel
    ganancias <- mcmapply(ArbolSimple,
        seq(qfolds),
        MoreArgs = list(param_rpart),
        SIMPLIFY = FALSE,
        mc.cores = detectCores()
    )

    # Remove fold column from dataset
    dataset[, fold := NULL]

    # Calculate average gain and normalize
    ganancia_promedio <- mean(unlist(ganancias))
    ganancia_promedio_normalizada <- ganancia_promedio * qfolds

    return(ganancia_promedio_normalizada)
}

#------------------------------------------------------------------------------
# Section 5: Objective Function for Bayesian Optimization
#------------------------------------------------------------------------------

# This function estimates the gain for a given set of hyperparameters
EstimarGanancia <- function(x) {
    GLOBAL_iteracion <<- GLOBAL_iteracion + 1

    xval_folds <- 5
    # Perform cross-validation and get normalized gain
    ganancia <- ArbolesCrossValidation(
        param_rpart = x,
        qfolds = xval_folds,
        pagrupa = "clase_ternaria",
        semilla = miAmbiente$semilla_primigenia
    )

    # Prepare results for logging
    xx <- x
    xx$xval_folds <- xval_folds
    xx$ganancia <- ganancia
    xx$iteracion <- GLOBAL_iteracion

    # Update and log best result if necessary
    if (ganancia > GLOBAL_mejor) {
        GLOBAL_mejor <<- ganancia
        Sys.sleep(2)
        loguear(xx, arch = archivo_log_mejor)
    }

    # Log current result
    Sys.sleep(2)
    loguear(xx, arch = archivo_log)

    return(ganancia)
}

#------------------------------------------------------------------------------
# Section 6: Main Program
#------------------------------------------------------------------------------

# Set working directory
setwd("~/buckets/b1/")

# Load environment variables
miAmbiente <- read_yaml("~/buckets/b1/miAmbiente.yml")

# Load and filter dataset
dataset <- fread(miAmbiente$dataset_pequeno)
dataset <- dataset[foto_mes == 202107]

# Create experiment directory
dir.create("./exp/", showWarnings = FALSE)
dir.create("./exp/HT3480/", showWarnings = FALSE)
setwd("./exp/HT3480/")

# Define log files
archivo_log <- "HT348.txt"
archivo_log_mejor <- "HT348_mejor.txt"
archivo_BO <- "HT348.RDATA"

# Initialize global variables
GLOBAL_iteracion <- 0
GLOBAL_mejor <- -Inf

# Load existing log if available
if (file.exists(archivo_log)) {
    tabla_log <- fread(archivo_log)
    GLOBAL_iteracion <- nrow(tabla_log)
    GLOBAL_mejor <- tabla_log[, max(ganancia)]
}

#------------------------------------------------------------------------------
# Section 7: Bayesian Optimization Configuration
#------------------------------------------------------------------------------

# Set the optimization function
funcion_optimizar <- EstimarGanancia

# Configure mlr settings
configureMlr(show.learner.output = FALSE)

# Create objective function for optimization
obj.fun <- makeSingleObjectiveFunction(
    fn = funcion_optimizar,
    minimize = FALSE,  # We want to maximize gain
    noisy = TRUE,      # The objective function is noisy
    par.set = PARAM$hs,
    has.simple.signature = FALSE
)

# Configure MBO control
ctrl <- makeMBOControl(
    save.on.disk.at.time = 600,  # Save every 10 minutes
    save.file.path = archivo_BO
)

# Set termination criteria
ctrl <- setMBOControlTermination(ctrl, iters = PARAM$BO_iter)

# Set infill criterion to Expected Improvement
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

# Configure surrogate model (Gaussian Process)
surr.km <- makeLearner("regr.km",
    predict.type = "se",
    covtype = "matern3_2", control = list(trace = TRUE)
)

#------------------------------------------------------------------------------
# Section 8: Run Bayesian Optimization
#------------------------------------------------------------------------------

# Start or continue Bayesian Optimization
if (!file.exists(archivo_BO)) {
    # Start new optimization run
    run <- mbo(
        fun = obj.fun,
        learner = surr.km,
        control = ctrl
    )
} else {
    # Continue from previous run
    run <- mboContinue(archivo_BO)
}

#------------------------------------------------------------------------------
# Section 9: Post-Processing
#------------------------------------------------------------------------------

# Copy results to another location
system("~/install/repobrutalcopy.sh")

# Shut down the virtual machine to save resources
system("~/install/apagar-vm.sh")