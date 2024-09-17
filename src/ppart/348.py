from typing import Dict, List, Optional, Any
import polars as pl
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import yaml
import os
from datetime import datetime
import dask
from dask.distributed import Client
import dask.dataframe as dd

# Set up Dask client for parallel processing
client = Client()

# Define Bayesian Optimization parameters
PARAM: Dict[str, Any] = {
    'BO_iter': 100,  # number of iterations for Bayesian Optimization
    'hs': {
        'cp': Real(-1, 0.1),
        'min_samples_split': Integer(1, 8000),
        'min_samples_leaf': Integer(1, 4000),
        'max_depth': Integer(3, 20)
    }
}

def log_results(reg: Dict[str, Any], arch: Optional[str] = None, folder: str = "./work/", ext: str = ".txt", verbose: bool = True) -> None:
    """
    Log results to a file and optionally print to console.

    Args:
        reg: Dictionary containing results to log.
        arch: Name of the log file. Defaults to None.
        folder: Folder to save the log file. Defaults to "./work/".
        ext: File extension for the log file. Defaults to ".txt".
        verbose: Whether to print results to console. Defaults to True.
    """
    archivo: str = arch if arch else f"{next(iter(reg))}{ext}"
    filepath: str = os.path.join(folder, archivo)
    
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write("fecha\t" + "\t".join(reg.keys()) + "\n")
    
    line: str = f"{datetime.now().strftime('%Y%m%d %H%M%S')}\t" + "\t".join(map(str, reg.values())) + "\n"
    
    with open(filepath, 'a') as f:
        f.write(line)
    
    if verbose:
        print(line)

def partition(data: pl.DataFrame, division: List[int], group_col: str = "", fold_col: str = "fold", start: int = 1, seed: Optional[int] = None) -> pl.DataFrame:
    """
    Partition the dataset into folds, optionally stratified by a group column.

    Args:
        data: Input dataframe.
        division: List specifying the size of each fold.
        group_col: Column to use for stratification. Defaults to "".
        fold_col: Name of the new fold column. Defaults to "fold".
        start: Starting value for fold numbering. Defaults to 1.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Dataframe with a new column indicating the fold for each row.
    """
    if seed is not None:
        np.random.seed(seed)
    
    blocks: np.ndarray = np.repeat(range(start, start + len(division)), division)
    
    if group_col:
        return data.with_column(
            pl.col(group_col).shuffle().over(group_col).map_groups(lambda x: np.random.choice(blocks, size=len(x))).alias(fold_col)
        )
    else:
        return data.with_column(
            pl.Series(name=fold_col, values=np.random.choice(blocks, size=len(data)))
        )

@dask.delayed
def simple_tree(fold_test: int, X: dd.DataFrame, y: dd.Series, param_tree: Dict[str, Any]) -> float:
    """
    Train a decision tree on a subset of data and evaluate it.

    Args:
        fold_test: The fold number to use for testing.
        X: Feature dataframe.
        y: Target series.
        param_tree: Parameters for the decision tree.

    Returns:
        The calculated gain for this fold.
    """
    mask = X['fold'] != fold_test
    X_train, y_train = X[mask], y[mask]
    X_test, y_test = X[~mask], y[~mask]
    
    model: DecisionTreeClassifier = DecisionTreeClassifier(**param_tree)
    model.fit(X_train.compute(), y_train.compute())
    
    prob_baja2: np.ndarray = model.predict_proba(X_test.compute())[:, 1]
    
    ganancia_testing: float = ((prob_baja2 > 1/40) * 
                        ((y_test.compute() == "BAJA+2") * 117000 - 
                         (y_test.compute() != "BAJA+2") * 3000)).sum()
    
    return ganancia_testing

def trees_cross_validation(param_tree: Dict[str, Any], qfolds: int, group_col: str, seed: int) -> float:
    """
    Perform cross-validation for decision trees.

    Args:
        param_tree: Parameters for the decision tree.
        qfolds: Number of folds for cross-validation.
        group_col: Column to use for stratification.
        seed: Random seed for reproducibility.

    Returns:
        Normalized average gain across all folds.
    """
    global dataset
    
    dataset = partition(dataset, [1]*qfolds, group_col=group_col, seed=seed)
    
    dask_dataset: dd.DataFrame = dd.from_pandas(dataset.to_pandas(), npartitions=qfolds)
    X: dd.DataFrame = dask_dataset.drop('clase_ternaria', axis=1)
    y: dd.Series = dask_dataset['clase_ternaria']
    
    ganancias: List[float] = [simple_tree(fold, X, y, param_tree) for fold in range(qfolds)]
    ganancias = dask.compute(*ganancias)
    
    dataset = dataset.drop('fold')
    
    ganancia_promedio: float = np.mean(ganancias)
    ganancia_promedio_normalizada: float = ganancia_promedio * qfolds
    
    return ganancia_promedio_normalizada

def estimate_gain(cp: float, min_samples_split: int, min_samples_leaf: int, max_depth: int) -> float:
    """
    Estimate the gain for a given set of decision tree parameters.

    Args:
        cp: Complexity parameter.
        min_samples_split: Minimum number of samples required to split an internal node.
        min_samples_leaf: Minimum number of samples required to be at a leaf node.
        max_depth: Maximum depth of the tree.

    Returns:
        Estimated gain.
    """
    global GLOBAL_iteracion, GLOBAL_mejor, archivo_log, archivo_log_mejor
    
    GLOBAL_iteracion += 1
    
    xval_folds: int = 5
    param_tree: Dict[str, Any] = {
        'ccp_alpha': cp,  # scikit-learn uses ccp_alpha instead of cp
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_depth': max_depth
    }
    
    ganancia: float = trees_cross_validation(
        param_tree, xval_folds, "clase_ternaria", mi_ambiente['semilla_primigenia']
    )
    
    log_data: Dict[str, Any] = {**param_tree, 'xval_folds': xval_folds, 'ganancia': ganancia, 'iteracion': GLOBAL_iteracion}
    
    if ganancia > GLOBAL_mejor:
        GLOBAL_mejor = ganancia
        log_results(log_data, arch=archivo_log_mejor)
    
    log_results(log_data, arch=archivo_log)
    
    return ganancia

# Main program
if __name__ == "__main__":
    # Set working directory
    os.chdir("~/buckets/b1/")

    # Load environment
    with open("~/buckets/b1/miAmbiente.yml", 'r') as file:
        mi_ambiente: Dict[str, Any] = yaml.safe_load(file)

    # Load dataset
    dataset: pl.DataFrame = pl.read_csv(mi_ambiente['dataset_pequeno'])
    dataset = dataset.filter(pl.col('foto_mes') == 202107)

    # Create experiment folder
    os.makedirs("./exp/HT3480/", exist_ok=True)
    os.chdir("./exp/HT3480/")

    archivo_log: str = "HT348.txt"
    archivo_log_mejor: str = "HT348_mejor.txt"
    archivo_BO: str = "HT348.pkl"

    GLOBAL_iteracion: int = 0
    GLOBAL_mejor: float = float('-inf')

    if os.path.exists(archivo_log):
        tabla_log: pl.DataFrame = pl.read_csv(archivo_log, separator='\t')
        GLOBAL_iteracion = len(tabla_log)
        GLOBAL_mejor = tabla_log['ganancia'].max()

    # Bayesian Optimization configuration
    opt: BayesSearchCV = BayesSearchCV(
        DecisionTreeClassifier(),
        PARAM['hs'],
        n_iter=PARAM['BO_iter'],
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    # Start Bayesian Optimization
    opt.fit(dataset.drop('clase_ternaria').to_pandas(), dataset['clase_ternaria'].to_pandas())

    # Close the Dask client
    client.close()