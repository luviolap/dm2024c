import os
from typing import List, Dict, Any
import polars as pl
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sympy import sieve, prime
import dask
from dask import delayed
from dask.distributed import Client

# Parameters
PARAM = {
    'semilla_primigenia': 102191,  # Replace with YOUR seed
    'qsemillas': 20,
    'training_pct': 70,  # between 1 and 99
    'dataset_nom': "~/datasets/vivencial_dataset_pequeno.csv"  # Choose your dataset
    # 'dataset_nom': "~/datasets/conceptual_dataset_pequeno.csv"  # Uncomment to use this dataset instead
}

def particionar(data: pl.DataFrame, division: List[int], agrupa: str = "", 
                campo: str = "fold", start: int = 1, seed: int = None) -> pl.DataFrame:
    """Perform stratified partitioning of the dataset."""
    if seed is not None:
        np.random.seed(seed)
    
    bloque = np.concatenate([np.repeat(i, d) for i, d in enumerate(division, start=start)])
    
    def assign_fold(group):
        size = group.shape[0]
        return pl.Series(np.random.choice(bloque, size=size, replace=True))
    
    return data.with_column(
        pl.struct([agrupa]).apply(lambda x: assign_fold(x)).alias(campo)
    )

@delayed
def ArbolEstimarGanancia(semilla: int, training_pct: int, param_basicos: Dict[str, Any]) -> Dict[str, Any]:
    """Train a decision tree model and estimate its performance."""
    global dataset
    
    # Partition the dataset
    dataset_partitioned = particionar(dataset.clone(), [training_pct, 100-training_pct], 
                                      agrupa="clase_ternaria", seed=semilla)
    
    # Split the data
    train = dataset_partitioned.filter(pl.col('fold') == 1)
    test = dataset_partitioned.filter(pl.col('fold') == 2)
    
    X_train = train.drop(['clase_ternaria', 'fold'])
    y_train = train.select('clase_ternaria').to_numpy().ravel()
    X_test = test.drop(['clase_ternaria', 'fold'])
    y_test = test.select('clase_ternaria').to_numpy().ravel()
    
    # Train the model
    model = DecisionTreeClassifier(**param_basicos, random_state=semilla)
    model.fit(X_train, y_train)
    
    # Make predictions
    prediccion = model.predict_proba(X_test)
    
    # Calculate gain
    ganancia_test = ((prediccion[:, model.classes_.tolist().index("BAJA+2")] > 0.025) * 
                     (y_test == "BAJA+2") * 117000 + 
                     (prediccion[:, model.classes_.tolist().index("BAJA+2")] > 0.025) * 
                     (y_test != "BAJA+2") * -3000).sum()
    
    ganancia_test_normalizada = ganancia_test / ((100 - training_pct) / 100)
    
    return {
        "semilla": semilla,
        **param_basicos,
        "ganancia_test": ganancia_test_normalizada
    }

def ArbolesMontecarlo(semillas: List[int], param_basicos: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run Monte Carlo simulations for tree models."""
    results = [ArbolEstimarGanancia(semilla, PARAM['training_pct'], param_basicos) for semilla in semillas]
    return dask.compute(*results)

if __name__ == "__main__":
    # Set up Dask client
    client = Client()

    os.chdir("~/buckets/b1/")
    
    # Generate prime numbers for seeds
    sieve.extend(1000000)
    primos = list(sieve.primerange(100000, 1000000))
    np.random.seed(PARAM['semilla_primigenia'])
    PARAM['semillas'] = np.random.choice(primos, PARAM['qsemillas'], replace=False)
    
    # Load data
    dataset = pl.read_csv(PARAM['dataset_nom'])
    dataset = dataset.filter(pl.col('clase_ternaria') != "")
    
    os.makedirs("~/buckets/b1/exp/HT2810", exist_ok=True)
    os.chdir("~/buckets/b1/exp/HT2810")
    
    # Initialize the grid search results dataframe
    tb_grid_search_detalle = pl.DataFrame(schema={
        "semilla": pl.Int32,
        "cp": pl.Float64,
        "max_depth": pl.Int32,
        "min_samples_split": pl.Int32,
        "min_samples_leaf": pl.Int32,
        "ganancia_test": pl.Float64
    })

    # Grid search
    for cp in [-1.0, -0.5, 0.0, 0.5, 1.0]:  # Adjust granularity as needed
        for max_depth in [4, 6, 8, 10, 12, 14]:
            for min_samples_split in [10, 50, 100, 200, 500, 1000]:
                for min_samples_leaf in [5, 10, 20, 50, 100]:
                    param_basicos = {
                        "ccp_alpha": -cp,  # scikit-learn uses ccp_alpha instead of cp
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf
                    }

                    ganancias = ArbolesMontecarlo(PARAM['semillas'], param_basicos)
                    
                    # Add results to the dataframe
                    tb_grid_search_detalle = tb_grid_search_detalle.vstack(pl.DataFrame(ganancias))

                    # Save detailed results after each iteration
                    tb_grid_search_detalle.write_csv("gridsearch_detalle.txt", separator="\t")

    # Generate and save summary
    tb_grid_search = tb_grid_search_detalle.groupby(
        ['cp', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    ).agg([
        pl.mean('ganancia_test').alias('ganancia_mean'),
        pl.count('ganancia_test').alias('qty')
    ]).sort('ganancia_mean', descending=True)

    tb_grid_search = tb_grid_search.with_row_count('id')
    tb_grid_search.write_csv("gridsearch.txt", separator="\t")

    # Close the Dask client
    client.close()