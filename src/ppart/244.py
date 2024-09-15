import os
from typing import List, Dict, Any
import polars as pl
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import prime, sieve
import dask
from dask import delayed
from dask.distributed import Client

# Parameters
PARAM: Dict[str, Any] = {
    'semilla_primigenia': 102191,
    'qsemillas': 200,
    'dataset_nom': "~/datasets/vivencial_dataset_pequeno.csv",
    'training_pct': 70,
    'rpart1': {
        'max_depth': 7,
        'min_samples_split': 170,
        'min_samples_leaf': 70
    },
    'rpart2': {
        'max_depth': 3,
        'min_samples_split': 1900,
        'min_samples_leaf': 800
    }
}

def particionar(data: pl.DataFrame, division: List[int], agrupa: str = "", 
                campo: str = "fold", start: int = 1, seed: int = None) -> pl.DataFrame:
    """
    Perform stratified partitioning of the dataset.

    Args:
    data: Input dataset
    division: List of partition sizes
    agrupa: Column name to group by for stratification
    campo: Name of the new column to store fold information
    start: Starting value for fold numbering
    seed: Random seed for reproducibility

    Returns:
    Dataset with added fold column
    """
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
def DosArbolesEstimarGanancia(semilla: int, training_pct: int, 
                              param_rpart1: Dict[str, Any], 
                              param_rpart2: Dict[str, Any]) -> Dict[str, float]:
    """
    Train two decision tree models and estimate their performance.

    Args:
    semilla: Random seed for reproducibility
    training_pct: Percentage of data to use for training
    param_rpart1: Parameters for the first decision tree model
    param_rpart2: Parameters for the second decision tree model

    Returns:
    Dictionary containing seed and normalized gains for both models
    """
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
    
    def train_and_evaluate(params: Dict[str, Any]) -> float:
        model = DecisionTreeClassifier(**params, random_state=semilla)
        model.fit(X_train, y_train)
        prediccion = model.predict_proba(X_test)
        
        ganancia_test = ((prediccion[:, model.classes_.tolist().index("BAJA+2")] > 0.025) * 
                         (y_test == "BAJA+2") * 117000 + 
                         (prediccion[:, model.classes_.tolist().index("BAJA+2")] > 0.025) * 
                         (y_test != "BAJA+2") * -3000).sum()
        
        return ganancia_test / ((100 - training_pct) / 100)
    
    # Train and evaluate both models
    ganancia1 = train_and_evaluate(param_rpart1)
    ganancia2 = train_and_evaluate(param_rpart2)
    
    return {
        "semilla": semilla,
        "ganancia1": ganancia1,
        "ganancia2": ganancia2
    }

if __name__ == "__main__":
    # Set up Dask client
    client = Client()
    
    os.chdir("~/buckets/b1/")
    
    # Generate prime numbers for seeds
    sieve.extend(1000000)
    primos: List[int] = list(sieve.primerange(100000, 1000000))
    np.random.seed(PARAM['semilla_primigenia'])
    PARAM['semillas'] = np.random.choice(primos, PARAM['qsemillas'], replace=False)
    
    # Load data
    dataset: pl.DataFrame = pl.read_csv(PARAM['dataset_nom'])
    dataset = dataset.filter(pl.col('clase_ternaria') != "")
    
    os.makedirs("~/buckets/b1/exp/EX2440", exist_ok=True)
    os.chdir("~/buckets/b1/exp/EX2440")
    
    # Create Dask delayed objects for parallel processing
    results: List[Any] = [DosArbolesEstimarGanancia(semilla, PARAM['training_pct'], PARAM['rpart1'], PARAM['rpart2']) 
                          for semilla in PARAM['semillas']]
    
    # Compute results in parallel
    salidas: List[Dict[str, float]] = dask.compute(*results)
    
    # Convert results to DataFrame
    tb_salida: pl.DataFrame = pl.DataFrame(salidas)
    
    # Plot densities
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=tb_salida.to_pandas(), x='ganancia1', shade=True, alpha=0.25, label='Model 1')
    sns.kdeplot(data=tb_salida.to_pandas(), x='ganancia2', shade=True, color='purple', alpha=0.10, label='Model 2')
    plt.title('Density of Gains for Two Models')
    plt.xlabel('Gain')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('densidad_dos.pdf')
    plt.close()
    
    # Print results
    print("Mean gains:")
    print(tb_salida.select([pl.col('ganancia1').mean(), pl.col('ganancia2').mean()]))
    print(f"Probability of Model 1 outperforming Model 2: {(tb_salida['ganancia1'] > tb_salida['ganancia2']).mean():.4f}")
    
    # Uncomment for Wilcoxon test
    # from scipy import stats
    # wt = stats.wilcoxon(tb_salida['ganancia1'].to_numpy(), tb_salida['ganancia2'].to_numpy())
    # print(f"Wilcoxon Test p-value: {wt.pvalue:.4f}")

    # Close the Dask client
    client.close()