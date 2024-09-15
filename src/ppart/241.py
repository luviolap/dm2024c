import os
import polars as pl
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.tree import DecisionTreeClassifier as DaskDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import List, Dict, Any
import random
from dask.distributed import Client
from sympy import sieve, prime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats

# Parameters
PARAM = {
    "semilla_primigenia": 102191,
    "qsemillas": 20,
    "dataset_nom": "~/datasets/vivencial_dataset_pequeno.csv",
    "training_pct": 70,
    "rpart1": {
        "max_depth": 7,
        "min_samples_split": 170,
        "min_samples_leaf": 70,
    },
    "rpart2": {
        "max_depth": 20,
        "min_samples_split": 250,
        "min_samples_leaf": 125,
    }
}

def particionar(data: dd.DataFrame, division: List[int], agrupa: str = "", 
                campo: str = "fold", start: int = 1, seed: int = None) -> dd.DataFrame:
    """Stratified partitioning of the dataset using Dask."""
    def assign_fold(group):
        n = len(group)
        bloque = [val for val, count in zip(range(start, start + len(division)), division) for _ in range(count)]
        return dd.from_pandas(pd.DataFrame({campo: random.sample(bloque * (n // len(bloque) + 1), n)}), npartitions=1)

    if seed is not None:
        random.seed(seed)

    return data.groupby(agrupa).apply(assign_fold, meta={campo: 'int64'}).reset_index(drop=True)

def DosArbolesEstimarGanancia(semilla: int, training_pct: int, param_rpart1: Dict[str, Any], param_rpart2: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate gain using two decision tree models with Dask."""
    global dask_dataset  # Use the global Dask dataset

    # Stratified partitioning of the dataset
    dataset_partitioned = particionar(dask_dataset, 
                          division=[training_pct, 100 - training_pct], 
                          agrupa="clase_ternaria", 
                          seed=semilla)

    # Prepare features and target
    features = [col for col in dataset_partitioned.columns if col not in ['clase_ternaria', 'fold']]
    X = dataset_partitioned[features]
    y = dataset_partitioned['clase_ternaria']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_pct/100, stratify=y, random_state=semilla)

    gains = []

    for params in [param_rpart1, param_rpart2]:
        model = DaskDecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=semilla
        )
        model.fit(X_train, y_train)

        prediction = model.predict_proba(X_test)
        prob_baja2 = prediction[:, model.classes_.tolist().index("BAJA+2")]

        gain_test = ((prob_baja2 > 0.025) * (y_test == "BAJA+2") * 117000 +
                     (prob_baja2 > 0.025) * (y_test != "BAJA+2") * -3000).sum().compute()

        gain_test_normalized = gain_test / ((100 - training_pct) / 100)
        gains.append(gain_test_normalized)

    return {
        "semilla": semilla,
        "ganancia1": gains[0],
        "ganancia2": gains[1]
    }

if __name__ == "__main__":
    # Set up Dask client
    client = Client()

    os.chdir("~/buckets/b1/")  # Set working directory

    # Generate prime numbers
    sieve.extend(1000000)
    primes = [p for p in sieve.primerange(100000, 1000000)]
    
    # Initialize random seed and select PARAM["qsemillas"] seeds
    random.seed(PARAM["semilla_primigenia"])
    PARAM["semillas"] = random.sample(primes, PARAM["qsemillas"])

    # Load data using Dask
    dask_dataset = dd.read_csv(PARAM["dataset_nom"])

    # Work only with data that has a class (202107)
    dask_dataset = dask_dataset[dask_dataset['clase_ternaria'] != ""]

    # Create directory
    os.makedirs("~/buckets/b1/exp/EX2410", exist_ok=True)
    os.chdir("~/buckets/b1/exp/EX2410")

    # Use Dask to parallelize DosArbolesEstimarGanancia for each seed
    results = []
    for semilla in tqdm(PARAM["semillas"]):
        result = DosArbolesEstimarGanancia(semilla, PARAM["training_pct"], PARAM["rpart1"], PARAM["rpart2"])
        results.append(result)

    # Convert the list of dictionaries to a Polars DataFrame
    tb_salida = pl.DataFrame(results)

    # Plot densities
    plt.figure(figsize=(10, 6))
    sns.kdeplot(tb_salida['ganancia1'], shade=True, label='Tree 1')
    sns.kdeplot(tb_salida['ganancia2'], shade=True, label='Tree 2', color='purple')
    plt.title('Density of ganancia for two trees')
    plt.xlabel('ganancia')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('densidad_dos.pdf')
    plt.close()

    # Print mean gains
    print("Mean gains:")
    print(tb_salida.select([
        pl.col('ganancia1').mean().alias('arbol1'),
        pl.col('ganancia2').mean().alias('arbol2')
    ]))

    # Print probability of model 1 being better than model 2
    print("Probability of model 1 being better than model 2:")
    print(tb_salida.select((pl.col('ganancia1') > pl.col('ganancia2')).mean()))

    # Wilcoxon signed-rank test
    wt = stats.wilcoxon(tb_salida['ganancia1'], tb_salida['ganancia2'])
    print(f"Wilcoxon Test p-value: {wt.pvalue}")

    # Close the Dask client
    client.close()