import os
from typing import List, Dict, Any, Tuple
import polars as pl
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sympy import sieve, prime
from scipy import stats
import dask
from dask import delayed
from dask.distributed import Client

# Parameters
PARAM: Dict[str, Any] = {
    'semilla_primigenia': 102191,
    'qsemillas_tope': 50,
    'dataset_nom': "~/datasets/vivencial_dataset_pequeno.csv",
    'training_pct': 50,  # between 1 and 99
    'rpart1': {
        'max_depth': 7,
        'min_samples_split': 800,
        'min_samples_leaf': 400,
    },
    'rpart2': {
        'max_depth': 6,
        'min_samples_split': 650,
        'min_samples_leaf': 300,
    }
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
def DosArbolesEstimarGanancia(semilla: int, training_pct: int, 
                              param_rpart1: Dict[str, Any], 
                              param_rpart2: Dict[str, Any]) -> Dict[str, float]:
    """Train two decision tree models and estimate their performance."""
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

def MejorArbol(qsemillas_tope: int, training_pct: int, 
               param_rpart1: Dict[str, Any], param_rpart2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two tree models and determine which is better."""
    # Generate prime numbers for seeds
    sieve.extend(1000000)
    primos = list(sieve.primerange(100000, 1000000))
    np.random.seed(PARAM['semilla_primigenia'])
    semillas = np.random.choice(primos, qsemillas_tope, replace=False)

    pvalue = 1.0
    isem = 0
    vgan1, vgan2 = [], []

    while (isem < qsemillas_tope) and (pvalue > 0.05):
        res = DosArbolesEstimarGanancia(semillas[isem], training_pct, param_rpart1, param_rpart2).compute()
        vgan1.append(res['ganancia1'])
        vgan2.append(res['ganancia2'])

        wt = stats.wilcoxon(vgan1, vgan2)
        pvalue = wt.pvalue

        print(f"{isem+1}, {res['ganancia1']}, {res['ganancia2']}, {pvalue}")
        isem += 1

    out = 0
    if pvalue < 0.05:
        out = 1 if np.mean(vgan1) > np.mean(vgan2) else 2

    return {
        "out": out,
        "qsemillas": len(vgan1),
        "m1": np.mean(vgan1),
        "m2": np.mean(vgan2)
    }

if __name__ == "__main__":
    # Set up Dask client
    client = Client()

    os.chdir("~/buckets/b1/")
    
    # Load data
    dataset = pl.read_csv(PARAM['dataset_nom'])
    dataset = dataset.filter(pl.col('clase_ternaria') != "")
    
    os.makedirs("~/buckets/b1/exp/EX2680", exist_ok=True)
    os.chdir("~/buckets/b1/exp/EX2680")
    
    comparacion = MejorArbol(
        PARAM['qsemillas_tope'],
        PARAM['training_pct'],
        PARAM['rpart1'],
        PARAM['rpart2']
    )

    print(comparacion)

    # Close the Dask client
    client.close()