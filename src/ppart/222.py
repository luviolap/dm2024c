import polars as pl
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import List, Dict, Any
import random
from multiprocessing import Pool, cpu_count
import os

# Parameters
PARAM = {
    "semillas": [102191, 200177, 410551, 552581, 892237],
    "dataset_nom": "~/datasets/vivencial_dataset_pequeno.csv",
    # "dataset_nom": "~/datasets/conceptual_dataset_pequeno.csv",
    "training_pct": 70,  # between 1 and 99
    "rpart": {
        "max_depth": 7,
        "min_samples_split": 170,
        "min_samples_leaf": 70,
    }
}

def particionar(data: pl.DataFrame, division: List[int], agrupa: str = "", 
                campo: str = "fold", start: int = 1, seed: int = None) -> pl.DataFrame:
    """Stratified partitioning of the dataset."""
    if seed is not None:
        random.seed(seed)
    
    def assign_fold(group):
        n = len(group)
        bloque = [val for val, count in zip(range(start, start + len(division)), division) for _ in range(count)]
        folds = random.sample(bloque * (n // len(bloque) + 1), n)
        return pl.Series(name=campo, values=folds)

    return data.groupby(agrupa).agg(
        [pl.all().exclude(agrupa), assign_fold().alias(campo)]
    ).explode(pl.all())

def ArbolEstimarGanancia(semilla: int, param_basicos: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate gain using a decision tree model."""
    global dataset  # Use the global dataset

    # Stratified partitioning of the dataset
    dataset_partitioned = particionar(dataset, 
                          division=[param_basicos["training_pct"], 100 - param_basicos["training_pct"]], 
                          agrupa="clase_ternaria", 
                          seed=semilla)

    # Generate the model
    features = [col for col in dataset_partitioned.columns if col not in ['clase_ternaria', 'fold']]
    X_train = dataset_partitioned.filter(pl.col('fold') == 1).select(features)
    y_train = dataset_partitioned.filter(pl.col('fold') == 1).select('clase_ternaria').to_series()

    model = DecisionTreeClassifier(
        max_depth=param_basicos["rpart"]["max_depth"],
        min_samples_split=param_basicos["rpart"]["min_samples_split"],
        min_samples_leaf=param_basicos["rpart"]["min_samples_leaf"],
        random_state=semilla
    )
    model.fit(X_train, y_train)

    # Apply the model to testing data
    X_test = dataset_partitioned.filter(pl.col('fold') == 2).select(features)
    prediction = model.predict_proba(X_test)

    # Calculate gain in testing
    test_data = dataset_partitioned.filter(pl.col('fold') == 2)
    test_data = test_data.with_column(pl.Series(name='prob_baja2', values=prediction[:, model.classes_.tolist().index("BAJA+2")]))
    
    gain_test = test_data.select(
        pl.when(pl.col('prob_baja2') > 0.025)
        .then(pl.when(pl.col('clase_ternaria') == "BAJA+2").then(117000).otherwise(-3000))
        .otherwise(0)
        .sum()
    ).item()

    # Scale the gain as if it were the entire dataset
    gain_test_normalized = gain_test / ((100 - param_basicos["training_pct"]) / 100)

    return {
        "semilla": semilla,
        "testing": test_data.shape[0],
        "testing_pos": test_data.filter(pl.col('clase_ternaria') == "BAJA+2").shape[0],
        "envios": test_data.filter(pl.col('prob_baja2') > 0.025).shape[0],
        "aciertos": test_data.filter((pl.col('prob_baja2') > 0.025) & (pl.col('clase_ternaria') == "BAJA+2")).shape[0],
        "ganancia_test": gain_test_normalized
    }

if __name__ == "__main__":
    os.chdir("~/buckets/b1/")  # Set working directory

    # Load data
    dataset = pl.read_csv(PARAM["dataset_nom"])

    # Work only with data that has a class (202107)
    dataset = dataset.filter(pl.col('clase_ternaria') != "")

    # Use multiprocessing to call ArbolEstimarGanancia for each seed
    with Pool(cpu_count()) as pool:
        salidas = pool.starmap(ArbolEstimarGanancia, [(semilla, PARAM) for semilla in PARAM["semillas"]])

    # Convert the list of dictionaries to a Polars DataFrame
    tb_salida = pl.DataFrame(salidas)

    print(tb_salida)

    # Calculate and print the average gain
    print(f"Average gain: {tb_salida['ganancia_test'].mean()}")

    # Calculate and print the standard deviation of gain
    print(f"Gain standard deviation: {tb_salida['ganancia_test'].std()}")