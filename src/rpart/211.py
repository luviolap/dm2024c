import polars as pl
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import List, Dict, Any
import random

# Parameters
PARAM = {
    "semilla": 102191,
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

# Load data
dataset = pl.read_csv("~/datasets/vivencial_dataset_pequeno.csv")

# Work only with data that has a class (202107)
dataset = dataset.filter(pl.col('clase_ternaria') != "")

# Stratified partitioning of the dataset 70%, 30%
dataset = particionar(dataset, 
                      division=[PARAM["training_pct"], 100 - PARAM["training_pct"]], 
                      agrupa="clase_ternaria", 
                      seed=PARAM["semilla"])

# Generate the model
features = [col for col in dataset.columns if col not in ['clase_ternaria', 'fold']]
X_train = dataset.filter(pl.col('fold') == 1).select(features)
y_train = dataset.filter(pl.col('fold') == 1).select('clase_ternaria').to_series()

model = DecisionTreeClassifier(
    max_depth=PARAM["rpart"]["max_depth"],
    min_samples_split=PARAM["rpart"]["min_samples_split"],
    min_samples_leaf=PARAM["rpart"]["min_samples_leaf"],
    random_state=PARAM["semilla"]
)
model.fit(X_train, y_train)

# Apply the model to testing data
X_test = dataset.filter(pl.col('fold') == 2).select(features)
prediction = model.predict_proba(X_test)

# Add a column for gains
dataset = dataset.with_column(
    pl.when(pl.col('clase_ternaria') == "BAJA+2")
    .then(117000)
    .otherwise(-3000)
    .alias('ganancia')
)

# Add probability for testing data
dataset = dataset.with_column(
    pl.Series(name='prob_baja2', values=prediction[:, model.classes_.tolist().index("BAJA+2")])
    .filter(pl.col('fold') == 2)
)

# Calculate gain in testing
gain_test = dataset.filter((pl.col('fold') == 2) & (pl.col('prob_baja2') > 0.025)).select('ganancia').sum().item()

# Scale the gain as if it were the entire dataset
gain_test_normalized = gain_test / ((100 - PARAM["training_pct"]) / 100)

stimuli = dataset.filter((pl.col('fold') == 2) & (pl.col('prob_baja2') > 0.025)).shape[0]
hits = dataset.filter((pl.col('fold') == 2) & (pl.col('prob_baja2') > 0.025) & (pl.col('clase_ternaria') == "BAJA+2")).shape[0]

print(f"Testing total: {dataset.filter(pl.col('fold') == 2).shape[0]}")
print(f"Testing BAJA+2: {dataset.filter((pl.col('fold') == 2) & (pl.col('clase_ternaria') == 'BAJA+2')).shape[0]}")
print(f"Stimuli: {stimuli}")
print(f"Hits (BAJA+2): {hits}")
print(f"Gain in testing (normalized): {gain_test_normalized}")