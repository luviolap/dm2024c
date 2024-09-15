import polars as pl
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# Set working directory (if needed)
# import os
# os.chdir("/path/to/your/directory")

def load_data(file_path: str) -> pl.DataFrame:
    """Load the dataset from a CSV file."""
    return pl.read_csv(file_path)

def split_data(data: pl.DataFrame, train_month: int, apply_month: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Split the data into training and application sets."""
    dtrain = data.filter(pl.col('foto_mes') == train_month)
    dapply = data.filter(pl.col('foto_mes') == apply_month)
    return dtrain, dapply

def train_model(X: pl.DataFrame, y: pl.Series, max_depth: int = 3) -> DecisionTreeClassifier:
    """Train the decision tree model."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X.to_numpy(), y.to_numpy())
    return model

def plot_model(model: DecisionTreeClassifier, feature_names: List[str], class_names: List[str]) -> None:
    """Plot the decision tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
    plt.show()

def apply_model(model: DecisionTreeClassifier, X: pl.DataFrame) -> np.ndarray:
    """Apply the model to new data and return probabilities."""
    return model.predict_proba(X.to_numpy())

def process_predictions(dapply: pl.DataFrame, predictions: np.ndarray, threshold: float = 1/40) -> pl.DataFrame:
    """Process predictions and add new columns to the dataframe."""
    return dapply.with_columns([
        pl.Series('prob_baja2', predictions[:, 1]),
        pl.Series('Predicted', (predictions[:, 1] > threshold).astype(int))
    ])

def save_results(results: pl.DataFrame, file_path: str) -> None:
    """Save the results to a CSV file."""
    results.select(['numero_de_cliente', 'Predicted']).write_csv(file_path)

def main() -> None:
    # Load data
    dataset = load_data("~/datasets/vivencial_dataset_pequeno.csv")

    # Split data
    dtrain, dapply = split_data(dataset, train_month=202107, apply_month=202109)

    # Prepare features and target
    features = [col for col in dtrain.columns if col not in ['clase_ternaria', 'foto_mes']]
    X_train = dtrain.select(features)
    y_train = dtrain.select('clase_ternaria').to_series()

    # Train model
    model = train_model(X_train, y_train)

    # Plot model
    plot_model(model, feature_names=features, class_names=model.classes_)

    # Apply model
    X_apply = dapply.select(features)
    predictions = apply_model(model, X_apply)

    # Process predictions
    results = process_predictions(dapply, predictions)

    # Save results
    save_results(results, "./exp/KA2001/K101_001_viv.csv")

if __name__ == "__main__":
    main()