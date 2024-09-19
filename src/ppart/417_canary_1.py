import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Any
import random
from collections import Counter

class Node:
    def __init__(self, feature: Optional[str] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None,
                 value: Optional[Any] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CanaryTree:
    def __init__(self, n_canaries: int = 155, max_depth: int = 30, min_samples_split: int = 2,
                 random_state: Optional[int] = None):
        self.n_canaries = n_canaries
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree: Optional[Node] = None
        self.features: List[str] = []
        np.random.seed(random_state)

    def _add_canaries(self, X: pd.DataFrame) -> pd.DataFrame:
        for i in range(self.n_canaries):
            X[f'canarito_{i}'] = np.random.rand(X.shape[0])
        return X

    def _weighted_gini(self, y: pd.Series, weights: pd.Series) -> float:
        classes, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(counts, dtype=float)
        for i, cls in enumerate(classes):
            weighted_counts[i] = weights[y == cls].sum()
        total_weight = weights.sum()
        return 1 - np.sum((weighted_counts / total_weight) ** 2)

    def _split(self, X: pd.DataFrame, y: pd.Series, weights: pd.Series, feature: str, threshold: float) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
        left_mask = X[feature] <= threshold
        right_mask = ~left_mask
        return (X[left_mask], y[left_mask], weights[left_mask],
                X[right_mask], y[right_mask], weights[right_mask])

    def _best_split(self, X: pd.DataFrame, y: pd.Series, weights: pd.Series) -> Tuple[str, float, float]:
        best_feature, best_threshold, best_gain = '', 0.0, -float('inf')
        current_gini = self._weighted_gini(y, weights)

        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                left_X, left_y, left_w, right_X, right_y, right_w = self._split(X, y, weights, feature, threshold)
                if len(left_y) < self.min_samples_split or len(right_y) < self.min_samples_split:
                    continue

                gain = current_gini - (left_w.sum() / weights.sum() * self._weighted_gini(left_y, left_w) +
                                       right_w.sum() / weights.sum() * self._weighted_gini(right_y, right_w))

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, weights: pd.Series, depth: int = 0) -> Optional[Node]:
        if depth >= self.max_depth or len(y.unique()) == 1 or len(X) < self.min_samples_split:
            return Node(value=y.mode().iloc[0])

        feature, threshold, gain = self._best_split(X, y, weights)

        if feature == '' or feature.startswith('canarito'):  # Stop if we hit a canary or couldn't find a valid split
            return Node(value=y.mode().iloc[0])

        left_X, left_y, left_w, right_X, right_y, right_w = self._split(X, y, weights, feature, threshold)

        left = self._build_tree(left_X, left_y, left_w, depth + 1)
        right = self._build_tree(right_X, right_y, right_w, depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None) -> 'CanaryTree':
        self.features = list(X.columns)
        X_with_canaries = self._add_canaries(X.copy())
        if weights is None:
            weights = pd.Series(np.ones(len(y)), index=y.index)
        self.tree = self._build_tree(X_with_canaries, y, weights)
        return self

    def _predict_single(self, x: pd.Series, node: Optional[Node] = None) -> Any:
        if node is None:
            node = self.tree
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_with_canaries = self._add_canaries(X.copy())
        return pd.Series([self._predict_single(row) for _, row in X_with_canaries.iterrows()], index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        predictions = self.predict(X)
        return pd.get_dummies(predictions)

    def _print_tree(self, node: Optional[Node], depth: int = 0) -> None:
        if node is None:
            return
        if node.value is not None:
            print('  ' * depth + f"Predict {node.value}")
        else:
            print('  ' * depth + f"{node.feature} <= {node.threshold:.2f}")
            self._print_tree(node.left, depth + 1)
            self._print_tree(node.right, depth + 1)

    def print_tree(self) -> None:
        self._print_tree(self.tree)

# Main script to use the CanaryTree with the actual dataset
if __name__ == "__main__":
    import os

    # Set working directory
    os.chdir("~/buckets/b1/")

    # Load the dataset
    dataset = pd.read_csv("~/datasets/vivencial_dataset_pequeno.csv")

    # Create experiment directories
    os.makedirs("./exp/CN4110/", exist_ok=True)
    os.chdir("./exp/CN4110/")

    # Prepare the data
    dtrain = dataset[dataset['foto_mes'] == 202107].copy()
    dapply = dataset[dataset['foto_mes'] == 202109].copy()

    dtrain['clase_binaria2'] = dtrain['clase_ternaria'].map({'CONTINUA': 'NEG', 'BAJA+1': 'POS', 'BAJA+2': 'POS'})
    dtrain = dtrain.drop('clase_ternaria', axis=1)

    # Prepare features and target
    features = [col for col in dtrain.columns if col not in ['clase_binaria2', 'foto_mes', 'numero_de_cliente']]
    X_train = dtrain[features]
    y_train = dtrain['clase_binaria2']

    # Prepare weights
    weights = dtrain['clase_binaria2'].map({'NEG': 1.0, 'POS': 5.0})

    # Create and fit the CanaryTree
    ct = CanaryTree(n_canaries=155, max_depth=30, min_samples_split=2, random_state=102191)
    ct.fit(X_train, y_train, weights)

    # Print the tree structure
    print("Tree structure:")
    ct.print_tree()

    # Make predictions
    X_apply = dapply[features]
    predictions = ct.predict_proba(X_apply)['POS']

    # Prepare the submission
    submission = dapply[['numero_de_cliente']].copy()
    submission['prob'] = predictions
    submission = submission.sort_values('prob', ascending=False)
    submission['Predicted'] = 0
    submission.loc[:11000, 'Predicted'] = 1

    # Save the results
    submission[['numero_de_cliente', 'Predicted']].to_csv("stopping_at_canary.csv", index=False)

    print("\nPredictions saved to 'stopping_at_canary.csv'")

    # You might want to add code here to create a visualization of the tree,
    # similar to the PDF creation in the R script. This would require additional
    # libraries like graphviz, which are not part of the standard library.