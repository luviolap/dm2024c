"""
CanaryTree: A decision tree classifier with early stopping based on canary variables.

This module implements a decision tree classifier that uses canary variables
(random noise features) to prevent overfitting. The tree stops growing when
it encounters a split on a canary variable.

Classes:
    Node: A node in the decision tree.
    CanaryTree: The main decision tree classifier.

Dependencies:
    - polars
    - numpy
"""

import polars as pl
import numpy as np
from typing import Optional, List, Tuple, Any
import random

class Node:
    """
    A node in the decision tree.

    Attributes:
        feature (str, optional): The feature used for splitting at this node.
        threshold (float, optional): The threshold value for the split.
        left (Node, optional): The left child node.
        right (Node, optional): The right child node.
        value (Any, optional): The predicted value if this is a leaf node.
    """

    def __init__(self, feature: Optional[str] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None,
                 value: Optional[Any] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CanaryTree:
    """
    A decision tree classifier with early stopping based on canary variables.

    This tree adds random noise features (canaries) to the dataset and stops
    growing when it encounters a split on a canary variable, helping to prevent overfitting.

    Attributes:
        n_canaries (int): Number of canary variables to add.
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        random_state (int, optional): Seed for random number generation.
        tree (Node): The root node of the decision tree.
        features (List[str]): List of feature names.
    """

    def __init__(self, n_canaries: int = 155, max_depth: int = 30, min_samples_split: int = 2,
                 random_state: Optional[int] = None):
        self.n_canaries = n_canaries
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree: Optional[Node] = None
        self.features: List[str] = []
        np.random.seed(random_state)

    def _add_canaries(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Add canary variables (random noise) to the dataset.

        Args:
            X (pl.DataFrame): Input features.

        Returns:
            pl.DataFrame: Input features with added canary variables.
        """
        for i in range(self.n_canaries):
            X = X.with_columns(pl.Series(f'canarito_{i}', np.random.rand(X.height)))
        return X

    def _weighted_gini(self, y: pl.Series, weights: pl.Series) -> float:
        """
        Calculate the weighted Gini impurity of a node.

        Args:
            y (pl.Series): Target values.
            weights (pl.Series): Sample weights.

        Returns:
            float: Weighted Gini impurity.
        """
        classes = y.unique()
        weighted_counts = pl.DataFrame({'class': classes, 'count': [0.0] * len(classes)})
        for cls in classes:
            weighted_counts = weighted_counts.with_columns(
                pl.when(pl.col('class') == cls)
                .then(weights.filter(y == cls).sum())
                .otherwise(pl.col('count'))
                .alias('count')
            )
        total_weight = weights.sum()
        return 1 - ((weighted_counts['count'] / total_weight) ** 2).sum()

    def _split(self, X: pl.DataFrame, y: pl.Series, weights: pl.Series, feature: str, threshold: float) -> Tuple[pl.DataFrame, pl.Series, pl.Series, pl.DataFrame, pl.Series, pl.Series]:
        """
        Split the dataset based on a feature and threshold.

        Args:
            X (pl.DataFrame): Input features.
            y (pl.Series): Target values.
            weights (pl.Series): Sample weights.
            feature (str): Feature to split on.
            threshold (float): Threshold value for the split.

        Returns:
            Tuple containing the split datasets (left and right) for X, y, and weights.
        """
        left_mask = X[feature] <= threshold
        right_mask = ~left_mask
        return (X.filter(left_mask), y.filter(left_mask), weights.filter(left_mask),
                X.filter(right_mask), y.filter(right_mask), weights.filter(right_mask))

    def _best_split(self, X: pl.DataFrame, y: pl.Series, weights: pl.Series) -> Tuple[str, float, float]:
        """
        Find the best split for a node.

        Args:
            X (pl.DataFrame): Input features.
            y (pl.Series): Target values.
            weights (pl.Series): Sample weights.

        Returns:
            Tuple containing the best feature, threshold, and information gain.
        """
        best_feature, best_threshold, best_gain = '', 0.0, -float('inf')
        current_gini = self._weighted_gini(y, weights)

        for feature in X.columns:
            thresholds = X[feature].unique().sort()
            for threshold in thresholds:
                left_X, left_y, left_w, right_X, right_y, right_w = self._split(X, y, weights, feature, threshold)
                if left_y.len() < self.min_samples_split or right_y.len() < self.min_samples_split:
                    continue

                gain = current_gini - (left_w.sum() / weights.sum() * self._weighted_gini(left_y, left_w) +
                                       right_w.sum() / weights.sum() * self._weighted_gini(right_y, right_w))

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: pl.DataFrame, y: pl.Series, weights: pl.Series, depth: int = 0) -> Optional[Node]:
        """
        Recursively build the decision tree.

        Args:
            X (pl.DataFrame): Input features.
            y (pl.Series): Target values.
            weights (pl.Series): Sample weights.
            depth (int): Current depth of the tree.

        Returns:
            Node: The root node of the built (sub)tree.
        """
        # Check stopping criteria
        if depth >= self.max_depth or y.n_unique() == 1 or X.height < self.min_samples_split:
            return Node(value=y.mode()[0])

        feature, threshold, gain = self._best_split(X, y, weights)

        # Stop if we hit a canary or couldn't find a valid split
        if feature == '' or feature.startswith('canarito'):
            return Node(value=y.mode()[0])

        left_X, left_y, left_w, right_X, right_y, right_w = self._split(X, y, weights, feature, threshold)

        left = self._build_tree(left_X, left_y, left_w, depth + 1)
        right = self._build_tree(right_X, right_y, right_w, depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X: pl.DataFrame, y: pl.Series, weights: Optional[pl.Series] = None) -> 'CanaryTree':
        """
        Fit the decision tree to the training data.

        Args:
            X (pl.DataFrame): Training features.
            y (pl.Series): Target values.
            weights (pl.Series, optional): Sample weights. If None, uniform weights are used.

        Returns:
            CanaryTree: The fitted tree.
        """
        self.features = X.columns
        X_with_canaries = self._add_canaries(X)
        if weights is None:
            weights = pl.Series([1.0] * X.height)
        self.tree = self._build_tree(X_with_canaries, y, weights)
        return self

    def _predict_single(self, x: pl.Series, node: Optional[Node] = None) -> Any:
        """
        Make a prediction for a single sample.

        Args:
            x (pl.Series): A single sample's features.
            node (Node, optional): The current node in the tree.

        Returns:
            Any: The predicted class.
        """
        if node is None:
            node = self.tree
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """
        Make predictions for multiple samples.

        Args:
            X (pl.DataFrame): Samples to predict.

        Returns:
            pl.Series: Predicted classes.
        """
        X_with_canaries = self._add_canaries(X)
        return pl.Series([self._predict_single(row) for row in X_with_canaries.iter_rows(named=True)])

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Predict class probabilities for samples in X.

        Args:
            X (pl.DataFrame): Samples to predict.

        Returns:
            pl.DataFrame: Class probabilities.
        """
        predictions = self.predict(X)
        return pl.get_dummies(predictions)

    def _print_tree(self, node: Optional[Node], depth: int = 0) -> None:
        """
        Helper method to print the tree structure.

        Args:
            node (Node): The current node.
            depth (int): The current depth in the tree.
        """
        if node is None:
            return
        if node.value is not None:
            print('  ' * depth + f"Predict {node.value}")
        else:
            print('  ' * depth + f"{node.feature} <= {node.threshold:.2f}")
            self._print_tree(node.left, depth + 1)
            self._print_tree(node.right, depth + 1)

    def print_tree(self) -> None:
        """Print the structure of the decision tree."""
        self._print_tree(self.tree)

# Example usage
if __name__ == "__main__":
    import os

    # Set working directory
    os.chdir("~/buckets/b1/")

    # Load the dataset
    dataset = pl.read_csv("~/datasets/vivencial_dataset_pequeno.csv")

    # Create experiment directories
    os.makedirs("./exp/CN4110/", exist_ok=True)
    os.chdir("./exp/CN4110/")

    # Prepare the data
    dtrain = dataset.filter(pl.col('foto_mes') == 202107)
    dapply = dataset.filter(pl.col('foto_mes') == 202109)

    dtrain = dtrain.with_columns(
        pl.when(pl.col('clase_ternaria') == 'CONTINUA')
        .then(pl.lit('NEG'))
        .otherwise(pl.lit('POS'))
        .alias('clase_binaria2')
    ).drop('clase_ternaria')

    # Prepare features and target
    features = [col for col in dtrain.columns if col not in ['clase_binaria2', 'foto_mes', 'numero_de_cliente']]
    X_train = dtrain.select(features)
    y_train = dtrain['clase_binaria2']

    # Prepare weights
    weights = dtrain['clase_binaria2'].map_dict({'NEG': 1.0, 'POS': 5.0})

    # Create and fit the CanaryTree
    ct = CanaryTree(n_canaries=155, max_depth=30, min_samples_split=2, random_state=102191)
    ct.fit(X_train, y_train, weights)

    # Print the tree structure
    print("Tree structure:")
    ct.print_tree()

    # Make predictions
    X_apply = dapply.select(features)
    predictions = ct.predict_proba(X_apply).select('POS')

    # Prepare the submission
    submission = dapply.select('numero_de_cliente').with_columns(predictions)
    submission = submission.sort('POS', descending=True)
    submission = submission.with_columns(pl.lit(0).alias('Predicted'))
    submission = submission.with_columns(
        pl.when(pl.col('POS').rank(method='dense', descending=True) <= 11000)
        .then(pl.lit(1))
        .otherwise(pl.col('Predicted'))
        .alias('Predicted')
    )

    # Save the results
    submission.select(['numero_de_cliente', 'Predicted']).write_csv("stopping_at_canaritos.csv")

    print("\nPredictions saved to 'stopping_at_canaritos.csv'")