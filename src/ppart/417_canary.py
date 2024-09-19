from typing import List, Tuple, Dict, Optional, Any
import random
from collections import Counter
import math

class Node:
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None,
                 value: Optional[Any] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CanaryTree:
    def __init__(self, n_canaries: int = 10, max_depth: int = 10, min_samples_split: int = 2,
                 random_state: Optional[int] = None):
        self.n_canaries = n_canaries
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree: Optional[Node] = None
        self.n_features: int = 0
        random.seed(random_state)

    def _add_canaries(self, X: List[List[float]]) -> List[List[float]]:
        return [x + [random.random() for _ in range(self.n_canaries)] for x in X]

    def _gini(self, y: List[Any]) -> float:
        counts = Counter(y)
        impurity = 1.0
        for count in counts.values():
            p = count / len(y)
            impurity -= p * p
        return impurity

    def _split(self, X: List[List[float]], y: List[Any], feature: int, threshold: float) -> Tuple[List[List[float]], List[Any], List[List[float]], List[Any]]:
        left_X, left_y, right_X, right_y = [], [], [], []
        for x, label in zip(X, y):
            if x[feature] <= threshold:
                left_X.append(x)
                left_y.append(label)
            else:
                right_X.append(x)
                right_y.append(label)
        return left_X, left_y, right_X, right_y

    def _best_split(self, X: List[List[float]], y: List[Any]) -> Tuple[int, float, float]:
        best_feature, best_threshold, best_gain = -1, 0.0, -float('inf')
        current_gini = self._gini(y)

        for feature in range(len(X[0])):
            thresholds = sorted(set(x[feature] for x in X))
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self._split(X, y, feature, threshold)
                if len(left_y) < self.min_samples_split or len(right_y) < self.min_samples_split:
                    continue

                gain = current_gini - (len(left_y) / len(y) * self._gini(left_y) +
                                       len(right_y) / len(y) * self._gini(right_y))

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: List[List[float]], y: List[Any], depth: int = 0) -> Optional[Node]:
        if depth >= self.max_depth or len(set(y)) == 1 or len(X) < self.min_samples_split:
            return Node(value=max(set(y), key=y.count))

        feature, threshold, gain = self._best_split(X, y)

        if feature == -1 or feature >= self.n_features:  # Stop if we hit a canary or couldn't find a valid split
            return Node(value=max(set(y), key=y.count))

        left_X, left_y, right_X, right_y = self._split(X, y, feature, threshold)

        left = self._build_tree(left_X, left_y, depth + 1)
        right = self._build_tree(right_X, right_y, depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X: List[List[float]], y: List[Any]) -> 'CanaryTree':
        self.n_features = len(X[0])
        X_with_canaries = self._add_canaries(X)
        self.tree = self._build_tree(X_with_canaries, y)
        return self

    def _predict_single(self, x: List[float], node: Optional[Node] = None) -> Any:
        if node is None:
            node = self.tree
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X: List[List[float]]) -> List[Any]:
        X_with_canaries = self._add_canaries(X)
        return [self._predict_single(x) for x in X_with_canaries]

    def _print_tree(self, node: Optional[Node], depth: int = 0) -> None:
        if node is None:
            return
        if node.value is not None:
            print('  ' * depth + f"Predict {node.value}")
        else:
            feature_name = f"feature_{node.feature}" if node.feature < self.n_features else f"canary_{node.feature - self.n_features}"
            print('  ' * depth + f"{feature_name} <= {node.threshold:.2f}")
            self._print_tree(node.left, depth + 1)
            self._print_tree(node.right, depth + 1)

    def print_tree(self) -> None:
        self._print_tree(self.tree)

# Example usage
if __name__ == "__main__":
    # Generate a sample dataset
    random.seed(42)
    X = [[random.random() for _ in range(5)] for _ in range(100)]
    y = [random.choice([0, 1]) for _ in range(100)]

    # Create and fit the CanaryTree
    ct = CanaryTree(n_canaries=3, max_depth=5, min_samples_split=2, random_state=42)
    ct.fit(X, y)

    # Print the tree structure
    print("Tree structure:")
    ct.print_tree()

    # Make predictions
    X_test = [[random.random() for _ in range(5)] for _ in range(10)]
    predictions = ct.predict(X_test)
    print("\nPredictions for 10 test samples:")
    print(predictions)