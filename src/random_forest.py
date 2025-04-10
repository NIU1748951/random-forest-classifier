from typing import List
from data import DataSet
from nodes.leaf import Leaf
from nodes.parent import Parent
import numpy as np
from logger_config import get_logger

logger = get_logger(__name__)

class RandomForestClassifier:
    def __init__(
        self,
        num_trees=100,
        min_size=1,
        max_depth=10,
        ratio_samples=0.8,
        num_random_features=0.33,
        criterion="gini",
    ):
        self._num_trees = num_trees
        self._min_size = min_size
        self._max_depth = max_depth
        self._ratio_samples = ratio_samples
        self._num_random_features = num_random_features
        self._criterion = criterion
        self._decison_trees = []
        logger.info("RandomForestClassifier inititalized with %d trees", num_trees)

    def fit(self, X, y):
        logger.info("Starting fit process with %d samples", len(X))
        dataset = DataSet(X, y, self._ratio_samples, len(X))
        self.make_decision_trees(dataset)
        logger.info("Fit process completed")

    def predict(self, X):
        logger.info("Starting prediction for %d samples", len(X))
        ypred = []

        for x in X:
            predictions = [root.predict(x) for root in self._decison_trees]

            #majority voting
            ypred.append(max(set(predictions), key = predictions.count))
        
        logger.info("Predictions completed")
        return np.array(ypred)

    def make_decision_trees(self, dataset: DataSet):
        self._decison_trees = []
        for i in range(self._num_trees):
            subset = dataset.random_sampling(
                self._ratio_samples
            ) 
            
            tree = self._make_node(subset, 1)
            self._decison_trees.append(tree)
            logger.info("Created a decision tree with %d samples", subset.num_samples)

    def _make_node(self, dataset, depth):
        if (
            depth == self._max_depth
            or dataset.num_samples <= self._min_size
            or len(np.unique(dataset.y)) == 1
        ):
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)

        return node

    def _make_leaf(self, dataset):
        label = NotImplemented  # la mes freqüent
        return Leaf(dataset.most_frequent_label())

    def _make_parent_or_leaf(self, dataset, depth):
        idx_features = np.random.choice(
            range(dataset.num_features), self._num_random_features, replace=False
        )

        best_feature_idx, best_threshold, minimum_cost, best_split = self._best_split(
            idx_features, dataset
        )

        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            return self._make_leaf(dataset)
        else:
            node = Parent(best_feature_idx, best_threshold)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            return node

    def _best_split(self, idx_features, dataset):
        best_feature_index, best_threshold, minimum_cost, best_split = (
            np.inf,
            np.inf,
            np.inf,
            None,
        )
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            for val in values:
                left_dataset, right_dataset = dataset.split(idx, val)
                cost = self._CART_cost(left_dataset, right_dataset)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, best_split = (
                        idx,
                        val,
                        cost,
                        [left_dataset, right_dataset],
                    )
        return best_feature_index, best_threshold, minimum_cost, best_split

    def _CART_cost(self, left_dataset, right_dataset):
        """
        Improved split cost calculation using weighted gini impurity
        """
        n_left = left_dataset.num_samples
        n_right = right_dataset.num_samples
        n_total = n_left + n_right
        
        if n_total == 0:
            return float('inf')  # Worst possible cost if no samples
            
        if self._criterion == 'gini':
            gini_left = self._gini(left_dataset) if n_left > 0 else 1
            gini_right = self._gini(right_dataset) if n_right > 0 else 1
        else:
            logger.error(f"Unsupported criterion: {self._criterion}")
            exit(1)
            
        # Weighted gini impurity with regularization
        cost = (n_left/n_total)*gini_left + (n_right/n_total)*gini_right
        cost += 0.01*(1/(n_left+1) + 1/(n_right+1))  # Small penalty for imbalanced splits
        
        return cost

    
    def _gini(self, dataset):
        """
        More robust gini impurity calculation
        """
        if dataset.num_samples == 0:
            return 1.0  # Maximum impurity for empty node
            
        _, counts = np.unique(dataset.y, return_counts=True)
        probabilities = counts / dataset.num_samples
        gini = 1.0 - np.sum(probabilities**2)
        
        # Add small epsilon to avoid perfect splits
        return gini + 1e-8
