from typing import List
from data import DataSet
from nodes.leaf import Leaf
from nodes.parent import Parent
import numpy as np


class RandomForestClassifier:
    def __init__(
        self,
        *,
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
        self._decison_trees = NotImplemented

    def fit(self, X, y) -> None:
        # a pair (X, y) is a dataset
        dataset = DataSet(X, y)
        self.make_decision_trees(dataset)

    def make_decision_trees(self, dataset):
        self._decison_trees = []
        for i in range(self._num_trees):
            subset = dataset.random_sampling(
                self._ratio_samples
            )  # IMPLEMENTAR EN DATASET
            tree = self.make_node(subset, 1)
            self._decison_trees.append(tree)

    def make_node(self, dataset, depth):
        if (
            depth == self._max_depth
            or dataset.num_samples <= self._min_size
            or len(np.unique(dataset.y)) == 1
        ):
            node = self._make_leaf(dataset)
        else:
            node = self._make_leaf(dataset)

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
            np.Inf,
            np.Inf,
            np.Inf,
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
        # IMPLEMENT GINI
        cost = None

        return cost

    @property
    def _gini(self, dataset):
        pass

    def predict(self, X) -> List[int]:
        pass
