from typing import List


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
        self.decison_trees = NotImplemented

    def fit(self, X, Y) -> None:
        pass

    def predict(self, X) -> List[int]:
        pass
