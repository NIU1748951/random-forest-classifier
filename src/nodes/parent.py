from node import Node


class Parent(Node):
    __slots__ = "_feature_index, _threshold"

    def __init__(self, feature_index, threshold):
        self._feature_index = feature_index
        self._threshold = threshold

    def predict(self, x):
        pass
