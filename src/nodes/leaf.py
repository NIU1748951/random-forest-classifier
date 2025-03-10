from node import Node


class Leaf(Node):
    __slots__ = "_label"

    def __init__(self, label):
        self._label = label

    def predict(self, x):
        pass

    def calc_gini(self):
        pass
