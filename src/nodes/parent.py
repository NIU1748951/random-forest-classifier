from node import Node


class Parent(Node):
    def __init__(self, feature_index: int, threshold:float):
        self._feature_index = feature_index
        self._threshold = threshold
        self.left_child: Node = None
        self.right_child: Node = None

    def predict(self, x: float):
        if x[self._feature_index] < self._threshold:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)
