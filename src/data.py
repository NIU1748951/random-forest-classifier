class Data:
    __slots__ = "_data, _target"

    def __init__(self, data, target):
        self._data, self._target = data, target


class DataSet:
    def __init__(self, table):
        self._table = table

    @property
    def y(self):
        pass

    @property
    def data(self):
        return self._table

    @property
    def num_samples(self):
        return len(self.data)

    @property
    def num_features(self):
        pass

    @property
    def X(self):
        pass

    def most_frequent_label(self):
        pass

    def ssplit(self, idx, val):
        pass
