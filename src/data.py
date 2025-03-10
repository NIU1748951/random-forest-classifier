class Data:
    __slots__ = "_data, _target"

    def __init__(self, data, target):
        self._data, self._target = data, target


class DataSet:
    def __inti__(self, table):
        self._table = table

    @property
    def data(self):
        return self._table
