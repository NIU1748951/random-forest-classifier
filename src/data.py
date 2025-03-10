import numpy as np

class Data:
    __slots__ = "_data, _target"

    def __init__(self, data, target):
        self._data, self._target = data, target


class DataSet:
    def __init__(self, table):
        self._table = table

    @classmethod
    def random_sampling(cls, table, ratio_samples):
        n_samples = int((len(table)*ratio_samples)) #nombre de mostres
        sampled_data = np.random.choice(table, size = n_samples, replace = True) #bootstrapping

        return cls(sampled_data)


    @property
    def data(self):
        return self._table
    
    @staticmethod
    def calc_gini():
        pass
