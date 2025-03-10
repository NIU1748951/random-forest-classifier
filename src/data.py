class Data:
    __slots__ = "_atributes, _result"

    def __init__(self, data, target):
        self._data, self._target = data, target




class dataTable:
    def __inti__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data
    
    @staticmethod
    def calc_gini():
        pass

    # ais∫