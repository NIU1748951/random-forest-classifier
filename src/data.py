import numpy as np

class DataSet:
    def __init__(self, X, y):
        self._X = np.array(X)
        self._y = np.array(y)

    @classmethod
    def random_sampling(cls, table, ratio_samples):
        n_samples = int((len(table)*ratio_samples)) #nombre de mostres
        sampled_data = np.random.choice(table, size = n_samples, replace = True) #bootstrapping

        return cls(sampled_data)
    
    @property
    def y(self):
        return self._y

    @property
    def data(self):
        return self._table

    @property
    def num_samples(self):
        return self._X.shape[0]

    @property
    def num_features(self):
        if self.num_samples > 0:
            return self._X.shape[1]
        else:
            return 0

    @property
    def X(self):
        return self.X

    def most_frequent_label(self):
        unique_labels, counts = np.unique(self._y, return_counts=True) #most common label
        return unique_labels[np.argmax(counts)]

    def split(self, idx_feature, value):
        # we split the data in two sub-sets: the ones that have the feature value in "idx" less than "val"
        # and the ones that have the same value or more
        left_mask = self._X[:, idx_feature] < value
        right_mask = ~left_mask

        #we filter the data by their masks
        left_data = self._X[left_mask]
        right_data = self._X[right_mask]
        left_labels = self._y[left_mask]
        right_labels = self._y[right_mask]

        #returning two datasets
        return DataSet(left_data, left_labels), DataSet(right_data, right_labels)
