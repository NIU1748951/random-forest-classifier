import numpy as np

class DataSet:
    def __init__(self, X, y):
        self._X = np.array(X)
        self._y = np.array(y)
        self.ratio_samples = 0
        self.num_samples = 0

    def random_sampling(self, ratio_samples):
        self._ratio_samples = ratio_samples
        self.num_samples = int(len(self._X) * ratio_samples)  # Calculate the number of samples to draw
        sampled_data = np.random.choice(a=self._X.flatten(), size=self.num_samples, replace=True)  # Bootstrapping
        print("Sampled table:", sampled_data)

        return DataSet(sampled_data, self._y)  # Return DataSet with sampled data
    
    @property
    def y(self):
        return self._y

    @property
    def data(self):
        return self._table

    @property
    def num_features(self):
        if self.num_samples > 0:
            return self._X.shape[1]
        else:
            return 0

    @property
    def X(self):
        return self._X

    def most_frequent_label(self):
        unique_labels, counts = np.unique(self._y, return_counts=True)  # Most common label
        return unique_labels[np.argmax(counts)]

    def split(self, idx_feature, value):
        left_mask = self._X[:, idx_feature] < value
        right_mask = ~left_mask

        left_data = self._X[left_mask]
        right_data = self._X[right_mask]
        left_labels = self._y[left_mask]
        right_labels = self._y[right_mask]

        return DataSet(left_data, left_labels), DataSet(right_data, right_labels)
