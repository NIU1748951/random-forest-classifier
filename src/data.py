import numpy as np
from logger_config import get_logger

logger = get_logger(__name__)
class DataSet:
    def __init__(self, X, y, ratio_samples, num_samples):
        self._X = np.array(X)
        self._y = np.array(y)
        self.ratio_samples = ratio_samples
        self.num_samples = num_samples

    def random_sampling(self, ratio_samples):
        self._ratio_samples = ratio_samples
        self.num_samples = int(len(self._X) * ratio_samples)  # Calculate the number of samples to draw
        sampled_indices = np.random.choice(a=self._X.shape[0], size=self.num_samples, replace=True)  # Bootstrapping
        
        sampled_X = self._X[sampled_indices]
        sampled_y = self._y[sampled_indices]

        logger.info("Ratio samples: %f", self._ratio_samples)
        logger.info("Sampled %d rows from the dataset", self.num_samples)

        return DataSet(sampled_X, sampled_y, self.num_samples, self._ratio_samples)
    
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

        return DataSet(left_data, left_labels, self.ratio_samples, self.num_samples), \
                DataSet(right_data, right_labels, self.ratio_samples, self.num_samples)
