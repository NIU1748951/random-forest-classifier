import numpy as np
import sklearn.datasets
from random_forest import RandomForestClassifier

from nodes.node import Node
from nodes.leaf import Leaf
from nodes.parent import Parent

from logger_config import get_logger

logger = get_logger(__name__)

iris = sklearn.datasets.load_iris() 

logger.info(iris.DESCR)

X, y = iris.data, iris.target 
ratio_train, ratio_test = 0.7, 0.3  

num_samples, num_features = X.shape  

idx = np.random.permutation(range(num_samples))  

num_samples_train = int(num_samples * ratio_train)  
num_samples_test = int(num_samples * ratio_test)   

idx_train = idx[:num_samples_train]  
idx_test = idx[num_samples_train:num_samples_train + num_samples_test]  


X_train, y_train = X[idx_train], y[idx_train]  
X_test, y_test = X[idx_test], y[idx_test]   


# Hyperparameters
max_depth = 10  
min_size_split = 5  
ratio_samples = 0.7  
num_trees = 10  
num_random_features = int(np.sqrt(num_features))  
# number of features to consider at each node when looking for the best split
criterion = 'gini' 

rf = RandomForestClassifier(
    max_depth, 
    min_size_split, 
    ratio_samples, 
    num_trees, 
    num_random_features, 
    criterion
)


if __name__ == "__main__":
    logger.info("Starting the Random Forest Classifier")
    # train = make the decision trees
    
    try:
        rf.fit(X_train, y_train)

        # classification
        ypred = rf.predict(X_test)

        # compute accuracy
        num_samples_test = len(y_test)
        num_correct_predictions = np.sum(ypred == y_test)
        accuracy = num_correct_predictions / float(num_samples_test)

        if (100 * np.round(accuracy, decimals=2)) < 50.0:
            logger.warning('accuracy {} %'.format(100 * np.round(accuracy, decimals=2)))
        else:
            logger.info('accuracy {} %'.format(100 * np.round(accuracy, decimals=2)))
    except Exception as e:
        logger.error("An error occurred: %s", str(e))