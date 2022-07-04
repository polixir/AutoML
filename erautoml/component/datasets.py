import numpy as np


class AutomlDataset:

    def __init__(self, features, labels, test_features=None, test_labels=None):

        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_labels = test_labels

        return


if __name__ == "__main__":
    print('Datasets.py')