

class BinaryClassifier:
    def __init__(self, input_dim):
        self._input_dim = input_dim

    def fit(self, data_train, target_train, vaidation_data=None):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def score(self, data):
        raise NotImplementedError
