import logging
from prepare_data import HCDALoader
from sklearn.model_selection import KFold
from models import LinearNN, GBC, ABC


def compare_models():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    loader = HCDALoader()

    # load index values from main table
    app_ix = loader.applications_train_index()

    # fit model using k-fold verification
    kf = KFold(n_splits=2, shuffle=True)
    for fold_indexes in kf.split(app_ix):
        # load training and validation data
        data_train, target_train, data_val, target_val = loader.load_train_val(fold_indexes[0], fold_indexes[1])

        # oversample troubled loans to make up for imbalance
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler()
        data_train_os, target_train_os = ros.fit_sample(data_train, target_train)

        # train on linear neural network
        linear_nn = LinearNN(data_train_os.shape[1])
        linear_nn.fit(data_train_os, target_train_os, data_val, target_val)
        # TODO: use predict on out of sample data and store results for each model

        # gradient boosting classifier
        gbc = GBC()
        gbc.fit(data_train_os, target_train_os)

        # adaboost classifier
        abc = ABC()
        abc.fit(data_train_os, target_train_os)

        # TODO: new model performing 1D convolution on bureau and loan balances
        # TODO: model using LSTM to analyze bureau and loan balances


def predict_test():
    pass


if __name__ == "__main__":
    compare_models()
