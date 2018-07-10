import logging
import numpy as np
from prepare_data import HCDRLoader
from sklearn.model_selection import KFold
from models import LinearNN, GBC, ABC, LSTMWithMetadata


def compare_models():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    loader = HCDRLoader()

    # load index values from main table
    app_ix = loader.applications_train_index()

    # fit model using k-fold verification
    kf = KFold(n_splits=2, shuffle=True)
    for fold_indexes in kf.split(app_ix):
        # load training and validation summary data
        data_train, target_train, data_val, target_val = loader.load_train_val(fold_indexes[0], fold_indexes[1])

        # oversample troubled loans to make up for imbalance
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler()
        data_train_os_index, target_train_os = ros.fit_sample(np.arange(data_train.shape[0]).reshape(-1, 1),
                                                              target_train)
        data_train_os = data_train[data_train_os_index.squeeze()]

        # TODO: create new dated log file for each run, store model name and oos accuracies for each fold

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

        # load time series data
        sequence_length = 25
        cc_data_train = loader.read_credit_card_balance(app_ix.values[fold_indexes[0]], t_max=sequence_length)
        cc_data_val = loader.read_credit_card_balance(app_ix.values[fold_indexes[1]], t_max=sequence_length)
        cc_data_train_os = cc_data_train[data_train_os_index.squeeze()]

        # determine input shape
        sequence_features = np.int(cc_data_train.shape[1] / sequence_length)
        meta_features = np.int(data_train.shape[1])

        # combined credit card balance lstm and dense metadata neural network
        lstm_nn = LSTMWithMetadata(sequence_length, sequence_features, meta_features)
        lstm_nn.fit(cc_data_train_os, data_train_os, target_train_os,
                    cc_data_val, data_val, target_val,
                    num_epochs=2)

        # TODO: grid search on parameters for credit card lstm network

        # TODO: lstm model for other time series data

        # TODO: new model performing 1D convolution on time series data


def predict_test():
    pass


if __name__ == "__main__":
    compare_models()
