import argparse
import logging
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
from prepare_data import HCDRLoader, HCDRDataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from models import LinearNN, GBC, ABC, LSTMWithMetadata, MultiLSTMWithMetadata


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
        ros = RandomOverSampler()
        data_train_os_index, target_train_os = ros.fit_sample(np.arange(data_train.shape[0]).reshape(-1, 1),
                                                              target_train)
        data_train_os = data_train[data_train_os_index.squeeze()]

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
                    cc_data_val, data_val, target_val)

        # TODO: lstm model for other time series data

        # TODO: new model performing 1D convolution on time series data


def lstm_grid_search():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    loader = HCDRLoader()

    # load index values from main table
    app_ix = loader.applications_train_index()
    sequence_length = 25

    hyperparameters = {
        'sequence_dense_layers': [0],
        'sequence_dense_width': [4, 8],
        'sequence_l2_reg': [0],
        'meta_dense_layers': [1, 2],
        'meta_dense_width': [64],
        'meta_l2_reg': [1e-5],
        'comb_dense_layers': [0, 1],
        'comb_dense_width': [32, 64],
        'lstm_units': [32, 64],
        'lstm_l2_reg': [1e-5, 1e-6],
        'batch_size': [2048],
        'lstm_gpu': [True]
    }

    keys, values = zip(*hyperparameters.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    exp_df = pd.DataFrame(experiments)

    cm = np.zeros((len(experiments), 2, 2), dtype=int)
    cm_df_cols = ['CM True Neg', 'CM False Pos', 'CM False Neg', 'CM True Pos']
    results_path = 'results/lstm_results_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())

    # fit model using k-fold verification
    k = 4
    kf = KFold(n_splits=k, shuffle=True)

    for j, fold_indexes in enumerate(kf.split(app_ix)):
        data_train, target_train, data_val, target_val = loader.load_train_val(fold_indexes[0], fold_indexes[1])
        cc_data_train = loader.read_credit_card_balance(app_ix.values[fold_indexes[0]], t_max=sequence_length)
        cc_data_val = loader.read_credit_card_balance(app_ix.values[fold_indexes[1]], t_max=sequence_length)

        sequence_features = np.int(cc_data_train.shape[1] / sequence_length)
        meta_features = np.int(data_train.shape[1])

        ros = RandomOverSampler()
        data_train_os_index, target_train_os = ros.fit_sample(np.arange(data_train.shape[0]).reshape(-1, 1), target_train)
        data_train_os = data_train[data_train_os_index.squeeze()]
        cc_data_train_os = cc_data_train[data_train_os_index.squeeze()]

        logging.debug('Fold {} of {}'.format(j + 1, k))
        
        for i, experiment in enumerate(experiments):
            logging.debug(experiment)
            lstm_nn = LSTMWithMetadata(sequence_length, sequence_features, meta_features,
                                       **experiment)
            history = lstm_nn.fit(cc_data_train_os, data_train_os, target_train_os,
                                  cc_data_val, data_val, target_val)
            predict_val = lstm_nn.predict(cc_data_val, data_val)

            cm_fold = confusion_matrix(target_val, predict_val[0].round())
            cm[i, :, :] = cm[i, :, :] + cm_fold
            cm_df = pd.DataFrame(cm.reshape((cm.shape[0], 4)), columns=cm_df_cols)
            results_df = exp_df.join(cm_df)
            results_df.to_csv(results_path)


def grid_search(model_class, data_loader, 
                loader_args=None, model_args=None, hp_file=NotImplemented, 
                folds=4, random_oversample=False):
    """
    Perform grid search over hyperparameters over given models.
    :param model_class:
    :param data_loader:
    :param loader_args:
    :param model_args:
    :param hp_file:
    :param folds:
    :param random_oversample:
    :return:
    """
    loader = data_loader(**loader_args)

    # load index values from main table
    data_ix = loader.get_index()
    
    # TODO: load hyperparameters from file
    hyperparameters = {
        'sequence_dense_layers': [0, 1],
        'sequence_dense_width': [4, 8],
        'sequence_l2_reg': [0],
        'meta_dense_layers': [0, 1],
        'meta_dense_width': [32, 64],
        'meta_l2_reg': [0, 1e-5],
        'comb_dense_layers': [0, 1],
        'lstm_units': [4, 8]
    }

    # create a list of dicts with hyperparameters for each experiment to run
    keys, values = zip(*hyperparameters.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    exp_df = pd.DataFrame(experiments)

    # prepare for storing confusion matrix and accuracy results to file
    # TODO: store oos accuracy per epoch
    cm = np.zeros((len(experiments), 2, 2), dtype=int)
    cm_df_cols = ['CM True Neg', 'CM False Pos', 'CM False Neg', 'CM True Pos']
    results_path = 'results/gridsearch_results_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())

    # fit model using k-fold verification
    kf = KFold(n_splits=folds, shuffle=True)

    for j, fold_indexes in enumerate(kf.split(data_ix)):
        # load train and validation data
        data_train, target_train, data_val, target_val = loader.load_train_val(fold_indexes[0], fold_indexes[1])

        # determine shape of input arrays
        input_shape = loader.get_input_shape()

        # oversample to correct for class imbalance
        if random_oversample:
            ros = RandomOverSampler()
            os_index, target_train_temp = ros.fit_sample(np.arange(data_train[0].shape[0]).reshape(-1, 1), target_train)
            data_train = [data_train_part[os_index.squeeze()] for data_train_part in data_train]
            target_train = target_train_temp

        logging.debug('Fold {} of {}'.format(j + 1, folds))
        
        for i, experiment in enumerate(experiments):
            logging.debug(experiment)
            model = model_class(input_shape, **model_args, **experiment)
            # TODO: track oos accuracy per epoch
            history = model.fit(data_train, target_train, data_val, target_val)
            predict_val = model.predict(data_val)

            # TODO: simple models may only have one item in predict_val
            cm[i, :, :] = cm[i, :, :] + confusion_matrix(target_val, predict_val[0].round())
            cm_df = pd.DataFrame(cm.reshape((cm.shape[0], 4)), columns=cm_df_cols)
            results_df = exp_df.join(cm_df)
            # TODO: also store run time
            results_df.to_csv(results_path)


def predict_test():
    pass


def new_main():
    loader_args = {
        'cc_tmax': 50
    }

    model_args = {
        'epochs': 25
    }

    grid_search(MultiLSTMWithMetadata, HCDRDataLoader,
                loader_args=loader_args, model_args=model_args,
                random_oversample=True)


if __name__ == "__main__":
    lstm_grid_search()
