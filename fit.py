import logging
from datetime import datetime
import numpy as np
import pandas as pd
from prepare_data import HCDRDataLoader
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from models import LinearNN, GBC, ABC, MultiLSTMWithMetadata
from grid_search import grid_search


def ensemble_fit_predict():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    loader_args = {
        'cc_tmax': 60,
        'bureau_tmax': 60,
        'pos_tmax': 60
    }

    loader = HCDRDataLoader(**loader_args)

    # load training and test data
    data_train, target_train = loader.load_train_data()
    data_val = loader.load_test_data()

    # oversample troubled loans to make up for imbalance
    ros = RandomOverSampler()
    os_index, target_train_os = ros.fit_sample(np.arange(data_train[0].shape[0]).reshape(-1, 1), target_train)
    data_train_os = [data_train_part[os_index.squeeze()] for data_train_part in data_train]

    # use predict on out of sample data and store results for each model
    num_models = 4
    train_samples = data_train[0].shape[0]
    test_samples = data_val[0].shape[0]
    train_results = np.empty((train_samples, num_models))
    val_results = np.empty((test_samples, num_models))

    # train on linear neural network
    # TODO: add dropout and regularization
    logging.debug('Training linear nn')
    linear_nn = LinearNN(data_train_os[0].shape[1], epochs=25)
    linear_nn.fit(data_train_os[0], target_train_os)

    train_results[:, 0] = linear_nn.predict(data_train[0]).squeeze()
    val_results[:, 0] = linear_nn.predict(data_val[0]).squeeze()

    # gradient boosting classifier
    logging.debug('Training gradient boosting classifier')
    gbc = GBC(
        max_depth=7,
        n_estimators=20
    )
    gbc.fit(data_train_os[0], target_train_os)

    train_results[:, 1] = gbc.predict(data_train[0]).squeeze()
    val_results[:, 1] = gbc.predict(data_val[0]).squeeze()

    # adaboost classifier
    # TODO: grid search on adaboost classifier
    logging.debug('Training adaboost classifier')
    abc = ABC()
    abc.fit(data_train_os[0], target_train_os)

    train_results[:, 2] = abc.predict(data_train[0]).squeeze()
    val_results[:, 2] = abc.predict(data_val[0]).squeeze()

    model_args = {
        'epochs': 50,
        'batch_size': 512,
        'lstm_gpu': False,
        'sequence_dense_layers': 0,
        'sequence_dense_width': 8,
        'sequence_l2_reg': 0,
        'meta_dense_layers': 1,
        'meta_dense_width': 64,
        'meta_l2_reg': 1e-5,
        'meta_dropout': 0.2,
        'comb_dense_layers': 3,
        'comb_dense_width': 64,
        'comb_l2_reg': 1e-6,
        'comb_dropout': 0.2,
        'lstm_units': 8,
        'lstm_l2_reg': 1e-7
    }

    input_shape = loader.get_input_shape()
    lstm_nn = MultiLSTMWithMetadata(input_shape, **model_args)

    logging.debug('Training multi lstm nn')
    lstm_nn.fit(data_train_os, target_train_os)

    train_results[:, 3] = lstm_nn.predict(data_train).squeeze()
    val_results[:, 3] = lstm_nn.predict(data_val).squeeze()

    lr = LogisticRegression(class_weight='balanced', C=0.1)
    lr.fit(train_results, target_train.values)

    y = lr.predict(val_results)

    results_path = 'data/results/results_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())
    results = pd.DataFrame({'SK_ID_CURR': loader.get_test_index().values, 'TARGET': y}).set_index('SK_ID_CURR')
    results.to_csv(results_path)

    raw_results = pd.DataFrame(np.concatenate([val_results, y.reshape(-1, 1)], axis=1))
    raw_results.to_csv('data/results/raw_results{:%Y%m%d_%H%M%S}.csv'.format(datetime.now()))


def ensemble_fit_val():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    loader_args = {
        'cc_tmax': 60,
        'bureau_tmax': 60,
        'pos_tmax': 60
    }

    loader = HCDRDataLoader(**loader_args)
    app_ix = loader.get_index()

    kf = KFold(n_splits=4, shuffle=True)
    for fold_indexes in kf.split(app_ix):
        # load training and test data
        data_train_ts, target_train_ts, data_val_ts, target_val_ts = loader.load_train_val(fold_indexes[0], fold_indexes[1])
        input_shape = loader.get_input_shape()

        # oversample troubled loans to make up for imbalance
        ros = RandomOverSampler()
        os_index, target_train_os = ros.fit_sample(np.arange(data_train_ts[0].shape[0]).reshape(-1, 1), target_train_ts)
        data_train_ts_os = [data_train_part[os_index.squeeze()] for data_train_part in data_train_ts]
        target_train_ts_os = target_train_ts.values[os_index.squeeze()]

        # data_train_os = data_train[os_index.squeeze()]
        # use predict on out of sample data and store results for each model
        num_models = 4
        train_samples = target_train_ts.shape[0]
        val_samples = target_val_ts.shape[0]
        train_results = np.empty((train_samples, num_models))
        val_results = np.empty((val_samples, num_models))

        # train on linear neural network
        linear_nn = LinearNN(data_train_ts_os[0].shape[1], epochs=25)
        linear_nn.fit(data_train_ts_os[0], target_train_ts_os, validation_data=(data_val_ts[0], target_val_ts))

        train_results[:, 0] = linear_nn.predict(data_train_ts[0]).squeeze()
        val_results[:, 0] = linear_nn.predict(data_val_ts[0]).squeeze()

        # gradient boosting classifier
        gbc = GBC()
        gbc.fit(data_train_ts_os[0], target_train_os)

        train_results[:, 1] = gbc.predict(data_train_ts[0]).squeeze()
        val_results[:, 1] = gbc.predict(data_val_ts[0]).squeeze()

        # adaboost classifier
        abc = ABC()
        abc.fit(data_train_ts_os[0], target_train_os)

        train_results[:, 2] = abc.predict(data_train_ts[0]).squeeze()
        val_results[:, 2] = abc.predict(data_val_ts[0]).squeeze()

        model_args = {
            'epochs': 1,
            'batch_size': 256,
            'lstm_gpu': False,
            'sequence_dense_layers': 0,
            'sequence_dense_width': 8,
            'sequence_l2_reg': 0,
            'meta_dense_layers': 1,
            'meta_dense_width': 64,
            'meta_l2_reg': 1e-5,
            'meta_dropout': 0.2,
            'comb_dense_layers': 3,
            'comb_dense_width': 64,
            'comb_l2_reg': 1e-6,
            'comb_dropout': 0.2,
            'lstm_units': 8,
            'lstm_l2_reg': 1e-7
        }

        lstm_nn = MultiLSTMWithMetadata(input_shape, **model_args)

        lstm_nn.fit(data_train_ts_os, target_train_os, validation_data=(data_val_ts, target_val_ts))

        train_results[:, 3] = lstm_nn.predict(data_train_ts).squeeze()
        val_results[:, 3] = lstm_nn.predict(data_val_ts).squeeze()

        lr = LogisticRegression(class_weight='balanced')
        lr.fit(train_results, target_train_ts.values)

        y = lr.predict(val_results)

        results = pd.DataFrame(np.concatenate([val_results, y.reshape(-1, 1), target_val_ts.values.reshape(-1, 1)], axis=1))
        results.to_csv('data/results.csv')

        results_path = 'data/results/results_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())
        results = pd.DataFrame({'TARGET': target_val_ts.values, 'PREDICTION': y})
        results.to_csv(results_path)

        raw_results = pd.DataFrame(np.concatenate([val_results, y.reshape(-1, 1), target_val_ts.values.reshape(-1, 1)], axis=1))
        raw_results.to_csv('data/results/raw_results{:%Y%m%d_%H%M%S}.csv'.format(datetime.now()))


def gbc_grid_search():
    loader_args = {
        'load_time_series': False
    }
    model_args = {
        'verbose': 1
    }

    grid_search(GBC, HCDRDataLoader,
                hp_file='gbc_grid_params.txt',
                loader_args=loader_args, model_args=model_args,
                random_oversample=True)


def abc_grid_search():
    loader_args = {
        'load_time_series': False
    }
    model_args = {
        'verbose': 1
    }

    grid_search(ABC, HCDRDataLoader,
                hp_file='abc_grid_params.txt',
                loader_args=loader_args, model_args=model_args,
                random_oversample=True)


def multi_lstm_grid_search():
    loader_args = {
        'cc_tmax': 60,
        'bureau_tmax': 60,
        'pos_tmax': 60
    }

    model_args = {
        'batch_size': 512,
        'lstm_gpu': True
    }

    grid_search(MultiLSTMWithMetadata, HCDRDataLoader,
                hp_file='grid_search_params.txt',
                loader_args=loader_args, model_args=model_args,
                random_oversample=True)


if __name__ == "__main__":
    abc_grid_search()
