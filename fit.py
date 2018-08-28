import logging
from datetime import datetime
import numpy as np
import pandas as pd
from prepare_data import HCDRDataLoader
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from models import DenseNN, GBC, ABC, DTC, MultiLSTMWithMetadata
from grid_search import grid_search
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix


def ensemble_fit_predict():
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

    # dense neural network
    logging.debug('Training dense neural network')
    model_args = {
        'hidden_dim': 64,
        'num_layers': 1,
        'l2_reg': 5e-5,
        'epochs': 20,
        'batch_size': 1024,
        'dropout': 0.4
    }

    dense_nn = DenseNN(data_train_os[0].shape[1], **model_args)
    dense_nn.fit(data_train_os[0], target_train_os)

    train_results[:, 0] = dense_nn.predict(data_train[0]).squeeze()
    val_results[:, 0] = dense_nn.predict(data_val[0]).squeeze()

    # gradient boosting classifier
    logging.debug('Training gradient boosting classifier')
    gbc = GBC(
        max_depth=5,
        n_estimators=30,
        min_samples_split=0.01,
        learning_rate=0.3
    )
    gbc.fit(data_train_os[0], target_train_os)

    train_results[:, 1] = gbc.predict(data_train[0]).squeeze()
    val_results[:, 1] = gbc.predict(data_val[0]).squeeze()

    # adaboost classifier
    logging.debug('Training adaboost classifier')
    abc = ABC(
        n_estimators=20,
        learning_rate=1.0
    )
    abc.fit(data_train_os[0], target_train_os)

    train_results[:, 2] = abc.predict(data_train[0]).squeeze()
    val_results[:, 2] = abc.predict(data_val[0]).squeeze()

    # multi lstm network with metadata
    logging.debug('Training multi lstm nn')
    model_args = {
        'epochs': 35,
        'batch_size': 8192,
        'lstm_gpu': True,
        'sequence_dense_layers': 0,
        'sequence_dense_width': 8,
        'sequence_l2_reg': 0,
        'meta_dense_layers': 3,
        'meta_dense_width': 64,
        'meta_l2_reg': 1e-5,
        'meta_dropout': 0.2,
        'comb_dense_layers': 3,
        'comb_dense_width': 64,
        'comb_l2_reg': 1e-5,
        'comb_dropout': 0.1,
        'lstm_units': 6,
        'lstm_l2_reg': 1e-5
    }

    input_shape = loader.get_input_shape()
    lstm_nn = MultiLSTMWithMetadata(input_shape, **model_args)
    lstm_nn.fit(data_train_os, target_train_os)

    train_results[:, 3] = lstm_nn.predict(data_train).squeeze()
    val_results[:, 3] = lstm_nn.predict(data_val).squeeze()

    # lr = LogisticRegression(class_weight='balanced', C=0.1, fit_intercept=False)
    # lr.fit(train_results, target_train.values)
    # y = lr.predict(val_results)

    weights = np.array([1.7, 3.0, 0.75, 0.35])
    y = logistic.cdf(np.dot(val_results - 0.5, weights))

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
        'pos_tmax': 60,
        'install_mos_max': 60
    }

    loader = HCDRDataLoader(**loader_args)
    app_ix = loader.get_index()
    scores = np.zeros(4)
    coefs = np.zeros((4, 4))
    scores_path = 'data/results/ensemble_scores_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())

    kf = KFold(n_splits=4, shuffle=True)
    for j, fold_indexes in enumerate(kf.split(app_ix)):
        # load training and test data
        data_train, target_train, data_val, target_val = loader.load_train_val(fold_indexes[0], fold_indexes[1])
        input_shape = loader.get_input_shape()

        # oversample troubled loans to make up for imbalance
        ros = RandomOverSampler()
        os_index, target_train_os = ros.fit_sample(np.arange(data_train[0].shape[0]).reshape(-1, 1), target_train)
        data_train_os = [data_train_part[os_index.squeeze()] for data_train_part in data_train]

        # use predict on out of sample data and store results for each model
        num_models = 4
        train_samples = target_train.shape[0]
        val_samples = target_val.shape[0]
        train_results = np.empty((train_samples, num_models))
        val_results = np.empty((val_samples, num_models))

        # dense neural network
        logging.debug('Training dense neural network')
        model_args = {
            'hidden_dim': 64,
            'num_layers': 1,
            'l2_reg': 5e-5,
            'epochs': 20,
            'batch_size': 1024,
            'dropout': 0.4
        }

        dense_nn = DenseNN(data_train_os[0].shape[1], **model_args)
        dense_nn.fit(data_train_os[0], target_train_os)

        train_results[:, 0] = dense_nn.predict(data_train[0]).squeeze()
        val_results[:, 0] = dense_nn.predict(data_val[0]).squeeze()

        # gradient boosting classifier
        logging.debug('Training gradient boosting classifier')
        gbc = GBC(
            max_depth=5,
            n_estimators=30,
            min_samples_split=0.01,
            learning_rate=0.3
        )
        gbc.fit(data_train_os[0], target_train_os)

        train_results[:, 1] = gbc.predict(data_train[0]).squeeze()
        val_results[:, 1] = gbc.predict(data_val[0]).squeeze()

        # adaboost classifier
        logging.debug('Training adaboost classifier')
        abc = ABC(
            n_estimators=20,
            learning_rate=1.0
        )
        abc.fit(data_train_os[0], target_train_os)

        train_results[:, 2] = abc.predict(data_train[0]).squeeze()
        val_results[:, 2] = abc.predict(data_val[0]).squeeze()

        model_args = {
            'epochs': 35,
            'batch_size': 8192,
            'lstm_gpu': True,
            'sequence_dense_layers': 0,
            'sequence_dense_width': 8,
            'sequence_l2_reg': 0,
            'meta_dense_layers': 3,
            'meta_dense_width': 64,
            'meta_l2_reg': 1e-5,
            'meta_dropout': 0.2,
            'comb_dense_layers': 3,
            'comb_dense_width': 64,
            'comb_l2_reg': 1e-5,
            'comb_dropout': 0.1,
            'lstm_units': 6,
            'lstm_l2_reg': 1e-5
        }

        lstm_nn = MultiLSTMWithMetadata(input_shape, **model_args)

        lstm_nn.fit(data_train_os, target_train_os, validation_data=(data_val, target_val))

        train_results[:, 3] = lstm_nn.predict(data_train).squeeze()
        val_results[:, 3] = lstm_nn.predict(data_val).squeeze()

        lr = LogisticRegression(class_weight='balanced', C=0.1, fit_intercept=False)
        lr.fit(train_results - 0.5, target_train.values)

        y = lr.predict(val_results - 0.5)

        # use logistic regression built-in scoring method to score out of sample accuracy
        scores[j] = lr.score(val_results - 0.5, target_val.values)
        # cm = confusion_matrix(target_val, y)

        vlr = LogisticRegression(class_weight='balanced', C=0.1, fit_intercept=False)
        vlr.fit(val_results - 0.5, target_val.values)
        coefs[j, :] = vlr.coef_.squeeze()
        coefs_path = 'data/results/ensemble_coef_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())
        coefs_df = pd.DataFrame(coefs)
        coefs_df.to_csv(coefs_path)

        results_path = 'data/results/results_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())
        results = pd.DataFrame({'TARGET': target_val.values, 'PREDICTION': y})
        results.to_csv(results_path)

        train_cls = pd.DataFrame(np.concatenate([train_results, target_train.values.reshape(-1, 1)], axis=1))
        train_cls.to_csv('data/results/train_results_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now()))

        raw_results = pd.DataFrame(np.concatenate([val_results, y.reshape(-1, 1), target_val.values.reshape(-1, 1)], axis=1))
        raw_results.to_csv('data/results/val_results_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now()))

        pd.DataFrame({'score': scores}).to_csv(scores_path)


# TODO: add method for reading validation results from a file
def ensemble_val_from_file(filename):
    raw_results = pd.read_csv(filename)
    raw_results.columns = [
        'dense_nn',
        'gbc',
        'abc',
        'lstm',
        'target'
    ]

    lr_params = {
        'C': 1,
        'class_weight': 'balanced',
        'fit_intercept': False
    }

    lr = LogisticRegression(**lr_params)
    lr.fit(raw_results[['dense_nn', 'gbc', 'abc', 'lstm']], raw_results['target'])


def gbc_grid_search():
    loader_args = {
        'load_time_series': False
    }
    model_args = {
        'verbose': 0
    }

    grid_search(GBC, HCDRDataLoader,
                hp_file='gbc_grid_params.txt',
                loader_args=loader_args, model_args=model_args,
                random_oversample=True)


def dense_nn_grid_search():
    loader_args = {
        'load_time_series': False
    }
    model_args = {
        'verbose': 1
    }

    grid_search(DenseNN, HCDRDataLoader,
                hp_file='dense_grid_params.txt',
                loader_args=loader_args, model_args=model_args,
                random_oversample=True)


def abc_grid_search():
    loader_args = {
        'load_time_series': False
    }
    model_args = {
    }

    grid_search(ABC, HCDRDataLoader,
                hp_file='abc_grid_params.txt',
                loader_args=loader_args, model_args=model_args,
                random_oversample=True)


def svc_grid_search():
    loader_args = {
        'load_time_series': False
    }
    model_args = {
        'class_weight': 'balanced',
        'verbose': 1,
        'dual': False,
        'tol': 1e-4,
        'max_iter': 1000
    }

    loader = HCDRDataLoader(**loader_args)
    app_ix = loader.get_index()

    kf = KFold(n_splits=4, shuffle=True)
    for j, fold_indexes in enumerate(kf.split(app_ix)):
        pass

    # load training and test data
    data_train, target_train, data_val, target_val = loader.load_train_val(fold_indexes[0], fold_indexes[1])

    svc = LinearSVC(**model_args)
    svc.fit(data_train, target_train)

    logging.debug(svc.score(data_val, target_val))
    logging.debug(confusion_matrix(target_val, svc.predict(data_val)))


def dtc_grid_search():
    loader_args = {
        'load_time_series': False
    }
    model_args = {
        'class_weight': 'balanced',
    }

    grid_search(DTC, HCDRDataLoader,
                hp_file='dtc_grid_params.txt',
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
                hp_file='lstm_grid_params.txt',
                loader_args=loader_args, model_args=model_args,
                random_oversample=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    file = 'data/results/raw_results20180819_044627.csv'
    # gbc_grid_search()
    # ensemble_val_from_file(file)
    # ensemble_fit_val()
    ensemble_fit_predict()
