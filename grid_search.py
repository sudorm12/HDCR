import logging
import ast
from datetime import datetime
import itertools
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def grid_search(model_class, data_loader, hp_file,
                loader_args=None, model_args=None,
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

    # load hyperparameters from file
    with open(hp_file, 'r') as f:
        hyperparameters = ast.literal_eval(f.read())

    # create a list of dicts with hyperparameters for each experiment to run
    keys, values = zip(*hyperparameters.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    exp_df = pd.DataFrame(experiments)

    # prepare for storing confusion matrix and accuracy results to file
    cm = np.zeros((len(experiments), 2, 2), dtype=int)
    cm_df_cols = ['CM True Neg', 'CM False Pos', 'CM False Neg', 'CM True Pos']
    results_path = 'data/results/gridsearch_results_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())

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
            os_index, target_train = ros.fit_sample(np.arange(len(fold_indexes[0])).reshape(-1, 1), target_train)
            if isinstance(data_train, list):
                data_train = [data_train_part[os_index.squeeze()] for data_train_part in data_train]
            else:
                data_train = data_train[os_index.squeeze()]

        logging.debug('Fold {} of {}'.format(j + 1, folds))

        for i, experiment in enumerate(experiments):
            logging.debug(experiment)
            model = model_class(input_shape, **model_args, **experiment)
            # TODO: track oos accuracy per epoch
            history = model.fit(data_train, target_train, validation_data=(data_val, target_val))
            predict_val = model.predict(data_val)

            cm[i, :, :] = cm[i, :, :] + confusion_matrix(target_val, predict_val.round())
            cm_df = pd.DataFrame(cm.reshape((cm.shape[0], 4)), columns=cm_df_cols)
            results_df = exp_df.join(cm_df)
            # TODO: also store run time
            results_df.to_csv(results_path)
