from prepare_data import HCDALoader
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense


def main():
    loader = HCDALoader()

    # load index values from main table
    app_ix = loader.applications_train_index()

    # fit model using k-fold verification
    kf = KFold(n_splits=2, shuffle=True)
    for fold_indexes in kf.split(app_ix):
        data_train, target_train, data_val, target_val = loader.load_train_val(fold_indexes[0], fold_indexes[1])

        # oversample troubled loans to make up for imbalance
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        data_train_os, target_train_os = ros.fit_sample(data_train, target_train)

        # TODO: move model to new method
        model = Sequential()

        model.add(Dense(units=64, activation='relu', input_dim=data_train_os.shape[1]))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        model.fit(data_train_os, target_train_os, epochs=5, batch_size=32, validation_data=(data_val, target_val))

        # TODO: new model perfomring 1D convolution on bureau and loan balances


if __name__ == "__main__":
    main()
