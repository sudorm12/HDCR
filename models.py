import logging
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, concatenate
from keras.regularizers import l2
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


class LinearNN:
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, l2_reg=0):
        self._model = Sequential()

        for i in range(num_layers):
            self._model.add(Dense(units=hidden_dim,
                                  activation='relu',
                                  kernel_regularizer=l2(l2_reg),
                                  input_dim=input_dim))

        self._model.add(Dense(units=1, activation='sigmoid'))

        self._model.compile(loss='binary_crossentropy',
                            optimizer='Adam',
                            metrics=['accuracy'])

    def fit(self, data_train, target_train, data_val, target_val, num_epochs=5, batch_size=32):
        self._model.fit(data_train, target_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_data=(data_val, target_val))

    def predict(self, data):
        return self._model.predict(data)


class GBC:
    def __init__(self, n_estimators=10):
        self._model = GradientBoostingClassifier(n_estimators=n_estimators)

    def fit(self, data_train, target_train):
        self._model.fit(data_train, target_train)

    def predict(self, data):
        self._model.predict(data)


class ABC:
    def __init__(self, n_estimators=10):
        self._model = AdaBoostClassifier(n_estimators=n_estimators)

    def fit(self, data_train, target_train):
        self._model.fit(data_train, target_train)

    def predict(self, data):
        self._model.predict(data)


class LSTMWithMetadata:
    def __init__(self, sequence_length, meta_length, num_dense_layers=1):
        # this code is for only one sequence per row
        main_input = Input(shape=(sequence_length,), dtype='int32', name='main_input')
        lstm = LSTM(32)(main_input)
        lstm_output = Dense(1, activation='sigmoid', name='lstm_output')(lstm)
        meta_input = Input(shape=(meta_length,))
        x = concatenate([lstm, meta_input])
        for i in range(num_dense_layers):
            x = Dense(64, activation='relu')(x)
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)

        self._model = Model(inputs=[main_input, meta_input], outputs=[main_output, lstm_output])

        self._model.compile(optimizer='rmsprop',
                            loss='binary_crossentropy',
                            loss_weights=[1., 0.2])

    def fit(self, ts_data_train, meta_data_train, target_train,
            ts_data_val, meta_data_val, target_val,
            num_epochs=5, batch_size=32):
        self._model.fit([ts_data_train, meta_data_train], [target_train, target_train],
                        validation_data=([ts_data_val, meta_data_val], [target_val, target_val]),
                        epochs=num_epochs, batch_size=batch_size)

    def predict(self, ts_data, meta_data):
        return self._model.predict([ts_data, meta_data])
