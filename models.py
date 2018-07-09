import logging
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Reshape, concatenate
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
    def __init__(self, sequence_length, sequence_features, meta_features,
                 sequence_dense_layers=1, meta_dense_layers=1, comb_dense_layers=1,
                 sequence_dense_width=32, meta_dense_width=32, comb_dense_width=32,
                 sequence_l2_reg=0, meta_l2_reg=0, comb_l2_reg=0):
        lstm_input = Input(shape=(sequence_length * sequence_features,), name='lstm_input')
        reshaped_input = Reshape((sequence_length, sequence_features), name='reshaped_input')(lstm_input)
        lstm = LSTM(8, activation='relu', go_backwards=False, name='lstm')(reshaped_input)
        for i in range(sequence_dense_layers):
            lstm = Dense(sequence_dense_width, activation='relu', kernel_regularizer=l2(sequence_l2_reg),
                         name='seq_dense_{}'.format(i))(lstm)
        lstm_output = Dense(1, activation='sigmoid', name='lstm_output')(lstm)
        meta_input = Input(shape=(meta_features,), name='meta_input')
        meta_dense = meta_input
        for i in range(meta_dense_layers):
            meta_dense = Dense(meta_dense_width, activation='relu', kernel_regularizer=l2(meta_l2_reg),
                               name='meta_dense_{}'.format(i))(meta_dense)
        x = concatenate([lstm, meta_dense], name='concatenate')
        for i in range(comb_dense_layers):
            x = Dense(comb_dense_width, activation='relu', kernel_regularizer=l2(comb_l2_reg),
                      name='combined_dense_{}'.format(i))(x)
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)

        self._model = Model(inputs=[lstm_input, meta_input], outputs=[main_output, lstm_output])

        self._model.compile(optimizer='rmsprop',
                            loss='binary_crossentropy',
                            loss_weights=[1., 0.2],
                            metrics=['accuracy'])

    def fit(self, ts_data_train, meta_data_train, target_train,
            ts_data_val, meta_data_val, target_val,
            num_epochs=5, batch_size=32):
        history = self._model.fit([ts_data_train, meta_data_train], [target_train, target_train],
                                  validation_data=([ts_data_val, meta_data_val], [target_val, target_val]),
                                  epochs=num_epochs, batch_size=batch_size)
        return history

    def predict(self, ts_data, meta_data):
        return self._model.predict([ts_data, meta_data])

    def model_summary(self):
        return self._model.summary()
