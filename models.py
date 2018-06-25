import logging
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2


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
