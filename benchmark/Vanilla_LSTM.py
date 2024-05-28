from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import os
import random
import numpy as np
import tensorflow as tf

class Vanilla_LSTM:

    def __init__(self, params):
        self.params = params

    def _Random(self, seed_value):
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

    def _build_model(self):
        self._Random(0)
        model = Sequential()
        model.add(LSTM(100,
                       activation='relu',
                       return_sequences=True,
                       input_shape=(self.params[0], self.n_features)))
        model.add(LSTM(100,
                       activation='relu'))
        model.add(Dense(self.n_features))
        model.compile(optimizer='adam',
                      loss='mae',
                      metrics=["mse"])
        return model

    def fit(self, X, y):
        self.n_features = X.shape[2]
        self.model = self._build_model()
        early_stopping = EarlyStopping(patience=10,
                                       verbose=0)
        reduce_lr = ReduceLROnPlateau(factor=0.1,
                                      patience=5,
                                      min_lr=0.0001,
                                      verbose=0)
        self.model.fit(X, y,
                  validation_split=self.params[3],
                  epochs=self.params[1],
                  batch_size=self.params[2],
                  verbose=0,
                  shuffle=False,
                  callbacks=[early_stopping, reduce_lr]
                  )

    def predict(self, data):

        return self.model.predict(data)
