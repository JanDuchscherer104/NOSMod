#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:12:53 2022

@author: admin


"""

import copy

# %matplotlib inline
import datetime
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras import Model, layers, models
from tensorflow.keras.layers import (
    LSTM,
    Concatenate,
    Conv1D,
    Conv1DTranspose,
    Dense,
    Input,
    Lambda,
    Layer,
    Reshape,
    SimpleRNN,
)

plt.close("all")

nos = 128 * 1
sps = 32
N_symbols_in = 30000
X_tmp = np.zeros((N_symbols_in, nos, 4))

for c in range(N_symbols_in):

    x_x = (np.array([np.random.uniform(0, 3, nos)]))[0]  # np.around
    s = x_x + 1
    x_x = np.exp(1j * np.pi * x_x / 2 + 1j * np.pi / 4) * np.sqrt(2)

    x_y = (np.array([np.random.uniform(0, 3, nos)]))[0]  # np.around
    x_y = np.exp(1j * np.pi * x_y / 2 + 1j * np.pi / 4) * np.sqrt(2)

    a = np.array([np.real(x_x), np.imag(x_x), np.real(x_y), np.imag(x_y)])
    X_tmp[c] = np.reshape(np.rot90(a, k=3), (nos, 4))

    # X_tmp[c] = np.reshape(np.rot90(np.array([x[0], x[0]])),(512,2))

# sys.exit()
# split into input (X) and output (Y) variables
test_ratio = 0.99
X1 = X_tmp[0 : (int(X_tmp.shape[0] * test_ratio)), :]
# X1 += np.random.uniform(-0.1, 0.1, (np.shape(X1)[0], nos, 4))
Y1 = X_tmp[0 : (int(X_tmp.shape[0] * test_ratio)), :]

X2 = X_tmp[(int(X_tmp.shape[0] * test_ratio)) :, :]
# X2 += np.random.uniform(-0.1, 0.1, (np.shape(X2)[0], nos, 4))
# Y2 = X_tmp[(int(X_tmp.shape[0]*test_ratio)):,:]
Y2 = X_tmp[(int(X_tmp.shape[0] * test_ratio)) :, :]


# create model
model = Sequential()
N_in = 2 * nos
N_out = nos
N_Layer_BP = 2 * nos * sps // 16
N_layer_pre = 2 * nos * sps
N_layer_post = 2 * nos
N_samples_in = nos * sps


def slice_sequence(x):
    return x[:, sps // 2 :: sps // 1, :]


def slice_feature(i):
    return Lambda(lambda x: x[:, :, i : i + 1])


@keras.saving.register_keras_serializable()
class RingPaddingLayer(Layer):
    def __init__(self, wow, **kwargs):
        super(RingPaddingLayer, self).__init__(**kwargs)
        self.wow = wow

    def build(self, input_shape):
        # Hier würden normalerweise Gewichte initialisiert werden, wenn nötig
        super(RingPaddingLayer, self).build(input_shape)

    def call(self, inputs):
        nof = inputs.shape[-1]  # Anzahl der Features

        # Ring-Padding hinzufügen
        padding = inputs[
            :, -self.wow + 1 :, :
        ]  # Wähle die letzten (wow-1) Features für das Padding
        inputs_with_ring_padding = tf.concat(
            [padding, inputs], axis=1
        )  # Füge Padding vorne hininzu

        # Erstelle eine Liste von Fenstern
        output_list = []
        for i in range(inputs.shape[1]):  # Gehe nur über die originalen Schritte
            window = inputs_with_ring_padding[:, i : i + self.wow, :]
            output_list.append(window)

        output = tf.stack(
            output_list, axis=1
        )  # Staple die Fenster entlang der Zeitachse

        # Flatten die letzten beiden Dimensionen um ein 3D-Tensor zu bekommen
        flattened_tensor = tf.reshape(output, [-1, inputs.shape[1], self.wow * nof])
        return flattened_tensor

    def compute_output_shape(self, input_shape):
        nof = input_shape[-1]
        # Die Länge der Sequenz bleibt unverändert, nur die Feature-Dimension vergrößert sich
        new_nof = self.wow * nof
        return (input_shape[0], input_shape[1], new_nof)


# Already implemented in PyTorch


class CustomReshapeLayer(Layer):
    def __init__(self):
        super(CustomReshapeLayer, self).__init__()

    def call(self, inputs):
        # Nehmen Sie jedes zweite Element, beginnend bei Index 0, bis zum Ende der Achse 1
        # Das Ergebnis hat dann die Shape (:, 64, 4)
        skipped = inputs[:, 64::128, :]

        # Stapeln Sie paarweise entlang der letzten Dimension
        # Also, reshape von (:, 64, 4) zu (:, 32, 8)
        # Hier kombinieren wir jeweils zwei aufeinanderfolgende (2*4=8) Features
        shape = tf.shape(skipped)
        new_shape = (shape[0], shape[1] // 2, shape[2] * 2)
        reshaped = tf.reshape(skipped, new_shape)

        return reshaped

    def compute_output_shape(self, input_shape):
        # Hier aktualisieren wir die Ausgabeform entsprechend der Transformation
        return (input_shape[0], input_shape[1] // 4, input_shape[2] * 2)


input_shape = (nos * 1, 4)
inputs = Input(shape=input_shape)

NoN = 121

# Slice neighbour haeppchen
sequence_layer = RingPaddingLayer(NoN)(inputs)


# Modulator

# Dense-Layer definieren, der auf jedes der nos Elemente angewandt werden soll
# dense_layer = TimeDistributed(Dense(4*sps))
No_Neuroms = sps // 4
mod_dense_layer_0 = Dense(
    No_Neuroms, activation="tanh", use_bias=True, kernel_initializer="random_normal"
)
mod_dense_layer_1 = Dense(
    4 * sps, activation="linear", use_bias=True, kernel_initializer="random_normal"
)


# TimeDistributed Layer auf den flatten_layer anwenden
output_layer_0 = mod_dense_layer_0(sequence_layer)
output_layer = mod_dense_layer_1(output_layer_0)


# In das Format für den Filter umwandeln.
reshaped_layer = Reshape((nos * sps, 4))(output_layer)

# Filter 1
splitted_tensors = [slice_feature(i)(reshaped_layer) for i in range(4)]

# Jeder Tensor geht durch seine eigene Dense Schicht
shared_dense_0 = Dense(
    N_samples_in, activation="linear", use_bias=False, kernel_initializer="Identity"
)

processed_tensor_0 = shared_dense_0(tf.squeeze(splitted_tensors[0], axis=-1))
processed_tensor_1 = shared_dense_0(tf.squeeze(splitted_tensors[1], axis=-1))
processed_tensor_2 = shared_dense_0(tf.squeeze(splitted_tensors[2], axis=-1))
processed_tensor_3 = shared_dense_0(tf.squeeze(splitted_tensors[3], axis=-1))

# Konkatenieren entlang der neuen Achse um die gewünschte Form zu erhalten: (None, N_samples_in, 4)
expanded_tensors = [
    Lambda(lambda x: tf.expand_dims(x, axis=2))(tensor)
    for tensor in [
        processed_tensor_0,
        processed_tensor_1,
        processed_tensor_2,
        processed_tensor_3,
    ]
]
concatenated_tensor = Concatenate(axis=2)(expanded_tensors)


# Sampeln
E_sampled = slice_sequence(concatenated_tensor)
# E_sampled = CustomReshapeLayer()(concatenated_tensor)

# Demodulieren
E_demod_fenster = RingPaddingLayer(5)(E_sampled)

# demod_dense_layer = TimeDistributed(Dense(4))
No_Neuroms = 8
demod_dense_layer_0 = Dense(
    No_Neuroms * 4, activation="relu", use_bias=True, kernel_initializer="random_normal"
)
demod_dense_layer_1 = Dense(
    No_Neuroms * 4, activation="relu", use_bias=True, kernel_initializer="random_normal"
)
demod_dense_layer_2 = Dense(
    4, activation="linear", use_bias=True, kernel_initializer="random_normal"
)

E_demod_output_layer_0 = demod_dense_layer_0(E_demod_fenster)
E_demod_output_layer_1 = demod_dense_layer_1(E_demod_output_layer_0)
E_demod_output_layer = demod_dense_layer_2(E_demod_output_layer_1)

model = Model(inputs=inputs, outputs=E_demod_output_layer)

# sys.exit()
# Set weight of sqrt(BP)
N_MOD_Layer = 12 + 1


# Set weight of TP
print("TP 1")
model_BP = keras.models.load_model("model_B20_sqrt_singlefeature.keras")
for c in range(1):
    layer_BP = model_BP.layers[1]
    weights_BP = layer_BP.get_weights()

    # BP 1
    layer = model.layers[N_MOD_Layer + 0 + c]
    layer.set_weights(weights_BP)
    layer.trainable = False


model.summary()
"""

del(X2)
X2=np.zeros((1,1*nos,4))
X2[0][:,0] = np.arange(nos*1)
X2[0][:,1] = np.arange(nos*1) + 1*nos*1
X2[0][:,2] = np.arange(nos*1) + 2*nos*1
X2[0][:,3] = np.arange(nos*1) + 3*nos*1

Y = model.predict(X2)
"""
# Compile model
log_dir = "/home/striegler/tensorlogs/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(
    loss="mse", optimizer="Adam", metrics=["accuracy"]
)  # mse, sparse_categorical_crossentropy
# model = keras.models.load_model("model_tmp.keras", safe_mode=False)
model.summary()
# model.compile(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name="RMSprop", metrics=['accuracy'])


# Fit the model
# model = keras.models.load_model("model_tmp.keras", safe_mode=False)
model.fit(
    X1, Y1, epochs=1500, batch_size=100, verbose=2, callbacks=[tensorboard_callback]
)
model.save("model_tmp.keras")


# Calculate predictions

N_symbols_in = 300
X2 = np.zeros((N_symbols_in, nos, 4))

for c in range(N_symbols_in):

    x_x = np.around(np.array([np.random.uniform(0, 3, nos)]))[0]  # np.around

    x_x = np.exp(1j * np.pi * x_x / 2 + 1j * np.pi / 4) * np.sqrt(2)

    x_y = np.around(np.array([np.random.uniform(0, 3, nos)]))[0]  # np.around
    x_y = np.exp(1j * np.pi * x_y / 2 + 1j * np.pi / 4) * np.sqrt(2)

    a = np.array([np.real(x_x), np.imag(x_x), np.real(x_y), np.imag(x_y)])
    X2[c] = np.reshape(np.rot90(a, k=3), (nos, 4))

Y2 = X2


Y = model.predict(X2)


# Plot actual vs predition for validation set
# ValResults = np.genfromtxt("valresults.csv", delimiter=",")
plt.figure(1)
a = np.shape(Y)[0] * np.shape(Y)[1] * np.shape(Y)[2]
plt.plot(np.reshape(Y2, (a)), np.reshape(Y, (a)), "go")
plt.plot(np.reshape(Y2, (a)), np.reshape(np.around(Y), (a)), "r.")


plt.title("Validation Set")
plt.xlabel("Actual")
plt.ylabel("Predicted")

n = 21
plt.figure(2)
plt.plot(Y[n, :, 0], "k")
plt.plot(Y2[n, :, 0], "r:")
plt.figure(2)
plt.plot(Y[n, :, 1], "k")
plt.plot(Y2[n, :, 1], "r:")
plt.figure(2)
plt.plot(Y[n, :, 2], "k")
plt.plot(Y2[n, :, 2], "r:")
plt.figure(2)
plt.plot(Y[n, :, 3], "k")
plt.plot(Y2[n, :, 3], "r:")


N_symbols_in = 30
X_test = np.zeros((N_symbols_in, nos, 4))

for c in range(N_symbols_in):

    x_x = np.around(np.array([np.random.uniform(0, 3, nos)]))[0]  # np.around
    s = x_x + 1
    x_x = np.exp(1j * np.pi * x_x / 2 + 1j * np.pi / 4) * np.sqrt(2)

    x_y = np.around(np.array([np.random.uniform(0, 3, nos)]))[0]  # np.around
    x_y = np.exp(1j * np.pi * x_y / 2 + 1j * np.pi / 4) * np.sqrt(2)

    a = np.array([np.real(x_x), np.imag(x_x), np.real(x_y), np.imag(x_y)])
    X_test[c] = np.reshape(np.rot90(a, k=3), (nos, 4))

Y_test = model.predict(X_test)


n = 20
plt.figure(3)
colco = "gr"
N_err = 0
for n in range(30):

    isuneql = np.logical_xor(np.around(Y_test[n, :, 0]), X_test[n, :, 0]) + 0
    N_err += np.sum(isuneql)
    for c in range(nos):
        plt.plot(Y_test[n, c, 0], Y_test[n, c, 1], colco[isuneql[c]] + ".", alpha=0.25)

# plt.plot(X_test[n,:,0], 'r.')
# plt.plot(isuneql*0.5, 'k.')

# plt.grid(b=True, which='major', color='#999999', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.show()
