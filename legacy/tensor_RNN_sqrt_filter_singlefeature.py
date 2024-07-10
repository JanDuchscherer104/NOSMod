#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:12:53 2022

@author: admin

import keras
model = keras.models.saving.load_model("model.keras")
model.save("model.keras")


layer = model.layers[1]  # z.B. der erste Dense Layer
weights = layer.get_weights()


layer.set_weights(weights)
# Einfrieren dieses Layers
layer.trainable = False


sqrt : 2400 Epochen, 10000 Samples
sqrt-SMP : 13 Epochen, 30e3 Samples

"""
import datetime

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from tensorflow.keras import Model, layers, models

# %matplotlib inline
from tensorflow.keras.layers import Dense, Input

log_dir = "/home/striegler/tensorlogs/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)
# log_dir = "/home/arne/tensorlogs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Read data for training and validation data
N_max = 10000
nos = 128 * 1
sps = 32
n_features = 4
samplen = False


X = np.zeros((N_max, 1 * nos * sps))
if samplen:
    Y = np.zeros((N_max, 1 * nos))
else:
    Y = np.zeros((N_max, 1 * sps * nos))


for c in range(N_max):

    X[c] = np.load("Data/E_in" + str(c) + ".npy")
    if samplen:
        Y[c] = np.load("Data/E_out" + str(c) + ".npy")[sps // 2 :: sps, :]
    else:
        Y[c] = np.load("Data/E_out" + str(c) + ".npy")


# split into input (X) and output (Y) variables
test_ratio = 0.99
X1 = X[0 : (int(X.shape[0] * test_ratio)), :]
Y1 = Y[0 : (int(Y.shape[0] * test_ratio))]

X2 = X[(int(X.shape[0] * test_ratio)) :, :]
Y2 = Y[(int(Y.shape[0] * test_ratio)) :]
# X2[N_max-1] = np.zeros(sps*nos*2)
# X2[N_max-1][sps*nos*2//4:sps*nos*2//4+1] = 0.5
# X2[N_max-1][sps*nos*2*3//4:sps*nos*2*3//4+1] = 0.5

# create model


if samplen:
    N_layer = 1 * nos
else:
    N_layer = sps * nos


N_samples_in = nos * sps
N_samples_out = nos * sps


input_shape = N_samples_in
inputs = Input(shape=input_shape)


shared_dense_0 = Dense(
    N_samples_in,
    activation="linear",
    use_bias=False,
    kernel_initializer="random_normal",
)

output = shared_dense_0(inputs)

model = Model(inputs=inputs, outputs=output)
model.summary()

# Modellübersicht ausgeben
model.summary()


# Compile model
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])  # mse
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# model = keras.models.load_model("model_B20_sqrt.keras", safe_mode=False)
# Fit the model
model.fit(
    X1, Y1, epochs=35, batch_size=100, verbose=2, callbacks=[tensorboard_callback]
)

# Calculate predictions
# Y = model.predict(X1)

Y = model.predict(X2)
model.save("model_B20_sqrt_singlefeature.keras")
"""
plt.figure(222)
colc= ['r', 'g', 'b', 'k']
for t in range(4):
    plt.plot(Y[t][0][:,0],'-'+colc[t], alpha = 0.25)
    plt.plot(X2[0,:,t],'.'+colc[t], alpha = 0.25)
sys.exit()
"""
# Plot actual vs predition for validation set
# ValResults = np.genfromtxt("valresults.csv", delimiter=",")
plt.figure(1)
plt.plot(Y2, Y, "g.")


plt.title("Validation Set")
plt.xlabel("Actual")
plt.ylabel("Predicted")

n = 0
plt.figure(2)
plt.plot(Y[n], "k")
plt.plot(Y2[n], "r:")

#  plt.plot(X2[n,:,1],'b:');


# plt.grid(b=True, which='major', color='#999999', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.show()
