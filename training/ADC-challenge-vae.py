#!/usr/bin/env python
# coding: utf-8


import numpy as np
import h5py
import math
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, BatchNormalization, Activation, Layer, ReLU, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.callbacks import ModelCheckpoint


folder = "../processed-dataset/"
filename = "BKG_dataset.h5"



with h5py.File(folder+filename,'r') as file:
    X_train  = np.array(file['X_train'])
    X_val  = np.array(file['X_val'])
    X_test  = np.array(file['X_test'])
X_train.shape[1]



input_shape = 57
latent_dimension = 6
#num_nodes=[40,30,20]

#num_nodes=[25,20]
num_nodes=[25,20]

EPOCHS = 20
BATCH_SIZE = 512


class custom_func(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batches = tf.shape(z_mean)[0]
        dimension = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batches, dimension))
        return z_mean + tf.exp(-1*(z_log_var)*(z_log_var)) * epsilon + tf.exp(-1*z_mean) * epsilon


inputArray = keras.Input(shape=(57))
x = Dense(num_nodes[0], activation='LeakyReLU',use_bias=False)(inputArray)
x = Dense(num_nodes[1], activation='LeakyReLU',use_bias=False)(x)

z_mean_1 = layers.Dense(latent_dimension, activation='ReLU', name="z_mean_1")(x)
z_log_var = layers.Dense(latent_dimension, activation='ReLU', name="z_log_var")(x)
z_1 = custom_func()([z_mean_1, z_log_var])

bottle_neck = Dense(latent_dimension, activation='LeakyReLU',use_bias=False)(z_1)

x = Dense(num_nodes[1], activation='LeakyReLU',use_bias=False)(bottle_neck)
x = Dense(num_nodes[0], activation='LeakyReLU',use_bias=False)(x)

decoder = Dense(input_shape)(x)

autoencoder = Model(inputs = inputArray, outputs=decoder)
autoencoder.summary()
autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.003),
    loss="mse",
    metrics=['ACC'])




tf.keras.utils.plot_model(
    autoencoder,
    to_file='result/model_arch_vae.png',
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=True,
    dpi=96,
    layer_range=None,
    show_layer_activations=False
)




#filepath = 'models/my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
filepath = 'models/best_model.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]




history = autoencoder.fit(X_train, X_train, epochs = EPOCHS, batch_size = 1024,
                  validation_data=(X_val, X_val),callbacks=callbacks)





plt.plot(autoencoder.history.history["loss"], label="Training loss")
plt.plot(autoencoder.history.history["val_loss"], label="Validation loss")
plt.legend()
plt.savefig('result/loss.png')
plt.figure(figsize=(10,8))




plt.plot(autoencoder.history.history["ACC"], label="Training accuracy")
plt.plot(autoencoder.history.history["val_ACC"], label="Validation accuracy")
plt.legend()
plt.savefig('result/accuracy.png')





def save_model(model_save_name, model):
    with open(model_save_name + '.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(model_save_name + '.h5')





def load_model(model_name, custom_objects=None):
    name = model_name + '.json'
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects=custom_objects)
    model.load_weights(filepath)
    return model



del X_train, X_val

model_name = 'model_ae_aman'
model_directory = 'result/'
save_model(model_directory+model_name, autoencoder)





bkg_prediction = autoencoder.predict(X_test)





signal_labels = ["Ato4l", "hChToTauNu", "hToTauTau", "leptoquark"]

# add correct path to signal files
signals_file = [
    "Ato4l_lepFilter_13TeV_dataset.h5",
    "hChToTauNu_13TeV_PU20_dataset.h5",
    "hToTauTau_13TeV_PU20_dataset.h5",
    "leptoquark_LOWMASS_lepFilter_13TeV_dataset.h5",
]





signal_data = []
for i, label in enumerate(signal_labels):
    with h5py.File(folder+signals_file[i], 'r') as file:
        test_data = np.array(file['Data'])
    signal_data.append(test_data)





signal_results = []

for i, label in enumerate(signal_labels):
    signal_prediction = autoencoder.predict(signal_data[i])
    signal_results.append([label, signal_data[i], signal_prediction]) # save [label, true, prediction] for signal





def mse_loss(true, prediction):
    # loss = tf.reduce_mean(tf.math.abs(1-tf.math.log(true - prediction)), axis=-1)
    loss = tf.reduce_mean(tf.math.square(true - prediction), axis=-1)
    # loss = - tf.reduce_mean(tf.math.log(1-(tf.math.square(true - prediction))),axis=-1)
    return loss





# compute loss value (true, predicted)
total_loss = []
total_loss.append(mse_loss(X_test, bkg_prediction.astype(np.float32)).numpy())
for i, signal_X in enumerate(signal_data):
    total_loss.append(mse_loss(signal_X, signal_results[i][2].astype(np.float32)).numpy())





bin_size=100

plt.figure(figsize=(10,8))
for i, label in enumerate(signal_labels):
    plt.hist(total_loss[i], bins=bin_size, label=label, density = True, histtype='step', fill=False, linewidth=1.5)
plt.yscale('log')
plt.xlabel("Autoencoder Loss")
plt.ylabel("Probability (a.u.)")
plt.title('MSE loss')
plt.legend(loc='best')
plt.savefig('result/mse_loss.png')





labels = np.concatenate([['Background'], np.array(signal_labels)])





target_background = np.zeros(total_loss[0].shape[0])

plt.figure(figsize=(10,8))
for i, label in enumerate(labels):
    if i == 0: continue # background events

    trueVal = np.concatenate((np.ones(total_loss[i].shape[0]), target_background)) # anomaly=1, bkg=0
    predVal_loss = np.concatenate((total_loss[i], total_loss[0]))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

    auc_loss = auc(fpr_loss, tpr_loss)

    plt.plot(fpr_loss, tpr_loss, "-", label='%s (auc = %.1f%%)'%(label,auc_loss*100.), linewidth=1.5)

    plt.semilogx()
    plt.semilogy()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc='center right')
    plt.grid(True)
    plt.tight_layout()
plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
plt.title("ROC AE")
plt.savefig('result/roc.png')
