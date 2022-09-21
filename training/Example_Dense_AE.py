#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Activation,
    Layer,
    ReLU,
    LeakyReLU,
)
from tensorflow.keras import backend as K


# In[2]:


#from func import load_model, save_model


# ## Load dataset

# In[3]:


folder = '../preprocessing-code/'
filename = "BKG_dataset.h5"


# In[4]:


# make sure input data has correct input shape - background training data
with h5py.File(folder+filename, "r") as file:
    X_train = np.array(file["X_train"])
    X_test = np.array(file["X_test"])
    X_val = np.array(file["X_val"])


# ## Define Dense NN architecture

# In[5]:


input_shape = 60
latent_dimension = 5
num_nodes = [20, 15]


# In[6]:


# encoder
inputArray = Input(shape=(input_shape))
x = Dense(num_nodes[0], use_bias=False)(inputArray)
x = Activation("relu")(x)
x = Dense(latent_dimension, use_bias=False)(x)
encoder = Activation("relu")(x)

# decoder
x = Dense(num_nodes[0], use_bias=False)(encoder)
x = Activation("relu")(x)
decoder = Dense(input_shape)(x)

# create autoencoder
autoencoder = Model(inputs=inputArray, outputs=decoder)
autoencoder.summary()


# In[7]:


autoencoder.compile(optimizer=keras.optimizers.Adam(), loss="mse")


# ## Train model

# In[8]:


EPOCHS = 10
BATCH_SIZE = 512


# In[9]:


history = autoencoder.fit(
    X_train,
    X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, X_val),
)


# In[10]:


model_name = "model_name"
model_directory = ""
#save_model(model_directory + model_name, autoencoder)


# ## Prediction - background

# In[11]:


bkg_prediction = autoencoder.predict(X_test)


# ## Prediction - signals

# In[ ]:


# add correct signal labels
signal_labels = ["Ato4l", "hChToTauNu", "hToTauTau", "leptoquark"]


# In[ ]:


# add correct path to signal files
signals_file = ["Ato4l_lepFilter_13TeV_dataset.h5", "hChToTauNu_13TeV_PU20_dataset.h5", "hToTauTau_13TeV_PU20_dataset.h5", "leptoquark_LOWMASS_lepFilter_13TeV_dataset.h5"]


# In[ ]:


# read signal data
signal_data = []
for i, label in enumerate(signal_labels):
    with h5py.File(folder+signals_file[i], "r") as file:
        test_data = np.array(file["Data"])
    signal_data.append(test_data)


# In[ ]:


signal_results = []

for i, label in enumerate(signal_labels):
    signal_prediction = autoencoder.predict(signal_data[i])
    signal_results.append(
        [label, signal_data[i], signal_prediction]
    )  # save [label, true, prediction] for signal


# ## Save results

# In[ ]:


save_file = "save_file"


# In[ ]:


with h5py.File(save_file, "w") as file:
    file.create_dataset("BKG_input", data=X_test)
    file.create_dataset("BKG_predicted", data=bkg_prediction)
    for i, sig in enumerate(signal_results):
        file.create_dataset("%s_input" % sig[0], data=sig[1])
        file.create_dataset("%s_predicted" % sig[0], data=sig[2])


# ## Evaluate results
#
# 1. Plot loss distribution after prediction (check loss value for signals)
# 2. Plot ROC curves - how good is anomaly detection for chosen FPR threshold

# # 1.

# In[ ]:


def mse_loss(true, prediction):
    loss = tf.reduce_mean(tf.math.square(true - prediction),axis=-1)
    return loss


# In[ ]:


# compute loss value (true, predicted)
total_loss = []
total_loss.append(mse_loss(X_test, bkg_prediction.astype(np.float32)).numpy())
for i, signal_X in enumerate(signal_data):
    total_loss.append(
        mse_loss(signal_X, signal_results[i][2].astype(np.float32)).numpy()
    )


# In[ ]:


bin_size = 100

plt.figure(figsize=(10, 8))
for i, label in enumerate(signal_labels):
    plt.hist(
        total_loss[i],
        bins=bin_size,
        label=label,
        density=True,
        histtype="step",
        fill=False,
        linewidth=1.5,
    )
plt.yscale("log")
plt.xlabel("Autoencoder Loss")
plt.ylabel("Probability (a.u.)")
plt.title("MSE loss")
plt.legend(loc="best")
plt.show()


# # 2.

# In[ ]:


from sklearn.metrics import roc_curve, auc


# In[ ]:


labels = np.concatenate([["Background"], np.array(signal_labels)])


# In[ ]:


target_background = np.zeros(total_loss[0].shape[0])

plt.figure(figsize=(10, 8))
for i, label in enumerate(labels):
    if i == 0:
        continue  # background events

    trueVal = np.concatenate(
        (np.ones(total_loss[i].shape[0]), target_background)
    )  # anomaly=1, bkg=0
    predVal_loss = np.concatenate((total_loss[i], total_loss[0]))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

    auc_loss = auc(fpr_loss, tpr_loss)

    plt.plot(
        fpr_loss,
        tpr_loss,
        "-",
        label="%s (auc = %.1f%%)" % (label, auc_loss * 100.0),
        linewidth=1.5,
    )

    plt.semilogx()
    plt.semilogy()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="center right")
    plt.grid(True)
    plt.tight_layout()
plt.plot(np.linspace(0, 1), np.linspace(0, 1), "--", color="0.75")
plt.axvline(
    0.00001, color="red", linestyle="dashed", linewidth=1
)  # threshold value for measuring anomaly detection efficiency
plt.title("ROC AE")
plt.show()


# In[ ]:
