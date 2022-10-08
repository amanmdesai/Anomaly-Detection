#!/usr/bin/env python


from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.utils.vis_utils import plot_model
from sklearn.metrics import auc, roc_curve
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import GlorotUniform

# from tensorflow.keras.layers import   # LeakyReLU,; ReLU,
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    Input,
    Layer,
)
from tensorflow.keras.models import Model
'''	
# In[2]:
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)],
        )
    except RuntimeError as e:
        print(e)



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[2]:


folder = "../processed-dataset/"
filename = "BKG_dataset.h5"


# In[3]:


with h5py.File(folder + filename, "r") as file:
    X_train = np.array(file["X_train"])
    X_test = np.array(file["X_test"])
    X_val = np.array(file["X_val"])


# In[8]:


input_shape = 57
latent_dimension = 6
num_nodes = [24, 16, 10]  # [25,15]#
EPOCHS = 5
BATCH_SIZE = 2048
activation = "LeakyReLU"  # LeakyReLU


# In[16]:


initializer = tf.keras.initializers.GlorotUniform()
inputArray = Input(shape=(input_shape))
x = Dense(num_nodes[0], use_bias=False, kernel_initializer=initializer)(inputArray)
x = Activation(activation)(x)
x = Dense(num_nodes[1], use_bias=False, kernel_initializer=initializer)(x)
x = Activation(activation)(x)
#x = Dense(num_nodes[2], use_bias=False, kernel_initializer=initializer)(x)
#x = Activation(activation)(x)
x = Dense(latent_dimension, use_bias=False, kernel_initializer=initializer)(x)
x = Activation(activation)(x)
encoder_1 = Dense(latent_dimension - 3, use_bias=False, kernel_initializer=initializer)(x)
encoder_act1 = Activation("linear")(encoder_1)
encoder_2 = Dense(latent_dimension - 2, use_bias=False, kernel_initializer=initializer)(x)
encoder_act2 = Activation("linear")(encoder_2)


# decoder
merged = Concatenate()([encoder_act1, encoder_act2])
x = Activation(activation)(merged)
#x = Dense(num_nodes[2], use_bias=False, kernel_initializer=initializer)(x)
#x = Activation(activation)(x)
x = Dense(num_nodes[1], use_bias=False, kernel_initializer=initializer)(x)
x = Activation(activation)(x)
x = Dense(num_nodes[0], use_bias=False, kernel_initializer=initializer)(x)
x = Activation(activation)(x)
decoder = Dense(input_shape)(x)

# create autoencoder
autoencoder = Model(inputs=inputArray, outputs=decoder)
autoencoder.summary()

plot_model(
    autoencoder, to_file="model_plot.png", show_shapes=True, show_layer_names=True
)

# In[17]:


autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss="mse",
    metrics=['AUC','ACC'],  # , "AUC"]
)  # ,metrics=['AUC','ACC','MSE']) #loss= "mse", mae, msle MeanSquaredLogarithmicError mean_squared_logarithmic_error


# In[18]:


callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0.002,
    patience=10,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)

#X_train = X_train[:1000,:]

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, X_val),
    callbacks=callbacks,
    verbose=1,
)


# In[19]:


# model_name = "model_name"
# model_directory = ""
# save_model(model_directory + model_name, autoencoder)

# ## Prediction - background

bkg_prediction = autoencoder.predict(X_test)

# add correct signal labels
signal_labels = ["Ato4l", "hChToTauNu", "hToTauTau", "leptoquark"]

# add correct path to signal files
signals_file = [
    "Ato4l_lepFilter_13TeV_dataset.h5",
    "hChToTauNu_13TeV_PU20_dataset.h5",
    "hToTauTau_13TeV_PU20_dataset.h5",
    "leptoquark_LOWMASS_lepFilter_13TeV_dataset.h5",
]


# In[ ]:


signal_data = []
for i, label in enumerate(signal_labels):
    with h5py.File(folder + signals_file[i], "r") as file:
        test_data = np.array(file["Data"])
    signal_data.append(test_data)
    # signal_data = scaler.transform(signal_data)


# In[ ]:


signal_results = []

for i, label in enumerate(signal_labels):
    signal_prediction = autoencoder.predict(signal_data[i])
    signal_results.append(
        [label, signal_data[i], signal_prediction]
    )  # save [label, true, prediction] for signal


# In[ ]:


def mse_loss(true, prediction):
    # loss = tf.reduce_mean(tf.math.abs(1-tf.math.log(true - prediction)), axis=-1)
    loss = tf.reduce_mean(tf.math.square(true - prediction), axis=-1)
    # loss = - tf.reduce_mean(tf.math.log(1-(tf.math.square(true - prediction))),axis=-1)
    return loss


# In[ ]:


# compute loss value (true, predicted)
total_loss = []
total_loss.append(mse_loss(X_test, bkg_prediction.astype(np.float32)).numpy())
for i, signal_X in enumerate(signal_data):
    total_loss.append(
        mse_loss(signal_X, signal_results[i][2].astype(np.float32)).numpy()
    )
print(total_loss)
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


labels = np.concatenate([["Background"], np.array(signal_labels)])


"""
plt.plot(autoencoder.history.history["loss"], label="Training loss")
plt.plot(autoencoder.history.history["val_loss"], label="Validation loss")
plt.legend()
plt.show()

plt.plot(autoencoder.history.history["ACC"], label="Training accuracy")
plt.plot(autoencoder.history.history["val_ACC"], label="Validation accuracy")
plt.legend()
plt.show()

"""

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
    for j in range(len(fpr_loss)):
        if fpr_loss[j] == 0.00001:
            print(label, tpr_loss[j])

    auc_loss = auc(fpr_loss, tpr_loss)

    plt.plot(
        fpr_loss,
        tpr_loss,
        "-",
        label=f"{label} (auc = {auc_loss * 100.0:.1f}%)",
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


# ##
