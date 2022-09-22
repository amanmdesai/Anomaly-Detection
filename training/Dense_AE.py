#!/usr/bin/env python


import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import auc, roc_curve
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Activation  # LeakyReLU,; ReLU,
from tensorflow.keras.layers import BatchNormalization, Dense, Input, Layer
from tensorflow.keras.models import Model

# In[2]:
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)],
        )
    except RuntimeError as e:
        print(e)


# from func import load_model, save_model


# ## Load dataset

# In[3]:


folder = "../preprocessing-code/"
filename = "BKG_dataset.h5"


# In[4]:


# make sure input data has correct input shape - background training data
with h5py.File(folder + filename, "r") as file:
    X_train = np.array(file["X_train"])
    X_test = np.array(file["X_test"])
    X_val = np.array(file["X_val"])

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train[:], X_train[:])
# X_test = scaler.transform(X_test[:])
# X_val = scaler.transform(X_val[:])
# ## Define Dense NN architecture

# In[5]:


input_shape = 60
latent_dimension = 5
num_nodes = [19, 14, 9]  # [25,15]#
EPOCHS = 20
BATCH_SIZE = 1024

activation = "LeakyReLU"  # LeakyReLU
# encoder

initializer = tf.keras.initializers.GlorotUniform()


inputArray = Input(shape=(input_shape))
x = Dense(num_nodes[0], use_bias=False, kernel_initializer=initializer)(inputArray)
x = Activation(activation)(x)
x = Dense(num_nodes[1], use_bias=False, kernel_initializer=initializer)(x)
x = Activation(activation)(x)
x = Dense(num_nodes[2], use_bias=False, kernel_initializer=initializer)(x)
x = Activation(activation)(x)
x = Dense(latent_dimension, use_bias=False, kernel_initializer=initializer)(x)
encoder = Activation(activation)(x)

# decoder
x = Dense(num_nodes[2], use_bias=False, kernel_initializer=initializer)(encoder)
x = Activation(activation)(x)
x = Dense(num_nodes[1], use_bias=False, kernel_initializer=initializer)(x)
x = Activation(activation)(x)
x = Dense(num_nodes[0], use_bias=False, kernel_initializer=initializer)(x)
x = Activation(activation)(x)
decoder = Dense(input_shape)(x)

# create autoencoder
autoencoder = Model(inputs=inputArray, outputs=decoder)
autoencoder.summary()


# In[7]:


autoencoder.compile(
    optimizer=keras.optimizers.Adam(), loss="mse", metrics=["ACC", "AUC"]
)  # ,metrics=['AUC','ACC','MSE']) #loss= "mse", mae, msle MeanSquaredLogarithmicError


# ## Train model

# learning_rate=0.0005


# In[9]:

callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    min_delta=0.003,
    patience=10,
    verbose=1,
    mode="max",
    baseline=None,
    restore_best_weights=True,
)

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, X_val),
    callbacks=callbacks,
)


# In[10]:


model_name = "model_name"
model_directory = ""
# save_model(model_directory + model_name, autoencoder)


# ## Prediction - background

# In[11]:


bkg_prediction = autoencoder.predict(X_test)


# ## Prediction - signals

# In[ ]:


# add correct signal labels
signal_labels = ["Ato4l", "hChToTauNu", "hToTauTau", "leptoquark"]


# In[ ]:


# add correct path to signal files
signals_file = [
    "Ato4l_lepFilter_13TeV_dataset.h5",
    "hChToTauNu_13TeV_PU20_dataset.h5",
    "hToTauTau_13TeV_PU20_dataset.h5",
    "leptoquark_LOWMASS_lepFilter_13TeV_dataset.h5",
]


# In[ ]:


# read signal data
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


# ## Save results

# In[ ]:


save_file = "save_file"


# In[ ]:

"""
with h5py.File(save_file, "w") as file:
    file.create_dataset("BKG_input", data=X_test)
    file.create_dataset("BKG_predicted", data=bkg_prediction)
    for i, sig in enumerate(signal_results):
        file.create_dataset("%s_input" % sig[0], data=sig[1])
        file.create_dataset("%s_predicted" % sig[0], data=sig[2])
"""

# ## Evaluate results
#
# 1. Plot loss distribution after prediction (check loss value for signals)
# 2. Plot ROC curves - how good is anomaly detection for chosen FPR threshold

# # 1.

# In[ ]:


def mse_loss(true, prediction):
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


# In[ ]:


bin_size = 100
"""
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
"""

# # 2.

# In[ ]:


# In[ ]:


labels = np.concatenate([["Background"], np.array(signal_labels)])


"""
plt.plot(model.history.history["loss"], label="Training loss")
plt.plot(model.history.history["val_loss"], label="Validation loss")
plt.legend()
plt.show()

plt.plot(model.history.history["ACC"], label="Training accuracy")
plt.plot(model.history.history["val_ACC"], label="Validation accuracy")
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


# In[ ]:
