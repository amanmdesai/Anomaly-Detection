import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, roc_curve

folder = "../preprocessing-code/"
filename = "BKG_dataset.h5"

with h5py.File(folder + filename, "r") as file:
    X_train = np.array(file["X_train"])
    X_test = np.array(file["X_test"])
    X_val = np.array(file["X_val"])


class Model:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators

    def IsolationForest(self):
        return IsolationForest(n_estimators=self.n_estimators, n_jobs=5, random_state=5)


model = Model(40).IsolationForest()

autoencoder = model.fit(X_train)
bkg_prediction = autoencoder.predict(X_test)

print("SM scanning completed")

signal_labels = ["Ato4l", "hChToTauNu"]  # , "hToTauTau", "leptoquark"]

# add correct path to signal files
signals_file = [
    "Ato4l_lepFilter_13TeV_dataset.h5",
    "hChToTauNu_13TeV_PU20_dataset.h5",
    #    "hToTauTau_13TeV_PU20_dataset.h5",
    #    "leptoquark_LOWMASS_lepFilter_13TeV_dataset.h5",
]

signal_data = []
for i, label in enumerate(signal_labels):
    with h5py.File(folder + signals_file[i], "r") as file:
        test_data = np.array(file["Data"])
    signal_data.append(test_data)

signal_results = []

for i, label in enumerate(signal_labels):
    print("now predicting for ", label)
    signal_prediction = autoencoder.predict(signal_data[i])
    signal_results.append(
        [label, signal_data[i], signal_prediction]
    )  # save [label, true, prediction] for signal


# print(signal_prediction.shape)


def mse_loss(true, prediction):
    # print(prediction)
    loss = tf.reduce_mean(tf.math.square(true - prediction), axis=-1)
    # loss = - tf.reduce_mean(tf.math.log(1-(tf.math.square(true - prediction))),axis=-1)
    return loss


# In[ ]:


# compute loss value (true, predicted)
total_loss = []
total_loss.append(
    mse_loss(np.ones(X_test.shape[0]), bkg_prediction.astype(np.float32)).numpy()
)
for i, signal_X in enumerate(signal_data):
    # print(signal_X.shape, signal_results[i][2])
    total_loss.append(
        mse_loss(
            -1 * np.ones(signal_X.shape[0]), signal_results[i][2].astype(np.float32)
        ).numpy()
    )

# print(total_loss)

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
# print(total_loss)
# target_background = np.zeros(total_loss[0].shape[0])

plt.figure(figsize=(10, 8))
for i, label in enumerate(labels):
    if i == 0:
        continue  # background events
    # trueVal = np.concatenate((np.ones))
    # trueVal = np.concatenate(
    #    (np.ones(total_loss[i].shape[0]), target_background)
    # )  # anomaly=1, bkg=0
    # predVal_loss = np.concatenate((total_loss[i], total_loss[0]))
    # print(signal_results[i][2],signal_results[i][2].shape)
    trueVal = -1 * np.ones(len(signal_prediction[i]))
    predVal_loss = signal_prediction[
        i
    ]  # (signal_results[i][2].astype(np.float32)).numpy() #,signal_results[1][2].astype(np.float32))
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
