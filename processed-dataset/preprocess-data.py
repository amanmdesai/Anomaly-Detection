#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


def create_datasets_dense(
    bkg_file,
    output_bkg_name,
    signals_files,
    output_signal_names,
    events=None,
    test_size=0.2,
    val_size=0.2,
    input_shape=57,
    folder="../ADC-new-dataset/",
):

    # read BACKGROUND data
    with h5py.File(folder + bkg_file, "r") as file:
        input_file = file["Particles"][:, :, :-1]
        np.random.shuffle(input_file)
        full_data = np.array(input_file).reshape(input_file.shape[0], 57)
        del input_file
        # print(full_data.shape)
        # nel = 4 * np.ones(full_data.shape[0])
        # nmu = 4 * np.ones(full_data.shape[0])
        # njet = 10 * np.ones(full_data.shape[0])
        # full_data = np.column_stack((full_data, nel, nmu, njet))
        """for i in range(full_data.shape[0]):
            for j in range(full_data.shape[1]):
                if (j % 3 == 0) & (j <= 54):
                    if (
                        (full_data[i, j] == 0)
                        & (full_data[i, j + 1] == 0)
                        & (full_data[i, j + 2] == 0)
                    ):
                        full_data[i, j] = -9
                        full_data[i, j + 1] = -9
                        full_data[i, j + 2] = -9
                    # if (j >= 3) & (j <= 14):
                    #     full_data[i, 57] -= 1
                    # elif (j >= 15) & (j <= 26):
                    #     full_data[i, 58] -= 1
                    # else:
                    #     full_data[i, 59] -= 1
                else:
                    continue
        """
    # print(full_data[:,57],full_data[0,3:15])

    # define training, test and validation datasets
    X_train, X_test = train_test_split(full_data, test_size=test_size, shuffle=True)
    X_train, X_val = train_test_split(X_train, test_size=val_size)
    del full_data


    # flatten the data for model input
    # X_train = X_train.reshape(X_train.shape[0], input_shape)
    # X_test = X_test.reshape(X_test.shape[0], input_shape)
    # X_val = X_val.reshape(X_val.shape[0], input_shape)

    with h5py.File(output_bkg_name + "_dataset.h5", "w") as h5f:
        h5f.create_dataset("X_train", data=X_train)
        h5f.create_dataset("X_test", data=X_test)
        h5f.create_dataset("X_val", data=X_val)

    print(X_train.shape, X_test.shape, X_val.shape)
    del X_train
    del X_test
    del X_val
    
    if signals_files:
        # read SIGNAL data
        for k, signal_file in enumerate(signals_files):
            f = h5py.File(folder + signal_file, "r")
            signal_data = f["Particles"][:, :, :-1]
            full_data = signal_data.reshape(signal_data.shape[0], input_shape)
            # nel = 4 * np.ones(full_data.shape[0])
            # nmu = 4 * np.ones(full_data.shape[0])
            # njet = 10 * np.ones(full_data.shape[0])
            # full_data = np.column_stack((full_data, nel, nmu, njet))
            """for i in range(full_data.shape[0]):
                for j in range(full_data.shape[1]):
                    if (j % 3 == 0) & (j <= 54):
                        if (
                            (full_data[i, j] == 0)
                            & (full_data[i, j + 1] == 0)
                            & (full_data[i, j + 2] == 0)
                        ):
                            full_data[i, j] = -9
                            full_data[i, j + 1] = -9
                            full_data[i, j + 2] = -9
                            # if (j >= 3) & (j <= 14):
                            #    full_data[i, 57] -= 1
                            # elif (j >= 15) & (j <= 26):
                            #    full_data[i, 58] -= 1
                            # else:
                            #    full_data[i, 59] -= 1
                    else:
                        continue
            """
            # print(full_data.shape)
            # del full_data, signal_data
            with h5py.File(output_signal_names[k] + "_dataset.h5", "w") as h5f2:
                h5f2.create_dataset("Data", data=full_data)
    return


# In[ ]:


create_datasets_dense(
    bkg_file="background_for_training.h5",
    output_bkg_name="BKG",
    signals_files=[
        "Ato4l_lepFilter_13TeV_filtered.h5",
        "hChToTauNu_13TeV_PU20_filtered.h5",
        "hToTauTau_13TeV_PU20_filtered.h5",
        "leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5",
    ],
    output_signal_names=[
        "Ato4l_lepFilter_13TeV",
        "hChToTauNu_13TeV_PU20",
        "hToTauTau_13TeV_PU20",
        "leptoquark_LOWMASS_lepFilter_13TeV",
    ],
)


# In[ ]:




