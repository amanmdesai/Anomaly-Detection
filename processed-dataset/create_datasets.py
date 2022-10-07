import argparse

import h5py
import numpy as np
from sklearn.model_selection import train_test_split


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
        full_data = file["Particles"][:, :, :-1]
        np.random.shuffle(full_data)
        if events:
            full_data = full_data[:events, :, :]

    # define training, test and validation datasets
    X_train, X_test = train_test_split(full_data, test_size=test_size, shuffle=True)
    X_train, X_val = train_test_split(X_train, test_size=val_size)

    del full_data

    # flatten the data for model input
    X_train = X_train.reshape(X_train.shape[0], input_shape)
    X_test = X_test.reshape(X_test.shape[0], input_shape)
    X_val = X_val.reshape(X_val.shape[0], input_shape)

    with h5py.File(output_bkg_name + "_dataset.h5", "w") as h5f:
        h5f.create_dataset("X_train", data=X_train)
        h5f.create_dataset("X_test", data=X_test)
        h5f.create_dataset("X_val", data=X_val)

    if signals_files:
        # read SIGNAL data
        for i, signal_file in enumerate(signals_files):
            f = h5py.File(folder + signal_file, "r")
            signal_data = f["Particles"][:, :, :-1]
            signal_data = signal_data.reshape(signal_data.shape[0], input_shape)
            with h5py.File(output_signal_names[i] + "_dataset.h5", "w") as h5f2:
                h5f2.create_dataset("Data", data=signal_data)
    return


def create_datasets_convolutional(
    bkg_file,
    output_bkg_name,
    signals_files,
    output_signal_names,
    events=None,
    test_size=0.2,
    val_size=0.2,
    input_shape=57,
):

    # read BACKGROUND data
    with h5py.File(bkg_file, "r") as file:
        full_data = file["Particles"][:, :, :-1]
        np.random.shuffle(full_data)
        if events:
            full_data = full_data[:events, :, :]

    # define training, test and validation datasets
    X_train, X_test = train_test_split(full_data, test_size=test_size, shuffle=True)
    X_train, X_val = train_test_split(X_train, test_size=val_size)

    del full_data

    with h5py.File(output_bkg_name + "_dataset.h5", "w") as h5f:
        h5f.create_dataset("X_train", data=X_train)
        h5f.create_dataset("X_test", data=X_test)
        h5f.create_dataset("X_val", data=X_val)

    if signals_files:
        # read SIGNAL data
        for i, signal_file in enumerate(signals_files):
            f = h5py.File(signal_file, "r")
            signal_data = f["Particles"][:, :, :-1]
            with h5py.File(output_signal_names[i] + "_dataset.h5", "w") as h5f2:
                h5f2.create_dataset("Data", data=signal_data)

    return


#if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--bkg_file', type=str)
    parser.add_argument('--output_bkg_name', type=str)
    parser.add_argument('--signals_files', type=str, action='append')
    parser.add_argument('--output_signal_names', type=str, action='append')
    parser.add_argument('--events', type=int, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--input_shape', type=int, default=57)
    args = parser.parse_args()
    #create_datasets_dense(**vars(args))
    
    create_datasets_dense(
        bkg_file="background_for_training.h5",
        output_bkg_name="BKG",
        signals_files=[
            "Ato4l_lepFilter_13TeV.h5",
            "hChToTauNu_13TeV_PU20.h5",
            "hToTauTau_13TeV_PU20.h5",
            "leptoquark_LOWMASS_lepFilter_13TeV.h5",
        ],
        output_signal_names=[
            "Ato4l_lepFilter_13TeV",
            "hChToTauNu_13TeV_PU20",
            "hToTauTau_13TeV_PU20",
            "leptoquark_LOWMASS_lepFilter_13TeV",
        ],
        events=1000000,
    )
    """
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

