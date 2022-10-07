#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')
#plt.rcParams["figure.dpi"] = 200


# In[2]:


def read_h5py(filename="filename"):
    with h5py.File(filename, "r") as file:
        input_file = file["Particles"][:, :, :-1]
        full_data = np.array(input_file).reshape(input_file.shape[0], 57)
        del input_file
        for i in range(full_data.shape[0]):
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
                else:
                    continue
        print(full_data.shape)
        return full_data


# In[ ]:


sm_bkg = read_h5py("background_for_training.h5")
print('1 done')
a24l = read_h5py("Ato4l_lepFilter_13TeV_filtered.h5")
print('2 done')
htaunu = read_h5py("hChToTauNu_13TeV_PU20_filtered.h5")
print('3 done')
htautau = read_h5py("hToTauTau_13TeV_PU20_filtered.h5")
print('4 done')
lepquark = read_h5py("leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5")
print('5 done')

# In[ ]:


def plot_mc_data():
    fig, ax = plt.subplots(figsize=(45, 45))
    fig.tight_layout()
    for i in range(57):
        ax = plt.subplot(10, 6, i + 1)
        min = np.array(
            [
                sm_bkg[:, i].min(),
                a24l[:, i].min(),
                htaunu[:, i].min(),
                htautau[:, i].min(),
                lepquark[:, i].min(),
            ]
        ).max()
        max = np.array(
            [
                sm_bkg[:, i].max(),
                a24l[:, i].max(),
                htaunu[:, i].max(),
                htautau[:, i].max(),
                lepquark[:, i].max(),
            ]
        ).max()
        bins = 50
        xlabel = ""
        ymax = 100000
        if i == 0:
            min = 0
            bins = 50
            xlabel = r"$p_T$"
        if i == 1:
            min = -max
            bins = 20
            ymax = 200000
            xlabel = r"$\eta$"
        if i == 2:
            min = -max
            bins = 20
            ymax = 200000
            xlabel = r"$\eta$"
        if (min == -9) & (i % 3 == 0):
            min = 0
            bins = 50
            xlabel = r"$p_T$"
        if (min == -9) & (max < 5):
            min = -max
            bins = 20
            ymax = 200000
            if (i - 1) % 3 == 0:
                xlabel = r"$\eta$"
            else:
                xlabel = r"$\phi$"
        ax.hist(
            sm_bkg[:, i],
            bins=bins,
            range=[min, max],
            log=True,
            histtype="step",
            label="SM",
        )
        ax.hist(
            a24l[:, i],
            bins=bins,
            range=[min, max],
            log=True,
            histtype="step",
            label=r"$A \rightarrow 4l$",
        )
        ax.hist(
            htaunu[:, i],
            bins=bins,
            range=[min, max],
            log=True,
            histtype="step",
            label=r"$H \rightarrow \tau \nu$",
        )
        ax.hist(
            htautau[:, i],
            bins=bins,
            range=[min, max],
            log=True,
            histtype="step",
            label=r"$H \pm \rightarrow \tau \tau$",
        )
        ax.hist(
            lepquark[:, i],
            bins=bins,
            range=[min, max],
            log=True,
            histtype="step",
            label=r"LQ $\rightarrow b \tau$",
        )
        ax.set_yticks(ticks=[10, 100, 1000, 10000, ymax])
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("Counts", fontsize=15)
        plt.legend()
    plt.savefig("input-variable-adc.pdf", dpi=350, format="pdf", bbox_inches="tight")
    return


# In[ ]:


plot_mc_data()

