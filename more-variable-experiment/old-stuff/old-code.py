"""



for i in range(full_data.shape[0]):
    for j in range(full_data.shape[1]):
        if (j % 3 == 0) & (j <= 54):
            if (
                (full_data[i, j] == 0)
                & (full_data[i, j + 1] == 0)
                & (full_data[i, j + 2] == 0)
            ):
                full_data[i, j] = -99
                full_data[i, j + 1] = -99
                full_data[i, j + 2] = -99
        else:
            continue
fig, ax = plt.subplots(figsize=(20, 20))
fig.tight_layout()
for i in range(57):
    ax = plt.subplot(12, 5, i + 1)
    min = full_data[:, i].min()
    max = full_data[:, i].max()
    bins = 50
    xlabel = ""
    ymax = 20000
    if (min == -99) & (i % 3 == 0):
        min = 0
        bins = 50
        xlabel = "p_T"
    if (min == -99) & (max < 5):
        min = -max
        bins = 20
        ymax = 200000
    ax.hist(full_data[:, i], bins=bins, range=[min, max], log=True, histtype="step")
    ax.set_yticks(ticks=[10, 1000, 10000, ymax])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
plt.show()

"""
