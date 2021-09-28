import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.datasets.SequenceDatasets import dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


signal_size = 1024

dataname= {0:["t_s_inner0.mat", "t_s_outer0.mat","t_s_ball0.mat", "t_s_holder0.mat", "t_s_normal0.mat"],
           1:["t_s_inner1.mat", "t_s_outer1.mat", "t_s_ball1.mat", "t_s_holder1.mat", "t_s_normal1.mat"],
           2:["un_supp_s_0.mat", "un_supp_s_1.mat", "un_supp_s_2.mat", "un_supp_s_3.mat", "un_supp_s_4.mat"]}

datasetname = ["condition0", "condition1", "condition2"]
axis = ["slot"]
label = [i for i in range(0, 5)]



root = 'D:\Data\condition'
def data_load(filename, axisname, label):
    fl = loadmat(filename)["Data"]
    fl = fl.reshape(-1, )

    start, end = 0, signal_size
    x = fl[start:end]
    x = np.fft.fft(x)
    x = np.abs(x) / len(x)
    x = x[range(int(x.shape[0] / 2))]
    x = x.reshape(-1, 1)

    return x

name = ['0', 'Inner race', 'Outer race', 'Ball', 'Holder', 'Normal']

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 6):
    x = data_load(os.path.join(root, datasetname[0], dataname[0][i - 1]), dataname[0][i - 1], label=None)
    ax = fig.add_subplot(2, 3, i)
    ax.plot(x)
    ax.set_title(name[i])
    ax.text(0.5, 0.5, str((2, 3, i)),
            fontsize=18, ha='center')

plt.show()