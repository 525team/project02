import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.datasets.SequenceDatasets import dataset
from src.datasets.sequence_aug import *
from tqdm import tqdm

#Digital data was collected at 12,000 samples per second
signal_size = 1024

dataname= {0:["t_s_inner0.mat", "t_s_outer0.mat","t_s_ball0.mat", "t_s_holder0.mat", "t_s_normal0.mat"],
           1:["t_s_inner1.mat", "t_s_outer1.mat", "t_s_ball1.mat", "t_s_holder1.mat", "t_s_normal1.mat"],
           2:["un_supp_s_0.mat", "un_supp_s_1.mat", "un_supp_s_2.mat", "un_supp_s_3.mat", "un_supp_s_4.mat"]}

datasetname = ["condition0", "condition1", "condition2"]
axis = ["slot"]
label = [i for i in range(0, 5)]


def get_files(root, N):
    '''
    This function is used to get normalized data and corresponding label.
    root:The location of the data set
    N: list, 其他代码中的 source_N / target_N example: N={0,1}, 即工况序列
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root, datasetname[N[k]], dataname[N[k]][n])
            # 加载数据
            data1, lab1 = data_load(path1, dataname[N[k]][n], label=label[n])
            # k是域标签,lab1代表的是类标签
            lab1 = k*100+np.array(lab1)
            lab1 = lab1.tolist()
            data += data1
            lab += lab1
    return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to 加载一种工况下一种故障的_DE_time数据.
    comment：这里的axisname命名不准确，实际发挥dataname，数据文件名的作用。
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = loadmat(filename)['Data']
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class competition(object):
    num_classes = 5
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        print('self.source_N',self.source_N)
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val

