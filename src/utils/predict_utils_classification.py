import logging
import warnings
import torch
from torch import nn
import numpy as np
import os
from scipy import io
from src import datasets
import src.models as models

# new version of load validation data
read_validation_samples = np.load('../2021_manufacturing_competition_data/validation_samples.npy', allow_pickle=True).item()
validation_samples_ball = read_validation_samples['ball']
validation_samples_holder = read_validation_samples['holder']
validation_samples_inner = read_validation_samples['inner']
validation_samples_normal = read_validation_samples['normal']
validation_samples_outer = read_validation_samples['outer']
fault_name = ['inner', 'outer', 'ball', 'holder', 'normal']
len_val_sample = 9

# construct new code
class predict_utils_classification():
    def __init__(self, args, model_pth):
        self.args = args
        self.model_pth = model_pth

    def setup(self):
        args = self.args
        # load model

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        Dataset = getattr(datasets, args.data_name)
        self.model = getattr(models, args.model_name)(args.pretrained)
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)
        self.model_all.load_state_dict(torch.load(self.model_pth))
        self.model_all.eval()


    def predict(self, fault_num, val_sample_num_per_fault):
        # predict data
        correct = []
        with torch.no_grad():
            for i in range(fault_num):
                correct_i = 0
                for j in range(val_sample_num_per_fault):
                    inputs = torch.from_numpy(read_validation_samples[fault_name[i]][j])
                    inputs = inputs.to(self.device)
                    label_output = self.model_all(inputs)
                    if label_output == i:
                        correct_i += 1
                correct.append(correct_i)

            correct_rate = correct / val_sample_num_per_fault
            correct_rate_overall = sum(correct_rate) / fault_num
        return correct_rate_overall, correct_rate




