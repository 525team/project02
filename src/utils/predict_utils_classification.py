import torch
import numpy as np
import os
from scipy import io
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
        self.model = getattr(models, args.model_name)(args.pretrained)
        self.model.load_state_dict(torch.load(self.model_pth))
        self.model.eval()

    def predict(self, fault_num, val_sample_num_per_fault):
        # predict data
        correct = []
        with torch.no_grad():
            for i in range(fault_num):
                correct_i = 0
                for j in range(val_sample_num_per_fault):
                    # 下面这行代码我不确定
                    label_output = self.model(read_validation_samples[fault_name[i]][j])
                    if label_output == i:
                        correct_i += 1
                correct.append(correct_i)

            correct_rate = correct / val_sample_num_per_fault
            correct_rate_overall = sum(correct_rate) / fault_num
        return correct_rate_overall, correct_rate




