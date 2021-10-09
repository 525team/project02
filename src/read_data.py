import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io

# --- training_samples.npy ---
# read_training_sample = np.load('../2021_manufacturing_competition_data/training_samples.npy', allow_pickle=True).item()
# condition0_ball = read_training_sample['condition0']['ball']
# condition0_holder = read_training_sample['condition0']['holder']
# condition0_inner = read_training_sample['condition0']['inner']
# condition0_normal = read_training_sample['condition0']['normal']
# condition0_outer = read_training_sample['condition0']['outer']

# #
# io.savemat('train_s.mat', {'Data': read_training_sample})
# io.savemat('train_s_ball0.mat', {'Data': condition0_ball})
# io.savemat('train_s_holder0.mat', {'Data': condition0_holder})
# io.savemat('train_s_inner0.mat', {'Data': condition0_inner})
# io.savemat('train_s_normal0.mat', {'Data': condition0_normal})
# io.savemat('train_s_outer0.mat', {'Data': condition0_outer})
#
# condition1_ball = read_training_sample['condition1']['ball']
# condition1_holder = read_training_sample['condition1']['holder']
# condition1_inner = read_training_sample['condition1']['inner']
# condition1_normal = read_training_sample['condition1']['normal']
# condition1_outer = read_training_sample['condition1']['outer']
# #
# io.savemat('train_s_ball1.mat', {'Data': condition1_ball})
# io.savemat('train_s_holder1.mat', {'Data': condition1_holder})
# io.savemat('train_s_inner1.mat', {'Data': condition1_inner})
# io.savemat('train_s_normal1.mat', {'Data': condition1_normal})
# io.savemat('train_s_outer1.mat', {'Data': condition1_outer})

# print(read_training_sample['condition0'])
# plt.plot(condition0_ball)
# plt.show()



# --- final_test.npy ---  200*1024
# read_final_test = np.load('../2021_manufacturing_competition_data/final_test.npy', allow_pickle=True)
# names = locals()
# for i in range(200):
#     names['final_test_' + str(i)] = read_final_test[i]
# #print(final_test_0)



# --- unlabeled_support_samples.npy ---  250*1024
# read_unlabeled_support_samples = np.load('../2021_manufacturing_competition_data/unlabeled_support_samples.npy', allow_pickle=True)
# io.savemat('unlabeled_support_samples.mat', {'Data':read_unlabeled_support_samples})
# names = locals()
# for i in range(0,50):
#     names['unl_support_samples_' + str(i)] = read_unlabeled_support_samples[i]
#     unl_support_samples_0 = np.append(unl_support_samples_0, read_unlabeled_support_samples[i])
#     io.savemat('unl_supp_s_0.mat',{'Data': unl_support_samples_0})
# for i in range(50,100):
#     names['unl_support_samples_' + str(i)] = read_unlabeled_support_samples[i]
#     unl_support_samples_1 = np.append(unl_support_samples_1, read_unlabeled_support_samples[i])
#     io.savemat('unl_supp_s_1.mat',{'Data': unl_support_samples_1})
# for i in range(100,150):
#     names['unl_support_samples_' + str(i)] = read_unlabeled_support_samples[i]
#     unl_support_samples_2 = np.append(unl_support_samples_2, read_unlabeled_support_samples[i])
#     io.savemat('unl_supp_s_2.mat',{'Data': unl_support_samples_2})
# for i in range(150,200):
#     names['unl_support_samples_' + str(i)] = read_unlabeled_support_samples[i]
#     unl_support_samples_3 = np.append(unl_support_samples_3, read_unlabeled_support_samples[i])
#     io.savemat('unl_supp_s_3.mat',{'Data': unl_support_samples_3})
# for i in range(200,250):
#     names['unl_support_samples_' + str(i)] = read_unlabeled_support_samples[i]
#     unl_support_samples_4 = np.append(unl_support_samples_4, read_unlabeled_support_samples[i])
#     io.savemat('unl_supp_s_4.mat',{'Data': unl_support_samples_4})

# print(unl_support_samples_0)

# --- validation_samples.npy ---  9*1024
read_validation_samples = np.load('../2021_manufacturing_competition_data/validation_samples.npy', allow_pickle=True).item()
validation_samples_ball = read_validation_samples['ball']
validation_samples_holder = read_validation_samples['holder']
validation_samples_inner = read_validation_samples['inner']
validation_samples_normal = read_validation_samples['normal']
validation_samples_outer = read_validation_samples['outer']

#
print(1)
# io.savemat("val_s_ball.mat", {'Data': validation_samples_ball})
# io.savemat("val_s_holder.mat", {'Data': validation_samples_holder})
# io.savemat("val_s_inner.mat", {'Data': validation_samples_inner})
# io.savemat("val_s_normal.mat", {'Data': validation_samples_normal})
# io.savemat("val_s_outer.mat", {'Data': validation_samples_outer})
