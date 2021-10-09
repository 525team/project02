import numpy as np
import os
# task: split source domain data (condition 0) in size 5*9*1024, and store the data in the source_domain_test.npy in the data directory
read_training_samples = np.load('../2021_manufacturing_competition_data/training_samples.npy', allow_pickle=True).item()
# read_validation_samples = read_training_samples['condition0']
condition0_ball = read_training_samples['condition0']['ball']
condition0_holder = read_training_samples['condition0']['holder']
condition0_inner = read_training_samples['condition0']['inner']
condition0_normal = read_training_samples['condition0']['normal']
condition0_outer = read_training_samples['condition0']['outer']

condition0_ball = np.array(np.array_split(condition0_ball[0:9216], 9))
condition0_holder = np.array(np.array_split(condition0_holder[0:9216], 9))
condition0_inner = np.array(np.array_split(condition0_inner[0:9216], 9))
condition0_normal = np.array(np.array_split(condition0_normal[0:9216], 9))
condition0_outer = np.array(np.array_split(condition0_outer[0:9216], 9))
#
training_part = {'ball': condition0_ball, 'holder': condition0_holder, 'inner': condition0_inner, 'normal': condition0_normal, 'outer': condition0_outer}
os.chdir(r'D:\python\project02\2021_manufacturing_competition_data')
np.save('training_part.npy', training_part)
# print(type(training_part))
print(training_part)
