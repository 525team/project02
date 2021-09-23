import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

read_training_sample = np.load('../2021_manufacturing_competition_data/training_samples.npy', allow_pickle=True).item()
# training_sample = read_training_sample[()]
# read_training_sample.item()
condition0_ball = read_training_sample['condition0']['ball']
print(read_training_sample['condition0'])

plt.plot(condition0_ball)
plt.show()