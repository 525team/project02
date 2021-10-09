import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import pywt


read_training_sample = np.load('../2021_manufacturing_competition_data/training_samples.npy', allow_pickle=True).item()
# training_sample = read_training_sample[()]
# read_training_sample.item()
condition0_ball = read_training_sample['condition0']['ball']
print(read_training_sample['condition0'])

#傅里叶分析，FFT，频域
fft_ball0 = fft(condition0_ball)
fft_ball0_real = fft_ball0.real   #获取实数
fft_ball0_imag = fft_ball0.imag   #获取虚数

yf=abs(fft(fft_ball0))                # 取绝对值
yf1=abs(fft(fft_ball0))/len(condition0_ball)           #归一化处理
yf2 = yf1[range(int(len(condition0_ball)/2))]  #由于对称性，只取一半区间

xf = np.arange(len(fft_ball0))        # 频率
xf1 = xf
xf2 = xf[range(int(len(condition0_ball)/2))]  #取一半区间
plt.subplot(221)
plt.plot(condition0_ball)
plt.title('Original wave')

plt.subplot(222)
plt.plot(xf,yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表

plt.subplot(223)
plt.plot(xf1,yf1,'g')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

plt.subplot(224)
plt.plot(xf2,yf2,'b')
plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')

#plt.plot(condition0_ball)
plt.show()