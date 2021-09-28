import torch
import os
from scipy import io
from models import cnn_1d

# PATH is the model path
PATH = '.\checkpoint\DA\s_0_1_t2\cnn_features_1d_0927-190704\8-0.3333-best_model.pth'
trained_model = cnn_1d.CNN()
len_val_sample = 9
validation_sample_dir = '.\data_files\\validation_samples'
validation_sample_name = ['val_s_ball.mat', 'val_s_holder.mat', 'val_s_inner.mat', 'val_s_normal.mat', 'val_s_outer.mat']

def predict(PATH, trained_model, len_val_sample):
    # trained_model.load_state_dict(torch.load(PATH), strict=False)
    trained_model.load_state_dict(torch.load(PATH))
    trained_model.eval()

    predict_correct = 0

    with torch.no_grad():
        for i in range(len(validation_sample_name)):
            val_data = io.loadmat(os.path.join(validation_sample_dir, validation_sample_name[i]))
            print(1)
            for j in range(len_val_sample):
                # 输入数据转化为输入神经网络的数据格式
                # 输入的数据是1024数据模式
                # val_sample =
                # 预测数据
                # output = trained_model(val_sample)
                # 下面的表述有问题
                # if output.data == correct_label[i]:

                predict_correct += 1

        # prediction_accuracy = predict_correct / len_val_sample
        prediction_accuracy = 0
    return prediction_accuracy


predict(PATH, trained_model, len_val_sample)