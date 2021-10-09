from src.utils.predict_utils_classification import predict_utils_classification as predictor
from train_advanced import parse_args

# test ------ predictor
PTH = '.\checkpoint\DA\s_0_1_t2\cnn_features_1d_1009-173233\\299-0.2222-best_model.pth'
fault_name = ['inner', 'outer', 'ball', 'holder', 'normal']
len_val_sample = 9
# len_test_sample = 200
args = parse_args()

predictor = predictor(args, PTH)
predictor.setup()

correct_rate_overall, correct_rate = predictor.predict(len(fault_name), len_val_sample)

# correct_rate_overall, correct_rate = predictor.predict(len(fault_name), len_test_sample)
# for i in range(len_test_sample):
#     label_output = predictor.predict(len(fault_name),len_test_sample)
#
# fei code
# for i in range(len(fault_name)):
#     for j in range(len_val_sample):
#         label_output = predictor.predict(len(fault_name),len_val_sample)

