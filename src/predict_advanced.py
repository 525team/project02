from src.utils.predict_utils_classification import predict_utils_classification as predictor
from train_advanced import parse_args

# test ------ predictor
PTH = '.\checkpoint\DA\s_0_1_t2\cnn_features_1d_0927-190704\8-0.3333-best_model.pth'
fault_name = ['inner', 'outer', 'ball', 'holder', 'normal']
len_val_sample = 9
args = parse_args()

predictor = predictor(args, PTH)
predictor.setup()
correct_rate_overall, correct_rate = predictor.predict(fault_name, len_val_sample)