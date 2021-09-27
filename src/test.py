from torch import nn


model = nn.load('9-0.2745-best_model.pth')
model.eval()
load(val_s.mat)