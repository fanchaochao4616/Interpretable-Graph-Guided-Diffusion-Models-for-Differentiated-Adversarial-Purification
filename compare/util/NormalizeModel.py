import torch
from utils import show_images
class NormalizedModel(torch.nn.Module):
    def __init__(self, model,raw_to_clf):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.raw_to_clf = raw_to_clf
    def forward(self, x):
        # 在前向传播时标准化输入
        x = self.raw_to_clf(x)
        return self.model(x)