import torch.nn as nn
import timm


class CovidGeneralModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=4, in_chans=3)

    def forward(self, x):
        output = self.model(x)
        return output
