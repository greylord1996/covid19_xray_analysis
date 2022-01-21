import torch.nn as nn
import timm


class CovidEffnetModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=4, in_chans=3)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.dropout_layer = nn.Dropout(0.5)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 4)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        # features = self.dropout_layer(features)
        pooled_features = self.pooling(features).view(bs, -1)
        pooled_features = self.dropout_layer(pooled_features)
        output = self.fc(pooled_features)
        return output
