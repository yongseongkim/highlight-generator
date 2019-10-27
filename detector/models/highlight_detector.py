import torch
import torch.nn as nn
import torchvision.models as models


class HighlightDetector(nn.Module):

    def __init__(self, features_dim, hidden_dim, layer_dim, resnet_pretrained):
        super(HighlightDetector, self).__init__()
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # 7장 중 하이라이트 장면에 포함되는 만큼 점수를 준다. ex) 10장 중 5장이 하이라이트 장면이면 output = 5
        self.lstm = nn.LSTM(features_dim, hidden_dim, layer_dim)
        self.resnet = models.resnet18(pretrained=resnet_pretrained)
        self.resnet.fc = nn.Linear(512, features_dim)
        self.h2o = nn.Linear(hidden_dim, 1)

    def forward(self, imgs):
        batch_size, timesteps, C, H, W = imgs.size()
        resnet_in = imgs.view(batch_size * timesteps, C, H, W)
        resnet_out = self.resnet(resnet_in)
        features = resnet_out.view(batch_size, timesteps, -1)
        out, (hn, cn) = self.lstm(features)
        out = self.h2o(out[:, -1, :])
        return out
