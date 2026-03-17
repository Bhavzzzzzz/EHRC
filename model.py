import torch
import torch.nn as nn
import torchvision.models as models

class MILModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()

        self.attention = nn.Sequential(
            nn.Linear(512,128),
            nn.Tanh(),
            nn.Linear(128,1)
        )

        self.classifier = nn.Linear(512,3)

    def forward(self, x):
        feats = self.encoder(x)  # (N,512)

        A = self.attention(feats)
        A = torch.softmax(A, dim=0)

        M = torch.sum(A * feats, dim=0)

        out = self.classifier(M)
        return out