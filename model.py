import torch
import torch.nn as nn
import torchvision.models as models

class MILModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder.fc = nn.Identity()

        # Separate attention for each biomarker
        self.attention_IDH1 = self._make_attention()
        self.attention_ATRX = self._make_attention()
        self.attention_P53 = self._make_attention()

        # Classifier
        self.classifier = nn.Linear(512, 3)

    def _make_attention(self):
        return nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, return_attention=False):
        feats = self.encoder(x)  # (N, 512)

        # Compute attention per biomarker
        A_idh1 = torch.softmax(self.attention_IDH1(feats), dim=0)
        A_atrx = torch.softmax(self.attention_ATRX(feats), dim=0)
        A_p53  = torch.softmax(self.attention_P53(feats), dim=0)

        # Aggregate features
        M_idh1 = torch.sum(A_idh1 * feats, dim=0)
        M_atrx = torch.sum(A_atrx * feats, dim=0)
        M_p53  = torch.sum(A_p53 * feats, dim=0)

        # Stack features
        M = torch.stack([M_idh1, M_atrx, M_p53], dim=0)

        # Classifier
        out = self.classifier(M.mean(dim=0))

        if return_attention:
            return out, [A_idh1, A_atrx, A_p53]

        return out