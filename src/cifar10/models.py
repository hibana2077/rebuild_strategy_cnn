import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Rebuild(nn.Module):
    def __init__(self):
        super(Rebuild, self).__init__()
        self.rebuild_A = nn.Sequential(
            nn.LayerNorm(32),
            nn.Conv2d(3, 32, 3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64),
        )# (B, 4, 4, 4)
        self.rebuild_B = nn.Sequential(
            nn.Conv2d(4, 256, 3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
        )# (B, 8, 8, 8)

    def forward(self, x):
        x = self.rebuild_A(x)
        x = x.view(x.size(0), 4, 4, 4)
        x = self.rebuild_B(x)
        return x
    
class QudaRebuildNet(nn.Module):
    def __init__(self,num_classes=101):
        super(QudaRebuildNet, self).__init__()
        self.rebuild_a = Rebuild()
        self.rebuild_b = Rebuild()
        self.rebuild_c = Rebuild()
        self.rebuild_d = Rebuild()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x_a = self.rebuild_a(x)
        x_b = self.rebuild_b(x)
        x_c = self.rebuild_c(x)
        x_d = self.rebuild_d(x)
        x = torch.cat([x_a, x_b, x_c, x_d], dim=1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    model = QudaRebuildNet()
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    print(f"Parameter count(M): {sum(p.numel() for p in model.parameters()) / 1e6}")
    print(f"Output shape: {logits.shape}")