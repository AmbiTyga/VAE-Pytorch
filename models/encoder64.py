import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                # input: 64
                nn.Conv2d(3, 64, kernel_size=4, padding=1, stride=2, bias=False), 
                nn.ReLU(True), 
                # input: 32
                nn.Conv2d(64, 64, kernel_size=4, padding=1, stride=2, bias=False),
                nn.ReLU(True),
                # input:16
                # nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2, bias=False), 
                # nn.ReLU(True),
                # # input : 8
                # nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False), 
                # nn.ReLU(True),

                )
        self.linear = nn.Sequential(
                nn.Linear( 16 * 16 * 64, 4096), 
                nn.ReLU(True),
                # nn.Linear(4096, 200), 
                # nn.Sigmoid()
                )

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, 16 * 16 * 64)
        out = self.linear(out)
        return out
