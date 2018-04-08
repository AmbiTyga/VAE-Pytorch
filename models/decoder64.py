import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
                nn.Linear(200, 4096), 
                nn.ReLU(True), 
                nn.Linear(4096, 16 * 16 * 64),
                nn.ReLU(True),
                )


        self.features = nn.Sequential(
                #input is 16 x 16
                nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2, bias=False),
                nn.ReLU(True),

                # input is 32x 32
                nn.ConvTranspose2d(64, 3, kernel_size=4, padding=1, stride=2, bias=False),
                # TODO: see if there is a need for a sigmoid here
                # nn.ReLU(True),
                nn.Sigmoid()
                )

    def forward(self, x):
        out = self.linear(x)
        out = out.view(-1, 64, 16, 16)
        out = self.features(out)
        return out
