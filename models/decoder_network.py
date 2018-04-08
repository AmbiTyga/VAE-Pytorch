import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class Decoder(nn.Module):
    def __init__(self, z_dim = 200):
        super().__init__()
        self.linear = nn.Sequential(
                nn.Linear(z_dim, 4096), 
                nn.ReLU(True), 
                nn.Linear(4096, 7 * 7 * 512),
                nn.ReLU(True), 
                )

        self.features = nn.Sequential(
                ## input 7x7
                nn.ConvTranspose2d(512, 512, kernel_size = 4, padding = 1, stride = 2, bias = False),
                nn.ConvTranspose2d(512, 256, kernel_size = 3, padding = 1, stride = 1, bias = False),
                nn.ReLU(True),


                ## input 14 x 14
                nn.ConvTranspose2d(256, 256, kernel_size = 4, padding = 1, stride = 2, bias = False),
                nn.ConvTranspose2d(256, 256, kernel_size = 3, padding = 1, stride = 1, bias = False),
                nn.ReLU(True),


                ## input 28 x 28
                nn.ConvTranspose2d(256, 256, kernel_size = 4, padding = 1, stride = 2, bias = False),
                nn.ConvTranspose2d(256, 128, kernel_size = 3, padding = 1, stride = 1, bias = False),
                nn.ReLU(True),

                # input 56 x 56
                nn.ConvTranspose2d(128, 128, kernel_size = 4, padding = 1, stride = 2, bias = False),
                nn.ConvTranspose2d(128, 64, kernel_size = 3, padding = 1, stride = 1, bias = False),
                nn.ReLU(True),
                

                # input 224 x 224
                nn.ConvTranspose2d(64, 64 , kernel_size = 4, padding = 1, stride = 2, bias = False),
                nn.ConvTranspose2d(64, 3, kernel_size = 3, padding = 1, stride = 1, bias = False),
                nn.Sigmoid(),
            )

        self.activation = F.relu


    def forward(self, latent):
        x = self.linear(latent)
        x = x.view(-1, 512, 7, 7)
        x = self.features(x)
        return x

