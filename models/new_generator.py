import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.linear_network import Embedder 
import pdb

class Generator(nn.Module):
    def __init__(self, z_dim = 300, init_weights = True):
        super().__init__()
        self.activation = F.relu
        # self.affine_layers = nn.Sequential(
                # nn.Linear(300, 1024), 
                # nn.ReLU(True), 
                # nn.Dropout(),
                # nn.Linear(1024, 4096), 
                # nn.ReLU(True), 
                # nn.Dropout(),
                # nn.Linear(4096, 512 * 7 * 7),
                # )
        self.conv_layers = nn.Sequential(
                ## input 1x1
                nn.ConvTranspose2d(z_dim, 512, kernel_size = 7, padding = 0, stride = 1, bias = False),
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
        self.embedder = Embedder(input_size = 44688)
        if init_weights:
            self._initialize_weights()


    def forward(self, embedding, noise):
        embedding = embedding.view(1, -1)
        z = self.embedder(embedding)
        z = z.view(1, 200) # hard coded right now.. do check later for the actual values
        # noise = noise.view(1, 100)
        latent = torch.cat((z, noise), 1) # possibility of facing an error here. 

        # x = self.affine_layers(latent)
        # x = x.view(1, 512, 7, 7)
        # pdb.set_trace()
        x = latent.view(1, 300, 1, 1)
        x = self.conv_layers(x)
        # pdb.set_trace()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
