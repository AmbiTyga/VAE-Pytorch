import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.features = nn.Sequential(
                # input 224x224
                nn.Conv2d(3, 64, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(64, 64, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 112 x 112
                nn.Conv2d(64, 128, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(128, 128, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # 56 x 56
                nn.Conv2d(128, 256, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(256, 256, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(256, 256, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                
                # 28 x 28
                nn.Conv2d(256, 512, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # 14 x 14
                nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # output is going to be now 7 x 7 x 512
            )


        self.classifier = nn.Sequential(
                nn.Linear( 512 * 7 * 7, 4096), 
                nn.ReLU(True), 
                nn.Dropout(), 
                nn.Linear(4096, 4096),
                # nn.ReLU(True), 
                # nn.Dropout(True), 
                # nn.Linear(4096, z_dim)
            )
                # nn.ReLU(True), 
                # nn.Dropout(),
                # nn.Lin

    def forward(self, x):
        ''' function to get the represeantation'''
        x = self.features(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x
