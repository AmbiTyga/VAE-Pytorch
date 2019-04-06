from __future__ import print_function
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pdb
# from dataset import FrameDataset
import h5py
import numpy as np
from dataloader import GraphVAEDataLoader
import ipdb
from models.encoder64 import Encoder
from models.decoder64 import Decoder


parser = argparse.ArgumentParser(description='VAE Pytorch Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type = float, default = 1e-3)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
z_dim = 200

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

hf_file_path = os.path.expanduser('~/datasets/mnist/final_dataset_with_mask.h5')
hf = h5py.File(hf_file_path, 'r')
dataset = GraphVAEDataLoader(hf, Tensor, return_mask=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mean = nn.Linear(4096, z_dim)
        self.logvar = nn.Linear(4096, z_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.encoder(x))
        return self.mean(h1), self.logvar(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.decoder(z)
        return h3

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # z = z.view(args.batch_size, z_dim, 1, 1)
        return self.decode(z), mu, logvar

model = VAE()
if use_cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr = args.lr)

# Reconstruction + KL divergence losses summed over all elements and batch
# A, B, C = 224, 224, 3
# A, B, C = 64, 64, 3
A, B, C = 64, 64, 1
image_size = A * B * C
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

device = torch.device("cuda")

def train(epoch):
    model.train()
    train_loss = 0
    # for batch_idx, (data) in enumerate(train_loader):
    for batch_idx, (data) in enumerate(dataloader):
        input_batch = next(iter(dataloader))
        batch_data_tensor = input_batch['image'].to(device)
        batch_mask_tensor = input_batch['mask'].to(device)
        batch_label_tensor = \
                input_batch['tuple'].type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch_data_tensor)
        loss = loss_function(recon_batch, batch_data_tensor, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader),
                100. * batch_idx / len(dataloader),
                loss.data[0] / len(data)))
            sample = Variable(torch.randn(bs, z_dim))
            if use_cuda:
                sample = sample.cuda()
            sample = sample.view(bs, z_dim)
            sample = model.decode(sample).cpu()
            save_image(sample.data.view(bs, C, A, B),
                       'results/sample_' + 
                       str(epoch) + 'count' + str(batch_idx) + '.png')
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataloader)))

bs = 10

for epoch in range(1, args.epochs + 1):
    train(epoch)
    # test(epoch)
    sample = Variable(torch.randn(bs, z_dim))
    if use_cuda:
        sample = sample.cuda()
    sample = sample.view(bs, z_dim)
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(bs, C, A, B),
               'results/sample_' + str(epoch) + '.png')
