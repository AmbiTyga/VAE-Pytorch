from __future__ import print_function
import argparse
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
from dataloader import GetDataset
from models.encoder_network import Encoder
from models.decoder_network import Decoder
from models.discriminator import Discriminator




parser = argparse.ArgumentParser(description='VAE Pytorch Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--discrim-lr', type=float, default=1e-4)
parser.add_argument('--lr', type = float, default = 1e-4)
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

filenames = '/home/tanya/VAE-Pytorch/datasets/cropped_224/*.png'
# filenames = '/home/tanya/dcgan/cropped_64/*.png'
# dataset = GetDataset(filenames)
dataset = GetDataset(filenames, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
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
        z = z.view(args.batch_size, z_dim)
        return self.decode(z), mu, logvar

model = VAE()
discrim = Discriminator()
if use_cuda:
    model.cuda()
    discrim.cuda()
optimizer = optim.Adam(model.parameters(), lr = args.lr)
discrim_optimizer = optim.Adam(discrim.parameters(), lr = args.discrim_lr)

# Reconstruction + KL divergence losses summed over all elements and batch
A, B, C = 224, 224, 3
image_size = A * B * C
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x.view(-1, image_size), x.view(-1, image_size), size_average=False)
    # BCE = F.binary_cross_entropy(recon_x, x)
    ## define the GAN loss here
    label = Variable(torch.ones(args.batch_size).type(Tensor))
    ## TODO: see if its a variable
    discrim_output = discrim(recon_x)
    BCE = F.binary_cross_entropy(discrim_output, label)


    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    discrim_loss = 0
    # for batch_idx, (data) in enumerate(train_loader):
    for batch_idx, (data) in enumerate(dataloader):
        data = Variable(data)
        if use_cuda:
            data = data.cuda()

        '''Add the discirm stuff here'''
        discrim_optimizer.zero_grad()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        recon_batch_clone = recon_batch.clone()
        loss = loss_function(recon_batch, data, mu, logvar) # and we use the discriminator there to find the loss
        # loss.backward()

        # loss wrt the discriminator....
        label_fake = Variable(torch.zeros(args.batch_size).type(Tensor))
        discrim_label_fake = discrim(recon_batch_clone.detach())
        fake_err = F.binary_cross_entropy(discrim_label_fake, label_fake)
        fake_err.backward()

        label_real = Variable(torch.ones(args.batch_size).type(Tensor))
        discrim_label_real = discrim(data)
        real_err = F.binary_cross_entropy(discrim_label_real, label_real)
        real_err.backward()

        err = fake_err + real_err
        discrim_loss += err.data[0]
       
        # loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        discrim_optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t discrimLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader),
                100. * batch_idx / len(dataloader),
                loss.data[0] / len(data), err.data[0]/len(data)))
            sample = Variable(torch.randn(bs, z_dim))
            if use_cuda:
                sample = sample.cuda()
            sample = sample.view(bs, z_dim)
            sample = model.decode(sample).cpu()
            save_image(sample.data.view(bs, 3, 224, 224),
                       'results/sample_' + str(epoch) + 'count' + str(batch_idx) + '.png')

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataloader)))

bs = 10

for epoch in range(1, args.epochs + 1):
    print('Learning Rate: %f' % (args.lr))
    train(epoch)
    # test(epoch)
    sample = Variable(torch.randn(bs, z_dim))
    if use_cuda:
        sample = sample.cuda()
    # sample = sample.view(bs, z_dim, 1, 1)
    sample = sample.view(bs, z_dim)
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(bs, 3, 224, 224),
               'results/sample_' + str(epoch) + '.png')
