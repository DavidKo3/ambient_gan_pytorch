from __future__ import print_function

import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


import argparse
import dcgan as model
print("Sdfsf")


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--niter', type=int, default=25, help='number of epoch to train for')
parser.add_argument('--manualseed', type=int, help='manual seed')


opt  = parser.parse_args()

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

print("Random seed : ", opt.manualseed)
random.seed(opt.manualseed)
torch.manual_seed(opt.manualseed)

if torch.cuda.is_available() :
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# data_loader
img_size = 64
isCrop = False
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

fixed_z = torch.randn((5*5, 100)).view(-1, 100, 1, 1)   # fixed noise
fixed_z = Variable(fixed_z.cuda(), volatile=True)


# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20


# data_loader
img_size = 64
transform = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('data' ,train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

# network
G = model.G(128)
D = model.D(128)


G.weight_init(G)
D.weight_init(D)
# D.apply(model.weight_init(D))
#
# G.weight_init(mean=0.0, std=0.02)
# D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()


# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    for x_, _ in train_loader:
        # train discriminator D
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real (lod(D(x))
        D.zero_grad()

        mini_batch = x_.size()[0] # len of train_loader
        # print("mini_batch :", mini_batch)
        y_real = torch.ones(mini_batch)
        y_fake = torch.zeros(mini_batch)  # [mini_batch]

        x_, y_real, y_fake = Variable(x_.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
        D_result = D(x_).squeeze()  # D(x)
        print(D(x_).shape)
        D_real_loss = BCE_loss(D_result, y_real)  # log(D(x))


        # train with fake (log(1 - D(G(z))
        z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1) # [x, 100] -> [x, 100, 1 , 1]
        z = Variable(z.cuda())
        G_result = G(z)

        D_result = D(G_result).squeeze() # D(G(z))
        D_fake_loss = BCE_loss(D_result, y_fake)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss   # log(D(x)) + log(1- D(G(z))

        D_train_loss.backward()
        D_optimizer.step()


        




        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # train generator G












































