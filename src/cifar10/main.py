'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import json
import argparse

from models import *
from utils import progress_bar

info = {}
train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []
BATCH_SIZE = 128
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# Our model
# net,net_name = QudaRebuildNet(num_classes=10), 'QudaRebuildNet'
# SE-ResNet
# net,net_name = timm.create_model("seresnet18", num_classes=10), 'seresnet18'
net,net_name = timm.create_model("seresnet34", num_classes=10), 'sereesnet34'
# net,net_name = timm.create_model("seresnet50", num_classes=10), 'seresnet50'
# net,net_name = timm.create_model("seresnet101", num_classes=10), 'seresnet101'
# net,net_name = timm.create_model("seresnet152", num_classes=10), 'seresnet152'
print('Number of parameters(M):', sum(p.numel() for p in net.parameters()) / 1e6)
print(net_name)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(net.parameters(), lr=4e-4, weight_decay=5e-4)
# optimizer = optim.AdamW(net.parameters(), lr=4e-4)
optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    train_loss_history.append(train_loss / len(trainloader))
    train_acc_history.append(100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    test_loss_history.append(test_loss / len(testloader))
    test_acc_history.append(100.*correct/total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()

# Save the training information
info['epoch'] = epoch
info['batch_size'] = BATCH_SIZE
info['net'] = net_name
info['best_acc'] = best_acc
info['parameters'] = sum(p.numel() for p in net.parameters())
info['train_loss'] = train_loss_history
info['train_acc'] = train_acc_history
info['test_loss'] = test_loss_history
info['test_acc'] = test_acc_history
info['image_size'] = 32

# check the directory
if not os.path.isdir('info'):
    os.mkdir('info')
# save the information
with open(f'./info/{net_name}.json', 'w') as f:
    json.dump(info, f)