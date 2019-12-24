import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm
from util import cal_loss, IOStream
from Qconv import QConv1d, QConv2d
from model import PointNet, DGCNN
from data import ModelNet40
from torch.utils.data import DataLoader

train_loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=8,
                          batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=8,
                         batch_size=32, shuffle=True, drop_last=False)

model = PointNet().cuda()
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
io = IOStream('checkpoints/run.log')
for epoch in range(100):
    #################
    #Train
    #################
    train_loss = 0.0
    count = 0.0
    model.train()
    train_pred = []
    train_true = []
    for data, label in tqdm(train_loader):
        data = data.permute(0,2,1)
        data, label = data.cuda(), label.squeeze().cuda()
        batch_size = data.size()[0]
        opt.zero_grad()
        logits = model(data)
        loss = criterion(logits, label)
        loss.backward()
        opt.step()
        preds = logits.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,train_loss*1.0/count,
                                                                             metrics.accuracy_score(train_true, train_pred),
                                                                             metrics.balanced_accuracy_score(train_true, train_pred))

    io.cprint(outstr)

    ####################
    # Test
    ####################
    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    for data, label in tqdm(test_loader):
        data = data.permute(0,2,1)
        data, label = data.cuda(), label.squeeze().cuda()
        batch_size = data.size()[0]
        logits = model(data)
        loss = criterion(logits, label)
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                          test_loss*1.0/count,
                                                                          test_acc,
                                                                          avg_per_class_acc)
    io.cprint(outstr)
