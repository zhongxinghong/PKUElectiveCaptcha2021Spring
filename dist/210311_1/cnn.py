#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: cnn.py
# Created Date: 2021-03-06
# Author: Rabbit
# --------------------------------
# Copyright (c) 2021 Rabbit

import os
import sys
import re
import time
import shutil
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from tqdm import tqdm
from const import DATA_CROP_DIR, DATA_TRAIN_DIR, DATA_TEST_DIR, MODEL_DIR, LOG_DIR

CAPTCHA_LABELS = '2345678abcdefghklmnpqrstuvwxy'
CAPTCHA_SIZE = 52

re_dataset_filename = re.compile(r'^([a-z0-9]{4})_c(\d)_([a-z0-9])_(\d+\-\d+)\.png$')
logfp = None

class CaptchaDataset(D.Dataset):

    def __init__(self, folder, train_set):
        N = CAPTCHA_SIZE
        captcha_labels_set = frozenset(CAPTCHA_LABELS)
        captcha_labels_indices = np.empty(0x80, dtype=np.uint8)

        if train_set:
            Mrot_list = [
                cv2.getRotationMatrix2D((N // 2, N // 2), 7.5 * i, 1.0)
                for i in range(-4, 5)
            ]

        for ix, c in enumerate(CAPTCHA_LABELS):
            captcha_labels_indices[ord(c)] = ix

        Xlist = []
        ylist = []

        for filename in tqdm(sorted(os.listdir(folder))):
            ## TEST ONLY !
            # sep = 0.01 if train_set else 0.01 * 4
            # if np.random.random() > sep:
            #     continue

            mat = re_dataset_filename.match(filename)
            assert mat is not None, filename

            filepath = os.path.join(folder, filename)
            X = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            y = mat.group(3)

            assert X.shape == (N, N), X.shape
            assert y in captcha_labels_set, y

            y = captcha_labels_indices[ord(y)]

            if train_set:
                for Mrot in Mrot_list:
                    Xrot = cv2.warpAffine(X, Mrot, X.shape)
                    Xlist.append(Xrot)
                    ylist.append(y)
            else:
                Xlist.append(X)
                ylist.append(y)

        self.X = np.array(Xlist, dtype=np.float32).reshape(-1, 1, N, N)
        self.y = np.array(ylist, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]


class CaptchaCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, len(CAPTCHA_LABELS)) # 29

    def forward(self, x):
        x = self.bn0(x)         # batch*1*52*52
        x = F.relu(x)
        x = self.conv1(x)       # batch*16*50*50
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)       # batch*32*48*48
        x = self.bn2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)  # batch*32*24*24
        x = self.conv3(x)       # batch*64*22*22
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)       # batch*128*20*20
        x = self.bn4(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)  # batch*128*10*10
        x = self.conv5(x)       # batch*256*8*8
        x = self.bn5(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)  # batch*256*4*4
        x = self.conv6(x)       # batch*512*2*2
        x = self.bn6(x)
        x = F.relu(x)
        x = torch.flatten(x, 1) # batch*2048
        x = self.fc1(x)         # batch*512
        x = F.relu(x)
        x = self.fc2(x)         # batch*128
        x = F.relu(x)
        x = self.fc3(x)         # batch*29
        x = F.log_softmax(x, dim=1)
        return x


def cnn_train(model, train_loader, optimizer, epoch):
    log_interval = max(1, int(len(train_loader) * 0.05))

    model.train()

    for ix, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if ix % log_interval == 0:
            line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                ix * len(data),
                len(train_loader.sampler),
                100.0 * ix / len(train_loader),
                loss.item()
            )
            print(line, file=sys.stdout, flush=True)
            print(line, file=logfp, flush=True)


def cnn_test(model, test_loader, epoch):
    test_loss = 0
    correct = 0
    confusion_matrix = np.zeros((len(CAPTCHA_LABELS), len(CAPTCHA_LABELS)), dtype=np.int)

    model.eval()

    with torch.no_grad():
        for Xlist, ylist in test_loader:
            if torch.cuda.is_available():
                Xlist = Xlist.cuda()
                ylist = ylist.cuda()

            output = model(Xlist)
            test_loss += F.nll_loss(output, ylist).item() / len(test_loader.sampler)
            ypred = output.argmax(dim=1, keepdim=True)
            correct += ypred.eq(ylist.view_as(ypred)).sum().item()
            for t, p in zip(ylist.view(-1), ypred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    line = '\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss,
        correct,
        len(test_loader.sampler),
        100.0 * correct / len(test_loader.sampler)
    )
    print(line, file=sys.stdout, flush=True)
    print(line, file=logfp, flush=True)

    df = pd.DataFrame(
        data=confusion_matrix,
        index=list(CAPTCHA_LABELS),
        columns=list(CAPTCHA_LABELS),
    )
    filepath = os.path.join(LOG_DIR, "confusion_matrix.epoch_%d.csv" % epoch)
    df.to_csv(filepath)


def build_train_test_set(train_size):
    shutil.rmtree(DATA_TRAIN_DIR)
    os.mkdir(DATA_TRAIN_DIR)
    shutil.rmtree(DATA_TEST_DIR)
    os.mkdir(DATA_TEST_DIR)

    train_cnt = 0
    test_cnt = 0

    for filename in tqdm(sorted(os.listdir(DATA_CROP_DIR))):
        src = os.path.join(DATA_CROP_DIR, filename)

        if random.random() <= train_size:
            dst = os.path.join(DATA_TRAIN_DIR, filename)
            train_cnt += 1
        else:
            dst = os.path.join(DATA_TEST_DIR, filename)
            test_cnt += 1

        shutil.copyfile(src, dst)

    print("Train set size: %d" % train_cnt)
    print("Test set size: %d" % test_cnt)
    print("Actual train_size: %.3f" % (train_cnt / (train_cnt + test_cnt)))


def train_model():
    global logfp

    logfp = open(os.path.join(LOG_DIR, "console.%d.log" % int(time.time() * 1000)), 'w')

    batch_size = 64
    epochs = 15

    train_dataset = CaptchaDataset(DATA_TRAIN_DIR, train_set=True)
    test_dataset = CaptchaDataset(DATA_TEST_DIR, train_set=False)

    train_loader = D.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = CaptchaCNN()

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, epochs+1):
        t1 = time.time()

        cnn_train(model, train_loader, optimizer, epoch)
        cnn_test(model, test_loader, epoch)

        t2 = time.time()

        line = 'Time cost: {} seconds\n'.format(int(t2 - t1))
        print(line, file=sys.stdout, flush=True)
        print(line, file=logfp, flush=True)

        scheduler.step()

        model_file = os.path.join(MODEL_DIR, "cnn.epoch_%02d.pt" % epoch)
        torch.save(model.state_dict(), model_file)

    logfp.close()


def determine_labels():

    labels = set()

    for filename in os.listdir(DATA_CROP_DIR):
        mat = re_dataset_filename.match(filename)
        assert mat is not None, filename
        ch = mat.group(3)
        labels.add(ch)

    labels = ''.join(sorted(labels))
    print(len(labels), labels)

    assert labels == '2345678abcdefghklmnpqrstuvwxy'


def main():
    # determine_labels()
    # build_train_test_set(train_size=0.8)
    train_model()


if __name__ == "__main__":
    main()
