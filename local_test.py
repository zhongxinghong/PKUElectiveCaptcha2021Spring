#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: validate.py
# Created Date: 2021-03-09
# Author: Rabbit
# --------------------------------
# Copyright (c) 2021 Rabbit

import os
import random
import torch
from tqdm import tqdm
from cnn import CaptchaCNN
from preprocess import image_generator
from bootstrap import predict_captcha
from const import MODEL_DIR

def main():
    model = CaptchaCNN()

    model_file = os.path.join(MODEL_DIR, 'cnn.20210309.1.pt')
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    right_cnt = 0
    wrong_cnt = 0

    for filepath, target_label, serial in tqdm(image_generator()):

        if random.random() > 0.015:
            continue

        with open(filepath, 'rb') as fp:
            im_data = fp.read()

        predict_label = predict_captcha(im_data, model)

        if predict_label == target_label:
            right_cnt += 1
        else:
            wrong_cnt += 1

    print("%d / %d, accuracy: %.4f" % (
        right_cnt,
        right_cnt + wrong_cnt,
        right_cnt / (right_cnt + wrong_cnt),
    ))


if __name__ == "__main__":
    main()
