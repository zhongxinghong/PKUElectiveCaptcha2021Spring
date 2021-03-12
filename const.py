#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: const.py
# Created Date: 2021-03-05
# Author: Rabbit
# --------------------------------
# Copyright (c) 2021 Rabbit

import os

ROOT_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(ROOT_DIR, "tmp/")
DATA_DIR = os.path.join(ROOT_DIR, "data/")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw/")
DATA_FRAME_DIR = os.path.join(DATA_DIR, "frame/")
DATA_CROP_DIR = os.path.join(DATA_DIR, "crop/")
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "train/")
DATA_TEST_DIR = os.path.join(DATA_DIR, "test/")
DATA_BOOTSTRAP_DIR = os.path.join(DATA_DIR, "bootstrap/")
DATA_MANUALLY_LABELED_DIR = os.path.join(DATA_DIR, "manually_labeled/")
DATA_TRASH_DIR = os.path.join(DATA_DIR, "trash/")
MODEL_DIR = os.path.join(ROOT_DIR, "model/")
LOG_DIR = os.path.join(ROOT_DIR, "log/")
WEB_DIR = os.path.join(ROOT_DIR, "web/")
CONFIG_INI = os.path.join(ROOT_DIR, "config.ini")

def _mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

_mkdir(TEMP_DIR)
_mkdir(DATA_DIR)
_mkdir(DATA_RAW_DIR)
_mkdir(DATA_FRAME_DIR)
_mkdir(DATA_CROP_DIR)
_mkdir(DATA_TRAIN_DIR)
_mkdir(DATA_TEST_DIR)
_mkdir(DATA_BOOTSTRAP_DIR)
_mkdir(DATA_MANUALLY_LABELED_DIR)
_mkdir(DATA_TRASH_DIR)
_mkdir(MODEL_DIR)
_mkdir(LOG_DIR)

del _mkdir
