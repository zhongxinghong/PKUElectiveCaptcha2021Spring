#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: preprocess.py
# Created Date: 2021-03-05
# Author: Rabbit
# --------------------------------
# Copyright (c) 2021 Rabbit

import os
import re
import numpy as np
import cv2
from PIL import Image, ImageSequence, ImageFilter
from tqdm import tqdm
from multiprocessing.pool import Pool
from const import DATA_RAW_DIR, DATA_FRAME_DIR, DATA_CROP_DIR, DATA_BOOTSTRAP_DIR,\
    DATA_MANUALLY_LABELED_DIR

re_label = re.compile(r'^([2345678abcdefghklmnpqrstuvwxy]{4})$')
re_data_raw_filename = re.compile(r'^([a-zA-Z0-9]{4})=(\d+\-\d+)\.gif$')
re_data_bootstrap_filename = re.compile(r'^([a-zA-Z0-9]{4})_(right|wrong)_(\d+\-\d+)\.gif$')
re_data_manually_labeled_filename = re.compile(r'^([a-zA-Z0-9]{4})_to_([a-zA-Z0-9]{4})_(\d+\-\d+)\.gif$')

def morph_erode_cross(im, ksize, iterations):
    assert im.mode == "L"
    assert ksize % 2 == 1

    M0 = None
    Mt = np.array(im, dtype=np.uint8)
    w, h = im.size
    k = ksize >> 1

    for _ in range(iterations):
        M0 = Mt.copy()
        for yc in range(k, h - k):
            for xc in range(k, w - k):
                Mt[yc, xc] = min(
                    np.min(M0[yc, xc-k : xc+k+1]),
                    np.min(M0[yc-k : yc+k+1, xc]),
                )

    return Image.fromarray(Mt, im.mode)

def morph_dilate_cross(im, ksize, iterations):
    assert im.mode == "L"
    assert ksize % 2 == 1

    M0 = None
    Mt = np.array(im, dtype=np.uint8)
    w, h = im.size
    k = ksize >> 1

    for _ in range(iterations):
        M0 = Mt.copy()
        for yc in range(k, h - k):
            for xc in range(k, w - k):
                Mt[yc, xc] = max(
                    np.max(M0[yc, xc-k : xc+k+1]),
                    np.max(M0[yc-k : yc+k+1, xc]),
                )

    return Image.fromarray(Mt, im.mode)

def morph_erode_rect(im, ksize, iterations):
    assert im.mode == "L"
    assert ksize % 2 == 1

    M0 = None
    Mt = np.array(im, dtype=np.uint8)
    w, h = im.size
    k = ksize >> 1

    for _ in range(iterations):
        M0 = Mt.copy()
        for yc in range(k, h - k):
            for xc in range(k, w - k):
                Mt[yc, xc] = np.min(M0[yc-k : yc+k+1, xc-k : xc+k+1])

    return Image.fromarray(Mt, im.mode)

def morph_dilate_rect(im, ksize, iterations):
    assert im.mode == "L"
    assert ksize % 2 == 1

    M0 = None
    Mt = np.array(im, dtype=np.uint8)
    w, h = im.size
    k = ksize >> 1

    for _ in range(iterations):
        M0 = Mt.copy()
        for yc in range(k, h - k):
            for xc in range(k, w - k):
                Mt[yc, xc] = np.max(M0[yc-k : yc+k+1, xc-k : xc+k+1])

    return Image.fromarray(Mt, im.mode)

def denoise(im, ksize, threshold, iterations):
    assert im.mode == "L"
    assert ksize % 2 == 1

    M0 = None
    Mt = np.array(im, dtype=np.uint8)
    w, h = im.size
    k = ksize >> 1
    threshold = threshold * 0xff

    for _ in range(iterations):
        M0 = Mt.copy()
        for yc in range(k, h - k):
            for xc in range(k, w - k):
                if M0[yc, xc] > 0xf8:
                    continue
                if np.sum(M0[yc-k : yc+k+1, xc-k : xc+k+1]) >= threshold:
                    Mt[yc, xc] = 0xff

    return Image.fromarray(Mt, im.mode)

def median_filter(im, ksize):
    assert im.mode == "L"
    assert ksize % 2 == 1

    M0 = None
    Mt = np.array(im, dtype=np.uint8)
    w, h = im.size
    k = ksize >> 1

    M0 = Mt.copy()
    for yc in range(k, h - k):
        for xc in range(k, w - k):
            Mt[yc, xc] = np.median(M0[yc-k : yc+k+1, xc-k: xc+k+1]).astype(M0.dtype)

    return Image.fromarray(Mt, im.mode)

def _crop(im):
    assert im.mode == "1"

    M = 1 - np.array(im, dtype=np.uint8)
    S0 = M.sum(axis=0).cumsum()
    w, h = im.size
    k = CROP_WIDTH = im.height

    max_pos = -1
    max_sum = -1

    for j in range(w - k):
        s = S0[j + k] - S0[j]
        if s >= max_sum:
            max_pos = j
            max_sum = s

    return im.crop((max_pos, 0, max_pos + k, h))

def _main():

    for filename in tqdm(sorted(os.listdir(DATA_RAW_DIR))[:64]):
        mat = re_data_raw_filename.match(filename)
        assert mat is not None, filename

        label = mat.group(1).lower()
        serial = mat.group(2)

        dirname = os.path.join(DATA_FRAME_DIR, "%s_%s" % (label, serial))
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        filepath = os.path.join(DATA_RAW_DIR, filename)
        im = Image.open(filepath)
        w, h = im.size

        frames = ImageSequence.all_frames(im)
        assert len(frames) == 16, len(frames)

        for ix, fim in enumerate(frames):
            filepath = os.path.join(dirname, "%02d.png" % ix)
            fim.save(filepath)

        for i in range(4):
            j = 4 * i + 3

            if i == 0:
                fim = frames[j]

                fim = fim.convert('L')
                fim = morph_dilate_cross(fim, ksize=3, iterations=1)
                fim = morph_erode_cross(fim, ksize=3, iterations=1)
                fim = denoise(fim, ksize=5, threshold=20, iterations=2)
                fim = fim.filter(ImageFilter.EDGE_ENHANCE_MORE)
                fim = fim.convert('1')

            else:
                M1 = np.array(frames[j - 4], dtype=np.int32)
                M2 = np.array(frames[j], dtype=np.int32)
                fim = Image.fromarray(0xff - np.abs(M2 - M1).astype(np.uint8))

                fim = fim.convert('L')
                fim = morph_erode_cross(fim, ksize=3, iterations=1)
                fim = morph_dilate_cross(fim, ksize=3, iterations=1)
                fim = denoise(fim, ksize=5, threshold=20, iterations=1)
                fim = fim.filter(ImageFilter.EDGE_ENHANCE_MORE)
                fim = fim.convert('1')

            filepath = os.path.join(dirname, "c%d_%s_raw.png" % (i, label[i]))
            fim.save(filepath)

            fim = crop(fim)

            filepath = os.path.join(dirname, "c%d_%s.png" % (i, label[i]))
            fim.save(filepath)

            filepath = os.path.join(DATA_CROP_DIR, "%s_c%d_%s_%s.png" % (label, i, label[i], serial))
            fim.save(filepath)

        im.close()

        # print(filename)
        # break

def extract_c0_v1(M0):
    _, M_threshold = cv2.threshold(M0, 0, 0xff, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    M_close1 = cv2.morphologyEx(M_threshold, cv2.MORPH_CLOSE, kernel1, iterations=1)
    M_blur1 = cv2.medianBlur(M_close1, 3)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    M_close2 = cv2.morphologyEx(M_threshold, cv2.MORPH_CLOSE, kernel2, iterations=1)
    M_blur2 = cv2.medianBlur(M_close2, 3)

    if np.sum(M_blur1[:, :40]) <= np.sum(M_blur2[:, :40]):
        Mt = M_blur1
    else:
        Mt = M_blur2
    return Mt

def extract_c0_v2(M0, M_merge):
    _, M_mask = cv2.threshold(M_merge, 0, 0xff, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    M_mask[:, 40:] = 0xff

    M_darken = M0.copy()
    M_darken[M_mask == 0x00] >>= 1

    return extract_c0_v1(M_darken)

def extract_c123(M0, M0_last):
    M_subtract = cv2.subtract(M0_last, M0)

    _, M_threshold = cv2.threshold(M_subtract, 0, 0xff, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    M_opened = cv2.morphologyEx(M_threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    Mt = cv2.medianBlur(M_opened, 3)
    return Mt

def _crop(Mt):
    S0 = (0xff - Mt).sum(axis=0).cumsum()
    h, w = Mt.shape
    k = h

    max_pos = -1
    max_sum = -1

    for i in range(w - k):
        s = S0[i + k] - S0[i]
        if s >= max_sum:
            max_pos = i
            max_sum = s

    Mt = Mt[:, max_pos : max_pos + k]
    return Mt

def crop(Mt, first):
    char_width = 32
    captcha_size = 52

    assert Mt.shape[0] == captcha_size

    M_blur = cv2.medianBlur(Mt, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11))
    M_opened = cv2.morphologyEx(M_blur, cv2.MORPH_OPEN, kernel, iterations=3)

    w = 50 if first else Mt.shape[1]

    S0 = (0xff - M_opened).sum(axis=0).cumsum()
    k = char_width

    max_sum = -1
    max_pos = -1

    for i in range(k, w):
        s = S0[i] - S0[i - k]
        if s > max_sum:
            max_sum = s
            max_pos = i

    w = max_pos - k - (captcha_size - k) // 2

    if w >= 0 and w + captcha_size <= Mt.shape[1]:
        return Mt[:, w : w+captcha_size]

    M_cropped = 0xff - np.zeros((captcha_size, captcha_size), np.uint8)
    if w < 0:
        M_cropped[:, -w : captcha_size] = Mt[:, : captcha_size+w]
    else:
        M_cropped[:, : Mt.shape[1]-w] = Mt[:, w :]
    return M_cropped


def image_generator():

    for filename in sorted(os.listdir(DATA_RAW_DIR)):
        filepath = os.path.join(DATA_RAW_DIR, filename)

        mat = re_data_raw_filename.match(filename)
        assert mat is not None, filename

        label = mat.group(1).lower()
        serial = mat.group(2)

        yield filepath, label, serial

    for filename in sorted(os.listdir(DATA_BOOTSTRAP_DIR)):
        filepath = os.path.join(DATA_BOOTSTRAP_DIR, filename)

        mat = re_data_bootstrap_filename.match(filename)
        assert mat is not None, filename

        label = mat.group(1).lower()
        valid_res = mat.group(2)
        serial = mat.group(3)

        if valid_res != 'right':
            continue

        yield filepath, label, serial

    for filename in sorted(os.listdir(DATA_MANUALLY_LABELED_DIR)):
        filepath = os.path.join(DATA_MANUALLY_LABELED_DIR, filename)

        mat = re_data_manually_labeled_filename.match(filename)
        assert mat is not None, filename

        label = mat.group(2).lower()
        serial = mat.group(3)

        yield filepath, label, serial


def task_preprocess_single(args):
    filepath, label, serial = args

    dirname = os.path.join(DATA_FRAME_DIR, "%s_%s" % (label, serial))
    if os.path.exists(dirname):
        return

    os.mkdir(dirname)

    im = Image.open(filepath)
    assert im.format == "GIF" and im.n_frames == 16

    # frames = ImageSequence.all_frames(im)

    # for ix, fim in enumerate(frames):
    #     filepath = os.path.join(dirname, "%02d.png" % ix)
    #     fim.save(filepath)

    w, h = im.size

    M0_list = []
    M0_last = None
    M_mask = np.zeros((h, w), dtype=np.uint8)
    M_merge = np.full((h, w), 0xff, dtype=np.uint8)

    for ix in (3, 7, 11, 15):
        im.seek(ix)
        M0 = np.array(im.convert('RGB'))
        M0 = cv2.cvtColor(M0, cv2.COLOR_RGB2GRAY)
        M0_list.append(M0)
        M_merge = np.min([M_merge, M0], axis=0)

    for cix, M0 in enumerate(M0_list):
        first = (M0_last is None)

        if first:
            # Mt = extract_c0_v1(M0)
            Mt = extract_c0_v2(M0, M_merge)
        else:
            Mt = extract_c123(M0, M0_last)

        M0_last = M0

        Mt_inv = 0xff - Mt
        Mt = 0xff - np.bitwise_and(Mt_inv, 0xff - M_mask)
        M_mask = np.bitwise_or(M_mask, Mt_inv)

        # filepath = os.path.join(dirname, "c%d_%s_raw.png" % (cix, label[cix]))
        # cv2.imwrite(filepath, Mt)

        Mt = crop(Mt, first)

        # filepath = os.path.join(dirname, "c%d_%s.png" % (cix, label[cix]))
        # cv2.imwrite(filepath, Mt)

        filepath = os.path.join(DATA_CROP_DIR, "%s_c%d_%s_%s.png" % (label, cix, label[cix], serial))
        cv2.imwrite(filepath, Mt)

    im.close()


def main():

    task_list = []

    for filepath, label, serial in image_generator():
        assert re_label.match(label) is not None, filepath

        task = (filepath, label, serial)
        task_list.append(task)

    with Pool(processes=4) as p:
        for _ in tqdm(p.imap(task_preprocess_single, task_list), total=len(task_list)):
            pass


if __name__ == "__main__":
    main()
