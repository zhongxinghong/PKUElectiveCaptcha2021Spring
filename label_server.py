#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: label_server.py
# Created Date: 2021-03-09
# Author: Rabbit
# --------------------------------
# Copyright (c) 2021 Rabbit

import os
import re
import base64
import shutil
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, url_for, redirect, request, jsonify
from preprocess import extract_c0, extract_c123, crop
from const import WEB_DIR, DATA_BOOTSTRAP_DIR, DATA_MANUALLY_LABELED_DIR

re_label = re.compile(r'^([2345678abcdefghklmnpqrstuvwxy]{4})$')
re_serial = re.compile(r'^\d{13}-\d{1,3}$')
re_data_bootstrap_filename = re.compile(r'^([a-zA-Z0-9]{4})_(right|wrong)_(\d+\-\d+)\.gif$')
re_data_labeled_filename = re.compile(r'^([a-zA-Z0-9]{4})_(\d+\-\d+)\.gif$')

def _image_generator():
    while True:
        found = False

        for filename in os.listdir(DATA_BOOTSTRAP_DIR):
            filepath = os.path.join(DATA_BOOTSTRAP_DIR, filename)

            mat = re_data_bootstrap_filename.match(filename)
            assert mat is not None, filename

            label = mat.group(1).lower()
            valid_res = mat.group(2)
            serial = mat.group(3)

            if valid_res != 'wrong':
                continue

            found = True

            while os.path.exists(filepath):
                yield filepath, label, serial

        if not found:
            break


app = Flask(__name__, template_folder=WEB_DIR)
im_gen = _image_generator()


@app.template_filter('img2url')
def img2url(M):
    assert len(M.shape) in [2, 3], M.shape

    if len(M.shape) == 2:
        im = Image.fromarray(M, 'L')
    else:
        im = Image.fromarray(M, 'RGB')

    bio = BytesIO()
    im.save(bio, 'PNG')
    data = bio.getvalue()
    bio.close()
    im.close()

    return 'data:image/png;base64,' + base64.b64encode(data).decode()


@app.route('/', methods=['GET'])
def root():
    filepath, label, serial = next(im_gen)

    im = Image.open(filepath)

    assert im.format == "GIF", im.format
    assert im.n_frames == 16, im.n_frames

    N = 52
    w, h = im.size

    images = []

    M0_old = None
    M_mask = np.zeros((h, w), np.uint8)

    for ix in (3, 7, 11, 15):
        first = (M0_old is None)
        im.seek(ix)

        im_row = []

        M0 = np.array(im.convert("RGB"))
        im_row.append(M0)

        M0 = cv2.cvtColor(M0, cv2.COLOR_RGB2GRAY)

        if first:
            im_row.append(M0)
        else:
            Mdiff = cv2.subtract(M0_old, M0)
            im_row.append(0xff - Mdiff)

        if first:
            Mt = extract_c0(M0)
        else:
            Mt = extract_c123(M0, M0_old)

        M0_old = M0

        Mt_inv = 0xff - Mt
        Mt = 0xff - np.bitwise_and(Mt_inv, 0xff - M_mask)
        M_mask = np.bitwise_or(M_mask, Mt_inv)
        im_row.append(Mt)

        Mt = crop(Mt, first)
        im_row.append(Mt)

        images.append(im_row)

    im.close()

    ctx = {
        "images": images,
        "label": label,
        "serial": serial,
    }
    return render_template("label_server.html", **ctx)


@app.route('/submit', methods=["POST"])
def submit():
    wrong_label = request.form["wrong_label"].lower()
    right_label = request.form["right_label"].lower()
    serial = request.form["serial"]

    assert re_label.match(wrong_label) is not None, wrong_label

    mat = re_label.match(right_label)
    if mat is None:
        return jsonify(errcode=1, errmsg="标签格式错误")

    mat = re_serial.match(serial)
    if mat is None:
        return jsonify(errcode=2, errmsg="序列号格式错误")

    if right_label == wrong_label:
        return jsonify(errcode=3, errmsg="与原标签相同")

    src_filepath = os.path.join(DATA_BOOTSTRAP_DIR, "%s_wrong_%s.gif" % (wrong_label, serial))
    dst_filepath = os.path.join(DATA_MANUALLY_LABELED_DIR, "%s_to_%s_%s.gif" % (wrong_label, right_label, serial))

    shutil.move(src_filepath, dst_filepath)

    return jsonify(errcode=0, errmsg="success")


if __name__ == "__main__":
    app.run("127.0.0.1", 9000, debug=True)
