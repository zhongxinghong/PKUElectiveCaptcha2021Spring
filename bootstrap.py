#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: bootstrap.py
# Created Date: 2021-03-07
# Author: Rabbit
# --------------------------------
# Copyright (c) 2021 Rabbit

import os
import re
import time
import random
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import torch
from requests import Session
from config import Config
from const import DATA_BOOTSTRAP_DIR, MODEL_DIR, TEMP_DIR
from cnn import CaptchaCNN, CAPTCHA_LABELS, CAPTCHA_SIZE
from preprocess import extract_c0_v2, extract_c123, crop

cfg = Config()
re_sida_sttp = re.compile(r'\?sida=(\S+?)&sttp=(?:bzx|bfx)')


def elective_login(s):
    s.cookies.clear()

    ## elective homepage

    r = s.get(
        url="https://elective.pku.edu.cn/"
    )
    r.raise_for_status()

    ## iaaa oauthlogin

    time.sleep(0.5)

    r = s.post(
        url="https://iaaa.pku.edu.cn/iaaa/oauthlogin.do",
        data={
            "appid": "syllabus",
            "userName": cfg.iaaa_id,
            "password": cfg.iaaa_password,
            "randCode": "",
            "smsCode": "",
            "otpCode": "",
            "redirUrl": "http://elective.pku.edu.cn:80/elective2008/ssoLogin.do",
        }
    )
    r.raise_for_status()

    token = r.json()["token"]

    ## elective ssologin

    r = s.get(
        url="https://elective.pku.edu.cn/elective2008/ssoLogin.do",
        params={
            "_rand": str(random.random()),
            "token": token,
        }
    )
    r.raise_for_status()

    if cfg.is_dual_degree:
        sida = re_sida_sttp.search(r.text).group(1)

    ## elective ssologin (dual degree)

    if cfg.is_dual_degree:
        time.sleep(0.5)

        r = s.get(
            url="https://elective.pku.edu.cn/elective2008/ssoLogin.do",
            params={
                "sida": sida,
                "sttp": "bzx",
            }
        )
        r.raise_for_status()

    ## elective HelpController

    r = s.get(
        url="https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/help/HelpController.jpf"
    )
    r.raise_for_status()

    ## elective SupplyCancel

    time.sleep(0.5)

    r = s.get(
        url="https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/SupplyCancel.do",
        headers={
            "Referer": "https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/help/HelpController.jpf"
        }
    )
    r.raise_for_status()


def predict_captcha(im_data, model):
    fp = BytesIO(im_data)
    im = Image.open(fp)

    assert im.format == "GIF", im.format
    assert im.n_frames == 16, im.n_frames

    N = CAPTCHA_SIZE
    w, h = im.size

    Xlist = []
    ylist = []

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
            Mt = extract_c0_v2(M0, M_merge)
        else:
            Mt = extract_c123(M0, M0_last)

        M0_last = M0

        Mt_inv = 0xff - Mt
        Mt = 0xff - np.bitwise_and(Mt_inv, 0xff - M_mask)
        M_mask = np.bitwise_or(M_mask, Mt_inv)

        Mt = crop(Mt, first)

        Xlist.append(Mt)

    im.close()
    fp.close()

    Xlist = np.array(Xlist, dtype=np.float32).reshape(-1, 1, N, N)
    ylist = model(torch.from_numpy(Xlist))

    code = ''.join( CAPTCHA_LABELS[ix] for ix in torch.argmax(ylist, dim=1) )
    return code


def main():
    model = CaptchaCNN()

    model_file = os.path.join(MODEL_DIR, 'cnn.20210311.1.pt')
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    right_cnt = 0
    wrong_cnt = 0
    error_cnt = 0

    s = Session()
    s.headers.update({
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36",
        "Upgrade-Insecure-Requests": "1"
    })

    elective_login(s)

    while True:

        ## elective DrawServlet

        r = s.get(
            url="https://elective.pku.edu.cn/elective2008/DrawServlet",
            params={
                "Rand": str(random.random() * 10000),
            },
            headers={
                "Referer": "https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/SupplyCancel.do"
            }
        )
        r.raise_for_status()

        im_data = r.content

        if not im_data.startswith(b'GIF89a'):
            error_cnt += 1
            print("Bad Captcha")
            time.sleep(0.5)
            elective_login(s)
            continue

        code = predict_captcha(im_data, model)

        r = s.post(
            url="https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/validate.do",
            data={
                "xh": cfg.iaaa_id,
                "validCode": code,
            },
            headers={
                "Referer": "https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/SupplyCancel.do"
            }
        )
        r.raise_for_status()

        try:
            rjson = r.json()
        except Exception as e:
            if "异常刷新" in r.text or "请重新登录" in r.text:
                error_cnt += 1
                time.sleep(0.5)
                elective_login(s)
                continue
            else:
                print(r.text)
            raise e

        if rjson.get('valid') != '2':
            wrong_cnt += 1
            valid_res = 'wrong'
        else:
            right_cnt += 1
            valid_res = 'right'

        print("%s %s, %d / %d, accuracy: %.4f, error count: %d" % (
            code,
            valid_res,
            right_cnt,
            right_cnt + wrong_cnt,
            right_cnt / (right_cnt + wrong_cnt),
            error_cnt
        ))

        serial = '%d-%d' % (time.time() * 1000, random.random() * 1000)
        filename = '%s_%s_%s.gif' % (code, valid_res, serial)
        filepath = os.path.join(DATA_BOOTSTRAP_DIR, filename)

        with open(filepath, 'wb') as fp:
            fp.write(im_data)

        time.sleep(2.0 + 2.0 * random.random())


if __name__ == "__main__":
    main()
