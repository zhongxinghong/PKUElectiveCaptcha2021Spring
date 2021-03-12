#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: config.py
# Created Date: 2021-03-05
# Author: Rabbit
# --------------------------------
# Copyright (c) 2021 Rabbit

from configparser import RawConfigParser
from const import CONFIG_INI

class Config(object):

    def __init__(self):
        self._config = RawConfigParser()
        self._config.read(CONFIG_INI, encoding='utf-8-sig')

    @property
    def iaaa_id(self):
        return self._config.get("user", "id")

    @property
    def iaaa_password(self):
        return self._config.get("user", "password")

    @property
    def is_dual_degree(self):
        return self._config.get("user", "dual_degree")
