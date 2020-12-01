#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.utils.py
Standard helper methods
"""
import os
import logging
import coloredlogs

log = logging.getLogger("utils")
coloredlogs.install(level="info", logger=log)


def confirm_directory(directory: os.PathLike):
    if not os.path.isdir(directory):
        log.info(f"The given directory '{directory}' does not exist. Creating it!")
        os.makedirs(directory)
    else:
        log.info(f"The directory '{directory}' exists. Ready to use.")
