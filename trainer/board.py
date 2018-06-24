#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 21:21:06 2018

@author: justo
"""

import keras.callbacks.TensorBoard as tb

def crearConfigTensorBoard(pathToLogs):
    return tb(log_dir=pathToLogs,
              histogram_freq=0,
              write_graph=True,
              write_images=False)