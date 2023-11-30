import json
import math
import os
import pickle
import sys
import yaml

import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
import cv2

from pathlib import Path


from seggpt_inference import run_inference


def run_seg_inference(input_predictions):
    if os.path.isdir(input_predictions) and os.listdir(input_predictions):
        files = os.listdir(input_predictions)
        jpg_files = [f for f in files if f.endswith('.jpg')]
        for file in jpg_files:
            run_inference('cpu', 'data/stuble', input_image=os.path.join(input_predictions, file),
                          input_video=None, prompt_image='/Users/samkoshythomas/Desktop/Projects/SEG-GPT/Painter/SegGPT/stubble-train/train/images/window_1280_11520_jpg.rf.7f11354471a9177766d29040a5c20ba1.jpg', prompt_target='/Users/samkoshythomas/Desktop/Projects/SEG-GPT/Painter/SegGPT/window_1280_11520_jpg.rf.7f11354471a9177766d29040a5c20ba1.jpg')
            print(file)


def main():
    run_seg_inference('./data/predictions')

if __name__ == "__main__":
    main()
