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


import supervision as sv
from autodistill_seggpt import SegGPT, FewShotOntology

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model features\n")
        sys.exit(1)
    input_predictions = sys.argv[1]
 
        
    
    supervision_dataset = sv.DetectionDataset.from_yolo(
        images_directory_path="/Users/samkoshythomas/Desktop/Research-Project-DS/AIML/WeedDetection/Sam-Work/stubble-3/train/images",
        annotations_directory_path="/Users/samkoshythomas/Desktop/Research-Project-DS/AIML/WeedDetection/Sam-Work/stubble-3/train/labels",
        data_yaml_path="/Users/samkoshythomas/Desktop/Research-Project-DS/AIML/WeedDetection/Sam-Work/stubble-3/data.yaml",
        force_masks=True
    )

    base_model = SegGPT(
        ontology=FewShotOntology(supervision_dataset)
    )
    
    base_model.label(input_predictions, extension=".jpg")
        # print(os.path.join(input_predictions,file))
    #   

# from roboflow import Roboflow
# rf = Roboflow(api_key="WRAR8pzooH3PWR6Kt9ue")
# project = rf.workspace("sam-thomas-m5kgt").project("stubble")
# dataset = project.version(3).download("yolov8")

if __name__ == "__main__":
    main()
