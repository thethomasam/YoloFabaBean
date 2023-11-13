import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def get_predictions(
    analysis_data: str,
    yolo_model: YOLO,  # Assuming YoloV4 is a class for your YOLO model
    threshold_area: float,
    output_path: str,
    save_dir: str
):
    """
    Generate predictions using a YOLO model and visualize the results on images.

    Parameters:
    - analysis_data (str): Path to the directory containing input images for analysis.
    - yolo_model (YoloV4): YOLO model instance for object detection.
    - threshold_area (float): Area threshold for considering detected objects.
    - output_path (str): Path to the directory where the visualized images will be saved.
    - save_dir (str): Directory to save intermediate results during YOLO predictions.

    Returns:
    - None

    This function performs object detection using the YOLO model on images from 'analysis_data'.
    Detected objects are visualized on the input images, and the visualized images are saved in
    the 'output_path' directory. The 'threshold_area' parameter filters out small objects based
    on their contour area.

    Example:
    >>> analysis_data = 'path/to/analysis/images'
    >>> yolo_model = YoloV4()  # Assuming YOLO model is initialized
    >>> threshold_area = 1000.0
    >>> output_path = 'path/to/output/images'
    >>> save_dir = 'path/to/save/intermediate/results'
    >>> get_predictions(analysis_data, yolo_model, threshold_area, output_path, save_dir)
    """
    # read files from directory
    for file in os.listdir(analysis_data):
        file_path=os.path.join(analysis_data,file)
        image=cv2.imread(file_path)
        # get predictions for each image
        results = yolo_model.predict(file_path,rect=True,save=True,save_dir=save_dir,conf=0.1)
        
        #iterate through bounding boxes
        for result in results:
            for x,y,w,h in result.boxes.xywh:
                x,y,w,h = int(x), int(y), int(w), int(h)
                #draw filled rectangles to hide faba beans
                cv2.rectangle(image,  (x - w//2, y - h//2),(x+w//2, y+w//2), 0, -3)

        print(os.path.join(output_path,str(file)+'.jpg'))
        cv2.imwrite(os.path.join(output_path,str(file)),image)
        
     


def main():
    params = yaml.safe_load(open("params.yaml"))["featurize"]
    threshold_area = params["threshold_area"]
    
    np.set_printoptions(suppress=True)

    if len(sys.argv) != 3 and len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
        sys.exit(1)

    analysis_data = sys.argv[1]
    best_model = sys.argv[2]
    yolov8_detector = YOLO(best_model)
    
    current_directory = os.getcwd()
    save_dir=os.path.join(current_directory,'Logs')
    output_path=os.path.join("data", "predictions")
    
    if os.path.isdir(output_path) and os.listdir(output_path) :
        print('No Files')
        files = os.listdir(output_path)
        if files:
            print(f"Files found in '{output_path}'. Removing them...")
        [os.remove(os.path.join(output_path, file)) for file in files]
    
    else:
        print(f"The directory '{output_path}' does not exist.")
        os.makedirs(output_path, exist_ok=True)
    
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    get_predictions(analysis_data,yolov8_detector,threshold_area,output_path,save_dir)
  
    

if __name__ == "__main__":
    main()
