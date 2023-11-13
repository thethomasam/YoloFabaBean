import json
import math
import os
import pickle
import sys
import yaml

import pandas as pd
from sklearn import metrics
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt
import numpy as np
import cv2
from skimage.segmentation import slic, mark_boundaries




def segment_weed(input_predictions,output_path):
    '''
     Parameters:
    - input_predictions (str): Path to the directory containing input images.
    - output_path (str): Path to the directory where segmented images will be saved.

    Returns:
    - None

    This function reads images from the 'input_predictions' directory, segments the weed areas
    based on the green color, calculates the percentage of weed area in each image, and saves
    the segmented images in the 'output_path' directory. It also creates a CSV file 'weed.csv'
    containing information about each segmented image, including the file name and the percentage
    of weed area.

    Example:
    >>> input_predictions = 'path/to/input/images'
    >>> output_path = 'path/to/output/segmented/images'
    >>> segment_weed(input_predictions, output_path)
    

    Need To Add: 
     -  Fine Tune Green Color Upper and Lower Bounds
    '''

    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    
    weed_data=[]
    if os.path.isdir(input_predictions) and os.listdir(input_predictions) :
        files = os.listdir(input_predictions)
        for file in files:
            image = cv2.imread(os.path.join(input_predictions,file))
            width,height,_=image.shape
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for the green color and its shades in HSV
            lower_green = np.array(params['lower_green'])  # Lower bound for green
            upper_green = np.array(params['upper_green'])  # Upper bound for green
            green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
            # Apply the mask to the original image to extract the green regions
            green_regions = cv2.bitwise_and(image, image, mask=green_mask)
            green_mask = (green_regions[:, :, 1] > 0)
            contours, _ = cv2.findContours(green_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            # Calculate the total area of green color
            total_area = 0
            for contour in contours:
                total_area += cv2.contourArea(contour)
            row={'file':str(file),'weed_area':(total_area/(width*height))*100}
            weed_data.append(row)
            result_image = image.copy()
            cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_path,file),result_image)    
    weed_data=pd.DataFrame(weed_data)
    weed_data.to_csv('./weed.csv')

    



def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    green_threshold = params["green_threshold"]
    n_segments = params["n_sgements"]
    
    print(len(sys.argv))

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model features\n")
        sys.exit(1)
    output_path=os.path.join("data", "eval")
    if  not os.path.isdir(output_path) :
        os.makedirs(output_path, exist_ok=True)
    
    #added new
    input_predictions = sys.argv[1]
    print(input_predictions )
    segment_weed(input_predictions,output_path,green_threshold,n_segments)

    


if __name__ == "__main__":
    main()
