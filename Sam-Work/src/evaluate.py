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

def reconstruct_image(window_folder_path:str, 
                      output_path:str,
                        window_size:tuple):
    """
    Reconstruct the original image from small window images.

    Parameters:
    - window_folder_path (str): Path to the directory containing small window images.
    - output_path (str): Path to save the reconstructed image.
    - window_size (tuple): Size of the small windows in pixels (width, height).

    Returns:
    - None

    This function reconstructs the original image from small window images. It assumes that the
    small window images are named in the format 'window_x_y.jpg', where 'x' and 'y' are the
    coordinates of the top-left corner of each window. The reconstructed image is saved at the
    specified 'output_path' as 'aggregated.jpg'.

    Example:
    >>> window_folder_path = 'path/to/small/windows'
    >>> output_path = 'path/to/output/reconstructed_image'
    >>> window_size = (100, 100)
    >>> reconstruct_image(window_folder_path, output_path, window_size)
    """
    # Get the size of the original image
    original_width, original_height = (2380,17059 ) # Replace with the actual size of the original image

    # Create an empty image of the original size
    reconstructed_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Iterate through the small images and paste them into the empty image
    for filename in os.listdir(window_folder_path):
        if filename.endswith(".jpg") and filename.startswith("window_"):
            # Extract x and y coordinates from the filename
            root, _ = os.path.splitext(filename)
        
            x, y = map(int, root.split('_')[1:3])

            # Read the small image
            window = cv2.imread(os.path.join(window_folder_path, filename))

            # Paste the small image into the corresponding position in the empty image
            reconstructed_image[y:y + window_size[1], x:x + window_size[0]] = window

    # Save the reconstructed 
    output_path+='/aggregated.jpg'
    print(output_path)
    cv2.imwrite(output_path, reconstructed_image)


def segment_weed(input_predictions:str
                 ,output_path:str
            ):
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    lower_green=params['lower_green']
    upper_green=params['upper_green']
    
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

    
    
    weed_data=[]
    if os.path.isdir(input_predictions) and os.listdir(input_predictions) :
        files = os.listdir(input_predictions)
        jpg_files = [file for file in files if file.endswith('.jpg')]

        for file in jpg_files:
            image = cv2.imread(os.path.join(input_predictions,file))
            width,height,_=image.shape
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the green color and its shades in HSV

            lower_green = np.array(lower_green)  # Lower bound for green
            upper_green = np.array(upper_green)  #   # Upper bound for green
      
            # Create a mask to extract green regions
            green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Apply the mask to the original image to extract the green regions
            green_regions = cv2.bitwise_and(image, image, mask=green_mask)

            green_mask = (green_regions[:, :, 1] > 0)
            contours, _ = cv2.findContours(green_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the total area of green color
            total_area = 0
            for contour in contours:
                total_area += cv2.contourArea(contour)
            result_image = image.copy()
            cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(output_path+'/'+file,result_image)
    weed_data=pd.DataFrame(weed_data)

    #chaged 
    weed_data.to_csv('./weed.csv')

    



def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    params_prepare = yaml.safe_load(open("params.yaml"))["prepare"]
    window_size=params_prepare['split']
    window_size=(window_size,window_size)
   



    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model features\n")
        sys.exit(1)
    output_path=os.path.join("data", "eval")
    if  not os.path.isdir(output_path) :
        os.makedirs(output_path, exist_ok=True)
    
    #added new
    input_predictions = sys.argv[1]
    
    segment_weed(input_predictions,output_path)
    # reconstruct_image(output_path, output_path, window_size)
    


if __name__ == "__main__":
    main()
