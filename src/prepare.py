import os
import random
import re
import sys
import xml.etree.ElementTree
import cv2
import yaml
import shutil




def extract_small_images(input_image_path, output_folder_path, window_size, step):
    input_image = cv2.imread(input_image_path)
    height, width, _ = input_image.shape
    print(height,width)

    for y in range(0, height - window_size[1] + 1, window_size[1]): # Adjusted step size to window_size[1]
        for x in range(0, width - window_size[0] + 1, window_size[0]): # Adjusted step size to window_size[0]
   
            # Define the coordinates for the window
            left = x
            top = y
            right = x + window_size[0]
            bottom = y + window_size[1]

            # Crop the image to the window
            window = input_image[top:bottom, left:right]

            # Save the cropped window
            cv2.imwrite(f'{output_folder_path}/window_{x}_{y}.jpg', window)

    

def main():
    input_image_path = "15-6-1-orig.jpg"
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    # Test data set split ratio
    split = params["split"]
    step= params["step"]
    window_size=(split,split)

    
    output_path=os.path.join("data", "prepared")
    if os.path.isdir(output_path) and os.listdir(output_path) :
        print('No Files')
        files = os.listdir(output_path)
        if files:
            print(f"Files found in '{output_path}'. Removing them...")
        [os.remove(os.path.join(output_path, file)) for file in files]
    
    else:
        print(f"The directory '{output_path}' does not exist.")
        os.makedirs(output_path, exist_ok=True)
    extract_small_images(input_image_path, output_path, window_size, step)


if __name__ == "__main__":
    main()
