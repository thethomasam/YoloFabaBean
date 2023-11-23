import os
import random
import re
import sys
import xml.etree.ElementTree
import cv2
import yaml
import shutil

# change
# change

def extract_small_images(input_image_path: str,
                          output_folder_path: str, 
                          window_size: tuple):
    """

    Extract small images (windows) from a larger input image.

    Parameters:
    - input_image_path (str): Path to the input image.
    - output_folder_path (str): Path to the folder where the extracted images will be saved.
    - window_size (tuple): Size of the extraction window in the format (width, height).
   

    Returns:
    - None

    This function reads the input image specified by 'input_image_path' and extracts small
    images (windows) with the specified 'window_size' . The extracted images
    are saved in the 'output_folder_path' with filenames indicating their position in the
    original image.

    Example:
    >>> input_image_path = 'path/to/input/image.jpg'
    >>> output_folder_path = 'path/to/output/folder'
    >>> window_size = (100, 100)
    >>> extract_small_images(input_image_path, output_folder_path, window_size)
    """
    input_image = cv2.imread(input_image_path)
    height, width, _ = input_image.shape
    print(height, width)
    img = input_image.copy()
    #
    for x in range(0, width, window_size[0]):
        # Adjusted step size to window_size[1]
        for y in range(0, height, window_size[1]):
         # Adjusted step size to window_size[0]
   
            # Define the coordinates for the window
            left = x
            top = y
            right = min(x + window_size[0], width)  # Adjust for the right edge
            bottom = min(y + window_size[1], height)
            cv2.rectangle(img, (int(left), int(top)),
                          (int(right), int(bottom)), (0, 255, 0), 2)

            # Crop the image to the window
            window = input_image[top:bottom, left:right]

            # Save the cropped window
            cv2.imwrite(f'{output_folder_path}/window_{x}_{y}.jpg', window)
            print(x, y)
    cv2.imwrite('./grid_plot.jpg', img)

    

def main():
    input_image_path = "/Users/samkoshythomas/Desktop/Research-Project-DS/AIML/WeedDetection/day-2.png"
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    # Test data set split ratio
    split = params["split"]
 
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
    extract_small_images(input_image_path, output_path, window_size)


if __name__ == "__main__":
    main()
