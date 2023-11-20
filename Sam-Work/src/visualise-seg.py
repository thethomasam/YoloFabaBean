import cv2
import numpy as np

# Load the original image
# original_image = cv2.imread('/Users/samkoshythomas/Desktop/Research-Project-DS/AIML/WeedDetection/Sam-Work/stubble-3/test/images/window_1280_10880_jpg.rf.f1bb8128571341d29a6ed93abbf48eae.jpg')

# # # Load segmentation labels from a text file
# with 
#     lines = file.readlines()

# Convert the labels to a NumPy array
import cv2
import numpy as np
from random import randint
import os

with open('/Users/samkoshythomas/Desktop/Research-Project-DS/AIML/WeedDetection/Sam-Work/stubble-3/test/labels/window_1280_10880_jpg.rf.f1bb8128571341d29a6ed93abbf48eae.txt', 'r') as f:
    labels = f.read().splitlines()
img = cv2.imread('/Users/samkoshythomas/Desktop/Research-Project-DS/AIML/WeedDetection/Sam-Work/stubble-3/test/images/window_1280_10880_jpg.rf.f1bb8128571341d29a6ed93abbf48eae.jpg')
h,w = img.shape[:2]
alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255  # Fully opaque

# Merge the original image and the alpha channel to create an RGBA image
rgba_image = cv2.merge((img, alpha_channel))
for label in labels:
    class_id, *poly = label.split(' ')
    print(label)
    
    poly = np.asarray(poly,dtype=np.float16).reshape(-1,2) # Read poly, reshape
    poly *= [w,h] # Unscale
    
    cv2.polylines(img, [poly.astype('int')], True, (205,133,63), 2) # Draw Poly Lines
    break
    # cv2.fillPoly(img, [poly.astype('int')], (randint(0,255),randint(0,255),randint(0,255)), cv2.LINE_AA) # Draw area

output_path=os.path.join("data", "stuble")
if  not os.path.isdir(output_path) :
    os.makedirs(output_path, exist_ok=True)
cv2.imwrite(os.path.join(output_path,'seg.jpg'),img)



