# Agricultural Image Processing Pipeline

This repository contains a Python-based pipeline for agricultural image processing, specifically designed for weed detection and assessment.

## Getting Started

Follow these steps to quickly get started with the Agricultural Image Processing Pipeline:

## Prerequisites
Make sure you have the following prerequisites installed on your system:

- [Python 3.x](https://www.python.org/downloads/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [NumPy](https://pypi.org/project/numpy/)
- [Pandas](https://pypi.org/project/pandas/)
- [SciPy](https://pypi.org/project/scipy/)
- [Ultralytics](https://pypi.org/project/ultralytics/) (for YOLO-based featurization)
```bash
pip install opencv-python numpy pandas scipy ultralytics
```

### Install Dvc
Read through this [link](https://dvc.org/doc/install) to dowload dvc.


### Quick Start
- Clone the Repository:
```bash 
git clone -b sc_1590 https://gitlab.aiml.team/products/aagi/faba-internship.git
cd Sam-Work

```
- Run DVC pipeline:
```bash 
dvc repro

```




## About the scripts
### Prepare
Script: prepare.py\
This script extracts small images (windows) from a larger input image.

Functionality:

- Extracts small windows from a larger image for further analysis.
- Supports customization of the window size. 

```bash
python src/prepare.py input-image-path
```
Output: \
- Small window images extracted from the input image are saved in the data/prepared directory.


## Featurization
Script: featurization.py \
This script performs object detection using a YOLO model and visualizes the results on input images.

Functionality:

- Uses a YOLO model to detect objects in images.
- Filters out small objects based on contour area.
- Visualizes the detected objects on input images.

```bash
python src/featurization.py analysis-data-path yolo-model-path
```
Output: \
- Visualized images with detected objects are saved in the data/predictions directory.

## Evaluation 
Script: evaluate.py\
This script segments the weed areas based on color, calculates the percentage of weed area, and visualizes the results.\

Functionality:

- Segments weed areas based on green color.
- Calculates the percentage of weed area in each image saves the result in weed.csv file.
- Visualizes and saves the segmented images.

Output: 

- Segmented images highlighting weed areas are saved in the data/eval directory.
- A CSV file named weed.csv is generated, containing information about each segmented image, including the file name and the percentage of weed area.


```bash
python src/evaluation.py analysis-data-path yolo-model-path
```


