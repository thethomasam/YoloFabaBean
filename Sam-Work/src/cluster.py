import json
import math
import os
import pickle
import sys
import yaml
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LinearRegression

import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
import cv2


def get_cluster(num_cluster, data):
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(data)
    return kmeans


def drawClusterBounds(num_clusters, kmeans, data, image):
    # Draw bounding box around each cluster
    for i in range(num_clusters):
        # Get the indices of points in the current cluster
        indices = np.where(kmeans.labels_ == i)[0]

        # Select (X, Y) coordinates of points in the current cluster
        cluster_points = data.iloc[indices, -2:].to_numpy()

        # Find the bounding box coordinates for the cluster
        if cluster_points.size > 0:
            x_min, y_min = np.min(cluster_points, axis=0)
            x_max, y_max = np.max(cluster_points, axis=0)
            cv2.rectangle(image, (int(x_min), int(y_min)),
                          (int(x_max), int(y_max)), (255, 0, 0), 2)
    return image


def get_straightness(data, image, num_clusters, kmeans):
    for i in range(num_clusters):
        # Get the indices of points in the current cluster
        indices = np.where(kmeans.labels_ == i)[0]

        # Select (X, Y) coordinates of points in the current cluster
        cluster_points = data.iloc[indices, -2:].to_numpy()

        # Ensure there are enough points to fit a model
        if len(cluster_points) > 1:
            model = LinearRegression()
            # Reshape data for linear regression, with Y as independent variable
            model.fit(cluster_points[:, 1].reshape(-1,
                      1), cluster_points[:, 0])
            slope = model.coef_[0]
            intercept = model.intercept_

            # Calculate line points
            y_min, y_max = np.min(cluster_points[:, 1]), np.max(
                cluster_points[:, 1])
            x_min = slope * y_min + intercept
            x_max = slope * y_max + intercept

            r_squared = model.score(cluster_points[:, 1].reshape(-1,
                                                                 1), cluster_points[:, 0])
            print(r_squared)

            # Draw the vertical line
            cv2.line(image, (int(x_min), int(y_min)),
                     (int(x_max), int(y_max)), (0, 255, 0), 5)

    return image
def main():
    params = yaml.safe_load(open("params.yaml"))["cluster"]
    num_clusters_X = params['num_clusters_X']
    num_clusters_Y = params['num_clusters_Y']


    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model features\n")
        sys.exit(1)

    img = cv2.imread(
        '/Users/samkoshythomas/Desktop/Research-Project-DS/AIML/WeedDetection/Sam-Work/15-6-1-orig.jpg')
    input_predictions = sys.argv[1]
    img_Y = img.copy()
    df = pd.read_csv(input_predictions)
    # for x, y in zip(df['X Coordinate'], df['Y Coordinate']):
    #     cv2.circle(img, (x, y), 5, (0, 0, 255), 3)  # Draws a red point

    # X = np.array(df[:, -2])

    X = np.array(df['X Coordinate']).reshape(-1, 1)
    Y = np.array(df['Y Coordinate']).reshape(-1, 1)
    kmeans_x = get_cluster(num_clusters_X, X)
    kmeans_y = get_cluster(num_clusters_Y, Y)
    img_Y = drawClusterBounds(num_clusters_Y, kmeans_y, df, img_Y)

    img = drawClusterBounds(num_clusters_X, kmeans_x, df, img)
    output_path = os.path.join("data", "cluster")
    if not (os.path.isdir(output_path)):
        os.makedirs(output_path, exist_ok=True)
    img = get_straightness(df, img, num_clusters_X, kmeans_x)
    # Save the image with drawn circles and rectangles
    cv2.imwrite(os.path.join(output_path, 'plotX.jpg'), img)
    cv2.imwrite(os.path.join(output_path, 'plotY.jpg'), img_Y)
    # centroids = kmeans.cluster_centers_
# labels = kmeans.labels_


if __name__ == "__main__":
    main()
