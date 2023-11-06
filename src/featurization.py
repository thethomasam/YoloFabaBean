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

def get_df(data):
    """Read the input data file and return a data frame."""
    df = pd.read_csv(
        data,
        encoding="utf-8",
        header=None,
        delimiter="\t",
        names=["id", "label", "text"],
    )
    sys.stderr.write(f"The input data frame {data} size is {df.shape}\n")
    return df


def save_matrix(df, matrix, names, output):
    """
    Save the matrix to a pickle file.

    Args:
        df (pandas.DataFrame): Input data frame.
        matrix (scipy.sparse.csr_matrix): Input matrix.
        names (list): List of feature names.
        output (str): Output file name.
    """
    id_matrix = sparse.csr_matrix(df.id.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T

    result = sparse.hstack([id_matrix, label_matrix, matrix], format="csr")

    msg = "The output matrix {} size is {} and data type is {}\n"
    sys.stderr.write(msg.format(output, result.shape, result.dtype))

    with open(output, "wb") as fd:
        pickle.dump((result, names), fd)
    pass


def generate_and_save_train_features(train_input, train_output, bag_of_words, tfidf):
    """
    Generate train feature matrix.

    Args:
        train_input (str): Train input file name.
        train_output (str): Train output file name.
        bag_of_words (sklearn.feature_extraction.text.CountVectorizer): Bag of words.
        tfidf (sklearn.feature_extraction.text.TfidfTransformer): TF-IDF transformer.
    """
    df_train = get_df(train_input)
    train_words = np.array(df_train.text.str.lower().values)

    bag_of_words.fit(train_words)

    train_words_binary_matrix = bag_of_words.transform(train_words)
    feature_names = bag_of_words.get_feature_names_out()

    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

    save_matrix(df_train, train_words_tfidf_matrix, feature_names, train_output)


def generate_and_save_test_features(test_input, test_output, bag_of_words, tfidf):
    """
    Generate test feature matrix.

    Args:
        test_input (str): Test input file name.
        test_output (str): Test output file name.
        bag_of_words (sklearn.feature_extraction.text.CountVectorizer): Bag of words.
        tfidf (sklearn.feature_extraction.text.TfidfTransformer): TF-IDF transformer.
    """
    df_test = get_df(test_input)
    test_words = np.array(df_test.text.str.lower().values)

    test_words_binary_matrix = bag_of_words.transform(test_words)
    test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)
    feature_names = bag_of_words.get_feature_names_out()

    save_matrix(df_test, test_words_tfidf_matrix, feature_names, test_output)


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
    for file in os.listdir(analysis_data):
        file_path=os.path.join(analysis_data,file)
        image=cv2.imread(file_path)
        results = yolov8_detector.predict(file_path,rect=True,save=True,save_dir=save_dir,conf=0.1)
        without_faba=image.copy()
        for result in results:
            for x,y,w,h in result.boxes.xywh:
                x,y,w,h = int(x), int(y), int(w), int(h)
                cv2.rectangle(without_faba,  (x - w//2, y - h//2),(x+w//2, y+w//2), 255, -1)
                cv2.rectangle(image,  (x - w//2, y - h//2),(x+w//2, y+w//2), (0,255,0), 3)
        hsv_image = cv2.cvtColor(without_faba, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])  # Lower bound for green in HSV
        upper_green = np.array([85, 255, 255])  # Upper bound for green in HSV
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
                
               
                x, y, w, h = cv2.boundingRect(contour)
                area=cv2.contourArea(contour)
                    #cv2.putText(image, hsv_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if area>threshold_area:
                    cv2.circle(image, (x, y), 20, (255, 0, 0), 2)
        print(os.path.join(output_path,str(file)+'.jpg'))
        cv2.imwrite(os.path.join(output_path,str(file)+'.jpg'),image)
        

if __name__ == "__main__":
    main()
