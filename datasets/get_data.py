# Script for assembling datasets from json files provided at:
# "https://www.nature.com/articles/s41597-023-02871-z#Sec9"
# To replicate, store extracts from zip folder in "raw_data/"
# Loaded datasets will be stored in "datasets/"

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import cv2

## Helper functions -----------------------------------------------------------------------------------

# Helper function to create image masks for stenosis
def create_mask(segmentations, image_shape=(512, 512)):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for segment in segmentations:
        x_coords = np.round(np.array(segment[::2]) * image_shape[1] / 512).astype(int)
        y_coords = np.round(np.array(segment[1::2]) * image_shape[0] / 512).astype(int)
        points = np.vstack((x_coords, y_coords)).T

        cv2.fillPoly(mask, [points], 255)

    mask = mask.astype(np.float16)
    return mask

# Helper function to pad cropped images back to squares so all tensor will have matching dimensions
def pad_to_square(image):
    height, width = image.shape[:2]
    diff = abs(height - width)
    pad_height1 = diff // 2 if height < width else 0
    pad_height2 = diff - pad_height1 if height < width else 0
    pad_width1 = diff // 2 if width < height else 0
    pad_width2 = diff - pad_width1 if width < height else 0

    # Apply padding to make the image square
    if len(image.shape) == 3:
        padding = ((pad_height1, pad_height2), (pad_width1, pad_width2), (0, 0))
    else:
        padding = ((pad_height1, pad_height2), (pad_width1, pad_width2))
    
    square_image = np.pad(image, padding, mode='constant', constant_values=0)
    return square_image

# Enhance visibility of small, light areas using white tophat
def apply_white_tophat(image, kernel_size=(50, 50)):
    kernel = np.ones(kernel_size, np.uint8)
    image_neg = 255 - image
    tophat = cv2.morphologyEx(image_neg, cv2.MORPH_TOPHAT, kernel)
    result = cv2.subtract(image, tophat)
    if len(result.shape) == 2:
        result = np.expand_dims(result, axis=-1)
    result = np.clip(result, 0, 255)
    return result

# Enhance contrast within highlighted areas usng Contrast Limited Adaptive
# Histogram Equalization (CLAHE)
def apply_clahe(image_array, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if image_array.dtype != np.uint8:
        image_array = (255 * image_array).astype(np.uint8)
    
    # Apply CLAHE to the grayscale image
    image_clahe = clahe.apply(image_array)
    if len(image_clahe.shape) == 2:
        image_clahe = np.expand_dims(image_clahe, axis=-1)
    return image_clahe

# Helper function to load images tensors
def load_data(path_to_data, label_table, is_test=False):
    image_ids = label_table.groupby(['image_id'])[['segmentation', 'bbox']].agg(list).index.to_list()
    segmentations = label_table.groupby(['image_id'])[['segmentation', 'bbox']].agg(list)
    images = []
    labels = []
    for image_id in image_ids:
        file_name = str(image_id) + '.png'
        file_path = os.path.join(path_to_data, file_name)
        image = load_img(file_path, color_mode='grayscale', target_size=(256,256))
        image_array = img_to_array(image)
        images.append(image_array)
        label = create_mask(segmentations.loc[image_id]['segmentation'], image_shape=(256, 256))
        labels.append(label)
        if is_test == False:
            image_array_processed = apply_white_tophat(image_array)
            image_array_processed = apply_clahe(image_array_processed)
            images.append(image_array_processed)
            labels.append(label)
    images = tf.io.serialize_tensor(tf.convert_to_tensor(images, dtype=tf.float16) / 255.0)
    labels = tf.io.serialize_tensor(tf.convert_to_tensor(np.stack(labels, axis=0), dtype=tf.float16))
    return images, labels

## Load Data ------------------------------------------------------------------------------------------
print("Opening datasets...")

# Load syntax training dataset
with open('../raw_data/syntax/train/annotations/train.json') as f:
    syntax_train = pd.json_normalize(json.load(f), 'annotations')

# Load syntax validation dataset
with open('../raw_data/syntax/val/annotations/val.json') as f:
    syntax_val = pd.json_normalize(json.load(f), 'annotations')

# Load syntax test dataset
with open('../raw_data/syntax/test/annotations/test.json') as f:
    syntax_test = pd.json_normalize(json.load(f), 'annotations')

# Load stenosis training dataset
with open('../raw_data/stenosis/train/annotations/train.json') as f:
    stenosis_train = pd.json_normalize(json.load(f), 'annotations')

# Load stenosis validation dataset
with open('../raw_data/stenosis/val/annotations/val.json') as f:
    stenosis_val = pd.json_normalize(json.load(f), 'annotations')

# Load stenosis test dataset
with open('../raw_data/stenosis/test/annotations/test.json') as f:
    stenosis_test = pd.json_normalize(json.load(f), 'annotations')

# Load label categories
with open('../raw_data/syntax/test/annotations/test.json') as f:
     categories = pd.json_normalize(json.load(f), 'categories')

# Drop area, iscrowd, and attributes.occluded fields, which add no information
print("Formatting datasets...")
syntax_train.drop(['area','iscrowd','attributes.occluded'], axis=1, inplace=True)
syntax_val.drop(['area','iscrowd','attributes.occluded'], axis=1, inplace=True)
syntax_test.drop(['area','iscrowd','attributes.occluded'], axis=1, inplace=True)
stenosis_train.drop(['area','iscrowd','attributes.occluded'], axis=1, inplace=True)
stenosis_val.drop(['area','iscrowd','attributes.occluded'], axis=1, inplace=True)
stenosis_test.drop(['area','iscrowd','attributes.occluded'], axis=1, inplace=True)

# Add actual Syntax categories in and drop category_id
syntax_train = syntax_train.merge(categories[['id','name']],
                                  left_on='category_id',
                                  right_on='id',
                                  how='left').drop(['category_id', 'id_x', 'id_y'],
                                                                  axis=1)
syntax_train = syntax_train.rename(columns={'name': 'syntax_label'})
syntax_val = syntax_val.merge(categories[['id','name']],
                              left_on='category_id',
                              right_on='id',
                              how='left').drop(['category_id', 'id_x', 'id_y'],
                                               axis=1)
syntax_val = syntax_val.rename(columns={'name': 'syntax_label'})
syntax_test = syntax_test.merge(categories[['id','name']],
                                left_on='category_id',
                                right_on='id',
                                how='left').drop(['category_id', 'id_x', 'id_y'],
                                                 axis=1)
syntax_test = syntax_test.rename(columns={'name': 'syntax_label'})
stenosis_train = stenosis_train.merge(categories[['id','name']],
                                      left_on='category_id',
                                      right_on='id',
                                      how='left').drop(['category_id', 'id_x', 'id_y'],
                                                                      axis=1)
stenosis_train = stenosis_train.rename(columns={'name': 'syntax_label'})
stenosis_val = stenosis_val.merge(categories[['id','name']],
                                  left_on='category_id', right_on='id',
                                  how='left').drop(['category_id', 'id_x', 'id_y'],
                                                   axis=1)
stenosis_val = stenosis_val.rename(columns={'name': 'syntax_label'})
stenosis_test = stenosis_test.merge(categories[['id','name']],
                                    left_on='category_id',
                                    right_on='id',
                                    how='left').drop(['category_id', 'id_x', 'id_y'],
                                                     axis=1)
stenosis_test = stenosis_test.rename(columns={'name': 'syntax_label'})

# Remove unneeded extra dimension for segmentation lists
syntax_train['segmentation'] = syntax_train['segmentation'].apply(lambda x: x[0])
syntax_val['segmentation'] = syntax_val['segmentation'].apply(lambda x: x[0])
syntax_test['segmentation'] = syntax_test['segmentation'].apply(lambda x: x[0])
stenosis_train['segmentation'] = stenosis_train['segmentation'].apply(lambda x: x[0])
stenosis_val['segmentation'] = stenosis_val['segmentation'].apply(lambda x: x[0])
stenosis_test['segmentation'] = stenosis_test['segmentation'].apply(lambda x: x[0])

# Write training data to parquet files for EDA
print("Writing training data to parquet files for EDA...")
syntax_train.to_parquet("syntax_train.parquet")
syntax_val.to_parquet("syntax_val.parquet")
syntax_test.to_parquet("syntax_test.parquet")
stenosis_train.to_parquet("stenosis_train.parquet")
stenosis_val.to_parquet("stenosis_val.parquet")
stenosis_test.to_parquet("stenosis_test.parquet")

## Load and Store tensors (Stenosis only) -------------------------------------------------------------
print("Loading image training data to tensors...")
X_train_stenosis, y_train_stenosis = load_data("../raw_data/stenosis/train/images/", stenosis_train)
print("Writing training tensors to disk...")
tf.io.write_file('X_train_stenosis.tf', X_train_stenosis)
tf.io.write_file('y_train_stenosis.tf', y_train_stenosis)
del X_train_stenosis
del y_train_stenosis

print("Loading image validation data to tensors...")
X_val_stenosis, y_val_stenosis = load_data("../raw_data/stenosis/val/images/", stenosis_val)
print("Writing validation tensors to disk...")
tf.io.write_file('X_val_stenosis.tf', X_val_stenosis)
tf.io.write_file('y_val_stenosis.tf', y_val_stenosis)
del X_val_stenosis
del y_val_stenosis

print("Loading image test data to tensors...")
X_test_stenosis, y_test_stenosis = load_data("../raw_data/stenosis/test/images/", stenosis_test, is_test=True)
print("Writing test tensors to disk...")
tf.io.write_file('X_test_stenosis.tf', X_test_stenosis)
tf.io.write_file('y_test_stenosis.tf', y_test_stenosis)
del X_test_stenosis
del y_test_stenosis

print("Dataset preapration complete!")
