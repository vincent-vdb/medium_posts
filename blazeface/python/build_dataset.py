import argparse
import os
import random

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd

random.seed(0)


def create_folder_architecture(path):
    suffixes = ['', '/images', '/images/train', '/images/val', '/labels', '/labels/train', '/labels/val']
    for suffix in suffixes:
        if not os.path.exists(path + suffix):
            os.mkdir(path + suffix)


def download_public_file_from_s3(s3_file_key, local_file_path, bucket_name: str = 'open-images-dataset'):
    # Create an S3 client with unsigned configuration
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    try:
        # Download the file
        s3_client.download_file(bucket_name, s3_file_key, local_file_path)
        print(f"File {s3_file_key} downloaded from bucket {bucket_name} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")


def download_image(image_id, path, subset):
    image_path = path + '/images/' + subset + '/' + image_id + '.jpg'
    download_public_file_from_s3(f'validation/{image_id}.jpg', image_path)


def build_labels(data, path, subset):
    output_df = pd.DataFrame({
        'class': [0] * len(data),
        'x_center': 0.5 * (data['XMin'] + data['XMax']),
        'y_center': 0.5 * (data['YMin'] + data['YMax']),
        'width': data['XMax'] - data['XMin'],
        'height': data['YMax'] - data['YMin']
    })
    output_path = path + '/labels/' + subset + '/' + data.iloc[0]['ImageID'] + '.txt'
    output_df.to_csv(output_path, header=None, sep=' ', index=None)


def build_dataset(label_name, path, val_split: float = 0.8):
    # Create folders if needed
    create_folder_architecture(path)
    # Download dataset metadata
    os.system('wget https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv')
    # Download dataset labels
    os.system('wget https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv')
    # Load data
    df_labels = pd.read_csv('validation-annotations-bbox.csv')
    df_meta = pd.read_csv('validation-images-with-rotation.csv')
    # Select label
    human_faces = df_labels[df_labels['LabelName'] == label_name]
    seen = set([])
    # Download the images if license is permissive
    for i in range(len(human_faces)):
        image_id = human_faces.iloc[i]['ImageID']
        if image_id not in seen:
            seen.add(image_id)
            # Check the license is OK
            meta = df_meta[df_meta['ImageID'] == image_id]
            if len(meta) > 0 and meta.iloc[0]['License'] == 'https://creativecommons.org/licenses/by/2.0/':
                subset = 'train'
                if random.uniform(0, 1) > val_split:
                    subset = 'val'
                download_image(image_id, path, subset)
                build_labels(human_faces[human_faces['ImageID'] == image_id], path, subset)


if __name__ == '__main__':
    # Parse the args
    parser = argparse.ArgumentParser(description='Build the dataset')
    parser.add_argument('--label_name', help='label name in the Open Images Dataset V7 (defauls to human faces)', type=str, default='/m/0dzct')
    parser.add_argument('--path', help='folder path to create the dataset', type=str, default='open_images_dataset')

    args = parser.parse_args()
    build_dataset(args.label_name, args.path)
