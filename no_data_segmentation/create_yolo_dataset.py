import argparse
from glob import glob
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
import shapely


def mask_to_polygons(mask: np.array, min_mask_threshold: float = 900) -> list:
    all_polygons = []
    if mask.max() <= 1.:
        mask = (mask * 255).astype(np.uint8)
    mask = mask[:, :, 0] > 128
    for shape, value in features.shapes(mask.astype(np.int16), mask=mask, transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        polygon = shapely.geometry.shape(shape)
        if polygon.area >= min_mask_threshold:
            all_polygons.append(np.array(list(polygon.exterior.segmentize(2).coords), dtype=np.int32))
    return all_polygons


def mask2yolo(image_path: str, mask_path: str, output: str):
    # Open output file
    file = open(output, "w")
    height, width = plt.imread(image_path).shape[:2]
    mask = plt.imread(mask_path)
    polygons = mask_to_polygons(mask)
    if len(polygons) == 0:
        print('no polygons found', mask_path)
        return None

    if 'horse' in image_path:
        file.write('0 ')
    elif 'lion' in image_path:
        file.write('1 ')
    elif 'tiger' in image_path:
        file.write('2 ')
    else:
        file.write('3 ')
    for polygon in polygons:
        for point in polygon:
            file.write(f'{str(point[0] / width)} {str(point[1] / height)} ')
    file.write('\n')


def generate_dataset_from_masks(images_path: str, masks_path: str, dataset_path: str, train_ratio: float = 0.8) -> None:
    all_images = glob(images_path + '*/*.png')
    if not os.path.exists(dataset_path + '/train'):
        os.mkdir(dataset_path + '/train')
        os.mkdir(dataset_path + '/train/images')
        os.mkdir(dataset_path + '/train/labels')
    if not os.path.exists(dataset_path + '/val'):
        os.mkdir(dataset_path + '/val')
        os.mkdir(dataset_path + '/val/images')
        os.mkdir(dataset_path + '/val/labels')
    for image in all_images:
        image_rootname = image.split('/')[-1].split('.')[0]
        # Get the associated mask if any
        mask = glob(masks_path + '*' + image_rootname + '*')
        if len(mask) == 1:
            if random.uniform(0, 1) < train_ratio:
                folder = 'train/'
            else:
                folder = 'val/'
            # Make the output path
            output = dataset_path + folder + 'labels/' + image_rootname + '.txt'
            mask2yolo(image, mask[0], output)
            # Copy the image in the datataset folder
            shutil.copy(image, dataset_path + folder + 'images/' + image_rootname + '.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLO dataset")
    parser.add_argument("--images_folder", type=str, help="Folder containing images", default='images/selected_images/')
    parser.add_argument("--selected_mask_folder", type=str, help="Folder containing selected masks", default="images/selected_masks/")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset folder", default="datasets/animals/")
    parser.add_argument("--train_ratio", type=float, help="Train ratio, between 0. and 1. Defaults to .8", default=0.8)
    args = parser.parse_args()

    generate_dataset_from_masks(args.images_folder, args.selected_mask_folder, args.dataset_path)
