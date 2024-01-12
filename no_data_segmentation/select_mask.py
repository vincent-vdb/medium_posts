import argparse
from glob import glob
import shutil

import cv2
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key


def on_press(key, left, right, mask_folder):
    if key == Key.left:
        destination = mask_folder + left.split('/')[-1]
        shutil.copy(left, destination)
        return False
    elif key == Key.right:
        destination = mask_folder + right.split('/')[-1]
        shutil.copy(right, destination)
        return False
    elif key == Key.up or key == Key.down:
        print("Discard mask")
        return False


def select_mask(images_folder: str, masks_folder: str):
    image_names = glob(images_folder + 'selected_images/*/*.png')

    for image_name in image_names:

        print(image_name)
        falsemask_path = args.images_folder + 'raw_masks/multimask__False'+ image_name.split('/')[-1][:-3] + 'jpg'
        truemask_path = args.images_folder + 'raw_masks/multimask__True'+ image_name.split('/')[-1][:-3] + 'jpg'

        img = cv2.imread(image_name)
        falsemask = cv2.imread(falsemask_path)

        truemask = cv2.imread(truemask_path)

        display_mask = np.concatenate([falsemask, truemask], axis=1)
        display_img = np.concatenate([img, img], axis=1)
        display = cv2.addWeighted(display_img, 1, display_mask, 0.3, 0.)

        cv2.imshow('Masks', display)
        if cv2.waitKey(5) & 0xFF == 27:
           break

        with keyboard.Listener(on_press=lambda event: on_press(event, left=falsemask_path, right=truemask_path, mask_folder=masks_folder)) as listener:
           listener.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select masks")
    parser.add_argument("--images_folder", type=str, help="Folder containing images", default='images/')
    parser.add_argument("--selected_masks_folder", type=str, help="Folder path to store selected masks", default="images/selected_masks/")
    args = parser.parse_args()

    select_mask(args.images_folder, args.selected_masks_folder)
