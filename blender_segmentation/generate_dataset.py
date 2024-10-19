import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_random_image(folder_path, target_size: int = None):
    """Load a random image from the specified folder."""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random_image_path = os.path.join(folder_path, random.choice(image_files))
    img = plt.imread(random_image_path)
    if target_size is not None:
        img = cv2.resize(img, (target_size, target_size))

    return img


def generate_composite_image(hand_folder, background_folder, output_folder, index, hand_target_size: int = 1024, background_resize_scale: float = 0.3):
    """Generate a composite image with a random hand on a random background."""
    # Load random hand and background images
    hand_image = load_random_image(hand_folder, target_size=hand_target_size)
    hand_mask = np.expand_dims(hand_image[:,:,3].astype(bool), -1)
    hand_image = (hand_image[:, :, :3]*255).astype(np.uint8)
    background_image = load_random_image(background_folder)
    resize_scale = random.uniform(background_resize_scale, 1.)
    resize_x = max(hand_image.shape[1], int(resize_scale*background_image.shape[1]))
    resize_y = max(hand_image.shape[0], int(resize_scale*background_image.shape[0]))
    background_image = cv2.resize(background_image, (resize_x, resize_y))
    # Generate random position for hand placement
    max_x = background_image.shape[1] - hand_image.shape[1]
    max_y = background_image.shape[0] - hand_image.shape[0]
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    # Create output mask
    mask = np.zeros(background_image.shape[:2], dtype=np.uint8)
    mask[y:y + hand_image.shape[0], x:x + hand_image.shape[1]] = hand_mask.squeeze(-1)
    # Create blended image
    blended_hand = hand_mask * hand_image + (1 - hand_mask)*background_image[y:y + hand_image.shape[0], x:x + hand_image.shape[1]]
    output = background_image.copy()
    output[y:y + hand_image.shape[0], x:x + hand_image.shape[1]] = blended_hand

    # Save the composite image
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, f'image_{index:04d}.jpg')
    plt.imsave(output_image_path, output)

    output_mask_path = os.path.join(output_folder, f'mask_{index:04d}.jpg')
    plt.imsave(output_mask_path, mask)


def main(args):
    for i in tqdm(range(args.num_images)):
        generate_composite_image(args.hand_folder, args.background_folder, args.output_folder, i + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate composite images with random hands on random backgrounds")
    parser.add_argument("--hand_folder", type=str, help="Folder containing hand images (PNG with alpha)")
    parser.add_argument("--background_folder", type=str, help="Folder containing background images (JPG)")
    parser.add_argument("--output_folder", type=str, help="Folder to save generated images")
    parser.add_argument("--num_images", type=int, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
