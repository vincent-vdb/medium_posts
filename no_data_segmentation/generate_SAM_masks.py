import argparse
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate SAM segmentation masks")
    parser.add_argument("--model_ckpt", type=str, help="Model checkpoint path", default='sam_vit_h_4b8939.pth')
    parser.add_argument("--model_type", type=str, help="SAM model type", default="vit_h")
    parser.add_argument("--device", type=str, help="Device", default="cuda")
    parser.add_argument("--images_folder", type=str, help="Path to images folder", default="images/selected_images/")
    parser.add_argument("--output_folder", type=str, help="Output masks folder path", default="images/masks/")
    parser.add_argument("--closing_kernel", type=int, help="Closing kernel size, 0 for no closing", default=7)
    args = parser.parse_args()

    sam = sam_model_registry[args.model_type](checkpoint=args.model_ckpt)
    sam.to(device=args.device)

    predictor = SamPredictor(sam)

    images = glob(args.images_folder + '*/*.png')
    # Using everytime the central point
    input_point = np.array([[256, 256]])
    input_label = np.array([1])
    # Creating a kernel
    kernel = np.ones((args.closing_kernel, args.closing_kernel), np.uint8)

    for idx in tqdm(range(len(images))):
        image = (plt.imread(images[idx])*255).astype(np.uint8)
        predictor.set_image(image)

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        for multimask in [True, False]:
            if multimask:
                mask = masks[scores.argmax()]
            else:
                mask = masks[0]
            if args.closing_kernel > 0:
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            filename = images[idx].split("/")[-1].split('.')[0]
            output_path = f"{args.output_folder}/multimask__{multimask}{filename}.jpg"
            plt.imsave(output_path, mask)
