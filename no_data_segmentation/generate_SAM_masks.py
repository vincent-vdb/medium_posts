import argparse
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate SAM segmentation masks")
    parser.add_argument("--model_ckpt", type=str, help="Model checkpoint path", default='sam_vit_h_4b8939.pth')
    parser.add_argument("--model_type", type=str, help="SAM model type", default="vit_h")
    parser.add_argument("--device", type=str, help="Device", default="cuda")
    parser.add_argument("--images_folder", type=str, help="Path to images folder", default="images/selected_images/")
    parser.add_argument("--output_folder", type=str, help="Output masks folder path", default="images/raw_masks/")
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
        for multimask in [True, False]:
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=multimask,
            )
            if multimask:
                mask = masks[scores.argmax()]
            else:
                mask = masks[0]
            if args.closing_kernel > 0:
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            filename = images[idx].split("/")[-1].split('.')[0]
            output_path = f"{args.output_folder}/multimask__{multimask}{filename}.jpg"
            plt.imsave(output_path, mask)
