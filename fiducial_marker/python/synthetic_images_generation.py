import argparse
from glob import glob

import albumentations as A
import cv2
import numpy as np

from rune_tag import Runetag

def generate_synthetic_images_with_tag(
    n_images_to_generate: int,
    path_to_store_images: str,
    background_images_path: str,
    tag_layers: int = 2,
    tag_dots_per_layer: int = 24,
):
    """Generate synthetic images with RUNE tags randomly placed in background images

    Params
    ------
    n_images_to_generate: int
        number of synthetic images to generate
    path_to_store_images: str
        folder path to store output images
    background_images_path: str
        path to folder containing background images. If None, it will just generate random tags

    """
    composition = [
        A.ColorJitter(brightness=[.5, 1.], p=0.1),
        A.Affine(scale=[0.3, 4.], rotate=[-180, 180], shear=[-30, 30], fit_output=True, p=.9, cval=[250, 252, 254]),
        A.Perspective(fit_output=True, p=.9, pad_val=[250, 252, 254]),
        A.MotionBlur(),
        A.Defocus(radius=(1, 5)),
        A.RandomScale(scale_limit=(0, 0), p=1.)
    ]
    transform = A.Compose(composition, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Generate a random tag
    rune_tag = Runetag(tag_layers, tag_dots_per_layer)
    tag_image = rune_tag.generate_random_tag(write_file=False)
    # Convert it to image
    tag_image = cv2.cvtColor((tag_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # Since the tag is centered and takes the full image, its label is just the following
    labels = [0.5, 0.5, 1, 1]
    if background_images_path is not None:
        bkg_images = glob(background_images_path + '/*.jpg')
    idx = 0
    while idx < n_images_to_generate:
        # Transform tag image
        transformed = transform(image=tag_image, bboxes=[labels], class_labels=['tag'])
        transformed_labels = np.array(transformed['bboxes'])
        transformed_image = transformed['image']

        if background_images_path is None:
            bkg_image = transformed_image
        else:
            h, w = transformed_image.shape[:2]
            # Get background image
            bkg_image = bkg_images[np.random.randint(len(bkg_images))]
            bkg_image = cv2.imread(bkg_image)
            # Check background image is big enough
            if h >= bkg_image.shape[0] or w >= bkg_image.shape[1]:
                continue
            # Compute mask image
            i = np.random.randint(bkg_image.shape[0] - h)
            j = np.random.randint(bkg_image.shape[1] - w)
            # Compute tag with original image as background
            mask = (transformed_image[:, :, 0] == 250) & (transformed_image[:, :, 1] == 252) & (
                        transformed_image[:, :, 2] == 254)
            mask = np.expand_dims(mask, -1).astype(int)
            tmp = transformed_image * (1 - mask) + bkg_image[i:i + h, j:j + w] * mask
            # Compute output image
            bkg_image[i:i + h, j:j + w] = tmp
            transformed_labels[:, 0] = (transformed_labels[:, 0]*transformed_image.shape[1] + j)/bkg_image.shape[1]
            transformed_labels[:, 1] = (transformed_labels[:, 1]*transformed_image.shape[0] + i)/bkg_image.shape[0]
            transformed_labels[:, 2] = transformed_labels[:, 2]*transformed_image.shape[0]/bkg_image.shape[0]
            transformed_labels[:, 3] = transformed_labels[:, 3]*transformed_image.shape[0]/bkg_image.shape[0]

        # Apply a final transformation
        final = A.Compose([
            A.ColorJitter(p=.1),
            A.GaussianBlur(p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        try:
            output = final(image=bkg_image, bboxes=transformed_labels, class_labels=['tag'])
        except ValueError:
            # Handle out of boundaries box
            continue

        transformed_labels = output['bboxes']
        bkg_image = output['image']
        # Save image and labels
        output_root_img = path_to_store_images + f'/images/synthetic_image_{idx}.jpg'
        output_root_txt = path_to_store_images + f'/labels/synthetic_image_{idx}.txt'

        cv2.imwrite(output_root_img, bkg_image)
        # save image with cv2.imwrite, save labels to file with savetxt
        output_arr = np.array([[0] + list(transformed_labels[0])])
        np.savetxt(output_root_txt, output_arr, fmt='%f')
        idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic images and labels with rune tags')
    parser.add_argument('--number', type=int, help='number of synthetic images to generate', default=2)
    parser.add_argument('--output_path', type=str, help='path to save the images', default='datasets/train/')
    parser.add_argument('--background_path', type=str, help='path to background images', default='background_images/train/')
    parser.add_argument('--num_layers', type=int, help='number of layers in the tag', default=2)
    parser.add_argument('--num_dots_per_layer', type=int, help='number of dots per layers in the tag', default=24)
    args = parser.parse_args()

    generate_synthetic_images_with_tag(
        n_images_to_generate=args.number,
        path_to_store_images=args.output_path,
        background_images_path=args.background_path,
        tag_layers=args.num_layers,
        tag_dots_per_layer=args.num_dots_per_layer,
    )
