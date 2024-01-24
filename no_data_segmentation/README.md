# No Data Segmentation

Python code of the Medium article [How to train an instance segmentation model with no training data](), published in [Towards Data Science](https://towardsdatascience.com/).

## How to compute raw SAM masks

To compute the SAM masks of the images generated with Stable Diffusion, you first need to download the SAM model in this folder,
available on the [SAM repository](https://github.com/facebookresearch/segment-anything), or directly [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
Then, run the following script:

```bash
python generate_SAM_masks.py --images_folder images/new_selected_images/ --output_folder images/new_raw_masks/
```

You can check the available script parameters to make it work for a custom dataset or another SAM model. The available params are the following:
- `--model_ckpt`, string, defaults to `sam_vit_h_4b8939.pth`, path to the SAM model
- `--model_type`, string, defaults to `vit_h`, check SAM documentation for other available models
- `--device` string, default to `cuda`, the device for SAM model computation. Use `cpu` if you have no GPU
- `--images_folder`, string, defaults to `images/selected_images/`, path to images folder
- `--output_folder` string, default to `images/raw_masks/` output mask folder
- `--closing_kernel`: int, default to 7, the closing kernel size

## How to select the masks

Once the images are in the folder `images/selected_images` and the computed masks in `images/raw_masks`, run the following: 
```bash
python select_mask.py 
```
The selected masks will be in `images/selected_masks` by default.

You may specify the following arguments to suit your need:
- `--images_folder`: string, defaults to `images`, path to the images, expecting the subfolders `selected_images` and `raw_masks`,
- `--selected_masks_folder`: string, defaults to `images/selected_masks`, path to output folder of selected masks

## How to create the YOLO dataset

Assuming the previous steps were respected with all the defaults parameters,
you can simply run the following script to create the YOLO dataset:
```bash
python train_yolo_model.py --input_model yolov8s.pt --epochs 100 --dataset_path datasets/dataset.yaml
```

At the end, you should get the following folder architecture: 
```
datasets
└── animals
    ├── train
    │   ├── images: N images (e.g., jpeg or png files)
    │   └── labels: N labels (txt files)
    ├── val
    │   ├── images: M images (e.g., jpeg or png files)
    │   └── labels: M labels (txt files)
    └── dataset.yaml
```

- `--images_folder`: string, defaults to `images/selected_images`, path to the selected images
- `--selected_masks_folder`: string, defaults to `images/selected_masks`, path to input folder of selected masks
- `--dataset_path`, string, defaults to `datasets/animals/`, the output path of the dataset
- `--train_ratio`, float, defaults to 0.8, the train validation ratio when creating the dataset

## How to train the YOLO model

Once the dataset is created, you can simply train the YOLOv8 model with the following script:

```bash
python train_yolo_model.py
```

The following arguments are available:
- `--epochs`: int, defaults to 20, the number of training epochs
- `--dataset_path`: string, defaults to `datasets/animals/dataset.yaml`, path to the input dataset yaml file
- `--input_model`, string, defaults to `yolov8n-seg.pt` (will be downloaded automatically), path to the input model to train

Note: a pre-trained model containing the presented results in the article is available in `assets/best.pt`
