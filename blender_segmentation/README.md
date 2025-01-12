# Blender Segmentation

This is the code to use Blender for image segmentation.

## Requirements

Before using it, `¶equirements.txt` must be installed.

For example, using conda, you can create a new environment and install the requirements libraries:
```bash
conda create -n blender python=3.11

conda activate blender

pip install -r requirements
```

## Generate the hand images

You can generate random images with the script `generate_blender_hands.py`.

```bash
python generate_blender_hands.py
```

The following parameters can be used:
- `--blend_file`: the blender file
- `--n_generation`: the number of images to generate
- `--camera_dist`: the distance of the camera from the hand center
- `--img_size`: the output image size
- `--seed`: the random seed

The generated images will be stored in a folder generated depending on the input blender filename.

## Generate dataset

You can then generate the dataset with the script `generate_dataset.py`:

The following parameters are available:
- `--hand_folder`: the folder containing the blender generated hands
- `--background_folder`: the folder containing the background images
- `--output_folder`: the output folder where to store images
- `--num_images`: the number of images to generate
- `--seed`: the random seed

## Train the model

Once the dataset is generated, the segmentation model can be trained with the script `segmentation_trainer.py`.

The following parameters are available:
- `--dataset`: the folder containing the dataset, containing the `train` and `valid` subfolders
- `--num_epochs`: the number of epochs
- `--batch_size`: the batch size
- `--learning_rate`: the learning rate

## Run the demo

You can run a demo on a webcam feed with a trained model with the script `run_segmentation_demo.py`.
