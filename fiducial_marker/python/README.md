# Fiducial Marker

Python code of the Medium article [Tired of QR codes? Build your own fiducial marker](https://medium.com/towards-data-science/tired-of-qr-codes-build-you-own-fiducial-marker-aab81cce1f25), published in [Towards Data Science](https://towardsdatascience.com/).

## How to generate random tags

To generate a random tag with the 2 circles and 16 dots per circle, 
just run the following command and store the output image in `image_tag.jpg`:

```bash
python tag_generation.py --num_layers 2 --num_dots_per_layer 20 --output_path image_tag.jpg
```
## How to generate synthetic images

To generate 800 training synthetic images for object detection training in the folder `datasets/train`, run the following command:

```bash
python synthetic_images_generation.py --number 800 --output_path datasets/train/ --background_path background_images/train/
```

The same way, generate for example 200 valid examples in the folder `dataset/valid`:

```bash
python synthetic_images_generation.py --number 200 --output_path datasets/valid/ --background_path background_images/valid/
```

You may specify the number of circles and dots per circle, as well as
the folder for background images and the output folder with the following arguments:
- `--output_path`: path to save the images, defaults to `datasets/train`
- `--background_path`: path to background images, defaults to `background_images/train`
- `--num_layers`: number of layers in the tag, defaults to `2`
- `--num_dots_per_layer`: number of dots per layers in the tag, defaults to `20`

## How to train an object detection model

Once the synthetic data has been created and placed in the right folders, the YOLO model can be trained with the following command:

```bash
python train_yolo_model.py --input_model yolov8s.pt --epochs 100 --dataset_path datasets/dataset.yaml
```

This command will train a YOLOv8 small model for 100 epochs.

The script parameters are the following:
- `--input_model`: path to the input model or architecture. Defaults to `yolov8n.pt` which means a pre-trained YOLOv8 nano model. Check [YOLOv8 documentation](https://docs.ultralytics.com/modes/train/#usage-examples) for more details.
- `--epochs`: number of epochs. Defaults to 10.
- `--dataset_path`: path to the required yaml file. Again, refer to YOLOv8 documentation for more details.

Once training is over, you may find all the results about it in the folder `runs/detect/train/`:
- `weights` folder contains the model weights, to be used for inference
- `results.csv` contains the results for each epoch
- there are a bunch of other useful files of viz and results

## How to test your object detection model on webcam feed

Once the model has been trained, you can test it on a webcam to check how it works in real condition.

To do so, just run the following script:

```bash
python run_yolo_webcam.py --input_model runs/detect/train/weights/best.pt 
```

The allowed parameters are the following:
- `--input_model`: path to the model weights, most likely in `runs/detect/train/weights/best.pt` if you train your own
- `--video_input`: id of the webcam, useful for openCV, defaults to `0`
- `--confidence_threshold`: confidence threshold for object detection, defaults to `0.25`

Note: a pre-trained model is available in `assets/best.pt`

## How to run the full pipeline

Once you have a working object detection model, you can run the pipeline with the following script:

```bash
python webcam_decoding.py --input_model runs/detect/train/weights/best.pt
```

This will run the pipeline on your webcam feed, detecting and decoding your fiducial marker, 
and printing the decoded code in the terminal.

The available parameters for this script are the following:
- `--input_model`: path to the model weights, most likely in `runs/detect/train/weights/best.pt` if you train your own
- `--video_input`: id of the webcam, useful for openCV, defaults to `0`
- `--confidence_threshold`: confidence threshold for object detection, defaults to `0.25`
- `--num_layers`: number of layers in the tag, defaults to `2`
- `--num_dots_per_layer`: number of dots per layers in the tag, defaults to `20`

Note: a pre-trained model is available in `assets/best.pt`
