# BlazeFace - Real time object detection in browser

This is the code for the Medium Article "BlazeFace: How to Run Real-time Object Detection in the Browser".

This has two separate folders:
- Model training with python
- Model inference and demo with javascript

## Model training


### Dataset 
To train the model yourself, you first need to download the Kaggle dataset 
named [Face-Detection-Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset?resource=download).

With default Kaggle parameters, the downloaded and extracted folder might be named `Archive` by default.
I would suggest to move it to the `python` folder of this repo and to rename it `dataset` to make it compliant with the default parameters of the following scripts.


### Environment set

You can then create a new environment and install all the required packages:
```bash
conda create -y -n blazeface python=3.10
conda activate blazeface
pip install -r python/requirements.txt
```

### Model training

You can then train the BlazeFace model for 50 epochs:
```bash
python trainer.py --epochs 50
```

This will create a file `weights/blazeface.pt` with the trained weights of the model.
Having a GPU is recommended, but not mandatory. Note that a pretrained TFLite model is available in `javascript/assets/`.

> N.B.: check the help of the `trainer.py` script for more about the available parameters

If during training you encounter the following error `RuntimeError: received 0 items of ancdata`,
just run `ulimit -n 1048576` in your terminal, and then rerun the training script.

### Convert model to TFLite

You can finally convert the model to TFLite:

```bash
python tf_lite_converter.py
```

This will create a `weights/blazeface.tflite`: the trained model converted to TFLite format.

> Again, you can check the help of the `trainer.py` script for more about the available parameters


## Face detection demo in the browser

This is allowing you to test the real time face detection in browser.

Just launch the `index.html` with your favorite web server (you can use you IDE such as PyCharm of VSCode).
