# BlazeFace - Real time object detection in browser

This is the code for the Medium Article "BlazeFace: How to Run Real-time Object Detection in the Browser".

This has two separate folders:
- Model training with Python
- Model inference and demo with JavaScript

## Model training with Python

All the commands in this section are expected to be used in the `python` folder of this repo.

### Environment setup

You can then create a new environment and install all the required packages:
```bash
conda create -y -n blazeface python=3.10
conda activate blazeface
pip install -r python/requirements.txt
```

### Dataset 
If you're willing to train the model yourself, you need to download and build the dataset first.

To do so, you just need to run the following script:
```bash
python build_dataset.py
```

This will download the dataset, compute the expected label format, and separate it into train and validation sets.
This will use the validation set of the [Open Images Dataset](https://storage.googleapis.com/openimages/web/download_v7.html), by Google.

More specifically, it will select only pictures with labeled human faces and a permissive enough license.


### Model training

You can then train the BlazeFace model for 100 epochs:
```bash
python trainer.py --epochs 100
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

> As for the training script, you can check the help of the `tf_lite_converter.py` script for more about the available parameters


## Face detection demo in the browser

This is allowing you to test the real time face detection in browser.

### Copy/paste your converted model

If you want to reuse the provided model of this repo, just go to the next step.

If you want to use your trained model, make sure you are in the `javascript` folder of this repo, and then copy/paste the trained model:
```bash
cp ../python/weights/blazeface.tflite assets/.
```

### Running the web demo

Just launch the `index.html` with your favorite web server (you can use you IDE such as PyCharm of VSCode).
