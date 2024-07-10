import {BlazeFaceDetector} from './blazeface.js';


function displayBoxes(src, detections, color = [0, 255, 0, 255]) {
    // Compute and display face box
    if (detections !== null) {
        let box = detections.dataSync();
        console.log('box', box);
        for (let i = 0; i < box.length / 4; i++)
        {
            let xCrop = Math.floor(box[4*i + 1]*webcamElement.width);
            let yCrop = Math.floor(box[4*i]*webcamElement.height);
            let wCrop = Math.floor((box[4*i + 3] - box[4*i + 1])*webcamElement.width);
            let hCrop = Math.floor((box[4*i + 2] - box[4*i])*webcamElement.height);
            let corner1 = new cv.Point(xCrop, yCrop);
            let corner2 = new cv.Point(xCrop + wCrop, yCrop + hCrop);
            cv.rectangle(src, corner1, corner2, color, 2);
        }
    }
}

function processTensor(img) {
    // Inference with BlazeFace
    let outputs = blazeFace.predictOnImage(img, 128);
    if (outputs['faceDetections'].shape[0] > 0) {
        // In case a face is well detected, replace the previous one
        tf.dispose(faceDetections);
        faceDetections = outputs['faceDetections'];
    }

    cap.read(src);
    // Compute and display face box
    displayBoxes(src, faceDetections);
    cv.imshow(canvasElement, src);
    tf.dispose(img);
}

const userAgent = navigator.userAgent;
var canvasElement = document.getElementById("canvas");
var canvasCtx = canvasElement.getContext("2d");
// Load the models
var blazeFace = await BlazeFaceDetector.instantiateDetector("./assets/blazeface.tflite", anchors_blazeface, 1., 1., 0.6, 0.6, 2);

var time = Date.now();
var fps = 0;
var faceDetections = null;

var webcamElement = document.getElementById('webcam');
// OpenCV placeholders
var src = new cv.Mat(webcamElement.height, webcamElement.width, cv.CV_8UC4);
var cap = new cv.VideoCapture(webcamElement);
var tensorToProcess = null;
var processing = false;

async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 640 },
      height: { ideal: 480 },
    },
  });
  // Set webcam size to native values
  let cameraSettings = stream.getVideoTracks()[0].getSettings();
  webcamElement.width = cameraSettings.width;
  webcamElement.height = cameraSettings.height;
  webcamElement.srcObject = stream;
  return new Promise((resolve) => {
    webcamElement.onloadedmetadata = () => {
      resolve(webcamElement);
    };
  });
}

async function captureFrames(webcam) {
  const webcamIterator = await tf.data.webcam(webcam);

  async function capture() {
    tensorToProcess =  await webcamIterator.capture();
    if (!processing) {
      processNextTensor();
    }
    requestAnimationFrame(capture); // Schedule the next frame capture
  }
  requestAnimationFrame(capture); // Start capturing frames
}

function processNextTensor() {
  processing = true;
  if (tensorToProcess !== null) {
    try {
      processTensor(tensorToProcess);
    } catch(error) {
      console.error('Error processing video:', error);
    }
    tensorToProcess.dispose(); // Dispose of the tensor to free up memory
    tensorToProcess = null;
  }
  processing = false;
  // Display information
  document.getElementById('console').innerText = `fps: ${Math.floor(fps*10)/10}`;
  fps = (0.8*fps + 0.2*1000 / (Date.now() - time));
  time = Date.now();
}

(async function main() {
  const webcam = await setupWebcam();
  captureFrames(webcam);
})();
