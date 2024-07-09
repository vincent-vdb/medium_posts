/** BlazeFaceDetector class - inspired from https://github.com/hollance/BlazeFace-PyTorch with Apache 2.0 License */
class BlazeFaceDetector {
    /**
     * Constructor of the BlazeFaceDetector Detector class
     * @param {tf.TFLiteModel} model - The object detection model
     * @param {tf.tensor} anchors - the model anchors
     * @param {float} boxeWidthFactor - a factor to multiply to box width
     * @param {float} boxeHeightFactor - a factor to multiply to box height
     * @param {float} detectionThreshold - the model detection threshold
     * @param {float} iouThreshold - the NMS iou threshold
     * @param {int} maxDetections - the max number of detections out of the NMS
     */
    constructor(
      model,
      anchors,
      boxeWidthFactor=1.,
      boxeHeightFactor=1.,
      detectionThreshold=0.5,
      iouThreshold=0.8,
      maxDetections=10,
    ) {
      this.model = model;
      this.anchors = anchors;
      this.boxWidthFactor = boxeWidthFactor;
      this.boxHeightFactor = boxeHeightFactor;
      this.detectionThreshold = detectionThreshold;
      this.iouThreshold = iouThreshold;
      this.maxDetections = maxDetections;
      this.size = 128;
    }
  
    /**
     * Instantiate a Detector from model filepath
     * @param {string} modelPath - The path to tflite model
     * @param {tf.tensor} anchors - the model anchors
     * @param {float} boxeWidthFactor - a factor to multiply to box width
     * @param {float} boxeHeightFactor - a factor to multiply to box height
     * @param {float} detectionThreshold - the model detection threshold
     * @param {float} iouThreshold - the NMS iou threshold
     * @param {int} maxDetections - the max number of detections out of the NMS
     * @return {Detector} the instantiated Detector object
     */
    static async instantiateDetector(
      modelPath="assets/bface_optim.tflite",
      anchors,
      boxeWidthFactor=1.,
      boxeHeightFactor=1.,
      detectionThreshold=0.5,
      iouThreshold=0.8,
      maxDetections=10,
    ) {
      const model = await tflite.loadTFLiteModel(modelPath);
      return new BlazeFaceDetector(
        model,
        anchors,
        boxeWidthFactor,
        boxeHeightFactor,
        detectionThreshold,
        iouThreshold,
        maxDetections,
      );
    }
  
    postprocessing(rawPreds, xOffset, xFactor, yOffset, yFactor) {
        let boxes = null;
        let faceScores = null;

        let locationData = rawPreds.slice([0, 0, 0], [-1, -1, 4]).squeeze(0);
        let confidenceData = rawPreds.slice([0, 0, 4], [-1, -1, -1]).softmax();
        faceScores = confidenceData.slice([0, 0, 1], [-1, -1, 1]).sigmoid().squeeze([0, 2]);
        // Compute boxes
        let tmpCenter = this.anchors.slice([0, 2], [-1, -1]).mul(locationData.slice([0, 0], [-1, 2])).mul(0.1);
        tmpCenter = tmpCenter.add(this.anchors.slice([0, 0], [-1, 2]));
        let tmpWH = tf.exp(locationData.slice([0, 2], [-1, -1]).mul(0.2))
        tmpWH = tmpWH.mul(this.anchors.slice([0, 2], [-1, -1])).mul(tf.tensor([[this.boxWidthFactor, this.boxHeightFactor]]));
        let tmpXYmin = tmpCenter.sub(tmpWH.mul(0.5)).reverse(-1);
        let tmpXYmax = tmpCenter.add(tmpWH.mul(0.5)).reverse(-1);
        // Apply padding correction
        let xMin = tmpXYmin.slice([0, 1], [-1, 1]).sub(xOffset).mul(xFactor);
        let xMax = tmpXYmax.slice([0, 1], [-1, 1]).sub(xOffset).mul(xFactor);
        let yMin = tmpXYmin.slice([0, 0], [-1, 1]).sub(yOffset).mul(yFactor);
        let yMax = tmpXYmax.slice([0, 0], [-1, 1]).sub(yOffset).mul(yFactor);
        boxes = tf.concat([yMin, xMin, yMax, xMax], 1);

        return {'boxes': boxes, 'faceScores': faceScores};
    }
  
    preprocessing(image, newX, newY, top, bottom, left, right) {
      let inferenceImageBlaze = tf.tidy(() => {
        let imgScaled = image.resizeBilinear([newY, newX]);
        imgScaled = imgScaled.pad([[top, bottom], [left, right], [0, 0]], 127);
        imgScaled = tf.mul(imgScaled, tf.scalar(1/127.5));
        const inferenceImage = tf.expandDims(imgScaled, 0);
        let inferenceImageBlaze = inferenceImage.transpose([0, 3, 1, 2]);
        inferenceImageBlaze = tf.sub(inferenceImageBlaze, tf.scalar(1.));
        return inferenceImageBlaze;
      })
      return inferenceImageBlaze;
    }

    /**
     * Get the predictions from input image: applied model inference and NMS
     * @param {tf.tensor4d} image - The image as a tf.tensor, of shape [batch_size, C, H, W], with rescaled values in [-1, 1]
     * @return {tf.tensor3d} The output bounding boxes as a tensor of shape [batch_size, 5, n_detections]
     * for n_detections detected boxes
     */
    predictOnImage(image, targetSize) {
      // Run preprocessing: image rescaling and padding
      let newX = targetSize, newY = targetSize;
      if (image.shape[0] > image.shape[1]) {
        newX = Math.floor(targetSize * image.shape[1] / image.shape[0]);
      } else {
        newY = Math.floor(targetSize * image.shape[0] / image.shape[1]);
      }
      let top = Math.floor(Math.max(0, newX - newY) / 2);
      let bottom = targetSize - newY - top;
      let left = Math.floor(Math.max(0, newY - newX) / 2);
      let right = targetSize - newX - left;
      let predImage = this.preprocessing(image, newX, newY, top, bottom, left, right);
      // Run model inference
      const rawPreds = this.model.predict(predImage);
      tf.dispose(predImage)
      let postPreds = tf.tidy(() => {
        return this.postprocessing(rawPreds, left / targetSize, targetSize / newX, top / targetSize, targetSize / newY);
     })
      // Apply NMS
      let boxes = postPreds['boxes'];
      let faceScores = postPreds['faceScores'];
      let nmsSelections = null;
      let faceDetections = null;
      if (faceScores !== null) {
        nmsSelections = tf.image.nonMaxSuppression(boxes, faceScores, this.maxDetections, this.iouThreshold, this.detectionThreshold);
        faceDetections = boxes.gather(nmsSelections, 0);
        tf.dispose(nmsSelections);
      }

      tf.dispose(rawPreds);
      tf.dispose(postPreds);
      tf.dispose(boxes);
      tf.dispose(faceScores);

      return {'faceDetections': faceDetections};
    }
  }


  export {BlazeFaceDetector};
