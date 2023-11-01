import argparse
import os

import cv2
import numpy as np

from ultralytics import YOLO


def run_yolo_webcam(video_input: int, input_model_path: str, confidence_threshold: float):
    model = YOLO(input_model_path)
    cap = cv2.VideoCapture(video_input)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        results = model(image, verbose=False)
        for box in results[0].boxes:
            conf = box.conf.item()
            if conf >= confidence_threshold:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

        cv2.imshow('YOLO', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Makeup gesture webcam demo')
    parser.add_argument('--video_input', type=int, help='Video input of the webcam', default=0)
    parser.add_argument('--input_model_path', type=str, help='Path to input yolo model', default='')
    parser.add_argument('--confidence_threshold', type=float, help='Minimum detection confidence threshold', default=0.25)
    args = parser.parse_args()

    run_yolo_webcam(
        video_input=args.video_input,
        input_model_path=args.input_model_path,
        confidence_threshold=args.confidence_threshold,
    )
