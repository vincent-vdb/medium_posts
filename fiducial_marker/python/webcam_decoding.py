import argparse

import cv2
from ultralytics import YOLO

from decoding_utils import raw_webcam_image_to_tags


def run_decoding_webcam(
        video_input: int, input_model_path: str, confidence_threshold: float, num_layers: int, num_dots_per_layer: int
):
    model = YOLO(input_model_path)
    cap = cv2.VideoCapture(video_input)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        codes, output, unwarped = raw_webcam_image_to_tags(image, model, num_layers, num_dots_per_layer, confidence_threshold)
        if len(codes) > 0:
            print(codes)
        cv2.imshow('YOLO', cv2.flip(output, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run marker decoding from webcam')
    parser.add_argument('--video_input', type=int, help='Video input of the webcam', default=0)
    parser.add_argument('--input_model_path', type=str, help='Path to input yolo model', default='')
    parser.add_argument('--confidence_threshold', type=float, help='Minimum detection confidence threshold', default=0.25)
    parser.add_argument('--num_layers', type=int, help='number of layers in the tag', default=2)
    parser.add_argument('--num_dots_per_layer', type=int, help='number of dots per layers in the tag', default=20)
    args = parser.parse_args()

    run_decoding_webcam(
        args.video_input, args.input_model_path, args.confidence_threshold, args.num_layers, args.num_dots_per_layer
    )
