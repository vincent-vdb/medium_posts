import argparse
from face_filter import FaceFilter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply webcam filter with optional parameters.')
    parser.add_argument('--display_face_points', action='store_true', help='Display face points on the output video.')
    parser.add_argument('--video', type=int, default=0, help='Video source index. Default is 0 for the primary camera.')

    args = parser.parse_args()

    face_filter = FaceFilter(display_face_points=args.display_face_points, video=args.video)
    face_filter.run()
