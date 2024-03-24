import csv
import math
import time
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import utils


class FaceFilter:
    def __init__(self, display_face_points: bool = False, video: int = 0, sigma: int = 50) -> None:
        """
        Initializes the FaceFilter class with configurations for filters, video source, and other settings.

        Args:
            display_face_points (bool): Whether to display face points on the video. Defaults to False.
            video (int): Video source. Defaults to 0.
            sigma (int): Sigma value for optical flow stabilization. Defaults to 50.
        """
        self.filters_config: dict = {
            'multicolor':
                [{'path': "assets/multicolor_transparent.png",
                  'anno_path': "assets/multicolor_labels.csv",
                  'has_alpha': True}],
            'anonymous':
                [{'path': "assets/anonymous_mask.png",
                  'anno_path': "assets/anonymous_labels.csv",
                  'has_alpha': True}],
        }
        self.iter_filter_keys = iter(self.filters_config.keys())
        self.filters, self.multi_filter_runtime = self.load_filter(next(self.iter_filter_keys))
        self.display_face_points = display_face_points
        self.sigma = sigma
        self.timer = time.time()
        self.points_dst_prev: Optional[np.ndarray] = None
        self.img_gray_prev: Optional[np.ndarray] = None
        self.video = video
        self.selected_keypoint_indices: list[int] = [
            127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55, 285,
            296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385, 387, 466,
            373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14, 178, 162, 54,
            67, 10,297, 284, 389,
        ]

    def load_filter(self, filter_name: str = "anonymous") -> tuple:
        """
        Loads filter configurations and runtime data for a given filter name.

        Args:
            filter_name (str): The name of the filter to load. Defaults to "anonymous".

        Returns:
            tuple: A tuple containing filter configurations and runtime data.
        """
        filters = self.filters_config[filter_name]
        multi_filter_runtime = []
        for filter in filters:
            temp_dict = {}
            img_src, img_src_alpha = self.load_filter_img(filter['path'], filter['has_alpha'])

            temp_dict['img'] = img_src
            temp_dict['img_a'] = img_src_alpha

            points = self.load_landmarks(filter['anno_path'])
            temp_dict['points'] = points

            indexes = np.arange(len(points)).reshape(-1, 1)
            landmarks = [points[str(i)] for i in range(len(indexes))]
            rect = (0, 0, img_src.shape[1], img_src.shape[0])
            dt = utils.calculate_delaunay_triangles(rect, landmarks)
            temp_dict['landmarks'] = landmarks
            temp_dict['indexes'] = indexes
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

            multi_filter_runtime.append(temp_dict)

        return filters, multi_filter_runtime

    def switch_filter(self) -> None:
        """
        Switches to the next filter in the configuration.
        """
        try:
            self.filters, self.multi_filter_runtime = self.load_filter(next(self.iter_filter_keys))
        except StopIteration:
            self.iter_filter_keys = iter(self.filters_config.keys())
            self.filters, self.multi_filter_runtime = self.load_filter(next(self.iter_filter_keys))

    @staticmethod
    def load_landmarks(annotation_file: str) -> dict[str, tuple[int, int]]:
        """
        Loads landmarks from a CSV annotation file.

        Args:
            annotation_file (str): Path to the CSV file containing landmarks.

        Returns:
            dict[str, tuple[int, int]]: A dictionary of landmarks.
        """
        with open(annotation_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            points = {}
            for i, row in enumerate(csv_reader):
                try:
                    x, y = int(row[1]), int(row[2])
                    points[row[0]] = (x, y)
                except ValueError:
                    continue
            return points

    @staticmethod
    def load_filter_img(img_path: str, has_alpha: bool) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Loads an image and optionally its alpha channel.

        Args:
            img_path (str): Path to the image file.
            has_alpha (bool): Whether the image has an alpha channel.

        Returns:
            tuple[np.ndarray, Optional[np.ndarray]]: The image and its alpha channel if present.
        """
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        alpha = None
        if has_alpha:
            b, g, r, alpha = cv2.split(img)
            img = cv2.merge((b, g, r))
        return img, alpha

    def optical_flow_stabilization(self, frame: np.ndarray, points_dst: list[tuple[int, int]], is_first_frame: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies optical flow stabilization to the given frame based on destination points.

        Args:
            frame (np.ndarray): The current video frame.
            points_dst (list[tuple[int, int]]): Destination points for optical flow stabilization.
            is_first_frame (bool): Indicates if this is the first frame of the video.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The stabilized frame, previous destination points, and previous grayscale image.
        """
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_first_frame:
            self.points_dst_prev = np.array(points_dst, np.float32)
            self.img_gray_prev = np.copy(img_gray)
            return frame, self.points_dst_prev, self.img_gray_prev

        lk_params = dict(
            winSize=(101, 101), maxLevel=15, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001)
        )
        points_dst_next, _, _ = cv2.calcOpticalFlowPyrLK(
            self.img_gray_prev,
            img_gray,
            self.points_dst_prev,
            np.array(points_dst, np.float32),
            **lk_params
        )

        for k in range(len(points_dst)):
            d = cv2.norm(np.array(points_dst[k]) - points_dst_next[k])
            alpha = math.exp(-d * d / self.sigma)
            points_dst[k] = (1 - alpha) * np.array(points_dst[k]) + alpha * points_dst_next[k]
            points_dst[k] = utils.constrain_point(points_dst[k], frame.shape[1], frame.shape[0])
            points_dst[k] = (int(points_dst[k][0]), int(points_dst[k][1]))

        self.points_dst_prev = np.array(points_dst, np.float32)
        self.img_gray_prev = img_gray

        return frame, self.points_dst_prev, self.img_gray_prev

    @staticmethod
    def visualize_face_points(frame: np.ndarray, points_dst: list[tuple[int, int]]) -> None:
        """
        Visualizes face points on the given frame.

        Args:
            frame (np.ndarray): The current video frame.
            points_dst (list[tuple[int, int]]): Face points to visualize.
        """
        viz = np.copy(frame)
        for idx, point in enumerate(points_dst):
            cv2.circle(viz, tuple(point), 2, (255, 0, 0), -1)
        cv2.imshow("landmarks", np.flip(viz, axis=1))

    @staticmethod
    def apply_morph_filter(
            frame: np.ndarray,
            img_src: np.ndarray,
            img_src_alpha: np.ndarray,
            filter_runtime: dict,
            points_dst: list[tuple[int, int]]
    ) -> np.ndarray:
        """
        Applies a morphing filter to the frame based on source and destination points.

        Args:
            frame (np.ndarray): The current video frame.
            img_src (np.ndarray): Source image for the filter.
            img_src_alpha (np.ndarray): Alpha channel of the source image.
            filter_runtime (dict): Runtime data for the filter.
            points_dst (list[tuple[int, int]]): Destination points for the filter.

        Returns:
            np.ndarray: The frame with the morphing filter applied.
        """
        indexes, dt, landmarks_src = filter_runtime['indexes'], filter_runtime['dt'], filter_runtime['landmarks']
        warped_img = np.copy(frame)
        landmarks_dst = [points_dst[i[0]] for i in indexes]
        mask_src = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
        mask_src = cv2.merge((mask_src, mask_src, mask_src))
        img_src_alpha_mask = cv2.merge((img_src_alpha, img_src_alpha, img_src_alpha))

        for i in range(len(dt)):
            t_src, t_dst = [], []
            for j in range(3):
                t_src.append(landmarks_src[dt[i][j]])
                t_dst.append(landmarks_dst[dt[i][j]])
            utils.warp_triangle(img_src, warped_img, t_src, t_dst)
            utils.warp_triangle(img_src_alpha_mask, mask_src, t_src, t_dst)

        mask_src = cv2.GaussianBlur(mask_src, (3, 3), 10)
        mask_dst = np.array([255.0, 255.0, 255.0]) - mask_src
        temp_src = np.multiply(warped_img, (mask_src * (1.0 / 255)))
        temp_dst = np.multiply(frame, (mask_dst * (1.0 / 255)))
        return temp_src + temp_dst

    def add_fps_text(self, output: np.ndarray) -> np.ndarray:
        """
        Adds FPS text to the output frame.

        Args:
            output (np.ndarray): The output frame to add FPS text to.

        Returns:
            np.ndarray: The output frame with FPS text added.
        """
        fps = 1 / (time.time() - self.timer)
        self.timer = time.time()
        output = np.uint8(output)
        output = np.flip(output, axis=1)
        return cv2.putText(
            output.astype(np.uint8), f"fps: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1
        )

    def get_landmarks(self, img: np.ndarray) -> Optional[list[tuple[int, int]]]:
        """
        Extracts relevant landmarks from the given image.

        Args:
            img (np.ndarray): The image to extract landmarks from.

        Returns:
            Optional[list[tuple[int, int]]]: A list of relevant landmarks or 0 if no face is detected.
        """
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        height, width = img.shape[:-1]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results = detector.detect(mp_image)
        if not results.face_landmarks:
            print('No face detected')
            return None

        for face_landmarks in results.face_landmarks:

            face_keypnts = []
            for normalized_landmark in face_landmarks:
                face_keypnts.append([normalized_landmark.x, normalized_landmark.y])

            face_keypnts = np.array(face_keypnts)
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []
            for i in self.selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
        return None

    def run(self) -> None:
        """
        Main loop for running the face filter application.
        """
        cap = cv2.VideoCapture(self.video)
        is_first_frame = True
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            points_dst = self.get_landmarks(frame)
            if points_dst is None or len(points_dst) != 75:
                continue

            frame, points_dst_prev, img_gray_prev = self.optical_flow_stabilization(frame, points_dst, is_first_frame)
            is_first_frame = False

            if self.display_face_points:
                self.visualize_face_points(frame, points_dst)

            for idx, filter in enumerate(self.filters):
                filter_run = self.multi_filter_runtime[idx]
                img_src, points_src, img_src_alpha = filter_run['img'], filter_run['points'], filter_run['img_a']
                output = self.apply_morph_filter(frame, img_src, img_src_alpha, filter_run, points_dst)
                output = self.add_fps_text(output)
                cv2.imshow("Face Filter", output)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('f'):
                self.switch_filter()

        cap.release()
        cv2.destroyAllWindows()
