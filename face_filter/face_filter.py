import mediapipe as mp
import cv2
import math
import numpy as np
import utils
import csv
import time

class FaceFilter:
    def __init__(self, display_face_points: bool = False, video: int = 0, sigma: int = 50):
        self.filters_config = {
            'multicolor':
                [{'path': "assets/multicolor_transparent.png",
                  'anno_path': "assets/multicolor_labels.csv",
                  'morph': True, 'animated': False, 'has_alpha': True}],
            'anonymous':
                [{'path': "assets/anonymous_mask.png",
                  'anno_path': "assets/anonymous_labels.csv",
                  'morph': True, 'animated': False, 'has_alpha': True}],
        }
        self.iter_filter_keys = iter(self.filters_config.keys())
        self.filters, self.multi_filter_runtime = self.load_filter(next(self.iter_filter_keys))
        self.display_face_points = display_face_points
        self.sigma = sigma
        self.timer = time.time()
        self.points_dst_prev = None
        self.img_gray_prev = None
        self.video = video
        self.selected_keypoint_indices = [
            127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55, 285,
            296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385, 387, 466,
            373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14, 178, 162, 54,
            67, 10,297, 284, 389,
        ]

    def load_filter(self, filter_name="anonymous"):
        filters = self.filters_config[filter_name]
        multi_filter_runtime = []
        for filter in filters:
            temp_dict = {}

            img_src, img_src_alpha = self.load_filter_img(filter['path'], filter['has_alpha'])

            temp_dict['img'] = img_src
            temp_dict['img_a'] = img_src_alpha

            points = self.load_landmarks(filter['anno_path'])

            temp_dict['points'] = points

            if filter['morph']:
                indexes = np.arange(len(points)).reshape(-1, 1)
                landmarks = [points[str(i)] for i in range(len(indexes))]
                rect = (0, 0, img_src.shape[1], img_src.shape[0])
                dt = utils.calculate_delaunay_triangles(rect, landmarks)

                temp_dict['landmarks'] = landmarks
                temp_dict['indexes'] = indexes
                temp_dict['dt'] = dt

                if len(dt) == 0:
                    continue

            if filter['animated']:
                filter_cap = cv2.VideoCapture(filter['path'])
                temp_dict['cap'] = filter_cap

            multi_filter_runtime.append(temp_dict)

        return filters, multi_filter_runtime

    def switch_filter(self):
        try:
            self.filters, self.multi_filter_runtime = self.load_filter(next(self.iter_filter_keys))
        except StopIteration:
            self.iter_filter_keys = iter(self.filters_config.keys())
            self.filters, self.multi_filter_runtime = self.load_filter(next(self.iter_filter_keys))

    @staticmethod
    def load_landmarks(annotation_file):
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
    def load_filter_img(img_path, has_alpha):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        alpha = None
        if has_alpha:
            b, g, r, alpha = cv2.split(img)
            img = cv2.merge((b, g, r))
        return img, alpha

    def optical_flow_stabilization(self, frame, points_dst, is_first_frame):
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

    def visualize_face_points(self, frame, points_dst):
        viz = np.copy(frame)
        for idx, point in enumerate(points_dst):
            cv2.circle(viz, tuple(point), 2, (255, 0, 0), -1)
        cv2.imshow("landmarks", np.flip(viz, axis=1))

    def apply_morph_filter(self, frame, img_src, img_src_alpha, filter_runtime, points_dst):
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

    def apply_similarity_transform(self, frame, img_src, img_src_alpha, points_src, points_dst):
        dst_points = [
            points_dst[int(list(points_src.keys())[0])],
            points_dst[int(list(points_src.keys())[16])],
            points_dst[int(list(points_src.keys())[71])],
        ]
        tform = utils.similarity_transform(
            [list(points_src.values())[0], list(points_src.values())[16], list(points_src.values())[71]],
            dst_points
        )
        trans_img = cv2.warpAffine(img_src, tform, (frame.shape[1], frame.shape[0]))
        trans_alpha = cv2.warpAffine(img_src_alpha, tform, (frame.shape[1], frame.shape[0]))
        mask_src = cv2.merge((trans_alpha, trans_alpha, trans_alpha))
        mask_src = cv2.GaussianBlur(mask_src, (3, 3), 10)
        mask_dst = np.array([255.0, 255.0, 255.0]) - mask_src
        temp_src = np.multiply(trans_img, (mask_src * (1.0 / 255)))
        temp_dst = np.multiply(frame, (mask_dst * (1.0 / 255)))
        return temp_src + temp_dst

    def add_fps_text(self, output):
        fps = 1 / (time.time() - self.timer)
        self.timer = time.time()
        output = np.uint8(output)
        output = np.flip(output, axis=1)
        return cv2.putText(
            output.astype(np.uint8), f"fps: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1
        )

    def get_landmarks(self, img):
        mp_face_mesh = mp.solutions.face_mesh
        height, width = img.shape[:-1]

        with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print('No face detected')
                return 0

            for face_landmarks in results.multi_face_landmarks:
                values = np.array(face_landmarks.landmark)
                face_keypnts = np.zeros((len(values), 2))

                for idx, value in enumerate(values):
                    face_keypnts[idx][0] = value.x
                    face_keypnts[idx][1] = value.y

                face_keypnts = face_keypnts * (width, height)
                face_keypnts = face_keypnts.astype('int')

                relevant_keypnts = []
                for i in self.selected_keypoint_indices:
                    relevant_keypnts.append(face_keypnts[i])
                return relevant_keypnts
        return 0

    def run(self):
        cap = cv2.VideoCapture(self.video)
        is_first_frame = True
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            points_dst = self.get_landmarks(frame)
            if not points_dst or len(points_dst) != 75:
                continue

            frame, points_dst_prev, img_gray_prev = self.optical_flow_stabilization(frame, points_dst, is_first_frame)
            is_first_frame = False

            if self.display_face_points:
                self.visualize_face_points(frame, points_dst)

            for idx, filter in enumerate(self.filters):
                filter_runtime = self.multi_filter_runtime[idx]
                img_src, points_src, img_src_alpha = filter_runtime['img'], filter_runtime['points'], filter_runtime['img_a']

                if filter['morph']:
                    output = self.apply_morph_filter(frame, img_src, img_src_alpha, filter_runtime, points_dst)
                else:
                    output = self.apply_similarity_transform(frame, img_src, img_src_alpha, points_src, points_dst)

                output = self.add_fps_text(output)
                cv2.imshow("Face Filter", output)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('f'):
                self.switch_filter()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_filter = FaceFilter()
    face_filter.run()
