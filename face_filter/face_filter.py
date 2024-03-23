import mediapipe as mp
import cv2
import math
import numpy as np
import utils
import csv
import time

class FaceFilter:
    def __init__(self):
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
        self.VISUALIZE_FACE_POINTS = True
        self.sigma = 50
        self.timer = time.time()
        self.points2Prev = None
        self.img2GrayPrev = None

    def load_filter(self, filter_name="anonymous"):
        filters = self.filters_config[filter_name]

        multi_filter_runtime = []

        for filter in filters:
            temp_dict = {}

            img1, img1_alpha = self.load_filter_img(filter['path'], filter['has_alpha'])

            temp_dict['img'] = img1
            temp_dict['img_a'] = img1_alpha

            points = self.load_landmarks(filter['anno_path'])

            temp_dict['points'] = points

            if filter['morph']:
                hullIndex = np.arange(len(points)).reshape(-1, 1)
                hull = [points[str(i)] for i in range(len(hullIndex))]
                sizeImg1 = img1.shape
                rect = (0, 0, sizeImg1[1], sizeImg1[0])
                dt = utils.calculate_delaunay_triangles(rect, hull)

                temp_dict['hull'] = hull
                temp_dict['hullIndex'] = hullIndex
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

    def run(self):
        cap = cv2.VideoCapture(0)
        isFirstFrame = True
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            points2 = self.get_landmarks(frame)
            if not points2 or len(points2) != 75:
                continue

            frame, points2Prev, img2GrayPrev = self.optical_flow_stabilization(frame, points2, isFirstFrame)
            isFirstFrame = False

            if self.VISUALIZE_FACE_POINTS:
                self.visualize_face_points(frame, points2)

            for idx, filter in enumerate(self.filters):
                filter_runtime = self.multi_filter_runtime[idx]
                img1, points1, img1_alpha = filter_runtime['img'], filter_runtime['points'], filter_runtime['img_a']

                if filter['morph']:
                    output = self.apply_morph_filter(frame, img1, img1_alpha, filter_runtime, points2)
                else:
                    output = self.apply_similarity_transform(frame, img1, img1_alpha, points1, points2)

                output = self.add_fps_text(output)
                cv2.imshow("Face Filter", output)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('f'):
                self.switch_filter()

        cap.release()
        cv2.destroyAllWindows()

    def optical_flow_stabilization(self, frame, points2, isFirstFrame):
        img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if isFirstFrame:
            self.points2Prev = np.array(points2, np.float32)
            self.img2GrayPrev = np.copy(img2Gray)
            return frame, self.points2Prev, self.img2GrayPrev

        lk_params = dict(winSize=(101, 101), maxLevel=15,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
        points2Next, _, _ = cv2.calcOpticalFlowPyrLK(self.img2GrayPrev, img2Gray, self.points2Prev,
                                                     np.array(points2, np.float32),
                                                     **lk_params)

        for k in range(len(points2)):
            d = cv2.norm(np.array(points2[k]) - points2Next[k])
            alpha = math.exp(-d * d / self.sigma)
            points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
            points2[k] = utils.constrain_point(points2[k], frame.shape[1], frame.shape[0])
            points2[k] = (int(points2[k][0]), int(points2[k][1]))

        self.points2Prev = np.array(points2, np.float32)
        self.img2GrayPrev = img2Gray

        return frame, self.points2Prev, self.img2GrayPrev

    def visualize_face_points(self, frame, points2):
        viz = np.copy(frame)
        for idx, point in enumerate(points2):
            cv2.circle(viz, tuple(point), 2, (255, 0, 0), -1)
        cv2.imshow("landmarks", np.flip(viz, axis=1))

    def apply_morph_filter(self, frame, img1, img1_alpha, filter_runtime, points2):
        hullIndex, dt, hull1 = filter_runtime['hullIndex'], filter_runtime['dt'], filter_runtime['hull']

        warped_img = np.copy(frame)
        hull2 = [points2[i[0]] for i in hullIndex]
        mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
        mask1 = cv2.merge((mask1, mask1, mask1))
        img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

        for i in range(len(dt)):
            t1, t2 = [], []
            for j in range(3):
                t1.append(hull1[dt[i][j]])
                t2.append(hull2[dt[i][j]])
            utils.warp_triangle(img1, warped_img, t1, t2)
            utils.warp_triangle(img1_alpha_mask, mask1, t1, t2)

        mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
        mask2 = (255.0, 255.0, 255.0) - mask1
        temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
        temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
        return temp1 + temp2

    def apply_similarity_transform(self, frame, img1, img1_alpha, points1, points2):
        dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[16])],
                      points2[int(list(points1.keys())[71])]]
        tform = utils.similarity_transform(
            [list(points1.values())[0], list(points1.values())[16], list(points1.values())[71]],
            dst_points
        )
        trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
        trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
        mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))
        mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
        mask2 = (255.0, 255.0, 255.0) - mask1
        temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
        temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
        return temp1 + temp2

    def add_fps_text(self, output):
        fps = 1 / (time.time() - self.timer)
        self.timer = time.time()
        output = np.uint8(output)
        output = np.flip(output, axis=1)
        return cv2.putText(output.astype(np.uint8), f"fps: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                           (255, 0, 0), 1)

    @staticmethod
    def get_landmarks(img):
        mp_face_mesh = mp.solutions.face_mesh
        selected_keypoint_indices = [
            127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55, 285,
            296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385, 387, 466,
            373,
            380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14, 178, 162, 54, 67,
            10,
            297, 284, 389
        ]
        height, width = img.shape[:-1]

        with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print('Face not detected!!!')
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

                for i in selected_keypoint_indices:
                    relevant_keypnts.append(face_keypnts[i])
                return relevant_keypnts
        return 0


if __name__ == "__main__":
    face_filter = FaceFilter()
    face_filter.run()
