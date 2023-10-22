import math

import cv2
import numpy as np


class MarkPoint:
    def __init__(self, x, y, angle, ellyradius):
        self.x = x
        self.y = y
        self.angle = angle
        self.ellyradius = ellyradius


class Runetag:
    def __init__(self, num_layers: int = 3, num_dots_per_layer: int = 43, gap_factor: float = 1.3):
        alpha = 2 * np.pi / num_dots_per_layer
        ellysize = alpha / (2 * gap_factor)
        radius_ratio = 1 / ellysize
        # parameters for whole
        self.gap_factor = gap_factor
        self.radius_ratio = radius_ratio
        self.alpha = alpha
        self.num_layers = num_layers
        self.num_dots_per_layer = num_dots_per_layer
        self.ellysize = ellysize
        # locations of marker points
        markpoints = []
        for i in range(num_dots_per_layer):
            for j in range(num_layers):
                angle = alpha * i
                radius = (num_layers + j + 1) / (2 * num_layers)
                markpoint = MarkPoint(radius * np.cos(angle), radius * np.sin(angle), angle, ellysize * radius)
                markpoints.append(markpoint)
        self.markpoints = markpoints
        # range
        self.xylim = [-1.2, 1.2]

    def get_keypoints_with_labels(self, keypoints):
        x_and_y_with_labels = [
            [markpoint.x, markpoint.y, label] for markpoint, label in zip(self.markpoints, keypoints)
        ]
        return x_and_y_with_labels

    def get_circle_radius(self):
        radius_list = [markpoint.ellyradius for markpoint in self.markpoints]
        return radius_list

    @staticmethod
    def slot_codes_to_binary_ids(codes, num_layers, is_outer_first = False):
        if num_layers==1:
          sub_codes = {0: [0], 1: [1]}
        else:
          sub_codes = {}
          for ii in range(2**num_layers-1):
              sub_code = []
              num = ii +1
              for _ in range(num_layers):
                  sub_code.append(num % 2)
                  num //=2
              if is_outer_first:
                  # from outer to inner
                  sub_code = sub_code[::-1]
              sub_codes[ii] = sub_code
        # from inside to outside
        binary_ids = []
        for code in codes:
            for ii in range(num_layers):
                binary_ids.append(sub_codes[code][ii])

        return binary_ids

    @staticmethod
    def add_circle(image, x, y, radius):
        tl = [int(x - radius - 1), int(y - radius - 1)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + radius + 1), int(y + radius + 1)]
        map_h, map_w = image.shape
        br[0] = min(br[0], map_w)
        br[1] = min(br[1], map_h)

        for map_y in range(tl[1], br[1]):
            for map_x in range(tl[0], br[0]):
                d2 = (map_x - x) * (map_x - x) + (map_y - y) * (map_y - y)
                d = math.sqrt(d2)
                if d < radius:
                    image[map_y, map_x] = 1

    def draw_runetag(self, scale_in_pixel, keypoints):
        xylim = self.xylim
        x_and_y_with_labels = self.get_keypoints_with_labels(keypoints)
        radius_list = self.get_circle_radius()

        w = int(scale_in_pixel * (xylim[1] - xylim[0]))
        image = np.zeros([w,w], dtype= np.float32)
        center = w/2 -0.5

        for radius, kpt_with_id in zip(radius_list, x_and_y_with_labels):
            if  kpt_with_id[2] == 0: continue
            x,y = kpt_with_id[:2]
            Runetag.add_circle(image, x * scale_in_pixel+ center, y * scale_in_pixel+  center, radius * scale_in_pixel)

        return 1-image


    def generate_random_tag(self, pixel_size: int = 128, output_path: str = None, write_file: bool = True):
        # Generate keypoint labels for tag generation
        codes = np.random.randint(0, 2**self.num_layers-1, self.num_dots_per_layer-1)
        keypoint_labels = Runetag.slot_codes_to_binary_ids(codes, num_layers=self.num_layers, is_outer_first=True)
        image = self.draw_runetag(scale_in_pixel=pixel_size, keypoints=keypoint_labels)
        # Mirror the image for consistency
        image = image[:, ::-1]
        # Write image
        if output_path is None:
            output_path = f'tag_{self.num_layers}layers{self.num_dots_per_layer}.png'
        else:
            output_path = output_path
        if write_file:
            cv2.imwrite(output_path, image*255)
        return image
