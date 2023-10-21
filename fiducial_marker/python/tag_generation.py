import argparse
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
    def __init__(self, kpt_labels, num_slots_for_layer: int = 43, gap_factor: float = 1.3, num_layers: int = 3):
        alpha = 2 * np.pi / num_slots_for_layer
        ellysize = alpha / (2 * gap_factor)
        radius_ratio = 1 / ellysize
        # parameters for whole
        self.gap_factor = gap_factor
        self.radius_ratio = radius_ratio
        self.alpha = alpha
        self.num_layers = num_layers
        self.num_slots_for_layer = num_slots_for_layer
        self.ellysize = ellysize
        # locations of marker points
        markpoints = []
        for i in range(num_slots_for_layer):
            for j in range(num_layers):
                angle = alpha * i
                radius = (num_layers + j + 1) / (2 * num_layers)
                markpoint = MarkPoint(radius * np.cos(angle), radius * np.sin(angle), angle, ellysize * radius)
                markpoints.append(markpoint)
        self.markpoints = markpoints
        # 0 or 1
        self.kpt_labels = kpt_labels
        # range
        self.xylim = [-1.2, 1.2]

    def get_keypoints_with_labels(self):
        x_and_y_with_labels = [
            [markpoint.x, markpoint.y, label] for markpoint, label in zip(self.markpoints, self.kpt_labels)
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

    def draw_runetag(self, scale_in_pixel):
        xylim = self.xylim
        x_and_y_with_labels = self.get_keypoints_with_labels()
        radius_list = self.get_circle_radius()

        w = int(scale_in_pixel * (xylim[1] - xylim[0]))
        image = np.zeros([w,w], dtype= np.float32)
        center = w/2 -0.5

        for radius, kpt_with_id in zip(radius_list, x_and_y_with_labels):
            if  kpt_with_id[2] == 0: continue
            x,y = kpt_with_id[:2]
            Runetag.add_circle(image, x * scale_in_pixel+ center, y * scale_in_pixel+  center, radius * scale_in_pixel)

        return 1-image


def generate_random_tag(
    num_layers: int = 3,
    num_dots_per_layer: int = 43,
    pixel_size: int = 128,
    corner_size: float = 0.,
    output_path: str = None
  ):
    # Generate keypoint labels for tag generation
    codes = np.random.randint(0, 2**num_layers-1, num_dots_per_layer-1)
    keypoint_labels = Runetag.slot_codes_to_binary_ids(codes, num_layers=num_layers, is_outer_first=True)
    runetag = Runetag(keypoint_labels, num_layers=num_layers, num_slots_for_layer=num_dots_per_layer)

    image = runetag.draw_runetag(scale_in_pixel=pixel_size)
    # Mirror the image for consistency
    image = image[:, ::-1]
    # Write image
    if output_path is None:
      output_path = 'tag'
      if corner_size > 0:
        output_path += '_corner'
      output_path += f'_{num_layers}layers{num_dots_per_layer}.png'
    else:
      output_path = output_path
    cv2.imwrite(output_path, image*255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate RUNE tag')
    parser.add_argument('--tag_id', type=int, help='any positive number between 0 and 17000', default=0)
    parser.add_argument('--pixel_size', type=int, help='pixel size of the square side of generated image', default=128)
    parser.add_argument('--corner_size', type=float, help='corner circle size, between 0. and 1., defaults to 0 for no circle', default=0.)
    parser.add_argument('--num_layers', type=int, help='number of layers in the tag', default=2)
    parser.add_argument('--num_dots_per_layer', type=int, help='number of dots per layers in the tag', default=24)
    parser.add_argument('--output_path', type=str, help='path to save the image', default=None)
    args = parser.parse_args()

    generate_random_tag(
        pixel_size=args.pixel_size,
        corner_size=args.corner_size,
        num_layers=args.num_layers,
        num_dots_per_layer=args.num_dots_per_layer,
        output_path=args.output_path,
    )
