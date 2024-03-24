import cv2
import numpy as np
from typing import Tuple, List


def constrain_point(p: Tuple[int, int], w: int, h: int) -> Tuple[int, int]:
  """
  Constrains a point to be inside a given boundary.

  Args:
    p: A tuple representing the point (x, y).
    w: The width of the boundary.
    h: The height of the boundary.

  Returns:
    A tuple representing the constrained point.
  """
  constrained_point = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
  return constrained_point


def rect_contains(rect: Tuple[int, int, int, int], point: Tuple[int, int]) -> bool:
  """
  Check if a point is inside a rectangle.

  Args:
    rect: A tuple representing the rectangle (x, y, width, height).
    point: A tuple representing the point (x, y).

  Returns:
    True if the point is inside the rectangle, False otherwise.
  """
  return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def calculate_delaunay_triangles(
        rect: Tuple[int, int, int, int],
        points: List[Tuple[int, int]],
) -> List[Tuple[int, int, int]]:
  """
  Calculate Delaunay triangles for a set of points.

  Args:
    rect: A tuple representing the rectangle in which to calculate the Delaunay triangles.
    points: A list of tuples representing the points.

  Returns:
    A list of tuples, each containing the indices of the 3 points forming a triangle.
  """
  subdiv = cv2.Subdiv2D(rect)

  for p in points:
    subdiv.insert((p[0], p[1]))

  triangle_list = subdiv.getTriangleList()
  delaunay_tri = []

  for t in triangle_list:
    pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])

    if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
      ind = []
      for j, pt in enumerate([pt1, pt2, pt3]):
        for k, p in enumerate(points):
          if abs(pt[0] - p[0]) < 1.0 and abs(pt[1] - p[1]) < 1.0:
            ind.append(k)
      if len(ind) == 3:
        delaunay_tri.append((ind[0], ind[1], ind[2]))

  return delaunay_tri


def apply_affine_transform(
        src: np.ndarray,
        src_tri: List[Tuple[int, int]],
        dst_tri: List[Tuple[int, int]],
        size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Apply affine transform calculated using srcTri and dstTri to src and output an image of size.

  Args:
    src: Source image.
    src_tri: List of tuples representing the vertices of the source triangle.
    dst_tri: List of tuples representing the vertices of the destination triangle.
    size: A tuple representing the size of the output image.

  Returns:
    The transformed image and the warp matrix.
  """
  warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
  dst = cv2.warpAffine(
    src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
  )

  return dst, warp_mat


def warp_triangle(
        img_src: np.ndarray,
        img_dst: np.ndarray,
        t_src: List[Tuple[int, int]],
        t_dst: List[Tuple[int, int]],
) -> np.ndarray:
  """
  Warps and alpha blends triangular regions from img_src and img_dst to img.

  Args:
    img_src: Source image.
    img_dst: Destination image.
    t_src: List of tuples representing the vertices of the triangle in the source image.
    t_dst: List of tuples representing the vertices of the triangle in the destination image.

  Returns:
    The warp matrix.
  """
  r_src = cv2.boundingRect(np.float32([t_src]))
  r_dst = cv2.boundingRect(np.float32([t_dst]))

  t_src_rect = [(pt[0] - r_src[0], pt[1] - r_src[1]) for pt in t_src]
  t_dst_rect = [(pt[0] - r_dst[0], pt[1] - r_dst[1]) for pt in t_dst]
  t_dst_rect_int = [tuple(map(int, pt)) for pt in t_dst_rect]

  mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
  cv2.fillConvexPoly(mask, np.array(t_dst_rect_int), (1.0, 1.0, 1.0), 16, 0)

  img_src_rect = img_src[r_src[1]:r_src[1] + r_src[3], r_src[0]:r_src[0] + r_src[2]]
  size = (r_dst[2], r_dst[3])

  img_dst_rect, warp_mat = apply_affine_transform(img_src_rect, t_src_rect, t_dst_rect, size)
  img_dst_rect = img_dst_rect * mask

  img_dst[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] = img_dst[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] * ((1.0, 1.0, 1.0) - mask)
  img_dst[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] = img_dst[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] + img_dst_rect

  return warp_mat
