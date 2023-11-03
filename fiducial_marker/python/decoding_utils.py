import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from ultralytics import YOLO

from rune_tag import Runetag


def get_blobs_keypoints_from_crop(image: np.array, resize_size: list[int] = (256, 256)):
    """Compute a otsu followed by blob keypoints computation

    Params
    ------
    image: np.array
      input image in black and white
    resize_size: list[int]
      target size of the output image

    Returns
    -------
    list[cv2.KeyPoint]: list of output keypoints, one for each blob detected
    np.array: the resized image after Otsu processing

    """
    # Grayscale conversion
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding & resizing
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(otsu, resize_size)
    # Detect blobs
    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 800
    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.7
    # Set Convexity filtering parameters
    params.filterByConvexity = False
    # Set inertia filtering parameters
    params.filterByInertia = False
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    return detector.detect(resized), resized


def fit_ellipse_from_keypoints(keypoints, n_points: int = 10, display: bool = False):
    """Fit ellipse from keypoints.
    Sort the provided keypoints by side in descending order. Take only the first n_points keypoints.
    Then fit an ellipse out of those points.
    The goal is to keep only the external layer dots.

    Params
    ------
    keypoints: list[cv2.KeyPoint]
      input keypoints
    n_points: int
      number of points to use to fit the ellipse
    display: bool
      True to plot the ellipse fitted on the data. False for no display.

    Returns
    -------
    x, center
    x: list[float]
      list of 5 floats, parameters of the fitted ellipse
    center:
      computed center of the ellipse

    """
    # Inspired from https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
    # Sort keypoints by size
    keypoints = sorted(keypoints, key=lambda x: x.size, reverse=True)
    X = np.array([kpt.pt[0] for kpt in keypoints]).reshape(-1, 1)
    Y = np.array([kpt.pt[1] for kpt in keypoints]).reshape(-1, 1)
    A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])[:n_points]
    b = np.ones_like(X[:n_points])
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

    # Compute the estimated center
    x_coord = np.linspace(X.min() - 10, X.max() + 10, 100)
    y_coord = np.linspace(Y.min() - 10, Y.max() + 10, 100)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    center = np.array([0.5*(X.min() + X.max()), 0.5*(Y.min() + Y.max())])

    if display:
        plt.figure(figsize=(8, 8))
        # Plot the ellipse and data
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
        plt.scatter(X, Y, label='Data Points')
        plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2, label='Ellipse fit')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return x, center


def get_keypoints_on_ellipse(keypoints, ellipse, threshold: float = 0.05) -> list[cv2.KeyPoint]:
    """Get keypoints close to the ellipse, given keypoints and ellipse equation

    Params
    ------
    keypoints: list[cv2.KeyPoint]
      input keypoints
    ellipse: list[float]
      ellipse equation
    threshold: float
      Threshold for discriminating whether a keypoint is on ellipse

    Returns
    -------
    list[cv2.KeyPoint]: the list of keypoints on the ellipse

    """
    res = []
    for kpt in keypoints:
        x, y = kpt.pt
        ellipse_pos = ellipse[0] * x * x + ellipse[1] * x * y + ellipse[2] * y * y + ellipse[3] * x + ellipse[4] * y
        if abs(1 - ellipse_pos) < threshold:
            res.append(kpt)
    return res


def compute_adjacency_matrix_size_corrected(keypoints: list[cv2.KeyPoint], distances: list[list[float]],
                                            sizes: list[float], min_distance: float, threshold_dist: float = 1.5) -> \
dict[int, list[int]]:
    """Compute the adjacency matrix of the keypoints, based on their distances and sizes.

    Any keypoints pair for which distance is < threshold_dist * min distance * relative_size is considered adjacent

    Params
    ------
    keypoints: list[cv2.KeyPoint]
      input keypoints
    distances: list[list[[float]]
      pairwise distances between keypoints
    sizes: list[float]
      sizes of keypoints
    min_distance: float
      Minimum distance found between 2 keypoints
    threshold_dist: float
      Threshold multiplier to apply to select keypoints as adjacent

    Returns
    -------
    dict[int, list[int]]: the adjacency matrix, for each index as key the list of indexes adjacent as value

    """
    adja = {}
    max_size = max(sizes)
    for i in range(len(keypoints)):
        for j in range(i + 1, len(keypoints)):
            threshold = min_distance * threshold_dist
            corrected_threshold = threshold * max_size / min(sizes[i], sizes[j])
            if distances[i][j] < corrected_threshold:
                if i not in adja:
                    adja[i] = []
                if j not in adja:
                    adja[j] = []
                adja[i].append(j)
                adja[j].append(i)

    for key in list(adja.keys()):
        if len(adja[key]) > 2:
            del adja[key]

    return adja


def compute_longest_connected_chain(adja: dict[int, list[int]]) -> list[int]:
    """Compute the longest chain of adjacent keypoints in the given adjacency matrix.

    Params
    ------
    adja: dict[int, list[int]]
      the input adjacency matrix

    Returns
    -------
    list[int]: the list of indexes of the longest chain

    """
    # Look for a list of closest neighboors longer than 4
    max_length = 0
    res = []
    visited = set()
    for edge in adja:
        if edge not in visited and len(adja[edge]) == 1:
            # If not visited and finding a 1-connected edge
            curr_length = 1
            visited.add(edge)
            curr = [edge]
            next_edge = adja[edge][0]
            while next_edge is not None:
                curr_length += 1
                curr.append(next_edge)
                visited.add(next_edge)
                nextnext = None
                if next_edge in adja:
                    # Find the next edge, the one not visited yet
                    for e in adja[next_edge]:
                        if e not in visited:
                            nextnext = e
                            break
                next_edge = nextnext
            if curr_length > max_length:
                max_length = curr_length
                res = curr
    return res


def find_longest_connected_points_on_ellipse(keypoints: list[cv2.KeyPoint], distance_threshold: float = 1.5):
    """Find the longest connected chain on points for given keypoints list.

    Will compute the pairwise distance, the adjacency matrix and the longest chain.
    Assume the given keypoints are already selected to be on ellipse.

    Params
    ------
    keypoints: list[cv2.KeyPoint]
      input keypoints
    distance_threshold: float
      distance threshold to use to compute adjacency matrix

    Returns
    -------
    connected, adja, min_dist
    connected: list[int]
      the list of indexes of the longest chain
    adja: dict[int, list[int]]
      the adjacency matrix
    min_dist: float
      the min distance between two keypoints

    """
    # Compute pairwise distances between keypoints
    distances = pairwise_distances([kpt.pt for kpt in keypoints])
    sizes = [kpt.size for kpt in keypoints]
    # Get the minimum distance
    min_dist = distances[distances > 0.].min()
    # Compute adjacency matrix
    adja = compute_adjacency_matrix_size_corrected(keypoints, distances, sizes, min_dist, distance_threshold)
    # Compute the longest connected chain of points
    connected = compute_longest_connected_chain(adja)
    # Return the points locations
    return connected, adja, min_dist


def is_point_symmetric_from_center(point1: np.array, point2: np.array, center: np.array, min_dist: float):
    """Compute symmetry of two points from a center, given a min_dist error threshold.

    Params
    ------
    point1: np.array
      position of point1
    point2: np.array
      position of point2
    center: np.array
      position of center
    min_dist: float
      threshold distance to consider it symmetric

    Returns
    -------
    is_symmetry, norm
    is_symmetry: bool
      True if symmetric, False otherwise
    norm: float
      the norm between the diff of the two vectors

    """
    vector1 = point1 - center
    vector2 = center - point2
    diff = vector1 - vector2
    norm = np.linalg.norm(diff)
    return norm < min_dist, norm


def find_symmetric_point_in_connected(connected: list[int], kpts_on_ellipse: list[cv2.KeyPoint], center: np.array,
                                      dist_threshold: float) -> list[int]:
    """Find a symmetric point between connected keypoints and kpts_on_ellipse keypoints wrt center.

    Params
    ------
    connected: list[int]
      indexes of connected keypoints in kpts_on_ellipse
    kpts_on_ellipse: list[cv2.KeyPoint]
      list of keypoints in the ellipse
    center: np.array
      position of center
    dist_threshold: float
      threshold distance to consider it symmetric

    Returns
    -------
    list[int]: the pair of keypoints indexes symmetric [keypoint index in connected, symmetric keypoint], None if no symmetry found

    """
    min_dist = None
    res = None
    for i in range(len(kpts_on_ellipse)):
        if i not in connected:
            for j in connected:
                is_symmetric, norm = is_point_symmetric_from_center(
                    np.array(kpts_on_ellipse[i].pt),
                    np.array(kpts_on_ellipse[j].pt),
                    center,
                    dist_threshold
                )
                if is_symmetric:
                    if min_dist is None or norm < min_dist:
                        min_dist = norm
                        res = [i, j]
    return res


def generate_ref_points(connected: list[int], best_symmetry_idx: list[int], num_layers: int = 2,
                        num_dots_per_layer: int = 24, symmetry_index_offset: int = 0) -> np.array:
    """Generate reference points position on which to match, given connected and symmetry index, as well as tag properties.

    Params
    ------
    connected: list[int]
      indexes of connected keypoints in kpts_on_ellipse
    best_symmetry_idx: list[int]
      indexes of keypoint on ellipse and symmetric counterpart
    num_layers: int
      number of layers in tag
    num_dots_per_layer: int
      number of dots in tag
    symmetry_index_offset: int
      offset to add to symmetrical index

    Returns
    -------
    np.array: the positions of the keypoints in the reference plane

    """
    tag = Runetag(num_layers=num_layers, num_dots_per_layer=num_dots_per_layer)
    all_ref_points = tag.generate_dots_keypoints_positions()
    # Keep only a list of linked outer layer points
    ref_points = [all_ref_points[i * num_layers + 1] for i in range(len(connected))]
    # Add the symmetry point if any
    if best_symmetry_idx:
        idx = connected.index(best_symmetry_idx[1]) + symmetry_index_offset
        dot_idx = (idx + 1) * num_layers + num_dots_per_layer + 1
        ref_points.append(all_ref_points[min(dot_idx, len(all_ref_points) - 1)])

    return np.array(ref_points)


def get_keypoints_positions_from_indices(keypoints_on_ellipse: list[cv2.KeyPoint], connected: list[int],
                                         best_symmetry_idx: list[int]) -> np.array:
    """Return the list connected + symmetric keypoints positions.

    Params
    ------
    keypoints_on_ellipse: list[cv2.KeyPoint]
      list of keypoints on the ellipse
    connected: list[int]
      indexes of connected keypoints in kpts_on_ellipse
    best_symmetry_idx: list[int]
      indexes of keypoint on ellipse and symmetric counterpart

    Returns
    -------
    np.array: the positions of the keypoints in the reference plane

    """
    res = [keypoints_on_ellipse[i].pt for i in connected]
    if best_symmetry_idx is not None:
        res.append(keypoints_on_ellipse[best_symmetry_idx[0]].pt)
    return np.array(res)


def unwarp_tag_image(image: np.array, num_layers: int, num_dots_per_layer: int, min_num_keypoints: int,
                     unwarping_offset: int = 2) -> list[np.array]:
    """Return the list connected + symmetric keypoints positions.

    Params
    ------
    image: np.array
      the input color image
    num_layers: int
      number of layers in tag
    num_dots_per_layer: int
      number of dots in tag
    min_num_keypoints: int
      the min number of keypoints to detect to process it*
    unwarping_offset: int
      the min and max dot offset to test for unwarping. If 5, will test offset in range [-5, 5], step of 1

    Returns
    -------
    list[np.array]: a list of 5 unwarped images, for different unwarping references

    """
    if image.max() <= 1.:
        image = (image * 255).astype(np.uint8)
    # Detect blobs
    keypoints, resize = get_blobs_keypoints_from_crop(image)
    if len(keypoints) < min_num_keypoints:
        return image, None
    # Compute the ellipse from those keypoints
    ellipse, center = fit_ellipse_from_keypoints(keypoints)
    # Get keypoints on ellipse
    kpts_on_ellipse = get_keypoints_on_ellipse(keypoints, ellipse)
    # Get connected keypoints indexes
    connected, _, min_dist = find_longest_connected_points_on_ellipse(kpts_on_ellipse)
    if len(connected) < 4:
        return image, None
    # Get symmetrical point from center if existing
    best_symmetry_idx = find_symmetric_point_in_connected(connected, kpts_on_ellipse, center, min_dist)

    corrected_images = []
    for offset_index in range(-unwarping_offset, unwarping_offset + 1):
        # Get reference points
        ref_points = generate_ref_points(connected, best_symmetry_idx, num_layers, num_dots_per_layer, offset_index)
        # Get destination points
        dst_points = get_keypoints_positions_from_indices(kpts_on_ellipse, connected, best_symmetry_idx)
        # Compute homography
        H, _ = cv2.findHomography(dst_points, ref_points, 0, 5.0)
        # Correct image
        if H is not None:
            corrected_image = cv2.warpPerspective(resize, H, (307, 307))
            corrected_images.append(corrected_image)

    return corrected_images


def crop_image_with_box(image: np.array, bbox: list[int], border: int = 5) -> np.array:
    """Return cropped image with given box and border offset.

    Params
    ------
    image: np.array
      the input color image
    bbox: list[int]
      the input bounding box [x_min, y_min, x_max, y_max]
    border: int
      a border offset to add to the box

    Returns
    -------
    np.array: the cropped image

    """
    return image[max(0, int(bbox[1]) - border):min(image.shape[0], int(bbox[3]) + border),
           max(0, int(bbox[0]) - border):min(image.shape[1], int(bbox[2]) + border)]


def unwarp_tags_from_detections(image: np.array, boxes: np.array, num_layers: int, num_dots_per_layer: int,
                                min_num_keypoints: int = 10):
    """Unwarp tags on image from detections.

    Params
    ------
    image: np.array
      the input color image
    boxes: np.array
      the bounding boxes
    num_layers: int
      number of layers in tag
    num_dots_per_layer: int
      number of dots in tag
    detection_threshold: float
      detection threshold to confidence score to apply for filtering
    min_num_keypoints: int
      the min number of keypoints to detect to process it

    Returns
    -------
    list[np.array]: list of unwarped detected tags.

    """
    res_images = []
    for box in boxes:
        crop = crop_image_with_box(image, box)
        unwarped_images = unwarp_tag_image(crop, num_layers, num_dots_per_layer, min_num_keypoints)
        res_images.append(unwarped_images)

    return res_images


def get_boxes_from_model(
        image: np.array, model: YOLO, detection_threshold: float = 0.5, draw_boxes: bool = False
):
    """Compute and get the bounding boxes from the model inference on the given image

    Params
    ------
    image: np.array
      the input color image
    model: YOLO
      YOLO model
    detection_threshold: float
      detection threshold to confidence score to apply for drawing, only applies if draw_boxes=True
    draw_boxes: bool
      True to draw the boxes on image, False otherwise

    Returns
    -------
    image, scores, boxes
    image: np.array
      the input image, with drawings if draw_boxes set to True
    scores:
      np.array: the detection scores
    boxes:
      np.array: the boxes

    """
    results = model(image, verbose=False)
    scores = []
    boxes = []
    for box in results[0].boxes:
        conf = box.conf.item()
        if conf >= detection_threshold:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            if draw_boxes:
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

    return image, scores, boxes


def raw_webcam_image_to_tags(image: np.array, model: YOLO, num_layers: int,
                             num_dots_per_layer: int, detection_threshold: float = 0.25, draw_boxes: bool = True,
                             display_tags: bool = True):
    """Unwarp tags on image from detections.

    Params
    ------
    image: np.array
      the input raw webcam image
    model: YOLO
      YOLO model
    num_layers: int
      number of layers in tag
    num_dots_per_layer: int
      number of dots in tag
    detection_threshold: float
      detection threshold to confidence score to apply for drawing, only applies if draw_boxes=True
    draw_boxes: bool
      True to draw the boxes on image, False otherwise

    Returns
    -------
    codes, output, unwarped
    codes: list[int]
      list of decoded codes
    output: np.array
      image with drawed boxes
    unwarped: list[list[images]]
      a list of list of unwarped images

    """
    # Convert and resize the image
    #image = cv2.resize(image, (640, 640))
    # Run tag detection model
    output, scores, boxes = get_boxes_from_model(
        image, model, detection_threshold=detection_threshold, draw_boxes=draw_boxes
    )
    # Unwarp the image
    unwarped = unwarp_tags_from_detections(image, boxes, num_layers, num_dots_per_layer)
    # Decode image
    codes = []
    if len(unwarped) > 0:
        for i in range(len(unwarped)):
            for offset in range(len(unwarped[i])):
                if unwarped[i][offset] is not None and unwarped[i][offset].shape == (307, 307):
                    code = decode_tag_from_image(unwarped[i][offset], num_layers, num_dots_per_layer)
                    if code != -1:
                        codes.append(code)
                        if display_tags:
                            cv2.putText(unwarped[i][offset], f'tag {code}', (127, 127), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (128, 128, 128), 2)
                            cv2.imshow(f'tag{i}', unwarped[i][offset])
                        break

    return codes, output, unwarped


def compute_binary_code_from_image(image: np.array, num_layers: int, num_dots_per_layer: int, area_radius: int = 4) -> \
list[int]:
    """Compute the tag binary code from an unwarped image.

    Params
    ------
    image: np.array
      the input unwparped, black and white image
    num_layers: int
      number of layers in tag
    num_dots_per_layer: int
      number of dots in tag
    area_radius: int
      radius in pixels in which to look for dot presence

    Returns
    -------
    list[int]: binary code of the tag

    """
    # Compute dot centers
    tag = Runetag(num_layers=num_layers, num_dots_per_layer=num_dots_per_layer)
    dot_centers = tag.generate_dots_keypoints_positions()
    # Define helper variables and output
    bin_code = []
    curr_layer = num_layers - 1
    curr_code = 0
    for dot_center in dot_centers:
        # Check if dot is present
        x, y = dot_center
        is_dot_area = np.any(
            image[int(y) - area_radius: int(y) + area_radius + 1, int(x) - area_radius:int(x) + area_radius + 1] < 128)
        # Compute code
        curr_code += is_dot_area * (2 ** curr_layer)
        curr_layer -= 1
        # If last layer, add code and reset all
        if curr_layer < 0:
            bin_code.append(curr_code - 1)
            curr_code = 0
            curr_layer = num_layers - 1
    # Make sure code right and on place
    if -1 not in bin_code:
        return None
    end_index = bin_code.index(-1)
    if end_index != len(bin_code) - 1:
        bin_code = bin_code[end_index + 1:] + bin_code[:end_index + 1]

    return bin_code


def decode_tag_from_image(image: np.array, num_layers: int,
                          num_dots_per_layer: int) -> int:
    """Decode tag from a given image (assumed unwarped) and a code table.

    Params
    ------
    image: np.array
      the input unwparped, black and white image
    codes_table: dict[str, int]
      the code map, with binary code as key, output code as value
    num_layers: int
      number of layers in tag
    num_dots_per_layer: int
      number of dots in tag


    Returns
    -------
    int: the decoded code, -1 if no code found

    """
    # Get the binary code
    bin_code = compute_binary_code_from_image(image, num_layers, num_dots_per_layer)
    if bin_code is None:
        return -1
    # Compute a final value
    power = 2 ** num_layers - 1
    res = 0
    for i in range(len(bin_code) - 1):
        if bin_code[i] == -1:
            return -1
        res += bin_code[i] * power ** i
    return res
