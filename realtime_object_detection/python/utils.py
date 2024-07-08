import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision


def od_collate_fn(batch: list) -> tuple:
    """Collate function for object detection.

    Args:
        batch (list): List of tuples containing images and targets.

    Returns:
        tuple: Batch of images and targets.
    """
    imgs = []
    targets = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


def encode(matched: torch.Tensor, priors: torch.Tensor, variances: list) -> torch.Tensor:
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.

    Args:
        matched (torch.Tensor): Coords of ground truth for each prior in point-form.
            Shape: [num_priors, 4].
        priors (torch.Tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances (list[float]): Variances of priorboxes.

    Returns:
        torch.Tensor: Encoded boxes, Shape: [num_priors, 4].
    """
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def jaccard(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute the jaccard overlap of two sets of boxes. The jaccard overlap
    is simply the intersection over union of two boxes. Here we operate on
    ground truth boxes and default boxes.

    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    Args:
        box_a (torch.Tensor): Ground truth bounding boxes, Shape: [num_objects, 4].
        box_b (torch.Tensor): Prior boxes from priorbox layers, Shape: [num_priors, 4].

    Returns:
        torch.Tensor: Jaccard overlap, Shape: [box_a.size(0), box_b.size(0)].
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def point_form(boxes: torch.Tensor) -> torch.Tensor:
    """Convert prior_boxes to (xmin, ymin, xmax, ymax) representation for comparison
    to point form ground truth data.

    Args:
        boxes (torch.Tensor): Center-size default boxes from priorbox layers.

    Returns:
        torch.Tensor: Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Resize both tensors to [A, B, 2] without new malloc and compute the area
    of intersect between box_a and box_b.

    Args:
        box_a (torch.Tensor): Bounding boxes, Shape: [A, 4].
        box_b (torch.Tensor): Bounding boxes, Shape: [B, 4].

    Returns:
        torch.Tensor: Intersection area, Shape: [A, B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def match(threshold: float, truths: torch.Tensor, priors: torch.Tensor, variances: list,
          labels: torch.Tensor, loc_t: torch.Tensor, conf_t: torch.Tensor, idx: int) -> None:
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.

    Args:
        threshold (float): The overlap threshold used when matching boxes.
        truths (torch.Tensor): Ground truth boxes, Shape: [num_obj, num_priors].
        priors (torch.Tensor): Prior boxes from priorbox layers, Shape: [n_priors, 4].
        variances (list[float]): Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels (torch.Tensor): All the class labels for the image, Shape: [num_obj].
        loc_t (torch.Tensor): Tensor to be filled with encoded location targets.
        conf_t (torch.Tensor): Tensor to be filled with matched indices for conf preds.
        idx (int): Current batch index.

    Returns:
        None
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1, num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1, num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors, 4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors, 4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh: float = 0.5, neg_pos: int = 3,
                 device: str = 'cpu', dbox_list: torch.Tensor = None) -> None:
        """Initialize MultiBoxLoss.

        Args:
            jaccard_thresh (float, optional): Jaccard threshold. Defaults to 0.5.
            neg_pos (int, optional): Negative-positive ratio. Defaults to 3.
            device (str, optional): Device to use. Defaults to 'cpu'.
            dbox_list (torch.Tensor, optional): Default box list. Defaults to None.
        """
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = neg_pos
        self.device = device
        self.dbox_list = dbox_list

    def forward(self, predictions: torch.Tensor, targets: list) -> tuple:
        """Forward pass for MultiBoxLoss.

        Args:
            predictions (torch.Tensor): Predictions from the model.
            targets (list): Ground truth targets.

        Returns:
            tuple: Localization loss and confidence loss.
        """
        loc_data = predictions[:, :, :4]
        conf_data = predictions[:, :, 4:]

        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        conf_t_label = torch.zeros((num_batch, num_dbox), dtype=torch.int64, device=self.device)
        loc_t = torch.zeros((num_batch, num_dbox, 4), dtype=torch.float32, device=self.device)

        for idx in range(num_batch):
            if len(targets[idx]) > 0:
                truths = targets[idx][:, :-1].to(self.device)  # BBox
                labels = targets[idx][:, -1].to(self.device)
                dbox = self.dbox_list.to(self.device)
                variance = [0.1, 0.2]
                match(self.jaccard_thresh, truths, dbox,
                      variance, labels, loc_t, conf_t_label, idx)

        pos_mask = conf_t_label > 0
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        batch_conf = conf_data.view(-1, num_classes)

        loss_c = F.cross_entropy(
            batch_conf, torch.clamp(conf_t_label.view(-1), 0, num_classes - 1), reduction='none')

        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_c = loss_c.view(num_batch, -1)
        loss_c[pos_mask] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox)

        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)
        ].view(-1, num_classes)

        conf_t_label_hnm = conf_t_label[(pos_mask + neg_mask).gt(0)]
        loss_c = F.cross_entropy(conf_hnm, torch.clamp(conf_t_label_hnm, 0, num_classes - 1), reduction='sum')

        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c


class Detect:
    def __init__(self, conf_thresh: float = 0.01, top_k: int = 200, nms_thresh: float = 0.45) -> None:
        """Initialize Detect.

        Args:
            conf_thresh (float, optional): Confidence threshold. Defaults to 0.01.
            top_k (int, optional): Top K results to keep. Defaults to 200.
            nms_thresh (float, optional): Non-maximum suppression threshold. Defaults to 0.45.
        """
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def decode(loc: torch.Tensor, dbox_list: torch.Tensor) -> torch.Tensor:
        """Decode bounding boxes from predictions using default boxes.

        Args:
            loc (torch.Tensor): Location predictions.
            dbox_list (torch.Tensor): Default box list.

        Returns:
            torch.Tensor: Decoded bounding boxes.
        """
        boxes = torch.cat((
            dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, :2],
            dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)

        # convert boxes to (xmin, ymin, xmax, ymax)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes

    def forward(self, args: tuple) -> torch.Tensor:
        """Forward pass for Detect.

        Args:
            args (tuple): Tuple containing location data, confidence data, and default box list.

        Returns:
            torch.Tensor: Detection results.
        """
        (loc_data, conf_data, dbox_list) = args
        num_batch = loc_data.shape[0]
        num_classes = conf_data.shape[2]
        conf_data = self.softmax(conf_data)

        # [batch, topk, 6]: 6 for [confidence, x_min, y_min, x_max, y_max, class]
        output = torch.zeros(num_batch, self.top_k, 6)

        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            decoded_boxes = self.decode(loc_data[i], dbox_list.to(device))

            conf_scores = conf_preds[i].clone()
            total_dets = 0
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)

                scores = conf_scores[cl][c_mask]

                if scores.nelement() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                ids = torchvision.ops.nms(boxes, scores, self.nms_thresh)
                if total_dets + len(ids) < self.top_k:
                    output[i, total_dets:total_dets+len(ids)] = torch.cat((scores[ids].unsqueeze(1), boxes[ids], torch.tensor((cl-1)*np.ones((len(ids), 1))).to(device)),1)
                    total_dets += len(ids)

        return output
