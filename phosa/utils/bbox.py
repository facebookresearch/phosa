# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from PIL import Image

from detectron2.structures.boxes import BoxMode


def crop_image_with_bbox(image, bbox):
    """
    Crops an image to a bounding box.

    Args:
        image (H x W x C).
        bbox (4): Bounding box in xywh format.

    Returns:
        np.ndarray
    """
    bbox = bbox_wh_to_xy(bbox)
    return np.array(Image.fromarray(image).crop(tuple(bbox)))


def make_bbox_square(bbox, bbox_expansion=0.0):
    """

    Args:
        bbox (4 or B x 4): Bounding box in xywh format.
        bbox_expansion (float): Expansion factor to expand the bounding box extents from
            center.

    Returns:
        Squared bbox (same shape as bbox).
    """
    bbox = np.array(bbox)
    original_shape = bbox.shape
    bbox = bbox.reshape(-1, 4)
    center = np.stack(
        (bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2), axis=1
    )
    b = np.expand_dims(np.maximum(bbox[:, 2], bbox[:, 3]), 1)
    b *= 1 + bbox_expansion
    square_bboxes = np.hstack((center - b / 2, b, b))
    return square_bboxes.reshape(original_shape)


def bbox_xy_to_wh(bbox):
    if not isinstance(bbox, (tuple, list)):
        original_shape = bbox.shape
        bbox = bbox.reshape((-1, 4))
    else:
        original_shape = None
    bbox = BoxMode.convert(
        box=bbox, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYWH_ABS
    )
    if original_shape is not None:
        return bbox.reshape(original_shape)
    return bbox


def bbox_wh_to_xy(bbox):
    if not isinstance(bbox, (tuple, list)):
        original_shape = bbox.shape
        bbox = bbox.reshape((-1, 4))
    else:
        original_shape = None
    bbox = BoxMode.convert(
        box=bbox, from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS
    )
    if original_shape is not None:
        return bbox.reshape(original_shape)
    return bbox


def check_overlap(bbox1, bbox2):
    """
    Checks if 2 boxes are overlapping. Also works for 2D tuples.

    Args:
        bbox1: [x1, y1, x2, y2] or [z1, z2]
        bbox2: [x1, y1, x2, y2] or [z1, z2]

    Returns:
        bool
    """
    if bbox1[0] > bbox2[2] or bbox2[0] > bbox1[2]:
        return False
    if len(bbox1) > 2:
        if bbox1[1] > bbox2[3] or bbox2[1] > bbox1[3]:
            return False
    return True


def compute_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def compute_iou(bbox1, bbox2):
    """
    Computes Intersection Over Union for two boxes.

    Args:
        bbox1 (np.ndarray or torch.Tensor): (x1, y1, x2, y2).
        bbox2 (np.ndarray or torch.Tensor): (x1, y1, x2, y2).
    """
    a1 = compute_area(bbox1)
    a2 = compute_area(bbox2)
    if isinstance(bbox1, np.ndarray):
        lt = np.maximum(bbox1[:2], bbox2[:2])
        rb = np.minimum(bbox1[2:], bbox2[2:])
        wh = np.clip(rb - lt, a_min=0, a_max=None)
    else:
        stack = torch.stack((bbox1, bbox2))
        lt = torch.max(stack[:, :2], 0).values
        rb = torch.min(stack[:, 2:], 0).values
        wh = torch.clamp_min(rb - lt, 0)
    inter = wh[0] * wh[1]
    return inter / (a1 + a2 - inter)
