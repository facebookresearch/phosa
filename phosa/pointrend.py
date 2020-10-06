# Copyright (c) Facebook, Inc. and its affiliates.
"""
Wrapper for PointRend Segmentation algorithm.
Reference: Kirillov et al. "PointRend: Image Segmentation as Rendering." (CVPR 2020).
"""
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import BitMasks

import point_rend
from phosa.constants import (
    BBOX_EXPANSION_FACTOR,
    IMAGE_SIZE,
    POINTREND_CONFIG,
    POINTREND_MODEL_WEIGHTS,
    REND_SIZE,
)
from phosa.utils.bbox import bbox_wh_to_xy, bbox_xy_to_wh, make_bbox_square


def get_pointrend_predictor(min_confidence=0.9, image_format="RGB"):

    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(POINTREND_CONFIG)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_confidence
    cfg.MODEL.WEIGHTS = POINTREND_MODEL_WEIGHTS
    cfg.INPUT.FORMAT = image_format
    return DefaultPredictor(cfg)


def get_class_masks_from_instances(
    instances,
    class_id=1,
    add_ignore=True,
    rend_size=REND_SIZE,
    bbox_expansion=BBOX_EXPANSION_FACTOR,
    min_confidence=0.0,
    image_size=IMAGE_SIZE,
):
    """
    Gets occlusion-aware masks for a specific class index and additional metadata from
    PointRend instances.

    Args:
        instances: Detectron2 Instances with segmentation predictions.
        class_id (int): Object class id (using COCO dense ordering).
        add_ignore (bool): If True, adds occlusion-aware masking.
        rend_size (int): Mask size.
        bbox_expansion (float): Amount to pad the masks. This is important to prevent
            ignoring background pixels right outside the bounding box.
        min_confidence (float): Minimum confidence threshold for masks.

    Returns:
        keep_masks (N x rend_size x rend_size).
        keep_annotations (dict):
            "bbox":
            "class_id":
            "segmentation":
            "square_bbox":
    """
    if len(instances) == 0:
        return [], []
    instances = instances.to(torch.device("cpu:0"))
    boxes = instances.pred_boxes.tensor.numpy()
    class_ids = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    keep_ids = np.logical_and(class_ids == class_id, scores > min_confidence)
    bit_masks = BitMasks(instances.pred_masks)

    keep_annotations = []
    keep_masks = []
    full_boxes = torch.tensor([[0, 0, image_size, image_size]] * len(boxes)).float()
    full_sized_masks = bit_masks.crop_and_resize(full_boxes, image_size)
    for k in np.where(keep_ids)[0]:
        bbox = bbox_xy_to_wh(boxes[k])
        square_bbox = make_bbox_square(bbox, bbox_expansion)
        square_boxes = torch.FloatTensor(
            np.tile(bbox_wh_to_xy(square_bbox), (len(instances), 1))
        )
        masks = bit_masks.crop_and_resize(square_boxes, rend_size).clone().detach()
        if add_ignore:
            ignore_mask = masks[0]
            for i in range(1, len(masks)):
                ignore_mask = ignore_mask | masks[i]
            ignore_mask = -ignore_mask.float().numpy()
        else:
            ignore_mask = np.zeros(rend_size, rend_size)
        m = ignore_mask.copy()
        mask = masks[k]
        m[mask] = mask[mask]
        keep_masks.append(m)
        keep_annotations.append(
            {
                "bbox": bbox,
                "class_id": class_ids[k],
                "mask": full_sized_masks[k],
                "score": scores[k],
                "square_bbox": square_bbox,
            }
        )
    return keep_masks, keep_annotations
