# Copyright (c) Facebook, Inc. and its affiliates.
"""
Wrapper for Human Pose Estimator using BodyMocap.
See: https://github.com/facebookresearch/frankmocap
"""
import numpy as np
import torch

from bodymocap.body_mocap_api import BodyMocap
from detectron2.structures.masks import BitMasks

from phosa.constants import BODY_MOCAP_REGRESSOR_CKPT, BODY_MOCAP_SMPL_PATH, IMAGE_SIZE
from phosa.utils import OrthographicRenderer, bbox_xy_to_wh, local_to_global_cam


def get_bodymocap_predictor():
    human_predictor = BodyMocap(
        regressor_checkpoint=BODY_MOCAP_REGRESSOR_CKPT, smpl_dir=BODY_MOCAP_SMPL_PATH
    )
    return human_predictor


def process_mocap_predictions(
    mocap_predictions, bboxes, image_size=IMAGE_SIZE, masks=None
):
    """
    Rescales camera to follow HMR convention, and then computes the camera w.r.t. to
    image rather than local bounding box.

    Args:
        mocap_predictions (list).
        bboxes (N x 4): Bounding boxes in xyxy format.
        image_size (int): Max dimension of image.
        masks (N x H x W): Bit mask of people.

    Returns:
        dict {str: torch.cuda.FloatTensor}
            bbox: Bounding boxes in xyxy format (N x 3).
            cams: Weak perspective camera (N x 3).
            masks: Bitmasks used for computing ordinal depth loss, cropped to image
                space (N x L x L).
            local_cams: Weak perspective camera relative to the bounding boxes (N x 3).
    """
    verts = np.stack([p["pred_vertices_smpl"] for p in mocap_predictions])
    # All faces are the same, so just need one copy.
    faces = np.expand_dims(mocap_predictions[0]["faces"].astype(np.int32), 0)
    max_dim = np.max(bbox_xy_to_wh(bboxes)[:, 2:], axis=1)
    local_cams = []
    for b, pred in zip(max_dim, mocap_predictions):
        local_cam = pred["pred_camera"].copy()
        scale_o2n = pred["bbox_scale_ratio"] * b / 224
        local_cam[0] /= scale_o2n
        local_cam[1:] /= local_cam[:1]
        local_cams.append(local_cam)
    local_cams = np.stack(local_cams)
    global_cams = local_to_global_cam(bboxes, local_cams, image_size)
    inds = np.argsort(bboxes[:, 0])  # Sort from left to right to make debugging easy.
    person_parameters = {
        "bboxes": bboxes[inds].astype(np.float32),
        "cams": global_cams[inds].astype(np.float32),
        "faces": faces,
        "local_cams": local_cams[inds].astype(np.float32),
        "verts": verts[inds].astype(np.float32),
    }
    for k, v in person_parameters.items():
        person_parameters[k] = torch.from_numpy(v).cuda()
    if masks is not None:
        full_boxes = torch.tensor([[0, 0, image_size, image_size]] * len(bboxes))
        full_boxes = full_boxes.float().cuda()
        masks = BitMasks(masks).crop_and_resize(boxes=full_boxes, mask_size=image_size)
        person_parameters["masks"] = masks[inds].cuda()
    return person_parameters


def visualize_orthographic(image, human_predictions):
    ortho_renderer = OrthographicRenderer(image_size=max(image.shape))
    new_image = image.copy()
    verts = human_predictions["verts"]
    faces = human_predictions["faces"]
    cams = human_predictions["cams"]
    for i in range(len(verts)):
        v = verts[i : i + 1]
        cam = cams[i : i + 1]
        new_image = ortho_renderer(
            vertices=v, faces=faces, cam=cam, color_name="blue", image=new_image
        )
    return (new_image * 255).astype(np.uint8)
