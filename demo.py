# Copyright (c) Facebook, Inc. and its affiliates.
"""
Demo script for running PHOSA: Perceiving Human-Object Spatial Arrangements.

Usage:
    # Run on image for bicycle class.
    python demo.py --filename my_image.jpg --class_name bicycle

    # Run on image without coarse interaction loss.
    python demo.py --filename my_image.jpg --class_name bicycle --lw_inter 0

    # Run on image with higher weight on depth ordering loss.
    python demo.py --filename my_image.jpg --class_name bicycle --lw_depth 1.
"""
import argparse
import json
import logging
import os

import numpy as np
from PIL import Image

from phosa.bodymocap import get_bodymocap_predictor, process_mocap_predictions
from phosa.constants import DEFAULT_LOSS_WEIGHTS, IMAGE_SIZE
from phosa.global_opt import optimize_human_object, visualize_human_object
from phosa.pointrend import get_pointrend_predictor
from phosa.pose_optimization import find_optimal_poses
from phosa.utils import bbox_xy_to_wh

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)


def get_args():
    parser = argparse.ArgumentParser(description="Optimize object meshes w.r.t. human.")
    parser.add_argument(
        "--filename", default="input/000000038829.jpg", help="Path to image."
    )
    parser.add_argument("--output_dir", default="output", help="Output directory.")
    parser.add_argument("--class_name", default="bicycle", help="Name of class.")
    parser.add_argument("--mesh_index", type=int, default=0, help="Index of mesh ")
    parser.add_argument(
        "--lw_inter",
        type=float,
        default=None,
        help="Loss weight for coarse interaction loss. (None: default weight)",
    )
    parser.add_argument(
        "--lw_depth",
        type=float,
        default=None,
        help="Loss weight for ordinal depth loss. (None: default weight)",
    )
    parser.add_argument(
        "--lw_inter_part",
        type=float,
        default=None,
        help="Loss weight for fine interaction loss. (None: default weight)",
    )
    parser.add_argument(
        "--lw_sil",
        type=float,
        default=None,
        help="Loss weight for mask loss. (None: default weight)",
    )
    parser.add_argument(
        "--lw_collision",
        type=float,
        default=None,
        help="Loss weight for collision loss. (None: default weight)",
    )
    parser.add_argument(
        "--lw_scale",
        type=float,
        default=None,
        help="Loss weight for object scale loss. (None: default weight)",
    )
    parser.add_argument(
        "--lw_scale_person",
        type=float,
        default=None,
        help="Loss weight for person scale loss. (None: default weight)",
    )
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="If added, saves computed metadata as filename.json.",
    )
    args = parser.parse_args()
    logger.info(f"Calling with args: {str(args)}")
    return args


def main(args):
    loss_weights = DEFAULT_LOSS_WEIGHTS[args.class_name]
    # Update defaults based on commandline args.
    for loss_name in loss_weights.keys():
        loss_weight = getattr(args, loss_name)
        if hasattr(args, loss_name) and getattr(args, loss_name) is not None:
            loss_weights[loss_name] = loss_weight
            logger.info(f"Updated {loss_name} with {loss_weight}")

    image = Image.open(args.filename).convert("RGB")
    w, h = image.size
    r = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
    w = int(r * w)
    h = int(r * h)
    image = np.array(image.resize((w, h)))
    segmenter = get_pointrend_predictor()
    instances = segmenter(image)["instances"]

    # Process Human Estimations.
    is_person = instances.pred_classes == 0
    bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()
    masks_person = instances[is_person].pred_masks
    human_predictor = get_bodymocap_predictor()
    mocap_predictions = human_predictor.regress(
        image[..., ::-1], bbox_xy_to_wh(bboxes_person)
    )
    person_parameters = process_mocap_predictions(
        mocap_predictions=mocap_predictions, bboxes=bboxes_person, masks=masks_person
    )

    object_parameters = find_optimal_poses(
        instances=instances, class_name=args.class_name, mesh_index=args.mesh_index
    )

    model = optimize_human_object(
        person_parameters=person_parameters,
        object_parameters=object_parameters,
        class_name=args.class_name,
        mesh_index=args.mesh_index,
        loss_weights=loss_weights,
    )
    frontal, top_down = visualize_human_object(model, image)

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.filename)
    ext = file_name[file_name.rfind(".") :]
    frontal_path = os.path.join(args.output_dir, file_name)
    Image.fromarray(frontal).save(frontal_path)
    logger.info(f"Saved rendered image to {frontal_path}.")
    top_down_path = frontal_path.replace(ext, "_top" + ext)
    Image.fromarray(top_down).save(top_down_path)
    logger.info(f"Saved top-down image to {top_down_path}.")
    if args.save_metadata:
        json_path = frontal_path.replace(ext, ".json")
        metadata = model.get_parameters()
        with open(json_path, "w") as f:
            json.dump(metadata, json_path)
        logger.info(f"Saved metadata to {json_path}.")


if __name__ == "__main__":
    main(get_args())
