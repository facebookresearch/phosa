# Copyright (c) Facebook, Inc. and its affiliates.
import os.path as osp

EXTERNAL_DIRECTORY = "./external"

# Configurations for PointRend.
POINTREND_PATH = osp.join(EXTERNAL_DIRECTORY, "detectron2/projects/PointRend")
POINTREND_CONFIG = osp.join(
    POINTREND_PATH, "configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
)
POINTREND_MODEL_WEIGHTS = (
    "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/"
    "164955410/model_final_3c3198.pkl"
)

# Configurations for BodyMocap.
BODY_MOCAP_PATH = osp.join(EXTERNAL_DIRECTORY, "frankmocap")
BODY_MOCAP_REGRESSOR_CKPT = osp.join(
    # BODY_MOCAP_PATH,
    "extra_data/body_module/pretrained_weights",
    "2020_05_31-00_50_43-best-51.749683916568756.pt",
)
BODY_MOCAP_SMPL_PATH = osp.join(
    # BODY_MOCAP_PATH,
    "extra_data/smpl"
)

# Configurations for PHOSA
FOCAL_LENGTH = 1.0
IMAGE_SIZE = 640
REND_SIZE = 256  # Size of target masks for silhouette loss.
BBOX_EXPANSION_FACTOR = 0.3  # Amount to pad the target masks for silhouette loss.
SMPL_FACES_PATH = "models/smpl_faces.npy"

# Mapping from class name to COCO contiguous id. You can double-check these using:
# >>> coco_metadata = MetadataCatalog.get("coco_2017_val")
# >>> coco_metadata.thing_classes
CLASS_ID_MAP = {
    "bat": 34,
    "bench": 13,
    "bicycle": 1,
    "car": 2,
    "laptop": 63,
    "motorcycle": 3,
    "skateboard": 36,
    "surfboard": 37,
    "tennis": 38,
}
MEAN_INTRINSIC_SCALE = {  # Empirical intrinsic scales learned by our method.
    "bat": 0.40,
    "bench": 0.95,
    "bicycle": 0.91,
    "laptop": 0.24,
    "motorcycle": 1.02,
    "skateboard": 0.35,
    "surfboard": 1.0,
    "tennis": 0.33,
}
MESH_MAP = {  # Class name -> list of paths to objs.
    "bicycle": ["models/meshes/bicycle_01.obj"]
}

# Dict[class_name: List[Tuple(path_to_parts_json, interaction_pairs_dict)]].
PART_LABELS = {
    "person": [("models/meshes/person_labels.json", {})],
    "bicycle": [
        (
            "models/meshes/bicycle_01_labels.json",
            {"seat": ["butt"], "handle": ["lhand", "rhand"]},
        )
    ],
}
INTERACTION_MAPPING = {
    "bat": ["lpalm", "rpalm"],
    "bench": ["back", "butt"],
    "bicycle": ["lhand", "rhand", "butt"],
    "laptop": ["lhand", "rhand"],
    "motorcycle": ["lhand", "rhand", "butt"],
    "skateboard": ["lfoot", "rfoot", "lhand", "rhand"],
    "surfboard": ["lfoot", "rfoot", "lhand", "rhand"],
    "tennis": ["lpalm", "rpalm"],
}
BBOX_EXPANSION = {
    "bat": 0.5,
    "bench": 0.3,
    "bicycle": 0.0,
    "bottle": 0.3,
    "chair": 0.3,
    "couch": 0.3,
    "cup": 0.3,
    "horse": 0.0,
    "laptop": 0.2,
    "motorcycle": 0.0,
    "skateboard": 0.8,
    "surfboard": 0,
    "tennis": 0.4,
    "wineglass": 0.3,
}
BBOX_EXPANSION_PARTS = {
    "bat": 2.5,
    "bench": 0.5,
    "bicycle": 0.7,
    "bottle": 0.3,
    "chair": 0.3,
    "couch": 0.3,
    "cup": 0.3,
    "horse": 0.3,
    "laptop": 0.0,
    "motorcycle": 0.7,
    "skateboard": 0.5,
    "surfboard": 0.2,
    "tennis": 2,
    "wineglass": 0.3,
}
INTERACTION_THRESHOLD = {
    "bat": 5,
    "bench": 3,
    "bicycle": 2,
    "laptop": 2.5,
    "motorcycle": 5,
    "skateboard": 3,
    "surfboard": 5,
    "tennis": 5,
}
DEFAULT_LOSS_WEIGHTS = {  # Loss weights.
    "default": {
        "lw_inter": 20,
        "lw_depth": 1,
        "lw_inter_part": 50,
        "lw_sil": 10.0,
        "lw_collision": 2.0,
        "lw_scale": 100,
        "lw_scale_person": 100,
    },
    "bat": {
        "lw_inter": 30,
        "lw_depth": 0.01,
        "lw_inter_part": 100,
        "lw_sil": 20.0,
        "lw_collision": 1.0,
        "lw_scale": 10000,
        "lw_scale_person": 1000,
    },
    "bench": {
        "lw_inter": 30,
        "lw_depth": 0.1,
        "lw_inter_part": 50,
        "lw_sil": 50.0,
        "lw_collision": 10.0,
        "lw_scale": 1000,
        "lw_scale_person": 100,
    },
    "bicycle": {
        "lw_inter": 20,
        "lw_depth": 1,
        "lw_inter_part": 50,
        "lw_sil": 10.0,
        "lw_collision": 2.0,
        "lw_scale": 100,
        "lw_scale_person": 100,
    },
    "laptop": {
        "lw_inter": 20,
        "lw_depth": 0.01,
        "lw_inter_part": 20,
        "lw_sil": 10.0,
        "lw_collision": 10,
        "lw_scale": 1e3,
        "lw_scale_person": 1e3,
    },
    "motorcycle": {
        "lw_inter": 0,
        "lw_depth": 1.0,
        "lw_inter_part": 100,
        "lw_sil": 20.0,
        "lw_collision": 2.0,
        "lw_scale": 100,
        "lw_scale_person": 100,
    },
    "surfboard": {
        "lw_inter": 50,
        "lw_depth": 10,
        "lw_inter_part": 100,
        "lw_sil": 10.0,
        "lw_collision": 20,
        "lw_scale": 1e3,
        "lw_scale_person": 1e3,
    },
    "skateboard": {
        "lw_inter": 10,
        "lw_depth": 0.01,
        "lw_inter_part": 50,
        "lw_sil": 10.0,
        "lw_collision": 100,
        "lw_scale": 1e4,
        "lw_scale_person": 1e3,
    },
    "tennis": {
        "lw_inter": 30,
        "lw_depth": 0.01,
        "lw_inter_part": 500,
        "lw_sil": 10.0,
        "lw_collision": 10,
        "lw_scale": 1e4,
        "lw_scale_person": 100,
    },
}
