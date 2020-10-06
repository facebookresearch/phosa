# Copyright (c) Facebook, Inc. and its affiliates.
"""
Utilities for computing initial object pose fits from instance masks.
"""
import matplotlib.pyplot as plt
import neural_renderer as nr
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.morphology import distance_transform_edt
from tqdm.auto import tqdm

from phosa.constants import (
    BBOX_EXPANSION_FACTOR,
    CLASS_ID_MAP,
    FOCAL_LENGTH,
    MESH_MAP,
    REND_SIZE,
)
from phosa.pointrend import get_class_masks_from_instances
from phosa.utils import (
    PerspectiveRenderer,
    center_vertices,
    compute_K_roi,
    compute_random_rotations,
    crop_image_with_bbox,
    matrix_to_rot6d,
    rot6d_to_matrix,
)


def compute_bbox_proj(verts, f, img_size=256):
    """
    Computes the 2D bounding box of vertices projected to the image plane.

    Args:
        verts (B x N x 3): Vertices.
        f (float): Focal length.
        img_size (int): Size of image in pixels.

    Returns:
        Bounding box in xywh format (Bx4).
    """
    xy = verts[:, :, :2]
    z = verts[:, :, 2:]
    proj = f * xy / z + 0.5  # [0, 1]
    proj = proj * img_size  # [0, img_size]
    u, v = proj[:, :, 0], proj[:, :, 1]
    x1, x2 = u.min(1).values, u.max(1).values
    y1, y2 = v.min(1).values, v.max(1).values
    return torch.stack((x1, y1, x2 - x1, y2 - y1), 1)


def compute_optimal_translation(bbox_target, vertices, f=1, img_size=256):
    """
    Computes the optimal translation to align the mesh to a bounding box using
    least squares.

    Args:
        bbox_target (list): bounding box in xywh.
        vertices (B x V x 3): Batched vertices.
        f (float): Focal length.
        img_size (int): Image size in pixels.

    Returns:
        Optimal 3D translation (B x 3).
    """
    bbox_mask = np.array(bbox_target)
    mask_center = bbox_mask[:2] + bbox_mask[2:] / 2
    diag_mask = np.sqrt(bbox_mask[2] ** 2 + bbox_mask[3] ** 2)
    B = vertices.shape[0]
    x = torch.zeros(B).cuda()
    y = torch.zeros(B).cuda()
    z = 2.5 * torch.ones(B).cuda()
    for _ in range(50):
        translation = torch.stack((x, y, z), -1).unsqueeze(1)
        v = vertices + translation
        bbox_proj = compute_bbox_proj(v, f=1, img_size=img_size)
        diag_proj = torch.sqrt(torch.sum(bbox_proj[:, 2:] ** 2, 1))
        delta_z = z * (diag_proj / diag_mask - 1)
        z = z + delta_z
        proj_center = bbox_proj[:, :2] + bbox_proj[:, 2:] / 2
        x += (mask_center[0] - proj_center[:, 0]) * z / f / img_size
        y += (mask_center[1] - proj_center[:, 1]) * z / f / img_size
    return torch.stack((x, y, z), -1).unsqueeze(1)


class PoseOptimizer(nn.Module):
    """
    Computes the optimal object pose from an instance mask and an exemplar mesh. We
    optimize an occlusion-aware silhouette loss that consists of a one-way chamfer loss
    and a silhouette matching loss.
    """

    def __init__(
        self,
        ref_image,
        vertices,
        faces,
        textures,
        rotation_init,
        translation_init,
        batch_size=1,
        kernel_size=7,
        K=None,
        power=0.25,
        lw_chamfer=0.5,
    ):
        assert ref_image.shape[0] == ref_image.shape[1], "Must be square."
        super(PoseOptimizer, self).__init__()

        self.register_buffer("vertices", vertices.repeat(batch_size, 1, 1))
        self.register_buffer("faces", faces.repeat(batch_size, 1, 1))
        self.register_buffer("textures", textures.repeat(batch_size, 1, 1, 1, 1, 1))

        # Load reference mask.
        # Convention for silhouette-aware loss: -1=occlusion, 0=bg, 1=fg.
        image_ref = torch.from_numpy((ref_image > 0).astype(np.float32))
        keep_mask = torch.from_numpy((ref_image >= 0).astype(np.float32))
        self.register_buffer("image_ref", image_ref.repeat(batch_size, 1, 1))
        self.register_buffer("keep_mask", keep_mask.repeat(batch_size, 1, 1))
        self.pool = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2)
        )
        self.rotations = nn.Parameter(rotation_init.clone().float(), requires_grad=True)
        if rotation_init.shape[0] != translation_init.shape[0]:
            translation_init = translation_init.repeat(batch_size, 1, 1)
        self.translations = nn.Parameter(
            translation_init.clone().float(), requires_grad=True
        )
        mask_edge = self.compute_edges(image_ref.unsqueeze(0)).cpu().numpy()
        edt = distance_transform_edt(1 - (mask_edge > 0)) ** (power * 2)
        self.register_buffer(
            "edt_ref_edge", torch.from_numpy(edt).repeat(batch_size, 1, 1).float()
        )
        # Setup renderer.
        if K is None:
            K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
        R = torch.eye(3).unsqueeze(0).cuda()
        t = torch.zeros(1, 3).cuda()
        self.renderer = nr.renderer.Renderer(
            image_size=ref_image.shape[0],
            K=K,
            R=R,
            t=t,
            orig_size=1,
            anti_aliasing=False,
        )
        self.lw_chamfer = lw_chamfer
        self.K = K

    def apply_transformation(self):
        """
        Applies current rotation and translation to vertices.
        """
        rots = rot6d_to_matrix(self.rotations)
        return torch.matmul(self.vertices, rots) + self.translations

    def compute_offscreen_loss(self, verts):
        """
        Computes loss for offscreen penalty. This is used to prevent the degenerate
        solution of moving the object offscreen to minimize the chamfer loss.
        """
        # On-screen means xy between [-1, 1] and far > depth > 0
        proj = nr.projection(
            verts,
            self.renderer.K,
            self.renderer.R,
            self.renderer.t,
            self.renderer.dist_coeffs,
            orig_size=1,
        )
        xy, z = proj[:, :, :2], proj[:, :, 2:]
        zeros = torch.zeros_like(z)
        lower_right = torch.max(xy - 1, zeros).sum(dim=(1, 2))  # Amount greater than 1
        upper_left = torch.max(-1 - xy, zeros).sum(dim=(1, 2))  # Amount less than -1
        behind = torch.max(-z, zeros).sum(dim=(1, 2))
        too_far = torch.max(z - self.renderer.far, zeros).sum(dim=(1, 2))
        return lower_right + upper_left + behind + too_far

    def compute_edges(self, silhouette):
        return self.pool(silhouette) - silhouette

    def forward(self):
        verts = self.apply_transformation()
        image = self.keep_mask * self.renderer(verts, self.faces, mode="silhouettes")
        loss_dict = {}
        loss_dict["mask"] = torch.sum((image - self.image_ref) ** 2, dim=(1, 2))
        loss_dict["chamfer"] = self.lw_chamfer * torch.sum(
            self.compute_edges(image) * self.edt_ref_edge, dim=(1, 2)
        )
        loss_dict["offscreen"] = 1000 * self.compute_offscreen_loss(verts)
        return loss_dict, image

    def render(self):
        """
        Renders objects according to current rotation and translation.
        """
        verts = self.apply_transformation()
        images = self.renderer(verts, self.faces, torch.tanh(self.textures))[0]
        images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        return images


def visualize_optimal_poses(model, image_crop, mask, score=0):
    """
    Visualizes the 8 best-scoring object poses.

    Args:
        model (PoseOptimizer).
        image_crop (H x H x 3).
        mask (M x M x 3).
        score (float): Mask confidence score (optional).
    """
    num_vis = 8
    rotations = model.rotations
    translations = model.translations
    verts = model.vertices[0]
    faces = model.faces[0]
    loss_dict, sil = model()
    losses = sum(loss_dict.values())
    K_roi = model.renderer.K
    inds = torch.argsort(losses)[:num_vis]
    obj_renderer = PerspectiveRenderer()

    fig = plt.figure(figsize=((10, 4)))
    ax1 = fig.add_subplot(2, 5, 1)
    ax1.imshow(image_crop)
    ax1.axis("off")
    ax1.set_title("Cropped Image")

    ax2 = fig.add_subplot(2, 5, 2)
    ax2.imshow(mask)
    ax2.axis("off")
    if score > 0:
        ax2.set_title(f"Mask Conf: {score:.2f}")
    else:
        ax2.set_title("Mask")

    for i, ind in enumerate(inds.cpu().numpy()):
        ax = fig.add_subplot(2, 5, i + 3)
        ax.imshow(
            obj_renderer(
                vertices=verts,
                faces=faces,
                image=image_crop,
                translation=translations[ind],
                rotation=rot6d_to_matrix(rotations)[ind],
                color_name="red",
                K=K_roi,
            )
        )
        ax.set_title(f"Rank {i}: {losses[ind]:.1f}")
        ax.axis("off")
    plt.show()


def find_optimal_pose(
    vertices,
    faces,
    mask,
    bbox,
    square_bbox,
    image_size,
    batch_size=500,
    num_iterations=50,
    num_initializations=2000,
    lr=1e-3,
):
    ts = 1
    textures = torch.ones(faces.shape[0], ts, ts, ts, 3, dtype=torch.float32).cuda()
    x, y, b, _ = square_bbox
    L = max(image_size)
    K_roi = compute_K_roi((x, y), b, L, focal_length=FOCAL_LENGTH)
    # Stuff to keep around
    best_losses = np.inf
    best_rots = None
    best_trans = None
    best_loss_single = np.inf
    best_rots_single = None
    best_trans_single = None
    loop = tqdm(total=np.ceil(num_initializations / batch_size) * num_iterations)
    for _ in range(0, num_initializations, batch_size):
        rotations_init = compute_random_rotations(batch_size, upright=False)
        translations_init = compute_optimal_translation(
            bbox_target=np.array(bbox) * REND_SIZE / L,
            vertices=torch.matmul(vertices.unsqueeze(0), rotations_init),
        )
        model = PoseOptimizer(
            ref_image=mask,
            vertices=vertices,
            faces=faces,
            textures=textures,
            rotation_init=matrix_to_rot6d(rotations_init),
            translation_init=translations_init,
            batch_size=batch_size,
            K=K_roi,
        )
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(num_iterations):
            optimizer.zero_grad()
            loss_dict, sil = model()
            losses = sum(loss_dict.values())
            loss = losses.sum()
            loss.backward()
            optimizer.step()
            if losses.min() < best_loss_single:
                ind = torch.argmin(losses)
                best_loss_single = losses[ind]
                best_rots_single = model.rotations[ind].detach().clone()
                best_trans_single = model.translations[ind].detach().clone()
            loop.set_description(f"loss: {best_loss_single.item():.3g}")
            loop.update()
        if best_rots is None:
            best_rots = model.rotations
            best_trans = model.translations
            best_losses = losses
        else:
            best_rots = torch.cat((best_rots, model.rotations), 0)
            best_trans = torch.cat((best_trans, model.translations), 0)
            best_losses = torch.cat((best_losses, losses))
        inds = torch.argsort(best_losses)
        best_losses = best_losses[inds][:batch_size].detach().clone()
        best_trans = best_trans[inds][:batch_size].detach().clone()
        best_rots = best_rots[inds][:batch_size].detach().clone()
    loop.close()
    # Add best ever:
    best_rotations = torch.cat((best_rots_single.unsqueeze(0), best_rots[:-1]), 0)
    best_translations = torch.cat((best_trans_single.unsqueeze(0), best_trans[:-1]), 0)
    model.rotations = nn.Parameter(best_rotations)
    model.translations = nn.Parameter(best_translations)
    return model


def find_optimal_poses(
    instances,
    vertices=None,
    faces=None,
    class_id=None,
    class_name=None,
    mesh_index=0,
    visualize=False,
    image=None,
    batch_size=500,
    num_iterations=50,
    num_initializations=2000,
):
    """
    Optimizes for pose with respect to a target mask using an occlusion-aware silhouette
    loss.

    Args:
        instances: PointRend or Mask R-CNN instances.
        vertices (N x 3): Mesh vertices (If not set, loads vertices using class name
            and mesh index).
        faces (F x 3): Mesh faces (If not set, loads faces using class name and mesh
            index).
        class_id (int): Class index if no vertices/faces nor class name is given.
        class_name (str): Name of class.
        mesh_index (int): Mesh index for classes with multiple mesh models.
        visualize (bool): If True, visualizes the top poses found.
        image (H x W x 3): Image used for visualization.

    Returns:
        dict {str: torch.cuda.FloatTensor}
            rotations (N x 3 x 3): Top rotation matrices.
            translations (N x 1 x 3): Top translations.
            target_masks (N x 256 x 256): Cropped occlusion-aware masks (for silhouette
                loss).
            masks (N x 640 x 640): Object masks (for depth ordering loss).
            K_roi (N x 3 x 3): Camera intrinsics corresponding to each object ROI crop.
    """
    if class_id is None:
        class_id = CLASS_ID_MAP[class_name]
    if vertices is None:
        vertices, faces = nr.load_obj(MESH_MAP[class_name][mesh_index])
        vertices, faces = center_vertices(vertices, faces)

    class_masks, annotations = get_class_masks_from_instances(
        instances=instances,
        class_id=class_id,
        add_ignore=True,
        rend_size=REND_SIZE,
        bbox_expansion=BBOX_EXPANSION_FACTOR,
        min_confidence=0.95,
    )
    object_parameters = {
        "rotations": [],
        "translations": [],
        "target_masks": [],
        "K_roi": [],
        "masks": [],
    }
    for _, (mask, annotation) in enumerate(zip(class_masks, annotations)):
        model = find_optimal_pose(
            vertices=vertices,
            faces=faces,
            mask=mask,
            bbox=annotation["bbox"],
            square_bbox=annotation["square_bbox"],
            image_size=instances.image_size,
            batch_size=batch_size,
            num_iterations=num_iterations,
            num_initializations=num_initializations,
        )
        if visualize:
            if image is None:
                image = np.zeros(instances.image_size, dtype=np.uint8)
            visualize_optimal_poses(
                model=model,
                image_crop=crop_image_with_bbox(image, annotation["square_bbox"]),
                mask=mask,
                score=annotation["score"],
            )
        object_parameters["rotations"].append(
            rot6d_to_matrix(model.rotations)[0].detach()
        )
        object_parameters["translations"].append(model.translations[0].detach())
        object_parameters["target_masks"].append(torch.from_numpy(mask).cuda())
        object_parameters["K_roi"].append(model.K.detach())
        object_parameters["masks"].append(annotation["mask"].cuda())
    for k, v in object_parameters.items():
        object_parameters[k] = torch.stack(v)
    return object_parameters
