# Copyright (c) Facebook, Inc. and its affiliates.
"""
Main functions for doing human-object optimization.
"""
import itertools
import json

import neural_renderer as nr
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from phosa.constants import (
    BBOX_EXPANSION,
    BBOX_EXPANSION_PARTS,
    DEFAULT_LOSS_WEIGHTS,
    IMAGE_SIZE,
    INTERACTION_MAPPING,
    INTERACTION_THRESHOLD,
    MEAN_INTRINSIC_SCALE,
    MESH_MAP,
    PART_LABELS,
    REND_SIZE,
    SMPL_FACES_PATH,
)
from phosa.utils.bbox import check_overlap, compute_iou
from phosa.utils.camera import (
    compute_transformation_ortho,
    compute_transformation_persp,
)
from phosa.utils.geometry import (
    center_vertices,
    combine_verts,
    compute_dist_z,
    matrix_to_rot6d,
    rot6d_to_matrix,
)


def get_faces_and_textures(verts_list, faces_list):
    """

    Args:
        verts_list (List[Tensor(B x V x 3)]).
        faces_list (List[Tensor(f x 3)]).

    Returns:
        faces: (1 x F x 3)
        textures: (1 x F x 1 x 1 x 1 x 3)
    """
    colors_list = [
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # red
        [0.65098039, 0.74117647, 0.85882353],  # blue
        [0.9, 0.7, 0.7],  # pink
    ]
    all_faces_list = []
    all_textures_list = []
    o = 0
    for verts, faces, colors in zip(verts_list, faces_list, colors_list):
        B = len(verts)
        index_offset = torch.arange(B).to(verts.device) * verts.shape[1] + o
        o += verts.shape[1] * B
        faces_repeat = faces.clone().repeat(B, 1, 1)
        faces_repeat += index_offset.view(-1, 1, 1)
        faces_repeat = faces_repeat.reshape(-1, 3)
        all_faces_list.append(faces_repeat)
        textures = torch.FloatTensor(colors).to(verts.device)
        all_textures_list.append(textures.repeat(faces_repeat.shape[0], 1, 1, 1, 1))
    all_faces_list = torch.cat(all_faces_list).unsqueeze(0)
    all_textures_list = torch.cat(all_textures_list).unsqueeze(0)
    return all_faces_list, all_textures_list


def project_bbox(vertices, renderer, parts_labels=None, bbox_expansion=0.0):
    """
    Computes the 2D bounding box of the vertices after projected to the image plane.

    TODO(@jason): Batch these operations.

    Args:
        vertices (V x 3).
        renderer: Renderer used to get camera parameters.
        parts_labels (dict): Dictionary mapping a part name to the corresponding vertex
            indices.
        bbox_expansion (float): Amount to expand the bounding boxes.

    Returns:
        If a part_label dict is given, returns a dictionary mapping part name to bbox.
        Else, returns the projected 2D bounding box.
    """
    proj = nr.projection(
        (vertices * torch.tensor([[1, -1, 1.0]]).cuda()).unsqueeze(0),
        K=renderer.K,
        R=renderer.R,
        t=renderer.t,
        dist_coeffs=renderer.dist_coeffs,
        orig_size=1,
    )
    proj = proj.squeeze(0)[:, :2]
    if parts_labels is None:
        parts_labels = {"": torch.arange(len(vertices)).to(vertices.device)}
    bbox_parts = {}
    for part, inds in parts_labels.items():
        bbox = torch.cat((proj[inds].min(0).values, proj[inds].max(0).values), dim=0)
        if bbox_expansion:
            center = (bbox[:2] + bbox[2:]) / 2
            b = (bbox[2:] - bbox[:2]) / 2 * (1 + bbox_expansion)
            bbox = torch.cat((center - b, center + b))
        bbox_parts[part] = bbox
    if "" in parts_labels:
        return bbox_parts[""]
    return bbox_parts


class Losses(object):
    def __init__(
        self,
        renderer,
        ref_mask,
        keep_mask,
        K_rois,
        class_name,
        labels_person,
        labels_object,
        interaction_map_parts,
    ):
        self.renderer = nr.renderer.Renderer(
            image_size=REND_SIZE, K=renderer.K, R=renderer.R, t=renderer.t, orig_size=1
        )
        self.ref_mask = ref_mask
        self.keep_mask = keep_mask
        self.K_rois = K_rois
        self.thresh = INTERACTION_THRESHOLD[class_name]  # z thresh for interaction loss
        self.mse = torch.nn.MSELoss()
        self.class_name = class_name
        self.labels_person = labels_person
        self.labels_object = labels_object

        self.expansion = BBOX_EXPANSION[class_name]
        self.expansion_parts = BBOX_EXPANSION_PARTS[class_name]
        self.interaction_map = INTERACTION_MAPPING[class_name]
        self.interaction_map_parts = interaction_map_parts

        self.interaction_pairs = None
        self.interaction_pairs_parts = None
        self.bboxes_parts_person = None
        self.bboxes_parts_object = None

    def assign_interaction_pairs(self, verts_person, verts_object):
        """
        Assigns pairs of people and objects that are interacting. Note that multiple
        people can be assigned to the same object, but one person cannot be assigned to
        multiple objects. (This formulation makes sense for objects like motorcycles
        and bicycles. Can be changed for handheld objects like bats or rackets).

        This is computed separately from the loss function because there are potential
        speed improvements to re-using stale interaction pairs across multiple
        iterations (although not currently being done).

        A person and an object are interacting if the 3D bounding boxes overlap:
            * Check if X-Y bounding boxes overlap by projecting to image plane (with
              some expansion defined by BBOX_EXPANSION), and
            * Check if Z overlaps by thresholding distance.

        Args:
            verts_person (N_p x V_p x 3).
            verts_object (N_o x V_o x 3).

        Returns:
            interaction_pairs: List[Tuple(person_index, object_index)]
        """
        with torch.no_grad():
            bboxes_object = [
                project_bbox(v, self.renderer, bbox_expansion=self.expansion)
                for v in verts_object
            ]
            bboxes_person = [
                project_bbox(v, self.renderer, self.labels_person, self.expansion)
                for v in verts_person
            ]
            num_people = len(bboxes_person)
            num_objects = len(bboxes_object)
            ious = np.zeros((num_people, num_objects))
            for part_person in self.interaction_map:
                for i_person in range(num_people):
                    for i_object in range(num_objects):
                        iou = compute_iou(
                            bbox1=bboxes_object[i_object],
                            bbox2=bboxes_person[i_person][part_person],
                        )
                        ious[i_person, i_object] += iou

            self.interaction_pairs = []
            for i_person in range(num_people):
                i_object = np.argmax(ious[i_person])
                if ious[i_person][i_object] == 0:
                    continue
                dist = compute_dist_z(verts_person[i_person], verts_object[i_object])
                if dist < self.thresh:
                    self.interaction_pairs.append((i_person, i_object))
            return self.interaction_pairs

    def assign_interaction_pairs_parts(self, verts_person, verts_object):
        """
        Assigns pairs of person parts and objects pairs that are interacting.

        This is computed separately from the loss function because there are potential
        speed improvements to re-using stale interaction pairs across multiple
        iterations (although not currently being done).

        A part of a person and a part of an object are interacting if the 3D bounding
        boxes overlap:
            * Check if X-Y bounding boxes overlap by projecting to image plane (with
              some expansion defined by BBOX_EXPANSION_PARTS), and
            * Check if Z overlaps by thresholding distance.

        Args:
            verts_person (N_p x V_p x 3).
            verts_object (N_o x V_o x 3).

        Returns:
            interaction_pairs_parts:
                List[Tuple(person_index, person_part, object_index, object_part)]
        """
        with torch.no_grad():
            bboxes_person = [
                project_bbox(v, self.renderer, self.labels_person, self.expansion_parts)
                for v in verts_person
            ]
            bboxes_object = [
                project_bbox(v, self.renderer, self.labels_object, self.expansion_parts)
                for v in verts_object
            ]
            self.interaction_pairs_parts = []
            for i_p, i_o in itertools.product(
                range(len(verts_person)), range(len(verts_object))
            ):
                for part_object in self.interaction_map_parts.keys():
                    for part_person in self.interaction_map_parts[part_object]:
                        bbox_object = bboxes_object[i_o][part_object]
                        bbox_person = bboxes_person[i_p][part_person]
                        is_overlapping = check_overlap(bbox_object, bbox_person)
                        z_dist = compute_dist_z(
                            verts_object[i_o][self.labels_object[part_object]],
                            verts_person[i_p][self.labels_person[part_person]],
                        )
                        if is_overlapping and z_dist < self.thresh:
                            self.interaction_pairs_parts.append(
                                (i_p, part_person, i_o, part_object)
                            )
            return self.interaction_pairs_parts

    def compute_sil_loss(self, verts, faces):
        loss_sil = torch.tensor(0.0).float().cuda()
        for i in range(len(verts)):
            v = verts[i].unsqueeze(0)
            K = self.K_rois[i]
            rend = self.renderer(v, faces[i], K=K, mode="silhouettes")
            image = self.keep_mask[i] * rend
            l_m = torch.sum((image - self.ref_mask[i]) ** 2) / self.keep_mask[i].sum()
            loss_sil += l_m
        return {"loss_sil": loss_sil / len(verts)}

    def compute_interaction_loss(self, verts_person, verts_object):
        """
        Computes interaction loss.
        """
        loss_interaction = torch.tensor(0.0).float().cuda()
        interaction_pairs = self.assign_interaction_pairs(verts_person, verts_object)
        for i_person, i_object in interaction_pairs:
            v_person = verts_person[i_person]
            v_object = verts_object[i_object]
            centroid_error = self.mse(v_person.mean(0), v_object.mean(0))
            loss_interaction += centroid_error
        num_interactions = max(len(interaction_pairs), 1)
        return {"loss_inter": loss_interaction / num_interactions}

    def compute_interaction_loss_parts(self, verts_person, verts_object):
        loss_interaction_parts = torch.tensor(0.0).float().cuda()
        interaction_pairs_parts = self.assign_interaction_pairs_parts(
            verts_person=verts_person, verts_object=verts_object
        )
        for i_p, part_p, i_o, part_o in interaction_pairs_parts:
            v_person = verts_person[i_p][self.labels_person[part_p]]
            v_object = verts_object[i_o][self.labels_object[part_o]]
            dist = self.mse(v_person.mean(0), v_object.mean(0))
            loss_interaction_parts += dist
        num_interactions = max(len(self.interaction_pairs_parts), 1)
        return {"loss_inter_part": loss_interaction_parts / num_interactions}

    def compute_intrinsic_scale_prior(self, intrinsic_scales, intrinsic_mean):
        return torch.sum((intrinsic_scales - intrinsic_mean) ** 2)

    def compute_ordinal_depth_loss(self, masks, silhouettes, depths):
        loss = torch.tensor(0.0).float().cuda()
        num_pairs = 0
        for i in range(len(silhouettes)):
            for j in range(len(silhouettes)):
                has_pred = silhouettes[i] & silhouettes[j]
                if has_pred.sum() == 0:
                    continue
                else:
                    num_pairs += 1
                front_i_gt = masks[i] & (~masks[j])
                front_j_pred = depths[j] < depths[i]
                m = front_i_gt & front_j_pred & has_pred
                if m.sum() == 0:
                    continue
                dists = torch.clamp(depths[i] - depths[j], min=0.0, max=2.0)
                loss += torch.sum(torch.log(1 + torch.exp(dists))[m])
        loss /= num_pairs
        return {"loss_depth": loss}

    @staticmethod
    def _compute_iou_1d(a, b):
        """
        a: (2).
        b: (2).
        """
        o_l = torch.min(a[0], b[0])
        o_r = torch.max(a[1], b[1])
        i_l = torch.max(a[0], b[0])
        i_r = torch.min(a[1], b[1])
        inter = torch.clamp(i_r - i_l, min=0)
        return inter / (o_r - o_l)


class PHOSA(nn.Module):
    def __init__(
        self,
        translations_object,
        rotations_object,
        verts_object_og,
        faces_object,
        cams_person,
        verts_person_og,
        faces_person,
        masks_object,
        masks_person,
        K_rois,
        target_masks,
        labels_person,
        labels_object,
        interaction_map_parts,
        class_name,
        int_scale_init=1.0,
    ):
        super(PHOSA, self).__init__()
        translation_init = translations_object.detach().clone()
        self.translations_object = nn.Parameter(translation_init, requires_grad=True)
        rotations_object = rotations_object.detach().clone()
        if rotations_object.shape[-1] == 3:
            rotations_object = matrix_to_rot6d(rotations_object)
        self.rotations_object = nn.Parameter(rotations_object, requires_grad=True)

        self.register_buffer("verts_object_og", verts_object_og)
        self.register_buffer("cams_person", cams_person)
        self.register_buffer("verts_person_og", verts_person_og)

        self.int_scales_object = nn.Parameter(
            int_scale_init * torch.ones(rotations_object.shape[0]).float(),
            requires_grad=True,
        )
        self.int_scale_object_mean = nn.Parameter(
            torch.tensor(int_scale_init).float(), requires_grad=False
        )
        self.int_scales_person = nn.Parameter(
            torch.ones(cams_person.shape[0]).float(), requires_grad=True
        )
        self.int_scale_person_mean = nn.Parameter(
            torch.tensor(1.0).float().cuda(), requires_grad=False
        )
        self.register_buffer("ref_mask", (target_masks > 0).float())
        self.register_buffer("keep_mask", (target_masks >= 0).float())
        self.register_buffer("K_rois", K_rois)
        self.register_buffer("faces_object", faces_object.unsqueeze(0))
        self.register_buffer(
            "textures_object", torch.ones(1, len(faces_object), 1, 1, 1, 3)
        )
        self.register_buffer(
            "textures_person", torch.ones(1, len(faces_person), 1, 1, 1, 3)
        )
        self.register_buffer("faces_person", faces_person.unsqueeze(0))
        self.cuda()

        # Setup renderer
        K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
        R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        t = torch.zeros(1, 3).cuda()
        self.renderer = nr.renderer.Renderer(
            image_size=IMAGE_SIZE, K=K, R=R, t=t, orig_size=1
        )
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = 0.3
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1, 1, 1]
        self.register_buffer("masks_human", masks_person)
        self.register_buffer("masks_object", masks_object)
        verts_object = self.get_verts_object()
        verts_person = self.get_verts_person()
        self.faces, self.textures = get_faces_and_textures(
            [verts_object, verts_person], [faces_object, faces_person]
        )

        self.losses = Losses(
            renderer=self.renderer,
            ref_mask=self.ref_mask,
            keep_mask=self.keep_mask,
            K_rois=self.K_rois,
            interaction_map_parts=interaction_map_parts,
            labels_person=labels_person,
            labels_object=labels_object,
            class_name=class_name,
        )

    def assign_human_masks(self, masks_human=None, min_overlap=0.5):
        """
        Uses a greedy matching algorithm to assign masks to human instances. The
        assigned human masks are used to compute the ordinal depth loss.

        If the human predictor uses the same instances as the segmentation algorithm,
        then this greedy assignment is unnecessary as the human instances will already
        have corresponding masks.

        1. Compute IOU between all human silhouettes and human masks
        2. Sort IOUs
        3. Assign people to masks in order, skipping people and masks that
            have already been assigned.

        Args:
            masks_human: Human bitmask tensor from instance segmentation algorithm.
            min_overlap (float): Minimum IOU threshold to assign the human mask to a
                human instance.

        Returns:
            N_h x
        """
        f = self.faces_person
        verts_person = self.get_verts_person()
        if masks_human is None:
            return torch.zeros(verts_person.shape[0], IMAGE_SIZE, IMAGE_SIZE).cuda()
        person_silhouettes = torch.cat(
            [self.renderer(v.unsqueeze(0), f, mode="silhouettes") for v in verts_person]
        ).bool()

        intersection = masks_human.unsqueeze(0) & person_silhouettes.unsqueeze(1)
        union = masks_human.unsqueeze(0) | person_silhouettes.unsqueeze(1)

        iou = intersection.sum(dim=(2, 3)).float() / union.sum(dim=(2, 3)).float()
        iou = iou.cpu().numpy()
        # https://stackoverflow.com/questions/30577375
        best_indices = np.dstack(np.unravel_index(np.argsort(-iou.ravel()), iou.shape))[
            0
        ]
        human_indices_used = set()
        mask_indices_used = set()
        # If no match found, mask will just be empty, incurring 0 loss for depth.
        human_masks = torch.zeros(verts_person.shape[0], IMAGE_SIZE, IMAGE_SIZE).bool()
        for human_index, mask_index in best_indices:
            if human_index in human_indices_used:
                continue
            if mask_index in mask_indices_used:
                continue
            if iou[human_index, mask_index] < min_overlap:
                break
            human_masks[human_index] = masks_human[mask_index]
            human_indices_used.add(human_index)
            mask_indices_used.add(mask_index)
        return human_masks.cuda()

    def get_verts_object(self):
        return compute_transformation_persp(
            meshes=self.verts_object_og,
            translations=self.translations_object,
            rotations=rot6d_to_matrix(self.rotations_object),
            intrinsic_scales=self.int_scales_object,
        )

    def get_verts_person(self):
        return compute_transformation_ortho(
            meshes=self.verts_person_og,
            cams=self.cams_person,
            intrinsic_scales=self.int_scales_person,
            focal_length=1.0,
        )

    def compute_ordinal_depth_loss(self):
        verts_object = self.get_verts_object()
        verts_person = self.get_verts_person()

        silhouettes = []
        depths = []

        for v in verts_object:
            _, depth, sil = self.renderer.render(
                v.unsqueeze(0), self.faces_object, self.textures_object
            )
            depths.append(depth)
            silhouettes.append((sil == 1).bool())
        for v in verts_person:
            _, depth, sil = self.renderer.render(
                v.unsqueeze(0), self.faces_person, self.textures_person
            )
            depths.append(depth)
            silhouettes.append((sil == 1).bool())
        masks = torch.cat((self.masks_object, self.masks_human))
        return self.losses.compute_ordinal_depth_loss(masks, silhouettes, depths)

    def forward(self, loss_weights=None):
        """
        If a loss weight is zero, that loss isn't computed (to avoid unnecessary
        compute).
        """
        loss_dict = {}
        verts_object = self.get_verts_object()
        verts_person = self.get_verts_person()
        if loss_weights is None or loss_weights["lw_sil"] > 0:
            loss_dict.update(
                self.losses.compute_sil_loss(
                    verts=verts_object, faces=[self.faces_object] * len(verts_object)
                )
            )
        if loss_weights is None or loss_weights["lw_inter"] > 0:
            loss_dict.update(
                self.losses.compute_interaction_loss(
                    verts_person=verts_person, verts_object=verts_object
                )
            )
        if loss_weights is None or loss_weights["lw_inter_part"] > 0:
            loss_dict.update(
                self.losses.compute_interaction_loss_parts(
                    verts_person=verts_person, verts_object=verts_object
                )
            )
        if loss_weights is None or loss_weights["lw_scale"] > 0:
            loss_dict["loss_scale"] = self.losses.compute_intrinsic_scale_prior(
                intrinsic_scales=self.int_scales_object,
                intrinsic_mean=self.int_scale_object_mean,
            )
        if loss_weights is None or loss_weights["lw_scale_person"] > 0:
            loss_dict["loss_scale_person"] = self.losses.compute_intrinsic_scale_prior(
                intrinsic_scales=self.int_scales_person,
                intrinsic_mean=self.int_scale_person_mean,
            )
        if loss_weights is None or loss_weights["lw_depth"] > 0:
            loss_dict.update(self.compute_ordinal_depth_loss())
        return loss_dict

    def render(self, renderer):
        verts_combined = combine_verts(
            [self.get_verts_object(), self.get_verts_person()]
        )
        image, _, mask = renderer.render(
            vertices=verts_combined, faces=self.faces, textures=self.textures
        )
        image = np.clip(image[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
        mask = mask[0].detach().cpu().numpy().astype(bool)
        return image, mask

    def get_parameters(self):
        """
        Computes a json-serializable dictionary of optimized parameters.

        Returns:
            parameters (dict): Dictionary mapping parameter name to list.
        """
        parameters = {
            "scales_object": self.int_scales_object,
            "scales_person": self.int_scales_person,
            "rotations_object": rot6d_to_matrix(self.rotations_object),
            "translations_object": self.translations_object,
        }
        for k, v in parameters.items():
            parameters[k] = v.detach().cpu().numpy().tolist()
        return parameters

    def save_obj(self, fname):
        with open(fname, "w") as fp:
            verts_combined = combine_verts(
                [self.get_verts_object(), self.get_verts_person()]
            )
            for v in tqdm.tqdm(verts_combined[0]):
                fp.write(f"v {v[0]:f} {v[1]:f} {v[2]:f}\n")
            o = 1
            for f in tqdm.tqdm(self.faces[0]):
                fp.write(f"f {f[0] + o:d} {f[1] + o:d} {f[2] + o:d}\n")


def optimize_human_object(
    person_parameters,
    object_parameters,
    class_name="bicycle",
    mesh_index=0,
    loss_weights=None,
    num_iterations=400,
    lr=1e-3,
):
    if loss_weights is None:
        loss_weights = DEFAULT_LOSS_WEIGHTS[class_name]

    # Load mesh data.
    mesh_path = MESH_MAP[class_name][mesh_index]
    verts_object_og, faces_object = nr.load_obj(mesh_path)
    verts_object_og, faces_object = center_vertices(verts_object_og, faces_object)
    faces_person = torch.IntTensor(np.load(SMPL_FACES_PATH).astype(int)).cuda()

    # Get part labels.
    with open(PART_LABELS["person"][0][0], "r") as f:
        labels_person = json.load(f)["labels"]
    labels_person = {k: torch.LongTensor(v).cuda() for k, v in labels_person.items()}
    part_fname, interaction_map_parts = PART_LABELS[class_name][mesh_index]
    if part_fname:
        with open(part_fname, "r") as f:
            labels_object = json.load(f)["labels"]
        # Sometimes list is empty.
        labels_object = {k: v for k, v in labels_object.items() if v}
    else:
        # If no json is defined, we just map all vertices to a body part called "all".
        labels_object = {"all": np.arange(len(verts_object_og))}
    labels_object = {k: torch.LongTensor(v).cuda() for k, v in labels_object.items()}

    model = PHOSA(
        translations_object=object_parameters["translations"],
        rotations_object=object_parameters["rotations"],
        verts_object_og=verts_object_og,
        faces_object=faces_object,
        target_masks=object_parameters["target_masks"],
        cams_person=person_parameters["cams"],
        verts_person_og=person_parameters["verts"],
        faces_person=faces_person,
        masks_object=object_parameters["masks"],
        masks_person=person_parameters["masks"],
        K_rois=object_parameters["K_roi"],
        labels_person=labels_person,
        labels_object=labels_object,
        interaction_map_parts=interaction_map_parts,
        class_name=class_name,
        int_scale_init=MEAN_INTRINSIC_SCALE[class_name],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loop = tqdm(range(num_iterations))
    for _ in loop:
        optimizer.zero_grad()
        loss_dict = model(loss_weights=loss_weights)
        loss_dict_weighted = {
            k: loss_dict[k] * loss_weights[k.replace("loss", "lw")] for k in loss_dict
        }
        loss = sum(loss_dict_weighted.values())
        loop.set_description(f"Loss {loss.data:.4f}")
        loss.backward()
        optimizer.step()
    return model


def visualize_human_object(model, image):
    # Rendered frontal image
    if image.max() > 1:
        image = image / 255.0
    rend, mask = model.render(model.renderer)
    h, w, c = image.shape
    L = max(h, w)
    new_image = np.pad(image.copy(), ((0, L - h), (0, L - w), (0, 0)))
    new_image[mask] = rend[mask]
    new_image = (new_image[:h, :w] * 255).astype(np.uint8)

    # Rendered top-down image
    theta = 1.3
    d = 3
    x, y = np.cos(theta), np.sin(theta)
    mx, my, mz = model.get_verts_object().mean(dim=(0, 1)).detach().cpu().numpy()
    K = model.renderer.K
    R2 = torch.cuda.FloatTensor([[[1, 0, 0], [0, x, -y], [0, y, x]]])
    t2 = torch.cuda.FloatTensor([mx, my + d, mz])
    top_renderer = nr.renderer.Renderer(
        image_size=IMAGE_SIZE, K=K, R=R2, t=t2, orig_size=1
    )
    top_renderer.background_color = [1, 1, 1]
    top_renderer.light_direction = [1, 0.5, 1]
    top_renderer.light_intensity_direction = 0.3
    top_renderer.light_intensity_ambient = 0.5
    top_renderer.background_color = [1, 1, 1]
    top_down, _ = model.render(top_renderer)
    top_down = (top_down * 255).astype(np.uint8)
    return new_image, top_down
