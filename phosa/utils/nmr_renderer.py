# Copyright (c) Facebook, Inc. and its affiliates.
import cv2
import neural_renderer as nr
import numpy as np
import torch

colors = {
    # colorblind/print/copy safe:
    "blue": [0.65098039, 0.74117647, 0.85882353],
    "pink": [0.9, 0.7, 0.7],
    "mint": [166 / 255.0, 229 / 255.0, 204 / 255.0],
    "mint2": [202 / 255.0, 229 / 255.0, 223 / 255.0],
    "green": [153 / 255.0, 216 / 255.0, 201 / 255.0],
    "green2": [171 / 255.0, 221 / 255.0, 164 / 255.0],
    "red": [251 / 255.0, 128 / 255.0, 114 / 255.0],
    "orange": [253 / 255.0, 174 / 255.0, 97 / 255.0],
    "yellow": [250 / 255.0, 230 / 255.0, 154 / 255.0],
    "white": [1, 1, 1],
}


class OrthographicRenderer(object):
    def __init__(self, image_size=256, texture_size=1):
        self.image_size = image_size
        self.renderer = nr.Renderer(
            image_size, camera_mode="look_at", perspective=False
        )
        self.t_size = texture_size
        self.set_light_dir([1, 0.5, -1], int_dir=0.3, int_amb=0.7)
        self.set_bgcolor([1, 1, 1.0])
        self.default_cam = torch.cuda.FloatTensor([[0.9, 0, 0]])

    def __call__(
        self, vertices, faces, cam=None, textures=None, color_name="white", image=None
    ):
        if vertices.ndimension() == 2:
            vertices = vertices.unsqueeze(0)
        if faces.ndimension() == 2:
            faces = faces.unsqueeze(0)
        if textures is None:
            textures = torch.ones(
                len(faces),
                faces.shape[1],
                self.t_size,
                self.t_size,
                self.t_size,
                3,
                dtype=torch.float32,
            ).cuda()
            color = torch.FloatTensor(colors[color_name]).cuda()
            textures = color * textures
        elif textures.ndimension() == 5:
            textures = textures.unsqueeze(0)
        if cam is None:
            cam = self.default_cam

        proj_verts = self.orthographic_proj(vertices, cam)
        # Flip the y-axis to align with image coordinates.
        proj_verts[:, :, 1] *= -1
        # Need to also flip the faces inside out.
        faces = faces[..., [2, 1, 0]]

        rend, depth, sil = self.renderer.render(proj_verts, faces, textures)
        rend = rend.cpu().numpy()[0].transpose((1, 2, 0))
        rend = np.clip(rend, 0, 1)

        if image is None:
            return rend
        sil = sil.cpu().numpy()[0]
        h, w, *_ = image.shape
        L = max(h, w)
        if image.max() > 1:
            image = image.astype(float) / 255.0
        new_image = np.pad(image, ((0, L - h), (0, L - w), (0, 0)))
        new_image = cv2.resize(new_image, (self.image_size, self.image_size))
        new_image[sil == 1] = rend[sil == 1]
        r = self.image_size / L
        new_image = new_image[: int(h * r), : int(w * r)]
        return new_image

    @classmethod
    def orthographic_proj(cls, X, cam, offset_z=0.0):
        """
        Computes scaled orthographic projection of vertices with no rotation. Uses same
        formulation as HMR:
            x = s * (X + t)

        Args:
            X (B x N x 3): Vertices.
            cam (B x 3): Weak-perspective camera [s, tx, ty].
            offset_z.
        """
        scale = cam[:, 0].contiguous().view(-1, 1, 1)
        trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)
        proj_xy = scale * (X[:, :, :2] + trans)
        proj_z = X[:, :, 2, None] + offset_z
        return torch.cat((proj_xy, proj_z), 2)

    def set_light_dir(self, direction=(1, 0.5, -1), int_dir=0.3, int_amb=0.7):
        self.renderer.light_direction = direction
        self.renderer.light_intensity_directional = int_dir
        self.renderer.light_intensity_ambient = int_amb

    def set_bgcolor(self, color):
        self.renderer.background_color = color


class PerspectiveRenderer(object):
    def __init__(self, image_size=256, texture_size=1):
        self.image_size = image_size
        self.default_K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
        self.R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        self.t = torch.zeros(1, 3).cuda()
        self.default_translation = torch.cuda.FloatTensor([[0, 0, 2]])
        self.t_size = texture_size
        self.renderer = nr.renderer.Renderer(
            image_size=image_size, K=self.default_K, R=self.R, t=self.t, orig_size=1
        )
        self.set_light_dir([1, 1, 0.4], int_dir=0.3, int_amb=0.7)
        self.set_bgcolor([0, 0, 0])

    def __call__(
        self,
        vertices,
        faces,
        textures=None,
        translation=None,
        rotation=None,
        image=None,
        color_name="white",
        K=None,
    ):
        # Batch currently not supported.
        # TODO(@jason): to add batch, just allow for multiple images.
        if vertices.ndimension() == 2:
            vertices = vertices.unsqueeze(0)
        if faces.ndimension() == 2:
            faces = faces.unsqueeze(0)
        if textures is None:
            textures = torch.ones(
                len(faces),
                faces.shape[1],
                self.t_size,
                self.t_size,
                self.t_size,
                3,
                dtype=torch.float32,
            ).cuda()
            color = torch.FloatTensor(colors[color_name]).cuda()
            textures = color * textures
        elif textures.ndimension() == 5:
            textures = textures.unsqueeze(0)

        if translation is None:
            translation = self.default_translation
        if not isinstance(translation, torch.Tensor):
            translation = torch.FloatTensor(translation).to(vertices.device)
        if translation.ndimension() == 1:
            translation = translation.unsqueeze(0)

        if rotation is not None:
            vertices = torch.matmul(vertices, rotation)
        vertices += translation

        if K is not None:
            self.renderer.K = K

        rend, depth, sil = self.renderer.render(vertices, faces, textures)
        rend = rend.detach().cpu().numpy().transpose(0, 2, 3, 1)  # B x H x W x C
        rend = np.clip(rend, 0, 1)[0]

        self.renderer.K = self.default_K  # Restore just in case.
        if image is None:
            return rend
        else:
            sil = sil.detach().cpu().numpy()[0]
            h, w, *_ = image.shape
            L = max(h, w)
            if image.max() > 1:
                image = image.astype(float) / 255.0
            new_image = np.pad(image, ((0, L - h), (0, L - w), (0, 0)))
            new_image = cv2.resize(new_image, (self.image_size, self.image_size))
            new_image[sil > 0] = rend[sil > 0]
            r = self.image_size / L
            new_image = new_image[: int(h * r), : int(w * r)]
            return new_image

    def set_light_dir(self, direction=(1, 0.5, -1), int_dir=0.3, int_amb=0.7):
        self.renderer.light_direction = direction
        self.renderer.light_intensity_directional = int_dir
        self.renderer.light_intensity_ambient = int_amb

    def set_bgcolor(self, color):
        self.renderer.background_color = color
