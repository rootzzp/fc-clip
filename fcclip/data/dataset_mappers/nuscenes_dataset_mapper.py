"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py
"""

import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances
from pyquaternion import Quaternion
import cv2
import torchvision

from PIL import Image

__all__ = ["NuscenesDatasetMapper"]

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class NuscenesDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOPanopticNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.data_aug_conf = {}
        self.data_aug_conf["final_dim"] = [512,512]
        self.data_aug_conf['resize_lim'] = (0.193, 0.225)
        self.data_aug_conf["bot_pct_lim"]=(0.0, 0.22)
        self.data_aug_conf["rot_lim"]=(-5.4, 5.4)
        self.data_aug_conf["rand_flip"]=True,
    
        self.nx = np.array([200,    200,    1])
        self.bx = np.array([-49.75, -49.75, 0])
        self.dx = np.array([0.5,    0.5,    20])

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img = Image.open(dataset_dict["file_name"])
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        # augmentation (resize, crop, horizontal flip, rotate)
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation(img)
        img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                resize=resize,
                                                resize_dims=resize_dims,
                                                crop=crop,
                                                flip=flip,
                                                rotate=rotate,
                                                )
        image_shape = img.size[:2]  # h, w
        dataset_dict["height"] = image_shape[0]
        dataset_dict["width"] = image_shape[1]
        dataset_dict["image"] = normalize_img(img)
        
        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3,dtype=torch.float64)
        post_rot = torch.eye(3,dtype=torch.float64)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2


        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        instances = Instances(image_shape)
        
        masks = []
        points = {}
        for segment_info in dataset_dict["annotations"]:
            class_id = segment_info["category_id"]
            category_name = segment_info["category_name"]
            pts = segment_info["pts"]
            if class_id not in points:
                points[class_id] = []
            points[class_id].append(pts)

        classes = []
        for k,pts in points.items():
            classes.append(k)
        classes = np.array([classes])
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        instances.post_tran = post_tran.unsqueeze(0)
        instances.post_rot = post_rot.unsqueeze(0)
        instances.intrin = torch.from_numpy(dataset_dict["intrin"]).unsqueeze(0)
        instances.rot = torch.from_numpy(dataset_dict["rot"]).unsqueeze(0)
        instances.tran = torch.from_numpy(dataset_dict["tran"]).unsqueeze(0)


        masks = []
        for k,pts in points.items():
            mask = self.get_binimg(pts)
            masks.append(mask)
        
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, 200, 200))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor.unsqueeze(0)

        dataset_dict["instances"] = instances

        return dataset_dict
    
    def sample_augmentation(self,img):
        H, W = img.size[0], img.size[1]
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate
    
    def img_transform(self, img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = get_rot(rotate/180*np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran
    
    def get_binimg(self, pts_list):
        img = np.zeros((self.nx[0], self.nx[1]))
        for pts in pts_list:
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return img
