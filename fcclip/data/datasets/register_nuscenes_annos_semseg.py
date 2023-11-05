"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_coco_panoptic_annos_semseg.py
"""

import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
# from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from . import openseg_classes

from detectron2.utils.file_io import PathManager

import numpy as np
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from nuscenes.utils.geometry_utils import  BoxVisibility
from nuscenes.utils.splits import create_splits_scenes

categories = [
             'adult',
             'child',
             'wheelchair',
             'stroller',
             'personal_mobility',
             'police_officer',
             'construction_worker',
             'animal',
             'car',
             'motorcycle',
             'bicycle',
             'bendy',
             'rigid',
             'truck',
             'construction',
             'ambulance',
             'police',
             'trailer',
             'barrier',
             'trafficcone',
             'pushable_pullable',
             'debris',
             'bicycle_rack']

def get_nuscenes_dicts(path="./", version='v1.0-mini', cam_name = "CAM_FRONT", is_train = True, categories=None):
    """
    This is a helper fuction that create dicts from nuscenes to detectron2 format.
    Nuscenes annotation use 3d bounding box, but for detectron we need 2d bounding box.
    The simplest solution is get max x, min x, max y and min y coordinates from 3d bb and
    create 2d box. So we lost accuracy, but this is not critical.
    :param path: <string>. Path to Nuscenes dataset.
    :param version: <string>. Nuscenes dataset version name.
    :param categories <list<string>>. List of selected categories for detection.
        Get from https://www.nuscenes.org/data-annotation
        Categories names:
            ['human.pedestrian.adult',
             'human.pedestrian.child',
             'human.pedestrian.wheelchair',
             'human.pedestrian.stroller',
             'human.pedestrian.personal_mobility',
             'human.pedestrian.police_officer',
             'human.pedestrian.construction_worker',
             'animal',
             'vehicle.car',
             'vehicle.motorcycle',
             'vehicle.bicycle',
             'vehicle.bus.bendy',
             'vehicle.bus.rigid',
             'vehicle.truck',
             'vehicle.construction',
             'vehicle.emergency.ambulance',
             'vehicle.emergency.police',
             'vehicle.trailer',
             'movable_object.barrier',
             'movable_object.trafficcone',
             'movable_object.pushable_pullable',
             'movable_object.debris',
             'static_object.bicycle_rack']
    :return: <dict>. Return dict with data annotation in detectron2 format.
    """
    assert(path[-1] == "/"), "Insert '/' in the end of path"
    nusc = NuScenes(version=version, dataroot=path, verbose=False)

    split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]

    scenes = create_splits_scenes()[split]
    samples = [samp for samp in nusc.sample]

    # remove samples that aren't in this split
    samples = [samp for samp in samples if
                nusc.get('scene', samp['scene_token'])['name'] in scenes]

    # sort by scene, timestamp (only to make chronological viz easier)
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
    ixes = samples

    # Select all catecategories if not set
    if categories == None:
        categories = [data["name"] for data in nusc.category]
    assert(isinstance(categories, list)), "Categories type must be list"

    dataset_dicts = []
    idx = 0
    for i in tqdm(range(0, len(ixes))):
        record = {}
        rec = ixes[i]
        samp = nusc.get('sample_data', rec['data'][cam_name])
        imgname = os.path.join(nusc.dataroot, samp['filename'])
        record["file_name"] = imgname

        sens = nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
        intrin = sens['camera_intrinsic']
        rot = Quaternion(sens['rotation']).rotation_matrix
        tran = sens['translation']
        record["intrin"] = np.array(intrin,dtype=np.float64)
        record["rot"] = np.array(rot,dtype=np.float64)
        record["tran"] = np.array(tran,dtype=np.float64)

        egopose = nusc.get('ego_pose',
                                nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse

        objs = []
        for tok in rec['anns']:
            inst = nusc.get('sample_annotation', tok)
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            c = inst['category_name']
            if '.' in c:
                c = c.split('.')[-1]
            obj = {
                "pts": pts,
                "category_name": c,
                "category_id": categories.index(c),
            }
            objs.append(obj)

        record["annotations"] = objs
        
        dataset_dicts.append(record)
    return dataset_dicts


def register_all_nuscenes(root):
    train_get_dicts = lambda p=root+"/nuscenes/", c=categories: get_nuscenes_dicts(path=p, categories=c, is_train=True)
    DatasetCatalog.register("nusc_mini_train", train_get_dicts)
    MetadataCatalog.get("nusc_mini_train").thing_classes = categories

    val_get_dicts = lambda p=root+"/nuscenes/", c=categories: get_nuscenes_dicts(path=p, categories=c, is_train=False)
    DatasetCatalog.register("nusc_mini_val", train_get_dicts)
    MetadataCatalog.get("nusc_mini_val").thing_classes = categories


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_nuscenes(_root)