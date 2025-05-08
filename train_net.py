    #!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (default_argument_parser, default_setup, hooks,
                               launch)
from detectron2.evaluation import verify_results
from detectron2.utils.logger import setup_logger
from DUONet import OpenDetTrainer, add_opendet_config, builtin
import argparse



from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

from detectron2.data.datasets.pascal_voc import load_voc_instances
from detectron2.structures import BoxMode
import math
import json

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from detectron2.utils.file_io import PathManager

import random
import numpy as np
import torch


class Register:
    
    """register my dataset"""
    #CLASS_NAMES = ['__background__', '1', '2', '3', '4', '5']  # 保留 background 类

    CLASS_NAMES = [
    # ShipRSImageNet_V1_CLASS_NAMES = [
    "Submarine", "Nimitz", "Midway", "Ticonderoga", "Atago DD", 
    "Hatsuyuki DD", "Hyuga DD", "Asagiri DD", "Perry FF", "Patrol",
    "YuTing LL", "YuDeng LL", "YuDao LL", "Austin LL", "Osumi LL",
    "LSD 41 LL", "LHA LL", "Commander", "Medical Ship",
    "Test Ship", "Training Ship", "Masyuu AS", "Sanantonio AS",
    "RoRo", "Cargo", "Barge", "Tugboat", "Ferry", "Yacht", "Wasp LL", "YuZhao LL",
    "Sailboat", "Fishing Vessel", "Oil Tanker", "Hovercraft", "Motorboat","Dock",
    # T2_CLASS_NAMES = [
    "EPF", "AOE", "Enterprise", "Container Ship", "Arleigh Burke DD",
    # UNK_CLASS = ["
    "unknown",
]

    ROOT = '/home/quchenyu/DUONet/datasets/ShipRSImageNet_V1'  # 数据集路径

    def __init__(self,):
        self.CLASS_NAMES = Register.CLASS_NAMES #or ['__background__', ]
        # 数据集路径
        self.DATASET_ROOT = Register.ROOT
        self.ANN_ROOT = self.DATASET_ROOT

        # 声明数据集的子集
        self.PREDEFINED_SPLITS_DATASET = {
            "coco_my_train1234": (self.DATASET_ROOT,'train','voc2007'),
            "coco_my_val1234": (self.DATASET_ROOT,'test','voc2007'),
        }


    def get_abbox_angle(self, annotation):
        centerx = (annotation[1] + annotation[3] + annotation[5] + annotation[7]) / 4
        centery = (annotation[2] + annotation[4] + annotation[6] + annotation[8]) / 4
        h = math.sqrt(math.pow((annotation[1] - annotation[3]), 2) + math.pow(
            (annotation[2] - annotation[4]), 2))
        w = math.sqrt(math.pow((annotation[1] - annotation[7]), 2) + math.pow(
            (annotation[2] - annotation[8]), 2))
        a = - math.degrees(math.atan2((annotation[8] - annotation[2]), (annotation[7] - annotation[1])))
        return a


    def get_dicts(self, dirname, split, class_names):
        """
        purpose: 用于定义自己的数据集格式，返回指定对象的数据
        :param ids: 需要返回数据的名称（id）号
        :return: 指定格式的数据字典
        """


        with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
            fileids = np.loadtxt(f, dtype=str)

        # Needs to read many small annotation files. Makes sense at local
        annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
        dicts = []
        for fileid in fileids:
            anno_file = os.path.join(annotation_dirname, fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".bmp")

            with PathManager.open(anno_file) as f:
                tree = ET.parse(f)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
                # "height": int(tree.findall("./size/h")[0].text),   ##DOSR
                # "width": int(tree.findall("./size/w")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text

                use_origin=True
                if use_origin:
                    bbox = obj.find("rotated_box")
                    angle = math.degrees(float(bbox.find('rot').text))

                else:
                    bbox = obj.find("rotated_box")
                    polygon = obj.find("polygon")
                    x1, y1 = float(polygon.find("x1").text), float(polygon.find("y1").text)
                    x2, y2 = float(polygon.find("x2").text), float(polygon.find("y2").text)
                    x3, y3 = float(polygon.find("x3").text), float(polygon.find("y3").text)
                    x4, y4 = float(polygon.find("x4").text), float(polygon.find("y4").text)
                    #array = [0,x1,y1,x2,y2,x3,y3,x4,y4]
                    array = [0,x2,y2,x1,y1,x4,y4,x3,y3]
                    angle = self.get_abbox_angle(array)

                bbox_values = [
                        #float(bbox.find(x).text) for x in ['cx', 'cy', 'w', 'h']
                        float(bbox.find(x).text) for x in ['cx', 'cy', 'width', 'height']
                ] + [angle]

                instances.append(
                    {"category_id": class_names.index(cls), "bbox": bbox_values, "bbox_mode": BoxMode.XYWHA_ABS}
                )
            r["annotations"] = instances
            dicts.append(r)
        return dicts

    def register_dataset(self,):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """
        for key, (dirname, split, year) in self.PREDEFINED_SPLITS_DATASET.items():

            self.register_dataset_instances(name=key,dirname=dirname,split=split,year=year,class_names=self.CLASS_NAMES)

    # @staticmethod
    def register_dataset_instances(self, name, dirname, split, year, class_names):
        """
        purpose: register datasets to DatasetCatalog,
                 register metadata to MetadataCatalog and set attribute
        注册数据集实例，加载数据集中的对象实例
        """
        print(10*'>','register_dataset_instances')
        # print(name,image_root,json_file)
        DatasetCatalog.register(name, lambda: self.get_dicts(dirname, split, class_names))
        # DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
        # MetadataCatalog.get(name).set(json_file=json_file,
        #                               image_root=image_root,
        #                               evaluator_type="coco")

        MetadataCatalog.get(name).set(
            thing_classes=list(class_names), dirname=dirname, year=year, split=split, evaluator_type='pascal_voc'
        )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    
    cfg = get_cfg()      
    Register().register_dataset()  # register my dataset

    # add opendet config
    add_opendet_config(cfg)    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Note: we use the key ROI_HEAD.NUM_KNOWN_CLASSES
    # for open-set data processing and evaluation.
    if 'RetinaNet' in cfg.MODEL.META_ARCHITECTURE:
        cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = cfg.MODEL.RETINANET.NUM_KNOWN_CLASSES
    # add output dir if not exist
    if cfg.OUTPUT_DIR == "./output":
        config_name = os.path.basename(args.config_file).split(".yaml")[0]
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, config_name)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR,
                 distributed_rank=comm.get_rank(), name="DUONet")
    return cfg


def main(args):
    cfg = setup(args)
    print(cfg.MODEL.WEIGHTS)
    if args.eval_only:
        model = OpenDetTrainer.build_model(cfg)
        
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = OpenDetTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(OpenDetTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = OpenDetTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
