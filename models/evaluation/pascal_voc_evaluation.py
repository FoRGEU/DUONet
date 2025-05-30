# Copyright (c) Facebook, Inc. and its affiliates.
# Code is modified from https://github.com/JosephKJ/OWOD

import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
from tabulate import tabulate
import math
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.evaluation.pascal_voc_evaluation import voc_ap
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.structures import BoxMode, RotatedBoxes, pairwise_iou_rotated
from detectron2 import _C


def obb2hbb(obb):
    
    cx, cy, w, h, angle = obb
    theta = np.radians(angle)
    
    # Calculate the four corners of the OBB
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate and translate the corners
    rotated_corners = np.dot(corners, rotation_matrix) + np.array([cx, cy])
    
    # Find the min and max points
    x_min, y_min = np.min(rotated_corners, axis=0)
    x_max, y_max = np.max(rotated_corners, axis=0)
    
    return [x_min, y_min, x_max, y_max]



class PascalVOCDetectionEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg=None):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)

        # Too many tiny files, download all to local for speed.
        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(
            meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        #assert meta.year in [2007, 2012], meta.year
        self.logger = logging.getLogger(__name__)
        self._is_2007 = True
        self._cpu_device = torch.device("cpu")
        if cfg is not None:
            self.output_dir = cfg.OUTPUT_DIR
            self.total_num_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            self.unknown_class_index = self.total_num_class - 1
            self.num_known_classes = cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES
            self.known_classes = self._class_names[:self.num_known_classes]

    def reset(self):
        # class name -> list of prediction strings
        self._predictions = defaultdict(list)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            
            
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()

            for box, score, cls in zip(boxes, scores, classes):
                cx, cy, width, height, rot = box
                #rot = -math.radians(rot)
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {cx:.1f} {cy:.1f} {width:.1f} {height:.1f} {rot:.1f}"
                )

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        wi_at_recall = {}
        # for r in range(1, 10):
        for r in [8]:
            r = r/10
            wi = self.compute_WI_at_a_recall_level(
                recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            for cls_id, rec in enumerate(recall):
                if cls_id in range(self.num_known_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(
                        rec[i] - recall_level))
                    tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][cls_id][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        #print(all_predictions)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self.logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        dirname = os.path.join(self.output_dir, 'pascal_voc_eval')
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        # with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
        res_file_template = os.path.join(dirname, "{}.txt")

        aps = defaultdict(list)  # iou -> ap per class
        recs = defaultdict(list)
        precs = defaultdict(list)
        all_recs = defaultdict(list)
        all_precs = defaultdict(list)
        unk_det_as_knowns = defaultdict(list)
        num_unks = defaultdict(list)
        tp_plus_fp_cs = defaultdict(list)
        fp_os = defaultdict(list)
        
        for cls_id, cls_name in enumerate(self._class_names):
            lines = predictions.get(cls_id, [""])
            
            #lines是空的
            with open(res_file_template.format(cls_name), "w") as f:
                f.write("\n".join(lines))
            
            for thresh in [50, ]:
                # for thresh in range(50, 100, 5):
                (rec, prec, ap, unk_det_as_known, num_unk,
                 tp_plus_fp_closed_set, fp_open_set) = voc_eval(
                    res_file_template,
                    self._anno_file_template,
                    self._image_set_path,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                    known_classes=self.known_classes
                )
                
                aps[thresh].append(ap * 100)
                unk_det_as_knowns[thresh].append(unk_det_as_known)
                num_unks[thresh].append(num_unk)
                all_precs[thresh].append(prec)
                all_recs[thresh].append(rec)
                tp_plus_fp_cs[thresh].append(tp_plus_fp_closed_set)
                fp_os[thresh].append(fp_open_set)
                try:
                    recs[thresh].append(rec[-1] * 100)
                    precs[thresh].append(prec[-1] * 100)
                except:
                    recs[thresh].append(0)
                    precs[thresh].append(0)
        
        results_2d = {}
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        results_2d['mAP'] = mAP[50]

        wi = self.compute_WI_at_many_recall_level(
            all_recs, tp_plus_fp_cs, fp_os)
        results_2d['WI'] = wi[0.8][50]
        
        total_num_unk_det_as_known = {iou: np.sum(
            x) for iou, x in unk_det_as_knowns.items()}
        # total_num_unk = num_unks[50][0]
        # self.logger.info('num_unk ' + str(total_num_unk))
        results_2d['AOSE'] = total_num_unk_det_as_known[50]

        # class-wise P-R
        # self.logger.info(self._class_names)
        # self.logger.info("AP50: " + str(['%.1f' % x for x in aps[50]]))
        # self.logger.info("P50: " + str(['%.1f' % x for x in precs[50]]))
        # self.logger.info("R50: " + str(['%.1f' % x for x in recs[50]]))

        # Known
        results_2d.update({
            "AP@K": np.mean(aps[50][:self.num_known_classes]),
            "P@K": np.mean(precs[50][:self.num_known_classes]),
            "R@K": np.mean(recs[50][:self.num_known_classes]),
        })

        # Unknown
        results_2d.update({
            "AP@U": np.mean(aps[50][-1]),
            "P@U": np.mean(precs[50][-1]),
            "R@U": np.mean(recs[50][-1]),
        })
        results_head = list(results_2d.keys())
        results_data = [[float(results_2d[k]) for k in results_2d]]
        table = tabulate(
            results_data,
            tablefmt="pipe",
            floatfmt=".5f",
            headers=results_head,
            numalign="left",
        )
        self.logger.info("\n" + table)

        return {metric:round(x,2) for metric, x in zip(results_head, results_data[0])}


def compute_iou_dt_gt(dt, gt):
    dt = RotatedBoxes(boxlist_to_tensor(dt, output_box_dim=5))
    gt = RotatedBoxes(boxlist_to_tensor(gt, output_box_dim=5))
    #return pairwise_iou_rotated(dt, gt)
    return _C.box_iou_rotated(dt.tensor, gt.tensor)



def boxlist_to_tensor(boxlist, output_box_dim=5):
    if type(boxlist) == np.ndarray:
        box_tensor = torch.from_numpy(boxlist)
        if box_tensor.dim() == 1:
          box_tensor = box_tensor.unsqueeze(0)
    elif type(boxlist) == list:
        if boxlist == []:
            return torch.zeros((0, output_box_dim), dtype=torch.float32)
        else:
            box_tensor = torch.FloatTensor(boxlist)
    else:
        raise Exception("Unrecognized boxlist type")
    
    input_box_dim = box_tensor.shape[1]
    if input_box_dim != output_box_dim:
        if input_box_dim == 4 and output_box_dim == 5:
            box_tensor = BoxMode.convert(box_tensor, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS)
        else:
            raise Exception(
                "Unable to convert from {}-dim box to {}-dim box".format(
                    input_box_dim, output_box_dim
                )
            )
    return box_tensor


@lru_cache(maxsize=None)
def parse_rec(filename, known_classes):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        cls_name = obj.find("name").text
        # translate unseen classes to unknown
        if cls_name not in known_classes:
            cls_name = 'unknown'

        obj_struct["name"] = cls_name
        # obj_struct["pose"] = obj.find("pose").text
        # obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        
        # hbb                        
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
           float(bbox.find("xmin").text),
           float(bbox.find("ymin").text),
           float(bbox.find("xmax").text),
           float(bbox.find("ymax").text)
        ]
        objects.append(obj_struct)
        
        # obb              
        # bbox = obj.find("rotated_box")
        # obj_struct["bbox"] = [
        #     float(bbox.find("cx").text),
        #     float(bbox.find("cy").text),
        #     float(bbox.find("width").text),
        #     float(bbox.find("height").text),
        #     # angle
        #     math.degrees(float(bbox.find("rot").text))
        # ]
        # objects.append(obj_struct)
        
        
# polygon        
        # polygon = obj.find("polygon")
        # x1, y1 = float(polygon.find("x1").text), float(polygon.find("y1").text)
        # x2, y2 = float(polygon.find("x2").text), float(polygon.find("y2").text)
        # x4, y4 = float(polygon.find("x4").text), float(polygon.find("y4").text)
        #
        # width = float(bbox.find("width").text)
        # dist12 = math.sqrt(math.pow(y2-y1, 2)+math.pow(x2-x1, 2))
        # dist14 = math.sqrt(math.pow(y4-y1, 2)+math.pow(x4-x1, 2))
        #
        #
        # if abs(dist12 - width) < abs(dist14 - width):
        #     vecy, vecx = y2 - y1, x2 - x1
        # else:
        #     vecy, vecx = y4 - y1, x4 - x1
        # radians = math.atan2(vecy, vecx)
        # angle = -math.degrees(radians)
        #
        # if angle >= 180:
        #     angle -= 360
        # elif angle < -180:
        #     angle += 360
        
        

    return objects


def compute_overlaps(BBGT, bb):
    # compute overlaps
    # intersection
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
        + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
        - inters
    )

    return inters / uni


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, known_classes=None):
    # first load gt
    # read list of images
    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(
            annopath.format(imagename), tuple(known_classes))
   
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox,
                                 "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)

    with open(detfile, "r") as f:
        lines = f.readlines()
    
    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]]
                  for x in splitlines]).reshape(-1, 5)
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        
        ########
        bb = BB[d, :].astype(float)
        
        bb = obb2hbb(bb)   
        ########
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)
        if BBGT.size > 0:
            ############
            #overlaps = compute_iou_dt_gt(bb, BBGT).numpy().reshape(-1,)
            overlaps = compute_overlaps(BBGT, bb)
            ############
            
            # if d == 0:
            #     print("#########BB###########")
            #     print(bb)
            #     print("#########BBGT###########")
            #     print(BBGT)
            # print("#########IOU###########")
            # print(overlaps)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    
    tp = np.cumsum(tp)
    
    rec = tp / float(npos)
    #print(rec)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    #print(prec)
    ap = voc_ap(rec, prec, use_07_metric)

    # compute unknown det as known
    unknown_class_recs = {}
    n_unk = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == 'unknown']
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[imagename] = {
            "bbox": bbox, "difficult": difficult, "det": det}

    if classname == 'unknown':
        return rec, prec, ap, 0, n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        R = unknown_class_recs[image_ids[d]]
        
        ##########
        bb = BB[d, :].astype(float)
        bb = obb2hbb(bb) 
        ##########
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            ###########
            #overlaps = compute_iou_dt_gt(bb, BBGT).numpy()
            overlaps = compute_overlaps(BBGT, bb)
            ###########
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set
