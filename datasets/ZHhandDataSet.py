import json
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import os
import numpy as np


class ImageDataSet(Dataset):
    """ load images to be used on the complete pipeline of object detection,
     pose estimation and classification """

    def __init__(self, image_root: str, ann_root: str, annotation_files: list, n_classes=9):
        super(ImageDataSet, self).__init__()
        self.ann_dir = ann_root
        self.ann_files = annotation_files
        self.image_root = image_root  # the root direction of images
        self.n_classes = n_classes  # default is 0~8, sometimes will except the class 0

    def __len__(self):
        return len(self.ann_files)  # just the length of images, not the number of hand

    def __getitem__(self, idx):
        ann_file = os.path.join(self.ann_dir, self.ann_files[idx])
        ann = json.load(open(ann_file, 'r'))  # the json file of COCO format
        img_info = ann['images']
        annotations_list = ann['annotations']

        img_file = os.path.join(self.image_root, img_info['file_name'])
        # print(img_file)
        img = cv.imread(img_file)
        img_shape = [img_info["width"], img_info["height"]]

        kpts = []
        labels = []
        bboxes = []
        for annotation in annotations_list:
            keypoints = annotation['keypoints']  # list
            keypoints = self.strip_conf(keypoints)
            kpts.append(keypoints)
            labels.append(annotation["category_id"])  # the hand posture, categories id in 0~9
            # todo: the bbox should be rescale to a spare of 256x256
            bbox = self.scale_bbox(annotation["bbox"], img_shape)  # [x1,y1,x2,y2]
            # bbox = self.xywh2xyxy(annotation["bbox"])  # [x1,y1,x2,y2]
            bboxes.append(bbox)

        return img, img_file, labels, kpts, bboxes

    def strip_conf(self, kpts):
        kpts_wo_conf = []
        for i in range(21):
            kpts_wo_conf.append(kpts[i * 3 + 0])  # xi
            kpts_wo_conf.append(kpts[i * 3 + 1])  # yi
        return kpts_wo_conf

    def xywh2xyxy(self, bbox):
        lx, ly, w, h = bbox
        bbox = [lx, ly, lx + w, ly + h]
        return bbox

    def scale_bbox(self, bbox, img_shape):
        x1, y1, x2, y2 = self.xywh2xyxy(bbox)

        # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
        correction_factor = (x2 - x1) / (y2 - y1)
        if correction_factor > 1:
            # increase y side
            center = y1 + (y2 - y1) // 2
            length = int(round((y2 - y1) * correction_factor))
            y1 = max(0, center - length // 2)
            y2 = min(img_shape[1], center + length // 2)
        elif correction_factor < 1:
            # increase x side
            center = x1 + (x2 - x1) // 2
            length = int(round((x2 - x1) * 1 / correction_factor))
            x1 = max(0, center - length // 2)
            x2 = min(img_shape[0], center + length // 2)

        return [x1, y1, x2, y2]


class ImgLoader:
    """load images"""

    def __init__(self, img_root='', ann_dir='./', n_classes=9, batch_size=1):
        self.img_root = img_root
        self.ann_dir = ann_dir
        self.ann_files = self.get_ann_files(ann_dir)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.num_workers = 4

    def test(self):
        test_set = ImageDataSet(image_root=self.img_root, n_classes=self.n_classes,
                                ann_root=self.ann_dir, annotation_files=self.ann_files)
        loader = DataLoader(dataset=test_set, batch_size=self.batch_size,
                            shuffle=True, num_workers=self.num_workers)
        return loader

    @staticmethod
    def get_ann_files(ann_dir):
        ann_files = os.listdir(ann_dir)
        for file in ann_files:
            if file.split('.')[-1] != "json":
                ann_files.remove(file)
        return ann_files
