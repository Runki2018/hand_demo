import os
import pickle
import random
from collections import OrderedDict
from collections import defaultdict

import cv2
import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from tqdm import tqdm

from misc.nms.nms import oks_nms_21

from misc.utils import affine_transform, get_affine_transform, evaluate_pck_accuracy
from .HumanPoseEstimation import HumanPoseEstimationDataset as Dataset


class ZHHandDataSet(Dataset):
    def __init__(self, data_path='/home/user/ln_home/data2/ZHhands/allocate_dataset/',
                 ann_file='first_subset/revise_totalSamples2.json',
                 is_train=True, image_width=256, image_height=256):
        super(ZHHandDataSet, self).__init__()
        self.data_path = data_path
        self.annotation_path = os.path.join(data_path, ann_file)
        self.is_train = is_train
        self.image_width = image_width
        self.image_height = image_height
        self.color_rgb = True
        self.use_gt_bboxes = True
        self.scale = True
        self.scale_factor = 0.35
        self.flip_prob = 0.5  # todo: need to reimplement the flip function without flip_pairs
        self.rotate_prob = 0.5  # todo: not use in classification, in which it is 0.0
        self.rotate_factor = 45.
        self.use_different_joints_weight = False
        self.heatmap_sigma = 3
        self.soft_nms = False

        self.image_size = (self.image_width, self.image_height)
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.heatmap_size = (int(self.image_width / 4), int(self.image_height / 4))
        self.heatmap_type = 'gaussian'
        self.pixel_std = 200

        self.nof_joints = 21
        # I think the wrist point and the fingertip points of thumb\first finger\little finger
        # play a more important role than other points
        self.joints_weight = np.asarray(
            [1.5, 1., 1., 1.2, 1.5,
             1., 1., 1.2, 1.5,
             1., 1., 1., 1.,
             1., 1., 1., 1.,
             1., 1., 1.2, 1.5], dtype=np.float32
        ).reshape((self.nof_joints, 1))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create COCO object then load images and annotations
        self.coco = COCO(self.annotation_path)
        self.imgIds = self.coco.getImgIds()

        self.data = []
        # load annotations for each image of COCO
        for imgId in tqdm(self.imgIds):
            ann_ids = self.coco.getAnnIds(imgIds=imgId, iscrowd=False)
            img = self.coco.loadImgs(imgId)[0]  # image info, including 'file_name', 'id'...

            objs = self.coco.loadAnns(ann_ids)  # a dict include the key 'bbox', 'keypoints'...
            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                # ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    continue

                x, y, w, h = obj['bbox']  # left_top coordinate x, y and width, height of box
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((img['width'] - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((img['height'] - 1, y1 + np.max((0, h - 1))))

                # Use only valid bounding boxes
                if x2 > x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            # for each annotation of this image, add the formatted annotation to self.data
            for obj in objs:
                joints = np.zeros((self.nof_joints, 2), dtype=np.float)
                joints_visibility = np.ones((self.nof_joints, 2), dtype=np.float)

                for pt in range(self.nof_joints):
                    joints[pt, 0] = obj["keypoints"][pt * 3 + 0]  # x
                    joints[pt, 1] = obj["keypoints"][pt * 3 + 1]  # y
                    t_vis = int(np.clip(obj['keypoints'][pt * 3 + 2], 0, 1))
                    # COCO:
                    # if visibility == 0 -> keypoint is not in the image
                    # if visibility == 1 -> keypoint is in the image but not visible
                    # if visibility == 2 -> keypoint looks clearly
                    # todo: directly set visibility equal 2 because COCO eval metric always is 0
                    # joints_visibility[pt, 0] = t_vis
                    # joints_visibility[pt, 1] = t_vis
                    joints_visibility[pt, 0] = 2
                    joints_visibility[pt, 1] = 2

                center, scale = self._box2cs(obj['clean_bbox'][:4])

                self.data.append({
                    'imgId': imgId,
                    'annId': obj['id'],
                    'label': img['label'],
                    'imgPath': os.path.join(self.data_path, img['file_name']),
                    'center': center,
                    'scale': scale,
                    'joints': joints,
                    'joints_visibility': joints_visibility
                })
        # done check if we need prepare_data -> we should not
        print('\nZHHand dataset loaded!')

        # Default values
        self.bbox_thre = 1.0
        self.image_thre = 0.0
        self.in_vis_thre = 0.2  # if predicted keypoint score > in_vis_thre, it means the point is visible and valid
        self.nms_thrs = 1.0
        self.oks_thre = 0.9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        joints_data = self.data[index].copy()

        image = cv2.imread(joints_data['imgPath'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if self.color_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise ValueError(' Fail to read %s' % image)

        joints = joints_data['joints']  # (nof_joints, 2)
        joints_vis = joints_data['joints_visibility']

        c = joints_data['center']
        s = joints_data['scale']
        score = joints_data['score'] if 'score' in joints_data else 1
        r = 0

        # Apply data augmentation
        if self.is_train:
            # todo: I did not use the half body augmentation and remove the related code

            sf = self.scale_factor  # 0.35
            rf = self.rotate_factor  # 45.  rotate angle

            if self.scale:
                s = s * np.clip(random.random() * sf + 1, 1 - sf, 1 + sf)  # s is between [1-sf, 1+sf]

            if self.rotate_prob and random.random() < self.rotate_prob:
                r = np.clip(random.random() * rf, -rf * 2, rf * 2)  # [-2*rf, 2*rf] = [-90., 90.]
            else:
                r = 0

            # todo: I replace the fliplr_joints() with flip_hand_joints() for hand pose flip
            if self.flip_prob and random.random() < self.flip_prob:
                image = image[:, ::-1, :]  # (h,w,c)  horizontal flip
                # joints, joints_vis = fliplr_joints(joints, joints_vis, image.shape[1], self.flip_pairs)
                joints, joints_vis = self.flip_hand_joints(joints, joints_vis, image.shape[1])
                c[0] = image.shape[1] - c[0] - 1

        # Apply affine transform on joints and image
        trans = get_affine_transform(c, s, self.pixel_std, r, self.image_size)
        image = cv2.warpAffine(
            image, trans,
            (int(self.image_width), int(self.image_height)),
            flags=cv2.INTER_LINEAR
        )

        for i in range(self.nof_joints):
            if joints_vis[i, 0] > 0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        # Convert image to tensor and normalize
        image = self.transform(image)

        target, target_weight = self._generate_target(joints, joints_vis)

        # update metadata
        joints_data['joints'] = joints
        joints_data['joints_visibility'] = joints_vis
        joints_data['center'] = c
        joints_data['scale'] = s
        joints_data['rotation'] = r
        joints_data['score'] = score

        return image, target.astype(np.float32), target_weight.astype(np.float32), joints_data

    def evaluate_accuracy(self, output, target, params=None):
        accs, avg_acc, cnt, joints_preds, joints_target = evaluate_pck_accuracy(output, target)
        return accs, avg_acc, cnt, joints_preds, joints_target

    def evaluate_overall_accuracy(self, predictions, bounding_boxes, image_ids, output_dir, rank=0., label=0):
        """add a new param: label, is the category of the hand pose,
            besides, I also change the 'image_paths' to image_ids.
        """
        res_folder = os.path.join(output_dir, 'result')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder, 0o755, exist_ok=True)
        res_file = os.path.join(res_folder, 'results_{}.json'.format(rank))

        # post process the path in image_paths
        # image_name_wo_suffix = []
        # for path in image_paths:
        #     path = path.split('/')[-1]  # data_path/file.jpg  -> file.jpg
        #     path = path.split['.'][0]  # file.jpg -> file
        #     image_name_wo_suffix.append(path)

        # data format -> person x keypoints
        _kpts = []
        for idx, kpt in enumerate(predictions):
            _kpts.append({
                'keypoints': kpt,
                'center': bounding_boxes[idx][0:2],
                'scale': bounding_boxes[idx][2:4],
                'area': bounding_boxes[idx][4],
                'score': bounding_boxes[idx][5],
                'image': image_ids[idx],  # image_name_wo_suffix[idx]
                'label': label
            })

        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.nof_joints  # 21
        in_vis_thre = self.in_vis_thre  # 0.2
        oks_thre = self.oks_thre  # 0.9
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_dict = kpts[img]
            for n_p in img_dict:  # num of person
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_j in range(0, num_joints):  # num of joints
                    joint_score = n_p['keypoints'][n_j][2]
                    if joint_score > in_vis_thre:
                        kpt_score += joint_score
                        valid_num += 1
                if valid_num != 0:
                    kpt_score /= valid_num

                n_p['score'] = kpt_score * box_score  # rescoring

            # if self.soft_nms:
            #     keep = soft_oks_nms([img_dict[i] for i in range(len(img_dict))], oks_thre)
            # else:
            #     keep = oks_nms([img_dict[i] for i in range(len(img_dict))], oks_thre)
            # just use oks_nms_21 modified by me in the base on oks_nms
            keep = oks_nms_21([img_dict[i] for i in range(len(img_dict))], oks_thre)

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_dict)
            else:
                oks_nmsed_kpts.append([img_dict[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)
        # todo: add not to avoid executing the code of True part
        # if not self.is_train:
        if self.is_train:
            info_str = self._do_python_keypoint_eval(res_file)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    # private methor

    def _generate_target(self, joints, joints_vis):
        """
        @param joints: [nof_joints, 2] = [21, 2]
        @param joints_vis: [nof_joints, 2] = [21, 2]
        @return target: [nof_joints, h_heatmap, w_heatmap]
        @return target_weight: [nof_joints, 2]
        """
        target_weight = np.ones((self.nof_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]  # visible == 1 , invisible == 0

        if self.heatmap_type == 'gaussian':
            target = np.zeros((self.nof_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]), dtype=np.float32)  # [21, h, w]

            tmp_size = self.heatmap_sigma * 3  # 3*3 = 9 ?  why use this variable?

            for joints_id in range(self.nof_joints):
                feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size)  # [4., 4.]
                mu_x = int(joints[joints_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joints_id][1] / feat_stride[1] + 0.5)
                # check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    target_weight[joints_id] = 0
                    continue

                # generate gaussian
                size = 2 * tmp_size + 1  # 19?
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # the gaussian is not normalized, we want the cneter value to equal 1
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

                # usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]  # (a, b)
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joints_id]
                if v > 0.5:
                    target[joints_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise NotImplementedError  # Only implement the gaussian heatmap

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32
        )
        if center[0] != -1:
            scale *= 1.25

        return center, scale

    @staticmethod
    def flip_hand_joints(joints, joints_vis, width):
        joints[:, 0] = width - joints[:, 0] - 1  # horizontal flip
        return joints, joints_vis

    def _write_coco_keypoint_results(self, img_data, res_file):
        results = self._coco_keypoint_results_one_category_kernel(img_data)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, img_data):
        cat_results = []

        for img_kpts in img_data:
            if len(img_kpts) == 0:
                continue
            _key_points = np.array([img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
                                   , dtype=np.float32)
            key_points = np.zeros((_key_points.shape[0], self.nof_joints * 3), dtype=np.float32)
            for ipt in range(self.nof_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]  # x
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]  # y
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # c, score

            result = [{
                'image_id': img_kpt['image'],
                'category_id': img_kpt['label'],
                'keypoints':list(key_points[k]),
                'score': img_kpt['score'].astype(np.float32),
                'center': list(img_kpt['center']),
                'scale': list(img_kpt['scale'])
                } for k, img_kpt in enumerate(img_kpts)
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR(M)', 'AR(L)']
        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        return info_str
