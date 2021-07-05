import cv2
import torch
import numpy as np
from SimpleHRNet import SimpleHRNet
from datasets.ZHhandDataSet import ImgLoader
from scripts.visualization_tools import Visualization
from models.classifier.Test_use_predicd_kpts import Classifier
from scripts.sort_rusult import sort_iou
from copy import deepcopy


def main():
    # img_root = "/home/user/ln_home/data2/ZHhands/JPEGImages/"
    ann_dir = '/home/user/ln_home/data2/ZHhands/allocate_dataset/first_subset/' \
              'annotations_mapping/'
    classifier_path = "./models/classifier/classification_model.pth"
    classes_names = ["0-other", "1-okay", "2-palm", "3-up", "4-down",
                     "5-right", "6-left", "7-finger_heart", "8-hush"]
    classes_colors = [(0, 0, 255), (147, 20, 255), (255, 0, 0),
                      (0, 255, 255), (0, 255, 255), (0, 70, 255),
                      (208, 224, 64), (130, 0, 75), (193, 182, 255)]

    loader = ImgLoader(ann_dir=ann_dir).test()

    # preparing models
    device = torch.device("cuda:0")
    model = SimpleHRNet(c=48,
                        checkpoint_path='./training/logs/train_HRNet_1/checkpoint_last.pth',
                        model_name='HRNet',
                        nof_joints=21,
                        resolution=(256, 256),
                        multiperson=True,
                        return_heatmaps=False,
                        return_bounding_boxes=True,
                        yolo_model_def='./models/detectors/yolo/config/yolov3-1class.cfg',
                        yolo_class_path="models/detectors/yolo/data/1classes.names",
                        yolo_weights_path="./models/detectors/yolo/weights/yolov3_1class.pth",
                        device=device)

    classifier = Classifier(device=device, conf_thr=0.88)  # fc + bn + leakyRelu

    print("models are ready!")
    cv2.namedWindow("image_window")
    ni = 0
    for img, img_file, labels, gt_kpts, bboxes in loader:
        print(img_file)
        # print(img.shape)
        img = img.squeeze()
        img = np.array(img)
        pred = model.predict(img)  # shape = (1,21,3) = (nof_person/images, nof_joints, xyc/yxc?)
        pt_bbox = np.array(pred[0])  # (x1,y1,x2,y2,conf,cls_conf, cls_pred)
        pt_keypoints = np.array(pred[-1])
        # print(pt_keypoints.shape)

        # get the keypoints (batch, 42) for classification
        # print(f'{len(pt_keypoints)=}')
        keypoints = []   # use to classify
        kpts = []
        for n_hand in range(len(pt_keypoints)):
            points_list = pt_keypoints[n_hand].tolist()  # [ [x1,y1,c1], [x2,y2,c2], ...]
            kpts.append([])
            keypoints_without_conf = []
            for point in points_list:
                kpts[-1].extend(point)  # [x,y,c]

            # print(f"{keypoints=}")
            for i in range(21):
                keypoints_without_conf.append(kpts[-1][i * 3 + 0])  # xi
                keypoints_without_conf.append(kpts[-1][i * 3 + 1])  # yi
            keypoints.append(keypoints_without_conf)

        # match object to corresponding label:
        if test_gt_kpts:
            gt_kpts_format = deepcopy(gt_kpts)  # # (n_hand, 42)
            gt_kpts_format = keypoints_mapping(gt_kpts_format, bboxes)  # (n_hand, 42)
            pred_labels = classifier.test(gt_kpts_format, labels)  # (batch, n_class), (batch)
        else:
            bbox_preds = pt_bbox[:, :4].tolist()
            labels = sort_iou(bbox_preds=bbox_preds, bbox_gts=bboxes,
                              labels=labels, iou_thre=0.4)
            keypoints = keypoints_mapping(keypoints, bbox_preds)
            pred_labels, confs = classifier.test(keypoints, labels)  # (batch, n_class), (batch)

        # draw the predict keypoints and bounding boxes on the image:
        if visual_flag:
            ni += 1
            if ni > 40:
                break
            vision = Visualization(img)   # visualize the keypoints, bbox, and text
            if test_gt_kpts:
                for n_hand in range(len(labels)):
                    idx = labels[n_hand]
                    vision.draw_point(gt_kpts[n_hand], strip_conf=True)

                    lx, ly, rx, ry = bboxes[n_hand]
                    vision.draw_bbox(lx, ly, rx, ry, color=classes_colors[idx])

                    text = classes_names[int(idx)]
                    vision.draw_text(text, [lx, ly, rx, ry], color=classes_colors[idx])

            else:
                for n_hand in range(len(pt_keypoints)):
                    pred_idx = pred_labels[n_hand]
                    vision.draw_point(kpts[n_hand])

                    lx, ly, rx, ry, conf, cls_conf, cls_pred = pt_bbox[n_hand].tolist()
                    vision.draw_bbox(lx, ly, rx, ry, color=classes_colors[pred_idx])
                    # print(f" {cls_conf=}, {cls_pred=}, {cls_conf=}")

                    text = classes_names[int(pred_labels[n_hand])]
                    vision.draw_text(text, [lx, ly, rx, ry], color=classes_colors[pred_idx])

            cv2.imshow("image_window", img)
            cv2.waitKey(0)
    if not visual_flag:
        classifier.eval()
    cv2.destroyWindow("image_window")


def keypoints_mapping(keypoints, bbox):
    """ Mapping the keypoints to bbox, and Normalize it to 0~1"""
    kpts = []
    for n_hand in range(len(bbox)):
        lx, ly, rx, ry = bbox[n_hand]
        w, h = rx - lx, ry - ly

        keypoint = keypoints[n_hand]
        kpts.append([])
        for i in range(21):
            x = (keypoint[i*2 + 0] - lx) / w
            y = (keypoint[i*2 + 1] - ly) / h
            kpts[-1].append(x)  # x
            kpts[-1].append(y)  # y
    return kpts


if __name__ == '__main__':
    visual_flag = False
    test_gt_kpts = False
    main()
