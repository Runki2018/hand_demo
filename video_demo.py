""" 打开摄像头，显示录像，并输出保存到本地 """
import cv2
import torch
import numpy as np
from SimpleHRNet import SimpleHRNet
from scripts.visualization_tools import Visualization
from models.classifier.Test_use_predicd_kpts import Classifier
from main_pipeline import keypoints_mapping
import time
import argparse


classes_names = ["0-other", "1-okay", "2-palm", "3-up", "4-down",
                 "5-right", "6-left", "7-finger_heart", "8-hush"]
classes_colors = [(0, 0, 255), (147, 20, 255), (255, 0, 0),
                  (0, 255, 255), (0, 255, 255), (0, 70, 255),
                  (208, 224, 64), (130, 0, 75), (193, 182, 255)]


def main(options):
    # preparing models
    if options.cpu:
        device = torch.device("cpu")
    else:
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

    # calling camera and test image captured by camera video
    if options.type == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 创建一个VideoCapture对象，此处插上摄像头，参数设置为0
    else:
        cap = cv2.VideoCapture(options.path)  # 创建一个VideoCapture对象，此处插上摄像头，参数设置为0

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取推荐的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f'fps = {fps}, size = {size}')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID 是一种开源视频编码/解释器，支持avi,mkv,mp4
    out = cv2.VideoWriter('./video/output_video1.avi', fourcc, fps, size)  # 原来fps = 20.0
    cv2.namedWindow("image_window")

    print(" please press 'q' to exit!")
    start = time.time()
    n_frame = 0
    while True:
        ret, frame = cap.read()  # 第一个参数返回一个布尔值（True/False），代表有没有读取到图片；第二个参数表示截取到一帧的图片
        if ret:

            n_frame += 1
            current = time.time()
            fps = int(n_frame/(current - start + 1e-16))

            # predict bbox and keypoints by simpleHRNet
            pred = model.predict(frame)  # shape = (1,21,3) = (nof_person/images, nof_joints, xyc/yxc?)
            pt_bbox = np.array(pred[0])  # (x1,y1,x2,y2,conf,cls_conf, cls_pred)
            # print('pt_bbpx \n', pt_bbox)

            if not pt_bbox.tolist():
                continue  # pt_bbox is empty, no hand was detected

            pt_keypoints = np.array(pred[-1])

            # get the keypoints (batch, 42) for classification
            keypoints = []  # use to classify
            kpts = []
            for n_hand in range(len(pt_keypoints)):
                keypoints_without_conf = []
                # [[x1,y1,c1], [x2,y2,c2],...] -> [x1,y1,c1,x2,y2,c2, ...]
                points_list = pt_keypoints[n_hand].reshape(-1).tolist()
                kpts.append(points_list)

                for i in range(21):
                    keypoints_without_conf.append(kpts[-1][i * 3 + 0])  # xi
                    keypoints_without_conf.append(kpts[-1][i * 3 + 1])  # yi
                keypoints.append(keypoints_without_conf)

            bbox_preds = pt_bbox[:, :4].tolist()
            conf_cc_cls = pt_bbox[:, 4:].tolist()

            labels = [0 for _ in range(len(keypoints))]
            keypoints = keypoints_mapping(keypoints, bbox_preds)
            pred_labels, confs = classifier.test(keypoints, labels)  # (batch, n_class), (batch)

            # draw the predict keypoints and bounding boxes on the image:
            vision = Visualization(frame)  # visualize the keypoints, bbox, and text
            for n_hand in range(len(pt_keypoints)):
                conf = conf_cc_cls[n_hand][0]
                if not conf:
                    continue


                pred_idx = pred_labels[n_hand]
                vision.draw_point(kpts[n_hand])

                lx, ly, rx, ry = bbox_preds[n_hand]

                vision.draw_bbox(lx, ly, rx, ry, color=classes_colors[pred_idx])

                text = classes_names[int(pred_labels[n_hand])]
                vision.draw_text(text, [lx, ly, rx, ry], color=classes_colors[pred_idx])
                text = str(fps) + " fps"
                vision.draw_text(text, [15, 15, 40, 40], color=classes_colors[0])

            cv2.imshow('image_window', frame)
            # out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="输入相应命令来执行摄像头测试或已有视频的测试")
    parser.add_argument('--type', type=int, default=1, help="0是摄像头测试，1是本地视频测试")
    parser.add_argument('--path', type=str, default="./video/hand_video001.avi", help="本地视频的路径")
    parser.add_argument('--cpu', action='store_true', default=False, help="是否用CPU运行，默认为假")
    args = parser.parse_args()
    # print(args.cpu)
    main(args)
