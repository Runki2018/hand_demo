import cv2 as cv
import os
import json
import numpy as np

class Visualization:
    """this clss can draw the bounding boxex, keypoints and text on a specific image read
        by cv2
    """
    def __init__(self, img):
        self.img = img  # img = cv2.imread("image path")

    def draw_bbox(self, lx, ly, rx, ry, color=(0, 0, 255)):
        """
            画出边界框的位置

        :param img: cv读入的图片
        :param lx: 左上角点的 x 坐标值
        :param ly: 左上角点的 y 坐标值
        :param rx: 右下角点的 x 坐标值
        :param ry: 右下角点的 y 坐标值
        :param color: BGR red default color
        """
        leftTop = (lx, ly)  # 左上角的点坐标 (x,y)
        rightBottom = (rx, ry)  # 右下角的点坐标= (x+w,y+h)
        point_color = color  # BGR
        thickness = 1
        lineType = 8
        cv.rectangle(self.img, leftTop, rightBottom, point_color, thickness, lineType)

    def draw_point(self, keypoints, strip_conf=False):
        """

        @param strip_conf: if False then [x1,y1,c1,...,x42,y42,c42] -> [x1,y1,...,x42,y42]
        @param img: the image read by cv2
        @param keypoints: the keypoints is a list, [x0,y0,c0, x1, y1, c1, ...]
        @return: nothing
        """
        points_list = []
        if not strip_conf:
            for i in range(0, len(keypoints), 3):
                point = int(keypoints[i]), int(keypoints[i + 1])  # (xi, yi)
                points_list.append(point)
        else:  # already strip confidence
            for i in range(21):
                point = int(keypoints[i*2 + 0]), int(keypoints[i*2 + 1])  # (xi, yi)
                points_list.append(point)

        color_list = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 140, 0), (0, 0, 255), (0, 255, 0)]  # RGB
        for i, (r, g, b) in enumerate(color_list):
            color_list[i] = (b, g, r)  # BGR
        point_size = 3
        thickness = 4  # 可以为 0 、4、8
        for i in range(21):
            if i == 0:
                cv.circle(self.img, points_list[i], point_size, color_list[i], thickness)
            else:
                index = (i - 1) // 4 + 1
                cv.circle(self.img, points_list[i], point_size, color_list[index], thickness)

    def draw_text(self, text, bbox, color=(0, 0, 255)):
        """

        @param text: the text you draw beside the bbox
        @param bbox: the coordinate list [x1,y1,x2,y2], which is the LT/RB point of bbox
        @param color: text color
        @return: nothing
        """
        # h, w, _ = self.img.shape
        lx, ly, rx, ry = bbox
        text_height = 25  # estimate by drawing text on a image
        y = ry + text_height if ly - text_height < 0 else ly
        xy = lx, y

        cv.putText(img=self.img, text=text,
                   org=xy,
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1,
                   color=color,
                   thickness=2)
