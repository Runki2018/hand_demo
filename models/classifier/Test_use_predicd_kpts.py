from models.classifier.test_model import plot_confusion_matrix, metric_eval
from models.classifier.utils.parse_config import parse_model_cfg
from models.classifier.model import classifier

import torch
import numpy as np


class Classifier:

    def __init__(self, device, conf_thr=0.01):
        #  get_classifier
        cfg_path = "./models/classifier/cfg/network.cfg"
        pt_file = "./models/classifier/weight/92acc_49epoch_9category.pt"
        # pt_file = "runs/2021-04-16/93acc_699epoch_9category.pt"
        net_block = parse_model_cfg(cfg_path)[0]  # [net]
        self.n_classes = net_block["n_classes"]
        # batch_size = net_block["batch"]
        self.device = device
        self.model = classifier(cfg_path).to(device)
        param_dict = torch.load(pt_file)
        self.model.load_state_dict(param_dict["model_state"])

        self.confidence_threshold = conf_thr  # 0.98 置信度阈值，大于该值才预测为真，否则pass

        nc = self.n_classes
        self.confusion_matrix = np.zeros((nc, nc), dtype=int)
        self.confidence_matrix = np.zeros((nc, nc, nc), dtype=float)  # 置信度和
        self.sum_matrix = np.zeros((nc, nc, nc), dtype=int)  # 每个置信度和由多少个样本得来

        self.classes = ['0-other', '1-OK', '2-palm', '3-up', '4-down', '5-right', '6-left', '7-heart', '8-hush']
        if self.n_classes == 8:
            self.classes.pop(0)

        self.model.eval()
        self.count = 0

    def test(self, keypoints: list, label: list, is_eval=False):
        """
        keypoints: (1, 42) -> [x0, y0, x1, y1, x2, y2, ...] or (batch, 42)-> [[x0...], [], [],...]
        label: (1) -> [a1] or (batch) -> [a1, .... ,a_batch]
        """

        with torch.no_grad():
            keypoints = torch.tensor(keypoints, device=self.device, dtype=torch.float)
            # print("keypoints:\n", keypoints)
            # print("size=", keypoints.size())
            y_predict = self.model(keypoints)  # (batch, n_class) 9个类别的置信度
            predict_index = y_predict.argmax(dim=1, keepdim=False)  # tensor([y0, y1, ..., y_batch]), (batch)
            print("y_predict = ", y_predict[:, predict_index])

            values, indexes = y_predict.topk(k=self.n_classes, dim=-1)  # 将置信度及其相应序号，按置信度由大到小排序
            print("values = ", values)
            print("indexes = ", indexes)

            if not is_eval:
                return predict_index, y_predict[:, predict_index]

            if self.n_classes == 8:  # for CrossEntropyLoss
                # [a1, a2, ..., a_batch], (batch)
                label = torch.tensor(label, dtype=torch.long, device=self.device) - 1
            else:
                label = torch.tensor(label, dtype=torch.long, device=self.device)

            i = -1
            for y_true, y_pred in zip(label, predict_index):  # [p1, p2, ..., p_batch]
                i += 1
                if y_predict[i, y_pred] < self.confidence_threshold:
                    continue
                self.confusion_matrix[y_true, y_pred] += 1  # 获取混淆矩阵
            values, indexes = y_predict.topk(k=self.n_classes, dim=-1)  # 将置信度及其相应序号，按置信度由大到小排序
            print("values = ", values)
            print("indexes = ", indexes)
            # todo: error occur when num of predict is not equal gt's
            batch_size = label.size()[0]
            # if keypoints.size()[0] < batch_size:
            #     batch_size = keypoints.size()[0]

            for i in range(batch_size):
                self.count += 1
                for j in range(self.n_classes):
                    self.confidence_matrix[label[i], indexes[i, j], j] += values[i, j]
                    self.sum_matrix[label[i], indexes[i, j], j] += 1

        return predict_index, y_predict[:, predict_index]

    def eval(self):
        print(f'number of images is {self.count}')

        # confidence_matrix.png /= sum_matrix
        with np.errstate(divide='ignore', invalid='ignore'):  # 防止报错
            self.confidence_matrix /= self.sum_matrix
            self.confidence_matrix = np.nan_to_num(self.confidence_matrix)  # 将数组中 nan值设置为0, inf设置为有限的大数

        confidence_matrix = self.confidence_matrix.round(3)  # 保留3个小数位

        for i in range(self.n_classes):
            print("类别：", self.classes[i])
            print(self.confidence_matrix[i])
            print("-" * 50)

        print(confidence_matrix[..., 0])

        # calculate Precision and Recall
        metric_eval(self.confusion_matrix)
        # draw the confusion_matrix
        plot_confusion_matrix(self.confusion_matrix, 'confusion_matrix.png', title='Confusion Matrix')  # 个数
        plot_confusion_matrix(confidence_matrix[..., 0], 'confidence_matrix.png',
                              title='Confidence Matrix')  # 概率
