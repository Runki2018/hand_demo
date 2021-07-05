import numpy as np


def compute_iou(rec1, rec2):
    """

    @param rec1:  (lx, ly, rx, ry) which reflects (left, top, right, bottom)
    @param rec2:  (lx, ly, rx, ry)
    @return: scala value of IoU
    """
    # compute area of each rectangles
    s_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    s_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = s_rec1 + s_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    top_line = max(rec1[1], rec2[1])
    right_line = min(rec1[2], rec2[2])
    bottom_line = min(rec1[3], rec2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def sort_iou(bbox_preds, bbox_gts, labels, iou_thre=0.7):
    """

    @param labels: the categories label of the hand objects
    @param iou_thre: iou threshold judge if the bbox_pred is a object
    @param bbox_gts: numpy (n_box, [lx, ly, rx, ry])
    @param bbox_preds:  numpy (n_box, [lx, ly, rx, ry])
    @return: index of labels
    """

    bbox_preds = np.array(bbox_preds)
    bbox_gts = np.array(bbox_gts)

    match_labels = []
    n_bbox_preds = len(bbox_preds)
    n_bbox_gts = len(bbox_gts)
    IOU = np.zeros((n_bbox_preds, n_bbox_gts), dtype=np.float)

    for i, bbox_pred in enumerate(bbox_preds):
        for j, bbox_gt in enumerate(bbox_gts):
            iou = compute_iou(bbox_pred, bbox_gt)
            IOU[i, j] = iou

    max_indexes = IOU.argmax(axis=1)

    for i, max_index in enumerate(max_indexes):
        max_iou = IOU[i, max_index]
        if max_iou >= iou_thre:
            match_labels.append(labels[max_index])
        else:
            match_labels.append(0)  # the label of the bbox is "0-other" without annotation
    # print(f"{IOU=}")
    # print(f"{max_indexes=}")
    # print(f"{match_labels=}")
    return match_labels


if __name__ == '__main__':
    b_preds = np.array([[1, 1, 3, 3], [2, 4, 5, 5], [5, 2, 9, 8]])
    b_gts = np.array([[6, 2, 9, 8], [2, 4, 5, 5]])
    cls = [1, 5]
    match_cls = sort_iou(bbox_preds=b_preds,
                         bbox_gts=b_gts,
                         labels=cls)
    # print(f"{match_cls=}")
