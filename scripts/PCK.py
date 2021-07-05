import numpy as np


def compute_pck_pckh(dt_kpts, gt_kpts, refer_kpts):
    """
    pck指标计算: 其中尺度因子，决定了计算pck还是pckh
     pck指标：躯干直径，左肩点－右臀点的欧式距离；
     pckh指标：头部长度，头部rect的对角线欧式距离；　

    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。　　
    :return: 相关指标
    """

    dt = np.array(dt_kpts)
    gt = np.array(gt_kpts)
    assert (len(refer_kpts) == 2)
    assert (dt.shape[0] == gt.shape[0])
    ranges = np.arange(0.0, 0.1, 0.01)
    kpts_num = gt.shape[2]
    ped_num = gt.shape[0]

    # compute dist
    scale = np.sqrt(np.sum(np.square(gt[:, :, refer_kpts[0]] - gt[:, :, refer_kpts[1]]), 1))
    dist = np.sqrt(np.sum(np.square(dt - gt), 1)) / np.tile(scale, (gt.shape[2], 1)).T

    # compute pck
    pck = np.zeros([ranges.shape[0], gt.shape[2] + 1])
    for idh, trh in enumerate(list(ranges)):
        for kpt_idx in range(kpts_num):
            pck[idh, kpt_idx] = 100 * np.mean(dist[:, kpt_idx] <= trh)
        # compute average pck
        pck[idh, -1] = 100 * np.mean(dist <= trh)
    return pck
