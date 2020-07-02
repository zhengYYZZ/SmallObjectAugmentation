import os
import numpy as np
import random
import Helpers as hp


def convert(size, box):
    """
    坐标转换为yolo需要的格式
    :param size: 图片尺寸
    :param box: 坐标点
    :return:
    """
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def read_label_txt(label_dir):
    """
    读取标签文件
    :param label_dir: 标签文件txt
    :return: label(shape(*,5))
    """
    labels = []
    with open(label_dir) as fp:
        for f in fp.readlines():
            labels.append(f.strip().split(' '))
    return labels


def check_dir(dir):
    """
    创建文件夹
    :param dir: 文件夹名
    :return: None
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def rescale_yolo_labels(labels, img_shape):
    """
    将yolo表示的坐标转换为普通坐标表示
    :param labels: yolo坐标
    :param img_shape: 图像shape
    :return: 普通坐标
    """
    height, width, nchannel = img_shape
    rescale_boxes = []
    for box in list(labels):
        x_c = float(box[1]) * width
        y_c = float(box[2]) * height
        w = float(box[3]) * width
        h = float(box[4]) * height
        x_left = x_c - w * .5
        y_left = y_c - h * .5
        x_right = x_c + w * .5
        y_right = y_c + h * .5
        rescale_boxes.append([box[0], int(x_left), int(y_left), int(x_right), int(y_right)])
    return rescale_boxes


def bbox_iou(box1, box2):
    """
    两个方形区域的相交系数,
    相交面积小-->相交面积大
           0 --> 1
    :param box1:方形区域1
    :param box2:方形区域1
    :return:相交系数
    """
    cl, b1_x1, b1_y1, b1_x2, b1_y2 = box1
    cl, b2_x1, b2_y1, b2_x2, b2_y2 = box2
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width > 0 and inter_height > 0:  # strong condition
        inter_area = inter_width * inter_height
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area)
    else:
        iou = 0
    return iou


def norm_sampling(search_space):
    """
    随机生成坐标点
    :param search_space: 生成点范围
    :return: 坐标[x,y]
    """
    search_x_left, search_y_left, search_x_right, search_y_right = search_space
    # print(search_space)
    # print(search_y_left,search_y_right)
    new_bbox_x_center = random.randint(search_x_left, search_x_right)
    new_bbox_y_center = random.randint(search_y_left, search_y_right)
    return [new_bbox_x_center, new_bbox_y_center]


def roi_box(points):
    # 四边形外接矩形
    x_left, y_left, x_right, y_right = points[0][0], points[0][1], 0, 0
    for p in points:
        if x_left > int(p[0]):
            x_left = p[0]
        if x_right < p[0]:
            x_right = p[0]
        if y_left > p[1]:
            y_left = p[1]
        if y_right < p[1]:
            y_right = p[1]
    return x_left, y_left, x_right, y_right


def random_add_patches(fg_img_shape, bg_labels, bg_img_shape, roi_points, cl=1, paste_number=1, iou_thresh=0):
    """
    随机生成预选框
    :param fg_img_shape:前景图shape
    :param bg_labels: 背景图标签
    :param bg_img_shape: 背景图shape
    :param roi_points: 背景图选框
    :param paste_number: 合成数目
    :param iou_thresh: 两个方框相交面积参数
    :return:
    """
    fg_img_h, fg_img_w, fg_img_c = fg_img_shape
    bg_img_h, bg_img_w, bg_img_c = bg_img_shape
    roi_rect = roi_box(roi_points.reshape(-1, 2).tolist())
    success_num = 0
    count = 0
    new_bboxes = []

    while success_num < paste_number:
        if count == 10:
            break

        new_bbox_x_center, new_bbox_y_center = norm_sampling(roi_rect)
        # 越界检查
        if new_bbox_x_center - 0.5 * fg_img_w < 0 or new_bbox_x_center + 0.5 * fg_img_w > bg_img_w:
            continue
        if new_bbox_y_center - 0.5 * fg_img_h < 0 or new_bbox_y_center + 0.5 * fg_img_h > bg_img_h:
            continue
        if hp.point_in_roi(roi_points, (new_bbox_x_center, new_bbox_y_center)) != 1:
            continue

        new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = new_bbox_x_center - 0.5 * fg_img_w, \
                                                                               new_bbox_y_center - 0.5 * fg_img_h, \
                                                                               new_bbox_x_center + 0.5 * fg_img_w, \
                                                                               new_bbox_y_center + 0.5 * fg_img_h
        new_bbox = [cl, int(new_bbox_x_left), int(new_bbox_y_left), int(new_bbox_x_right), int(new_bbox_y_right)]

        ious_bg = [bbox_iou(new_bbox,bbox_bg) for bbox_bg in bg_labels]
        ious_fg = [bbox_iou(new_bbox,bbox_fg) for bbox_fg in new_bboxes]
        if ious_fg == []:
            ious_fg.append(0)
        if ious_bg == []:
            ious_bg.append(0)
        if max(ious_bg) <= iou_thresh and max(ious_fg) <= iou_thresh:
            success_num +=1
            new_bboxes.append(new_bbox)
        else:
            count += 1
            continue

    return new_bboxes


if __name__ == "__main__":
    test = [[532, 306], [1432, 292], [836, 868], [204, 664]]
    print(roi_box(test) == (204, 868, 1432, 292))
