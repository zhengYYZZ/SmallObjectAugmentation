import cv2 as cv2
import numpy as np
import random


def replace_labels(path):
    """
    加载label.txt
    更改后缀,eg:test.jpg->test.txt
    :param path:jpg文件列表
    :return:txt文件列表
    """
    label_path = []
    for p in path:
        label_path.append(p.replace('.jpg', '.txt'))
    return label_path


def draw_roi(img):
    """
    鼠标画出感兴趣区域
    :param img: 图像
    :return: 点坐标集shape(*,1,2)
    """
    temp = [[]]

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        '''鼠标回调,获取点击坐标'''
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            temp[0].append([x, y])
            # print(a)
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("imageROI", img)

    cv2.namedWindow("imageROI")
    cv2.setMouseCallback("imageROI", on_EVENT_LBUTTONDOWN)
    cv2.imshow("imageROI", img)
    cv2.waitKey(0)
    roi_point = np.array(temp)
    cv2.polylines(img, roi_point, 1, 255)
    cv2.imshow("imageROI", img)
    cv2.waitKey(0)
    return roi_point


def point_in_roi(roi_points, point):
    """
    判断点是否在区域ROI中
    :param roi_point: ROI顶点坐标集shape(*,*,2)
    :param point: 点 (x,y)
    :return: 在区域内返回 1,不在区域中返回 -1,刚好落在边上返回 0
    """
    temp = cv2.pointPolygonTest(roi_points, point, False)
    return temp


def rand_list(list_dir):
    """
    随机返回链表中的一个元素
    :param list_dir:
    :return:
    """
    list_len = len(list_dir) - 1
    ran = random.randint(0, list_len)
    return list_dir[ran]


def img_resize(image, area_max=2000, area_min=1000):
    """
     改变图片面积大小
    :param image: 图片
    :param area_max: 最大面积
    :param area_min: 最小面积
    :return:
    """

    height, width, channels = image.shape

    while (height * width) > area_max:
        image = cv2.resize(image, (int(width * 0.9), int(height * 0.9)))
        height, width, channels = image.shape
        height, width = int(height * 0.9), int(width * 0.9)

    while (height * width) < area_min:
        image = cv2.resize(image, (int(width * 1.1), int(height * 1.1)))
        height, width, channels = image.shape
        height, width = int(height * 1.1), int(width * 1.1)

    return image


def gaussian_blurImg(image):
    # 高斯模糊
    ran = random.randint(0, 9)
    if ran % 2 == 1:
        image = cv2.GaussianBlur(image, ksize=(ran, ran), sigmaX=0, sigmaY=0)
    else:
        pass
    return image


def r(val):
    """
    生成0~val范围的随机数
    :param val:
    :return:
    """
    return int(np.random.random() * val)
