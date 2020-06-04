import Helpers as hp
import os
import util
import cv2
import augment as aug
from tqdm import tqdm


def draw_background_roi(path_list):
    box_list = []
    for back_img_path in path_list:
        img = cv2.imread(back_img_path)
        img_h, img_w, _ = img.shape
        img = cv2.resize(img, (img_w // 2, img_h // 2))
        roi = hp.draw_roi(img)

        roi = roi * 2
        box_list.append([back_img_path, roi])
    return box_list


def set_path(bg='train.txt', fg='small.txt', save_path='save'):
    '''

    :param bg:
    :param fg:
    :param save_path:
    :return: save_base_dir=保存文件夹,img_dir=背景图path,labels_dir=背景图标签,small_img_dir=前景图path
    '''
    base_dir = os.getcwd()
    save_base_dir = os.path.join(base_dir, save_path)
    util.check_dir(save_base_dir)
    img_dir = [f.strip() for f in open(os.path.join(base_dir, bg)).readlines()]  # 读取train.txt文件内容，背景图片
    labels_dir = hp.replace_labels(img_dir)  # 读取背景图片标签
    small_img_dir = [f.strip() for f in open(os.path.join(base_dir, fg)).readlines()]  # 读取small.txt文件内容，前景图片
    return save_base_dir, img_dir, labels_dir, small_img_dir


if __name__ == '__main__':
    save_dir, bg_img_dir, bg_label_dir, fg_img_dir = set_path()
    box = draw_background_roi(bg_img_dir)
    for bb in tqdm(box):
        img, label = aug.synthetic_img(bb[0], bg_label_dir[0], bb[1], fg_img_dir, num=5)
        yolo_txt_name = os.path.join(save_dir, os.path.basename(bb[0].replace('.jpg', '_augment.txt')))
        img_file_name = os.path.join(save_dir, os.path.basename(bb[0].replace('.jpg', '_augment.jpg')))
        cv2.imwrite(img_file_name, img)
        aug.save_label_txt(img.shape, label, yolo_txt_name)
        # print(f'img={img}')
        # img = cv2.resize(img,(512,512))
        # cv2.imshow("imgssss",img)
        # cv2.waitKey(0)
