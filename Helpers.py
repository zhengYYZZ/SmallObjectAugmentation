import glob
import cv2 as cv2
from tqdm import tqdm


def read_images(path):
    images = glob.glob(path)
    return images


def load_images_from_path(path):
    image_list = []
    for p in tqdm(path):
        image = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        image_list.append(image)
    return image_list


def replace_labels(path):
    label_path = []
    for p in path:
        label_path.append(p.replace('.jpg', '.txt'))
    return label_path
