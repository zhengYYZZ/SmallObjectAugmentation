import aug as am
import Helpers as hp
from util import *
import os
from os.path import join
from tqdm import tqdm
import random

base_dir = os.getcwd()
save_base_dir = join(base_dir, 'save')
check_dir(save_base_dir)

img_dir = [f.strip() for f in open(join(base_dir, 'train.txt')).readlines()]
labels_dir = hp.replace_labels(img_dir)
small_img_dir = [f.strip() for f in open(join(base_dir, 'small.txt')).readlines()]
random.shuffle(small_img_dir)

for image_dir, label_dir in tqdm(zip(img_dir, labels_dir)):
    small_img = []
    for x in range(8):
        if not small_img_dir:
            small_img_dir = [f.strip() for f in open(join(base_dir, 'small.txt')).readlines()]
            random.shuffle(small_img_dir)
        small_img.append(small_img_dir.pop())
    am.copysmallobjects(image_dir, label_dir, save_base_dir, small_img)
print(f'The picture is saved in {save_base_dir}')