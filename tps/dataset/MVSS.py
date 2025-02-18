import numpy as np
import torch

from advent.utils.serialization import json_load
from tps.dataset.base_dataset import BaseDataset
from pathlib import Path

class MVSSDataSet(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 set='val',
                 max_iters=None,
                 crop_size=(321, 321),
                 mean=(128, 128, 128),
                 load_labels=True,
                 info_path='',
                 labels_size=None,
                 interval=1):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)
        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=str)
        self.mapping = np.array(self.info['label2train'], dtype=int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        self.interval = interval
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / name
        label_name = name.replace("visible", "label")
        label_file = self.root / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name_cf = self.files[index]
        label_file_parts_list = list(label_file.parts)

        if 'v.jpg' in label_file_parts_list[-1]:
            label_file_parts_list[-1] = label_file_parts_list[-1].replace('v.jpg', 'l.png')

        if 'v.png' in label_file_parts_list[-1]:
            label_file_parts_list[-1] = label_file_parts_list[-1].replace('v.png', 'l.png')
        label_file_parts_tuple = tuple(label_file_parts_list)
        label_file = Path(*label_file_parts_tuple)
        label = self.get_labels(label_file)
        label = self.map_labels(label).copy()

        folder, _, image_name = name_cf.split('/')

        I_flag = 0
        img_flag = 0
        if image_name[0] == 'I':
            I_flag = 1
            image_number = image_name.split('I')[1]
            image_number, prefix = image_number.split('v')
        elif 'img_' in image_name:
            img_flag = 1
            image_number = image_name.split('img_')[1]
            image_number, prefix = image_number.split('v')
        else:
            image_number, prefix = image_name.split('v')

        prev_image_name1 = f'{int(image_number) - self.interval - 1:0{len(image_number)}}'
        prev_image_name2 = f'{int(image_number) - 1:0{len(image_number)}}'
        next_image_name = f'{int(image_number) - self.interval:0{len(image_number)}}'

        if I_flag == 1:
            prev_image_name1 = 'I' + prev_image_name1
            prev_image_name2 = 'I' + prev_image_name2
            next_image_name = 'I' + next_image_name

        if img_flag == 1:
            prev_image_name1 = 'img_' + prev_image_name1
            prev_image_name2 = 'img_' + prev_image_name2
            next_image_name = 'img_' + next_image_name

        prev_image_name1 = prev_image_name1 + 'v' + prefix
        prev_image_name2 = prev_image_name2 + 'v' + prefix
        next_image_name = next_image_name + 'v' + prefix

        # current
        name_cf = name_cf
        file_cf = img_file
        d = self.get_image(file_cf)
        image_d = self.get_image(file_cf)
        image_d = self.preprocess(image_d)

        file_kf = self.root / folder / 'visible' / prev_image_name2
        c = self.get_image(file_kf)
        image_c = self.get_image(file_kf)
        image_c = self.preprocess(image_c)

        # previous
        file_kf = file_kf = self.root / folder / 'visible' / next_image_name
        b = self.get_image(file_kf)
        image_b = self.get_image(file_kf)
        image_b = self.preprocess(image_b)

        file_kf = file_kf = self.root / folder / 'visible' / prev_image_name1
        a = self.get_image(file_kf)
        image_a = self.get_image(file_kf)
        image_a = self.preprocess(image_a)

        frames = [folder+'/'+image_name, folder+'/'+prev_image_name2, folder+'/'+prev_image_name2, folder+'/'+prev_image_name1]

        if self.set == 'train':
            return image_d.copy(), image_c.copy(), image_b.copy(), image_a.copy(), d.transpose(2, 0, 1), label, name_cf, frames
        else:
            return image_d.copy(), label, image_c.copy(), image_b.copy(), image_a.copy(), name_cf, prev_image_name2, folder
