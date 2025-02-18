import numpy as np

from advent.utils.serialization import json_load
from tps.dataset.base_dataset import BaseDataset
from pathlib import Path

class CityscapesSeq_src_DataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)
        # map to cityscape's ids
        #self.id_to_trainid = {3: 0, 4: 1, 2: 2, 5: 3, 7: 4, 15: 5, 9: 6, 6: 7, 1: 8, 10: 9, 11: 10, 8: 11,}
        #if mvss_id:
        self.id_to_trainid = {13: 0, 15: 1, 17: 2, 18: 3, 11: 4, 12: 5, 5: 6, 6: 7, 7: 8, 10: 9, 0: 10, 1:11, 8:12, 9:13, 2:14}

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def get_metadata(self, name):
        new_root = '../../../TPSold/data/Cityscapes'
        self.root = Path(new_root)
        img_file = self.root / 'leftImg8bit_sequence' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name_cf = self.files[index]
        label = self.get_labels(label_file)
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = label_copy
        frame_cf = int(name_cf.split('/')[-1].replace('_leftImg8bit.png','')[-6:])

        # current
        name_cf = name_cf
        file_cf = img_file
        d = self.get_image(file_cf)
        image_d = self.get_image(file_cf)
        image_d = self.preprocess(image_d)

        name_kf = name_cf.replace(str(frame_cf).zfill(6) + '_leftImg8bit.png', str(frame_cf - 1).zfill(6) + '_leftImg8bit.png')
        label_file_c = name_kf.replace('leftImg8bit.png', 'gtFine_labelIds.png')
        file_kf = self.root / 'leftImg8bit_sequence' / self.set / name_kf
        label_file_c = self.root / 'gtFine' / self.set / label_file_c
        label_c = self.get_labels(label_file_c)
        label_c_copy = 255 * np.ones(label_c.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_c_copy[label_c == k] = v
        label_c = label_c_copy
        c = self.get_image(file_kf)
        image_c = self.get_image(file_kf)
        image_c = self.preprocess(image_c)

        # previous
        name_kf = name_cf.replace(str(frame_cf).zfill(6) + '_leftImg8bit.png', str(frame_cf - 1 - 1).zfill(6) + '_leftImg8bit.png')
        label_file_b = name_kf.replace('leftImg8bit.png', 'gtFine_labelIds.png')
        file_kf = self.root / 'leftImg8bit_sequence' / self.set / name_kf
        label_file_b = self.root / 'gtFine' / self.set / label_file_b
        label_b = self.get_labels(label_file_b)
        label_b_copy = 255 * np.ones(label_b.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_b_copy[label_b == k] = v
        label_b = label_b_copy
        b = self.get_image(file_kf)
        image_b = self.get_image(file_kf)
        image_b = self.preprocess(image_b)

        return image_d.copy(), label.copy(), image_c.copy(), label_c.copy(), image_b.copy(), label_b.copy(), np.array(image_d.shape), name_cf, d.transpose((2, 0, 1)), c.transpose((2, 0, 1)), b.transpose((2, 0, 1))
