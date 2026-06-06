import os

import cv2
import numpy as np
import torch
import torch.utils.data as data


class ChangeDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None):
        super().__init__()
        self._split_name = split_name
        self._A_format = setting['A_format']
        self._B_format = setting['B_format']
        self._gt_format = setting['gt_format']
        self._root_path = setting['root']
        self.class_names = setting['class_names']
        self._A_dir = setting.get('A_dir', 'A')
        self._B_dir = setting.get('B_dir', 'B')
        self._gt_dir = setting.get('gt_dir', 'gt')
        self._B_grayscale = setting.get('B_grayscale', False)
        self._legacy_bgr_input = bool(setting.get('legacy_bgr_input', False))
        self._file_names = self._get_file_names(split_name)
        self.preprocess = preprocess

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, index):
        item_name = self._file_names[index]
        A_path = os.path.join(self._root_path, self._A_dir, item_name + self._A_format)
        B_path = os.path.join(self._root_path, self._B_dir, item_name + self._B_format)
        gt_path = os.path.join(self._root_path, self._gt_dir, item_name + self._gt_format)

        A = self._open_image(A_path, cv2.IMREAD_COLOR)
        if not self._legacy_bgr_input:
            A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)

        if self._B_grayscale:
            B = self._open_image(B_path, cv2.IMREAD_GRAYSCALE)
            B = B[:, :, np.newaxis]
        else:
            B = self._open_image(B_path, cv2.IMREAD_COLOR)
            if not self._legacy_bgr_input:
                B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)

        has_gt = os.path.isfile(gt_path)
        if has_gt:
            gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        elif self._split_name == 'test':
            gt = np.full(A.shape[:2], 255, dtype=np.uint8)
        else:
            raise FileNotFoundError(gt_path)

        if self.preprocess is not None:
            if hasattr(self.preprocess, 'set_sample_id'):
                self.preprocess.set_sample_id(item_name)
            A, B, gt = self.preprocess(A, B, gt)

        if self._split_name == 'train':
            A = torch.from_numpy(np.ascontiguousarray(A)).float()
            B = torch.from_numpy(np.ascontiguousarray(B)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()

        output_dict = dict(A=A, B=B, gt=gt, fn=str(item_name), n=len(self._file_names))
        if not has_gt:
            output_dict['has_gt'] = False
        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        source = os.path.join(self._root_path, split_name + '.txt')

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            if not file_name:
                continue
            file_name = os.path.basename(file_name)
            file_name = os.path.splitext(file_name)[0]
            file_name = self._normalize_sample_stem(file_name)
            file_names.append(file_name)
        return file_names

    def _normalize_sample_stem(self, file_name):
        for modal_format in (self._A_format, self._B_format):
            if not modal_format or modal_format.startswith('.'):
                continue
            modal_stem = os.path.splitext(modal_format)[0]
            if modal_stem and file_name.endswith(modal_stem):
                file_name = file_name[:-len(modal_stem)]
        return file_name

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = cv2.imread(filepath, mode)
        if img is None:
            raise FileNotFoundError(filepath)
        return np.array(img, dtype=dtype)

    @staticmethod
    def _gt_transform(gt):
        return gt - 1

    @classmethod
    def get_class_colors(cls):
        def uint82bin(n, count=8):
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        n_classes = 41
        cmap = np.zeros((n_classes, 3), dtype=np.uint8)
        for i in range(n_classes):
            r, g, b = 0, 0, 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap.tolist()
