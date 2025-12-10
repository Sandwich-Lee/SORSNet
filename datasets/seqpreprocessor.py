from __future__ import absolute_import
import os.path as osp
import random

import torch
import numpy as np
from PIL import Image


class SeqTrainPreprocessor(object):
    def __init__(self, seqset, dataset, seq_len, sample='evenly', transform=None):
        super(SeqTrainPreprocessor, self).__init__()
        self.seqset = seqset
        self.sample = sample
        # self.identities = dataset.identities
        self.transform = transform
        self.seq_len = seq_len
        # self.root = [dataset.images_dir]
        # self.root.append(dataset.other_dir)

    def __len__(self):
        return len(self.seqset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        img_paths, pid, camid, trackid = self.seqset[index]
        num = len(img_paths)
        if self.sample != "intelligent":  # random consecutive clip
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices1 = frame_indices[begin_index:end_index]
            indices = []
            for index in indices1:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
        else:  # k clip，one frame/clip
            # frame_indices = range(num)
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    indices.append(random.randint(min(i * each, num - 1), min((i + 1) * each - 1, num - 1)))
                else:
                    indices.append(random.randint(min(i * each, num - 1), num - 1))
            # print(len(indices), indices, num )
        imgs = []
        # labels = []
        targt_cam = []

        for index in indices:
            index = int(index)
            img_path = img_paths[index]

            # img = read_image(img_path)
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                print("error: ", img_path)
            imgs.append(img)
            targt_cam.append(camid)
        while len(imgs) < self.seq_len:
            imgs.append(img)
        # print(imgs[0][3][0])
        seq = [imgs]
        if self.transform is not None:
            seq = self.transform(seq)
        imgs = torch.stack(seq[0], dim=0)
        return imgs, None, pid, camid


class SeqTestPreprocessor(object):
    def __init__(self, seqset, dataset, seq_len, transform=None):
        super(SeqTestPreprocessor, self).__init__()
        self.seqset = seqset
        self.identities = dataset.identities
        self.transform = transform
        self.seq_len = seq_len
        self.root = [dataset.images_dir]
        self.root.append(dataset.other_dir)

    def __len__(self):
        return len(self.seqset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):

        start_ind, end_ind, pid, label, camid = self.seqset[index]

        imgseq = []
        # flowseq = []
        for ind in range(start_ind, end_ind):
            fname = self.identities[pid][camid][ind]
            fpath_img = osp.join(self.root[0], fname)
            imgrgb = Image.open(fpath_img).convert('RGB')
            imgseq.append(imgrgb)

        while len(imgseq) < self.seq_len:
            imgseq.append(imgrgb)
        seq = [imgseq]

        if self.transform is not None:
            seq = self.transform(seq)

        img_tensor = torch.stack(seq[0], 0)

        return img_tensor, None, pid, camid
