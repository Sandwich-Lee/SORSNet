from collections import defaultdict
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, batch_size, num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size  # 16
        self.num_instances = num_instances  # 4
        self.num_pids_per_batch = self.batch_size // self.num_instances  # 4
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)  # 625

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)  # 每个id对应的track数量
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances  # 7532

    def __iter__(self):

        batch_idxs_dict = defaultdict(list)

        for pid in self.pids: # 所有训练的id
            idxs = copy.deepcopy(self.index_dic[pid])  # 第pid个id有idxs个track
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)  # 每个id选择4个序列
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:  # 每个id是否取了self.num_instances个track了
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:  # 选择4个ID
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySamplerForSeq(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, batch_size, num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size  # 16
        self.num_instances = num_instances  # K=4
        self.num_pids_per_batch = self.batch_size // self.num_instances  # N=batch_size//K
        self.index_dic = defaultdict(list)

        # for index, (a, b, pid, c, d) in enumerate(data_source):  # index：每个track的下标索引
        #     self.index_dic[pid].append(index)  # 训练集所有数据，相同id的不同track
        # self.pids = list(self.index_dic.keys())  # 把id变成列表

        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.num_identities = len(self.pids)  # 625

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:  # 625个pid
            idxs = self.index_dic[pid]  # 一个id的track索引们
            num = len(idxs)  # 每个id对应有几个track
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):

        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:  # 625个pid
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:  # 想要self.num_instances 个track，实际不够
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)  # 随机选取track重复
            random.shuffle(idxs)  # 将同一个id的track们乱序
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:  # tracks够了（4）
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)  # 625
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)  # 随机选取4个id
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

# class RandomIdentitySamplerWYQ(Sampler):
#     """
#     Randomly sample N identities, then for each identity,
#     randomly sample K instances, therefore batch size is N*K.
#
#     Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.
#
#     Args:
#         data_source (Dataset): dataset to sample from.
#         num_instances (int): number of instances per identity.
#     """
#     def __init__(self, data_source, num_instances=4):
#         super(RandomIdentitySampler).__init__()
#         self.data_source = data_source
#         self.num_instances = num_instances
#         self.index_dic = defaultdict(list)
#         for index, (_, pid, _, _) in enumerate(data_source):
#             self.index_dic[pid].append(index)
#         self.pids = list(self.index_dic.keys())
#         self.num_identities = len(self.pids)
#
#     def __iter__(self):
#         indices = torch.randperm(self.num_identities)
#         ret = []
#         for i in indices:
#             pid = self.pids[i]
#             t = self.index_dic[pid]
#             replace = False if len(t) >= self.num_instances else True
#             t = np.random.choice(t, size=self.num_instances, replace=replace)
#             ret.extend(t)
#         # print(ret)
#         return iter(ret)
#
#     def __len__(self):
#         return self.num_identities * self.num_instances