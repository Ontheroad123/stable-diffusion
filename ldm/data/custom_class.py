# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import albumentations
from torch.utils.data import Dataset

from ldm.data.base import ImagePaths


class CustomBase(Dataset):
    def __init__(self, size, txt_file="data/ct/train.txt"):
        super().__init__()
        self.class_labels = []
        self.txt_file = txt_file

        with open(txt_file, "r") as f:
            self.paths = f.read().splitlines()

        for p in self.paths:
            #name = p.split('/')[-2].split('.')[-1]
            name = p.split('/')[-2].split('_')[-1]
            self.class_labels.append(name)
        #print(set(self.class_labels))
        sorted_classes = {x: i for i, x in enumerate(sorted(set(self.class_labels)))}
        #print(sorted_classes)
        self.new_classes = {}
        for key in  sorted_classes.keys():
            self.new_classes[sorted_classes[key]] = key
        classes = [sorted_classes[x] for x in self.class_labels]

        self.labels = {
            "class_label": np.array(classes),
        }
        #print(labels)
        self.data = ImagePaths(paths=self.paths,
                               labels=self.labels,
                               size=size,
                               random_crop=True,
                               random_flip=True,
                               random_rotate=True)
        #print(self.data.__getitem__)
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        example = self.data[i]
        example['human_label'] = self.new_classes[self.labels['class_label'][i]]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file="data/ct/train.txt"):
        super().__init__(size, training_images_list_file)
        self.size = size
        self.txt_file = training_images_list_file


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file="data/ct/test.txt"):
        super().__init__(size, test_images_list_file)

        self.size = size
        self.txt_file = test_images_list_file

# import tqdm
# class_labels = []
# with open("data/ct/train.txt", "r") as f:
#     paths = f.read().splitlines()

#     for p in paths:
#         #name = p.split('/')[-2].split('.')[-1]
#         name = p.split('/')[-2].split('_')[-1]
#         class_labels.append(name)
#     #print(set(self.class_labels))
#     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_labels)))}
#     print(sorted_classes)
#     new_classes = {}
#     for key in  sorted_classes.keys():
#         new_classes[sorted_classes[key]] = key
#     classes = [sorted_classes[x] for x in class_labels]
#     print(new_classes)
#     labels = {
#         "class_label": np.array(classes),
#     }
#     print(labels)
#     data = ImagePaths(paths=paths,
#                         labels=labels,
#                         size=1,
#                         random_crop=True,
#                         random_flip=True,
#                         random_rotate=True)
#     example = data[0]
    
#     print(example)

# datas = CustomBase(4)
# print(datas.__getitem__(0))
