import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL
from PIL import Image as im
import numpy as np
import pandas as pd
import nibabel as nib
from utils import *
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import collections
import matplotlib.pyplot as plt

import json
import cv2
import imageio
from IPython.display import Image
import torchio as tio
from random import random

"""Hyperparameters"""
batch_size = 1

"""Taken from official keras implementation: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9-L37"""


def to_categorical(y, num_classes):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype='int')
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class BRATS_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        folder_name = self.annotations.iloc[index, 4]
        folder_path = os.path.join(self.root_dir, self.annotations.iloc[index, 4])
        flair_path = str(folder_path) + "/" + str(folder_name) + "_flair.nii.gz"
        seg_path = str(folder_path) + "/" + str(folder_name) + "_seg.nii.gz"
        t1_path = str(folder_path) + "/" + str(folder_name) + "_t1.nii.gz"
        t1ce_path = str(folder_path) + "/" + str(folder_name) + "_t1ce.nii.gz"
        t2_path = str(folder_path) + "/" + str(folder_name) + "_t2.nii.gz"

        flair_image = np.array(nib.load(flair_path).get_fdata())
        seg_image = np.array(nib.load(seg_path).get_fdata())
        t1_image = np.array(nib.load(t1_path).get_fdata())
        t1ce_image = np.array(nib.load(t1ce_path).get_fdata())
        t2_image = np.array(nib.load(t2_path).get_fdata())

        seg_image[seg_image == 4] = 3

        """Normalize image"""

        flair_image = cv2.normalize(flair_image[:, :, :], None, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        #         seg_image = cv2.normalize(seg_image[:, :, :], None, alpha=0, beta=255,
        #                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        t1_image = cv2.normalize(t1_image[:, :, :], None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        t1ce_image = cv2.normalize(t1ce_image[:, :, :], None, alpha=0, beta=255,
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        t2_image = cv2.normalize(t2_image[:, :, :], None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

#         print("Before "+str(flair_image.shape))
        if self.transform:
            flair_image = self.transform(flair_image)
            seg_image = self.transform(seg_image)
            t1_image = self.transform(t1_image)
            t1ce_image = self.transform(t1ce_image)
            t2_image = self.transform(t2_image)
            
#         print("After "+str(flair_image.shape))
        flair_image = flair_image.permute(1,2,0)
        seg_image = seg_image.permute(1,2,0)
        t1_image = t1_image.permute(1,2,0)
        t1ce_image = t1ce_image.permute(1,2,0)
        t2_image = t2_image.permute(1,2,0)
        
#         print("After "+str(flair_image.shape))
        

        final_image = np.zeros((4, flair_image.shape[0], flair_image.shape[1], flair_image.shape[2]))
        final_image[0] = flair_image
        final_image[1] = t1_image
        final_image[2] = t1ce_image
        final_image[3] = t2_image
        
        random_value = random()
        
        if(random_value>0.5):
            final_image = tio.RandomAffine(
                scales=(0.9, 1.2),
                degrees=(-20, 20),
                translation=(0.1, 0.1),
                isotropic=True,
            )(final_image)

        label_cat = to_categorical(seg_image, num_classes=4).astype(np.uint8)
        final_seg = np.zeros((label_cat.shape[3], label_cat.shape[0], label_cat.shape[1], label_cat.shape[2]))

        final_seg[0] = label_cat[:, :, :, 0]
        final_seg[1] = label_cat[:, :, :, 1]
        final_seg[2] = label_cat[:, :, :, 2]
        final_seg[3] = label_cat[:, :, :, 3]


        return (final_image, final_seg)



