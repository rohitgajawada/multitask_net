from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset
from torchvision import transforms

class SynthDataset(Dataset):
    def __init__(self, data_dir, type):
        self.image_dir = os.path.join(data_dir, "football_imgs")
        self.anno_path = os.path.join(data_dir, "annotations.txt")
        f = open(self.anno_path, "r")
        self.coords_anno = f.readlines()
        self.image_paths = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, "track_" + str(idx + 2) + ".png")
        img = io.imread(img_name)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img / 255.0).contiguous()
        normtransform = transforms.Compose([
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

        img = normtransform(img)
        players_anno = []
        names = []
        anno = self.coords_anno[idx]
        anno = anno.split(';')
        for i in range(14):
            players_anno.append(anno[i].split(',')[1:])
            names.append(anno[i].split(',')[0])
        sample = {'image': img.float(), 'anno': players_anno, 'names': names}
        return sample

class GazeFollow():
    def __init__(self, opt):
        self.train_data = SynthDataset(opt.data_dir, 'train')
        self.train_loader = torch.utils.data.DataLoader(self.train_gaze, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        # self.val_data = SynthDataset(opt.data_dir, 'test')
        # self.val_loader = torch.utils.data.DataLoader(self.val_gaze,
        # batch_size=opt.testbatchsize, shuffle=True, num_workers=opt.workers)


a = SynthDataset('./synth_football', 'train')
sample = a[0]
print(sample['anno'])
