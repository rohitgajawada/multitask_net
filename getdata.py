from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

class SynthDataset(Dataset):
    def __init__(self, data_dir, type, img_size):
        self.image_dir = os.path.join(data_dir, "football_imgs")
        self.anno_path = os.path.join(data_dir, "annotation.txt")
        f = open(self.anno_path, "r")
        self.coords_anno = f.readlines()
        self.image_paths = os.listdir(self.image_dir)
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths) - 1 #to ensure the dataset has n-1 pairs

    def __getitem__(self, idx):

        normtransform = transforms.Compose([
				transforms.Normalize([0.3796, 0.5076, 0.2408], [0.0519, 0.0332, 0.1166])
			])

        img_name_prev = os.path.join(self.image_dir, "track_" + str(idx + 2) + ".png")
        img_prev = io.imread(img_name_prev)
        oldht, oldwid, chans = img_prev.shape

        # scale = [1.0, 1.0]

        img_prev = img_prev.transpose((2, 0, 1))
        img_prev = transform.resize(img_prev, self.img_size)
        img_prev = torch.from_numpy(img_prev / 1.0).contiguous()
        img_prev = normtransform(img_prev)

        img_name = os.path.join(self.image_dir, "track_" + str(idx + 2 + 1) + ".png")
        img = io.imread(img_name)
        img = img.transpose((2, 0, 1))
        img = transform.resize(img, self.img_size)
        img = torch.from_numpy(img / 1.0).contiguous()
        img = normtransform(img)

        players_anno_prev = []
        names_prev = []
        anno_prev = self.coords_anno[idx]
        anno_prev = anno_prev.split(';')
        for i in range(14):
            l = [float(z) for z in anno_prev[i].split(',')[1:]]
            # l = [a[i] * scale[i] for i in range(3)]
            players_anno_prev.append(l)
            names_prev.append(anno_prev[i].split(',')[0])
        players_anno_prev = [j for i in players_anno_prev for j in i]
        players_anno_prev = torch.FloatTensor(players_anno_prev) / 224.0

        players_anno = []
        names = []
        anno = self.coords_anno[idx + 1]
        anno = anno.split(';')
        for i in range(14):
            ll = [float(z) for z in anno[i].split(',')[1:]]
            # ll = [aa[i] * scale[i] for i in range(3)]
            players_anno.append(ll)
            names.append(anno[i].split(',')[0])
        players_anno = [j for i in players_anno for j in i]
        players_anno = torch.FloatTensor(players_anno) / 224.0

        sample = [img_prev.float(), img.float(), players_anno_prev, players_anno]
        return sample

class SynthLoader():
    def __init__(self, opt):
        self.train_data = SynthDataset(opt.data_dir, 'train', opt.img_size)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        # self.val_data = SynthDataset(opt.data_dir, 'test')
        # self.val_loader = torch.utils.data.DataLoader(self.val_gaze,
        # batch_size=opt.testbatchsize, shuffle=True, num_workers=opt.workers)


# a = SynthDataset('./synth_football', 'train', (3, 224, 224))
# for i, data in enumerate(a, 0):
#     b = data[2]
#     print(b)
#     break
