from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class SynthDataset(Dataset):
    def __init__(self, data_dir, type, img_size):
        self.image_dir = os.path.join(data_dir, "football_imgs")
        self.anno_path = os.path.join(data_dir, "annotations.txt")
        f = open(self.anno_path, "r")
        self.coords_anno = f.readlines()
        self.image_paths = os.listdir(self.image_dir)
        self.img_size = img_size
        self.rescaler = Rescale(img_size)

    def __len__(self):
        return len(self.image_paths) - 1 #to ensure the dataset has n-1 pairs

    def __getitem__(self, idx):

        normtransform = transforms.Compose([
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

        img_name_prev = os.path.join(self.image_dir, "track_" + str(idx + 2) + ".png")
        img_prev = io.imread(img_name_prev)
        img_prev = img_prev.transpose((2, 0, 1))
        img_prev = transform.resize(img_prev, self.img_size)
        img_prev = torch.from_numpy(img_prev / 255.0).contiguous()
        img_prev = normtransform(img_prev)

        img_name = os.path.join(self.image_dir, "track_" + str(idx + 2 + 1) + ".png")
        img = io.imread(img_name)
        img = img.transpose((2, 0, 1))
        img = transform.resize(img, self.img_size)
        img = torch.from_numpy(img / 255.0).contiguous()
        img = normtransform(img)

        players_anno_prev = []
        names_prev = []
        anno_prev = self.coords_anno[idx]
        anno_prev = anno_prev.split(';')
        for i in range(14):
            players_anno_prev.append(anno_prev[i].split(',')[1:])
            names_prev.append(anno_prev[i].split(',')[0])
        players_anno_prev = [j for i in players_anno_prev for j in i]
        players_anno_prev = torch.FloatTensor([float(i) for i in players_anno_prev])

        players_anno = []
        names = []
        anno = self.coords_anno[idx + 1]
        anno = anno.split(';')
        for i in range(14):
            players_anno.append(anno[i].split(',')[1:])
            names.append(anno[i].split(',')[0])
        players_anno = [j for i in players_anno for j in i]
        players_anno = torch.FloatTensor([float(i) for i in players_anno])

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
#     b = data['anno_prev']
#     break
