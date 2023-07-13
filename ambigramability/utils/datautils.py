import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch import Tensor
import torchvision.transforms.functional as F

import blobfile as bf
from PIL import Image
from glob import glob
import random
from omegaconf import OmegaConf
import numpy as np


class PartialRandomAffine(transforms.RandomAffine):
    def __init__(self, ratio, degrees, translate=None, scale=None, shear=None, interpolation=F.InterpolationMode.NEAREST, fill=0, fillcolor=None, resample=None, center=None):
        super().__init__(degrees, translate, scale, shear, interpolation, fill, fillcolor, resample, center)
        self.ratio = ratio
        
    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F.get_image_size(img)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        return F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)

class LabeledImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        DA_configs={},
        resolution=64,
        shard=0,
        num_shards=1,
    ):
        super().__init__()      
        if not data_dir:
            raise ValueError("unspecified data directory")
        
        all_files = []
        for type in ["jpg", "jpeg", "png", "gif"]:
            all_files += glob(f'{data_dir}/*/*.{type}')
        self.local_images = all_files[shard:][::num_shards]
        
        self.classes = {}
        class_labels = sorted(glob(f'{data_dir}/*'))
        num = 0
        for label in class_labels:
            self.classes[label.split('/')[-1]] = num
            num += 1
        self.num_classes = num
        
        if DA_configs is None:
            self.DA_configs = OmegaConf.load('configs/DA_ambigram_configs.yaml')
        else:
            self.DA_configs = DA_configs
        
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            PartialRandomAffine(ratio=self.DA_configs.re_scale.ratio, degrees=0, translate=None, scale=(self.DA_configs.re_scale.min_scale, self.DA_configs.re_scale.max_scale), shear=None, interpolation=F.InterpolationMode.BILINEAR, fill=255, center=None),
            PartialRandomAffine(ratio=self.DA_configs.rotate.ratio, degrees=(-self.DA_configs.rotate.max_angle, self.DA_configs.rotate.max_angle), translate=None, scale=None, shear=None, interpolation=F.InterpolationMode.BILINEAR, fill=255, center=None),
            PartialRandomAffine(ratio=self.DA_configs.move.ratio, degrees=0, translate=(self.DA_configs.move.scale, self.DA_configs.move.scale), scale=None, shear=None, interpolation=F.InterpolationMode.NEAREST, fill=255, center=None),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((224, 224)),
        ])

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        class_label = path.split('/')[-2]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        tensor = self.transform(pil_image)
        
        if random.random() < .5:
            in_label = torch.unsqueeze(torch.from_numpy(np.array(self.classes[class_label])), 0).to(torch.int64)
            out_label = torch.nn.functional.one_hot(torch.tensor(0), num_classes=2).float()
        else:
            class_list = list(self.classes.values())
            class_list.pop(self.classes[class_label])
            in_label = torch.unsqueeze(torch.from_numpy(np.array(np.random.choice(class_list))), 0).to(torch.int64)
            out_label = torch.nn.functional.one_hot(torch.tensor(1), num_classes=2).float()
            
        return tensor, in_label, out_label



class PairedImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        DA_configs={},
        resolution=64,
        shard=0,
        num_shards=1,
    ):
        super().__init__()      
        if not data_dir:
            raise ValueError("unspecified data directory")
        
        self.classes = {}
        class_labels = sorted(glob(f'{data_dir}/*'))
        num = 0
        for label in class_labels:
            self.classes[label.split('/')[-1]] = num
            num += 1
        
        all_files = []
        for type in ["jpg", "jpeg", "png", "gif"]:
            all_files += glob(f'{data_dir}/*/*.{type}')
        
        self.local_images = all_files[shard:][::num_shards]        
        self.local_images_per_class = [[] for _ in range(num)]
        for label in class_labels:
            for files in all_files:
                self.local_images_per_class[self.classes[label.split('/')[-1]]].append(files)

        if DA_configs is None:
            self.DA_configs = OmegaConf.load('configs/DA_ambigram_configs.yaml')
        else:
            self.DA_configs = DA_configs
        
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            PartialRandomAffine(ratio=self.DA_configs.re_scale.ratio, degrees=0, translate=None, scale=(self.DA_configs.re_scale.min_scale, self.DA_configs.re_scale.max_scale), shear=None, interpolation=F.InterpolationMode.BILINEAR, fill=255, center=None),
            PartialRandomAffine(ratio=self.DA_configs.rotate.ratio, degrees=(-self.DA_configs.rotate.max_angle, self.DA_configs.rotate.max_angle), translate=None, scale=None, shear=None, interpolation=F.InterpolationMode.BILINEAR, fill=255, center=None),
            PartialRandomAffine(ratio=self.DA_configs.move.ratio, degrees=0, translate=(self.DA_configs.move.scale, self.DA_configs.move.scale), scale=None, shear=None, interpolation=F.InterpolationMode.NEAREST, fill=255, center=None),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Resize((224, 224)),
        ])

    def __len__(self):
        return int(len(self.local_images) * 1) ## any num_paris available.

    def __getitem__(self, idx):
        path = self.local_images[idx]
        class_label = path.split('/')[-2]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")        
        tensor = self.transform(pil_image)
                
        if random.random() < 0.5:
            path_y = 0
        
        if random.random() < 0.5:
            in_label = torch.unsqueeze(torch.from_numpy(np.array(self.classes[class_label])), 0).to(torch.int64)
            out_label = torch.nn.functional.one_hot(torch.tensor(0), num_classes=2).float()
        else:
            class_list = list(self.classes.values())
            class_list.pop(self.classes[class_label])
            in_label = torch.unsqueeze(torch.from_numpy(np.array(np.random.choice(class_list))), 0).to(torch.int64)
            out_label = torch.nn.functional.one_hot(torch.tensor(1), num_classes=2).float()
            
        return tensor, in_label, out_label

