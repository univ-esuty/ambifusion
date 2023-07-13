from glob import glob

import math
import random
from PIL import Image
import cv2
from omegaconf import OmegaConf
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def load_DAambigram_dataset(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    _DA_conf=None,
):
    """
    For a Image dataset with class_cond, create a generator over (images, kwargs) pairs.
    Directory structure is `datset_root/[class_label]/[file_name]`.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    all_files = []
    for type in ["jpg", "jpeg", "png", "gif"]:
        all_files += glob(f'{data_dir}/*/*.{type}')
    
    classes = {}
    if class_cond:
        class_labels = sorted(glob(f'{data_dir}/*'))
        num = 0
        for label in class_labels:
            classes[label.split('/')[-1]] = num
            num += 1
    
    if _DA_conf is None:
        _DA_conf = OmegaConf.load('configs/DA_ambigram_configs.yaml')
        
    dataset = DAambigramDataset_legacy(
        image_size,
        all_files,
        _DA_conf,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    
    while True:
        yield from loader    


class DAambigramDataset_legacy(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        DA_conf,
        shard=0,
        num_shards=1,
        classes={},
    ):
        super().__init__()
        self.resolution = resolution
        self.DA_conf = DA_conf
        self.local_images = image_paths[shard:][::num_shards]
        self.classese = classes

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        class_label = path.split('/')[-2]
        
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB").resize((self.resolution, self.resolution))
        arr = np.array(pil_image)
        
        ## Ambigram DA
        if random.random() < self.DA_conf.re_scale.ratio:
            arr = self.re_scale(arr, min_scale=self.DA_conf.re_scale.min_scale, max_scale=self.DA_conf.re_scale.max_scale)
            
        if random.random() < self.DA_conf.rotate.ratio:
            arr = self.rotate(arr, self.DA_conf.rotate.max_angle)
            
        if random.random() < self.DA_conf.move.ratio:
            arr = self.move(arr, self.DA_conf.move.scale)
        
        arr = arr.astype(np.float32) / 127.5 - 1
        out_dict = {}
        
        if self.classese != {}:
            out_dict["y"] = np.array(self.classese[class_label], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


    def move(self, arr, scale):
        base = np.full((self.resolution*2, self.resolution*2, 3), fill_value=255, dtype=np.uint8)
        base[self.resolution//2:self.resolution//2*3, self.resolution//2:self.resolution//2*3, :] = arr
        dx = self.resolution * (random.random()*scale*2 - scale)
        dy = self.resolution * (random.random()*scale*2 - scale)
        afin_matrix = np.float32([[1,0,dx],[0,1,dy]])
        base = cv2.warpAffine(base, afin_matrix, (self.resolution*2, self.resolution*2), flags=cv2.INTER_NEAREST)
        arr = base[self.resolution//2:self.resolution//2*3, self.resolution//2:self.resolution//2*3, :]
        return arr
    
    def re_scale(self, arr, min_scale, max_scale):
        base = np.full((self.resolution*2, self.resolution*2, 3), fill_value=255, dtype=np.uint8)
        size = int(self.resolution + self.resolution * (min_scale + (max_scale-min_scale)*random.random()))
        arr_resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST) #cv2.INTER_LINEAR isn't apporopriate for character image.
        top = left = (self.resolution*2 - size) // 2
        base[top:(top+size), left:(left+size), :] = arr_resized
        arr = base[self.resolution//2:self.resolution//2*3, self.resolution//2:self.resolution//2*3, :]
        return arr
    
    def rotate(self, arr, max_angle):
        angle = random.random()*max_angle*2 - max_angle
        M = cv2.getRotationMatrix2D((self.resolution//2, self.resolution//2), angle, 1.)
        arr = cv2.warpAffine(arr, M, dsize=(self.resolution, self.resolution), borderValue=(255, 255, 255), flags=cv2.INTER_LINEAR) #cv2.INTER_LINEAR isn't apporopriate for character image.
        return arr
