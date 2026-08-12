import numpy as np
import torch
import math
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import v2 as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F 
from torchvision.tv_tensors import Mask

class PadToDivisible:
    def __init__(self, divisor=8, image_fill=0, mask_fill=255):
        self.divisor = divisor
        self.image_fill = image_fill
        self.mask_fill = mask_fill

    def __call__(self, img, mask):
        h, w = img.shape[-2:]

        new_h = ((h + self.divisor - 1) // self.divisor) * self.divisor
        new_w = ((w + self.divisor - 1) // self.divisor) * self.divisor

        pad_h = new_h - h
        pad_w = new_w - w

        if pad_h == 0 and pad_w == 0:
            return img, mask

        img = F.pad(
            img,
            padding=[0, 0, pad_w, pad_h],
            fill=self.image_fill,
        )

        mask = F.pad(
            mask,
            padding=[0, 0, pad_w, pad_h],
            fill=self.mask_fill,
        )

        return img, mask
    
class PadToPSPNetSize:
    def __init__(self, divisor=8, image_fill=0, mask_fill=255):
        self.divisor = divisor
        self.image_fill = image_fill
        self.mask_fill = mask_fill

    def _next_valid_size(self, size):
        if (size - 1) % self.divisor == 0:
            return size
        return ((size - 1 + self.divisor - 1) // self.divisor) * self.divisor + 1

    def __call__(self, img, mask):
        h, w = img.shape[-2:]

        new_h = self._next_valid_size(h)
        new_w = self._next_valid_size(w)

        pad_h = new_h - h
        pad_w = new_w - w

        if pad_h == 0 and pad_w == 0:
            return img, mask

        img = F.pad(
            img,
            padding=[0, 0, pad_w, pad_h],
            fill=self.image_fill,
        )

        mask = F.pad(
            mask,
            padding=[0, 0, pad_w, pad_h],
            fill=self.mask_fill,
        )

        return img, mask

class LongSideResize:
    def __init__(self, long_side=512):
        self.long_side = long_side
        
    def __call__(self, img, mask):
        _, h, w = F.get_dimensions(img)
        scale = self.long_side / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        
        img = F.resize(
            img,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        
        mask = F.resize(
            mask,
            size=[new_h, new_w],
            interpolation=InterpolationMode.NEAREST,
            antialias=False
        )
        
        return img, mask
    
class VOC2012Dataset(VOCSegmentation):
    num_classes = 21
    ignore_index = 255
    
    classes = (
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    
    def __init__(
        self,
        root,
        split='train',
        year='2012',
        crop_size=(513, 513),
        augment=True,
        scale_range=(0.5, 2.0),
        base_size=513,
        eval_long_side=513,
        prob_class_crop=0.0,
        download=False,
        **kwargs
    ):
        super().__init__(
            root=root,
            year=year,
            image_set=split,
            download=download,
            **kwargs
        )
        
        self.augment = augment
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.base_size = base_size
        self.eval_long_side = eval_long_side
        
        self.to_img = transforms.ToImage()
        
        if augment:
            self.joint_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResize(
                    min_size=int(base_size * scale_range[0]),
                    max_size=int(base_size * scale_range[1]),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomCrop(
                    size=crop_size,
                    pad_if_needed=True,
                    fill={torch.Tensor: 0, Mask: self.ignore_index},
                    padding_mode='constant',
                )
            ])
        else:
            self.joint_tf = transforms.Compose([
                LongSideResize(long_side=eval_long_side),
                #PadToDivisible(divisor=8, image_fill=0, mask_fill=self.ignore_index)
                PadToPSPNetSize(divisor=8, image_fill=0, mask_fill=self.ignore_index)
            ])
            
        self.img_tf = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True)
        ])
        
    def _load_mask(self, mask_pil):
        mask = torch.as_tensor(np.array(mask_pil), dtype=torch.long)
        valid = ((mask >= 0) & (mask <= 20)) | (mask == self.ignore_index)
        mask = torch.where(
            valid,
            mask,
            torch.full_like(mask, self.ignore_index)
        )
        return Mask(mask)
    
    def get_raw_mask(self, index):
        _, mask = super().__getitem__(index)
        return torch.as_tensor(np.array(mask), dtype=torch.long)
    
    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        
        img = self.to_img(img)
        mask = self._load_mask(mask)
        
        if self.joint_tf is not None:
            img, mask = self.joint_tf(img, mask)
        
        img = self.img_tf(img)
        mask = torch.as_tensor(mask, dtype=torch.long)
        
        return img, mask