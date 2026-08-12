import numpy as np
import random
import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms import v2 as transforms
from torchvision.transforms import InterpolationMode
from torchvision.tv_tensors import Mask
from torchvision.transforms import functional as TF

import random
import torch
from torchvision.transforms import functional as TF


class ClassAwareRandomCrop:
    '''
    Class-aware random crop for RL frameworks
    '''
    def __init__(
        self,
        size,
        num_classes,
        ignore_index=255,
        min_pixels_in_image=64,
        min_target_pixels=256,
        max_attempts=10,
        pad_val=0,
        prob_class_crop=0.3,
        prob_weights=None,
    ):
        self.size = size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.min_pixels_in_image = min_pixels_in_image
        self.min_target_pixels = min_target_pixels
        self.max_attempts = max_attempts
        self.pad_val = pad_val
        self.prob_class_crop = prob_class_crop

        if prob_weights is None:
            self.prob_weights = None
        else:
            self.prob_weights = torch.as_tensor(
                prob_weights,
                dtype=torch.float32,
            ).clone()

    # Coordinates for random cropping
    def _random_crop_coordinates(self, h, w):
        crop_h, crop_w = self.size

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        return top, left

    # Determines suitable crop coordinates containing a specific pixel
    def _crop_coordinates_containing_pixel(
        self,
        y,
        x,
        h,
        w,
    ):
        crop_h, crop_w = self.size

        top_min = max(0, y - crop_h + 1)
        top_max = min(y, h - crop_h)

        left_min = max(0, x - crop_w + 1)
        left_max = min(x, w - crop_w)

        if top_min <= top_max:
            top = random.randint(top_min, top_max)
        else:
            top = max(0, min(y - crop_h // 2, h - crop_h))

        if left_min <= left_max:
            left = random.randint(left_min, left_max)
        else:
            left = max(0, min(x - crop_w // 2, w - crop_w))

        return top, left

    # Crop is applied with specified coordinates
    def _apply_crop(self, img, mask, top, left):
        crop_h, crop_w = self.size

        return (
            TF.crop(
                img,
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
            ),
            TF.crop(
                mask,
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
            ),
        )

    # Executed when an image is retrieved from the dataset
    def __call__(self, img, mask):
        crop_h, crop_w = self.size
        _, h, w = img.shape

        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)

        # Padding may be necessary
        if pad_h > 0 or pad_w > 0:
            padding = [0, 0, pad_w, pad_h]

            img = TF.pad(
                img=img,
                padding=padding,
                fill=self.pad_val,
            )

            mask = TF.pad(
                img=mask,
                padding=padding,
                fill=self.ignore_index,
            )

        _, h, w = img.shape

        # Determines whether a random crop or a class-aware crop is performed
        if random.random() >= self.prob_class_crop:
            top, left = self._random_crop_coordinates(h, w)
            return self._apply_crop(img, mask, top, left)

        # Class-aware crop is performed in this case
        mask_long = torch.as_tensor(mask, dtype=torch.long)

        valid = (mask_long != self.ignore_index) & (mask_long >= 0) & (mask_long < self.num_classes)

        counts = torch.bincount(mask_long[valid].flatten(), minlength=self.num_classes)

        present_classes = torch.where(counts >= self.min_pixels_in_image)[0]

        if present_classes.numel() == 0:
            top, left = self._random_crop_coordinates(h, w)
            return self._apply_crop(img, mask, top, left)

        if self.prob_weights is None:
            class_weights = torch.ones(present_classes.numel(), dtype=torch.float32)
        else:
            class_weights = self.prob_weights[present_classes.cpu()].float()

        class_weights = torch.nan_to_num(
            class_weights,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if class_weights.sum() <= 0:
            class_weights = torch.ones_like(class_weights)

        class_probabilities = class_weights / class_weights.sum()

        # Samples a target class
        sampled_position = torch.multinomial(class_probabilities, num_samples=1).item()

        target_class = int(present_classes[sampled_position].item())

        ys, xs = torch.where(mask_long == target_class)

        best_top = None
        best_left = None
        best_target_count = -1

        # Iterativel determines crop coordinates and keeps the best ones
        for _ in range(self.max_attempts):
            pixel_idx = torch.randint(low=0, high=ys.numel(),size=(1,)).item()

            y = int(ys[pixel_idx].item())
            x = int(xs[pixel_idx].item())

            top, left = self._crop_coordinates_containing_pixel(
                y=y,
                x=x,
                h=h,
                w=w,
            )

            target_count = (mask_long[top:top + crop_h, left:left + crop_w] == target_class).sum().item()

            if target_count > best_target_count:
                best_target_count = target_count
                best_top = top
                best_left = left

            if target_count >= self.min_target_pixels:
                break

        return self._apply_crop(
            img,
            mask,
            best_top,
            best_left,
        )
            
        

class CityscapesDataset(Cityscapes):
    
    num_classes = 19
    ignore_index = 255
    
    def __init__(
        self, 
        root, 
        split='train', 
        crop_size=(449, 449),
        eval_size=(512, 1024), 
        augment=True, 
        scale_range=(0.75, 1.75),
        prob_class_crop=0.5,
        **kwargs
    ):
        super().__init__(
            root=root,
            split=split,
            mode='fine',
            target_type='semantic',
            **kwargs
        )
        
        self.split = split
        self.augment = augment and split == "train"
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.eval_size = eval_size
        self.prob_class_crop = prob_class_crop
        
        self.labelid_to_trainid = torch.full((256,), self.ignore_index, dtype=torch.long)
        self.label_mapping = {
            7: 0,   # road
            8: 1,   # sidewalk
            11: 2,  # building
            12: 3,  # wall
            13: 4,  # fence
            17: 5,  # pole
            19: 6,  # traffic light
            20: 7,  # traffic sign
            21: 8,  # vegetation
            22: 9,  # terrain
            23: 10, # sky
            24: 11, # person
            25: 12, # rider
            26: 13, # car
            27: 14, # truck
            28: 15, # bus
            31: 16, # train
            32: 17, # motorcycle
            33: 18, # bicycle
        }
        
        for k, v in self.label_mapping.items():
            self.labelid_to_trainid[k] = v
        
        self.to_img = transforms.ToImage()
        
        self.flip_tf = transforms.RandomHorizontalFlip(0.5)

        self.resize_tf = transforms.RandomResize(
            min_size=int(1024 * scale_range[0]),
            max_size=int(1024 * scale_range[1]),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        
        prob_weights = torch.ones(self.num_classes)
        prob_weights[3]  = 4.0  # wall
        prob_weights[4]  = 4.0  # fence
        prob_weights[5]  = 3.0  # pole
        prob_weights[6]  = 4.0  # traffic light
        prob_weights[7]  = 3.0  # traffic sign
        prob_weights[11] = 3.0  # person
        prob_weights[12] = 6.0  # rider
        prob_weights[14] = 4.0  # truck
        prob_weights[15] = 4.0  # bus
        prob_weights[16] = 6.0  # train
        prob_weights[17] = 6.0  # motorcycle
        prob_weights[18] = 4.0  # bicycle

        #print(f"using cityscapes with crop_size={crop_size}, prob_class_crop={prob_class_crop}")
        
        self.crop_tf = ClassAwareRandomCrop(
            size=crop_size,
            num_classes=self.num_classes,
            prob_class_crop=self.prob_class_crop,
            prob_weights=prob_weights
        )

        self.img_tf = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
        ])
        
    def get_raw_mask(self, index):
        _, raw_mask = super().__getitem__(index)
        raw_mask = torch.as_tensor(np.array(raw_mask), dtype=torch.long)
        return self.labelid_to_trainid[raw_mask]
                
    def __getitem__(self, index):
        #print(f"[getitem] split={self.split}, augment={self.augment}", flush=True)
        img, mask = super().__getitem__(index)

        img = self.to_img(img)
        
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask = self.labelid_to_trainid[mask]
        mask = Mask(mask.to(torch.uint8))

        if self.augment:
            img, mask = self.flip_tf(img, mask)
            img, mask = self.resize_tf(img, mask)
            img, mask = self.crop_tf(img, mask)
            
        else:
            img = transforms.functional.resize(
                img,
                size=self.eval_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True
            )
            mask = transforms.functional.resize(
                mask,
                size=self.eval_size,
                interpolation=InterpolationMode.NEAREST,
                antialias=False
            )

        img = self.img_tf(img)
        mask = torch.as_tensor(mask, dtype=torch.long)

        return img, mask