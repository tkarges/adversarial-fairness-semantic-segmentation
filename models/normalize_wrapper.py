import torch
import torch.nn as nn

# Channel-wise mean and standard deviations for the ImageNet dataset 
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class NormalizeWrapper(nn.Module):
    '''
    Normalizes model inputs such that they have the distribution expected by the ResNet
    backbone, which has been pretrained on ImageNet.
    
    The dataloaders emit images in the pixel range [0,1]. This is very important for the
    current attack implementations. The models themselves then normalize these values before
    the forward pass.
    '''
    def __init__(self, model, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.model = model
        
        # Imagenet mean
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        
        # Imagenet standard deviation
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )
        
    def forward(self, x, return_aux=False):
        '''
        Forward pass wrapper around the model.
        
        An image is normalized using the mean and std buffers before the actual
        forward pass is performed.
        '''
        x = (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)
        out = self.model(x, return_aux=return_aux)
        
        if isinstance(out, tuple) and not return_aux:
            out = out[-1]
            
        return out