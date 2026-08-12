import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from models.pspnet import PSPNet, DeepLabV3
from models.normalize_wrapper import NormalizeWrapper

def load_dense_model(model_name, num_classes, device, distributed, load_existing=False, smp_model=False):  
    base_model = None
    if model_name == 'pspnet':
        base_model = load_pspnet(num_classes=num_classes, load_existing=load_existing, smp_model=smp_model)
    
    elif model_name == 'deeplabv3':
        base_model = load_deeplabv3(num_classes=num_classes, load_existing=load_existing, smp_model=smp_model)
    
    if base_model == None:
        raise ValueError(f'Unsupported model: {model_name}')
    
    base_model.to(device)
    
    if distributed:
        if device.type == 'cuda':
            base_model = nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            
        model = DistributedDataParallel(
            base_model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=True,
            find_unused_parameters=False
        )
        
        return model
    
    return base_model
            
    
def load_pspnet(num_classes, load_existing, smp_model=False):
    if smp_model:
        pass
    else:
        raw_model = PSPNet(
            layers=50,
            bins=(1, 2, 3, 6),
            dropout=0.1,
            classes=num_classes,
            zoom_factor=8,
            use_ppm=True,
            pretrained=not load_existing,
            BatchNorm=nn.BatchNorm2d
        )
        model = NormalizeWrapper(raw_model)
        return model
  
    
def load_deeplabv3(num_classes, load_existing, smp_model=False):
    if smp_model:
        pass
    else:
        raw_model = DeepLabV3(
            layers=50,
            atrous_rates=(6, 12, 18),
            dropout=0.1,
            classes=num_classes,
            zoom_factor=8,
            use_aspp=True,
            BatchNorm=nn.BatchNorm2d,
            pretrained=not load_existing,
        )
        model = NormalizeWrapper(raw_model)
        return model