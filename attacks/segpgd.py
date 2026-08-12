'''
Implementation of the SegPGD adversarial attack
'''
import torch
import torch.nn.functional as F

class SegPGD:
    def __init__(
        self, 
        iterations, 
        epsilon, 
        alpha, 
        num_classes, 
        ignore_index, 
        clamp_min=0.0, 
        clamp_max=1.0,
        scale_margins=False
    ):
        self.iterations = iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # This is used for DAFA
        self.scale_margins = scale_margins
        
    def prepare_mask(self, mask):
        '''
        Ensures that mask has shape [B,H,W] before the attack
        '''
        if mask.ndim == 4:
            if mask.shape[1] == 1:
                mask = mask[:, 0]
            elif mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

        if mask.ndim != 3:
            raise ValueError(f"Mask must be [B,H,W], got {mask.shape}")

        return mask.long()
        
    def init_linf(self, X, margins_map):
        '''
        Initialize the adversarial example as X + U(-epsilon,epsilon)
        Clamping is applied such that all pixel values stay within the valid [0,1] range
        '''
        X_clean = X.detach()
        X_adv = X_clean.clone()
        noise = (torch.rand_like(X_adv) * 2.0 - 1.0) * margins_map
        X_adv += noise
        X_adv = torch.clamp(X_adv, min=self.clamp_min, max=self.clamp_max)
        return X_clean, X_adv
    
    def lambda_t(self, iteration):
        return iteration / (2.0 * max(self.iterations, 1))
    
    def segpgd_loss(self, logits, mask, current_iteration):
        '''
        Loss computation as specified in the SegPGD paper
        '''
        valid = mask != self.ignore_index
        
        pixel_ce = F.cross_entropy(
            logits,
            mask,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == mask) & valid
            incorrect = (pred != mask) & valid
            lam = self.lambda_t(current_iteration)
            pixel_weights = torch.zeros_like(pixel_ce)
            pixel_weights[correct] = 1 - lam
            pixel_weights[incorrect] = lam
            
        denom = valid.float().sum().clamp_min(1.0)
        loss = (pixel_ce * pixel_weights).sum() / denom
        return loss
    
    def prepare_class_weights(
        self,
        class_weights,
        reference,
    ):
        '''
        Checks class weights and puts them on the correct device
        '''
        if class_weights is None:
            return torch.ones(self.num_classes, device=reference.device, dtype=reference.dtype)
    
        weights = class_weights.to(device=reference.device, dtype=reference.dtype)
    
        if not torch.isfinite(weights).all():
            raise ValueError(f"Non-finite DAFA weights: {weights}")
    
        if (weights <= 0).any():
            raise ValueError(f"DAFA weights must be positive: {weights}")
    
        return weights
    
    def build_margin_maps(
        self,
        X,
        mask,
        class_weights,
    ):
        '''
        Segmentation needs a margin map for scaling cross-entropy pixel maps
        '''
        weights = self.prepare_class_weights(class_weights=class_weights, reference=X)
    
        valid = (mask != self.ignore_index) & (mask >= 0) & (mask < self.num_classes)
    
        scale_map = torch.ones(mask.shape, device=X.device, dtype=X.dtype)
    
        scale_map[valid] = weights[mask[valid]]
        scale_map = scale_map.unsqueeze(1)
    
        if self.scale_margins:
            epsilon_map = self.epsilon * scale_map
        else:
            epsilon_map = torch.full_like(scale_map, self.epsilon)

        alpha_map = torch.full_like(epsilon_map, self.alpha)
    
        return epsilon_map, alpha_map
    
             
    def attack(self, model, X, mask, class_weights=None, clean_logits=None):
        '''
        Performs the SegPGD attack (clean_logits is a placeholder that is unused)
        '''
        attack_model = model.module if hasattr(model, 'module') else model
        was_training = attack_model.training
        attack_model.eval()
        
        try:
            mask = self.prepare_mask(mask)
            
            if X.ndim != 4:
                raise ValueError(f"SegPGD input X must have shape [B,C,H,W], got {X.shape}")
            
            epsilon_map, alpha_map = self.build_margin_maps(
                X=X,
                mask=mask,
                class_weights=class_weights
            )
            
            X_clean, X_adv = self.init_linf(X, epsilon_map)
            
            # This represents the attack loop in the SegPGD algorithm
            for step in range(self.iterations):
                X_adv = X_adv.detach().requires_grad_(True)
                
                logits = attack_model(X_adv, return_aux=False)
                loss = self.segpgd_loss(logits=logits, mask=mask, current_iteration=step)
                    
                grad = torch.autograd.grad(loss, X_adv, retain_graph=False, create_graph=False)[0]
                
                with torch.no_grad():
                    X_adv = X_adv + alpha_map * grad.sign()
                    delta = torch.clamp(X_adv - X_clean, -epsilon_map, epsilon_map)
                    X_adv = torch.clamp(X_clean + delta, self.clamp_min, self.clamp_max)
            
            return X_adv.detach()
                    
        finally:
            if was_training:
                attack_model.train()