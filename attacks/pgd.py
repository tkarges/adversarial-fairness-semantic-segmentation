import torch
import torch.nn.functional as F

class PGD:
    def __init__(
        self, 
        iterations, 
        epsilon, 
        alpha, 
        num_classes, 
        ignore_index, 
        clamp_min=0.0, 
        clamp_max=1.0,
        random_start=True
    ):
        self.iterations = iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.random_start = random_start
        
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
    
    def prepare_class_weights(self, reference, class_weights=None):
        if class_weights is None:
            return torch.ones(self.num_classes, device=reference.device, dtype=reference.dtype)
        
        weights = class_weights.to(device=reference.device, dtype=reference.dtype)
        
        if (weights < 0).any() or not torch.isfinite(weights).all():
            raise ValueError(f"Invalid weights for PGD attack margin scaling: {weights}")
        
        return weights
    
    def build_margin_map(self, X, mask, class_weights=None):
        weights = self.prepare_class_weights(reference=X, class_weights=class_weights)
        valid = (mask != self.ignore_index) & (mask >= 0) & (mask < self.num_classes)
        scale_map = torch.ones(mask.shape, device=X.device, dtype=X.dtype)
        scale_map[valid] = weights[mask[valid]]
        scale_map = scale_map.unsqueeze(1)
        epsilon_map = self.epsilon * scale_map
        return epsilon_map
        
            
    def init_linf(self, X, margins_map):
        '''
        Initialize the adversarial example as X + U(-epsilon,epsilon)
        Clamping is applied such that all pixel values stay within the valid [0,1] range
        '''
        X_clean = X.detach()
        
        if self.random_start:
            noise = (torch.rand_like(X_clean) * 2.0 - 1.0) * margins_map
            X_adv = X_clean + noise
        else:
            X_adv = X_clean.clone()
        
        X_adv = X_adv.clamp(self.clamp_min, self.clamp_max)
        
        return X_clean, X_adv
    
    def pgd_loss(self, logits, mask):
        valid = (mask != self.ignore_index) & (mask >= 0) & (mask < self.num_classes)
        if not valid.any():
            return logits.sum() * 0.0
        
        pixel_loss = F.cross_entropy(
            logits,
            mask,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        
        return pixel_loss[valid].mean()
             
    def attack(self, model, X, mask, class_weights=None):
        attack_model = model.module if hasattr(model, 'module') else model
        was_training = attack_model.training
        attack_model.eval()
        
        try:
            mask = self.prepare_mask(mask)
            
            if X.ndim != 4:
                raise ValueError(f"SegPGD input X must have shape [B,C,H,W], got {X.shape}")
            
            margins_map = self.build_margin_map(X=X, mask=mask, class_weights=class_weights)
            
            X_clean, X_adv = self.init_linf(X, margins_map)
            
            for _ in range(self.iterations):
                X_adv = X_adv.detach().requires_grad_(True)
                
                logits = attack_model(X_adv, return_aux=False)
                loss = self.pgd_loss(logits=logits, mask=mask)
                    
                grad = torch.autograd.grad(loss, X_adv, retain_graph=False, create_graph=False, only_inputs=True)[0]
                
                with torch.no_grad():
                    X_adv = X_adv + self.alpha * grad.sign()
                    delta = torch.clamp(X_adv - X_clean, -margins_map, margins_map)
                    X_adv = torch.clamp(X_clean + delta, self.clamp_min, self.clamp_max)
            
            return X_adv.detach()
                    
        finally:
            if was_training:
                attack_model.train()

class TradesPGD:
    '''
    TRADES PGD attack
    '''
    def __init__(
        self,
        iterations,
        epsilon,
        alpha,
        num_classes,
        ignore_index,
        clamp_min=0.0,
        clamp_max=1.0,
        random_start=True,
        scale_margins=False,
        scale_step_size=False
    ):
        self.iterations = iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.random_start = random_start
        self.scale_margins = scale_margins
        self.scale_step_size = scale_step_size

    def prepare_mask(self, mask):
        if mask.ndim == 4:
            if mask.shape[1] == 1:
                mask = mask[:, 0]
            elif mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                raise ValueError(
                    f"Unexpected mask shape: {mask.shape}"
                )

        if mask.ndim != 3:
            raise ValueError(
                f"Mask must be [B,H,W], got {mask.shape}"
            )

        return mask.long()

    def extract_logits(self, output, target_size):
        if isinstance(output, tuple):
            logits = output[0]
        elif isinstance(output, dict):
            logits = output["out"]
        else:
            logits = output

        return logits

    def prepare_class_weights(
        self,
        class_weights,
        reference,
    ):
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
        weights = self.prepare_class_weights(class_weights=class_weights, reference=X)

        valid = ((mask != self.ignore_index) & (mask >= 0) & (mask < self.num_classes))

        scale_map = torch.ones(mask.shape, device=X.device, dtype=X.dtype)

        scale_map[valid] = weights[mask[valid]]
        scale_map = scale_map.unsqueeze(1)

        if self.scale_margins:
            epsilon_map = self.epsilon * scale_map
        else:
            epsilon_map = torch.full_like(scale_map, self.epsilon)

        if self.scale_step_size:
            alpha_map = self.alpha * scale_map
        else:
            alpha_map = torch.full_like(epsilon_map, self.alpha)

        return epsilon_map, alpha_map

    def attack(
        self,
        model,
        X,
        mask,
        clean_logits,
        class_weights=None,
    ):
        attack_model = model.module if hasattr(model, "module") else model

        was_training = attack_model.training
        attack_model.eval()

        try:
            mask = self.prepare_mask(mask)

            if X.ndim != 4:
                raise ValueError(
                    f"X must be [B,C,H,W], got {X.shape}"
                )

            clean_logits = self.extract_logits(clean_logits, target_size=mask.shape[-2:]).detach()

            clean_probs = F.softmax(clean_logits, dim=1)

            epsilon_map, alpha_map = self.build_margin_maps(
                X=X,
                mask=mask,
                class_weights=class_weights,
            )

            X_clean = X.detach()

            if self.random_start:
                noise = (torch.rand_like(X_clean) * 2.0 - 1.0) * epsilon_map
                X_adv = X_clean + noise
            else:
                X_adv = X_clean.clone()

            X_adv = X_adv.clamp(self.clamp_min, self.clamp_max)

            valid = ((mask != self.ignore_index) & (mask >= 0) & (mask < self.num_classes))

            for _ in range(self.iterations):
                X_adv = X_adv.detach().requires_grad_(True)

                try:
                    output_adv = attack_model(X_adv, return_aux=False)
                except TypeError:
                    output_adv = attack_model(X_adv)

                logits_adv = self.extract_logits(
                    output_adv,
                    target_size=mask.shape[-2:],
                )

                kl_map = F.kl_div(
                    F.log_softmax(
                        logits_adv,
                        dim=1,
                    ),
                    clean_probs,
                    reduction="none",
                ).sum(dim=1)

                if valid.any():
                    loss = kl_map[valid].mean()
                else:
                    loss = logits_adv.sum() * 0.0

                grad = torch.autograd.grad(
                    loss,
                    X_adv,
                    retain_graph=False,
                    create_graph=False,
                    only_inputs=True,
                )[0]

                with torch.no_grad():
                    X_adv = X_adv + alpha_map * grad.sign()

                    delta = X_adv - X_clean

                    delta = torch.maximum(
                        torch.minimum(
                            delta,
                            epsilon_map,
                        ),
                        -epsilon_map
                    )

                    X_adv = (X_clean + delta).clamp(self.clamp_min, self.clamp_max)

            return X_adv.detach()

        finally:
            if was_training:
                attack_model.train()