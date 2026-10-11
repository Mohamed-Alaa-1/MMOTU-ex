import torch
import logging

class GradientMonitor:
    def __init__(self, model: torch.nn.Module, logger: logging.Logger, clip_value: float = 1.0, explosion_threshold: float = 10.0, skip_threshold: float = 100.0):
        """Monitor, log, and clip gradients to prevent explosions or NaN updates."""
        self.model = model
        self.logger = logger
        self.clip_value = clip_value
        self.explosion_threshold = explosion_threshold
        self.skip_threshold = skip_threshold

    def get_grad_norm(self) -> float:
        """Return current total L2 gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def check_and_clip(self, loss: torch.Tensor) -> bool:
        """
        Check for NaNs/Infs and explosive gradients.
        Returns True if the batch should be skipped.
        """
        # 1. Detect NaN/Inf in gradients
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    self.logger.warning("NaN/Inf detected in gradients. Skipping batch.")
                    self.model.zero_grad()
                    return True

        # 2. Compute total gradient norm
        grad_norm = self.get_grad_norm()

        # 3. Skip threshold check
        if grad_norm > self.skip_threshold:
            self.logger.critical(f"Gradient norm ({grad_norm:.2f}) exceeds skip threshold ({self.skip_threshold}). Skipping batch.")
            self.model.zero_grad()
            return True

        # 4. Explosion warning check (Silenced)
        # if grad_norm > self.explosion_threshold:
        #     self.logger.warning(f"Gradient norm ({grad_norm:.2f}) exceeds explosion threshold ({self.explosion_threshold}).")

        # 5. Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

        # 6. Proceed
        return False

    def log_layer_norms(self, step: int):
        """Log per-layer gradient norms every N steps for debugging."""
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                norm = p.grad.data.norm(2).item()
                self.logger.debug(f"Step {step} - Layer {name} grad norm: {norm:.4f}")
