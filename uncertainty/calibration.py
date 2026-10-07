import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Scalar temperature fit by NLL minimisation. T<1 sharpens (model underconfident), T>1 softens."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.nll_before = self.nll_after = float("nan")

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 500) -> float:
        logits, labels = logits.detach().float(), labels.detach().long()
        with torch.no_grad():
            self.nll_before = F.cross_entropy(logits, labels).item()
        opt = torch.optim.LBFGS([self.log_temperature], lr=1.0, max_iter=max_iter,
                                tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            self.nll_after = F.cross_entropy(logits / self.temperature, labels).item()
            if self.nll_after > self.nll_before + 1e-6:
                logger.warning("LBFGS did not improve NLL; resetting T=1.")
                self.log_temperature.zero_()
                self.nll_after = self.nll_before
        return self.temperature.item()


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs(acc[m].mean() - conf[m].mean())
    return float(ece)
