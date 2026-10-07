import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.dataset import get_dataloaders
from models.factory import get_model
from utils.checkpoint import load_checkpoint
from .calibration import TemperatureScaler, expected_calibration_error
from .conformal import MondrianConformalPredictor, evaluate_conformal_sets
from .selective_prediction import risk_coverage_curve
from .estimators import MCDropoutEstimator
from .uncertainty_stats import compute_per_image_risk_contributions, run_repeated_measures_tests
from visualization.uncertainty_plots import (plot_reliability_diagram, plot_risk_coverage_curves,
                                             plot_coverage_per_class, plot_set_size_distribution)

logger = logging.getLogger(__name__)


@torch.no_grad()
def _collect_logits(model, loader, device):
    model.eval()
    zs, ys = [], []
    for batch in loader:
        zs.append(model(batch[0].to(device, non_blocking=True)).float().cpu())
        ys.append(batch[2])
    return torch.cat(zs), torch.cat(ys).long()


def _mc_entropy(model, loader, device, n_samples):
    est = MCDropoutEstimator(copy.deepcopy(model), device, n_samples)   # deepcopy: keep model clean
    return np.concatenate([est.predict(b[0])["predictive_entropy"] for b in loader])


def _resolve_ckpt(name, path, config):
    swa = Path(config.output.checkpoints_dir) / f"{config.experiment.run_name}_{name}_swa.pt"
    return str(swa) if swa.exists() else str(path)


def run_uncertainty_pipeline(trained_models: dict, config, device: torch.device) -> pd.DataFrame:
    cfg = getattr(config, "uncertainty", config) # fallback
    K = config.training.num_classes
    out_dir, fig_dir = Path(config.output.results_dir), Path(config.output.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader, calib_loader, test_loader = get_dataloaders(config.data.splits_csv, config)

    rows, per_class_rows = [], []
    rc_by_bb, conf_by_bb, sets_by_bb, risk_by_bb = {}, {}, {}, {}
    ref_labels = None

    for name, ckpt in trained_models.items():
        base = name.split("_fold")[0]
        model, _ = get_model(base, num_classes=K)
        load_checkpoint(_resolve_ckpt(name, ckpt, config), model)
        model = model.to(device).eval()

        val_z, val_y = _collect_logits(model, val_loader, device)
        cal_z, cal_y = _collect_logits(model, calib_loader, device)
        test_z, test_y = _collect_logits(model, test_loader, device)
        test_y_np, cal_y_np = test_y.numpy(), cal_y.numpy()
        if ref_labels is None:
            ref_labels = test_y_np
        assert np.array_equal(ref_labels, test_y_np), "Test order differs across backbones"

        fit_z, fit_y = (val_z, val_y) if getattr(cfg, "temperature_fit_split", "val") == "val" else (cal_z, cal_y)
        scaler = TemperatureScaler()
        T = scaler.fit(fit_z, fit_y)
        if not 0.05 < T < 10:
            logger.warning(f"{name}: unusual temperature {T:.3f}")

        sm = lambda z, t=1.0: F.softmax(z / t, dim=1).numpy()
        cal_p, test_p, test_raw = sm(cal_z, T), sm(test_z, T), sm(test_z)

        ece_pre = expected_calibration_error(test_raw, test_y_np, getattr(cfg, "ece_n_bins", 15))
        ece_post = expected_calibration_error(test_p, test_y_np, getattr(cfg, "ece_n_bins", 15))
        plot_reliability_diagram(test_raw, test_p, test_y_np, str(fig_dir / f"{name}_reliability.png"),
                                 n_bins=getattr(cfg, "ece_n_bins", 15))

        rc_soft = risk_coverage_curve(test_p, test_y_np, -np.log(np.clip(test_p.max(1), 1e-12, None)))
        rc_mc = risk_coverage_curve(test_p, test_y_np, _mc_entropy(model, test_loader, device,
                                                                    getattr(cfg, "mc_dropout_samples", 30)))
        with open(out_dir / f"{name}_riskcoverage.json", "w") as f:
            json.dump({**rc_soft, "aurc_mc_dropout": rc_mc["aurc"],
                       "coverage_at_5pct_risk_mc_dropout": rc_mc["coverage_at_5pct_risk"]}, f)
        rc_by_bb[name] = rc_soft
        risk_by_bb[name] = compute_per_image_risk_contributions(test_p, test_y_np)

        alpha_sweep = getattr(cfg, "alpha_sweep", [0.05, 0.10, 0.20])
        alpha_default = getattr(cfg, "alpha", 0.10)
        score_method = getattr(cfg, "score", "lac")
        rare_policy = getattr(cfg, "rare_class_policy", "global")
        min_cal = getattr(cfg, "min_calibration_per_class", 10)

        for alpha in alpha_sweep:
            cp = MondrianConformalPredictor(alpha, min_cal, score_method, rare_policy)
            cp.calibrate(cal_p, cal_y_np)
            sets = cp.predict_sets(test_p)
            ev = evaluate_conformal_sets(sets, test_y_np, K)

            rows.append({"Backbone": name, "Alpha": alpha, "Temperature": T,
                         "NLL_Before": scaler.nll_before, "NLL_After": scaler.nll_after,
                         "ECE_Pre": ece_pre, "ECE_Post": ece_post,
                         "Test_Accuracy": float((test_p.argmax(1) == test_y_np).mean()),
                         "Marginal_Coverage": ev["marginal_coverage"], "Avg_Set_Size": ev["avg_set_size"],
                         "Singleton_Rate": ev["singleton_rate"],
                         "AURC_Softmax": rc_soft["aurc"], "AURC_MC_Dropout": rc_mc["aurc"],
                         "Cov_at_5pct_Risk": rc_soft["coverage_at_5pct_risk"],
                         "Fallback_Classes": ",".join(map(str, cp.fallback_classes)),
                         "Calib_N": int(cp.class_n.sum())})
            for c, cov in ev["per_class_coverage"].items():
                lo, hi = ev["per_class_coverage_ci"][c]
                per_class_rows.append({"Backbone": name, "Alpha": alpha, "Class": c,
                                       "N_Calib": int(cp.class_n[c]), "N_Test": ev["per_class_n"][c],
                                       "Coverage": cov, "CI_Low": lo, "CI_High": hi,
                                       "Avg_Set_Size": ev["per_class_set_size"][c],
                                       "Global_Fallback": c in cp.fallback_classes})
            if np.isclose(alpha, alpha_default):
                conf_by_bb[name] = {"per_class_coverage": ev["per_class_coverage"]}
                sets_by_bb[name] = sets

        del model
        torch.cuda.empty_cache()

    plot_risk_coverage_curves(rc_by_bb, str(fig_dir / "all_risk_coverage.png"))
    plot_coverage_per_class(conf_by_bb, str(fig_dir / "all_class_coverage.png"), alpha=alpha_default)
    plot_set_size_distribution(sets_by_bb, str(fig_dir / "all_set_sizes.png"))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "uncertainty_conformal_summary.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out_dir / "uncertainty_per_class_coverage.csv", index=False)
    run_repeated_measures_tests(risk_by_bb).to_csv(out_dir / "backbone_uncertainty_friedman.csv", index=False)
    return summary
