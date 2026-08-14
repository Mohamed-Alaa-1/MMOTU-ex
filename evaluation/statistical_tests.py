import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, f_oneway, mannwhitneyu

# NOTE: pairwise_tukeyhsd (ANOVA post-hoc) has been removed.
# Tukey HSD assumes normality and equal variances, which do not hold for
# per-image AURC/risk-contribution distributions.  Paired bootstrap
# comparisons are used instead (compare_backbones).


class StatisticalAnalyzer:
    def correct_vs_incorrect_exbale(self, results_df: pd.DataFrame,
                                  cam_threshold: float = 0.5) -> pd.DataFrame:
        """
        Compares ExBale distributions between correctly and incorrectly classified images
        for each XAI method at a specific threshold.
        """
        from statsmodels.stats.multitest import multipletests

        # Filter by threshold
        df_thresh = results_df[results_df['cam_threshold'] == cam_threshold].copy()

        methods = df_thresh['xai_method'].unique()
        stats_list = []

        for method in methods:
            m_df = df_thresh[df_thresh['xai_method'] == method]
            correct_exbale = m_df[m_df['is_correct'] == True]['exbale'].dropna()
            incorrect_exbale = m_df[m_df['is_correct'] == False]['exbale'].dropna()

            if len(correct_exbale) < 2 or len(incorrect_exbale) < 2:
                stats_list.append({
                    "xai_method": method,
                    "correct_mean": correct_exbale.mean() if not correct_exbale.empty else 0,
                    "incorrect_mean": incorrect_exbale.mean() if not incorrect_exbale.empty else 0,
                    "delta": (correct_exbale.mean() - incorrect_exbale.mean()) if not (correct_exbale.empty or incorrect_exbale.empty) else 0,
                    "wilcoxon_p": 1.0
                })
                continue

            # Using Mann-Whitney U test as the groups are independent
            res = mannwhitneyu(correct_exbale, incorrect_exbale, alternative='two-sided')

            stats_list.append({
                "xai_method": method,
                "correct_mean": correct_exbale.mean(),
                "incorrect_mean": incorrect_exbale.mean(),
                "delta": correct_exbale.mean() - incorrect_exbale.mean(),
                "wilcoxon_p": res.pvalue
            })

        stats_df = pd.DataFrame(stats_list)

        # Bonferroni correction
        if not stats_df.empty:
            p_vals = stats_df['wilcoxon_p'].values
            rejected, p_corrected, _, _ = multipletests(p_vals, method='bonferroni')
            stats_df['bonferroni_p'] = p_corrected
            stats_df['significant'] = rejected

        return stats_df

    def compare_xai_methods(self, results_df: pd.DataFrame, metric: str = "exbale", cam_threshold: float = 0.5) -> dict:
        """
        Pairwise Wilcoxon signed-rank test between XAI methods.
        Bonferroni correction is applied internally or reported raw.
        """
        if 'cam_threshold' in results_df.columns:
            df_thresh = results_df[np.isclose(results_df['cam_threshold'], cam_threshold)].copy()
        else:
            df_thresh = results_df.copy()

        # Pivot the dataframe to have images as rows and methods as columns.
        # One row per image is the paired sample unit for the signed-rank test.
        pivot_df = df_thresh.pivot_table(index='image_path', columns='xai_method', values=metric).dropna()
        methods = pivot_df.columns.tolist()
        n_images = len(pivot_df)

        n_methods = len(methods)
        p_values = np.ones((n_methods, n_methods))
        effect_sizes = np.zeros((n_methods, n_methods))

        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                g1 = pivot_df[methods[i]].values
                g2 = pivot_df[methods[j]].values

                try:
                    res = wilcoxon(g1, g2)
                    p_val = res.pvalue
                except Exception:
                    p_val = 1.0

                p_values[i, j] = p_val
                p_values[j, i] = p_val

                d = self.cohens_d(g1, g2)
                effect_sizes[i, j] = d
                effect_sizes[j, i] = -d

        # Bonferroni correction
        num_comparisons = n_methods * (n_methods - 1) / 2
        p_values = np.clip(p_values * num_comparisons, 0, 1.0)

        return {
            "methods": methods,
            "p_values": p_values.tolist(),
            "effect_sizes": effect_sizes.tolist(),
            "n_images": n_images,
            "cam_threshold": cam_threshold
        }

    def compare_backbones(self, all_results_dfs: dict, metric: str = "exbale", cam_threshold: float = 0.5) -> dict:
        """
        Compare backbone models using paired bootstrap hypothesis tests for AURC/risk metrics.

        Replaces the previous Tukey HSD approach, which assumed normality and equal
        variances — assumptions that do not hold for per-image NLL/risk-contribution
        distributions. Paired bootstrap is distribution-free and appropriate here.

        Args:
            all_results_dfs: Dict mapping backbone name -> results DataFrame.
            metric: Column name of the metric to compare (default 'exbale').
            cam_threshold: Filter rows to this CAM threshold.

        Returns:
            dict with:
              anova_p          : One-way Kruskal-Wallis p-value (non-parametric analogue)
              anova_F          : Kruskal-Wallis statistic
              n_per_backbone   : {name: n_samples}
              bootstrap_results: DataFrame of pairwise bootstrap comparisons
        """
        from scipy.stats import kruskal

        combined_data = []
        n_per_backbone = {}
        for bb, df in all_results_dfs.items():
            if 'cam_threshold' in df.columns:
                df_thresh = df[np.isclose(df['cam_threshold'], cam_threshold)].copy()
            else:
                df_thresh = df

            vals = df_thresh[metric].dropna().values
            combined_data.append(vals)
            n_per_backbone[bb] = len(vals)

        # Non-parametric omnibus test (Kruskal-Wallis replaces ANOVA)
        try:
            F, p = kruskal(*combined_data)
        except Exception:
            F, p = 0.0, 1.0

        # Pairwise paired bootstrap comparisons
        backbones = list(all_results_dfs.keys())
        bootstrap_rows = []
        rng = np.random.default_rng(seed=42)

        for i in range(len(backbones)):
            for j in range(i + 1, len(backbones)):
                b1, b2 = backbones[i], backbones[j]
                v1, v2 = combined_data[i], combined_data[j]
                # For unpaired samples use bootstrap permutation test
                n1, n2 = len(v1), len(v2)
                obs_diff = float(np.mean(v1) - np.mean(v2))
                pooled = np.concatenate([v1, v2])
                n_boot = 5000
                boot_diffs = np.array([
                    np.mean(rng.choice(pooled, n1, replace=True))
                    - np.mean(rng.choice(pooled, n2, replace=True))
                    for _ in range(n_boot)
                ])
                # Two-tailed p-value
                p_boot = float(np.mean(np.abs(boot_diffs) >= np.abs(obs_diff)))
                # 95% bootstrap CI for the difference
                ci_lo = float(np.percentile(boot_diffs, 2.5))
                ci_hi = float(np.percentile(boot_diffs, 97.5))
                cohen_d = self.cohens_d(v1, v2)
                bootstrap_rows.append({
                    "backbone_1": b1,
                    "backbone_2": b2,
                    "mean_1": float(np.mean(v1)),
                    "mean_2": float(np.mean(v2)),
                    "observed_diff": obs_diff,
                    "bootstrap_p": p_boot,
                    "ci_95_lo": ci_lo,
                    "ci_95_hi": ci_hi,
                    "cohens_d": cohen_d,
                    "significant_0.05": p_boot < 0.05,
                })

        bootstrap_df = pd.DataFrame(bootstrap_rows)

        return {
            "kruskal_p": p,
            "kruskal_stat": F,
            "n_per_backbone": n_per_backbone,
            "cam_threshold": cam_threshold,
            "bootstrap_results": bootstrap_df,
            # Keep field name anova_p / anova_F for backward compatibility
            "anova_p": p,
            "anova_F": F,
        }

    # ------------------------------------------------------------------ #
    # Bootstrap CI helpers for classification metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def bootstrap_accuracy_ci(
        correct: np.ndarray,
        n_boot: int = 5000,
        ci: float = 0.95,
    ) -> dict:
        """Compute bootstrap confidence interval for top-1 accuracy.

        Args:
            correct: Binary array (1 = correct, 0 = incorrect) of length N.
            n_boot:  Number of bootstrap resamples.
            ci:      Confidence level (default 0.95).

        Returns:
            dict with keys: accuracy, ci_lo, ci_hi.
        """
        rng = np.random.default_rng(seed=42)
        correct = np.asarray(correct, dtype=float)
        acc = float(correct.mean())
        boot_accs = np.array([
            rng.choice(correct, size=len(correct), replace=True).mean()
            for _ in range(n_boot)
        ])
        lo = float(np.percentile(boot_accs, 100 * (1 - ci) / 2))
        hi = float(np.percentile(boot_accs, 100 * (1 - (1 - ci) / 2)))
        return {"accuracy": acc, "ci_lo": lo, "ci_hi": hi}

    @staticmethod
    def bootstrap_metric_ci(
        values: np.ndarray,
        n_boot: int = 5000,
        ci: float = 0.95,
    ) -> dict:
        """Compute bootstrap CI for the mean of any scalar metric array.

        Suitable for Dice, IoU, Recall, Precision, AURC, etc.

        Args:
            values: 1-D array of per-sample metric values.
            n_boot: Number of bootstrap resamples.
            ci:     Confidence level (default 0.95).

        Returns:
            dict with keys: mean, std, ci_lo, ci_hi.
        """
        rng = np.random.default_rng(seed=42)
        values = np.asarray(values, dtype=float)
        mn  = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        boot_means = np.array([
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(n_boot)
        ])
        lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
        hi = float(np.percentile(boot_means, 100 * (1 - (1 - ci) / 2)))
        return {"mean": mn, "std": std, "ci_lo": lo, "ci_hi": hi}

    # ------------------------------------------------------------------ #

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (np.mean(group1) - np.mean(group2)) / pooled_std
