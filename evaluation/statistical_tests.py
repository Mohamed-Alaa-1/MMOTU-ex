import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, f_oneway, tukey_hsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy.stats import wilcoxon, f_oneway, tukey_hsd, mannwhitneyu

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
                except:
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
        One-way ANOVA across backbone mean ExBale scores.
        """
        # Combine all dfs
        combined_data = []
        n_per_backbone = {}
        for bb, df in all_results_dfs.items():
            if 'cam_threshold' in df.columns:
                df_thresh = df[np.isclose(df['cam_threshold'], cam_threshold)].copy()
            else:
                df_thresh = df

            # Drop na
            vals = df_thresh[metric].dropna().values
            combined_data.append(vals)
            n_per_backbone[bb] = len(vals)
            
        try:
            F, p = f_oneway(*combined_data)
        except:
            F, p = 0.0, 1.0
            
        # Flatten for Tukey
        flat_vals = []
        labels = []
        for bb, df in all_results_dfs.items():
            if 'cam_threshold' in df.columns:
                df_thresh = df[np.isclose(df['cam_threshold'], cam_threshold)].copy()
            else:
                df_thresh = df

            vals = df_thresh[metric].dropna().values
            flat_vals.extend(vals)
            labels.extend([bb] * len(vals))
            
        try:
            tukey = pairwise_tukeyhsd(endog=flat_vals, groups=labels, alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        except Exception as e:
            tukey_df = pd.DataFrame()
            
        return {
            "anova_p": p,
            "anova_F": F,
            "n_per_backbone": n_per_backbone,
            "cam_threshold": cam_threshold,
            "tukey_results_df": tukey_df
        }

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (np.mean(group1) - np.mean(group2)) / pooled_std
