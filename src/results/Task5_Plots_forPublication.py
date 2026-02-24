#!/usr/bin/env python3
"""
Task 5: Meta-Analysis Visualization
Forest Plot and Funnel Plot for Screen Time and Depression Studies

This script generates:
1. A forest plot with pooled estimate and I-squared heterogeneity measure
2. A funnel plot (exploratory) for visual inspection of publication bias

Author: AI-assisted meta-analysis workflow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(filepath):
    """Load effect size data from CSV file."""
    df = pd.read_csv(filepath)
    return df

# =============================================================================
# META-ANALYSIS CALCULATIONS (DerSimonian-Laird Random Effects)
# =============================================================================

def calculate_se_from_ci(lower, upper):
    """Calculate standard error from 95% CI on log scale."""
    log_lower = np.log(lower)
    log_upper = np.log(upper)
    se = (log_upper - log_lower) / (2 * 1.96)
    return se

def random_effects_meta(effect_sizes, standard_errors):
    """
    Perform DerSimonian-Laird random effects meta-analysis.
    
    Parameters:
    -----------
    effect_sizes : array-like
        Log-transformed effect sizes (log OR)
    standard_errors : array-like
        Standard errors of log effect sizes
    
    Returns:
    --------
    dict : Contains pooled estimate, CI, weights, tau2, I2, Q
    """
    y = np.array(effect_sizes)
    se = np.array(standard_errors)
    k = len(y)
    
    # Fixed effects weights
    w_fe = 1 / (se ** 2)
    
    # Fixed effects pooled estimate
    theta_fe = np.sum(w_fe * y) / np.sum(w_fe)
    
    # Cochran's Q statistic
    Q = np.sum(w_fe * (y - theta_fe) ** 2)
    
    # Degrees of freedom
    df = k - 1
    
    # Calculate C
    C = np.sum(w_fe) - (np.sum(w_fe ** 2) / np.sum(w_fe))
    
    # Between-study variance (tau-squared)
    tau2 = max(0, (Q - df) / C)
    
    # Random effects weights
    w_re = 1 / (se ** 2 + tau2)
    
    # Random effects pooled estimate
    theta_re = np.sum(w_re * y) / np.sum(w_re)
    
    # Standard error of pooled estimate
    se_theta = np.sqrt(1 / np.sum(w_re))
    
    # 95% CI for pooled estimate
    ci_lower = theta_re - 1.96 * se_theta
    ci_upper = theta_re + 1.96 * se_theta
    
    # I-squared
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0
    
    # P-value for Q (chi-squared test)
    from scipy import stats
    p_heterogeneity = 1 - stats.chi2.cdf(Q, df)
    
    return {
        'pooled_log': theta_re,
        'pooled_or': np.exp(theta_re),
        'ci_lower_log': ci_lower,
        'ci_upper_log': ci_upper,
        'ci_lower_or': np.exp(ci_lower),
        'ci_upper_or': np.exp(ci_upper),
        'se_pooled': se_theta,
        'weights': w_re / np.sum(w_re) * 100,  # Percentage weights
        'tau2': tau2,
        'I2': I2,
        'Q': Q,
        'df': df,
        'p_heterogeneity': p_heterogeneity
    }

# =============================================================================
# FOREST PLOT
# =============================================================================

def create_forest_plot(df, meta_results, output_prefix):
    """
    Create a forest plot with individual study estimates and pooled estimate.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_studies = len(df)
    y_positions = np.arange(n_studies, 0, -1)
    
    # Plot individual studies
    for i, (idx, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        
        # Effect size point
        ax.plot(row['Final_ES'], y, 'ko', markersize=8)
        
        # Confidence interval line
        ax.hlines(y, row['Final_Lower'], row['Final_Upper'], colors='black', linewidth=1.5)
        
        # Whiskers
        ax.plot([row['Final_Lower'], row['Final_Lower']], [y - 0.1, y + 0.1], 'k-', linewidth=1)
        ax.plot([row['Final_Upper'], row['Final_Upper']], [y - 0.1, y + 0.1], 'k-', linewidth=1)
    
    # Plot pooled estimate as diamond
    pooled_y = 0
    pooled_or = meta_results['pooled_or']
    pooled_lower = meta_results['ci_lower_or']
    pooled_upper = meta_results['ci_upper_or']
    
    diamond_height = 0.3
    diamond = Polygon([
        (pooled_lower, pooled_y),
        (pooled_or, pooled_y + diamond_height),
        (pooled_upper, pooled_y),
        (pooled_or, pooled_y - diamond_height)
    ], closed=True, facecolor='darkblue', edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    
    # Reference line at OR = 1
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Set x-axis to log scale
    ax.set_xscale('log')
    
    # X-axis formatting
    ax.set_xlim(0.5, 12)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(['0.5', '1', '2', '4', '8'])
    ax.set_xlabel('Odds Ratio (log scale)', fontsize=12, fontweight='bold')
    
    # Y-axis labels
    y_labels = list(df['Study_ID']) + ['Pooled Estimate']
    ax.set_yticks(list(y_positions) + [pooled_y])
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-1, n_studies + 1)
    
    # Add effect size annotations on the right (aligned outside plot area)
    import matplotlib.transforms as mtransforms
    trans_right = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    # Header for the right column
    ax.text(1.02, y_positions[0] + 0.6, 'OR [95% CI]', transform=trans_right,
            fontsize=10, fontweight='bold', ha='left', va='bottom')

    # Per-study annotations
    for i, (idx, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        text = f"{row['Final_ES']:.2f} [{row['Final_Lower']:.2f}, {row['Final_Upper']:.2f}]"
        ax.text(1.02, y, text, transform=trans_right, fontsize=9, va='center', ha='left')

    # Pooled estimate annotation
    pooled_text = f"{pooled_or:.2f} [{pooled_lower:.2f}, {pooled_upper:.2f}]"
    ax.text(1.02, pooled_y, pooled_text, transform=trans_right, fontsize=9,
            fontweight='bold', va='center', ha='left')

    # Make room for the right-side text column
    plt.subplots_adjust(right=0.80)

    # Heterogeneity statistics
    het_text = (f"Heterogeneity: I² = {meta_results['I2']:.1f}%, "
                f"Q = {meta_results['Q']:.2f}, df = {meta_results['df']}, "
                f"p = {meta_results['p_heterogeneity']:.3f}\n"
                f"Random effects model (DerSimonian-Laird)")
    ax.text(0.02, -0.08, het_text, transform=ax.transAxes, fontsize=9, 
            style='italic', va='top')
    
    # Title
    ax.set_title('Forest Plot: Screen Time and Depression Risk\n(Prospective Cohort Studies)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add "Favors lower risk" and "Favors higher risk" labels
    ax.text(0.7, -0.12, '← Lower Risk', transform=ax.transAxes, fontsize=9, ha='center')
    ax.text(1.0, -0.12, 'Higher Risk →', transform=ax.transAxes, fontsize=9, ha='center')
    
    # Horizontal line separating pooled from individual studies
    ax.axhline(y=0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{output_prefix}_forest_plot_forPublication.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_forest_plot_forPublication.pdf', dpi=300, bbox_inches='tight')
    print(f"Forest plot saved: {output_prefix}_forest_plot_forPublication.png/pdf")
    
    plt.close()

# =============================================================================
# FUNNEL PLOT
# =============================================================================

def create_funnel_plot(df, meta_results, output_prefix):
    """
    Create a funnel plot for visual inspection of publication bias.
    Note: Exploratory only due to small number of studies.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate log effect sizes and standard errors
    log_es = np.log(df['Final_ES'])
    se = np.array([calculate_se_from_ci(row['Final_Lower'], row['Final_Upper']) 
                   for _, row in df.iterrows()])
    
    # Plot individual studies
    ax.scatter(log_es, se, s=100, c='darkblue', edgecolors='black', alpha=0.7, zorder=5)
    
    # Pooled estimate line
    pooled_log = meta_results['pooled_log']
    ax.axvline(x=pooled_log, color='darkblue', linestyle='-', linewidth=2, 
               label=f'Pooled estimate (log OR = {pooled_log:.3f})')
    
    # Reference line at log(1) = 0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='No effect (OR = 1)')
    
    # Pseudo 95% confidence limits (funnel)
    se_range = np.linspace(0.001, max(se) * 1.2, 100)
    lower_limit = pooled_log - 1.96 * se_range
    upper_limit = pooled_log + 1.96 * se_range
    
    ax.fill_betweenx(se_range, lower_limit, upper_limit, alpha=0.1, color='gray',
                     label='95% pseudo-confidence region')
    ax.plot(lower_limit, se_range, 'k--', linewidth=0.5, alpha=0.5)
    ax.plot(upper_limit, se_range, 'k--', linewidth=0.5, alpha=0.5)
    
    # Invert y-axis (SE = 0 at top)
    ax.invert_yaxis()
    
    # Labels
    ax.set_xlabel('Log Odds Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Error', fontsize=12, fontweight='bold')
    ax.set_title('Funnel Plot: Screen Time and Depression Risk\n(EXPLORATORY - Small number of studies)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add study labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.annotate(row['Study_ID'].split()[0], 
                    (log_es.iloc[i], se[i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    alpha=0.7)
    
    # Legend (place outside upper-right to avoid overlapping data / funnel shading)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.00),
              borderaxespad=0.0, fontsize=9, framealpha=0.9)

    # Warning note (place outside right side, below legend; avoids data and funnel shading)
    warning_text = ("NOTE: Funnel plot is exploratory only.\n"
                    "With only 6 studies, formal tests for publication bias\n"
                    "have low statistical power and are not recommended.")
    ax.text(1.02, 0.72, warning_text, transform=ax.transAxes, fontsize=9,
            style='italic', va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Leave space on the right for legend and note
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    
    # Save plots
    plt.savefig(f'{output_prefix}_funnel_plot_forPublication.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_funnel_plot_forPublication.pdf', dpi=300, bbox_inches='tight')
    print(f"Funnel plot saved: {output_prefix}_funnel_plot_forPublication.png/pdf")
    
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # File paths
    input_csv = r'C:\Users\pc\Desktop\recomp-src\results\Task5_Statistics_PubMed_Search.csv'
    output_prefix = r'C:\Users\pc\Desktop\recomp-src\results\Task5'
    
    print("="*60)
    print("META-ANALYSIS VISUALIZATION")
    print("Screen Time and Depression Risk - Prospective Cohort Studies")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data(input_csv)
    print(f"   Loaded {len(df)} studies")
    print(df.to_string(index=False))
    
    # Calculate log effect sizes and standard errors
    print("\n2. Calculating meta-analysis statistics...")
    log_es = np.log(df['Final_ES'])
    se = np.array([calculate_se_from_ci(row['Final_Lower'], row['Final_Upper']) 
                   for _, row in df.iterrows()])
    
    # Perform random effects meta-analysis
    meta_results = random_effects_meta(log_es, se)
    
    print(f"\n   POOLED ESTIMATE (Random Effects):")
    print(f"   OR = {meta_results['pooled_or']:.3f} "
          f"(95% CI: {meta_results['ci_lower_or']:.3f} - {meta_results['ci_upper_or']:.3f})")
    print(f"\n   HETEROGENEITY:")
    print(f"   I² = {meta_results['I2']:.1f}%")
    print(f"   Q = {meta_results['Q']:.2f}, df = {meta_results['df']}, p = {meta_results['p_heterogeneity']:.4f}")
    print(f"   τ² = {meta_results['tau2']:.4f}")
    
    print(f"\n   STUDY WEIGHTS:")
    for i, (idx, row) in enumerate(df.iterrows()):
        print(f"   {row['Study_ID']}: {meta_results['weights'][i]:.1f}%")
    
    # Create forest plot
    print("\n3. Creating forest plot...")
    create_forest_plot(df, meta_results, output_prefix)
    
    # Create funnel plot
    print("\n4. Creating funnel plot...")
    create_funnel_plot(df, meta_results, output_prefix)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {output_prefix}_forest_plot_forPublication.png")
    print(f"  - {output_prefix}_forest_plot_forPublication.pdf")
    print(f"  - {output_prefix}_funnel_plot_forPublication.png")
    print(f"  - {output_prefix}_funnel_plot_forPublication.pdf")

if __name__ == "__main__":
    main()
