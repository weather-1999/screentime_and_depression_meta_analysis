#!/usr/bin/env python3
"""
Meta-Analysis Visualization Script (OR-Only Primary Analysis)
==============================================================
Generates forest plot and funnel plot for screen time-depression meta-analysis.
EXCLUDES Kandola et al. (2021) due to IRR effect measure.
Uses DerSimonian-Laird random effects model for pooled estimate.

Author: Generated for systematic review
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sys
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_and_filter_data(filepath):
    """Load effect size data from CSV file and filter to OR-only studies."""
    df = pd.read_csv(filepath)
    
    # Filter to include ONLY OR effect sizes
    df_or = df[df['ES_Type'] == 'OR'].copy()
    
    # Check that we have exactly 8 OR studies
    n_or_studies = len(df_or)
    
    if n_or_studies != 8:
        print(f"ERROR: Expected 8 OR studies, but found {n_or_studies}.")
        print(f"Studies found with ES_Type == 'OR':")
        for idx, row in df_or.iterrows():
            print(f"  - {row['Study_ID']}")
        print(f"\nStudies excluded (non-OR):")
        df_excluded = df[df['ES_Type'] != 'OR']
        for idx, row in df_excluded.iterrows():
            print(f"  - {row['Study_ID']} (ES_Type: {row['ES_Type']})")
        sys.exit(1)
    
    return df_or

def calculate_log_es(df):
    """Convert effect sizes to log scale and calculate standard errors."""
    df = df.copy()
    
    # Log-transform effect sizes and confidence intervals
    df['log_ES'] = np.log(df['Final_ES'])
    df['log_Lower'] = np.log(df['Final_Lower'])
    df['log_Upper'] = np.log(df['Final_Upper'])
    
    # Calculate standard error from CI (assuming 95% CI)
    # SE = (log_Upper - log_Lower) / (2 * 1.96)
    df['SE'] = (df['log_Upper'] - df['log_Lower']) / (2 * 1.96)
    
    # Calculate variance and weights
    df['Variance'] = df['SE'] ** 2
    df['Weight_FE'] = 1 / df['Variance']
    
    return df

# =============================================================================
# Meta-Analysis Calculations (DerSimonian-Laird Random Effects)
# =============================================================================

def dersimonian_laird(log_es, se):
    """
    Perform DerSimonian-Laird random effects meta-analysis.
    
    Parameters:
    -----------
    log_es : array-like
        Log-transformed effect sizes
    se : array-like
        Standard errors of log effect sizes
    
    Returns:
    --------
    dict : Contains pooled estimate, CI, tau2, I2, Q statistic
    """
    log_es = np.array(log_es)
    se = np.array(se)
    k = len(log_es)
    
    # Fixed effects weights
    w = 1 / (se ** 2)
    
    # Fixed effects pooled estimate
    pooled_fe = np.sum(w * log_es) / np.sum(w)
    
    # Cochran's Q statistic
    Q = np.sum(w * (log_es - pooled_fe) ** 2)
    
    # Degrees of freedom
    df = k - 1
    
    # Calculate tau-squared (between-study variance)
    c = np.sum(w) - (np.sum(w ** 2) / np.sum(w))
    
    if Q > df:
        tau2 = (Q - df) / c
    else:
        tau2 = 0
    
    # Random effects weights
    w_re = 1 / (se ** 2 + tau2)
    
    # Random effects pooled estimate
    pooled_re = np.sum(w_re * log_es) / np.sum(w_re)
    
    # Standard error of pooled estimate
    se_pooled = np.sqrt(1 / np.sum(w_re))
    
    # 95% CI for pooled estimate
    ci_lower = pooled_re - 1.96 * se_pooled
    ci_upper = pooled_re + 1.96 * se_pooled
    
    # I-squared (percentage of variability due to heterogeneity)
    if Q > 0:
        I2 = max(0, (Q - df) / Q * 100)
    else:
        I2 = 0
    
    # P-value for Q statistic (chi-squared distribution)
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(Q, df)
    
    return {
        'pooled_log': pooled_re,
        'pooled_es': np.exp(pooled_re),
        'ci_lower_log': ci_lower,
        'ci_upper_log': ci_upper,
        'ci_lower': np.exp(ci_lower),
        'ci_upper': np.exp(ci_upper),
        'se_pooled': se_pooled,
        'tau2': tau2,
        'tau': np.sqrt(tau2),
        'I2': I2,
        'Q': Q,
        'df': df,
        'p_value': p_value
    }

# =============================================================================
# Forest Plot
# =============================================================================

def create_forest_plot(df, meta_results, output_prefix):
    """
    Create publication-quality forest plot for OR-only studies.
    
    Parameters:
    -----------
    df : DataFrame
        Study data with log-transformed effect sizes (OR only)
    meta_results : dict
        Results from meta-analysis
    output_prefix : str
        Prefix for output file names
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    n_studies = len(df)
    y_positions = np.arange(n_studies, 0, -1)
    
    # Plot individual studies
    for i, (idx, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        
        # Point estimate (square, size proportional to weight)
        weight_normalized = row['Weight_FE'] / df['Weight_FE'].max()
        marker_size = 50 + 150 * weight_normalized
        
        ax.scatter(row['Final_ES'], y, s=marker_size, c='#2c3e50', 
                   marker='s', zorder=3, edgecolors='black', linewidths=0.5)
        
        # Confidence interval (horizontal line)
        ax.hlines(y, row['Final_Lower'], row['Final_Upper'], 
                  colors='#2c3e50', linewidth=1.5, zorder=2)
        
        # Whiskers at CI ends
        whisker_height = 0.15
        ax.vlines(row['Final_Lower'], y - whisker_height, y + whisker_height,
                  colors='#2c3e50', linewidth=1.5, zorder=2)
        ax.vlines(row['Final_Upper'], y - whisker_height, y + whisker_height,
                  colors='#2c3e50', linewidth=1.5, zorder=2)
    
    # Pooled estimate (diamond)
    pooled_y = 0
    pooled_es = meta_results['pooled_es']
    pooled_lower = meta_results['ci_lower']
    pooled_upper = meta_results['ci_upper']
    
    diamond_height = 0.4
    diamond = Polygon([
        (pooled_lower, pooled_y),
        (pooled_es, pooled_y + diamond_height),
        (pooled_upper, pooled_y),
        (pooled_es, pooled_y - diamond_height)
    ], closed=True, facecolor='#e74c3c', edgecolor='#c0392b', linewidth=1.5, zorder=4)
    ax.add_patch(diamond)
    
    # Reference line at OR = 1
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Y-axis labels (study names)
    study_labels = df['Study_ID'].tolist()
    study_labels.append('Pooled (Random Effects)')
    all_y_positions = list(y_positions) + [pooled_y]
    
    ax.set_yticks(all_y_positions)
    ax.set_yticklabels(study_labels, fontsize=11)
    
    # X-axis (log scale)
    ax.set_xscale('log')
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12, fontweight='bold')
    
    # Set x-axis limits and ticks
    ax.set_xlim(0.5, 3.0)
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xticklabels(['0.5', '0.75', '1.0', '1.5', '2.0', '2.5', '3.0'])
    
    # Grid
    ax.xaxis.grid(True, alpha=0.3, linestyle='-')
    ax.yaxis.grid(False)
    
    # Y-axis limits
    ax.set_ylim(-1.5, n_studies + 1)
    
    # Add effect size annotations on the right
    ax.text(3.2, n_studies + 0.5, 'OR [95% CI]', fontsize=10, fontweight='bold',
            ha='left', va='center', transform=ax.get_xaxis_transform())
    
    for i, (idx, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        text = f"{row['Final_ES']:.2f} [{row['Final_Lower']:.2f}, {row['Final_Upper']:.2f}]"
        ax.text(3.2, y, text, fontsize=9, ha='left', va='center',
                transform=ax.get_xaxis_transform())
    
    # Pooled estimate text
    pooled_text = f"{pooled_es:.2f} [{pooled_lower:.2f}, {pooled_upper:.2f}]"
    ax.text(3.2, pooled_y, pooled_text, fontsize=9, fontweight='bold', 
            ha='left', va='center', transform=ax.get_xaxis_transform())
    
    # Title (indicating OR-only and Kandola exclusion)
    title = ('Forest Plot: Screen Time and Depression Risk\n'
             'Primary analysis (OR only); Kandola et al. (2021) excluded due to IRR')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # Heterogeneity annotation
    het_text = (f"Heterogeneity: Q = {meta_results['Q']:.2f}, df = {meta_results['df']}, "
                f"p = {meta_results['p_value']:.3f}\n"
                f"I² = {meta_results['I2']:.1f}%, τ² = {meta_results['tau2']:.4f}")
    
    ax.text(0.02, 0.02, het_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    # Direction labels
    ax.text(0.6, -1.2, '← Lower Risk', fontsize=9, ha='center', 
            style='italic', color='#27ae60')
    ax.text(2.0, -1.2, 'Higher Risk →', fontsize=9, ha='center', 
            style='italic', color='#c0392b')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Save plots
    plt.savefig(f'{output_prefix}_Forest_ORonly.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_prefix}_Forest_ORonly.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Forest plot saved: {output_prefix}_Forest_ORonly.png")
    print(f"Forest plot saved: {output_prefix}_Forest_ORonly.pdf")
    
    plt.close()

# =============================================================================
# Funnel Plot
# =============================================================================

def create_funnel_plot(df, meta_results, output_prefix):
    """
    Create funnel plot for publication bias assessment (OR-only studies).
    Note: Exploratory only due to small number of studies (n=8).
    Egger's test is NOT performed.
    
    Parameters:
    -----------
    df : DataFrame
        Study data with log-transformed effect sizes (OR only)
    meta_results : dict
        Results from meta-analysis
    output_prefix : str
        Prefix for output file names
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot individual studies
    ax.scatter(df['log_ES'], df['SE'], s=80, c='#3498db', 
               edgecolors='#2980b9', linewidths=1, alpha=0.8, zorder=3)
    
    # Add study labels
    for idx, row in df.iterrows():
        ax.annotate(row['Study_ID'], (row['log_ES'], row['SE']),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, alpha=0.7)
    
    # Pooled estimate vertical line
    ax.axvline(x=meta_results['pooled_log'], color='#e74c3c', 
               linestyle='-', linewidth=2, label='Pooled estimate', zorder=2)
    
    # Reference line at log(1) = 0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, 
               alpha=0.7, label='No effect (OR=1)', zorder=1)
    
    # Funnel shape (pseudo 95% CI)
    se_range = np.linspace(0, df['SE'].max() * 1.2, 100)
    pooled = meta_results['pooled_log']
    
    # Left and right boundaries of funnel
    left_bound = pooled - 1.96 * se_range
    right_bound = pooled + 1.96 * se_range
    
    ax.fill_betweenx(se_range, left_bound, right_bound, 
                     color='lightgray', alpha=0.3, label='95% CI region')
    ax.plot(left_bound, se_range, 'k--', alpha=0.5, linewidth=0.8)
    ax.plot(right_bound, se_range, 'k--', alpha=0.5, linewidth=0.8)
    
    # Invert y-axis (convention: smaller SE at top)
    ax.invert_yaxis()
    
    # Labels
    ax.set_xlabel('Log Odds Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Error', fontsize=12, fontweight='bold')
    
    # Title with caveat (indicating OR-only and Kandola exclusion)
    title = ('Funnel Plot: Screen Time and Depression Risk\n'
             'Primary analysis (OR only, n=8); Kandola et al. (2021) excluded due to IRR\n'
             '(EXPLORATORY - Small number of studies)')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Add secondary x-axis showing OR values
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    or_ticks = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    log_or_ticks = [np.log(x) for x in or_ticks]
    ax2.set_xticks(log_or_ticks)
    ax2.set_xticklabels([str(x) for x in or_ticks])
    ax2.set_xlabel('Odds Ratio', fontsize=10)
    
    # Legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-')
    
    # Caveat text box
    caveat_text = ("Note: This funnel plot is exploratory.\n"
                   "With only 8 studies, visual inspection for\n"
                   "publication bias has limited reliability.\n"
                   "Egger's test is NOT performed (requires ≥10 studies).")
    
    ax.text(0.98, 0.02, caveat_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                      alpha=0.8, edgecolor='orange'))
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{output_prefix}_Funnel_ORonly.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_prefix}_Funnel_ORonly.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Funnel plot saved: {output_prefix}_Funnel_ORonly.png")
    print(f"Funnel plot saved: {output_prefix}_Funnel_ORonly.pdf")
    
    plt.close()

# =============================================================================
# Summary Report
# =============================================================================

def print_summary(df, meta_results):
    """Print summary of meta-analysis results."""
    print("\n" + "="*70)
    print("META-ANALYSIS SUMMARY (OR-Only Primary Analysis)")
    print("="*70)
    print(f"\nNumber of studies included: {len(df)}")
    print(f"Exclusion: Kandola et al. (2021) - IRR effect measure")
    print(f"\nStudies included:")
    for idx, row in df.iterrows():
        print(f"  - {row['Study_ID']}: OR = {row['Final_ES']:.2f} "
              f"[{row['Final_Lower']:.2f}, {row['Final_Upper']:.2f}]")
    print(f"\nPooled Odds Ratio (Random Effects): {meta_results['pooled_es']:.3f}")
    print(f"95% Confidence Interval: [{meta_results['ci_lower']:.3f}, {meta_results['ci_upper']:.3f}]")
    print(f"\nHeterogeneity Statistics:")
    print(f"  - Cochran's Q: {meta_results['Q']:.3f} (df = {meta_results['df']}, p = {meta_results['p_value']:.4f})")
    print(f"  - I² (inconsistency): {meta_results['I2']:.1f}%")
    print(f"  - τ² (between-study variance): {meta_results['tau2']:.4f}")
    print(f"  - τ (between-study SD): {meta_results['tau']:.4f}")
    print("\n" + "="*70)
    
    # Interpretation
    if meta_results['I2'] < 25:
        het_interp = "low"
    elif meta_results['I2'] < 50:
        het_interp = "low-to-moderate"
    elif meta_results['I2'] < 75:
        het_interp = "moderate"
    else:
        het_interp = "high"
    
    print(f"\nInterpretation:")
    print(f"  - The pooled OR of {meta_results['pooled_es']:.2f} suggests a {(meta_results['pooled_es']-1)*100:.0f}% increase")
    print(f"    in odds of depression associated with higher screen time.")
    print(f"  - Heterogeneity is {het_interp} (I² = {meta_results['I2']:.1f}%).")
    print("="*70 + "\n")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to run the OR-only meta-analysis visualization."""
    
    # File paths
    input_file = r'C:\Users\pc\Desktop\recomp-src\results\Task2_Statistics_9_Studies.csv'
    output_prefix = r'C:\Users\pc\Desktop\recomp-src\results\Task2'
    
    print("="*70)
    print("OR-ONLY META-ANALYSIS VISUALIZATION")
    print("Excluding Kandola et al. (2021) due to IRR effect measure")
    print("="*70)
    
    print("\nLoading and filtering data (OR only)...")
    df = load_and_filter_data(input_file)
    
    print(f"\n✓ Successfully loaded {len(df)} OR-based studies")
    print("\nStudies included in analysis:")
    for idx, row in df.iterrows():
        print(f"  - {row['Study_ID']}: {row['ES_Type']} = {row['Final_ES']:.2f} "
              f"[{row['Final_Lower']:.2f}, {row['Final_Upper']:.2f}]")
    
    print("\nCalculating log-transformed effect sizes...")
    df = calculate_log_es(df)
    
    print("Performing DerSimonian-Laird random effects meta-analysis...")
    meta_results = dersimonian_laird(df['log_ES'], df['SE'])
    
    print_summary(df, meta_results)
    
    print("Generating forest plot (OR only)...")
    create_forest_plot(df, meta_results, output_prefix)
    
    print("Generating funnel plot (OR only)...")
    create_funnel_plot(df, meta_results, output_prefix)
    
    print("\n" + "="*70)
    print("✅ All visualizations complete!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - {output_prefix}_Forest_ORonly.png")
    print(f"  - {output_prefix}_Forest_ORonly.pdf")
    print(f"  - {output_prefix}_Funnel_ORonly.png")
    print(f"  - {output_prefix}_Funnel_ORonly.pdf")

if __name__ == "__main__":
    main()
