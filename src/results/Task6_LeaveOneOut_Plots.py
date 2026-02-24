#!/usr/bin/env python3
"""
Task6_LeaveOneOut_Plots.py

Leave-One-Out Sensitivity Analysis Visualization
Meta-analysis of screen time and depression risk

This script generates:
1. Leave-one-out influence plot (pooled OR vs omitted study)
2. Forest-style summary of leave-one-out pooled estimates

Author: Generated for systematic review analysis
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
INPUT_FILE = 'Task6_LeaveOneOut_Results.csv'
FULL_MODEL_OR = 1.3098  # Pooled OR from full 6-study model
FULL_MODEL_LOWER = 1.0994
FULL_MODEL_UPPER = 1.5606
DPI = 300

# =============================================================================
# Load Data
# =============================================================================
def load_loo_data(filepath):
    """Load leave-one-out results from CSV"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} leave-one-out scenarios")
    print(df.to_string(index=False))
    return df

# =============================================================================
# Plot 1: Leave-One-Out Influence Plot
# =============================================================================
def plot_loo_influence(df, full_or, full_lower, full_upper, output_prefix='loo_influence'):
    """
    Generate leave-one-out influence plot showing how pooled OR changes
    when each study is omitted.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Sort by pooled OR for better visualization
    df_sorted = df.sort_values('Pooled_OR', ascending=True).reset_index(drop=True)
    
    y_positions = np.arange(len(df_sorted))
    
    # Plot each LOO estimate with error bars
    for i, row in df_sorted.iterrows():
        # Error bar (95% CI)
        xerr_lower = row['Pooled_OR'] - row['Lower_CI']
        xerr_upper = row['Upper_CI'] - row['Pooled_OR']
        
        ax.errorbar(
            row['Pooled_OR'], 
            i,
            xerr=[[xerr_lower], [xerr_upper]],
            fmt='s',
            markersize=8,
            color='#2c7bb6',
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            markeredgecolor='black',
            markeredgewidth=0.5
        )
    
    # Reference line: Full model pooled OR
    ax.axvline(x=full_or, color='#d7191c', linestyle='--', linewidth=2, 
               label=f'Full Model OR = {full_or:.2f}')
    
    # Shaded region for full model 95% CI
    ax.axvspan(full_lower, full_upper, alpha=0.15, color='#d7191c',
               label=f'Full Model 95% CI [{full_lower:.2f}, {full_upper:.2f}]')
    
    # Reference line at OR = 1.0 (null effect)
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
               label='Null Effect (OR = 1.0)')
    
    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Omit: {s}" for s in df_sorted['Omitted_Study']], fontsize=10)
    
    # Log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlim(0.9, 2.0)
    
    # Formatting
    ax.set_xlabel('Pooled Odds Ratio (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Study Omitted', fontsize=12, fontweight='bold')
    ax.set_title('Leave-One-Out Sensitivity Analysis\nInfluence of Individual Studies on Pooled Effect',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # Add I² annotations on the right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f"I²={row['I_squared']:.1f}%" for _, row in df_sorted.iterrows()],
                        fontsize=9, color='#666666')
    ax2.tick_params(axis='y', length=0)
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f'{output_prefix}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_prefix}.pdf', dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_prefix}.png and {output_prefix}.pdf")
    
    plt.close()

# =============================================================================
# Plot 2: Forest-Style Leave-One-Out Summary
# =============================================================================
def plot_loo_forest(df, full_or, full_lower, full_upper, output_prefix='loo_forest'):
    """
    Generate forest-style summary plot of leave-one-out estimates
    with detailed annotations.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Add full model as first row
    n_studies = len(df)
    y_positions = np.arange(n_studies + 1, 0, -1)
    
    # Plot full model at top
    ax.errorbar(
        full_or, 
        n_studies + 1,
        xerr=[[full_or - full_lower], [full_upper - full_or]],
        fmt='D',
        markersize=10,
        color='#d7191c',
        capsize=5,
        capthick=2,
        elinewidth=2,
        markeredgecolor='black',
        markeredgewidth=1,
        label='Full Model (All 6 Studies)'
    )
    
    # Plot each LOO estimate
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_studies))
    
    for i, (idx, row) in enumerate(df.iterrows()):
        y_pos = n_studies - i
        xerr_lower = row['Pooled_OR'] - row['Lower_CI']
        xerr_upper = row['Upper_CI'] - row['Pooled_OR']
        
        ax.errorbar(
            row['Pooled_OR'], 
            y_pos,
            xerr=[[xerr_lower], [xerr_upper]],
            fmt='s',
            markersize=8,
            color=colors[i],
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            markeredgecolor='black',
            markeredgewidth=0.5
        )
    
    # Reference lines
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=full_or, color='#d7191c', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Y-axis labels
    y_labels = ['Full Model (n=6)'] + [f"Omit: {row['Omitted_Study']}" for _, row in df.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Log scale
    ax.set_xscale('log')
    ax.set_xlim(0.85, 2.1)
    
    # Add text annotations for OR [95% CI]
    for i, (idx, row) in enumerate(df.iterrows()):
        y_pos = n_studies - i
        text = f"{row['Pooled_OR']:.2f} [{row['Lower_CI']:.2f}, {row['Upper_CI']:.2f}]"
        ax.annotate(text, xy=(1.85, y_pos), fontsize=9, va='center', ha='left',
                    fontfamily='monospace')
    
    # Full model annotation
    text_full = f"{full_or:.2f} [{full_lower:.2f}, {full_upper:.2f}]"
    ax.annotate(text_full, xy=(1.85, n_studies + 1), fontsize=9, va='center', ha='left',
                fontfamily='monospace', fontweight='bold', color='#d7191c')
    
    # Column header
    ax.annotate('OR [95% CI]', xy=(1.85, n_studies + 1.8), fontsize=10, va='center', 
                ha='left', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Pooled Odds Ratio (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Leave-One-Out Meta-Analysis: Forest Plot Summary\nScreen Time and Depression Risk',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    
    # Horizontal separator line after full model
    ax.axhline(y=n_studies + 0.5, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f'{output_prefix}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_prefix}.pdf', dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_prefix}.png and {output_prefix}.pdf")
    
    plt.close()

# =============================================================================
# Plot 3: Combined Summary with I² Analysis
# =============================================================================
def plot_loo_combined(df, full_or, full_lower, full_upper, output_prefix='loo_combined'):
    """
    Generate combined plot showing both pooled OR and I² changes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    studies = df['Omitted_Study'].tolist()
    x_pos = np.arange(len(studies))
    
    # =========================
    # Left panel: Pooled OR
    # =========================
    ax1.bar(x_pos, df['Pooled_OR'], width=0.6, color='#2c7bb6', 
            edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Error bars
    yerr_lower = df['Pooled_OR'] - df['Lower_CI']
    yerr_upper = df['Upper_CI'] - df['Pooled_OR']
    ax1.errorbar(x_pos, df['Pooled_OR'], yerr=[yerr_lower, yerr_upper],
                 fmt='none', color='black', capsize=4, capthick=1.5)
    
    # Reference lines
    ax1.axhline(y=full_or, color='#d7191c', linestyle='--', linewidth=2,
                label=f'Full Model OR = {full_or:.2f}')
    ax1.axhspan(full_lower, full_upper, alpha=0.1, color='#d7191c')
    ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Null Effect')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.replace(' ', '\n') for s in studies], fontsize=8, rotation=0)
    ax1.set_ylabel('Pooled Odds Ratio', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Study Omitted', fontsize=11, fontweight='bold')
    ax1.set_title('A. Pooled OR When Study Omitted', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0.9, 1.9)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # =========================
    # Right panel: I² statistic
    # =========================
    full_i2 = 79.10  # From full model
    
    colors = ['#fdae61' if i2 < full_i2 else '#abd9e9' for i2 in df['I_squared']]
    
    ax2.bar(x_pos, df['I_squared'], width=0.6, color=colors,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax2.axhline(y=full_i2, color='#d7191c', linestyle='--', linewidth=2,
                label=f'Full Model I² = {full_i2:.1f}%')
    ax2.axhline(y=75, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                label='High Heterogeneity (75%)')
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.replace(' ', '\n') for s in studies], fontsize=8, rotation=0)
    ax2.set_ylabel('I² Statistic (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Study Omitted', fontsize=11, fontweight='bold')
    ax2.set_title('B. Heterogeneity When Study Omitted', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add text annotations
    for i, i2 in enumerate(df['I_squared']):
        ax2.annotate(f'{i2:.1f}%', xy=(i, i2 + 2), ha='center', fontsize=8)
    
    plt.suptitle('Leave-One-Out Sensitivity Analysis Summary', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f'{output_prefix}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_prefix}.pdf', dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_prefix}.png and {output_prefix}.pdf")
    
    plt.close()

# =============================================================================
# Main Execution
# =============================================================================
def main():
    print("="*60)
    print("LEAVE-ONE-OUT SENSITIVITY ANALYSIS VISUALIZATION")
    print("="*60)
    
    # Load data
    df = load_loo_data(INPUT_FILE)
    
    print("\n" + "-"*60)
    print("Generating plots...")
    print("-"*60)
    
    # Generate plots
    plot_loo_influence(df, FULL_MODEL_OR, FULL_MODEL_LOWER, FULL_MODEL_UPPER, 
                       'Task6_LOO_Influence')
    
    plot_loo_forest(df, FULL_MODEL_OR, FULL_MODEL_LOWER, FULL_MODEL_UPPER,
                    'Task6_LOO_Forest')
    
    plot_loo_combined(df, FULL_MODEL_OR, FULL_MODEL_LOWER, FULL_MODEL_UPPER,
                      'Task6_LOO_Combined')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Summary statistics
    print("\nSummary:")
    print(f"  Full Model Pooled OR: {FULL_MODEL_OR:.4f} [{FULL_MODEL_LOWER:.4f}, {FULL_MODEL_UPPER:.4f}]")
    print(f"  LOO Pooled OR Range: {df['Pooled_OR'].min():.4f} - {df['Pooled_OR'].max():.4f}")
    print(f"  LOO I² Range: {df['I_squared'].min():.2f}% - {df['I_squared'].max():.2f}%")
    
    # Identify most influential study
    max_change_idx = (df['Pooled_OR'] - FULL_MODEL_OR).abs().idxmax()
    most_influential = df.loc[max_change_idx, 'Omitted_Study']
    print(f"  Most Influential Study: {most_influential}")

if __name__ == '__main__':
    main()
