import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import os
import numpy as np
import pandas as pd
import seaborn.objects as so


class Plotter:
    def __init__(self):
        pass
    
    @staticmethod
    def plot_latent_factors(lfs, loadings, outdir=None, title='Significant Latent Factors'):
        """
        Plot genes for each latent factor, colored by their sign and ordered by absolute loading values.
        
        Parameters:
        - lfs: Dictionary where keys are latent factor names and values are dictionaries
               containing 'positive' and 'negative' lists of genes
        - loadings: Dictionary of gene loadings for each latent factor
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """
        # Set up colors and style
        colors = {'pos': '#FF4B4B', 'neg': '#4B4BFF'}  # Bright red and blue
        plt.style.use('default')
        
        # Calculate dimensions
        n_lfs = len(lfs)
        max_genes = max(len(lf['pos']) + len(lf['neg']) for lf in lfs.values())
        fig_width = min(20, max(10, n_lfs * 1.5))
        fig_height = min(15, max(8, max_genes * 0.3))
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Plot each latent factor
        for i, (lf_name, genes_dict) in enumerate(lfs.items()):
            # Combine and sort genes by absolute loading value
            all_genes = genes_dict['pos'] + genes_dict['neg']
            sorted_genes = sorted(all_genes, 
                                key=lambda x: abs(loadings[lf_name][x]), 
                                reverse=True)
            
            # Plot genes
            for j, gene in enumerate(sorted_genes):
                # Determine color based on sign
                color = colors['pos'] if loadings[lf_name][gene] > 0 else colors['neg']
                
                # Add gene name with background
                ax.text(i, j, gene, 
                       color=color,
                       fontsize=10,
                       fontweight='bold',
                       ha='center',
                       va='center',
                       bbox=dict(facecolor='white',
                               edgecolor='lightgray',
                               alpha=0.7,
                               boxstyle='round,pad=0.3'))
        
        # Customize plot appearance
        ax.set_title(title.replace('_', ' '), pad=20, fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, n_lfs - 0.5)
        ax.set_ylim(-1, max_genes)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add grid and customize ticks
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks(range(n_lfs))
        ax.set_xticklabels(lfs.keys(), ha='center')
        ax.set_yticks([])
        
        # Add legend
        # legend_elements = [
        #     Patch(facecolor=colors['pos'], label='Positive Loading', alpha=0.7),
        #     Patch(facecolor=colors['neg'], label='Negative Loading', alpha=0.7)
        # ]
        # ax.legend(
        #     handles=legend_elements, 
        #     loc='best'
        # )
        
        # Adjust layout and save
        plt.tight_layout()
        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white')
        return fig
    
    @staticmethod
    def plot_scores(scores, outdir=None, title='AUC'):
        """
        Plot performance scores for different latent factor configurations using a boxplot.
        
        Parameters:
        - scores: Dictionary where keys are latent factor configurations (e.g., 'z_matrix', 'marginals', 'marginals&interactions')
                 and values are lists of performance scores (e.g., AUC values)
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """
        # Convert dictionary to DataFrame for plotting
        color_dict = {
            'X': '#c65999',
            'z_matrix': '#7aa456',
            's1': '#777acd',
            's2': '#777acd',
            's3': '#777acd',
            's1_random': '#c96d44',
            's2_random': '#c96d44',
            's3_random': '#c96d44',
        }

        data = []
        for config, score_list in scores.items():
            for score in score_list:
                data.append({'Configuration': config, 'Score': score})
        df = pd.DataFrame(data)
        
        # Create figure with white background
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Create boxplot with custom colors for each configuration
        for i, config in enumerate(df['Configuration'].unique()):
            config_data = df[df['Configuration'] == config]
            sns.boxplot(data=config_data, x='Configuration', y='Score', ax=ax,
                       color=color_dict[config], width=0.6, fliersize=3)
        
        # Add individual points with jitter
        sns.stripplot(data=df, x='Configuration', y='Score',
                     size=4, alpha=0.3, color='black', jitter=True)
        
        # Add mean values above boxes
        means = df.groupby('Configuration')['Score'].mean()
        for i, config in enumerate(df['Configuration'].unique()):
            mean = means[config]
            ax.text(i, 0.3, f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize plot appearance
        ax.set_title(f'Performance plot', fontsize=14, pad=15, fontweight='bold')
        ax.set_ylabel('AUC', fontsize=12, labelpad=10)
        ax.set_xlabel('Group', fontsize=12, labelpad=10)
        ax.set_ylim(0, 1.1) 
        
        # Customize grid and spines
        ax.grid(True, linestyle='--', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Rotate x-axis labels for better readability
        plt.xticks(ha='center')
        plt.tight_layout()
        
        # Save plot if outdir is provided
        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
        return fig

    @staticmethod
    def plot_controlplot(scores, outdir=None, title='Control Plot'):
        """
        Plot control plot for different latent factor configurations.
        
        Parameters:
        - scores: Dictionary where keys are latent factor configurations (e.g., 'z_matrix', 'marginals', 'marginals&interactions')
                 and values are lists of performance scores (e.g., AUC values)
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """
        # Create figure with white background
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Plot density for s1_random and s2_random with filled area
        sns.kdeplot(scores['full_random'], label='full_random', ax=ax, fill=True, alpha=0.3, color='blue')
        sns.kdeplot(scores['partial_random'], label='partial_random', ax=ax, fill=True, alpha=0.3, color='green')
        
        # Add vertical line at s1 median
        # s1_median = np.median(scores['s1'])
        # ax.axvline(x=s1_median, color='purple', linestyle='--', label=f's1 median: {s1_median:.3f}', alpha=0.3)
        # s2_median = np.median(scores['s2'])
        # ax.axvline(x=s2_median, color='orange', linestyle='--', label=f's2 median: {s2_median:.3f}', alpha=0.3)
        s3_median = np.median(scores['s3'])
        ax.axvline(x=s3_median, color='red', linestyle='--', label=f's3 median: {s3_median:.3f}')
        
        # Customize plot appearance
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel('Score', fontsize=12, labelpad=10)
        ax.set_ylabel('Density', fontsize=12, labelpad=10)
        
        # Customize grid and spines
        ax.grid(True, linestyle='--', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.title(title)
        
        # Save plot if outdir is provided
        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
        return fig
