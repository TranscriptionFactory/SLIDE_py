import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

class Plotter:
    def __init__(self):
        pass
    
    @staticmethod
    def plot_latent_factors(lfs, loadings, outdir=None, title='Significant Latent Factors'):
        """
        Plot genes for each latent factor, colored by their sign.
        
        Parameters:
        - lfs: Dictionary where keys are latent factor names and values are dictionaries
               containing 'pos' and 'neg' lists of genes
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """
        color_dict = {'pos': 'red', 'neg': 'blue'}  
        marker_dict = {'pos': '', 'neg': ''}  
        
        # Calculate optimal figure size based on number of latent factors
        n_lfs = len(lfs)
        fig_width = min(20, max(8, n_lfs * 2))  # Width between 8 and 20 inches
        fig_height = min(15, max(6, n_lfs * 1.5))  # Height between 6 and 15 inches
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Plot each latent factor
        for i, (lf_name, genes_dict) in enumerate(lfs.items()):
            loading = loadings[lf_name]
            lf_genes = lfs[lf_name]['pos'].tolist() + lfs[lf_name]['neg'].tolist()
            lf_genes = np.abs(loading[lf_genes]).sort_values(ascending=False).index

            # Add latent factor label at the top
            ax.text(i, -0.1, lf_name, ha='center', va='center', 
                   fontsize=10, fontweight='bold')

            # Calculate optimal vertical spacing based on number of genes
            n_genes = len(lf_genes)
            spacing = min(0.1, max(0.01, 1.0/n_genes))  # Spacing between 0.02 and 0.1

            for j, gene in enumerate(lf_genes):
                color = color_dict['pos'] if loading[gene] > 0 else color_dict['neg']
                marker = marker_dict['pos'] if loading[gene] > 0 else marker_dict['neg']
                
                # Plot gene names with markers
                ax.text(i, j * spacing, f"{marker} {gene}",
                       ha='center', va='center',
                       fontsize=8,
                       color=color,
                       fontweight='medium')
        
        # Turn off axes
        ax.axis('off')
        
        # # Add legend
        # legend_elements = [
        #     plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=color_dict['pos'],
        #               markersize=10, label='Positive'),
        #     plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=color_dict['neg'],
        #               markersize=10, label='Negative')
        # ]
        # ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=8)
        
        # Set x-axis limits to push latent factors inward
        ax.set_xlim(-0.5, len(lfs) - 0.5)
        
        # Adjust layout with more padding
        plt.tight_layout(pad=1.5)
        
        # Save or return
        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    @staticmethod
    def plot_scores(scores, outdir=None, title='AUC'):
        """
        Plot scores for each latent factor using seaborn boxplot.
        
        Parameters:
        - scores: Dictionary where keys are latent factor names and values are lists of scores
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """
        # Convert dictionary to long format DataFrame
        data = []
        for lf_name, score_list in scores.items():
            for score in score_list:
                data.append({'Latent Factor': lf_name, 'Score': score})
        df = pd.DataFrame(data)
        
        # Set style
        plt.style.use('ggplot')
        
        # Create figure with custom style
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create boxplot with custom colors and style
        palette = sns.color_palette("husl", len(scores))
        sns.boxplot(data=df, x='Latent Factor', y='Score', ax=ax, 
                   palette=palette, width=0.6, fliersize=3)
        
        # Add individual points with jitter
        sns.stripplot(data=df, x='Latent Factor', y='Score', 
                     size=4, alpha=0.3, color='black', jitter=True)
        
        # Add mean values above boxes
        means = df.groupby('Latent Factor')['Score'].mean()
        for i, (lf, mean) in enumerate(means.items()):
            ax.text(i, mean-0.1, f'{mean:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Customize plot appearance
        ax.set_title(title, fontsize=12, pad=15, fontweight='bold')
        ax.set_xlabel('Latent Factor', fontsize=10, labelpad=10)
        ax.set_ylabel('Score', fontsize=10, labelpad=10)
        ax.set_ylim(0, 1)
        
        # Customize grid and spines
        ax.grid(True, linestyle='--', alpha=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or return
        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            return fig
