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
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot each latent factor
        for i, (lf_name, genes_dict) in enumerate(lfs.items()):
            loading = loadings[lf_name]
            lf_genes = lfs[lf_name]['pos'].tolist() + lfs[lf_name]['neg'].tolist()
            lf_genes = np.abs(loading[lf_genes]).sort_values(ascending=False).index

            # Add latent factor label at the top
            ax.text(i, -0.1, lf_name, ha='center', va='center', 
                   fontsize=12, fontweight='bold')

            for j, gene in enumerate(lf_genes):
                color = color_dict['pos'] if loading[gene] > 0 else color_dict['neg']
                marker = marker_dict['pos'] if loading[gene] > 0 else marker_dict['neg']
                
                # Plot gene names with markers - reduced vertical spacing
                ax.text(i, j * 0.04, f"{marker} {gene}",
                       ha='center', va='center',
                       fontsize=10,
                       color=color,
                       fontweight='medium')
        
        # Set title
        ax.text(-1, len(lfs)/2, title, rotation=90, ha='center', va='center', 
               fontsize=14, fontweight='bold')
        # Turn off axes
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=color_dict['pos'],
                      markersize=12, label='Positive'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=color_dict['neg'],
                      markersize=12, label='Negative')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
        
        # Set x-axis limits to push latent factors inward
        ax.set_xlim(-0.5, len(lfs) - 0.5)
        
        # Adjust layout with more padding
        plt.tight_layout(pad=2.0)
        
        # Save or return
        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    @staticmethod
    def plot_scores(scores, outdir=None, title='Scores'):
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
        
        # Create boxplot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.boxplot(data=df, x='Latent Factor', y='Score', ax=ax)
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel('Latent Factor')
        ax.set_ylabel('Score')
        plt.xticks(rotation=45)
        
        # Save or return
        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
