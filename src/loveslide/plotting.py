import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import os
import numpy as np
import pandas as pd
import seaborn.objects as so
import networkx as nx


class Plotter:
    def __init__(self):
        pass
    
    @staticmethod
    def plot_latent_factors(lfs, outdir=None, title='Significant Latent Factors'):
        """
        Plot genes for each latent factor, colored by their sign and ordered by absolute loading values.
        
        Parameters:
        - lfs: Dataframe where index is the latent factors name and columns are 'loading', 'AUC', 'corr', 'color'
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """
        # Set up colors and style
        colors = {'red': '#FF4B4B', 'gray': '#808080', 'blue': '#4B4BFF'}  # Bright red and blue
        plt.style.use('default')
        
        # Calculate dimensions
        n_lfs = len(lfs)
        fig_width = min(20, max(10, n_lfs * 2.5)) + 3 # for the title
        fig_height = min(18, max(5, n_lfs * 1.5))
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Plot each latent factor
        for i, (lf_name, lr_info) in enumerate(lfs.items()):
            # Reverse the order of genes (top to bottom, descending loading)
            lr_info = lr_info.sort_values(by='loading', ascending=True)

            # Plot genes
            for j, (gene, row) in enumerate(lr_info.iterrows()):
                # Determine color based on sign
                color = colors[row['color']] 

                ax.text(i, j, gene, 
                       color=color,
                       fontsize=24,  
                       fontweight='bold',
                       ha='center',
                       va='center')
        
        # Customize plot appearance
        ax.text(-0.5, 2, title.replace('_', ' '), 
                fontsize=14, fontweight='bold', 
                rotation=90, va='center')
        
        ax.set_xlim(-0.5, n_lfs - 0.5)
        ax.set_ylim(-1, fig_height)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add grid and customize ticks
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks(range(n_lfs))
        ax.set_xticklabels(lfs.keys(), ha='center')
        ax.set_yticks([])
        
        # Adjust layout and save
        plt.tight_layout()
        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white')
        return fig
    
    @staticmethod
    def plot_corr_network(X, lf_dict, outdir=None, minimum=0.25):

        colors = {'red': '#FF4B4B', 'gray': '#808080', 'blue': '#4B4BFF'}
        
        for lf, lf_loadings in lf_dict.items():
            lf_genes = lf_loadings.index.tolist()
            color_dict = lf_loadings['color'].map(colors).to_dict()
            
            features = X[lf_genes]
            corr = features.corr().where(lambda x: x > minimum, 0)
            np.fill_diagonal(corr.values, 0)

            G = nx.from_pandas_adjacency(corr)

            for gene in G.nodes():
                G.nodes[gene]['color'] = color_dict[gene]

            fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
            ax.grid(False)
            # pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
            pos = nx.shell_layout(G)
            nx.draw_networkx_nodes(G, pos, 
                                 node_color=[G.nodes[node]['color'] for node in G.nodes()],
                                 node_size=600,
                                 alpha=0.4,
                                 ax=ax)
            # Draw edges with alpha based on correlation strength
            for (node1, node2, data) in G.edges(data=True):
                weight = abs(data['weight'])
                nx.draw_networkx_edges(G, pos,
                                     edgelist=[(node1, node2)],
                                     width=weight*5,
                                     alpha=min(weight, 1.0),
                                     edge_color='gray')
                
            nx.draw_networkx_labels(G, pos, font_size=10)
            # nx.draw_networkx_edge_labels(G, pos, font_size=10)

            plt.tight_layout()
            plt.gca().set_aspect('equal')
            plt.savefig(os.path.join(outdir, f'corr_{lf}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')

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
        for score_type, color in [('full_random', 'blue'), ('partial_random', 'green'), ('s3', 'red')]:
            if len(set(scores[score_type])) == 1:  # If all values are the same
                ax.axvline(x=scores[score_type][0], color=color, label=f'{score_type}', linewidth=2)
            else:
                sns.kdeplot(scores[score_type], label=score_type, ax=ax, fill=True, alpha=0.3, color=color)
            
        s3_max = np.max([x for x in scores['s3'] if x is not None])
        ax.axvline(x=s3_max, color='red', linestyle='--', label=f's3 best: {s3_max:.3f}')
        
        # Customize plot appearance
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel('Score', fontsize=12, labelpad=10)
        ax.set_ylabel('Density', fontsize=12, labelpad=10)
        ax.set_xlim(-0.1, 1.1)
        
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
    