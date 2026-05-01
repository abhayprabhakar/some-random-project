import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StatisticalValidators:
    """
    Computes statistical and visual distance metrics to evaluate 
    the fidelity of AI-generated synthetic 5G-NIDD network traffic.
    """
    
    @staticmethod
    def compute_ks_test(real_data: np.ndarray, fake_data: np.ndarray) -> dict:
        """
        Computes Kolmogorov-Smirnov test for continuous features.
        Checks if real and fake data come from the same distribution.
        """
        logging.info("Computing KS-Test across features...")
        num_features = real_data.shape[1]
        ks_scores = []
        p_values = []
        
        for i in range(num_features):
            res = ks_2samp(real_data[:, i], fake_data[:, i])
            ks_scores.append(res.statistic)
            p_values.append(res.pvalue)
            
        avg_ks = np.mean(ks_scores)
        logging.info(f"Average KS Distance: {avg_ks:.4f} (Closer to 0.0 is better)")
        return {'average_ks_dist': avg_ks, 'feature_ks_scores': ks_scores, 'p_values': p_values}

    @staticmethod
    def compute_js_divergence(real_data: np.ndarray, fake_data: np.ndarray) -> float:
        """
        Computes Jensen-Shannon divergence for categorical/discrete feature distributions.
        """
        logging.info("Computing Jensen-Shannon Divergence...")
        js_scores = []
        num_features = real_data.shape[1]
        
        for i in range(num_features):
            # Compute histograms (probability distributions)
            hist_real, bin_edges = np.histogram(real_data[:, i], bins=50, density=True)
            hist_fake, _ = np.histogram(fake_data[:, i], bins=bin_edges, density=True)
            
            # Add small epsilon to avoid division by zero
            hist_real = hist_real + 1e-10
            hist_fake = hist_fake + 1e-10
            
            js_score = distance.jensenshannon(hist_real, hist_fake)
            js_scores.append(js_score)
            
        avg_js = np.mean(js_scores)
        logging.info(f"Average JS Divergence: {avg_js:.4f} (Closer to 0.0 is better)")
        return avg_js

    @staticmethod
    def plot_pca(real_data: np.ndarray, fake_data: np.ndarray, output_dir: str):
        """
        Visually plots 2D Principal Component Analysis overlap.
        """
        logging.info("Generating PCA overlap plot...")
        pca = PCA(n_components=2)
        
        # We sample data if it's too large to keep plotting fast
        sample_size = min(5000, len(real_data), len(fake_data))
        real_sample = real_data[np.random.choice(real_data.shape[0], sample_size, replace=False)]
        fake_sample = fake_data[np.random.choice(fake_data.shape[0], sample_size, replace=False)]
        
        # Combine and apply PCA
        combined = np.concatenate((real_sample, fake_sample), axis=0)
        pca_result = pca.fit_transform(combined)
        
        real_pca = pca_result[:sample_size]
        fake_pca = pca_result[sample_size:]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.3, label='Real')
        plt.scatter(fake_pca[:, 0], fake_pca[:, 1], c='red', alpha=0.3, label='Synthetic')
        plt.title('PCA Overlap of Real vs Synthetic 5G Traffic')
        plt.legend()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pca_overlap.png'))
        plt.close()
        logging.info(f"PCA plot saved to {output_dir}/pca_overlap.png")

if __name__ == "__main__":
    # Quick Test Execution Using Dummy Data
    dummy_real = np.random.normal(loc=0.5, scale=0.1, size=(1000, 10))
    dummy_fake = np.random.normal(loc=0.52, scale=0.12, size=(1000, 10))
    
    validators = StatisticalValidators()
    validators.compute_ks_test(dummy_real, dummy_fake)
    validators.compute_js_divergence(dummy_real, dummy_fake)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    validators.plot_pca(dummy_real, dummy_fake, os.path.join(project_root, 'docs'))
