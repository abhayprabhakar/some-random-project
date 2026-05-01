import os
import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

sys.path.append(os.path.join(project_root, 'src', 'preprocessing'))
from data_parser import NIDDParser
from preprocessor import DataPreprocessor
from sequence_generator import SequenceGenerator

from validators import StatisticalValidators
from ml_efficacy import MLEfficacyTester

def run_evaluation_pipeline():
    """
    Loads empirical (real) data and the generated (synthetic) data,
    then runs the full suite of statistical, visual, and ML efficacy tests.
    """
    logging.info("--- PHASE 3: EVALUATION PIPELINE ---")
    
    # 1. Load Synthetic Data
    synth_path = os.path.join(project_root, "data", "synthetic", "synthetic_raw_output.npy")
    if not os.path.exists(synth_path):
        logging.error(f"Cannot find synthetic data at {synth_path}. Run generate.py first!")
        return
        
    synth_data = np.load(synth_path)
    logging.info(f"Loaded Synthetic Data. Shape: {synth_data.shape}")
    
    # 2. Load and Preprocess Real Data (Quick sample for comparison)
    logging.info("Loading Real 5G-NIDD Data for comparison...")
    real_path = os.path.join(project_root, "data", "raw", "Combined.csv")
    parser = NIDDParser(real_path)
    df_real = parser.load_data().head(10000) # Sample to match synthetic size roughly
    
    preprocessor = DataPreprocessor()
    df_real_processed = preprocessor.fit_transform(df_real)
    
    # Assuming label is the last column or we know its index. 
    # For TSTR, we need X (features) and y (target).
    label_col = 'Attack Type'
    label_idx = df_real_processed.columns.get_loc(label_col)
    
    real_data = df_real_processed.values
    
    # Ensure shapes match (Truncate if necessary for comparison)
    min_samples = min(len(real_data), len(synth_data))
    real_data_eval = real_data[:min_samples]
    synth_data_eval = synth_data[:min_samples]
    
    # 3. Statistical & Visual Validation
    validators = StatisticalValidators()
    
    # We compare features column by column in the latent/scaled space
    validators.compute_ks_test(real_data_eval, synth_data_eval)
    validators.compute_js_divergence(real_data_eval, synth_data_eval)
    
    # Generates pca_overlap.png in the docs folder!
    docs_dir = os.path.join(project_root, "docs")
    validators.plot_pca(real_data_eval, synth_data_eval, output_dir=docs_dir)
    
    # 4. Machine Learning Efficacy (TSTR)
    logging.info("--- TSTR (Train Synthetic, Test Real) BENCHMARK ---")
    
    # Split Features (X) and Labels (y)
    # We assume 'label_idx' is the target column
    X_synth = np.delete(synth_data_eval, label_idx, axis=1)
    y_synth = np.round(synth_data_eval[:, label_idx]).astype(int) # Clean AI outputs to integers
    
    X_real = np.delete(real_data_eval, label_idx, axis=1)
    y_real = real_data_eval[:, label_idx].astype(int)
    
    tester = MLEfficacyTester()
    tester.evaluate_tstr(X_synth, y_synth, X_real, y_real)
    
    logging.info(f"Evaluation complete! Check {docs_dir}/pca_overlap.png for the visual results.")

if __name__ == "__main__":
    run_evaluation_pipeline()
