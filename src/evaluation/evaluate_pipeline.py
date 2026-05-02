import os
import sys
import numpy as np
import logging
import joblib
import pandas as pd

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

def log_label_distribution(name: str, labels: np.ndarray):
    vals, counts = np.unique(labels.astype(int), return_counts=True)
    logging.info(f"{name} label distribution: {dict(zip(vals, counts))}")

def stratified_sample(df: pd.DataFrame, label_col: str, total_samples: int, random_state: int = 42) -> pd.DataFrame:
    if label_col not in df.columns:
        logging.warning(f"Label column '{label_col}' missing for stratified sampling. Using full dataset.")
        return df

    value_counts = df[label_col].value_counts()
    classes = value_counts.index.tolist()
    if len(classes) == 0:
        return df

    per_class = max(1, total_samples // len(classes))
    parts = []
    for label in classes:
        group = df[df[label_col] == label]
        n = min(per_class, len(group))
        if n > 0:
            parts.append(group.sample(n=n, random_state=random_state))

    if not parts:
        return df

    sampled = pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logging.info(f"Stratified sample size: {len(sampled)} (target per class: {per_class})")
    return sampled

def run_evaluation_pipeline(real_sample_size: int = 50000):
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
    df_real = parser.load_data()
    
    preprocessor = DataPreprocessor()
    # Sample a large random chunk from full dataset for richer class coverage
    if real_sample_size and real_sample_size > 0:
        sample_size = min(real_sample_size, len(df_real))
        df_real = df_real.sample(n=sample_size, random_state=42).reset_index(drop=True)
    df_real_processed = preprocessor.fit_transform(df_real)

    artifact_dir = os.path.join(project_root, "models", "artifacts")
    feature_columns = joblib.load(os.path.join(artifact_dir, "feature_columns.pkl"))
    
    # Assuming label is the last column or we know its index. 
    # For TSTR, we need X (features) and y (target).
    label_col = 'Attack Type'
    if label_col not in feature_columns:
        logging.error("Label column 'Attack Type' missing from feature metadata.")
        return
    if list(df_real_processed.columns) != feature_columns:
        df_real_processed = df_real_processed[feature_columns]
    label_idx = feature_columns.index(label_col)
    
    desired_total = min(len(df_real_processed), len(synth_data))
    df_real_eval = stratified_sample(df_real_processed, label_col, desired_total)
    real_data = df_real_eval.values
    
    # Ensure shapes match (Truncate if necessary for comparison)
    min_samples = min(len(real_data), len(synth_data))
    real_data_eval = real_data[:min_samples]
    synth_data_eval = synth_data[:min_samples]
    
    # 3. Statistical & Visual Validation
    log_label_distribution("Synthetic", synth_data_eval[:, label_idx])
    log_label_distribution("Real", real_data_eval[:, label_idx])
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
