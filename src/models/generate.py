import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add necessary paths to import models and preprocessor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preprocessing'))

from generator import SequenceGenerator
from autoencoder import SequenceAutoencoder
from preprocessor import DataPreprocessor

def generate_synthetic_data(num_samples: int, seq_len: int = 20, noise_dim: int = 16, hidden_dim: int = 24, input_dim: int = 51):
    """
    Loads the trained PyTorch Generator and Autoencoder to generate synthetic 5G traffic,
    then uses the saved Scikit-Learn transformers to map the AI numbers back to readable strings/bytes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # 1. Load Preprocessor Artifacts (Scalers and Encoders)
    artifact_dir = os.path.join(project_root, "models", "artifacts")
    try:
        continuous_scalers = joblib.load(os.path.join(artifact_dir, 'continuous_scalers.pkl'))
        categorical_encoders = joblib.load(os.path.join(artifact_dir, 'categorical_encoders.pkl'))
        feature_columns = joblib.load(os.path.join(artifact_dir, 'feature_columns.pkl'))
        continuous_cols = joblib.load(os.path.join(artifact_dir, 'continuous_cols.pkl'))
        categorical_cols = joblib.load(os.path.join(artifact_dir, 'categorical_cols.pkl'))
        logging.info("Preprocessing artifacts loaded successfully.")
    except Exception as e:
        logging.error(f"Could not load scalers/encoders. Did you run the training/preprocessing pipeline first? Error: {e}")
        return

    num_classes = len(categorical_encoders['Attack Type'].classes_)
    input_dim = len(feature_columns)
    categorical_sizes = [len(categorical_encoders[col].classes_) for col in categorical_cols]
    if "Attack Type" not in feature_columns:
        logging.error("'Attack Type' column missing from feature metadata.")
        return
    attack_type_idx = feature_columns.index("Attack Type")

    # 2. Initialize Models
    autoencoder = SequenceAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        continuous_dim=len(continuous_cols),
        categorical_sizes=categorical_sizes
    ).to(device)
    generator = SequenceGenerator(noise_dim=noise_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    
    # 3. Load Trained Checkpoint
    checkpoint_path = os.path.join(project_root, "checkpoints", "latest_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        logging.error(f"No trained checkpoint found at {checkpoint_path}. Train the model first!")
        return
        
    logging.info(f"Loading trained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(checkpoint['autoencoder_state'])
    generator.load_state_dict(checkpoint['generator_state'])
    
    autoencoder.eval()
    generator.eval()
    
    # 4. Generate Fake Data!
    logging.info(f"Generating {num_samples} synthetic network flow sequences...")
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(num_samples, seq_len, noise_dim).to(device)
        
        # Balanced attack classes to avoid label collapse
        repeats = (num_samples + num_classes - 1) // num_classes
        labels = torch.arange(num_classes, device=device).repeat(repeats)[:num_samples]
        labels = labels[torch.randperm(num_samples, device=device)]
        
        # Pass through Generator to get the 24-Dimensional Latent sequence
        fake_latent_seqs = generator(z, labels)
        
        # Pass through Autoencoder Decoder to recover continuous + categorical logits
        fake_cont_seqs, fake_cat_logits = autoencoder.decoder(fake_latent_seqs)

        # Sample categorical features using Gumbel-Softmax
        fake_cat_indices = []
        for logits in fake_cat_logits:
            one_hot = F.gumbel_softmax(logits, tau=0.5, hard=True, dim=-1)
            fake_cat_indices.append(one_hot.argmax(dim=-1))

        # Reconstruct full feature tensor in original column order
        batch_size = fake_latent_seqs.size(0)
        seq_len = fake_latent_seqs.size(1)
        full_features = torch.zeros(
            batch_size,
            seq_len,
            len(feature_columns),
            device=fake_latent_seqs.device
        )

        if fake_cont_seqs is not None:
            for idx, col in enumerate(continuous_cols):
                col_idx = feature_columns.index(col)
                full_features[:, :, col_idx] = fake_cont_seqs[:, :, idx]

        for idx, col in enumerate(categorical_cols):
            col_idx = feature_columns.index(col)
            full_features[:, :, col_idx] = fake_cat_indices[idx].float()

        # Force Attack Type to match the conditioning label
        full_features[:, :, attack_type_idx] = labels.unsqueeze(1).expand(-1, seq_len).float()

    # 5. Inverse Transform (Convert back to Pandas DataFrame)
    # The neural network outputs 3D tensors (Samples, Seq_Len, Features)
    # We must flatten it to 2D (Samples * Seq_Len, Features) to save as a CSV
    flattened_fake_data = full_features.cpu().numpy().reshape(-1, input_dim)
    
    # Create DataFrame (We assume the column order matches how it was processed)
    # Note: For production, you should save & load the exact column names array during training.
    # Here we outline the concept using the preprocessor dictionaries.
    
    # Example logic to map the generated 0-1 values back to real Bytes and Strings
    # Implementation depends on exact column index order maintained during DataPreprocessor.
    
    logging.info(f"Generated Raw Tensor Shape: {flattened_fake_data.shape}")
    logging.info("To complete the pipeline, align the columns with saved metadata to inverse transform.")
    
    # 6. Save Raw Synthetic Output
    out_dir = os.path.join(project_root, "data", "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_raw_output.npy")
    np.save(out_path, flattened_fake_data)
    logging.info(f"Saved synthetic array to {out_path}")

if __name__ == "__main__":
    # Generate 100 sequences of 20 flows (2,000 fake flows total)
    generate_synthetic_data(num_samples=100)