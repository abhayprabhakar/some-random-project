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

def generate_synthetic_data(
    num_samples: int,
    seq_len: int = 20,
    noise_dim: int = 16,
    hidden_dim: int = 24,
    input_dim: int = 51,
    balanced_labels: bool = False,
):
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

    # Build a label distribution from the same raw sample used for training.
    # Matching the real label prior is usually better for downstream fidelity
    # than forcing a perfectly uniform synthetic class mix.
    label_probs = None
    if not balanced_labels:
        try:
            real_path = os.path.join(project_root, "data", "raw", "Combined.csv")
            parser = DataPreprocessor()
            df_real = pd.read_csv(real_path).head(100000).copy()
            df_real_processed = parser.fit_transform(df_real)
            attack_type_idx = df_real_processed.columns.get_loc('Attack Type')
            label_values, label_counts = np.unique(df_real_processed.iloc[:, attack_type_idx].astype(int).values, return_counts=True)
            label_probs = label_counts / label_counts.sum()
            label_values = label_values.astype(int)
            logging.info(f"Using real label prior for synthetic sampling: {dict(zip(label_values.tolist(), label_probs.round(4).tolist()))}")
        except Exception as e:
            logging.warning(f"Could not derive real label distribution; falling back to uniform labels. Error: {e}")
            label_probs = None

    num_classes = len(categorical_encoders['Attack Type'].classes_)
    input_dim = len(feature_columns)
    categorical_sizes = [len(categorical_encoders[col].classes_) for col in categorical_cols]

    # 2. Initialize Models
    autoencoder = SequenceAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        continuous_dim=len(continuous_cols),
        categorical_sizes=categorical_sizes
    ).to(device)
    generator = SequenceGenerator(noise_dim=noise_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    
    # 3. Load Trained Checkpoint
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    preferred_paths = [
        os.path.join(checkpoint_dir, "model_final.pt"),
        os.path.join(checkpoint_dir, "latest_checkpoint.pt"),
    ]
    checkpoint_path = next((path for path in preferred_paths if os.path.exists(path)), None)
    if checkpoint_path is None:
        logging.error(
            f"No trained checkpoint found in {checkpoint_dir}. Train the model first!"
        )
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
        
        # Use balanced class labels by default so synthetic data covers every attack type.
        # This is more useful for evaluation than drawing labels from a small random sample.
        if balanced_labels:
            base_labels = np.tile(np.arange(num_classes), int(np.ceil(num_samples / num_classes)))[:num_samples]
            np.random.shuffle(base_labels)
            labels = torch.LongTensor(base_labels).to(device)
        elif label_probs is not None:
            # Sample labels from the observed real class prior.
            labels_np = np.random.choice(label_values, size=num_samples, p=label_probs)
            labels = torch.LongTensor(labels_np).to(device)
        else:
            labels = torch.randint(0, num_classes, (num_samples,)).to(device)
        
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
    # Generate a larger synthetic set using the real class distribution by default.
    generate_synthetic_data(num_samples=2000, balanced_labels=False)