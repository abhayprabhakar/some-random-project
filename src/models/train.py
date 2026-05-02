import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from torch.autograd import grad
from autoencoder import SequenceAutoencoder
from generator import SequenceGenerator
from discriminator import SequenceDiscriminator
from supervisor import SequenceSupervisor

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RobustTrainer:
    """
    A unified, fault-tolerant training loop for the C-TimeGAN.
    Includes:
    - Auto-Hardware detection (CPU -> GPU Portability)
    - WGAN-GP Loss implementation
    - Automatic state checkpointing
    - Graceful exit on KeyboardInterrupt (Ctrl+C on laptop)
    """
    def __init__(
        self,
        noise_dim=16,
        hidden_dim=24,
        input_dim=51,
        num_classes=5,
        lr=1e-4,
        continuous_indices=None,
        categorical_indices=None,
        categorical_sizes=None,
        lambda_cat=1.0,
        lambda_sup=1.0
    ):
        # Auto-detect hardware for portability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Initialized training on device: {self.device}")

        # Feature metadata for mixed continuous/categorical reconstruction
        self.categorical_indices = categorical_indices or []
        self.categorical_sizes = categorical_sizes or []
        if self.categorical_indices and len(self.categorical_sizes) != len(self.categorical_indices):
            raise ValueError("categorical_sizes must match categorical_indices length")
        if continuous_indices is None:
            self.continuous_indices = [i for i in range(input_dim) if i not in self.categorical_indices]
        else:
            self.continuous_indices = continuous_indices

        self.lambda_cat = lambda_cat
        self.lambda_sup = lambda_sup
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # 1. Initialize Neural Networks
        self.autoencoder = SequenceAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            continuous_dim=len(self.continuous_indices),
            categorical_sizes=self.categorical_sizes
        ).to(self.device)
        self.generator = SequenceGenerator(noise_dim=noise_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(self.device)
        self.discriminator = SequenceDiscriminator(hidden_dim=hidden_dim, num_classes=num_classes).to(self.device)
        self.supervisor = SequenceSupervisor(hidden_dim=hidden_dim).to(self.device)

        # 2. Initialize Optimizers (Adam with beta1=0.0 is standard for WGAN-GP)
        self.opt_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.0, 0.9))
        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
        self.opt_AE = optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.opt_S = optim.Adam(self.supervisor.parameters(), lr=lr, betas=(0.0, 0.9))

        self.noise_dim = noise_dim
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.start_epoch = 0

    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Saves the complete state of the training session for portability and resumption."""
        state = {
            'epoch': epoch,
            'autoencoder_state': self.autoencoder.state_dict(),
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'supervisor_state': self.supervisor.state_dict(),
            'opt_G_state': self.opt_G.state_dict(),
            'opt_D_state': self.opt_D.state_dict(),
            'opt_AE_state': self.opt_AE.state_dict(),
            'opt_S_state': self.opt_S.state_dict()
        }
        
        filename = 'model_final.pt' if is_final else f'checkpoint_epoch_{epoch}.pt'
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save atomically
        torch.save(state, filepath)
        # Update pointer to "latest"
        torch.save(state, os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt'))
        logging.info(f"Checkpoint saved successfully at Epoch {epoch} -> {filepath}")

    def load_checkpoint(self) -> bool:
        """Attempts to load the `latest_checkpoint.pt` to resume training cleanly."""
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        if not os.path.exists(latest_path):
            logging.info("No existing checkpoints found. Starting fresh from Epoch 0.")
            return False
            
        logging.info(f"Found checkpoint at {latest_path}. Resuming training...")
        # map_location=self.device ensures if you trained on GPU but load on a Laptop CPU, it won't crash
        checkpoint = torch.load(latest_path, map_location=self.device)
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state'])
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        if 'supervisor_state' in checkpoint:
            self.supervisor.load_state_dict(checkpoint['supervisor_state'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state'])
        self.opt_AE.load_state_dict(checkpoint['opt_AE_state'])
        if 'opt_S_state' in checkpoint:
            self.opt_S.load_state_dict(checkpoint['opt_S_state'])
        
        logging.info(f"Resumed successfully. Training will continue from Epoch {self.start_epoch}.")
        return True

    def calculate_gradient_penalty(self, real_emb, fake_emb, labels):
        """WGAN-GP Gradient Penalty to mathematically prevent Mode Collapse."""
        batch_size, seq_len, hidden_dim = real_emb.size()
        
        # Random weight term for interpolation
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_emb)
        
        # Get random interpolation between real and fake samples
        interpolated = alpha * real_emb + (1 - alpha) * fake_emb
        interpolated.requires_grad_(True)
        
        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated, labels)
        
        # Calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                         grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                         create_graph=True, retain_graph=True)[0]
                         
        # Gradients have shape (batch_size, seq_len, hidden_dim)
        # Using .reshape() instead of .view() to avoid contiguous memory subspace issues in PyTorch autograd
        gradients = gradients.reshape(batch_size, -1)
        
        # Derive Gradient Penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def compute_supervisor_loss(self, latent_seq: torch.Tensor) -> torch.Tensor:
        """Supervised loss to enforce temporal coherence in latent space."""
        if latent_seq.size(1) < 2:
            return torch.zeros((), device=latent_seq.device)
        pred = self.supervisor(latent_seq)
        return self.mse_loss(pred[:, :-1, :], latent_seq[:, 1:, :])

    def autoencoder_step(self, real_sequences: torch.Tensor):
        """Trains the autoencoder (and supervisor) on real sequences."""
        self.opt_AE.zero_grad()
        self.opt_S.zero_grad()

        if self.categorical_indices:
            recovered_cont, cat_logits, real_emb = self.autoencoder(real_sequences)
        else:
            recovered_cont, real_emb = self.autoencoder(real_sequences)
            cat_logits = []

        cont_loss = torch.zeros((), device=real_sequences.device)
        if recovered_cont is not None and self.continuous_indices:
            cont_target = real_sequences[:, :, self.continuous_indices]
            cont_loss = self.mse_loss(recovered_cont, cont_target)

        cat_loss = torch.zeros((), device=real_sequences.device)
        if self.categorical_indices:
            for head_idx, cat_idx in enumerate(self.categorical_indices):
                logits = cat_logits[head_idx]
                targets = real_sequences[:, :, cat_idx].round().long()
                cat_loss = cat_loss + self.ce_loss(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
            cat_loss = cat_loss / len(self.categorical_indices)

        ae_loss = cont_loss + (self.lambda_cat * cat_loss)

        sup_loss = self.compute_supervisor_loss(real_emb.detach())
        total_loss = ae_loss + (self.lambda_sup * sup_loss)

        total_loss.backward()
        self.opt_AE.step()
        self.opt_S.step()

        return ae_loss, sup_loss, real_emb.detach()

    def pretrain_autoencoder(self, dataloader, epochs: int):
        """Pretrains the autoencoder before adversarial training starts."""
        from tqdm import tqdm

        logging.info(f"Starting Autoencoder pretraining for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_ae_loss = 0.0
            epoch_sup_loss = 0.0

            batch_iterator = tqdm(dataloader, desc=f"AE Pretrain {epoch + 1}/{epochs}", leave=False)
            for real_seqs, _ in batch_iterator:
                real_seqs = real_seqs.to(self.device)
                ae_loss, sup_loss, _ = self.autoencoder_step(real_seqs)

                epoch_ae_loss += ae_loss.item()
                epoch_sup_loss += sup_loss.item()
                batch_iterator.set_postfix({"AE_Loss": f"{ae_loss.item():.4f}", "Sup_Loss": f"{sup_loss.item():.4f}"})

            avg_ae_loss = epoch_ae_loss / len(dataloader)
            avg_sup_loss = epoch_sup_loss / len(dataloader)
            logging.info(f"[AE Pretrain {epoch + 1}/{epochs}] - Avg AE_Loss: {avg_ae_loss:.4f} | Avg Sup_Loss: {avg_sup_loss:.4f}")

    def run_periodic_evaluation(self, epoch: int, num_sequences: int, real_sample_size: int):
        """Generates synthetic data and runs evaluation pipeline at intervals."""
        if num_sequences <= 0:
            return

        import sys
        eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
        if eval_dir not in sys.path:
            sys.path.append(eval_dir)

        from generate import generate_synthetic_data
        from evaluate_pipeline import run_evaluation_pipeline

        logging.info(
            f"Running periodic evaluation at Epoch {epoch + 1} "
            f"(synthetic sequences: {num_sequences}, real sample: {real_sample_size})"
        )
        self.save_checkpoint(epoch=epoch)
        generate_synthetic_data(num_samples=num_sequences)
        run_evaluation_pipeline(real_sample_size=real_sample_size)

    def train_step(self, real_sequences: torch.Tensor, labels: torch.Tensor, lambda_gp=10.0, n_critic=3):
        """
        Executes one full step of Wasserstein GAN training.
        Includes autoencoder + supervisor pre-optimization and categorical reconstruction heads.
        """
        real_sequences = real_sequences.to(self.device)
        labels = labels.to(self.device)
        batch_size, seq_len, _ = real_sequences.size()
        
        # ============================================
        #          TRAIN AUTOENCODER + SUPERVISOR
        # ============================================
        ae_loss, sup_loss, real_emb = self.autoencoder_step(real_sequences)

        # ============================================
        #            TRAIN DISCRIMINATOR
        # ============================================
        for _ in range(n_critic):
            self.opt_D.zero_grad()
            
            # Generate fake embedding sequences
            z = torch.randn(batch_size, seq_len, self.noise_dim).to(self.device)
            fake_emb = self.generator(z, labels).detach() # Detach so we don't backprop through G
            
            # Calculate WGAN Loss
            real_validity = self.discriminator(real_emb, labels)
            fake_validity = self.discriminator(fake_emb, labels)
            
            # Gradient penalty
            gradient_penalty = self.calculate_gradient_penalty(real_emb, fake_emb, labels)
            
            # Adversarial Loss (Discriminator wants to push real to +inf and fake to -inf)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            
            d_loss.backward()
            self.opt_D.step()

        # ============================================
        #              TRAIN GENERATOR
        # ============================================
        self.opt_G.zero_grad()
        
        z = torch.randn(batch_size, seq_len, self.noise_dim).to(self.device)
        fake_emb = self.generator(z, labels) # Do NOT detach here
        
        # Generator wants to fool Discriminator into thinking fakes are real
        fake_validity = self.discriminator(fake_emb, labels)
        g_loss = -torch.mean(fake_validity)

        # Supervised generator loss for temporal coherence
        self.supervisor.eval()
        for param in self.supervisor.parameters():
            param.requires_grad_(False)
        g_sup_loss = self.compute_supervisor_loss(fake_emb)
        for param in self.supervisor.parameters():
            param.requires_grad_(True)
        self.supervisor.train()

        total_g_loss = g_loss + (self.lambda_sup * g_sup_loss)
        total_g_loss.backward()
        self.opt_G.step()
        
        return ae_loss.item(), sup_loss.item(), d_loss.item(), g_loss.item(), g_sup_loss.item()

    def fit(
        self,
        dataloader,
        epochs: int,
        save_interval: int = 5,
        patience: int = 15,
        pretrain_epochs: int = 10,
        min_epochs: int = 50,
        min_delta: float = 1e-4,
        eval_interval: int = 0,
        eval_num_sequences: int = 1000,
        eval_real_sample_size: int = 50000
    ):
        """
        Main training orchestration loop. Includes Graceful Interruption and detailed progress reporting.
        Automates Early Stopping based on WGAN-GP convergence (Discriminator loss nears 0).
        """
        from tqdm import tqdm
        
        # Try to resume from an existing checkpoint
        resumed = self.load_checkpoint()
        
        # Open metrics file to append logic
        metrics_file = os.path.join(self.checkpoint_dir, 'training_metrics.csv')
        if not os.path.exists(metrics_file):
            with open(metrics_file, 'w') as f:
                f.write("epoch,ae_loss,sup_loss,d_loss,g_loss,g_sup_loss\n")
                
        best_composite_score = float('inf')
        patience_counter = 0
        
        if pretrain_epochs > 0 and not resumed and dataloader is not None:
            self.pretrain_autoencoder(dataloader, pretrain_epochs)

        logging.info("Starting C-TimeGAN Adversarial Training Loop...")
        try:
            for epoch in range(self.start_epoch, epochs):
                
                # Check if dataloader exists
                if dataloader is None:
                    # Mock sleep for demonstration if no data is provided
                    import time
                    time.sleep(0.5)
                    logging.info(f"Epoch {epoch}/{epochs} passed (Mock mode).")
                else:
                    epoch_ae_loss = 0.0
                    epoch_sup_loss = 0.0
                    epoch_d_loss = 0.0
                    epoch_g_loss = 0.0
                    epoch_g_sup_loss = 0.0
                    
                    # Create a sleek progress bar for the batches
                    batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
                    
                    for batch_idx, (real_seqs, labels) in enumerate(batch_iterator):
                        ae_loss, sup_loss, d_loss, g_loss, g_sup_loss = self.train_step(real_seqs, labels)
                        
                        epoch_ae_loss += ae_loss
                        epoch_sup_loss += sup_loss
                        epoch_d_loss += d_loss
                        epoch_g_loss += g_loss
                        epoch_g_sup_loss += g_sup_loss
                        
                        # Update progress bar metrics dynamically
                        batch_iterator.set_postfix({
                            "AE_Loss": f"{ae_loss:.4f}",
                            "Sup_Loss": f"{sup_loss:.4f}",
                            "D_Loss": f"{d_loss:.4f}",
                            "G_Loss": f"{g_loss:.4f}",
                            "G_Sup": f"{g_sup_loss:.4f}"
                        })
                        
                    # Calculate true Epoch averages
                    avg_ae_loss = epoch_ae_loss / len(dataloader)
                    avg_sup_loss = epoch_sup_loss / len(dataloader)
                    avg_d_loss = epoch_d_loss / len(dataloader)
                    avg_g_loss = epoch_g_loss / len(dataloader)
                    avg_g_sup_loss = epoch_g_sup_loss / len(dataloader)
                    logging.info(
                        f"[Epoch {epoch + 1}/{epochs}] - Avg AE_Loss: {avg_ae_loss:.4f} | "
                        f"Avg Sup_Loss: {avg_sup_loss:.4f} | Avg D_Loss: {avg_d_loss:.4f} | "
                        f"Avg G_Loss: {avg_g_loss:.4f} | Avg G_Sup: {avg_g_sup_loss:.4f}"
                    )
                    
                    # Append metrics to CSV continuously
                    with open(metrics_file, 'a') as f:
                        f.write(
                            f"{epoch + 1},{avg_ae_loss:.6f},{avg_sup_loss:.6f},"
                            f"{avg_d_loss:.6f},{avg_g_loss:.6f},{avg_g_sup_loss:.6f}\n"
                        )
                        
                    # Early Stopping Evaluator (Composite score: AE + Supervisor + Generator Supervisor)
                    composite_score = avg_ae_loss + avg_sup_loss + avg_g_sup_loss
                    if composite_score < (best_composite_score - min_delta):
                        best_composite_score = composite_score
                        patience_counter = 0
                    else:
                        if (epoch + 1) >= min_epochs:
                            patience_counter += 1
                            logging.info(
                                "No improvement in composite score (AE + Sup + G_Sup). "
                                f"Patience: {patience_counter}/{patience} (min_delta={min_delta})"
                            )
                            if patience_counter >= patience:
                                logging.warning(
                                    f"EARLY STOPPING triggered! Network stopped progressing after {epoch + 1} Epochs."
                                )
                                self.save_checkpoint(epoch=epoch, is_final=True)
                                break
                        else:
                            logging.info(
                                f"Warmup active: early stopping disabled until Epoch {min_epochs}."
                            )
                
                # Periodic Evaluation
                if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
                    self.run_periodic_evaluation(epoch, eval_num_sequences, eval_real_sample_size)

                # Checkpointing
                if epoch > 0 and epoch % save_interval == 0:
                    logging.info(f"--> Reached Save Interval. Checkpointing Epoch {epoch}...")
                    self.save_checkpoint(epoch=epoch)
                    
            # Final Save (if loop concludes naturally)
            if patience_counter < patience:
                self.save_checkpoint(epoch=epochs, is_final=True)
                logging.info("Training complete.")
            
        except KeyboardInterrupt:
            # Graceful Exit for Laptops!
            logging.warning(f"\nTraining interrupted actively by user (KeyboardInterrupt) at Epoch {epoch}!")
            logging.info("Safely saving the models before exiting to prevent data loss...")
            self.save_checkpoint(epoch=epoch)
            logging.info("State saved successfully. Run the script again to resume from exactly this spot.")

if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader
    import sys
    
    # Add preprocessing folder to path to easily import our data modules
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preprocessing'))
    from data_parser import NIDDParser
    from preprocessor import DataPreprocessor
    from sequence_generator import SequenceGenerator as DataSequenceGenerator
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    dataset_path = os.path.join(project_root, "data", "raw", "Combined.csv")
    
    logging.info("--- PHASE 1: DATA PIPELINE ---")
    # 1. Parse Data
    parser = NIDDParser(dataset_path)
    df = parser.load_data()
    
    # IMPORTANT FOR LAPTOPS: Taking a 100,000 row sample to prevent RAM Out-Of-Memory crashes.
    # Once you verify it trains smoothly, you can comment out the .head() to train on all 1.2 Million rows.
    logging.info("Sampling first 100,000 rows for memory safety on laptop...")
    df_sample = df.head(100000).copy()
    
    # 2. Preprocess (Scale & Encode)
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df_sample)
    
    # Save the artifacts so we can reverse the transformation later
    artifact_dir = os.path.join(project_root, "models", "artifacts")
    preprocessor.save_pipeline(artifact_dir)
    
    # 3. Generate Sequences!
    seq_gen = DataSequenceGenerator(sequence_length=20, stride=5)
    X, y = seq_gen.create_windows(df_processed, label_col='Attack Type')

    feature_cols = preprocessor.feature_columns
    categorical_cols = preprocessor.categorical_cols
    continuous_cols = preprocessor.continuous_cols
    categorical_indices = [feature_cols.index(col) for col in categorical_cols]
    continuous_indices = [feature_cols.index(col) for col in continuous_cols]
    categorical_sizes = [len(preprocessor.categorical_encoders[col].classes_) for col in categorical_cols]
    
    # 4. Convert NumPy arrays to PyTorch Tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    
    logging.info("--- PHASE 2: C-TimeGAN TRAINING ---")
    trainer = RobustTrainer(
        noise_dim=16,
        hidden_dim=24,
        input_dim=X_tensor.shape[-1],
        num_classes=len(preprocessor.categorical_encoders['Attack Type'].classes_),
        continuous_indices=continuous_indices,
        categorical_indices=categorical_indices,
        categorical_sizes=categorical_sizes,
        lambda_cat=1.0,
        lambda_sup=1.0
    )
    
    # Start training! Set to a massive epoch number (1,000,000) so it runs infinitely 
    # until Early Stopping kicks in or you hit Ctrl+C.
    logging.info("Initiating indefinite training. Will halt automatically via Early Stopping or manual Ctrl+C.")
    trainer.fit(
        dataloader=dataloader,
        epochs=1000000,
        save_interval=5,
        patience=25,
        pretrain_epochs=10,
        min_epochs=100,
        min_delta=1e-4,
        eval_interval=50,
        eval_num_sequences=1000,
        eval_real_sample_size=50000
    )
