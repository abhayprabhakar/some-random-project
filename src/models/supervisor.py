import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SequenceSupervisor(nn.Module):
    """
    Supervisor network for TimeGAN-style temporal coherence.
    Predicts the next latent step given the current latent sequence.
    """
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super(SequenceSupervisor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        # h: (batch, seq_len, hidden_dim)
        out, _ = self.rnn(h)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    batch_size = 16
    seq_len = 20
    hidden = 24

    dummy_latent = torch.randn(batch_size, seq_len, hidden)
    supervisor = SequenceSupervisor(hidden_dim=hidden)
    predicted = supervisor(dummy_latent)

    logging.info(f"Input latent shape: {dummy_latent.shape}")
    logging.info(f"Predicted latent shape: {predicted.shape}")
