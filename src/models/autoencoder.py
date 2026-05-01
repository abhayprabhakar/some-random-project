import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SequentialEncoder(nn.Module):
    """
    Encoder network that compresses a sequence of network flows into a lower-dimensional latent space.
    This helps the C-TimeGAN generate complex sequences without dealing with 50+ raw features.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super(SequentialEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM to process the time-series
        self.rnn = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Linear projection to finalize the embedding representation for each timestep
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # x is of shape (batch, seq_len, input_dim)
        out, _ = self.rnn(x)  
        # out shape: (batch, seq_len, hidden_dim)
        out = self.linear(out)
        out = self.activation(out)
        return out


class SequentialDecoder(nn.Module):
    """
    Decoder network that maps latent embeddings back to continuous features and categorical logits.
    """
    def __init__(self, hidden_dim: int, continuous_dim: int, categorical_sizes=None, num_layers: int = 2):
        super(SequentialDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.continuous_dim = continuous_dim
        self.categorical_sizes = categorical_sizes or []
        self.has_categorical = len(self.categorical_sizes) > 0
        
        self.rnn = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.continuous_head = None
        if self.continuous_dim > 0:
            self.continuous_head = nn.Linear(hidden_dim, continuous_dim)
        
        self.categorical_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, size) for size in self.categorical_sizes]
        )

    def forward(self, x):
        # x is of shape (batch, seq_len, hidden_dim)
        out, _ = self.rnn(x)
        
        cont_out = None
        if self.continuous_head is not None:
            # Continuous features are MinMax scaled to [0, 1]
            cont_out = torch.sigmoid(self.continuous_head(out))
        
        cat_logits = [head(out) for head in self.categorical_heads]
        return cont_out, cat_logits


class SequenceAutoencoder(nn.Module):
    """
    Combines the SequentialEncoder and SequentialDecoder.
    Used purely to train the embedding/recovery mappings before connecting the GAN.
    """
    def __init__(self, input_dim: int, hidden_dim: int, continuous_dim: int, categorical_sizes=None, num_layers: int = 2):
        super(SequenceAutoencoder, self).__init__()
        self.encoder = SequentialEncoder(input_dim, hidden_dim, num_layers)
        self.decoder = SequentialDecoder(hidden_dim, continuous_dim, categorical_sizes, num_layers)
        self.has_categorical = bool(categorical_sizes)
        
    def forward(self, x):
        latent_h = self.encoder(x)
        cont_out, cat_logits = self.decoder(latent_h)
        if self.has_categorical:
            return cont_out, cat_logits, latent_h
        return cont_out, latent_h

if __name__ == "__main__":
    # Quick Test Execution
    batch_size = 32
    seq_len = 20
    features = 51
    hidden = 24  # Compressing 51 features down to an embedding space of 24
    
    # 1. Create dummy input tensor matching what the sequence_generator output shape is
    dummy_input = torch.rand((batch_size, seq_len, features))
    
    # 2. Instantiate Autoencoder (single continuous head for quick test)
    autoencoder = SequenceAutoencoder(input_dim=features, hidden_dim=hidden, continuous_dim=features, categorical_sizes=[])
    
    # 3. Pass through the network
    recovered_data, latent_space = autoencoder(dummy_input)
    
    logging.info(f"Input Data Shape: {dummy_input.shape}")
    logging.info(f"Compressed Latent Shape: {latent_space.shape}")
    logging.info(f"Recovered Output Shape: {recovered_data.shape}")
