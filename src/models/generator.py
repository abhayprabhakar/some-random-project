import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SequenceGenerator(nn.Module):
    """
    Generator network for C-TimeGAN.
    Takes in random noise sequence and generates a fake 24-dimensional latent embedding sequence.
    Conditioned on attack class 'c'.
    """
    def __init__(self, noise_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2):
        super(SequenceGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # We embed the conditional class label into a dense vector (e.g. 10 dims)
        self.class_emb_dim = 10
        self.class_embedding = nn.Embedding(num_classes, self.class_emb_dim)
        
        # The input to the LSTM at each time step will be noise size + embedded class size
        lstm_input_dim = noise_dim + self.class_emb_dim
        
        self.rnn = nn.LSTM(
            input_size=lstm_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        # Using Tanh to output values between -1 and 1 or Sigmoid for [0, 1]
        # Since the autoencoder embeds into Sigmoid [0,1], we match it.
        self.activation = nn.Sigmoid()

    def forward(self, z, c):
        # z: (batch, seq_len, noise_dim)
        # c: (batch) -> a single class label per sequence 
        # (Alternatively, could provide a seq of labels, but typically it's one attack type per window)
        batch_size, seq_len, _ = z.size()
        
        # Embed class: (batch, emb_dim)
        c_emb = self.class_embedding(c)
        
        # Expand condition to all time steps: (batch, seq_len, emb_dim)
        c_emb_expanded = c_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate noise and condition: (batch, seq_len, noise_dim + emb_dim)
        rnn_input = torch.cat([z, c_emb_expanded], dim=2)
        
        out, _ = self.rnn(rnn_input)
        out = self.linear(out)
        out = self.activation(out)
        
        return out

if __name__ == "__main__":
    batch_size = 32
    seq_len = 20
    noise = 16
    hidden = 24
    classes = 5
    
    dummy_noise = torch.randn((batch_size, seq_len, noise))
    dummy_labels = torch.randint(0, classes, (batch_size,))
    
    generator = SequenceGenerator(noise_dim=noise, hidden_dim=hidden, num_classes=classes)
    generated_embedding = generator(dummy_noise, dummy_labels)
    
    logging.info(f"Noise Input Shape: {dummy_noise.shape}")
    logging.info(f"Labels Shape: {dummy_labels.shape}")
    logging.info(f"Generated Latent Sequence Shape: {generated_embedding.shape}")
