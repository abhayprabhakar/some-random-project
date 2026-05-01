import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SequenceDiscriminator(nn.Module):
    """
    Discriminator network for C-TimeGAN.
    Takes in a 24-dimensional latent embedding sequence and predicts if it's real or generated.
    Also conditioned on the attack class 'c'.
    """
    def __init__(self, hidden_dim: int, num_classes: int, num_layers: int = 2):
        super(SequenceDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.class_emb_dim = 10
        self.class_embedding = nn.Embedding(num_classes, self.class_emb_dim)
        
        # Input to discriminator LSTM is the generated/real embedding + class embedding
        lstm_input_dim = hidden_dim + self.class_emb_dim
        
        self.rnn = nn.LSTM(
            input_size=lstm_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Output a single score per sequence (or per timestep)
        # Using 1 dimension since it's a binary real/fake scalar score
        # For WGAN, we don't use sigmoid on the output scalar, just linear.
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, h, c):
        # h: (batch, seq_len, hidden_dim) - the latent sequence
        # c: (batch)
        batch_size, seq_len, _ = h.size()
        
        c_emb = self.class_embedding(c)
        c_emb_expanded = c_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        rnn_input = torch.cat([h, c_emb_expanded], dim=2)
        
        # We can either inspect the final step only or all steps. 
        # Typically timeGAN discriminates all steps
        out, _ = self.rnn(rnn_input)
        # out: (batch, seq_len, hidden_dim)
        
        score = self.linear(out) 
        # scalar score: (batch, seq_len, 1)
        
        # Pool to one score per sequence by averaging time steps
        score = score.mean(dim=1) 
        # final shape: (batch, 1)
        
        return score

if __name__ == "__main__":
    batch_size = 32
    seq_len = 20
    hidden = 24
    classes = 5
    
    # Simulating the generated latent embedding from the generator
    dummy_latent = torch.rand((batch_size, seq_len, hidden))
    dummy_labels = torch.randint(0, classes, (batch_size,))
    
    discriminator = SequenceDiscriminator(hidden_dim=hidden, num_classes=classes)
    validity_score = discriminator(dummy_latent, dummy_labels)
    
    logging.info(f"Latent Sequence Input Shape: {dummy_latent.shape}")
    logging.info(f"Labels Shape: {dummy_labels.shape}")
    logging.info(f"Discriminator Score Shape: {validity_score.shape}")
