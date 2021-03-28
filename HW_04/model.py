# import torch
import torch.nn as nn


class GRUNet(nn.Module):
    """Recurrent Network with GRU Layers"""
    def __init__(
            self,
            embedding,
            hidden_dim=150,
            num_layers=1,
            dropout_rate=0.4,
            fix_embedding=True
            ):
        super(GRUNet, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)

        # Fix the embedding in training or not
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # GRU layer after embedding layer
        # TODO: add batchnorm or layernorm
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate)

        self.classifier = nn.Sequential(
            # nn.LayerNorm(hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
            )

    def forward(self, inputs):
        inputs = self.embedding(inputs)

        x, _ = self.gru(inputs, None)

        # Use final layer of GRU
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
