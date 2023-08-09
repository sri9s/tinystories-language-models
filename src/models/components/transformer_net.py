"""Transformer model."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.data.tokenizer import Tokenizer
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding.

    Parameters
    ----------
    nn
        nn module
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    """Test code."""
    # Model hyperparameters
    ntoken = 10000
    d_model = 512
    nhead = 8
    d_hid = 2048
    nlayers = 6
    dropout = 0.5

    model = TransformerModel(
        ntoken=ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers
    )
    log.info(model)

    seq_len = 10
    batch_size = 8
    # input shape: [seq_len, batch_size]
    # output shape: [seq_len, batch_size, ntoken]
    # target shape: [seq_len, batch_size]
    input_tensor = torch.randint(low=0, high=ntoken, size=(seq_len, batch_size))
    target_tensor = torch.randint(low=0, high=ntoken, size=(seq_len, batch_size))
    output = model(input_tensor)
    print("Output shape:", output.shape)
    print("Target shape:", target_tensor.shape)
    probabilities = torch.softmax(output[-1, 0, :] / 0.7, dim=0)
    next_token = torch.multinomial(probabilities, num_samples=1).item()
    loss = F.cross_entropy(output.view(-1, ntoken), target_tensor.view(-1))
    enc = Tokenizer()
    print(f"Input shape: {input_tensor.shape}")
    print("Output shape:", output.view(-1, ntoken).shape)
    print("Target shape:", target_tensor.view(-1).shape)
    print("probabilities", probabilities.shape)
    print("next_token:", next_token)
    print("loss:", loss)
