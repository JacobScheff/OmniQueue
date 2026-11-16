"""Park-wide GRU, categorical loss, and autoregressive sampling."""

import torch
from torch import nn
from torch.nn import functional as F

from ml.data import CLOSED, DOWN, REFURBISHMENT, UNKNOWN, VOCAB_SIZE


def select_device(name="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and check the NVIDIA driver.")
    return device


def token_loss(logits, targets):
    """CLOSED, REFURBISHMENT, and unavailable observations contribute no loss."""
    targets = targets.masked_fill(targets > DOWN, -100)
    losses = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction="sum")
    count = (targets != -100).sum()
    return losses / count.clamp_min(1), count


class WaitPredictor(nn.Module):
    def __init__(self, num_rides, embedding_size=16, hidden_size=128):
        super().__init__()
        self.num_rides = num_rides
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_size)
        # Fixed ride slots preserve identity while mixing information across the park.
        self.input = nn.Linear(num_rides * embedding_size + 6, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, num_rides * VOCAB_SIZE)

    def forward(self, tokens, next_time_features, hidden=None):
        embedded = self.embedding(tokens).flatten(start_dim=2)
        inputs = torch.cat((embedded, next_time_features), dim=-1)
        states, hidden = self.gru(torch.relu(self.input(inputs)), hidden)
        logits = self.output(states).reshape(*tokens.shape, VOCAB_SIZE)
        return logits, hidden

    @torch.inference_mode()
    def forecast(self, history, time_features, steps=24, samples=100, temperature=1.0):
        """Consume all supplied day history and return [samples, steps, rides].

        time_features contains next-tick features for each history input, followed
        by the remaining future ticks. No future ride observations are consumed.
        """
        if history.ndim != 2 or history.shape[0] < 1 or history.shape[1] != self.num_rides:
            raise ValueError("History must have shape [nonempty ticks, checkpoint rides]")
        if steps < 1 or samples < 1 or temperature <= 0:
            raise ValueError("Steps, samples, and temperature must be positive")
        if time_features.shape != (history.shape[0] + steps - 1, 6):
            raise ValueError("Calendar features do not cover the history and forecast")
        self.eval()
        logits, hidden = self(history.unsqueeze(0), time_features[:len(history)].unsqueeze(0))
        logits = logits[:, -1].expand(samples, -1, -1)
        hidden = hidden.expand(-1, samples, -1).contiguous()
        locked = history[-1] == REFURBISHMENT
        absent = (history == UNKNOWN).all(dim=0)
        paths = []
        for step in range(steps):
            # These tokens have no prediction targets. They remain input/context tokens.
            logits = logits.clone()
            logits[..., [CLOSED, REFURBISHMENT, UNKNOWN]] = -torch.inf
            probabilities = (logits / temperature).softmax(dim=-1)
            token = torch.multinomial(probabilities.reshape(-1, VOCAB_SIZE), 1).reshape(samples, self.num_rides)
            token[:, locked] = REFURBISHMENT
            token[:, absent] = UNKNOWN
            paths.append(token)
            if step + 1 < steps:
                features = time_features[len(history) + step].expand(samples, 1, -1)
                logits, hidden = self(token.unsqueeze(1), features, hidden)
                logits = logits[:, 0]
        return torch.stack(paths, dim=1)
