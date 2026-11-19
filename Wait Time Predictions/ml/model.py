"""Shared ride GRU with cross-ride attention and autoregressive sampling."""

import torch
from torch import nn
from torch.nn import functional as F

from ml.data import CLOSED, DOWN, REFURBISHMENT, UNKNOWN, VOCAB_SIZE
from ml.availability import advance_ages, history_ages


def select_device(name="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and check the NVIDIA driver.")
    return device


def token_loss(logits, targets, *, predict_closed=False):
    """Train availability as well as waits; ignore unknown/refurbishment/padding."""
    targets = targets.masked_fill(targets > (CLOSED if predict_closed else DOWN), -100)
    losses = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction="sum")
    count = (targets != -100).sum()
    return losses / count.clamp_min(1), count

class WaitPredictor(nn.Module):
    def __init__(self, num_rides, embedding_size=16, hidden_size=128, *,
                 architecture="park_gru_v1", attention_blocks=2, attention_heads=4,
                 dropout=0.1, predict_closed=False, breakdown_hazards=None):
        super().__init__()
        if min(num_rides, embedding_size, hidden_size) < 1:
            raise ValueError("Ride count, embedding size, and hidden size must be positive")
        if architecture not in ("park_gru_v1", "ride_gru_attention_v1"):
            raise ValueError(f"Unsupported model architecture: {architecture}")
        self.architecture = architecture
        # Defaults preserve checkpoints trained before CLOSED became a target.
        self.predict_closed = predict_closed
        self.breakdown_hazards = breakdown_hazards
        self.num_rides = num_rides
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_size)
        if architecture == "park_gru_v1":
            # Checkpoints without an architecture field retain their original layout.
            self.input = nn.Linear(num_rides * embedding_size + 6, hidden_size)
            self.output = nn.Linear(hidden_size, num_rides * VOCAB_SIZE)
        else:
            if attention_blocks < 1 or attention_heads < 1 or hidden_size % attention_heads:
                raise ValueError("Attention needs positive block/head counts and hidden size divisible by heads")
            if not 0 <= dropout < 1:
                raise ValueError("Dropout must be between 0 (inclusive) and 1 (exclusive)")
            self.ride_embedding = nn.Embedding(num_rides, embedding_size)
            self.input = nn.Linear(2 * embedding_size + 6, hidden_size)
            self.attention = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size, nhead=attention_heads,
                    dim_feedforward=2 * hidden_size, dropout=dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                ) for _ in range(attention_blocks)
            ])
            self.output_norm = nn.LayerNorm(hidden_size)
            self.output = nn.Linear(hidden_size, VOCAB_SIZE)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, tokens, next_time_features, hidden=None):
        """Return next-tick logits and state with the batch on hidden axis 1.

        Ride-GRU state is [1, batch, rides, hidden], so forecast branches can
        expand/concatenate batches without mixing a ride's recurrent history.
        Attention operates only across rides at the same tick, never future ticks.
        """
        embedded = self.embedding(tokens)
        if self.architecture == "park_gru_v1":
            inputs = torch.cat((embedded.flatten(start_dim=2), next_time_features), dim=-1)
            states, hidden = self.gru(torch.relu(self.input(inputs)), hidden)
            return self.output(states).reshape(*tokens.shape, VOCAB_SIZE), hidden

        batch, ticks, rides = tokens.shape
        identities = self.ride_embedding.weight.view(1, 1, rides, -1).expand(batch, ticks, -1, -1)
        calendar = next_time_features.unsqueeze(2).expand(-1, -1, rides, -1)
        inputs = torch.relu(self.input(torch.cat((embedded, identities, calendar), dim=-1)))
        inputs = inputs.permute(0, 2, 1, 3).reshape(batch * rides, ticks, -1)
        state = None if hidden is None else hidden.reshape(1, batch * rides, -1).contiguous()
        states, state = self.gru(inputs, state)
        hidden = state.reshape(1, batch, rides, -1)
        states = states.reshape(batch, rides, ticks, -1).permute(0, 2, 1, 3)
        states = states.reshape(batch * ticks, rides, -1)
        for block in self.attention:
            states = block(states)
        logits = self.output(self.output_norm(states)).reshape(batch, ticks, rides, VOCAB_SIZE)
        return logits, hidden

    def prediction_logits(self, logits, ages, temperature=1.0):
        """Use the same output policy in production and held-out evaluation."""
        logits = logits.clone() / temperature
        logits[..., [REFURBISHMENT, UNKNOWN]] = -torch.inf
        if not self.predict_closed:
            logits[..., CLOSED] = -torch.inf
        if self.breakdown_hazards is None:
            return logits
        hazards = torch.as_tensor(self.breakdown_hazards, device=logits.device, dtype=logits.dtype)
        hazards = hazards.unsqueeze(0).expand(logits.shape[0], -1, -1)
        hazard = hazards.gather(-1, (ages - 1).clamp(0, hazards.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
        probabilities = logits.softmax(-1)
        available_mass = 1 - probabilities[..., CLOSED]
        wait_logits = logits[..., :DOWN]
        # A neural head with no finite wait logits must still permit reopening.
        wait_logits = torch.where(torch.isfinite(wait_logits).any(-1, keepdim=True), wait_logits, 0.)
        waits = wait_logits.softmax(-1) * (available_mass * hazard).unsqueeze(-1)
        calibrated = probabilities.clone()
        calibrated[..., :DOWN] = waits
        calibrated[..., DOWN] = available_mass * (1 - hazard)
        return torch.where((ages > 0).unsqueeze(-1), calibrated.log(), logits)

    @torch.inference_mode()
    def forecast(self, history, time_features, steps=24, samples=100, temperature=1.0,
                 history_timestamps=None):
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
        hidden = hidden.expand(-1, samples, *hidden.shape[2:]).contiguous()
        locked = history[-1] == REFURBISHMENT
        absent = (history == UNKNOWN).all(dim=0)
        ages = history_ages(history, history_timestamps).expand(samples, -1)
        paths = []
        for step in range(steps):
            probabilities = self.prediction_logits(logits, ages, temperature).softmax(dim=-1)
            token = torch.multinomial(probabilities.reshape(-1, VOCAB_SIZE), 1).reshape(samples, self.num_rides)
            token[:, locked] = REFURBISHMENT
            token[:, absent] = UNKNOWN
            ages = advance_ages(ages, token)
            paths.append(token)
            if step + 1 < steps:
                features = time_features[len(history) + step].expand(samples, 1, -1)
                logits, hidden = self(token.unsqueeze(1), features, hidden)
                logits = logits[:, 0]
        return torch.stack(paths, dim=1)
