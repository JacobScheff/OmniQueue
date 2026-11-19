"""Causal park forecasters, including a structured wait-transition model."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ml.data import CLOSED, DOWN, OVERFLOW, REFURBISHMENT, UNKNOWN, VOCAB_SIZE
from ml.availability import advance_ages, history_ages


def select_device(name="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and check the NVIDIA driver.")
    return device


def token_loss(logits, targets, *, predict_closed=False, ordinal_weight=0.0):
    """Train availability as well as waits; ignore unknown/refurbishment/padding."""
    targets = targets.masked_fill((targets < 0) | (targets > (CLOSED if predict_closed else DOWN)), -100)
    losses = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction="sum")
    count = (targets != -100).sum()
    loss = losses / count.clamp_min(1)
    if ordinal_weight:
        # Ranked probability score on exact numeric waits only. Status labels
        # and the open-ended overflow bucket have no numeric distance.
        numeric = (targets >= 0) & (targets < OVERFLOW)
        cdf = logits[..., :OVERFLOW].softmax(-1).cumsum(-1)[..., :-1]
        observed_cdf = targets.unsqueeze(-1) <= torch.arange(OVERFLOW - 1, device=logits.device)
        distances = (cdf - observed_cdf.to(cdf.dtype)).square().mean(-1)
        loss = loss + ordinal_weight * (distances * numeric).sum() / numeric.sum().clamp_min(1)
    return loss, count

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

    def forward(self, tokens, next_time_features, hidden=None, *, return_state_sequence=False):
        """Return next-tick logits and state with the batch on hidden axis 1.

        Ride-GRU state is [1, batch, rides, hidden], so forecast branches can
        expand/concatenate batches without mixing a ride's recurrent history.
        Attention operates only across rides at the same tick, never future ticks.
        """
        embedded = self.embedding(tokens)
        if self.architecture == "park_gru_v1":
            inputs = torch.cat((embedded.flatten(start_dim=2), next_time_features), dim=-1)
            states, hidden = self.gru(torch.relu(self.input(inputs)), hidden)
            state = states.unsqueeze(0) if return_state_sequence else hidden
            return self.output(states).reshape(*tokens.shape, VOCAB_SIZE), state

        batch, ticks, rides = tokens.shape
        identities = self.ride_embedding.weight.view(1, 1, rides, -1).expand(batch, ticks, -1, -1)
        calendar = next_time_features.unsqueeze(2).expand(-1, -1, rides, -1)
        inputs = torch.relu(self.input(torch.cat((embedded, identities, calendar), dim=-1)))
        inputs = inputs.permute(0, 2, 1, 3).reshape(batch * rides, ticks, -1)
        state = None if hidden is None else hidden.reshape(1, batch * rides, -1).contiguous()
        states, state = self.gru(inputs, state)
        states = states.reshape(batch, rides, ticks, -1).permute(0, 2, 1, 3)
        hidden = (states.unsqueeze(0) if return_state_sequence
                  else state.reshape(1, batch, rides, -1))
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

    def greedy_prediction(self, logits, ages, survival):
        """Greedy waits; median residual lifetime for a calibrated outage."""
        supported = self.prediction_logits(logits, ages)
        prediction = supported.argmax(-1)
        if self.breakdown_hazards is not None:
            probabilities = supported.softmax(-1)
            nonclosed = (1 - probabilities[..., CLOSED]).clamp_min(1e-12)
            survival = torch.where(ages > 0, survival * probabilities[..., DOWN] / nonclosed, 1.)
            operating = supported[..., :DOWN].argmax(-1)
            reopening = torch.where(survival > .5, DOWN, operating)
            prediction = torch.where((ages > 0) & (probabilities[..., CLOSED] < .5), reopening, prediction)
        return prediction, survival

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


class TransitionWaitPredictor(WaitPredictor):
    """Ride recurrence with park context and a local/global wait mixture.

    The two extra state channels hold the last token and its run length. They
    are causal observations, not learned recurrent units. Keeping them in the
    batchable state makes full-day, streaming, and branched forecasts identical.
    """

    def __init__(self, num_rides, embedding_size=16, hidden_size=128, *,
                 architecture="ride_transition_v2", attention_blocks=2, attention_heads=4,
                 dropout=0.1, predict_closed=False, breakdown_hazards=None):
        if architecture != "ride_transition_v2":
            raise ValueError(f"Unsupported transition architecture: {architecture}")
        super().__init__(num_rides, embedding_size, hidden_size,
                         architecture="ride_gru_attention_v1", attention_blocks=attention_blocks,
                         attention_heads=attention_heads, dropout=dropout,
                         predict_closed=predict_closed, breakdown_hazards=breakdown_hazards)
        self.architecture = architecture
        self.hidden_size = hidden_size
        self.input = nn.Linear(2 * embedding_size + 6 + 5, hidden_size)
        self.park_context = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.GELU(),
                                          nn.LayerNorm(hidden_size))
        self.delta_output = nn.Linear(hidden_size, 25)  # -60 through +60 minutes
        self.local_gate = nn.Linear(hidden_size, 1)
        self.availability_output = nn.Linear(hidden_size, 3)  # operating, DOWN, CLOSED
        self.register_buffer("wait_deltas", torch.arange(-12, 13), persistent=False)

    def forward(self, tokens, next_time_features, hidden=None, *, return_state_sequence=False):
        batch, ticks, rides = tokens.shape
        if hidden is None:
            previous = torch.full_like(tokens[:, :1], UNKNOWN)
            run = next_time_features.new_zeros(batch, 1, rides)
            recurrent = None
        else:
            recurrent = hidden[..., :self.hidden_size].reshape(1, batch * rides, self.hidden_size).contiguous()
            previous = hidden[0, ..., self.hidden_size].long().unsqueeze(1)
            run = hidden[0, ..., self.hidden_size + 1].unsqueeze(1)
        before = torch.cat((previous, tokens[:, :-1]), dim=1)
        indices = torch.arange(1, ticks + 1, device=tokens.device).view(1, ticks, 1)
        starts = torch.where(tokens != before, indices, 0).cummax(dim=1).values
        duration = torch.where(starts == 0, run + indices, indices - starts + 1).clamp_max(288)
        numeric = (tokens >= 0) & (tokens < OVERFLOW)
        pair = numeric & (before >= 0) & (before < OVERFLOW)
        numeric_features = torch.stack((
            torch.where(numeric, tokens / (OVERFLOW - 1), 0.), numeric.float(),
            torch.where(pair, (tokens - before) / 12., 0.), pair.float(),
            torch.log1p(duration) / math.log(289),
        ), dim=-1)
        identities = self.ride_embedding.weight.view(1, 1, rides, -1).expand(batch, ticks, -1, -1)
        calendar = next_time_features.unsqueeze(2).expand(-1, -1, rides, -1)
        inputs = torch.relu(self.input(torch.cat((self.embedding(tokens), identities, calendar, numeric_features), -1)))
        present = ((tokens != UNKNOWN) & (tokens != REFURBISHMENT)).unsqueeze(-1)
        pooled = (inputs * present).sum(2, keepdim=True) / present.sum(2, keepdim=True).clamp_min(1)
        inputs = self.park_context(torch.cat((inputs, pooled.expand_as(inputs)), -1))
        inputs = inputs.permute(0, 2, 1, 3).reshape(batch * rides, ticks, -1)
        states, _ = self.gru(inputs, recurrent)
        states = states.reshape(batch, rides, ticks, -1).permute(0, 2, 1, 3)
        all_states = torch.cat((states, tokens.unsqueeze(-1), duration.unsqueeze(-1)), -1)
        hidden = all_states.unsqueeze(0) if return_state_sequence else all_states[:, -1].unsqueeze(0).contiguous()
        context = states.reshape(batch * ticks, rides, -1)
        for block in self.attention:
            context = block(context)
        context = self.output_norm(context).reshape(batch, ticks, rides, -1)
        return self.wait_distribution(context, tokens), hidden

    def wait_distribution(self, context, tokens):
        """Factor availability from a mixture of local and global wait buckets."""
        numeric = (tokens >= 0) & (tokens < OVERFLOW)
        global_wait = self.output(context)[..., :DOWN].softmax(-1)
        local_wait = torch.zeros_like(global_wait)
        columns = (tokens.unsqueeze(-1) + self.wait_deltas).clamp(0, OVERFLOW)
        local_wait.scatter_add_(-1, columns, self.delta_output(context).softmax(-1))
        gate = self.local_gate(context).sigmoid() * numeric.unsqueeze(-1)
        wait = gate * local_wait + (1 - gate) * global_wait
        availability = self.availability_output(context).log_softmax(-1)
        supported = torch.cat((wait.clamp_min(1e-12).log() + availability[..., :1], availability[..., 1:]), -1)
        return F.pad(supported, (0, VOCAB_SIZE - CLOSED - 1), value=-torch.inf)


class HorizonWaitPredictor(TransitionWaitPredictor):
    """Direct lead-conditioned forecasts from a fixed, causal observed prefix.

    One-step recurrence remains useful for history encoding and immediate
    forecasts. Longer leads query that encoding directly, avoiding recursively
    treating a guessed wait as a fresh measurement.
    """

    def __init__(self, num_rides, embedding_size=16, hidden_size=128, *,
                 architecture="ride_horizon_v3", **settings):
        if architecture != "ride_horizon_v3":
            raise ValueError(f"Unsupported horizon architecture: {architecture}")
        super().__init__(num_rides, embedding_size, hidden_size, **settings)
        self.architecture = architecture
        self.horizon_query = nn.Sequential(
            nn.Linear(hidden_size + 2 * embedding_size + 6 + 3, 2 * hidden_size), nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size), nn.LayerNorm(hidden_size),
        )

    def horizon_logits(self, origin_states, origin_tokens, future_features, leads):
        """[batch, leads, rides, vocab]; no future observations or generated inputs."""
        batch, ticks, _ = future_features.shape
        leads = torch.as_tensor(leads, device=future_features.device, dtype=future_features.dtype)
        leads = leads.expand(batch, ticks)
        horizon = torch.stack((leads / 48., leads.log1p() / math.log(49), leads.rsqrt()), -1)
        recurrent = origin_states[0, ..., :self.hidden_size].unsqueeze(1).expand(-1, ticks, -1, -1)
        tokens = origin_tokens.unsqueeze(1).expand(-1, ticks, -1)
        identities = self.ride_embedding.weight.view(1, 1, self.num_rides, -1).expand(batch, ticks, -1, -1)
        features = torch.cat((future_features, horizon), -1).unsqueeze(2).expand(-1, -1, self.num_rides, -1)
        context = self.horizon_query(torch.cat((recurrent, self.embedding(tokens), identities, features), -1))
        context = context.reshape(batch * ticks, self.num_rides, -1)
        for block in self.attention:
            context = block(context)
        context = self.output_norm(context).reshape(batch, ticks, self.num_rides, -1)
        return self.wait_distribution(context, tokens)

    @torch.inference_mode()
    def forecast(self, history, time_features, steps=24, samples=100, temperature=1.0,
                 history_timestamps=None):
        if history.ndim != 2 or history.shape[0] < 1 or history.shape[1] != self.num_rides:
            raise ValueError("History must have shape [nonempty ticks, checkpoint rides]")
        if steps < 1 or samples < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Steps, samples, and temperature must be positive and finite")
        if time_features.shape != (history.shape[0] + steps - 1, 6):
            raise ValueError("Calendar features do not cover the history and forecast")
        self.eval()
        immediate, state = self(history.unsqueeze(0), time_features[:len(history)].unsqueeze(0))
        features = time_features[len(history) - 1:].unsqueeze(0)
        logits = self.horizon_logits(state, history[-1:].contiguous(), features,
                                     torch.arange(1, steps + 1, device=history.device))
        logits[:, 0] = immediate[:, -1]
        locked = history[-1] == REFURBISHMENT
        absent = (history == UNKNOWN).all(0)
        ages = history_ages(history, history_timestamps).expand(samples, -1)
        paths = []
        for step in range(steps):
            probabilities = self.prediction_logits(logits[:, step].expand(samples, -1, -1), ages, temperature).softmax(-1)
            token = torch.multinomial(probabilities.reshape(-1, VOCAB_SIZE), 1).reshape(samples, self.num_rides)
            token[:, locked] = REFURBISHMENT
            token[:, absent] = UNKNOWN
            ages = advance_ages(ages, token)
            paths.append(token)
        return torch.stack(paths, 1)


def build_model(**settings):
    """Reconstruct architecture from checkpoint metadata, including legacy ones."""
    architectures = {"ride_transition_v2": TransitionWaitPredictor, "ride_horizon_v3": HorizonWaitPredictor}
    model = architectures.get(settings.get("architecture"), WaitPredictor)
    return model(**settings)
