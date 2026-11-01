# Small CPU tensor policy with frozen random features and low-rank adapters.
# This is a discrete synthetic policy, not an adapter around a pretrained LLM.

import torch
import torch.nn as nn

from .config import Config
from .features import FeatureBuilder


class Head(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, skip_features: int | None = None, weights_trainable: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / (in_features ** 0.5)), requires_grad=weights_trainable)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=weights_trainable)
        self.weights_trainable = weights_trainable
        self.adapter_down = nn.Parameter(torch.zeros(rank, in_features))
        self.adapter_up = nn.Parameter(torch.zeros(out_features, rank))
        # A zero up-projection starts the hidden-state adapter contribution at zero.
        nn.init.normal_(self.adapter_down, std=1.0 / (rank ** 0.5))
        self.skip_features = skip_features
        if skip_features is None:
            self.skip_down = None
            self.skip_up = None
        else:
            self.skip_down = nn.Parameter(torch.zeros(rank, skip_features))
            self.skip_up = nn.Parameter(torch.randn(out_features, rank) * (0.10 / (rank ** 0.5)))
            nn.init.normal_(self.skip_down, std=1.0 / (rank ** 0.5))

    def forward(self, hidden: torch.Tensor, adapter_scale: float, skip: torch.Tensor | None = None) -> torch.Tensor:
        base = torch.nn.functional.linear(hidden, self.weight, self.bias)
        mixed = torch.nn.functional.linear(hidden, self.adapter_down)
        low_rank = torch.nn.functional.linear(mixed, self.adapter_up)
        if self.skip_down is not None and skip is not None:
            mixed_skip = torch.nn.functional.linear(skip, self.skip_down)
            low_rank = low_rank + torch.nn.functional.linear(mixed_skip, self.skip_up)
        return base + adapter_scale * low_rank

    def trainable(self) -> list:
        parts = [self.adapter_up, self.adapter_down]
        if self.weights_trainable:
            parts.append(self.weight)
            parts.append(self.bias)
        if self.skip_down is not None:
            parts.append(self.skip_down)
            parts.append(self.skip_up)
        return parts

    def state(self) -> dict:
        entry = {"down": self.adapter_down.detach().cpu().clone(), "up": self.adapter_up.detach().cpu().clone()}
        if self.weights_trainable:
            entry["weight"] = self.weight.detach().cpu().clone()
            entry["bias"] = self.bias.detach().cpu().clone()
        if self.skip_down is not None:
            entry["skip_down"] = self.skip_down.detach().cpu().clone()
            entry["skip_up"] = self.skip_up.detach().cpu().clone()
        return entry

    def load(self, entry: dict) -> None:
        with torch.no_grad():
            self.adapter_down.copy_(entry["down"])
            self.adapter_up.copy_(entry["up"])
            if self.weights_trainable:
                self.weight.copy_(entry["weight"])
                self.bias.copy_(entry["bias"])
            if self.skip_down is not None:
                self.skip_down.copy_(entry["skip_down"])
                self.skip_up.copy_(entry["skip_up"])


class Policy(nn.Module):
    def __init__(self, config: Config, features: FeatureBuilder):
        super().__init__()
        self.config = config
        self.features = features
        self.hidden = config.hidden
        self.trunk_one = nn.Parameter(torch.randn(self.hidden, features.dimension) * (1.0 / (features.dimension ** 0.5)), requires_grad=False)
        self.trunk_one_bias = nn.Parameter(torch.zeros(self.hidden), requires_grad=False)
        self.trunk_two = nn.Parameter(torch.randn(self.hidden, self.hidden) * (1.0 / (self.hidden ** 0.5)), requires_grad=False)
        self.trunk_two_bias = nn.Parameter(torch.zeros(self.hidden), requires_grad=False)
        self.trunk_adapter = Head(self.hidden, self.hidden, config.adapter_rank)
        self.call_head = Head(self.hidden, 2, config.adapter_rank, features.dimension, config.heads_trainable)
        self.decision_head = Head(self.hidden, 3, config.adapter_rank, features.dimension, config.heads_trainable)
        self.answer_head = Head(self.hidden, config.num_values, config.adapter_rank, None, config.answer_head_trainable)
        # Fixed answer readout supplies memory/copy behavior without answer-head training.
        copy = torch.zeros(config.num_values, features.dimension)
        copy[:, features.memory_start:features.memory_start + config.num_values] = 0.15 * torch.eye(config.num_values)
        copy[:, features.candidate_start:features.candidate_start + config.num_values] = 3.00 * torch.eye(config.num_values)
        copy[:, features.consumed_start:features.consumed_start + config.num_values] = 0.40 * torch.eye(config.num_values)
        copy[:, features.adopted_start:features.adopted_start + config.num_values] = 4.20 * torch.eye(config.num_values)
        self.copy_readout = nn.Parameter(copy, requires_grad=False)
        self.decision_head.bias = nn.Parameter(torch.tensor([0.0, 1.40, -0.60]), requires_grad=config.heads_trainable)
        self.call_head.bias = nn.Parameter(torch.tensor([1.00, -0.50]), requires_grad=config.heads_trainable)
        self.adapter_scale = 1.0
        if not config.trunk_adapter_trainable:
            for parameter in self.trunk_adapter.trainable():
                parameter.requires_grad_(False)
        if not config.answer_head_trainable:
            for parameter in self.answer_head.trainable():
                parameter.requires_grad_(False)

    def hidden_state(self, inputs: torch.Tensor, adapted: bool) -> torch.Tensor:
        first = torch.tanh(torch.nn.functional.linear(inputs, self.trunk_one, self.trunk_one_bias))
        second = torch.tanh(torch.nn.functional.linear(first, self.trunk_two, self.trunk_two_bias))
        if adapted:
            second = self.trunk_adapter(second, self.adapter_scale)
        return second

    def forward(self, inputs: torch.Tensor) -> tuple:
        hidden = self.hidden_state(inputs, adapted=True)
        call_logits = self.call_head(hidden, self.adapter_scale, inputs)
        decision_logits = self.decision_head(hidden, self.adapter_scale, inputs)
        answer_logits = self.answer_head(hidden, self.adapter_scale, inputs) + torch.nn.functional.linear(inputs, self.copy_readout)
        return call_logits, decision_logits, answer_logits

    # Disable adapter paths. When base heads are trainable, their current weights are
    # reused here; this is not a separately frozen initial-policy snapshot.
    @torch.no_grad()
    def reference(self, inputs: torch.Tensor) -> tuple:
        hidden = self.hidden_state(inputs, adapted=False)
        call_logits = torch.nn.functional.linear(hidden, self.call_head.weight, self.call_head.bias)
        decision_logits = torch.nn.functional.linear(hidden, self.decision_head.weight, self.decision_head.bias)
        answer_logits = torch.nn.functional.linear(hidden, self.answer_head.weight, self.answer_head.bias) + torch.nn.functional.linear(inputs, self.copy_readout)
        return call_logits, decision_logits, answer_logits

    def heads(self) -> tuple:
        return (self.trunk_adapter, self.call_head, self.decision_head, self.answer_head)

    # Despite the name, this can include base head weights enabled by the config.
    def adapter_parameters(self) -> list:
        trainable: list = []
        for head in self.heads():
            for parameter in head.trainable():
                if parameter.requires_grad:
                    trainable.append(parameter)
        return trainable

    def snapshot(self) -> dict:
        return {"adapters": {str(id(head)): head.state() for head in self.heads()}}

    def restore(self, snapshot: dict) -> None:
        state = snapshot["adapters"]
        for head in self.heads():
            head.load(state[str(id(head))])
