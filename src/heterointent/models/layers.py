from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int | None = None, dropout: float = 0.0) -> nn.Sequential:
    dims = [input_dim] + hidden_dims
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
    if output_dim is not None:
        layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = mlp(input_dim, [hidden_dim], output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PLELayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_tasks: int,
        shared_experts: int,
        task_experts: int,
        dropout: float,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.shared = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim, dropout) for _ in range(shared_experts)])
        self.task_specific = nn.ModuleList(
            [
                nn.ModuleList([Expert(input_dim, hidden_dim, output_dim, dropout) for _ in range(task_experts)])
                for _ in range(num_tasks)
            ]
        )
        self.task_gates = nn.ModuleList(
            [nn.Linear(input_dim, shared_experts + task_experts) for _ in range(num_tasks)]
        )
        self.shared_gate = nn.Linear(input_dim, shared_experts + num_tasks * task_experts)

    @staticmethod
    def _mix(expert_outputs: list[torch.Tensor], gate_logits: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(expert_outputs, dim=1)
        weights = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
        return (stacked * weights).sum(dim=1)

    def forward(
        self, shared_input: torch.Tensor, task_inputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        shared_outputs = [expert(shared_input) for expert in self.shared]
        task_outputs: list[torch.Tensor] = []
        task_gate_weights: list[torch.Tensor] = []
        all_task_expert_outputs: list[torch.Tensor] = []

        for task_idx, task_input in enumerate(task_inputs):
            specific_outputs = [expert(task_input) for expert in self.task_specific[task_idx]]
            all_task_expert_outputs.extend(specific_outputs)
            gate_logits = self.task_gates[task_idx](task_input)
            task_gate_weights.append(torch.softmax(gate_logits, dim=-1))
            task_outputs.append(self._mix(shared_outputs + specific_outputs, gate_logits))

        shared_gate_logits = self.shared_gate(shared_input)
        new_shared = self._mix(shared_outputs + all_task_expert_outputs, shared_gate_logits)
        return new_shared, task_outputs, task_gate_weights


class PLERanker(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_tasks: int = 3,
        shared_experts: int = 4,
        task_experts: int = 2,
        ple_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(ple_layers):
            layers.append(
                PLELayer(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_tasks=num_tasks,
                    shared_experts=shared_experts,
                    task_experts=task_experts,
                    dropout=dropout,
                )
            )
            current_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.towers = nn.ModuleList([mlp(hidden_dim, [hidden_dim], 1, dropout) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shared = x
        tasks = [x, x, x]
        gate_weights = []
        for layer in self.layers:
            shared, tasks, gates = layer(shared, tasks)
            gate_weights.append(torch.stack(gates, dim=1))
        logits = torch.cat([tower(task) for tower, task in zip(self.towers, tasks)], dim=-1)
        return logits, {"ple_gate_weights": gate_weights[-1] if gate_weights else torch.empty(0, device=x.device)}


class SharedBottomRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_tasks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.bottom = mlp(input_dim, [hidden_dim, hidden_dim], hidden_dim, dropout)
        self.towers = nn.ModuleList([mlp(hidden_dim, [hidden_dim], 1, dropout) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.bottom(x)
        logits = torch.cat([tower(h) for tower in self.towers], dim=-1)
        return logits, {}


class MMoERanker(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_tasks: int = 3,
        num_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, hidden_dim, dropout) for _ in range(num_experts)])
        self.gates = nn.ModuleList([nn.Linear(input_dim, num_experts) for _ in range(num_tasks)])
        self.towers = nn.ModuleList([mlp(hidden_dim, [hidden_dim], 1, dropout) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        task_reprs = []
        gate_weights = []
        for gate in self.gates:
            weights = torch.softmax(gate(x), dim=-1)
            gate_weights.append(weights)
            task_reprs.append((expert_outputs * weights.unsqueeze(-1)).sum(dim=1))
        logits = torch.cat([tower(h) for tower, h in zip(self.towers, task_reprs)], dim=-1)
        return logits, {"mmoe_gate_weights": torch.stack(gate_weights, dim=1)}
