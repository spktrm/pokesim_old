import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pokesim.structs import ModelOutput
from pokesim.rl_utils import _legal_log_policy, _legal_policy
from pokesim.model.embedding import EntityEmbedding


def _layer_init(
    layer: nn.Module, mean: float = None, std: float = None, bias_value: float = None
):
    if hasattr(layer, "weight"):
        if isinstance(layer, nn.Embedding):
            init_func = nn.init.normal_
        elif isinstance(layer, nn.Linear):
            init_func = nn.init.trunc_normal_
        if std is None:
            n = getattr(layer, "num_embeddings", None) or getattr(layer, "in_features")
            std = math.sqrt(1 / n)
        init_func(layer.weight, mean=(mean or 0), std=std)
    if hasattr(layer, "bias") and getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, val=(bias_value or 0))
    return layer


class ResBlock(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()

        self.lin1 = _layer_init(nn.Linear(size, size))
        self.lin2 = _layer_init(nn.Linear(size, size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin1(F.relu(x))
        out = self.lin1(F.relu(x))
        return out + x


class Model(nn.Module):
    def __init__(self, size: int = 256):
        super().__init__()
        self.embedding = EntityEmbedding()
        self.move_embeddings = _layer_init(
            nn.Embedding(self.embedding.moves_shape[0] + 1, size)
        )
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(7))

        self.boosts_embedding = _layer_init(nn.Linear(2 * 7, size, bias=False))

        self.pseudoweather_onehot = nn.Embedding.from_pretrained(torch.eye(9)[..., 1:])
        self.weather_onehot = nn.Embedding.from_pretrained(torch.eye(9))
        self.terrain_onehot = nn.Embedding.from_pretrained(torch.eye(6))

        self.field_embedding = _layer_init(nn.Linear(8 + 9 + 6, size, bias=False))

        self.sidecon_onehot = nn.Embedding.from_pretrained(torch.eye(16)[..., 1:])
        self.sidecon_embedding = _layer_init(nn.Linear(2 * 15, size, bias=False))

        self.volatile_onehot = nn.Embedding.from_pretrained(torch.eye(106)[..., 1:])
        self.volatile_embedding = _layer_init(nn.Linear(2 * 105, size, bias=False))

        self.ee1 = nn.Sequential(
            _layer_init(nn.Linear(888, size, bias=False)),
            nn.ReLU(),
            _layer_init(nn.Linear(size, size, bias=False)),
        )
        self.ee2 = nn.Sequential(
            nn.ReLU(),
            _layer_init(nn.Linear(3 * size, size, bias=False)),
        )

        # self.state_lin = _layer_init(nn.Linear(5 * size, 5 * size))
        # self.gate_lin = _layer_init(nn.Linear(5 * size, 5 * size))

        self.denom = math.sqrt(size)

        self.torso = nn.Sequential(*[ResBlock(size) for _ in range(2)])
        self.value = nn.Sequential(
            ResBlock(size),
            ResBlock(size),
            nn.ReLU(),
            _layer_init(nn.Linear(size, 1)),
        )

    def forward(
        self,
        teams: torch.Tensor,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        field: torch.Tensor,
        mask: torch.Tensor,
    ):
        T, B, *_ = mask.shape

        teams_ = teams + 1
        species_token = teams_[..., 0]
        item_token = teams_[..., 1]
        ability_token = teams_[..., 2]
        hp = (teams[..., 3] / 1000).unsqueeze(-1)
        active_token = teams[..., 4].clamp(min=0)
        fainted_token = teams[..., 5].clamp(min=0)
        status_token = teams_[..., 6]
        move_tokens = teams_[..., -4:]
        with torch.no_grad():
            (
                species_embedding,
                item_embedding,
                ability_embedding,
                moveset_embedding,
            ) = self.embedding(species_token, ability_token, item_token, move_tokens)
        entity_embedding = torch.cat(
            (
                species_embedding,
                item_embedding,
                ability_embedding,
                moveset_embedding,
                hp,
                self.active_onehot(active_token),
                self.fainted_onehot(fainted_token),
                self.status_onehot(status_token),
            ),
            dim=-1,
        )
        entities_embedding = self.ee1(entity_embedding)
        side_embedding = self.ee2(entities_embedding.max(-2).values.flatten(2))
        boosts_embedding = self.boosts_embedding(boosts.flatten(2) / 6)

        pseudoweather = field[..., :9].view(T, B, 3, 3)
        pseudoweather_tokens = pseudoweather[..., 0]
        psuedoweather_onehot = self.pseudoweather_onehot(pseudoweather_tokens).sum(-2)
        weather_onehot = self.weather_onehot(field[..., 10])
        terrain_onehot = self.terrain_onehot(field[..., 13])

        field_onehot = torch.cat(
            (psuedoweather_onehot, weather_onehot, terrain_onehot), dim=-1
        )
        field_embedding = self.field_embedding(field_onehot)

        volatile_onehot = (
            self.volatile_onehot(volatile_status[..., 0, :]).sum(-2).flatten(2)
        )
        volatile_embedding = self.volatile_embedding(volatile_onehot)

        sidecon_onehot = (
            self.sidecon_onehot((side_conditions > 0).to(torch.long)).sum(-2).flatten(2)
        )
        sidecon_embedding = self.sidecon_embedding(sidecon_onehot)

        # state_embedding = torch.stack(
        #     (
        #         side_embedding,
        #         boosts_embedding,
        #         field_embedding,
        #         volatile_embedding,
        #         sidecon_embedding,
        #     ),
        #     dim=-2,
        # )
        # shape = state_embedding.shape
        # flat_state = state_embedding.flatten(-2)
        # state_embedding = self.state_lin(flat_state).view(*shape)
        # gate = self.gate_lin(flat_state).view(*shape)
        # gate = gate.softmax(-2)

        # state_embedding = (state_embedding * gate).sum(-2)

        state_embedding = (
            side_embedding
            + boosts_embedding
            + field_embedding
            + volatile_embedding
            + sidecon_embedding
        )

        state_embedding = self.torso(state_embedding)

        switch_embeddings = entities_embedding[..., 0, :6, :]
        move_embeddings = self.move_embeddings(move_tokens[..., 0, 0, :])

        key = state_embedding.unsqueeze(-2)
        logits = torch.cat(
            (
                key @ move_embeddings.transpose(-2, -1),
                key @ switch_embeddings.transpose(-2, -1),
            ),
            dim=-1,
        ).squeeze(-2)

        value = self.value(state_embedding)
        policy = _legal_policy(logits, mask)
        log_policy = _legal_log_policy(logits, mask)
        return ModelOutput(
            policy=policy, value=value, log_policy=log_policy, logits=logits
        )
