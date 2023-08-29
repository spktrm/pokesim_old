import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pokesim.structs import ModelOutput
from pokesim.rl_utils import _legal_log_policy, _legal_policy
from pokesim.model.embedding import EntityEmbedding
from pokesim.constants import _NUM_HISTORY


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

        self.net = nn.Sequential(
            nn.LeakyReLU(),
            _layer_init(nn.Linear(size, size)),
            nn.LayerNorm(size),
            nn.LeakyReLU(),
            _layer_init(nn.Linear(size, size)),
            nn.LayerNorm(size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class Model(nn.Module):
    def __init__(self, size: int = 64):
        super().__init__()
        self.embedding = EntityEmbedding()
        self.move_embeddings = _layer_init(
            nn.Embedding(self.embedding.moves_shape[0] + 1, size)
        )
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(7))

        self.pseudoweather_onehot = nn.Embedding.from_pretrained(torch.eye(9)[..., 1:])
        self.weather_onehot = nn.Embedding.from_pretrained(torch.eye(9))
        self.terrain_onehot = nn.Embedding.from_pretrained(torch.eye(6))
        self.sidecon_onehot = nn.Embedding.from_pretrained(torch.eye(16)[..., 1:])
        self.volatile_onehot = nn.Embedding.from_pretrained(torch.eye(106)[..., 1:])

        self.ee1 = nn.Sequential(
            _layer_init(nn.Linear(888, size, bias=False)),
            nn.LeakyReLU(),
            _layer_init(nn.Linear(size, size, bias=False)),
            nn.LayerNorm(size),
        )
        self.ee2 = nn.Sequential(
            nn.LeakyReLU(),
            _layer_init(nn.Linear(3 * size, size, bias=False)),
            nn.LayerNorm(size),
        )

        self.context_embedding = _layer_init(
            nn.Linear(2 * 7 + 8 + 9 + 6 + 2 * 15 + 2 * 105, size)
        )

        self.coeff = 1 / math.sqrt(size)

        self.torso1 = nn.Sequential(*[ResBlock(size) for _ in range(2)])
        self.torso2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(_NUM_HISTORY, 32, 3, 2, bias=False),
            nn.MaxPool1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, 3, 2, bias=False),
            nn.MaxPool1d(2),
        )
        self.torso3 = nn.Sequential(
            nn.LeakyReLU(),
            _layer_init(nn.Linear((size - 16) * 4, size)),
            nn.LayerNorm(size),
        )
        self.queries = nn.Sequential(
            nn.LeakyReLU(),
            _layer_init(nn.Linear(size, size)),
            nn.LayerNorm(size),
            nn.LeakyReLU(),
            _layer_init(nn.Linear(size, size)),
            nn.LayerNorm(size),
        )
        self.value = nn.Sequential(
            ResBlock(size),
            ResBlock(size),
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
        T, B, H, *_ = teams.shape

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
        side_embedding = self.ee2(entities_embedding.max(-2).values.flatten(3))

        pseudoweather = field[..., :9].view(T, B, H, 3, 3)
        pseudoweather_tokens = pseudoweather[..., 0]
        psuedoweather_onehot = self.pseudoweather_onehot(pseudoweather_tokens).sum(-2)
        weather_onehot = self.weather_onehot(field[..., 10])
        terrain_onehot = self.terrain_onehot(field[..., 13])

        context_onehot = torch.cat(
            (
                boosts.flatten(3) / 6,
                psuedoweather_onehot,
                weather_onehot,
                terrain_onehot,
                self.volatile_onehot(volatile_status[..., 0, :]).sum(-2).flatten(3),
                self.sidecon_onehot((side_conditions > 0).to(torch.long))
                .sum(-2)
                .flatten(3),
            ),
            dim=-1,
        )

        context_embedding = self.context_embedding(context_onehot)
        state_embedding = side_embedding + context_embedding

        hist_mask = (teams.flatten(3).sum(-1) != 0).unsqueeze(-1)
        state_embedding = self.torso1(state_embedding) * hist_mask
        state_embedding = self.torso2(state_embedding.flatten(0, 1)).view(T, B, -1)
        state_embedding = self.torso3(state_embedding)

        switch_embeddings = entities_embedding[..., -1, 0, :6, :]
        move_embeddings = self.move_embeddings(move_tokens[..., -1, 0, 0, :])

        key = state_embedding.unsqueeze(-2)
        queries = torch.cat((move_embeddings, switch_embeddings), dim=-2)
        logits = (key @ self.queries(queries).transpose(-2, -1)).squeeze(-2)
        logits *= self.coeff

        value = self.value(state_embedding)
        policy = _legal_policy(logits, mask)
        log_policy = _legal_log_policy(logits, mask)
        return ModelOutput(
            policy=policy, value=value, log_policy=log_policy, logits=logits
        )
