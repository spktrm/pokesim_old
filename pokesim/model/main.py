import torch
import torch.nn as nn

from pokesim.structs import ModelOutput
from pokesim.rl_utils import _legal_log_policy, _legal_policy
from pokesim.model.embedding import EntityEmbedding


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EntityEmbedding()
        self.move_embeddings = nn.Embedding(self.embedding.moves_shape[0] + 1, 128)
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(7))

        self.boosts_embedding = nn.Linear(2 * 7, 128)

        self.pseudoweather_onehot = nn.Embedding.from_pretrained(torch.eye(9)[..., 1:])
        self.weather_onehot = nn.Embedding.from_pretrained(torch.eye(9))
        self.terrain_onehot = nn.Embedding.from_pretrained(torch.eye(6))

        self.field_embedding = nn.Linear(8 + 9 + 6, 128)

        self.sidecon_onehot = nn.Embedding.from_pretrained(torch.eye(16)[..., 1:])
        self.sidecon_embedding = nn.Linear(2 * 15, 128)

        self.volatile_onehot = nn.Embedding.from_pretrained(torch.eye(106)[..., 1:])
        self.volatile_embedding = nn.Linear(2 * 105, 128)

        self.torso1 = nn.Sequential(nn.Linear(888, 128), nn.ReLU(), nn.Linear(128, 128))
        self.torso2 = nn.Sequential(nn.ReLU(), nn.Linear(3 * 128, 128))
        self.torso3 = nn.Sequential(
            nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)
        )
        self.value = nn.Sequential(nn.ReLU(), nn.Linear(128, 1))

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
        entities_embedding = self.torso1(entity_embedding)
        side_embedding = self.torso2(entities_embedding.mean(-2).flatten(2))
        side_embedding = side_embedding + self.boosts_embedding(boosts.flatten(2) / 6)

        pseudoweather = field[..., :9].view(T, B, 3, 3)
        pseudoweather_tokens = pseudoweather[..., 0]
        psuedoweather_onehot = self.pseudoweather_onehot(pseudoweather_tokens).sum(-2)
        weather_onehot = self.weather_onehot(field[..., 10])
        terrain_onehot = self.terrain_onehot(field[..., 13])

        field_onehot = torch.cat(
            (psuedoweather_onehot, weather_onehot, terrain_onehot), dim=-1
        )
        side_embedding = side_embedding + self.field_embedding(field_onehot)

        volatile_onehot = (
            self.volatile_onehot(volatile_status[..., 0, :]).sum(-2).flatten(2)
        )
        side_embedding = side_embedding + self.volatile_embedding(volatile_onehot)

        sidecon_onehot = (
            self.sidecon_onehot((side_conditions > 0).to(torch.long)).sum(-2).flatten(2)
        )
        side_embedding = side_embedding + self.sidecon_embedding(sidecon_onehot)

        switch_embeddings = entities_embedding[..., 0, :6, :]
        move_embeddings = self.move_embeddings(move_tokens[..., 0, 0, :])

        key = side_embedding.unsqueeze(-2)
        logits = torch.cat(
            (
                key @ move_embeddings.transpose(-2, -1),
                key @ switch_embeddings.transpose(-2, -1),
            ),
            dim=-1,
        ).squeeze(-2)

        value = self.value(side_embedding)
        policy = _legal_policy(logits, mask)
        log_policy = _legal_log_policy(logits, mask)
        return ModelOutput(
            policy=policy, value=value, log_policy=log_policy, logits=logits
        )
