import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence

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


def ghostmax(x, dim=None):
    # subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    # compute exponentials
    exp_x = torch.exp(x)
    # compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class ResBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        output_size: int = None,
        num_layers: int = 2,
        bias: bool = True,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        output_size = output_size or input_size
        hidden_size = hidden_size or input_size
        sizes = (
            [input_size]
            + [hidden_size for _ in range(max(0, num_layers - 1))]
            + [output_size]
        )
        layers = []
        for size1, size2 in zip(sizes, sizes[1:]):
            layer = [
                nn.ReLU(),
                _layer_init(nn.Linear(size1, size2, bias=bias)),
            ]
            if use_layer_norm:
                layer.insert(0, RMSNorm(size1))
            layers += layer
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class ResNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        output_size: int = None,
        num_resblocks: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    use_layer_norm=use_layer_norm,
                )
                for i in range(num_resblocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int] = None,
        bias: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        layers = []
        for size1, size2 in zip(layer_sizes, layer_sizes[1:]):
            layer = [
                nn.ReLU(),
                _layer_init(nn.Linear(size1, size2, bias=bias)),
            ]
            if use_layer_norm:
                layer.insert(0, RMSNorm(size1))
            layers += layer
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x: torch.Tensor):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        key_size: int,
        with_bias: bool = True,
        value_size: int = None,
        model_size: int = None,
    ):
        super().__init__()
        self.key_size = key_size
        self.num_heads = num_heads
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.denom = 1 / math.sqrt(key_size)

        self.queries = _layer_init(
            nn.Linear(self.model_size, num_heads * self.key_size, bias=with_bias)
        )
        self.keys = _layer_init(
            nn.Linear(self.model_size, num_heads * self.key_size, bias=with_bias)
        )
        self.values = _layer_init(
            nn.Linear(self.model_size, num_heads * self.value_size, bias=with_bias)
        )
        self.final_proj = _layer_init(
            nn.Linear(self.value_size * num_heads, self.model_size)
        )

    def _linear_projection(
        self, x: torch.Tensor, mod: nn.Module, head_size: int
    ) -> torch.Tensor:
        y = mod(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        *leading_dims, sequence_length, _ = query.shape

        query_heads = self._linear_projection(query, self.queries, self.key_size)
        key_heads = self._linear_projection(key, self.keys, self.key_size)
        value_heads = self._linear_projection(value, self.values, self.value_size)

        attn_logits = torch.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = attn_logits * self.denom
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = torch.where(mask, attn_logits, -1e30)
        attn_weights = attn_logits.softmax(-1)

        attn = torch.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = torch.reshape(attn, (*leading_dims, sequence_length, -1))

        return self.final_proj(attn)


class Transformer(nn.Module):
    def __init__(
        self,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_value_size: int,
        transformer_model_size: int,
        resblocks_num_before: int,
        resblocks_num_after: int,
        resblocks_hidden_size: int = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.attn = nn.ModuleList(
            [
                MultiHeadAttention(
                    transformer_num_heads,
                    transformer_key_size,
                    value_size=transformer_value_size,
                    model_size=transformer_model_size,
                )
                for i in range(transformer_num_layers)
            ]
        )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.ModuleList(
                [RMSNorm(transformer_model_size) for _ in range(transformer_num_layers)]
            )
        self.resnet_before = ResNet(
            input_size=transformer_model_size,
            hidden_size=resblocks_hidden_size,
            output_size=transformer_model_size,
            num_resblocks=resblocks_num_before,
            use_layer_norm=use_layer_norm,
        )
        self.resnet_after = ResNet(
            input_size=transformer_model_size,
            hidden_size=resblocks_hidden_size,
            output_size=transformer_model_size,
            num_resblocks=resblocks_num_after,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.resnet_before(x)
        for i, attn in enumerate(self.attn):
            x1 = x
            if self.use_layer_norm:
                ln = self.ln[i]
                x1 = ln(x1)
            x1 = F.relu(x1)
            logits_mask = mask[..., None, None, :]
            x1 = attn(x1, x1, x1, logits_mask)
            x1 = torch.where(mask.unsqueeze(-1), x1, 0)
            x = x + x1
        x = self.resnet_after(x)
        x = torch.where(mask.unsqueeze(-1), x, 0)
        return x


class ToVector(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.net = MLP([input_size] + hidden_sizes, use_layer_norm=use_layer_norm)

        out_layers = [
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_sizes[-1], hidden_sizes[-1])),
        ]
        if use_layer_norm:
            out_layers.insert(0, RMSNorm(hidden_sizes[-1]))
        self.out = nn.Sequential(*out_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.mean(-2)
        x = self.out(x)
        return x


class PointerLogits(nn.Module):
    def __init__(
        self,
        query_input_size: int,
        keys_input_size: int,
        num_layers_query: int = 2,
        num_layers_keys: int = 2,
        key_size: int = 64,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.query_mlp = MLP(
            [query_input_size]
            + [query_input_size for _ in range(num_layers_query - 1)]
            + [key_size],
            use_layer_norm=use_layer_norm,
        )
        self.keys_mlp = MLP(
            [keys_input_size]
            + [keys_input_size for _ in range(num_layers_keys - 1)]
            + [key_size],
            use_layer_norm=use_layer_norm,
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        query = self.query_mlp(query)
        keys = self.keys_mlp(keys)

        logits = keys @ query.transpose(-2, -1)
        return logits


class Model(nn.Module):
    def __init__(self, entity_size: int = 32, vector_size: int = 128):
        super().__init__()
        self.embedding = EntityEmbedding()

        self.switch_embedding = nn.Parameter(torch.randn(entity_size))
        self.move_embeddings = _layer_init(
            nn.Embedding(self.embedding.moves_shape[1] + 1, entity_size)
        )
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(7))

        self.units_lin = _layer_init(nn.Linear(890, entity_size))
        self.units_mlp = MLP([2 * entity_size, entity_size])

        self.entity_transformer = Transformer(
            transformer_num_layers=1,
            transformer_num_heads=2,
            transformer_key_size=entity_size // 2,
            transformer_value_size=entity_size // 2,
            transformer_model_size=entity_size,
            resblocks_num_before=1,
            resblocks_num_after=1,
            resblocks_hidden_size=entity_size // 2,
        )
        self.to_vector = ToVector(entity_size, [entity_size, vector_size])

        self.pseudoweather_onehot = nn.Embedding.from_pretrained(torch.eye(9)[..., 1:])
        self.weather_onehot = nn.Embedding.from_pretrained(torch.eye(9))
        self.terrain_onehot = nn.Embedding.from_pretrained(torch.eye(6))
        self.sidecon_onehot = nn.Embedding.from_pretrained(torch.eye(16)[..., 1:])
        self.volatile_onehot = nn.Embedding.from_pretrained(torch.eye(106)[..., 1:])
        self.side_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.public_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.context_embedding = nn.Sequential(
            _layer_init(nn.Linear(2 * 7 + 8 + 9 + 6 + 2 * 15 + 2 * 105, vector_size)),
            ResNet(vector_size, vector_size),
        )

        self.action_hist = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(4, 8, 3, 1, bias=False, padding="same"),
            nn.AvgPool1d(2),
            nn.ReLU(),
            nn.Conv1d(8, 16, 3, 1, bias=False, padding="same"),
            nn.AvgPool1d(2),
            nn.Flatten(-2),
            MLP([_NUM_HISTORY * entity_size, vector_size]),
        )

        self.torso1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(4, 8, 3, 1, bias=False, padding="same"),
            nn.AvgPool1d(2),
            nn.ReLU(),
            nn.Conv1d(8, 16, 3, 1, bias=False, padding="same"),
            nn.AvgPool1d(2),
            nn.Flatten(-2),
            MLP([_NUM_HISTORY * vector_size, vector_size]),
        )
        self.torso2 = ResNet(vector_size)

        # self.action_transformer = Transformer(
        #     transformer_num_layers=1,
        #     transformer_num_heads=2,
        #     transformer_key_size=entity_size // 2,
        #     transformer_value_size=entity_size // 2,
        #     transformer_model_size=entity_size,
        #     resblocks_num_before=1,
        #     resblocks_num_after=1,
        #     resblocks_hidden_size=entity_size // 2,
        # )
        self.query_resnet = ResNet(vector_size)
        self.keys_mlp = MLP([entity_size, entity_size // 4])
        self.pointer = PointerLogits(
            vector_size,
            self.keys_mlp.layer_sizes[-1],
            key_size=self.keys_mlp.layer_sizes[-1],
            num_layers_keys=0,
            num_layers_query=1,
        )

        self.value = MLP([vector_size, vector_size, vector_size, 1])

    def forward(
        self,
        teams: torch.Tensor,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        field: torch.Tensor,
        mask: torch.Tensor,
        action_hist: torch.Tensor,
    ):
        T, B, H, *_ = teams.shape

        teams_ = teams + 1
        species_token = teams_[..., 0]
        side_token = torch.zeros_like(species_token)
        side_token[..., 2:, :] = 1
        public_token = torch.zeros_like(species_token)
        public_token[..., 1:, :] = 1
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

        entity_embeddings = torch.cat(
            (
                species_embedding,
                item_embedding,
                ability_embedding,
                moveset_embedding,
                hp,
                self.active_onehot(active_token),
                self.fainted_onehot(fainted_token),
                self.status_onehot(status_token),
                self.side_onehot(side_token),
            ),
            dim=-1,
        )
        entity_embeddings = self.units_lin(entity_embeddings)
        entity_embeddings = self.units_mlp(
            torch.stack(
                (
                    torch.cat(
                        (
                            entity_embeddings[..., 0, :, :],
                            entity_embeddings[..., 1, :, :],
                        ),
                        dim=-1,
                    ),
                    torch.cat(
                        (
                            entity_embeddings[..., 2, :, :],
                            entity_embeddings[..., 2, :, :],
                        ),
                        dim=-1,
                    ),
                ),
                dim=-3,
            )
        )
        entity_embeddings = entity_embeddings.flatten(-3, -2)
        entity_embeddings = self.entity_transformer(
            entity_embeddings,
            torch.ones_like(entity_embeddings[..., 0].squeeze(-1), dtype=torch.bool),
        )
        entities_embedding = self.to_vector(entity_embeddings)

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

        user = action_hist[..., 0].clamp(min=0)
        user = torch.where(user >= 12, user - 6, user)
        move = action_hist[..., 1] + 1

        action_move_embeddings = self.move_embeddings(move)
        action_embeddings = torch.where(
            (move > 0).unsqueeze(-1),
            action_move_embeddings,
            self.switch_embedding.expand(T, B, H, 4, -1),
        )
        action_hist_mask = action_hist[..., 0] >= 0

        user_index = torch.arange(T * B * H, device=user.device).unsqueeze(-1)
        user_index *= entity_embeddings.shape[-2]
        user_index = user_index + user.flatten(0, -2)
        user_embeddings = torch.embedding(entity_embeddings.flatten(0, -2), user_index)
        user_embeddings = user_embeddings.view(T, B, H, 4, -1)

        action_hist_embeddings = (
            (action_embeddings + user_embeddings) * action_hist_mask.unsqueeze(-1)
        ).flatten(0, -3)
        action_hist_embedding = self.action_hist(action_hist_embeddings)
        action_hist_embedding = action_hist_embedding.view(T, B, H, -1)

        state_embedding = entities_embedding + context_embedding + action_hist_embedding

        hist_mask = (teams.flatten(3).sum(-1) != 0).unsqueeze(-1)
        state_embedding = self.torso1((state_embedding * hist_mask).flatten(0, 1))
        state_embedding = self.torso2(state_embedding)
        state_embedding = state_embedding.view(T, B, -1)

        switch_embeddings = entity_embeddings[..., -1, :6, :]
        move_embeddings = self.move_embeddings(move_tokens[..., -1, 0, 0, :])

        context_actions = torch.cat(
            (
                switch_embeddings[..., 0, :]
                .unsqueeze(-2)
                .expand(T, B, move_embeddings.shape[2], -1)
                * torch.sigmoid(move_embeddings),
                switch_embeddings
                * torch.sigmoid(self.switch_embedding).expand(
                    T, B, switch_embeddings.shape[2], -1
                ),
            ),
            dim=-2,
        )
        # context_actions = self.action_transformer(
        #     context_actions.flatten(0, 1),
        #     mask.flatten(0, 1),
        # ).view(T, B, 10, -1)

        query = self.query_resnet(state_embedding).unsqueeze(-2)
        keys = self.keys_mlp(context_actions)
        logits = self.pointer(query, keys).flatten(2)

        value = self.value(state_embedding)
        policy = _legal_policy(logits, mask)
        log_policy = _legal_log_policy(logits, mask)

        return ModelOutput(policy, value, log_policy, logits)
