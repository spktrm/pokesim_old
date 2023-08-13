import os
import json
import uvloop
import asyncio
import numpy as np

import torch
import torch.nn as nn

from typing import Dict, List, Any, NamedTuple
from abc import ABC, abstractmethod
from tqdm import tqdm

_NUM_CPUS = os.cpu_count()
print(f"Running on `{_NUM_CPUS} Workers`")

_N_STATE_BITS = 227
_STATE_BYTES = 2 * _N_STATE_BITS
_LINE_FEED = 10
_DEFAULT_ACTION = 255

_NUM_ACTIVE = 1
_NUM_SIDES = 2
_POKEMON_SIZE = 11

_TEAM_SIZE = _POKEMON_SIZE * 6 * _NUM_SIDES

_SIDE_CON_OFFSET = _TEAM_SIZE
_SIDE_CON_SIZE = 15 * _NUM_SIDES

_VOLAILTE_OFFSET = _SIDE_CON_OFFSET + _SIDE_CON_SIZE
_VOLAILTE_SIZE = 10 * _NUM_SIDES * _NUM_ACTIVE

_BOOSTS_OFFSET = _VOLAILTE_OFFSET + _VOLAILTE_SIZE
_BOOSTS_SIZE = 7 * _NUM_SIDES * _NUM_ACTIVE

_FIELD_OFFSET = _BOOSTS_OFFSET + _BOOSTS_SIZE


progress = tqdm()


TensorType = np.ndarray | torch.Tensor
TensorDict = Dict[str, TensorType | Any]


def unpackbits(x: np.ndarray, num_bits: int):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


class EnvStep(NamedTuple):
    game_id: np.ndarray
    player_id: np.ndarray
    raw_obs: np.ndarray
    rewards: np.ndarray
    valid: np.ndarray
    legal: np.ndarray

    @classmethod
    def from_data(cls, data: bytes) -> "EnvStep":
        _r = lambda arr: arr.reshape(1, 1, -1)
        state = _r(get_arr(data))
        return cls(
            game_id=state[..., 1],
            player_id=state[..., 2],
            raw_obs=state[..., 6:-10],
            rewards=_r(np.array([0, 0])),
            valid=(1 - state[..., 3]).astype(bool),
            legal=state[..., -10:].astype(bool),
        )

    def get_leading_dims(self):
        return self.valid.shape

    def preprocess(self):
        leading_dims = self.get_leading_dims()
        reshape_args = (*leading_dims, _NUM_SIDES, _NUM_ACTIVE, -1)

        volatile_status = self.raw_obs[
            ..., _VOLAILTE_OFFSET : _VOLAILTE_OFFSET + _VOLAILTE_SIZE
        ].reshape(*leading_dims, _NUM_SIDES, _NUM_ACTIVE, -1)
        volatile_status_token = volatile_status >> 8
        volatile_status_level = volatile_status & 0xFF

        teams = self.raw_obs[..., :_SIDE_CON_OFFSET]
        side_conditions = self.raw_obs[..., _SIDE_CON_OFFSET:_BOOSTS_OFFSET]
        boosts = self.raw_obs[..., _BOOSTS_OFFSET:_FIELD_OFFSET]
        field = self.raw_obs[..., _FIELD_OFFSET:]

        return {
            "teams": teams.reshape(*leading_dims, _NUM_SIDES, -1, _POKEMON_SIZE),
            "side_conditions": side_conditions.reshape(*leading_dims, _NUM_SIDES, -1),
            "volatile_status": np.stack(
                [volatile_status_token, volatile_status_level], axis=-2
            ),
            "boosts": boosts.reshape(*reshape_args),
            "field": field.reshape(*leading_dims, -1),
            "legal": self.legal,
        }


class ModelOutput(NamedTuple):
    policy: TensorType
    value: TensorType
    log_policy: TensorType
    logits: TensorType


class ActorStep(NamedTuple):
    policy: np.ndarray
    action: np.ndarray


class TimeStep(NamedTuple):
    id: str
    actor: ActorStep
    env: EnvStep


class Batch(NamedTuple):
    player_id: TensorType
    obs: TensorType
    rewards: TensorType
    valid: TensorType
    legal: TensorType
    policy: TensorType
    action: TensorType

    def _to(self, callback: callable):
        return Batch(**{key: callback(value) for key, value in self._asdict().items()})

    def to_numpy(self):
        def _trans(value):
            if isinstance(value, np.ndarray):
                return value
            else:
                return value.cpu().numpy()

        return self._to(_trans)

    def to_torch(self, device: str = "cpu"):
        def _trans(value):
            if isinstance(value, torch.Tensor):
                return value
            else:
                return torch.from_numpy(value).to(device, non_blocking=True)

        return self._to(_trans)


class Action:
    def __init__(self, game_id: int, player_id: int):
        self.game_id = game_id
        self.player_id = player_id

    @classmethod
    def from_env_step(cls, env_step: EnvStep):
        return cls(env_step.game_id.item(), env_step.player_id.item())

    def select_action(self, action_index: int):
        return bytearray([self.game_id, self.player_id, action_index, _LINE_FEED])

    def default(self):
        return self.select_action(_DEFAULT_ACTION)


class Actor(ABC):
    @abstractmethod
    def choose_action(self):
        raise NotImplementedError


class SelfplayActor(Actor):
    def __init__(self, model: nn.Module):
        self.model = model

    def choose_action(self, env_step: EnvStep):
        obs = env_step.preprocess()
        model_output = self.model(obs)
        return Action.from_env_step(env_step).default()


def get_arr(data) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.short)
    arr.setflags(write=False)
    return arr


class Manager:
    def __init__(
        self, process: asyncio.subprocess.Process, actor1: Actor, actor2: Actor
    ):
        self.process = process
        self.actor1 = actor1
        self.actor2 = actor2

    async def run(self):
        traj = []
        dones = 0
        while True:
            data = await self.process.stdout.read(_STATE_BYTES)

            env_step = EnvStep.from_data(data)

            if env_step.player_id == 0:
                policy_fn = self.actor1.choose_action

            elif env_step.player_id == 1:
                policy_fn = self.actor2.choose_action

            action = policy_fn(env_step)

            self.process.stdin.write(action)

            dones += 1 - env_step.valid
            traj.append(env_step)

            if dones >= 2:
                progress.update(len(traj))
                dones = 0
                traj = []


async def stderr(process: asyncio.subprocess.Process):
    while True:
        err = await process.stderr.readline()
        print(err)


class EntityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        with open("src/data.json", "w") as f:
            moves_df = json.loads(f)
        with open("abilities.csv", "w") as f:
            abilities_df = json.loads(f)
        with open("items.csv", "w") as f:
            items_df = json.loads(f)

        abilities_init = abilities_df[
            list(sorted([c for c in abilities_df.columns if c != "Species"]))
        ].values.astype(np.float32)

        items_init = items_df[
            list(sorted([c for c in items_df.columns if c != "Species"]))
        ].values.astype(np.float32)

        moves_init = moves_df[
            list(sorted([c for c in moves_df.columns if c != "Species"]))
        ].values.astype(np.float32)

        embeddings = torch.eye(moves_df.shape[0] + 1)[:, 1:]
        self.species_onehot = nn.Embedding.from_pretrained(embeddings)

        self.all_abilities = nn.Embedding.from_pretrained(
            torch.from_numpy(abilities_init)
        )
        self.abilities_onehot = nn.Embedding.from_pretrained(
            torch.eye(abilities_init.shape[1])
        )
        self.all_items = nn.Embedding.from_pretrained(torch.from_numpy(items_init))
        self.items_onehot = nn.Embedding.from_pretrained(torch.eye(items_init.shape[1]))

        self.all_moves = nn.Embedding.from_pretrained(torch.from_numpy(moves_init))
        self.moves_onehot = nn.Embedding.from_pretrained(
            torch.eye(moves_init.shape[1] + 1)[:, 1:]
        )

    def forward_species(self, token: torch.Tensor, mask: torch.Tensor):
        # Assume shape is [..., 6]
        mask = mask.unsqueeze(-1)
        onehot = self.species_onehot(token)
        unknown = torch.ones_like(onehot) - onehot.sum(-2, keepdim=True)
        t_unknown = unknown.sum(-1, keepdim=True)
        out = torch.where(mask, onehot, unknown / t_unknown.clamp(min=1))
        return out

    def forward_ability(
        self,
        species_embedding: torch.Tensor,
        ability_token: torch.Tensor,
        mask: torch.Tensor,
    ):
        mask = mask.unsqueeze(-1)
        known = self.abilities_onehot((ability_token - 1).clamp(min=0))
        unknown = species_embedding @ self.all_abilities.weight
        unknown = unknown / unknown.sum(-1, keepdim=True).clamp(min=1)
        return torch.where(mask, known, unknown)

    def forward_item(
        self,
        species_embedding: torch.Tensor,
        item_token: torch.Tensor,
        mask: torch.Tensor,
    ):
        mask = mask.unsqueeze(-1)
        known = self.items_onehot((item_token - 1).clamp(min=0))
        unknown = species_embedding @ self.all_items.weight
        unknown = unknown / unknown.sum(-1, keepdim=True).clamp(min=1)
        return torch.where(mask, known, unknown)

    def forward_moveset(
        self,
        species_embedding: torch.Tensor,
        move_tokens: torch.Tensor,
        mask: torch.Tensor,
    ):
        known = self.moves_onehot(move_tokens).sum(-2)
        all_unknown = species_embedding @ self.all_moves.weight
        unknown = all_unknown - known
        unknown = unknown / unknown.sum(-1, keepdim=True).clamp(min=1)
        num_missing = 4 - (known > 0).sum(-1, keepdim=True)
        return torch.where(mask, known + num_missing * unknown, 4 * unknown)

    def forward(
        self,
        species_token: torch.Tensor,
        ability_token: torch.Tensor,
        item_token: torch.Tensor,
        move_tokens: torch.Tensor,
    ):
        species_mask = species_token > 0
        species_embedding = self.forward_species(species_token, species_mask)

        ability_mask = ability_token > 0
        ability_embedding = self.forward_ability(
            species_embedding, ability_token, ability_mask
        )

        item_mask = ability_token > 0
        item_embedding = self.forward_item(species_embedding, item_token, item_mask)

        move_mask = (species_mask & (move_tokens.sum(-1) != 0)).unsqueeze(-1)
        move_embedding = self.forward_moveset(species_embedding, move_tokens, move_mask)

        return species_embedding, ability_embedding, item_embedding, move_embedding


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EntityEmbedding()

    def forward(self, x):
        return x


async def run_process(worker_index: int, model: nn.Module):
    command = f"./sim.sim {worker_index}"

    process = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    manager = Manager(
        process=process,
        actor1=SelfplayActor(model),
        actor2=SelfplayActor(model),
    )
    await manager.run()


async def _datagen(model: nn.Module):
    await asyncio.gather(
        *[run_process(worker_index, model) for worker_index in range(_NUM_CPUS)]
    )


def datagen(model: nn.Module):
    uvloop.install()
    asyncio.run(_datagen(model))


def main():
    model = Model()
    datagen(model)


if __name__ == "__main__":
    main()
