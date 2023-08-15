import os
import time
import random
import uvloop
import asyncio
import functools
import numpy as np

import torch
import pandas as pd
import torch.nn as nn

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, NamedTuple
from abc import ABC, abstractmethod
from tqdm import tqdm


_DEBUG = False
_NUM_CPUS = 1 if _DEBUG else os.cpu_count()
print(f"Running on `{_NUM_CPUS} Workers`")

_POOL = ThreadPoolExecutor()

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


def preprocess(obs: torch.Tensor) -> TensorDict:
    T, B, *_ = leading_dims = obs.shape
    leading_dims = (T, B)
    reshape_args = (*leading_dims, _NUM_SIDES, _NUM_ACTIVE, -1)

    volatile_status = obs[
        ..., _VOLAILTE_OFFSET : _VOLAILTE_OFFSET + _VOLAILTE_SIZE
    ].reshape(*leading_dims, _NUM_SIDES, _NUM_ACTIVE, -1)
    volatile_status_token = volatile_status >> 8
    volatile_status_level = volatile_status & 0xFF

    teams = obs[..., :_SIDE_CON_OFFSET]
    side_conditions = obs[..., _SIDE_CON_OFFSET:_BOOSTS_OFFSET]
    boosts = obs[..., _BOOSTS_OFFSET:_FIELD_OFFSET]
    field = obs[..., _FIELD_OFFSET:]

    return {
        "teams": teams.reshape(*leading_dims, _NUM_SIDES, -1, _POKEMON_SIZE),
        "side_conditions": side_conditions.reshape(*leading_dims, _NUM_SIDES, -1),
        "volatile_status": np.stack(
            [volatile_status_token, volatile_status_level], axis=-2
        ),
        "boosts": boosts.reshape(*reshape_args),
        "field": field.reshape(*leading_dims, -1),
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
    def __init__(self, inference: "Inference"):
        self.inference = inference
        self.loop = asyncio.get_running_loop()

    def _get_policy_and_action(self, policy: np.ndarray):
        action = random.choices(
            population=list(range(policy.shape[-1])),
            k=1,
            weights=policy.squeeze().tolist(),
        )
        return action[0]

    async def choose_action(self, env_step: EnvStep):
        fut = await self.inference.compute(env_step)
        await fut
        actor_step = fut.result()
        return Action.from_env_step(env_step).default()
        # return Action.from_env_step(env_step).select_action(action_index)


class Inference:
    def __init__(
        self,
        model,
        device,
        batch_size: int = None,
        timeout: float = 0.1,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size or 1024
        self.timeout = timeout

        self.full_queue = asyncio.Queue()

        self.optimized = False if batch_size is None else True
        self.run_in_executor = False

        self.loop = asyncio.get_event_loop()

    async def compute(self, step: EnvStep) -> ActorStep:
        fut = self.loop.create_future()
        await self.full_queue.put((step, fut))
        return fut

    async def _get(self):
        if self.optimized:
            item = await self.full_queue.get()
        else:
            item = await asyncio.wait_for(self.full_queue.get(), timeout=self.timeout)
        return item

    async def _run_callback(self):
        last_optim_time = None
        futs = []
        buffer = []
        while True:
            try:
                step, fut = await self._get()
            except:
                if buffer:
                    last_optim_time = time.time()
                    self.batch_size = len(buffer)
                    print(f"Updating `batch_size={self.batch_size}`")

                    yield futs, buffer

                    futs.clear()
                    buffer.clear()
            else:
                futs.append(fut)
                buffer.append(step)

                if len(buffer) >= self.batch_size:
                    if (
                        not self.optimized
                        and last_optim_time
                        and (time.time() - last_optim_time) >= 60
                    ):
                        self.optimized = True
                        print("Using Optimized `await queue.get()`")

                    yield futs, buffer

                    futs.clear()
                    buffer.clear()

    async def run(self):
        outputs = None
        loader = self._run_callback()
        while True:
            batch = await anext(loader)
            futs, buffer = batch
            outputs = await self._process_batch(buffer)
            self._set_outputs(outputs, futs)

    async def _process_batch(self, buffer):
        return await self.loop.run_in_executor(
            _POOL, functools.partial(self._forward_model, buffer)
        )

    def _set_outputs(self, outputs, futs):
        for fut, *output in zip(futs, *(o.squeeze(0) for o in outputs)):
            model_output = ModelOutput(*(arr.cpu().squeeze().numpy() for arr in output))
            fut.set_result(model_output)

    def _forward_model(self, buffer: List[EnvStep]) -> ActorStep:
        batch = preprocess(np.stack([step.raw_obs for step in buffer], axis=1))
        batch = {
            k: torch.from_numpy(v.astype(np.int64)).to(self.device)
            for k, v in batch.items()
        }
        mask = torch.from_numpy(
            np.stack([step.legal for step in buffer], axis=1).astype(np.bool_)
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**batch, mask=mask)
        return out


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

            action = await policy_fn(env_step)

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

        moves_df = pd.read_csv("./dist/src/moves.csv", index_col="species")
        abilities_df = pd.read_csv("./dist/src/abilities.csv", index_col="species")
        items_df = pd.read_csv("./dist/src/items.csv", index_col="species")

        abilities_init = abilities_df.values.astype(np.float32)
        items_init = items_df.values.astype(np.float32)
        moves_init = moves_df.values.astype(np.float32)

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
        return torch.where(mask, onehot, unknown / t_unknown.clamp(min=1))

    def forward_ability(
        self,
        species_embedding: torch.Tensor,
        ability_token: torch.Tensor,
        mask: torch.Tensor,
    ):
        mask = mask.unsqueeze(-1)
        known = self.abilities_onehot((ability_token - 1).clamp(min=0))
        unknown = torch.matmul(species_embedding, self.all_abilities.weight)
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
        unknown = torch.matmul(species_embedding, self.all_items.weight)
        unknown = unknown / unknown.sum(-1, keepdim=True).clamp(min=1)
        return torch.where(mask, known, unknown)

    def forward_moveset(
        self,
        species_embedding: torch.Tensor,
        move_tokens: torch.Tensor,
        mask: torch.Tensor,
    ):
        known = self.moves_onehot(move_tokens).sum(-2)
        all_unknown = torch.matmul(species_embedding, self.all_moves.weight)
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

        item_mask = item_token > 0
        item_embedding = self.forward_item(species_embedding, item_token, item_mask)

        move_mask = (species_mask & (move_tokens.sum(-1) != 0)).unsqueeze(-1)
        move_embedding = self.forward_moveset(species_embedding, move_tokens, move_mask)

        return species_embedding, ability_embedding, item_embedding, move_embedding


@torch.jit.script
def _legal_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """A soft-max policy that respects legal_actions."""
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    l_min = logits.min(dim=-1, keepdim=True).values
    logits = torch.where(legal_actions, logits, l_min)
    logits -= logits.max(dim=-1, keepdim=True).values
    logits *= legal_actions
    exp_logits = torch.where(
        legal_actions, torch.exp(logits), 0
    )  # Illegal actions become 0.
    exp_logits_sum = torch.sum(exp_logits, dim=-1, keepdim=True)
    return exp_logits / exp_logits_sum


@torch.jit.script
def _legal_log_policy(
    logits: torch.Tensor, legal_actions: torch.Tensor
) -> torch.Tensor:
    """Return the log of the policy on legal action, 0 on illegal action."""
    # logits_masked has illegal actions set to -inf.
    logits_masked = logits + torch.log(legal_actions)
    max_legal_logit = logits_masked.max(dim=-1, keepdim=True).values
    logits_masked = logits_masked - max_legal_logit
    # exp_logits_masked is 0 for illegal actions.
    exp_logits_masked = torch.exp(logits_masked)

    baseline = torch.log(torch.sum(exp_logits_masked, dim=-1, keepdim=True))
    # Subtract baseline from logits. We do not simply return
    #     logits_masked - baseline
    # because that has -inf for illegal actions, or
    #     legal_actions * (logits_masked - baseline)
    # because that leads to 0 * -inf == nan for illegal actions.
    log_policy = torch.multiply(legal_actions, (logits - max_legal_logit - baseline))
    return log_policy


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EntityEmbedding()

        self.torso = nn.Sequential(nn.Linear(876, 128), nn.ReLU(), nn.Linear(128, 128))
        self.logits = nn.Sequential(nn.ReLU(), nn.Linear(2 * 128, 10))
        self.value = nn.Sequential(nn.ReLU(), nn.Linear(2 * 128, 1))

    def forward(
        self,
        teams: torch.Tensor,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        field: torch.Tensor,
        mask: torch.Tensor,
    ):
        species_token = teams[..., 0] + 1
        item_token = teams[..., 1] + 1
        ability_token = teams[..., 2] + 1
        move_tokens = teams[..., -4:] + 1
        (
            species_embedding,
            item_embedding,
            ability_embedding,
            moveset_embedding,
        ) = self.embedding(species_token, ability_token, item_token, move_tokens)
        entity_embedding = torch.cat(
            (species_embedding, item_embedding, ability_embedding, moveset_embedding),
            dim=-1,
        )
        entities_embedding = self.torso(entity_embedding)
        side_embedding = entities_embedding.mean(-2).flatten(2)
        logits = self.logits(side_embedding)
        value = self.value(side_embedding)
        policy = _legal_policy(logits, mask)
        log_policy = _legal_log_policy(logits, mask)
        return ModelOutput(
            policy=policy, value=value, log_policy=log_policy, logits=logits
        )


async def run_process(worker_index: int, inference: Inference):
    command = f"node ./sim.js {worker_index}"

    process = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    manager = Manager(
        process=process,
        actor1=SelfplayActor(inference),
        actor2=SelfplayActor(inference),
    )
    await manager.run()


async def _datagen(model: nn.Module):
    inference = Inference(model, "cpu", batch_size=_NUM_CPUS)

    acting_tasks = [
        run_process(worker_index, inference) for worker_index in range(_NUM_CPUS)
    ]
    inference_task = asyncio.create_task(inference.run())
    all_tasks = acting_tasks + [inference_task]
    await asyncio.gather(*all_tasks)


def datagen(model: nn.Module):
    uvloop.install()
    asyncio.run(_datagen(model))


def main():
    model = Model()
    datagen(model)


if __name__ == "__main__":
    main()
