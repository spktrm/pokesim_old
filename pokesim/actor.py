import wandb
import random
import asyncio

import torch
import numpy as np

from abc import ABC, abstractmethod

from typing import Tuple

from pokesim.utils import preprocess
from pokesim.inference import Inference
from pokesim.constants import _DEFAULT_ACTION, _LINE_FEED
from pokesim.structs import EnvStep, ActorStep


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
    async def _choose_action(self, env_step: EnvStep):
        try:
            return await self.choose_action(env_step)
        except Exception:
            import traceback

            traceback.print_exc()

    @abstractmethod
    async def choose_action(self, env_step: EnvStep) -> Tuple[Action, ActorStep]:
        raise NotImplementedError

    def _done_callback(self, reward: int):
        pass


def _get_action(policy: np.ndarray):
    action = random.choices(
        population=list(range(policy.shape[-1])),
        k=1,
        weights=policy.squeeze().tolist(),
    )
    return action[0]


class SelfplayActor(Actor):
    def __init__(self, inference: Inference):
        self.inference = inference
        self.loop = asyncio.get_running_loop()

    async def choose_action(self, env_step: EnvStep) -> Tuple[Action, ActorStep]:
        batch = preprocess(env_step.raw_obs)
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        mask = torch.from_numpy(env_step.legal.astype(np.bool_))
        model_output = self.inference.model(**batch, mask=mask)
        policy = model_output.policy.detach().numpy()
        action = _get_action(policy)
        return (
            Action.from_env_step(env_step).select_action(action),
            # Action.from_env_step(env_step).default(),
            ActorStep(policy=policy, action=np.array([[action]])),
        )


class EvalActor(Actor):
    def __init__(self, eval_queue) -> None:
        self.eval_queue = eval_queue

    async def choose_action(self, env_step: EnvStep) -> Tuple[Action, ActorStep]:
        mask = torch.from_numpy(env_step.legal.astype(np.bool_))
        policy = torch.masked_fill(
            torch.zeros((10,)),
            ~mask,
            float("-inf"),
        ).softmax(-1)
        action = _get_action(policy)
        return (
            Action.from_env_step(env_step).select_action(action),
            None,
        )

    def _done_callback(self, reward: int):
        self.eval_queue.put((1 - reward))
