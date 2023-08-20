import torch
import numpy as np

from pokesim.types import TensorDict
from pokesim.constants import (
    _NUM_PLAYERS,
    _NUM_ACTIVE,
    _VOLAILTE_OFFSET,
    _SIDE_CON_OFFSET,
    _FIELD_OFFSET,
    _BOOSTS_OFFSET,
    _POKEMON_SIZE,
)


def unpackbits(x: np.ndarray, num_bits: int):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def preprocess(obs: torch.Tensor) -> TensorDict:
    T, B, *_ = leading_dims = obs.shape
    leading_dims = (T, B)
    reshape_args = (*leading_dims, _NUM_PLAYERS, _NUM_ACTIVE, -1)

    volatile_status = obs[..., _VOLAILTE_OFFSET:_BOOSTS_OFFSET].reshape(
        *leading_dims, _NUM_PLAYERS, _NUM_ACTIVE, -1
    )
    volatile_status_token = volatile_status >> 8
    volatile_status_level = volatile_status & 0xFF

    teams = obs[..., :_SIDE_CON_OFFSET]
    side_conditions = obs[..., _SIDE_CON_OFFSET:_VOLAILTE_OFFSET]
    boosts = obs[..., _BOOSTS_OFFSET:_FIELD_OFFSET]
    field = obs[..., _FIELD_OFFSET:]

    return {
        "teams": teams.reshape(*leading_dims, 3, -1, _POKEMON_SIZE).astype(np.int64),
        "side_conditions": side_conditions.reshape(
            *leading_dims, _NUM_PLAYERS, -1
        ).astype(np.int64),
        "volatile_status": np.stack(
            [volatile_status_token, volatile_status_level], axis=-2
        ).astype(np.int64),
        "boosts": boosts.reshape(*reshape_args).astype(np.int64),
        "field": field.reshape(*leading_dims, -1).astype(np.int64),
    }


def get_arr(data) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.short)
    arr.setflags(write=False)
    return arr
