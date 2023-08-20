import pytest

import numpy as np

from pokesim.structs import Trajectory, _DTYPES


def test_serialize_trajectory():
    t = Trajectory(
        **{
            k: np.random.random((4, 100, 1)).astype(_DTYPES[k])
            for k in Trajectory._fields
        }
    )
    st = t.serialize()
    t_ = Trajectory.deserialize(st)

    for k in Trajectory._fields:
        assert np.all(getattr(t, k) == getattr(t_, k))
        assert getattr(t, k).shape == getattr(t_, k).shape
