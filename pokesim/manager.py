import asyncio
import multiprocessing as mp

from pokesim.structs import EnvStep, TimeStep, Trajectory
from pokesim.actor import Actor, Action, EvalActor
from pokesim.constants import _STATE_BYTES


class Manager:
    def __init__(
        self,
        worker_index: int,
        process: asyncio.subprocess.Process,
        actor1: Actor,
        actor2: Actor,
    ):
        self.worker_index = worker_index
        self.process = process
        self.actor1 = actor1
        self.actor2 = actor2
        self.is_eval = isinstance(actor1, EvalActor) or isinstance(actor2, EvalActor)

    async def run(self, progress_queue: mp.Queue, learn_queue: mp.Queue = None):
        traj = {}
        dones = {}

        while True:
            data = await self.process.stdout.readuntil(b"\n\n")

            env_step = EnvStep.from_data(data)

            if env_step.player_id == 0:
                actor = self.actor1

            elif env_step.player_id == 1:
                actor = self.actor2

            policy_fn = actor._choose_action

            done = 1 - env_step.valid
            game_id = env_step.game_id.item()

            if game_id not in dones:
                dones[game_id] = 0

            dones[game_id] += done
            if game_id not in traj:
                traj[game_id] = []

            if not done:
                if env_step.legal.sum() > 0:
                    action, actor_step = await policy_fn(env_step)
                    if learn_queue is not None:
                        time_step = TimeStep(
                            id=game_id,
                            actor=actor_step,
                            env=env_step,
                        )
                        traj[game_id].append(time_step)
                else:
                    action = Action.from_env_step(env_step).default()

                self.process.stdin.write(action)
                await self.process.stdin.drain()
            else:
                reward = env_step.rewards[..., env_step.player_id.item()]
                actor._done_callback(reward.item())

            if dones[game_id] >= 2:
                if learn_queue is not None:
                    time_step = TimeStep(
                        id=game_id,
                        actor=actor_step,
                        env=env_step,
                    )
                    traj[game_id].append(time_step)
                    trajectory = Trajectory.from_env_steps(traj[game_id])
                    learn_queue.put(trajectory.serialize())

                progress_queue.put(len(traj[game_id]))
                dones[game_id] = 0
                traj[game_id] = []
