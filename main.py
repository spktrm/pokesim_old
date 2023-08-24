import os
import wandb

os.environ["OMP_NUM_THREADS"] = "1"

import uvloop
import asyncio

import torch
import torch.nn as nn

import threading
import multiprocessing as mp

from typing import Iterator
from tqdm import tqdm

from pokesim.inference import Inference
from pokesim.manager import Manager
from pokesim.actor import SelfplayActor, EvalActor
from pokesim.structs import Batch, Trajectory
from pokesim.learner import Learner

_DEBUG = False
_NUM_CPUS = 1 if _DEBUG else os.cpu_count()
print(f"Running on `{_NUM_CPUS} Workers`")


async def _eval(
    worker_index: int, model: nn.Module, progress_queue: mp.Queue, eval_queue: mp.Queue
):
    inference = Inference(model, "cpu", batch_size=2)
    command = f"node ./sim.js {worker_index}"

    process = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    manager = Manager(
        worker_index=worker_index,
        process=process,
        actor1=SelfplayActor(inference),
        actor2=EvalActor(eval_queue),
    )

    await manager.run(progress_queue)


def run_eval(
    worker_index: int, model: nn.Module, progress_queue: mp.Queue, eval_queue: mp.Queue
):
    torch.set_grad_enabled(False)
    asyncio.run(_eval(worker_index, model, progress_queue, eval_queue))


async def _selfplay(
    worker_index: int, model: nn.Module, progress_queue: mp.Queue, learn_queue: mp.Queue
):
    inference = Inference(model, "cpu", batch_size=2)
    command = f"node ./sim.js {worker_index}"

    process = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    manager = Manager(
        worker_index=worker_index,
        process=process,
        actor1=SelfplayActor(inference),
        actor2=SelfplayActor(inference),
    )

    await manager.run(progress_queue, learn_queue)


def run_selfplay(
    worker_index: int, model: nn.Module, progress_queue: mp.Queue, learn_queue: mp.Queue
):
    torch.set_grad_enabled(False)
    asyncio.run(_selfplay(worker_index, model, progress_queue, learn_queue))


def read_prog(progress_queue: mp.Queue):
    progress1 = tqdm(desc="games")
    progress2 = tqdm(desc="steps", leave=False)
    while True:
        l = progress_queue.get()
        progress1.update(1)
        progress2.update(l)


def read_eval(eval_queue: mp.Queue):
    n = 0
    while True:
        r = eval_queue.get()
        if not _DEBUG:
            wandb.log({"n": n, "r": r})
        n += 1


def dataloader(queue: mp.Queue, batch_size: int = 16) -> Iterator[Batch]:
    batch = []
    while True:
        sample = queue.get()
        trajectory = Trajectory.deserialize(sample)
        if trajectory.is_valid():
            batch.append(trajectory)
        if len(batch) >= batch_size:
            batch = Batch.from_trajectories(batch)
            yield batch
            batch = []


def learn(learner: Learner, queue: mp.Queue):
    progress = tqdm(desc="Learning")
    env_steps = 0
    for batch in dataloader(queue, learner.config.batch_size):
        env_steps += batch.valid.sum()

        alpha, update_target_net = learner._entropy_schedule(learner.learner_steps)
        logs = learner.update_parameters(batch, alpha, update_target_net)

        learner.learner_steps += 1

        logs["avg_length"] = batch.valid.sum(0).mean()
        logs["learner_steps"] = learner.learner_steps
        logs["env_steps"] = env_steps

        if not _DEBUG:
            wandb.log(logs)

        if learner.learner_steps % 2500 == 0:
            torch.save(
                learner.params.state_dict(), f"ckpts/ckpt-{learner.learner_steps}.pt"
            )

        progress.update()


def main():
    init = None
    # init = torch.load("ckpts/ckpt-90000.pt")
    learner = Learner(init)

    if not _DEBUG:
        wandb.init(
            # set the wandb project where this run will be logged
            project="meloettav2",
            # track hyperparameters and run metadata
            config=learner.config.__dict__,
        )

    progress_queue = mp.Queue()
    eval_queue = mp.Queue()
    learn_queue = mp.Queue(maxsize=100)

    progress_thread = threading.Thread(target=read_prog, args=(progress_queue,))
    progress_thread.start()

    eval_thread = threading.Thread(target=read_eval, args=(eval_queue,))
    eval_thread.start()

    learn_thread = threading.Thread(
        target=learn,
        args=(learner, learn_queue),
    )
    learn_thread.start()

    uvloop.install()

    procs = [
        mp.Process(
            target=run_selfplay,
            args=(worker_index, learner.params_actor, progress_queue, learn_queue),
        )
        for worker_index in range(_NUM_CPUS - 1)
    ]
    procs += [
        mp.Process(
            target=run_eval,
            args=(len(procs) + 1, learner.params_actor, progress_queue, eval_queue),
        )
    ]
    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

    progress_thread.join()
    learn_thread.join()
    eval_thread.join()


if __name__ == "__main__":
    main()
