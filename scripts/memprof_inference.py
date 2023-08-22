import math

import torch
import torch.nn as nn

from memory_profiler import profile


@profile(precision=4)
def optimized_forward(module: nn.Module, inputs, batch_size: int = 4):
    results = []

    first_key = next(iter(inputs.keys()))
    length = inputs[first_key].shape[0]

    for i in range(math.ceil(length / batch_size)):
        minibatch = {
            k: v[i * batch_size : (i + 1) * batch_size] for k, v in inputs.items()
        }
        results.append(module(**minibatch))

    return tuple(map(lambda x: torch.cat(x), zip(*results)))


def my_func1(mod, inp):
    return mod(**inp)


@profile(precision=4)
def my_func2(mod, inp):
    return optimized_forward(mod, inp)


class MyModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return (x,)


def main():
    mod = MyModule()
    inp = {"x": torch.randn(2**9, 128)}

    with torch.no_grad():
        out1 = my_func1(mod, inp)
        out2 = my_func2(mod, inp)

    for t1, t2 in zip(out1, out2):
        print((t2 - t1).mean())


if __name__ == "__main__":
    main()
