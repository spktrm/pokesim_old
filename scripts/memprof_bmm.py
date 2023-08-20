import torch

from memory_profiler import profile


# @profile(precision=4)
def my_func1(x, y):
    return torch.bmm(x.expand(192, 4096, 4096), y)


# @profile(precision=4)
def my_func2(x, y):
    return torch.matmul(x, y)


def main():
    x = torch.randn(1, 4096, 4096, device="cuda")
    y = torch.randn(192, 4096, 1, device="cuda")
    out1 = my_func1(x, y)
    out2 = my_func2(x, y)
    assert torch.all(out1 == out2)


if __name__ == "__main__":
    main()
