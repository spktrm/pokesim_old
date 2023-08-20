import torch

from memory_profiler import profile


@profile(precision=4)
def my_func1(x, y, m):
    return torch.where(m, x, y)


@profile(precision=4)
def my_func2(x, y, m):
    return x * m + y * ~m


def main():
    s = (100, 100)
    device = "cpu"

    m = torch.randint(0, 1, s, dtype=torch.bool, device=device)
    x = torch.randn(s, device=device)
    y = torch.randn(s, device=device)

    out1 = my_func1(x, y, m)
    out2 = my_func2(x, y, m)

    assert torch.all(out1 == out2)


if __name__ == "__main__":
    main()
