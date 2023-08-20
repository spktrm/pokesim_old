import torch

from memory_profiler import profile


onehot = torch.nn.Embedding.from_pretrained(torch.eye(10))


def one_hot_encode(tensor, num_classes):
    # Get the shape of the tensor, and create a new shape for the one-hot encoded tensor
    shape = list(tensor.shape) + [num_classes]
    # Create a tensor filled with zeros of the new shape
    one_hot = torch.zeros(shape, device=tensor.device, dtype=torch.float32)
    # Use the scatter method to fill the one-hot tensor
    one_hot.scatter_(-1, tensor.unsqueeze(-1), 1)
    return one_hot


def multi_hot_encode(indices_tensor, num_classes):
    # Get the shape of the tensor, and create a new shape for the multi-hot encoded tensor
    shape = list(indices_tensor.shape[:-1]) + [num_classes]
    # Create a tensor filled with zeros of the new shape
    multi_hot = torch.zeros(
        shape, device=indices_tensor.device, dtype=indices_tensor.dtype
    )
    # Use scatter_add to add the ones to the multi_hot tensor at the specified indices
    multi_hot.scatter_add_(
        -1, indices_tensor, torch.ones_like(indices_tensor, dtype=indices_tensor.dtype)
    )
    return multi_hot.to(torch.float32)


@profile(precision=4)
def my_func1(inp):
    return onehot(inp).sum(-2)


@profile(precision=4)
def my_func2(inp):
    return multi_hot_encode(inp, onehot.weight.shape[-1])


if __name__ == "__main__":
    inp = torch.randint(0, onehot.weight.shape[-1] - 1, (100, 100, 2))
    out1 = my_func1(inp)
    out2 = my_func2(inp)
    assert torch.all(out1 == out2)
