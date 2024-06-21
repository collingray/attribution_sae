


def reconstruction(x, x_hat):
    return ((x_hat - x) ** 2).mean(dim=-1)

def sparsity(f):
    return f.abs().mean(dim=-1)

def