import torch
import timeit

def compute_sens(many_xs, t, m, a, b, device):
    # Clamp and sort all xs's at once, assuming shape is (num_iter, n)
    many_xs = many_xs.clamp(a, b).sort(dim=1).values
    n = many_xs.size(1)
    num_iter = many_xs.size(0)

    # Concat [b, a] to the end of every xs so that indexing -1 gives a and indexing n gives b, then clamp indices to -1, n
    many_xs = torch.cat((many_xs, torch.full((num_iter, 1), b, device=device), torch.full((num_iter, 1), a, device=device)), dim=1)

    # Generate indices now so they don't need to be every time (will be xs[idx1] - xs[idx2]), this doesn't need to be efficient but w/e
    ks = torch.arange(0, n+1, device=device) # distances
    ls = torch.arange(0, n+2, device=device)
    # Use all l values then take lower triangular part of matrix plus (with diagonal shifted by one) to remove values where l > k+1
    idx1 = torch.tril(n - m + 1 + ks.reshape(-1, 1) - ls, diagonal=1).clamp(-1, n)
    #print("IDX1:")
    #print(idx1)
    idx2 = (m + 1 - ls).clamp(-1, n)
    #print("IDX2:")
    #print(idx2)

    scalar = torch.exp(-1 * ks * t)

    out = torch.empty(num_iter)
    for i in range(num_iter):
        xs = many_xs[i]

        diffs = torch.tril(xs[idx1] - xs[idx2], diagonal=1)
        #print("Diffs:")
        #print(diffs)

        inner_max = diffs.max(dim=1).values
        #print("Inner max:")
        #print(inner_max*scalar)

        outer_max = (inner_max*scalar).max()
        #print("Result:")
        #print(outer_max / (n - 2*m))
        out[i] = outer_max / (n - 2*m)

    return out

def time(n=256, num_weights=100000, num_exec=1, device="cpu"):
    inp = torch.empty((num_weights, n)).normal_(5, 2).to(device)
    print(timeit.timeit(lambda: compute_sens(inp, 1, 2, 0, 10, device=device), number=num_exec))

if __name__ == "__main__":
    print(compute_sens(torch.tensor([[7,2,3,4,5,6,1], [7,2,3,4,5,6,1]]), .1, 2, 0, 10, "cpu"))
    time(num_weights=100)
