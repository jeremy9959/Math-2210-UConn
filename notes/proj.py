# %%
import numpy as np


def GS(M):
    N = M[:, 0].reshape(-1, 1)
    for i in range(1, M.shape[1]):
        X = GS_step(M[:, i].reshape(-1, 1), N)
        print(X)
        N = np.column_stack((N, X))
    return N


def GS_step(v, M):
    A = v - M @ np.linalg.inv(M.transpose() @ M) @ (M.transpose() @ v)
    return A


# %%
M = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]])
N = GS(M)
print(N)

# %%
