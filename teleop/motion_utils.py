import numpy as np
import torch


def mat_update(prev_mat, mat):
    if np.linalg.det(mat) == 0:
        return prev_mat
    else:
        return mat


def tensor_update(prev_tensor, tensor):
    if torch.det(tensor) == 0:
        return prev_tensor
    else:
        return tensor


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret
