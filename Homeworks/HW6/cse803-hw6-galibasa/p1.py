import pandas as pd
import numpy as np
from utils import *

def get_pair(p2d, p3d):
    u, v = p2d
    x, y, z = p3d
    return [[x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u],
            [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]]


def read_data(filename = None):
    with open(filename, 'r') as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            pts.append(np.array([float(x) for x in line.strip().split()]))
    return pts

points_in_2d = read_data('data/pts2d-norm-pic.txt')
points_in_3d = read_data('data/pts3d-norm.txt')

pair_matrix = get_pair(points_in_2d[0], points_in_3d[0])
for p2d, p3d in zip(points_in_2d[1:], points_in_3d[1:]):
    pair_matrix = pair_matrix + get_pair(p2d, p3d)
[U, S, V] = np.linalg.svd(pair_matrix)
P = V.T[:, -1].reshape(3,4)
print("P:", P)

points_in_3d_homo = np.concatenate( [np.array(points_in_3d), np.ones((len(points_in_3d), 1))], axis=1)
points_in_2d_homo = np.concatenate( [np.array(points_in_2d), np.ones((len(points_in_2d), 1))], axis=1)
prediction = (P@(points_in_3d_homo.T)).T



