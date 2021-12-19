import numpy as np
import cv2
import open3d
from utils import *

temple = np.load('data/temple.npz')
print(temple.files)
K1, K2  = temple["K1"], temple["K2"]
p1, p2 = temple["pts1"], temple["pts2"]

F = cv2.findFundamentalMat(p1, p2)[0]
print("F:", F)
E = K1 @ F @ K2.T
print("E:", E)

R1, R2, t = cv2.decomposeEssentialMat(E)

P1 = np.concatenate([K1, np.zeros((3,1))], axis=1)
Rt = np.concatenate([R2, t], axis=1)
P2 = K2.dot(Rt)

print("P1:", P1)
print("P2:", P2)

coord_3d = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float32), p2.T.astype(np.float32))

print(coord_3d.T.shape)

pcd = coord_3d.T[:, :3] / coord_3d.T[:,3].reshape((-1, 1))

visualize_pcd(pcd)