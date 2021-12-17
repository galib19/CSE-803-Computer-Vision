import cv2
from PIL import Image
from matplotlib import pyplot
import pandas as pd
import numpy as np
from utils import *
from functools import reduce


temple = np.load('data/temple.npz')
p1, p2 = temple["pts1"], temple["pts2"]


# Loading Images
img1 = cv2.imread('data/im1.png')
img2 = cv2.imread('data/im2.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


def normalize_data(data):
    return data/ 640


p1_normalized = normalize_data(p1)
p2_normalized = normalize_data(p2)

def normalize(points):
    n = len(points)
    image1_points, image2_points = [], []
    for a, b, c, d in points:
        image2_points.append([a, b])
        image1_points.append([c, d])
    sum1 = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), image1_points)
    sum2 = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), image2_points)

    mean1 = [val / n for val in sum1]
    mean2 = [val / n for val in sum2]

    s1 = (n * 2) ** 0.5 / (sum([((x - mean1[0]) ** 2 + (y - mean1[1]) ** 2) ** 0.5 for x, y in image1_points]))
    s2 = (2 * n) ** 0.5 / (sum([((x - mean2[0]) ** 2 + (y - mean2[1]) ** 2) ** 0.5 for x, y in image2_points]))

    T1 = np.array([[s1, 0, -mean1[0] * s1], [0, s1, -mean1[1] * s1], [0, 0, 1]])
    T2 = np.array([[s2, 0, -mean2[0] * s2], [0, s2, -mean2[1] * s2], [0, 0, 1]])

    points = [[T1 @ [c, d, 1], T2 @ [a, b, 1]] for a, b, c, d in points]
    points = [[l[0], l[1], r[0], r[1]] for l, r in points]
    return points, T1, T2


init_matrix = np.concatenate([np.array(p1), np.array(p2)], axis=1)
norm_points, T1, T2 = normalize(init_matrix)


def calc_F(init_matrix):
    matrix = np.zeros((len(init_matrix), 9))
    # img1 x' y' x y im2
    for i in range(len(init_matrix)):
        matrix[i][0] = init_matrix[i][0] * init_matrix[i][2]
        matrix[i][1] = init_matrix[i][1] * init_matrix[i][2]
        matrix[i][2] = init_matrix[i][2]
        matrix[i][3] = init_matrix[i][0] * init_matrix[i][3]
        matrix[i][4] = init_matrix[i][1] * init_matrix[i][3]
        matrix[i][5] = init_matrix[i][3]
        matrix[i][6] = init_matrix[i][0]
        matrix[i][7] = init_matrix[i][1]
        matrix[i][8] = 1.0

    _, _, v = np.linalg.svd(matrix)
    fundamental_vector = v.transpose()[:, 8]
    fundamental_matrix = np.reshape(fundamental_vector, (3, 3))

    s, v, d = np.linalg.svd(fundamental_matrix)
    fundamental_matrix = s @ np.diag([*v[:2], 0]) @ d
    return fundamental_matrix

fundamental_matrix = calc_F(init_matrix)
print("F before normalization:", fundamental_matrix / fundamental_matrix[2, 2])

fundamental_matrix = calc_F(norm_points)


def restore(fundamental_matrix, T1, T2):
    f_mat = T2.transpose() @ fundamental_matrix @ T1
    return f_mat


f_mat = restore(fundamental_matrix, T1, T2)
print("F after normalization:", f_mat / f_mat[2, 2])

f = f_mat / f_mat[2, 2]


pyplot.figure(figsize=(15, 8))
draw_epipolar(img1, img2, f, p1[:10], p2[:10])