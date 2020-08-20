import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict


def add_space(sims, qc, gc, la=1.0):
    topo = np.array([[0.20634, 0.11704, 0.21065, 0.29363, 0.17235],
              [0.2029 , 0.17888, 0.20383, 0.21945, 0.19494],
              [0.20608, 0.11115, 0.21078, 0.30260, 0.16940],
              [0.20604, 0.11043, 0.21078, 0.30373, 0.16903],
              [0.20641, 0.13280, 0.20976, 0.27158, 0.17944]])
    new_sims = sims.copy()
    topo_sims = topo[qc][:, gc]
    new_sims = new_sims * topo_sims
    return sims - new_sims


distmats = ['log/test_a/distmat.npy', 'log/test_b/distmat.npy', 'log/test_101/distmat.npy', 'log/test_hr/distmat.npy',
            'log/test_a_large/distmat.npy', 'log/test_b_large/distmat.npy', 'log/test_101_large/distmat.npy', 'log/test_hr_large/distmat.npy']
weights = [2, 1, 2, 1,
           2, 1, 2, 1]
camdistmats = ['log/test_camera_101/camdistmat.npy', 'log/test_camera_152/camdistmat.npy',
               'log/test_camera_101_a/camdistmat.npy', 'log/test_camera_hr/camdistmat.npy']
cam_weights = [1, 1, 1, 1]
q_pred = np.load('log/test_camera_101_a/q_pred.npy')
g_pred = np.load('log/test_camera_101_a/g_pred.npy')

final_distmat = np.zeros((len(q_pred), len(g_pred))).astype(np.float16)
for distmat, weight in zip(distmats, weights):
    final_distmat += np.load(distmat) * weight
final_distmat /= sum(weights)

camdistmat = np.zeros((len(q_pred), len(g_pred))).astype(np.float16)
for distmat, weight in zip(camdistmats, cam_weights):
    camdistmat += np.load(distmat) * weight
camdistmat /= sum(cam_weights)

final_distmat -= camdistmat * 0.1

campredmat = np.equal(q_pred.reshape(-1, 1), g_pred.T)
final_distmat += campredmat * 1.0

final_distmat = add_space(final_distmat, q_pred, g_pred)

indices = np.argsort(final_distmat, axis=1)
np.savetxt('result.txt', indices[:, :100], fmt='%05d')
