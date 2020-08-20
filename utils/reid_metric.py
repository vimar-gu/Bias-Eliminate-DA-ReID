# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import numpy as np
import torch
from ignite.metrics import Metric
import collections

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking

import sklearn.metrics.cluster
from sklearn.cluster import DBSCAN
from collections import Counter
from layers.triplet_loss import euclidean_dist


def compute_P2(qf, gf, gc, la=3.0):
    X = gf
    neg_vec = {}
    u_cams = np.unique(gc)
    P = {}
    for cam in u_cams:
        curX = gf[gc == cam]
        neg_vec[cam] = np.mean(curX, axis=0)
        P[cam] = np.linalg.inv(curX.T.dot(curX)+curX.shape[0]*la*np.eye(X.shape[1]))
    return P, neg_vec


def meanfeat_sub(P, neg_vec, in_feats, in_cams):
    out_feats = []
    for i in range(in_feats.shape[0]):
        camid = in_cams[i]
        feat = in_feats[i] - neg_vec[camid]
        feat = P[camid].dot(feat)
        feat = feat/np.linalg.norm(feat, ord=2)
        out_feats.append(feat)
    out_feats = np.vstack(out_feats)
    return out_feats


def mergesetfeat(X, cams, gX, gcams, beta=0.08, knn=20):
    for i in range(X.shape[0]):
        if i % 1000 == 0:
            print('merge:%d/%d' % (i, X.shape[0]))
        knnX = gX
        sim = knnX.dot(X[i, :])
        knnX = knnX[sim>0]
        sim = sim[sim>0]
        if len(sim) > 0:
            idx = np.argsort(-sim)
            if len(sim)>2*knn:
                sim = sim[idx[:2*knn]]
                knnX = knnX[idx[:2*knn],:]
            else:
                sim = sim[idx]
                knnX = knnX[idx,:]
                knn = min(knn,len(sim))
            knn_pos_weight = np.exp((sim[:knn]-1)/beta)
            knn_neg_weight = np.ones(len(sim)-knn)
            knn_pos_prob = knn_pos_weight/np.sum(knn_pos_weight)
            knn_neg_prob = knn_neg_weight/np.sum(knn_neg_weight)
            X[i,:] += 0.2*(knn_pos_prob.dot(knnX[:knn,:]) - knn_neg_prob.dot(knnX[knn:,:]))
            X[i,:] /= np.linalg.norm(X[i,:])
    return X


def add_space(sims, qc, gc, la=1.0):
    topo = np.array([[0.20634, 0.11704, 0.21065, 0.29363, 0.17235],
              [0.2029 , 0.17888, 0.20383, 0.21945, 0.19494],
              [0.20608, 0.11115, 0.21078, 0.30260, 0.16940],
              [0.20604, 0.11043, 0.21078, 0.30373, 0.16903],
              [0.20641, 0.13280, 0.20976, 0.27158, 0.17944]])
    new_sims = sims.copy()
    topo_sims = topo[qc][:, gc]
    new_sims *= topo_sims
    return sims - new_sims * la


class R1_mAP(Metric):
    def __init__(self, num_query, validation_flag, output_flag, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.validation_flag = validation_flag
        self.output_flag = output_flag

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, _, _, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            #print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        distmat = euclidean_dist(qf, gf).cpu().numpy()

        if self.validation_flag:
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        else:
            cmc, mAP = np.zeros((self.max_rank,)), 0.0

        if self.output_flag:
            indices = np.argsort(distmat, axis=1)
            np.savetxt("result.txt", indices[:, :100], fmt="%05d")

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, validation_flag, output_flag, output_dir, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.validation_flag = validation_flag
        self.output_flag = output_flag
        self.output_dir = output_dir

    def reset(self):
        self.feats = []
        self.camera_scores = np.array([])
        self.camera_feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, camscore, cam_feat, pid, camid = output
        self.feats.append(feat)
        self.camera_scores = np.append(self.camera_scores, camscore.unsqueeze(0).cpu().numpy())
        self.camera_feats.append(cam_feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        camera_feats = torch.cat(self.camera_feats, dim=0)
        if self.feat_norm == 'yes':
            #print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            camera_feats = torch.nn.functional.normalize(camera_feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_campred = np.asarray(self.camera_scores[:self.num_query], np.int)
        qcf = camera_feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_campred = np.asarray(self.camera_scores[self.num_query:], np.int)
        gcf = camera_feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        qf = qf.cpu().numpy().astype(np.float16)
        gf = gf.cpu().numpy().astype(np.float16)
        P, neg_vec = compute_P2(qf, gf, g_campred, 0.02)
        qf = meanfeat_sub(P, neg_vec, qf, q_campred)
        gf = meanfeat_sub(P, neg_vec, gf, g_campred)
        gf_new = gf.copy()
        for _ in range(3):
            gf_new = mergesetfeat(gf_new, g_campred, gf, g_campred, 0.03, 50)
        qf_new = qf.copy()
        for _ in range(3):
            qf_new = mergesetfeat(qf_new, q_campred, gf_new, g_campred, 0.03, 50)
        qf = torch.from_numpy(qf_new).cuda()
        gf = torch.from_numpy(gf_new).cuda()

        distmat = re_ranking(qf, gf, k1=30, k2=8, lambda_value=0.3)

        camdistmat = euclidean_dist(qcf, gcf).cpu().numpy()

        if self.output_flag:
            np.save(os.path.join(self.output_dir, 'distmat.npy'), distmat)
            np.save(os.path.join(self.output_dir, 'q_pred.npy'), q_campred)
            np.save(os.path.join(self.output_dir, 'g_pred.npy'), g_campred)
            np.save(os.path.join(self.output_dir, 'camdistmat.npy'), camdistmat)

        distmat -= camdistmat * 0.1

        campredmat = np.equal(q_campred.reshape(-1,1), g_campred.T)
        distmat += campredmat * 1.0

        distmat = add_space(distmat, q_campred, g_campred)

        if self.validation_flag:
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        else:
            cmc, mAP = np.zeros((self.max_rank,)), 0.0

        if self.output_flag:
            indices = np.argsort(distmat, axis=1)
            np.savetxt("result.txt", indices[:, :100], fmt="%05d")

        return cmc, mAP


class Cluster(Metric):
    def __init__(self, topk, dist_thrd, feat_norm='yes', finetune=False):
        super(Cluster, self).__init__()
        self.feat_norm = feat_norm
        self.topk = topk
        self.dist_thrd = dist_thrd
        self.epoch = 0
        self.indep_thres = 0
        self.finetune = finetune

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.trkids = []
        self.camfeats = []

    def update(self, output):
        feat, camfeat, pid, camid, trkid = output
        self.feats.append(feat)
        self.camfeats.append(camfeat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.trkids.extend(np.asarray(trkid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        camfeats = torch.cat(self.camfeats, dim=0)
        if self.feat_norm == 'yes':
            #print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            camfeats = torch.nn.functional.normalize(camfeats, dim=1, p=2)
        feats = feats.cpu().numpy().astype(np.float16)

        # merge by trkid
        self.pids = np.asarray(self.pids)
        trkids = list(set(self.trkids))
        merged_feats = np.zeros((len(trkids),feats.shape[1]))
        merged_labels = np.zeros(len(trkids)) - 1
        for i in range(len(trkids)):
            tmpidxs = np.where(self.trkids==trkids[i])
            tmpfeats = feats[tmpidxs]
            merged_feats[i] = np.mean(tmpfeats,axis=0)
            merged_labels[i] = self.pids[tmpidxs][0]
        feats = torch.from_numpy(merged_feats).cuda()

        distmat = re_ranking(feats, feats, 20, 6, 0.3)
        if self.finetune:
            camdistmat = euclidean_dist(camfeats, camfeats).cpu().numpy()
            distmat -= camdistmat * 0.1
        pos_bool = (distmat < 0)
        distmat[pos_bool] = 0

        eps = self.dist_thrd
        cluster = DBSCAN(eps=eps,min_samples=4, metric='precomputed', n_jobs=2)
        ret = cluster.fit_predict(distmat.astype(np.double))

        # most count
        labelset = Counter(ret[ret>=0])
        labelset = labelset.most_common()
        labelset = [i[0] for i in labelset]

        labelset = labelset[:self.topk]
        idxs = np.where(np.in1d(ret,labelset))[0]
        psolabels = ret[idxs]
        pids = merged_labels[idxs]
        acc = sklearn.metrics.cluster.normalized_mutual_info_score(psolabels,pids)

        # back to train images
        outret = np.zeros(len(self.trkids)) - 1
        for i in range(len(trkids)):
            if ret[i] >= 0:
                outret[self.trkids==trkids[i]] = ret[i]

        # relabel
        outret_set = set(outret)
        outret_set_len = len(labelset) - 1
        outliers = 0
        pid2label = {}
        for pid in outret_set:
            if pid in labelset:
                pid2label[pid] = labelset.index(pid)
            elif self.finetune and outliers <= 200:
                pid2label[pid] = outret_set_len + outliers
                outliers += 1
            else:
                pid2label[pid] = -1
        outret = np.asarray([pid2label[i] for i in outret])

        return outret,acc,len(outret[outret!=-1])
