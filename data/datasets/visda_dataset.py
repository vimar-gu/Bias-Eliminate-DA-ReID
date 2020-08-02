# encoding: utf-8

import glob
import re

import os.path as osp
from collections import defaultdict

from .bases import BaseImageDataset


class VisDADataset(BaseImageDataset):

    def __init__(self, root_train='', root_val='', setid=0, verbose=True, transfered='original', **kwargs):
        super(VisDADataset, self).__init__()
        if transfered == 'original':
            self.train_dir = osp.join(root_train, 'trainval')
            self.train_camstyle_dir = osp.join(root_train, 'trainval_camstyle')
        else:
            self.train_dir = osp.join(root_train, 'trainval_' + transfered)
            self.train_camstyle_dir = osp.join(root_train, 'trainval_' + transfered + '_camstyle')
        self.query_dir = osp.join(root_val, 'test_probe')
        self.gallery_dir = osp.join(root_val, 'test_gallery')
        self.setid = setid

        if verbose: 
            print ('*' * 50)
            print ('load train_dir: {}'.format(self.train_dir))
            print ('load train_camstyle_dir: {}'.format(self.train_camstyle_dir))
            print ('load query_dir: {}'.format(self.query_dir))
            print ('load gallery_dir: {}'.format(self.gallery_dir))
            print ('*' * 50)

        #self._check_before_run()

        train, pid2label = self._process_dir(self.train_dir, relabel=True)
        train_camstyle, camstyle_dict = self._process_camstyle_dir(self.train_camstyle_dir, relabel=True, pid2label=pid2label)
        query, _ = self._process_dir(self.query_dir, relabel=False)
        gallery, _ = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            self.print_dataset_statistics(train, train_camstyle, query, gallery)

        self.train = train
        self.train_camstyle = train_camstyle
        self.camstyle_dict = camstyle_dict
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_train_camstyle_pids, self.num_train_camstyle_imgs, self.num_train_camstyle_cams = self.get_imagedata_info(self.train_camstyle)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.train_camstyle_dir):
            raise RuntimeError("'{}' is not available".format(self.train_camstyle_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        if dir_path == '':
            return []

        img_paths = sorted(glob.glob(osp.join(dir_path, '*.png')) + glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'([\d]+)_([\d]+)_([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            pid, camid, pidx = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, self.setid, i, pidx))

        return dataset, pid2label

    def _process_camstyle_dir(self, dir_path, relabel=False, pid2label=None):
        if dir_path == '':
            return []

        img_paths = sorted(glob.glob(osp.join(dir_path, '*.png')) + glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'([\d]+)_([\d]+)_([\d]+)')

        dataset = []
        label2pid = {}
        pid_cam_dict = defaultdict(list)
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            opid, camid, pidx = map(int, pattern.search(img_path).groups())
            if opid == -1: continue  # junk images are just ignored
            if relabel: 
                pid = pid2label[opid]
                label2pid[pid] = opid
            dataset.append((img_path, pid, camid, self.setid, i, pidx))
            pid_cam_dict[str(pid) + '_' + str(camid) + '_' + str(pidx)].append(img_path)
        for i in pid_cam_dict.keys():
            if len(pid_cam_dict[i]) != 6 and len(pid_cam_dict[i]) != 8:
                raise RuntimeError("camestyle data error")

        return dataset, pid_cam_dict
