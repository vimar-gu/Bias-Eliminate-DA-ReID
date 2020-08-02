# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .visda_dataset import VisDADataset
from .dataset_loader import ImageDataset, CamStyleImageDataset, CameraImageDataset


__factory = {
    'visda_dataset': VisDADataset,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)


def add_psolabels(dataset,psolabels):
    assert(len(dataset)==len(psolabels))
    outset = []
    for i in range(len(dataset)):
        img_path, pid, camid, setid, trkid, pidx = dataset[i]
        outset.append((img_path,psolabels[i],camid,setid,trkid, pidx))
    return outset


def add_camstyle_psolabels(dataset, camstyle_dict, psolabels):
    assert(len(dataset)==len(psolabels))
    outset = []
    for i in range(len(dataset)):
        img_path, pid, camid, setid, trkid, pidx = dataset[i]
        camstyle_arr = camstyle_dict[str(pid) + '_' + str(camid) + '_' + str(pidx)]
        for camstyle_img_path in camstyle_arr:
            outset.append((camstyle_img_path,psolabels[i],camid,setid,trkid, pidx))
    return outset
