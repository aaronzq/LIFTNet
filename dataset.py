from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from os.path import splitext
from os import listdir
from typing import List

from torch.utils.data import DataLoader, random_split

class liftDataset(Dataset):
    def __init__(
        self, 
        lf_dir: str, 
        gt_dir: str, 
        image_size: List = [256, 256],
        n_slices: int =21,
        n_ang: int =25,
        fmt: str = '.tif',
        preload: bool = False
    ) -> None:
                
        self.lf_dir = lf_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.n_slices = n_slices
        self.n_ang = n_ang
        self.fmt = fmt
        self.preload = preload
        self.lfs = None
        self.gts = None

        self.lf_ids = [splitext(file)[0] for file in listdir(lf_dir) 
                        if not file.startswith('.') and splitext(file)[1] == fmt]
        self.gt_ids = [splitext(file)[0] for file in listdir(gt_dir) 
                        if not file.startswith('.') and splitext(file)[1] == fmt]

        # assert self.lf_ids == self.gt_ids, "LIFT and Ground truth images are mismatched"


        if preload:
            self.lfs = np.zeros([len(self.lf_ids), self.n_ang, image_size[0], image_size[1]])
            self.gts = np.zeros([len(self.gt_ids), self.n_slices, image_size[0], image_size[1]])
            for idx in range(len(self.lf_ids)):
                lf = get_img3d_fn(self.lf_ids[idx]+self.fmt, self.lf_dir, normalize_fn=normalize_1)
                # gt = get_img3d_fn(self.gt_ids[idx]+self.fmt, self.gt_dir, normalize_fn=normalize_1)
                gt = get_img2d_fn(self.gt_ids[idx]+self.fmt, self.gt_dir, normalize_fn=normalize_1)

                assert lf.shape == (self.n_ang, *self.image_size) , "Lift images size wrong"
                assert gt.shape == (self.n_slices, *self.image_size) , "Ground truth 3D images size wrong"
                self.lfs[idx, :, :, :] = lf
                self.gts[idx, :, :, :] = gt
                print('\rLoading in {}/{} Dataset: {}'.format(idx+1, len(self.lf_ids), self.lf_ids[idx]), end='')
            print()
            self.lfs = torch.from_numpy(self.lfs).to(torch.float32)
            self.gts = torch.from_numpy(self.gts).to(torch.float32)
            

    def __len__(self):
        return len(self.lf_ids)

    def __getitem__(self, i):
        
        if self.preload:

            return {'lf': self.lfs[i,:,:,:], 'gt': self.gts[i,:,:,:]}
        
        else:

            lf = get_img3d_fn(self.lf_ids[i]+self.fmt, self.lf_dir, normalize_fn=normalize_1)
            gt = get_img3d_fn(self.gt_ids[i]+self.fmt, self.gt_dir, normalize_fn=normalize_1)            
            assert lf.shape == (self.n_ang, *self.image_size) , "Lift 3D images size wrong"
            assert gt.shape == (self.n_slices, *self.image_size) , "Ground truth 3D images size wrong"
            return {'lf': torch.from_numpy(lf).to(torch.float32), 'gt': torch.from_numpy(gt).to(torch.float32)}


if __name__ == '__main__':

    lf_dir = './data/lf/'
    gt_dir = './data/gt/'

    dataset = liftDataset(lf_dir, gt_dir, image_size=[256,256], n_slices=21, n_ang=25)
    print(type(dataset))
    print(dataset.lf_ids, dataset.gt_ids)
    print(len(dataset))


    train_dataset, val_dataset = random_split(dataset, [18, 2])

    a=1

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    
    # print(len(dataloader))

    # for batch in dataloader:
    #     print(batch)
    

    

