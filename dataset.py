import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from os.path import splitext
from os import listdir


from torch.utils.data import DataLoader, random_split

class vcdDataset(Dataset):
    def __init__(self, lf_dir, gt_dir, image_size=[176, 176], n_slices=51, n_num=11, fmt='.tif', preload=False):
        self.lf_dir = lf_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.n_slices = n_slices
        self.n_num = n_num
        self.n_channels_lf = n_num * n_num
        self.view_size = [image_size[0]//n_num, image_size[1]//n_num]
        self.fmt = fmt
        self.preload = preload
        self.lfs = None
        self.gts = None

        self.lf_ids = [splitext(file)[0] for file in listdir(lf_dir) 
                        if not file.startswith('.') and splitext(file)[1] == fmt]
        self.gt_ids = [splitext(file)[0] for file in listdir(gt_dir) 
                        if not file.startswith('.') and splitext(file)[1] == fmt]

        assert image_size[0]%n_num==0 and image_size[1]%n_num==0, "light field 2D images size and n number are mismatched"
        assert self.lf_ids == self.gt_ids, "Light field and Ground truth images are mismatched"


        if preload:
            self.lfs = np.zeros([len(self.lf_ids), n_num*n_num, image_size[0]//n_num, image_size[1]//n_num])
            self.gts = np.zeros([len(self.gt_ids), n_slices, image_size[0], image_size[1]])
            for idx in range(len(self.lf_ids)):
                lf = get_lf_views(self.lf_ids[idx]+self.fmt, self.lf_dir, normalize_fn=normalize_1, n_num=self.n_num)
                gt = get_img3d_fn(self.gt_ids[idx]+self.fmt, self.gt_dir, normalize_fn=normalize_1)                
                assert lf.shape == (self.n_channels_lf, *self.view_size) , "light field 2D images size wrong"
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

            lf = get_lf_views(self.lf_ids[i]+self.fmt, self.lf_dir, normalize_fn=normalize_1, n_num=self.n_num)
            gt = get_img3d_fn(self.gt_ids[i]+self.fmt, self.gt_dir, normalize_fn=normalize_1)            
            assert lf.shape == (self.n_channels_lf, *self.view_size) , "light field 2D images size wrong"
            assert gt.shape == (self.n_slices, *self.image_size) , "Ground truth 3D images size wrong"
            return {'lf': torch.from_numpy(lf).to(torch.float32), 'gt': torch.from_numpy(gt).to(torch.float32)}


if __name__ == '__main__':

    lf_dir = './data/lftest/LF/'
    gt_dir = './data/lftest/WF/'

    dataset = vcdDataset(lf_dir, gt_dir, n_slices=61)
    print(type(dataset))
    print(dataset.lf_ids, dataset.gt_ids)
    print(len(dataset))


    train_dataset, val_dataset = random_split(dataset, [1, 1])

    a=1
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    
    # print(len(dataloader))

    # for batch in dataloader:
    #     print(batch)
    

    

