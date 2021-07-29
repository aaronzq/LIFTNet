import numpy as np
import torch

from model import UNet
from utils import *
from os import listdir
from os.path import splitext
import time

from typing import List



def read_test_imgs(
    path: str,
    img_size: List = [256, 256],
    n_ang: int = 25,
    fmt: str = '.tif'):

    img_list = sorted([img_name for img_name in listdir(path) if not img_name.startswith('.') and splitext(img_name)[1] == fmt])

    assert len(img_list) > 0, "no image is loaded from the target directory" 

    imgs = np.zeros([len(img_list), n_ang, img_size[0], img_size[1]])
    for idx, img_name in enumerate(img_list):
        img = get_img3d_fn(img_name, path, normalize_fn=normalize_1)
        imgs[idx, :, :, :] = img
        print('Load {} with size: '.format(img_name), img.shape)
        
    print('Read {} images from {}'.format(len(img_list), path))
    return imgs


def test(config, epoch, batch_size=1):
    
    start_time0 = time.time()

    exists_or_mkdir(config.test.saving_path)

    device = torch.device("cuda:{}".format(config.test.device) if torch.cuda.is_available() else "cpu")
    print("Testing on {}".format(device))

    net = UNet(config.n_slices , config.n_ang).to(device)

    ckpt_list = listdir(config.test.ckpt_loading_path)
    ckpt_name = '{}_liftnet_epoch{}.pth'.format(config.label, epoch)
    if ckpt_name in ckpt_list:
        ckpt = torch.load(config.test.ckpt_loading_path+ckpt_name, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.to(device)
        print('Load checkpoint: {}'.format(ckpt_name))
    else:
        raise Exception('No such checkpoint file')
    

    test_imgs = read_test_imgs(config.test.lf2d_path, config.test.img_size, config.n_ang)
    test_imgs = torch.from_numpy(test_imgs).to(device, dtype=torch.float32)

    net.eval()
    for idx in range(0, test_imgs.shape[0], batch_size):
        with torch.no_grad():
            start_time = time.time()
            test_batch = test_imgs[idx:idx+batch_size, :, :, :]
            pred = net(test_batch)
            print("Recon time elapsed : %4.4fs " % (time.time() - start_time))
            save_img3d('test_epoch{}_{}.tif'.format(epoch, idx), config.test.saving_path, pred, bitdepth=16)
            print("Recon+IO time elapsed : %4.4fs " % (time.time() - start_time))  
            print('Finished %d / %d ...' % (idx + 1, test_imgs.shape[0]))

    print("Total time elapsed : %4.4fs " % (time.time() - start_time0))


if __name__ == "__main__":
    config = Config(label = 'simulated_beads_25_21', n_ang=25, n_slices=21)    
    config.test.img_size = [256, 256]
    config.test.lf3d_path = 'data/test/'
    config.test.saving_path = 'data/test/{}/'.format(config.label)

    config.show_basic_paras()
    config.show_test_paras()
    test(config, epoch=20, batch_size=1)