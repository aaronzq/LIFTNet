import numpy as np
import torch
from torch.utils.data import DataLoader

from model import VCDNet
from dataset import vcdDataset
from utils import *
from os import listdir
from os.path import splitext
import time




def read_test_imgs(path, img_size, n_num=11, fmt='.tif'):

    img_list = sorted([img_name for img_name in listdir(path) if not img_name.startswith('.') and splitext(img_name)[1] == fmt])

    assert img_size[0]%n_num==0 and img_size[1]%n_num==0, "light field 2D images size and n number are mismatched"
    assert len(img_list) > 0, "no image is loaded from the target directory" 

    imgs = np.zeros([len(img_list), n_num*n_num, img_size[0]//n_num, img_size[1]//n_num])
    for idx, img_name in enumerate(img_list):
        img = get_lf_views(img_name, path, normalize_1, n_num)
        imgs[idx, :, :, :] = img
        print('Load {} with size: '.format(img_name), img.shape)
        
    print('Read {} images from {}'.format(len(img_list), path))
    return imgs


def test(config, epoch, batch_size=1):
    
    start_time0 = time.time()

    exists_or_mkdir(config.test.saving_path)

    device = torch.device("cuda:{}".format(config.test.device) if torch.cuda.is_available() else "cpu")
    print("Testing on {}".format(device))

    net = VCDNet(config.n_num, config.n_slices, config.test.img_size).to(device)

    ckpt_list = listdir(config.test.ckpt_loading_path)
    ckpt_name = '{}_vcdnet_epoch{}.pth'.format(config.label, epoch)
    if ckpt_name in ckpt_list:
        ckpt = torch.load(config.test.ckpt_loading_path+ckpt_name, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.to(device)
        print('Load checkpoint: {}'.format(ckpt_name))
    else:
        raise Exception('No such checkpoint file')
    

    test_imgs = read_test_imgs(config.test.lf2d_path, config.test.img_size, config.n_num)
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
    config = Config(label = 'rbcDSRED_[m76-76]_step2um_N11_bgsub_20210112_bs1_netv5_s1s2_trans_cstr2_cust12', n_num=11, n_slices=77)    
    config.test.img_size = [330, 330]
    config.show_basic_paras()
    config.show_test_paras()
    # read_test_imgs('./data/lftest/LF/', [176,176], n_num=11, fmt='.tif')
    test(config, epoch=26, batch_size=1)



