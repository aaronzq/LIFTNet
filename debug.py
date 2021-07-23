import numpy as np
import torch
from os import listdir
from model import VCDNet
from utils import *


class Features:
    def __init__(self):
        self._outputs = []
    def __call__(self, module, module_in, module_out):
        self._outputs.append(module_out)
    def clear(self):
        self._outputs = []
    def get_outputs(self):
        return self._outputs

def debug(config, epoch, img_name, img_path, img_size, save_path):
    
    exists_or_mkdir(save_path)

    device = torch.device("cuda:{}".format(config.train.device) if torch.cuda.is_available() else "cpu")
    print("Testing on {}".format(device))

    net = VCDNet(config.n_num, config.n_slices, img_size).to(device)

    # add hook to intermediate layers
    features = Features()
    hook_handles = []
    for layer in net.modules():
        if (len(list(layer.children())))==0:
            handle = layer.register_forward_hook(features)
            hook_handles.append(handle)

    ckpt_list = listdir(config.train.ckpt_saving_path)
    ckpt_name = '{}_vcdnet_epoch{}.pth'.format(config.label, epoch)
    if ckpt_name in ckpt_list:
        ckpt = torch.load(config.train.ckpt_saving_path+ckpt_name, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.to(device)
        print('Load checkpoint: {}'.format(ckpt_name))
    else:
        raise Exception('No such checkpoint file')


    debug_img = get_lf_views(img_name, img_path, normalize_1, config.n_num, padding=False)
    debug_img = debug_img[np.newaxis, :, :, :]
    debug_img = torch.from_numpy(debug_img).to(device, dtype=torch.float32)

    net.eval()
    with torch.no_grad():
        pred = net(debug_img)
        save_img3d('debug_input.tif'.format(epoch), save_path, debug_img, bitdepth=16)
        save_img3d('debug_output_epoch{}.tif'.format(epoch), save_path, pred, bitdepth=16)       
        print(len(features.get_outputs()))
        for ind, f in enumerate(features.get_outputs()):
            save_img3d('debug_epoch{}_feature{}.tif'.format(epoch, ind), save_path, f, bitdepth=16, norm_mode=2)


if __name__ == "__main__":
    config = Config(label = 'rbcDSRED_[m76-76]_step2um_N11_bgsub_20210112_bs1_netv5_s1s2_trans_cstr2_cust12', n_num=11, n_slices=77)    
    
    config.show_basic_paras() 
    img_size = [176, 176]
    img_name = '003-000046.tif'
    img_path = './data/train/rbcDSRED_[m76-76]_step2um_N11_bgsub_20210112/LF/'
    save_path = './debug/{}/'.format(config.label)
    epoch = 26

    debug(config, epoch, img_name, img_path, img_size, save_path) 





