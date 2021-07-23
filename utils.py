import numpy as np
import imageio
from easydict import EasyDict as edict
from pathlib import Path
from os.path import splitext
import torch

def get_img2d_fn(filename, filepath, normalize_fn):

    img2d = np.array(imageio.imread(filepath+filename))
    assert img2d.dtype == 'uint16' or img2d.dtype == 'uint8', "Only supports 8bit and 16bit image" 
    bitdepth = 16 if img2d.dtype == 'uint16' else 8    
    img2d = img2d.astype(np.float32)
    assert img2d.ndim == 2, "Only supports 2D grey images"
    img2d = img2d[np.newaxis, :, :]
    return normalize_fn(img2d, bitdepth) # channel(1) height width

def get_img3d_fn(filename, filepath, normalize_fn):

    img3d = np.array(imageio.volread(filepath+filename))
    assert img3d.dtype == 'uint16' or img3d.dtype == 'uint8', "Only supports 8bit and 16bit image" 
    bitdepth = 16 if img3d.dtype == 'uint16' else 8    
    img3d = img3d.astype(np.float32)    
    assert img3d.ndim == 3, "Only supports 3D grey images"
    return normalize_fn(img3d, bitdepth) # depth height width

def get_lf_views(filename, filepath, normalize_fn, n_num=11, padding=False):

    lf3d = get_img2d_fn(filename, filepath, normalize_fn)  # channel(1) height width
    return lf_views_fn(lf3d, n_num, padding) # channel(n_num*n_num) height width

def normalize_1(x, bitdepth=16):
    assert bitdepth==8 or bitdepth==16, "Only supports 8bit and 16bit image"
    x = x / 65535 if bitdepth==16 else x / 255
    return x

def normalize_2(x, bitdepth=16):
    assert bitdepth==8 or bitdepth==16, "Only supports 8bit and 16bit image"
    x = x / (65535. / 2.) if bitdepth==16 else x / (255. / 2.)
    x = x - 1
    return x

def lf_views_fn(lf3d, n_num=11, padding=False):

    n = n_num
    c, h, w = lf3d.shape # channel(1) height width
    if padding:
        lf_views = np.zeros([n*n, h, w])
        d = 0
        for i in range(n):
            for j in range(n):
                lf_views[d, i:h:n, j:w:n] = lf3d[0, i:h:n, j:w:n]
                d = d + 1
    else:
        lf_views = np.zeros([n*n, int(h/n), int(w/n)])
        d = 0
        for i in range(n):
            for j in range(n):
                lf_views[d, :, :] = lf3d[0, i:h:n, j:w:n]
                d = d + 1
    return lf_views

def save_img2d_(filename, filepath, img, bitdepth=16, norm_mode=1):

    assert bitdepth==8 or bitdepth==16, "Only supports 8bit and 16bit image output"
    if norm_mode == 1:
        img = img * 65535 if bitdepth==16 else img * 255
    elif norm_mode == 2:
        img = img + 1
        img = img * 65535. / 2. if bitdepth==16 else img * 255. / 2.
    else:
        raise NotImplementedError
    img = img * 65535 if bitdepth==16 else img * 255
    img = np.clip(img, 0, 65535).astype(np.uint16) if bitdepth==16 else np.clip(img, 0, 255).astype(np.uint8)
    imageio.imwrite(filepath+filename, img[0,:,:])

def save_img3d_(filename, filepath, img, bitdepth=16, norm_mode=1):

    assert bitdepth==8 or bitdepth==16, "Only supports 8bit and 16bit image output"
    if norm_mode == 1:
        img = img * 65535 if bitdepth==16 else img * 255
    elif norm_mode == 2:
        img = img + 1
        img = img * 65535. / 2. if bitdepth==16 else img * 255. / 2.
    else:
        raise NotImplementedError
    img = np.clip(img, 0, 65535).astype(np.uint16) if bitdepth==16 else np.clip(img, 0, 255).astype(np.uint8) 
    imageio.volwrite(filepath+filename, img) ## depth height width

def save_img3d(filename, filepath, img, bitdepth=16, norm_mode=1):
# img: (batch) x depth x height x width
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    if img.ndim == 3:
        img = img[np.newaxis, :, :, :]
    elif img.ndim == 4:
        img = img
    else:
        print("Image dimension should be 3 or 4, (batch) x depth x height x width")
        raise NotImplementedError
    b, d, h, w = img.shape
    if b == 1:
        save_img3d_(filename, filepath, img[0,:,:,:], bitdepth, norm_mode)
    else:
        for i in range(b):
            name = splitext(filename)[0] + "_{:02d}".format(i) + splitext(filename)[1] #batchsize < 100
            save_img3d_(name, filepath, img[i,:,:,:], bitdepth, norm_mode)




def exists_or_mkdir(dir):
    try:
        Path(dir).mkdir(parents=True, exist_ok=False)
        print('Create folder: {}'.format(dir))
    except FileExistsError:
        print('Folder already exists: {}'.format(dir))     



class Config():
    def __init__(self, label='liftnet_test', n_ang=25, n_slices=21):
        self.label = label 
        self.n_ang = n_ang
        self.n_slices = n_slices
        self.train = edict()
        self.test = edict()

        self.train.img_size = [256, 256]
        self.train.batch_size = 1
        
        self.train.lr_init = 1e-4
        self.train.beta1 = 0.9
        self.train.lr_decay = 0.1

        self.train.n_val = 2        
        self.train.n_epoch = 50
        self.train.decay_every = 30
        self.train.ckpt_saving_interval = 10

        self.train.valid_saving_path = "sample/valid/{}/".format(label)
        self.train.ckpt_saving_path = "checkpoint/{}/".format(label)
        self.train.target3d_path = 'data/train/gt/'
        self.train.lf3d_path = 'data/train/lf/'

        self.dataset_preload = True
        self.train.device = 0

        self.test.ckpt_loading_path = "checkpoint/{}/".format(label)
        self.test.img_size = [330, 330]
        self.test.lf3d_path = 'data/test/'
        self.test.saving_path = 'data/test/{}/'.format(label)

        self.test.device = 0
    
    def change_label(self, new_label):
        self.label = new_label
        self.train.valid_saving_path = "sample/valid/{}/".format(new_label)
        self.train.ckpt_saving_path = "checkpoint/{}/".format(new_label)
        self.test.ckpt_loading_path = "checkpoint/{}/".format(new_label)
        return new_label       

    def show_basic_paras(self):
        print('{:<30s}: {}'.format('label', self.label))
        print('{:<30s}: {}'.format('n_ang', self.n_ang))
        print('{:<30s}: {}'.format('n_slices', self.n_slices))

    def show_train_paras(self):
        print('{:<30s}:'.format('train.img_size'), self.train.img_size)
        print('{:<30s}:'.format('train.batch_size'), self.train.batch_size)
        print('{:<30s}:'.format('train.lr_init'), self.train.lr_init)
        print('{:<30s}:'.format('train.beta1'), self.train.beta1)
        print('{:<30s}:'.format('train.n_val'), self.train.n_val)
        print('{:<30s}:'.format('train.n_epoch'), self.train.n_epoch)
        print('{:<30s}:'.format('train.decay_every'), self.train.decay_every)
        print('{:<30s}:'.format('train.ckpt_saving_interval'), self.train.ckpt_saving_interval)        
        print('{:<30s}: {}'.format('train.valid_saving_path', self.train.valid_saving_path))
        print('{:<30s}: {}'.format('train.ckpt_saving_path', self.train.ckpt_saving_path)) 
        print('{:<30s}: {}'.format('train.target3d_path', self.train.target3d_path)) 
        print('{:<30s}: {}'.format('train.lf3d_path', self.train.lf3d_path)) 
        print('{:<30s}:'.format('dataset_preload'), self.dataset_preload)  
        print('{:<30s}:'.format('train.device'), self.train.device)  
    
    def show_test_paras(self):
        print('{:<30s}: {}'.format('test.ckpt_loading_path', self.test.ckpt_loading_path)) 
        print('{:<30s}:'.format('test.img_size'), self.test.img_size)
        print('{:<30s}: {}'.format('test.lf3d_path', self.test.lf3d_path)) 
        print('{:<30s}: {}'.format('test.saving_path', self.test.saving_path)) 
        print('{:<30s}:'.format('train.device'), self.train.device)          
       



if __name__ == '__main__':

    # img2d = get_img2d_fn('LF.tif', './data/', normalize_fn)
    # lf_views = lf_views_fn(img2d)
    # print(img2d.shape)
    # print(img2d.dtype)
    # img3d = get_img3d_fn('GT.tif', './data/', normalize_fn)
    # print(img3d)
    # save_img3d_('GT_save.tif', './data/', img3d)
    # save_img3d_('LF_views_save.tif', './data/', lf_views)


    # save_img2d_('LF_save.tif', './data/', img2d, bitdepth=8)

    config = Config(label = 'rbcDSRED_[m76-76]_step2um_N11_bgsub_20210112')

    exists_or_mkdir('data/test/')

    a=1

