import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from model import UNet
from dataset import liftDataset
from utils import *
from os import listdir
import time


def train(config, begin_epoch=0):


    exists_or_mkdir(config.train.valid_saving_path)
    exists_or_mkdir(config.train.ckpt_saving_path)

    device = torch.device("cuda:{}".format(config.train.device) if torch.cuda.is_available() else "cpu")
    print("Training on {}".format(device))
    
    net = UNet(config.n_slices , config.n_ang).to(device) 
    
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=config.train.lr_init, betas=(config.train.beta1, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, config.train.decay_every, config.train.lr_decay)

    dataset = liftDataset(config.train.lf3d_path, config.train.target3d_path, config.train.img_size, config.n_slices, config.n_ang, fmt='.tif', preload=config.dataset_preload)
    n_val = config.train.n_val
    n_train = len(dataset) - n_val
    
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, pin_memory=True)


    if begin_epoch != 0:
        ckpt_list = listdir(config.train.ckpt_saving_path)
        ckpt_name = '{}_vcdnet_epoch{}.pth'.format(config.label, begin_epoch)
        if ckpt_name in ckpt_list:
            ckpt = torch.load(config.train.ckpt_saving_path+ckpt_name, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            net = net.to(device)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('Load checkpoint file : {}'.format(ckpt_name))
        else:
            raise Exception('No such checkpoint file')    

    losses_train = []
    losses_val = []

    for epoch in range(begin_epoch, config.train.n_epoch+1):

        net.train()
        step_time = time.time()
        running_loss = 0
        for i, batch in enumerate(train_loader):

            net.zero_grad()

            lf_batch = batch['lf'].to(device)
            gt_batch = batch['gt'].to(device)

            pred = net(lf_batch)
            loss = criterion(pred, gt_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print("\rEpoch:[%d/%d] iter:[%d/%d] times: %4.3fs, loss:%.6f" % (epoch, config.train.n_epoch,
                i+1, len(train_loader), time.time() - step_time, loss.item()), end='')
        running_loss = running_loss / len(train_loader)
        losses_train.append(running_loss)
        
        net.eval()
        running_loss = 0
        for idx, batch in enumerate(val_loader):
            with torch.no_grad():
                lf_batch = batch['lf'].to(device)
                gt_batch = batch['gt'].to(device)

                pred = net(lf_batch)
                loss = criterion(pred, gt_batch)
                running_loss += loss.item()
                if epoch == begin_epoch:
                    save_img3d('gt_{}.tif'.format(idx), config.train.valid_saving_path, gt_batch, bitdepth=16)
                    save_img3d('lf_{}.tif'.format(idx), config.train.valid_saving_path, lf_batch, bitdepth=16)
                elif (epoch%config.train.ckpt_saving_interval == 0):
                    save_img3d('val_epoch{}_{}.tif'.format(epoch, idx), config.train.valid_saving_path, pred, bitdepth=16)                
        running_loss = running_loss / len(val_loader)
        print(", val_loss:%.6f" % (running_loss), end='')
        losses_val.append(running_loss)
        if (epoch != begin_epoch) and (epoch%config.train.ckpt_saving_interval == 0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_train': losses_train[-1],
                'loss_val': losses_val[-1]
                }, config.train.ckpt_saving_path + '{}_liftnet_epoch{}.pth'.format(config.label, epoch))
                # todo the saved file seems to be too big
        
        print(", learning rate:%.6f" % (optimizer.param_groups[0]["lr"]), end='\n')
        scheduler.step()
    return net

if __name__ == '__main__':

    config = Config(label = 'simulated_beads_25_21', n_ang=25, n_slices=21)
    config.train.target3d_path = 'data/train/gt/'
    config.train.lf3d_path = 'data/train/lf/'
    config.train.n_epoch = 50
    config.train.ckpt_saving_interval = 2
    config.train.decay_every = 40
    config.train.n_val = 2
    config.train.batch_size = 1
    config.train.lr_init = 1e-4
    config.show_basic_paras()
    config.show_train_paras()
    train(config, begin_epoch=0)