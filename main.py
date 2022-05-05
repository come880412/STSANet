'''
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import argparse
import os
import csv
import cv2
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import DHF1KDataset, DIEM_data
from utils import blur, cc, kldiv
from models.STSANet import STSANet

import warnings
warnings.filterwarnings("ignore")

#torch.autograd.set_detect_anomaly(True)
def train(epochs, epoch, model, train_loader, optimizer):
    pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, epochs), unit=" step")
    # set model to train mode
    model.train()

    # statistics
    train_loss = 0
    for batch in train_loader:
        if (opt.dataset == 'DHF1k') or (opt.dataset == 'DIEM'):
            image, label = batch
        else:
            image, label, _, _ = batch

        image = image.permute(0,2,1,3,4).cuda(1)
        label = label.cuda(1)

        optimizer.zero_grad()
        pred_map = model(image)

        loss = kldiv(pred_map, label) - cc(pred_map, label)
        if torch.any(torch.isnan(kldiv(pred_map, label))):
            print('kldiv blows up!')
            pbar.update()
            optimizer.zero_grad()
            continue

        loss.backward()
        optimizer.step()

        train_loss += loss.data

        pbar.update()
        pbar.set_postfix(
        train_loss = f"{train_loss:.4f}",
        )
    pbar.close()
    
    return train_loss / len(train_loader)


def valid(epochs, epoch, model, valid_loader, dataset):
    pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Valid[%d/%d]"%(epoch, epochs), unit=" step")
    # set model to eval mode
    model.eval()

    # statistics
    valid_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            if (opt.dataset == 'DHF1k') or (opt.dataset == 'DIEM'):
                image, label = batch
            else:
                image, label, _, _ = batch

            image = image.permute(0,2,1,3,4).cuda(1)

            pred_map = model(image)

            label = label.squeeze(0).numpy()
            pred_map = pred_map.cpu().squeeze(0).numpy()
            pred_map = cv2.resize(pred_map, (label.shape[1], label.shape[0]))
            pred_map = blur(pred_map).unsqueeze(0).cuda(1)

            label = torch.FloatTensor(label).unsqueeze(0).cuda(1)

            loss = kldiv(pred_map, label) - cc(pred_map, label)
            valid_loss += loss.data

            pbar.update()
            pbar.set_postfix(
            valid_loss = f"{valid_loss:.4f}",
            )
    pbar.close()
    
    return valid_loss / len(valid_loader)


def main(epochs, model, train_loader, valid_loader, optimizer, scheduler):
    # record
    record = open('./metrics.tsv', 'a+')
    record_writer = csv.writer(record, delimiter='\t')
    record_writer.writerow(['Epoch', 'train_loss', 'valid_loss'])

    # save model threshold
    min_loss = 10000
    save_name = './checkpoints/{}.pth'.format(opt.dataset)
    print('start tarining...')
    for epoch in range(epochs):
        train_loss = train(epochs, epoch, model, train_loader, optimizer)
        valid_loss = valid(epochs, epoch, model, valid_loader, opt.dataset)

        print('epoch: {}  train_loss: {}  valid_loss: {}\n'.format(epoch+1, train_loss, valid_loss))
        record_writer.writerow([epoch+1, train_loss, valid_loss])

        if valid_loss <= min_loss:
            min_loss = valid_loss
            torch.save({'model': model.state_dict()}, save_name)
        
        scheduler.step(valid_loss)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """train_model"""
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs you want to train")
    parser.add_argument("--root", type=str, default='../dataset/DIEM/', help="path to dataset")
    parser.add_argument("--dataset", type=str, default='DIEM', help= 'DHF1K/DIEM')

    parser.add_argument("--backbone_pretrained", type=str, default='./checkpoints/S3D_kinetics400.pt', help="path to pretrained backbone weight")
    parser.add_argument("--load", type=str, default='', help="path to model checkpoints")

    parser.add_argument("--batch_size", type=int, default=1, help="number of batch_size")
    parser.add_argument("--workers", type=int, default=0, help="number of threads")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")

    parser.add_argument("--image_width", type=int, default=384, help="image width")
    parser.add_argument("--image_height", type=int, default=224, help="image height")
    parser.add_argument("--temporal", type=int, default=32, help="Temporal dimension")
    opt = parser.parse_args()

    if opt.dataset == 'DHF1k':
        train_data = DHF1KDataset(opt, path_data=os.path.join(opt.root, 'train'), len_snippet=opt.temporal, mode="train")
        valid_data = DHF1KDataset(opt, path_data=os.path.join(opt.root, 'val'), len_snippet=opt.temporal, mode="val")
    elif opt.dataset == 'DIEM':
        train_data = DIEM_data(opt, opt.root, 'train')
        valid_data = DIEM_data(opt, opt.root, 'test')
    
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.workers)
    valid_loader = DataLoader(valid_data, batch_size=1, num_workers=opt.workers)
    
    # define model
    model = STSANet(opt.temporal, opt.image_width, opt.image_height).cuda(1)
    """Load backbone pretrained weight"""
    if opt.load:
        print('Load pretrained model!')
        model.load_state_dict(torch.load(opt.load)['model'])
    else:
        if os.path.isfile(opt.backbone_pretrained):
            print ('loading weight file')
            weight_dict = torch.load(opt.backbone_pretrained)
            model_dict = model.backbone.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if 'base.' in name:
                    bn = int(name.split('.')[1])
                    sn_list = [0, 5, 8, 14]
                    sn = sn_list[0]
                    if bn >= sn_list[1] and bn < sn_list[2]:
                        sn = sn_list[1]
                    elif bn >= sn_list[2] and bn < sn_list[3]:
                        sn = sn_list[2]
                    elif bn >= sn_list[3]:
                        sn = sn_list[3]
                    name = '.'.join(name.split('.')[2:])
                    name = 'base%d.%d.'%(sn_list.index(sn)+1, bn-sn)+name
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        print (' size? ' + name, param.size(), model_dict[name].size())
                else:
                    print (' name? ' + name)

            model.backbone.load_state_dict(model_dict)
        else:
            print ('weight file?')

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1)

    # start training
    main(opt.n_epochs, model, train_loader, valid_loader, optimizer, scheduler)
