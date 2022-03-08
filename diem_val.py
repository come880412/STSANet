'''
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import os
import numpy as np
import cv2
import torch
from models.STSANet import STSANet
from scipy.ndimage.filters import gaussian_filter
import argparse
import tqdm

from utils import *
from torchvision import transforms
from os.path import join
import scipy.io as sio
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

def validate(args, model):
    ''' read frames in path_indata and generate frame-wise saliency maps in path_output '''
    # optional two command-line arguments
    path_indata = args.root
    len_temporal = args.temporal

    torch.backends.cudnn.benchmark = False
    model.eval()

    # iterate over the path_indata directory
    list_indata = []
    with open(os.path.join(path_indata, './DIEM_list_test_fps.txt'), 'r') as f:
        for line in f.readlines():
            name = line.split(' ')[0].strip()
            list_indata.append(name)
    # list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()

    frame_sim_loss = 0
    frame_cc_loss = 0
    frame_nss_loss = 0
    frame_aucj_loss = 0
    frame_cnt = 0

    avg_video_sim_loss = 0
    avg_video_cc_loss = 0
    avg_video_nss_loss = 0
    avg_video_aucj_loss = 0
    num_videos = 0
    valid_loss = 0

	# list_indata = ['BBC_wildlife_serpent_1280x704']
    pbar = tqdm.tqdm(total=len(list_indata), ncols=0, desc="valid", unit=" step")
    for idx, dname in enumerate(list_indata):
        list_frames = [f for f in os.listdir(os.path.join(path_indata, 'data', dname)) if os.path.isfile(os.path.join(path_indata, 'data', dname, f))]
        list_frames.sort()

        video_sim_loss = 0
        video_cc_loss = 0
        video_nss_loss = 0
        video_aucj_loss = 0
        num_frames = 0

        snippet = []
        for i in range(0, len(list_frames) + 16, 1):
            if i < len(list_frames):
                torch_img, img_size = torch_transform(args, os.path.join(path_indata, 'data', dname, list_frames[i]))
                snippet.append(torch_img)
            else: # for last 16 frames
                torch_img, img_size = torch_transform(args, os.path.join(path_indata, 'data', dname, list_frames[len(list_frames)-1]))
                snippet.append(torch_img)
            
            if i < len_temporal / 2 - 1: # first 15 frames
                torch_img, img_size = torch_transform(args, os.path.join(path_indata, 'data', dname, list_frames[0]))

                frame = []
                repeat_frame_num = int(len_temporal / 2 - i)
                for j in range(repeat_frame_num):
                    frame.append(torch_img)
                for j in range(1, i + len_temporal//2 + 1, 1):
                    torch_img, img_size = torch_transform(args, os.path.join(path_indata, 'data', dname, list_frames[j]))
                    frame.append(torch_img)

                clip = torch.FloatTensor(torch.stack(frame, dim=0)).unsqueeze(0)
                clip = clip.permute((0,2,1,3,4))
                
                sim_loss, cc_loss, nss_loss, aucj_loss, loss = process(model, clip, path_indata, dname, i+1)

            if i >= len_temporal - 1: # 16~ last frame
                clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                clip = clip.permute((0,2,1,3,4))
                sim_loss, cc_loss, nss_loss, aucj_loss, loss = process(model, clip, path_indata, dname, i-15)
                del snippet[0]

            if np.isnan(sim_loss) or np.isnan(cc_loss) or np.isnan(nss_loss): # no saliency map
                continue
            else:
                frame_sim_loss += sim_loss
                frame_nss_loss += nss_loss
                frame_cc_loss += cc_loss
                frame_aucj_loss += aucj_loss
                frame_cnt += 1

                video_sim_loss += sim_loss
                video_nss_loss += nss_loss
                video_cc_loss += cc_loss
                video_aucj_loss += aucj_loss
                num_frames += 1

                valid_loss += loss.data

        num_videos += 1
        avg_video_sim_loss += video_sim_loss / num_frames
        avg_video_nss_loss += video_nss_loss / num_frames
        avg_video_cc_loss += video_cc_loss / num_frames
        avg_video_aucj_loss += video_aucj_loss / num_frames

        pbar.update()
        pbar.set_postfix(
        valid_loss = f"{valid_loss / (idx+1):.4f}",
        )
        

    
    pbar.set_postfix(
    SIM = f"{frame_sim_loss/frame_cnt:.3f}",
    CC = f"{frame_cc_loss/frame_cnt:.3f}",
    NSS = f"{frame_nss_loss/frame_cnt:.3f}",
    AUCJ = f"{frame_aucj_loss/frame_cnt:.3f}",
    Avg_SIM = f"{avg_video_sim_loss/num_videos:.3f}",
    Avg_CC = f"{avg_video_cc_loss/num_videos:.3f}",
    Avg_NSS = f"{avg_video_nss_loss/num_videos:.3f}",
    Avg_AUCJ = f"{avg_video_aucj_loss/num_videos:.3f}",
    )

    return valid_loss / frame_cnt

		
def torch_transform(opt, path):
	img_transform = transforms.Compose([
			transforms.Resize((opt.image_height, opt.image_width)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
	])
	img = Image.open(path).convert('RGB')
	sz = img.size
	img = img_transform(img)
	return img, sz

def blur(img):
	k_size = 11
	bl = cv2.GaussianBlur(img,(k_size,k_size),0)
	return torch.FloatTensor(bl)

def get_fixation(path_indata, dname, frame_no):
	info = sio.loadmat(join(path_indata, 'annotation', dname, 'fixMap_%05d.mat' % frame_no))
	return info['eyeMap']

def process(model, clip, path_indata, dname, frame_no):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ''' process one clip and save the predicted saliency map '''
    with torch.no_grad():
        smap = model(clip.to(device)).cpu().data[0]

    smap = smap.numpy()
    
    gt = cv2.imread(join(path_indata, 'annotation', dname, 'maps', 'eyeMap_%05d.jpg' % frame_no), 0)
    smap = cv2.resize(smap, (gt.shape[1], gt.shape[0]))
    fix = get_fixation(path_indata, dname, frame_no)
    smap = blur(smap)

    gt = torch.FloatTensor(gt).unsqueeze(0)
    fix = torch.FloatTensor(fix).unsqueeze(0)
    smap = smap.unsqueeze(0)
    loss = kldiv(smap, gt) - cc(smap, gt)
	# print(smap.size(), gt.size())
    sim_loss = similarity(smap, gt)
    cc_loss = cc(smap, gt)
    nss_loss = nss(smap, fix)
    aucj_loss = auc_judd(smap, fix)

    if np.isnan(sim_loss) or np.isnan(cc_loss) or np.isnan(nss_loss):
        assert gt.numpy().max()==0, gt.numpy().max()
    return sim_loss, cc_loss, nss_loss, aucj_loss, loss

	
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """train_model"""
    parser.add_argument("--root", type=str, default='../dataset/DIEM/', help="path to dataset")
    parser.add_argument("--load", type=str, default='./checkpoints/DIEM.pth', help="path to model checkpoints")
    parser.add_argument("--image_width", type=int, default=384, help="image width")
    parser.add_argument("--image_height", type=int, default=224, help="image height")
    parser.add_argument("--temporal", type=int, default=32, help="Temporal dimension")
    opt = parser.parse_args()
    
    model = STSANet(opt.temporal, opt.image_width, opt.image_height).cuda()
    # """Load backbone pretrained weight"""
    if opt.load:
        print('Load pretrined model!!')
        model.load_state_dict(torch.load(opt.load)['model'])
    validate(opt, model)