'''
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse


class DHF1KDataset(Dataset):
	def __init__(self, path_data, len_snippet, mode="train", multi_frame=0, alternate=1):
		''' mode: train, val, save '''
		self.path_data = path_data
		self.len_snippet = len_snippet
		self.mode = mode
		self.multi_frame = multi_frame
		self.alternate = alternate
		self.image_width, self.image_height = opt.image_width, opt.image_height
		self.img_transform = transforms.Compose([
			transforms.Resize((self.image_height, self.image_width)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		if self.mode == "train":
			self.video_names = os.listdir(path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(path_data,d,'images'))) for d in self.video_names]
		elif self.mode=="val":
			self.list_num_frame = []
			for v in os.listdir(path_data):
				for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))- self.alternate * self.len_snippet, 4*self.len_snippet):
					self.list_num_frame.append((v, i))
		else:
			self.list_num_frame = []
			for v in os.listdir(path_data):
				for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))-self.alternate * self.len_snippet, self.len_snippet):
					self.list_num_frame.append((v, i))
				self.list_num_frame.append((v, len(os.listdir(os.path.join(path_data,v,'images')))-self.len_snippet))

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		if self.mode == "train":
			file_name = self.video_names[idx]
			start_idx = np.random.randint(0, self.list_num_frame[idx]-self.alternate * self.len_snippet+1)
		elif self.mode == "val" or self.mode=="save":
			(file_name, start_idx) = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, file_name, 'images')
		path_annt = os.path.join(self.path_data, file_name, 'maps')

		clip_img = []
		clip_gt = []
		for i in range(self.len_snippet):
			img = Image.open(os.path.join(path_clip, 'frame%d.png'%(start_idx+self.alternate*i))).convert('RGB')
			sz = img.size

			if self.mode!="save":
				gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+self.alternate*i+1))).convert('L'))
				gt = gt.astype('float')
				
				if self.mode == "train":
					gt = cv2.resize(gt, (self.image_width, self.image_height))
				
				if np.max(gt) > 1.0:
					gt = gt / 255.0
				clip_gt.append(torch.FloatTensor(gt))

			clip_img.append(self.img_transform(img))
			
		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
		if self.mode!="save":
			clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))
		if self.mode=="save":
			return clip_img, start_idx, file_name, sz
		else:
			if self.multi_frame==0:
				return clip_img, clip_gt[15]
			return clip_img, clip_gt

class DIEM_data(Dataset):
	def __init__(self, opt, root, mode):
		self.root = root
		self.mode = mode
		self.opt = opt
		self.temporal = opt.temporal
		self.image_width, self.image_height = opt.image_width, opt.image_height

		self.img_transform = transforms.Compose([
			transforms.Resize((self.image_height, self.image_width)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])

		self.data_root = os.path.join(self.root, 'data')
		self.ground_truth_root = os.path.join(self.root, 'annotation')

		self.list_num_frame = []
		self.frame_name = []
        
		if self.mode == 'train':
			label_root = os.path.join(self.root, 'DIEM_list_train_fps.txt')
			label_info = np.loadtxt(label_root, delimiter=' ', dtype=np.str)
			for info in label_info:
				data_name = info[0]
				data_path = os.path.join(self.data_root, data_name)
				self.frame_name.append(data_name)
				self.list_num_frame.append(len(os.listdir(data_path)))
	
		elif self.mode == 'test':
			label_root = os.path.join(self.root, 'DIEM_list_test_fps.txt')
			label_info = np.loadtxt(label_root, delimiter=' ', dtype=np.str)
			self.list_num_frame = []
			for info in label_info:
				data_name = info[0]
				path_data = os.path.join(self.data_root, data_name)
				for i in range(0, len(os.listdir(path_data))- self.temporal-1, 4*self.temporal):
					self.list_num_frame.append((data_name, i))
	def check_frame(self, path):
		img = cv2.imread(path, 0)
		return img.max()!=0

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		if self.mode == 'train':
			frame_name = self.frame_name[idx]
			start_idx = np.random.randint(1, self.list_num_frame[idx]- self.temporal + 1)

			path_annt = os.path.join(self.ground_truth_root, frame_name, 'maps', 'eyeMap_%05d.jpg' % (start_idx + 15))

			clip_gt = np.array(Image.open(path_annt).convert('L'))
			clip_gt = clip_gt.astype('float')
			clip_gt = cv2.resize(clip_gt, (self.opt.image_height, self.opt.image_width))
			if np.max(clip_gt) > 1.0:
				clip_gt = clip_gt / 255.0
			clip_gt = torch.FloatTensor(clip_gt)

			clip_img = []
			for i in range(start_idx, start_idx + 32, 1):
				path_frame = os.path.join(self.data_root, frame_name, 'img_%05d.jpg' % (i))
				img = Image.open(path_frame).convert('RGB')

				clip_img.append(self.img_transform(img))
			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
			return clip_img, clip_gt
		
		elif self.mode == 'test':
			(video_name, start_idx) = self.list_num_frame[idx]

			path_clip = os.path.join(self.data_root, video_name)
			path_annt = os.path.join(self.ground_truth_root, video_name, 'maps')

			clip_img = []
			
			for i in range(self.temporal):
				img = Image.open(os.path.join(path_clip, 'img_%05d.jpg'%(start_idx+i+1))).convert('RGB')
				sz = img.size		
				clip_img.append(self.img_transform(img))
				
			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
			
			gt = np.array(Image.open(os.path.join(path_annt, 'eyeMap_%05d.jpg'%(start_idx+16))).convert('L'))
			gt = gt.astype('float')
			
			gt = cv2.resize(gt, (self.image_width, self.image_height))

			if np.max(gt) > 1.0:
				gt = gt / 255.0
			#assert gt.max()!=0, (start_idx, video_name)
			return clip_img, gt

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs you want to train")
	parser.add_argument("--root", type=str, default='../dataset/DIEM/', help="path to dataset")
	parser.add_argument("--dataset", type=str, default='DHF1k', help= 'DHF1K/DIEM')

	parser.add_argument("--backbone_pretrained", type=str, default='./checkpoints/S3D_kinetics400.pt', help="path to pretrained backbone weight")
	parser.add_argument("--load", type=str, default='', help="path to model checkpoints")

	parser.add_argument("--batch_size", type=int, default=1, help="number of batch_size")
	parser.add_argument("--workers", type=int, default=0, help="number of threads")
	parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")

	parser.add_argument("--image_width", type=int, default=384, help="image width")
	parser.add_argument("--image_height", type=int, default=224, help="image height")
	parser.add_argument("--temporal", type=int, default=32, help="Temporal dimension")
	opt = parser.parse_args()

	dataset = DIEM_data(opt, '../dataset/DIEM/', 'test')
	dataloader = DataLoader(dataset, batch_size=3, num_workers=0)

	data_iter = iter(dataloader)

	image, gt = data_iter.next()
	print(image.shape, gt.shape)
