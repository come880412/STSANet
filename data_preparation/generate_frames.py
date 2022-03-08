'''
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import os
import cv2

# data path
root = '/mnt/sdb_path/JayChao/Project/NTU/Saliency_Detection/dataset/DHF1k/'


def generate(root, mode='train'):
    video_path = os.path.join(root, 'video')
    train_folders = os.listdir(os.path.join(root, mode))
    train_folders.sort()
    save_path = os.path.join(root, mode)

    for folder_name in train_folders:
        folder_name_ = folder_name[1:]

        if not os.path.isdir(os.path.join(save_path, folder_name, 'images')):
            os.mkdir(os.path.join(save_path, folder_name, 'images'))
        
        cap = cv2.VideoCapture(os.path.join(video_path, '{}.AVI'.format(folder_name_)))
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(os.path.join(save_path, folder_name, 'images', 'frame{}.png'.format(count)), frame)
            count += 1



if __name__ == '__main__':
    # generate(root, mode='train')
    generate(root, mode='val')
