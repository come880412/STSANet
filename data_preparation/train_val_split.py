'''
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import os
from distutils.dir_util import copy_tree


# data path
root = '/mnt/sdb_path/JayChao/Project/NTU/Saliency_Detection/dataset/DHF1k'


def split():
    folder_names = os.listdir(os.path.join(root, 'annotation'))
    folder_names.sort()
    
    folder_train = folder_names[:600]
    folder_val = folder_names[600:]

    save_train_folder = os.path.join(root, 'train')
    save_val_folder = os.path.join(root, 'val')

    for folder in folder_train:
        if not os.path.isdir(os.path.join(save_train_folder, folder)):
            os.mkdir(os.path.join(save_train_folder, folder))
            copy_tree(os.path.join(root, 'annotation', folder), os.path.join(save_train_folder, folder))
    
    for folder in folder_val:
        if not os.path.isdir(os.path.join(save_val_folder, folder)):
            os.mkdir(os.path.join(save_val_folder, folder))
            copy_tree(os.path.join(root, 'annotation', folder), os.path.join(save_val_folder, folder))



if __name__ == '__main__':
    split()
