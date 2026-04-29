import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import json
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG','bmp', 'PNG', 'gif'])

num_classes = 8
classes = ('background', 'farmland', 'city', 'village' ,'water', 'forest', 'road', 'others')
rgb_lable_img = [[0, 0, 0], [204, 102, 0],[255, 0, 0],[255, 255, 0],[0, 0, 255],[85, 167, 0],[0, 255, 255], [153, 102, 153]]
PALETTE = [[0, 0, 0], [10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40], [50, 50, 50], [60, 60, 60], [70, 70, 70]]

def rgb_to_2D_label(_label):  # H W C
    label_seg = np.zeros((_label.shape[0], _label.shape[1]), dtype=np.uint8)#320 320
    for idex in range(num_classes):
        color_mask = PALETTE[idex]
        class_mask = np.all(_label == color_mask, axis=-1)
        label_seg[class_mask] = idex
    return label_seg

class DataLoaderTrain(Dataset):
    def __init__(self, img_dir,text_path):
        super(DataLoaderTrain, self).__init__()

        inp_ir_files = sorted(os.listdir(os.path.join(img_dir, 'ir', 'input')))
        tar_ir_files = sorted(os.listdir(os.path.join(img_dir, 'ir', 'target')))
        inp_rgb_files = sorted(os.listdir(os.path.join(img_dir, 'rgb', 'input')))
        tar_rgb_files = sorted(os.listdir(os.path.join(img_dir, 'rgb', 'target')))
        tar_seg_files = sorted(os.listdir(os.path.join(img_dir, 'seg')))
        self.inp_ir_filenames = [os.path.join(img_dir, 'ir', 'input', x)  for x in inp_ir_files if is_image_file(x)]
        self.tar_ir_filenames = [os.path.join(img_dir, 'ir', 'target', x) for x in tar_ir_files if is_image_file(x)]
        self.inp_rgb_filenames = [os.path.join(img_dir, 'rgb', 'input', x) for x in inp_rgb_files if is_image_file(x)]
        self.tar_rgb_filenames = [os.path.join(img_dir, 'rgb', 'target', x) for x in tar_rgb_files if is_image_file(x)]
        self.tar_seg_filenames = [os.path.join(img_dir, 'seg', x) for x in tar_seg_files if is_image_file(x)]
        self.sizex       = len(self.tar_ir_filenames)  # get the size of target

        with open(text_path,'r',encoding = 'utf-8') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_ir_path = self.inp_ir_filenames[index_]
        tar_ir_path = self.tar_ir_filenames[index_]
        inp_rgb_path = self.inp_rgb_filenames[index_]
        tar_rgb_path = self.tar_rgb_filenames[index_]
        tar_seg_path = self.tar_seg_filenames[index_]

        inp_ir_img = Image.open(inp_ir_path).convert('RGB')
        tar_ir_img = Image.open(tar_ir_path).convert('RGB')
        inp_rgb_img = Image.open(inp_rgb_path).convert('RGB')
        tar_rgb_img = Image.open(tar_rgb_path).convert('RGB')
        tar_seg_img = Image.open(tar_seg_path).convert('RGB')

        # ---------rgb_to_2D_label-----------#
        tar_seg_array = np.array(tar_seg_img)
        # print(tar_seg_array.shape)
        target_seg = rgb_to_2D_label(tar_seg_array)
        # --------------------#

        inp_ir_img = TF.to_tensor(inp_ir_img)
        tar_ir_img = TF.to_tensor(tar_ir_img)
        inp_rgb_img = TF.to_tensor(inp_rgb_img)
        tar_rgb_img = TF.to_tensor(tar_rgb_img)
        #-----------------------------
        tar_seg_img = torch.from_numpy(target_seg).long()

        filename = os.path.splitext(os.path.split(tar_ir_path)[-1])[0]

        # 取两个模态的文本标注
        infrared_prompt = self.annotations[filename]["infrared_prompt"]
        visible_prompt = self.annotations[filename]["visible_prompt"]

        return tar_ir_img, inp_ir_img, tar_rgb_img, inp_rgb_img, tar_seg_img, infrared_prompt, visible_prompt, filename


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir,text_path):
        super(DataLoaderTest, self).__init__()
        inp_ir_files = sorted(os.listdir(os.path.join(inp_dir, 'ir')))
        inp_rgb_files = sorted(os.listdir(os.path.join(inp_dir, 'rgb')))
        tar_seg_files = sorted(os.listdir(os.path.join(inp_dir,'seg')))

        self.inp_ir_filenames = [os.path.join(inp_dir, 'ir', x) for x in inp_ir_files if is_image_file(x)]
        self.inp_rgb_filenames = [os.path.join(inp_dir, 'rgb', x) for x in inp_rgb_files if is_image_file(x)]
        self.tar_seg_filenames = [os.path.join(inp_dir,'seg',x) for x in tar_seg_files if is_image_file(x) ]
        
        self.inp_ir_size = len(self.inp_ir_filenames)
        self.inp_rgb_size = len(self.inp_rgb_filenames)
        self.tar_seg_size = len(self.tar_seg_filenames)

        with open(text_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return self.inp_ir_size

    def __getitem__(self, index):
        path_ir_inp = self.inp_ir_filenames[index]
        path_rgb_inp = self.inp_rgb_filenames[index]
        path_seg_tar = self.tar_seg_filenames[index]

        filename = os.path.splitext(os.path.split(path_ir_inp)[-1])[0]
        inp_ir = Image.open(path_ir_inp).convert('RGB')
        inp_rgb = Image.open(path_rgb_inp).convert('RGB')
        tar_seg = Image.open(path_seg_tar).convert('RGB')

        # ---------------------seg彩色图->[H,W]类别标签--------------------------------
        tar_seg_array = np.array(tar_seg)
        target_seg = rgb_to_2D_label(tar_seg_array)

         #----------------------图像转tensor--------------------------------------
        inp_ir = TF.to_tensor(inp_ir)
        inp_rgb = TF.to_tensor(inp_rgb)
        # ---------------------------seg转long tensor--------------------------------
        tar_seg_img = torch.from_numpy(target_seg).long()

        infrared_prompt = self.annotations[filename]["infrared_prompt"]
        visible_prompt = self.annotations[filename]["visible_prompt"]

        return inp_ir, inp_rgb, tar_seg_img, infrared_prompt, visible_prompt,filename
