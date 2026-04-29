import os
import json
from click import prompt
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG','bmp', 'PNG', 'gif'])

# 六类实例的值
ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0]) # label 1
LowVeg = np.array([255, 255, 0]) # label 2
Tree = np.array([0, 255, 0]) # label 3
Car = np.array([0, 255, 255]) # label 4
Clutter = np.array([0, 0, 255]) # label 5

num_classes = 6
classes = ('ImSurf', 'Building', 'LowVeg' ,'Tree', 'Car', 'Clutter')
rgb_label_img = [[255, 255, 255],[0, 0, 255],[0, 255, 255],[0, 255, 0],[255, 255, 0],[255, 0, 0]]

def rgb_to_2D_label(_label):  # H W C
    label_seg = np.full((_label.shape[0], _label.shape[1]), 255, dtype=np.uint8)
    for idex in range(num_classes):
        color_mask = np.array(rgb_label_img[idex], dtype=np.uint8)
        class_mask = np.all(_label == color_mask, axis=-1)
        label_seg[class_mask] = idex

    return label_seg

def onehot_to_mask(img):
    mask = np.argmax(img,axis=-1)
    return mask

"""def load_prompt_csv(csv_path):
    
    读取文本标注csv文件，返回：
        prompt_dict = {
            "1.png":"a photo of ...",
            ...
        }
    
    prompt_dict = {}

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prompt csv not found:{csv_path}")

    with open (csv_path,'r',encoding='utf-8-sig',newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"Empty csv file:{csv_path}")

    # 默认跳过第一行表头
    for row in rows[1:]:
        if len(row) < 2:
            continue
        img_name = row[0].strip()
        prompt = row[1].strip()

        if img_name == "":
           continue

        prompt_dict[img_name] = prompt

    return prompt_dict"""



class DataLoaderTrain(Dataset):
    def __init__(self, img_dir, text_path_train):
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

        # 读取两个模态各自的文本标注
        with open(text_path_train,'r',encoding = 'utf-8') as f:
            self.annotations = json.load(f)

        self.sizex       = len(self.tar_ir_filenames)  # get the size of target

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
        tar_seg_img = torch.from_numpy(target_seg).long()

        filename = os.path.splitext(os.path.split(tar_ir_path)[-1])[0]

        prompt_rgb = self.annotations[filename]["infrared_prompt"]
        prompt_ir = self.annotations[filename]["visible_prompt"]

        return tar_ir_img, inp_ir_img, tar_rgb_img, inp_rgb_img, tar_seg_img,prompt_ir,prompt_rgb, filename

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir,text_path_test):
        super(DataLoaderTest, self).__init__()
        inp_ir_files = sorted(os.listdir(os.path.join(inp_dir, 'ir')))
        inp_rgb_files = sorted(os.listdir(os.path.join(inp_dir, 'rgb')))
        tar_seg_files = sorted(os.listdir(os.path.join(inp_dir,'seg')))

        self.inp_ir_filenames = [os.path.join(inp_dir, 'ir', x) for x in inp_ir_files if is_image_file(x)]
        self.inp_rgb_filenames = [os.path.join(inp_dir, 'rgb', x) for x in inp_rgb_files if is_image_file(x)]
        self.tar_seg_filenames = [os.path.join(inp_dir,'seg',x) for x in tar_seg_files if is_image_file(x) ]

        # 读取测试集两个模态的文本标注
        with open(text_path_test, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self.inp_ir_size = len(self.inp_ir_filenames)
        self.inp_rgb_size = len(self.inp_rgb_filenames)


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

        prompt_rgb = self.annotations[filename]["visible_prompt"]
        prompt_ir = self.annotations[filename]["infrared_prompt"]


        return inp_ir, inp_rgb, tar_seg_img, prompt_ir,prompt_rgb, filename
