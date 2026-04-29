import os
import kornia  # 用于图像处理
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import utils
#from data_RGB_mfnet import get_training_data, get_test_data
from data_RGB_potsdam import get_training_data, get_test_data
# from data_RGB_whu import get_training_data, get_test_data
from warmup_scheduler import GradualWarmupScheduler  # 先用小学习率热身，等模型稳定了再用正常学习率
from tqdm import tqdm  # for循环显示进度条
import torch.utils.data
from utils.seg_util import *
from utils.dice import *
import warnings
from utils.MEF_SSIM_loss import th_SSIM_LOSS, Y_Upper
import logging
import argparse
from Evaluator import *
from model import MMFNet as Net
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.seg_metrics import SegmentationMetric
from utils.save_img import save_img

print("A torch.cuda.is_available() =", torch.cuda.is_available())
print("A torch.cuda.device_count() =", torch.cuda.device_count())


def show_img(img):
    """
    显示形状为 (3, 320, 320) 的图像

    参数:
    img: 可以是 numpy array 或 PyTorch Tensor
    """
    # 如果是 PyTorch Tensor，转换为 numpy
    if isinstance(img, torch.Tensor):
        if img.requires_grad:
            img = img.detach().cpu().numpy()
        else:
            img = img.cpu().numpy()

    # 确保是 numpy 数组
    img = np.asarray(img)

    # 检查形状
    if img.shape != (3, 320, 320):
        raise ValueError(f"Expected shape (3, 320, 320), got {img.shape}")

    # 转换维度: (C, H, W) -> (H, W, C)
    img_display = img.transpose(1, 2, 0)

    # 显示图像
    plt.figure(figsize=(5, 5))
    plt.imshow(img_display)
    plt.axis('off')
    plt.title('3×320×320 Image Display')
    plt.show()


def rgb_to_ycbcr(img):
    """
    将R、G、B转换成Y、Cr、Cb

    """
    ycbcr = kornia.color.rgb_to_ycbcr(img)
    return ycbcr


# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)


def evaluation_one(ir_name, vi_name, f_name):
    """
    计算评价指标，计算的函数均来自Evaluator中创建的class Evaluator
    """
    ir = image_read_cv2(ir_name, 'GRAY')
    vi = image_read_cv2(vi_name, 'GRAY')
    fi = image_read_cv2(f_name, 'GRAY')
    EN = Evaluator.EN(fi)  # 熵
    SD = Evaluator.SD(fi)  # 标准差，表示对比度，SD越高->图像越清晰，亮度变化越丰富
    SF = Evaluator.SF(fi)  # 空间频率，表征图像纹理、细节多少。SF越大->边缘、纹理越丰富
    AG = Evaluator.AG(fi)  # 平均梯度,描述图像尖锐程度，AG越大->图像越锐利（细节更清楚）
    SCD = Evaluator.SCD(fi, ir, vi)  # 结构相似度度量
    VIFF = Evaluator.VIFF(fi, ir, vi)  # 视觉信息忠实度
    return EN, SD, SF, AG, SCD, VIFF


EXP_NAME = "potsdam_three_IPID_3"  # 实验名称：baseline/seg_high/seg_low
SEG_LOSS_WEIGHT = 1.0  # baseline=1.0；高权重=2.0； 低权重=0.5

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用哪块GPU运行以下程序‘0’代表第一块，‘1’代表第二块
NUM_EPOCHS = 30
BATCH_SIZE = 2  # 为什么设置这么小的batch_size
LR_INITIAL = 2e-5
LR_MIN = 1e-6
data = 'potsdam'

model_dir = os.path.join('./checkpoints', data, EXP_NAME, 'models')  # './checkpoints/potsdam/models'
best_model_path = os.path.join(model_dir, 'best_model.pth')
train_dir = os.path.join('./dataset/train/', data)
# 具体文本标注路径
text_path_train = "./dataset/train/potsdam/annotations/train.json"
text_path_test = "./dataset/test/potsdam/annotations/test.json"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
print("device:", device)

print("B before Net, cuda available =", torch.cuda.is_available())

model = Net(6)  # mfnet ;potsdam是6类；whu是7类

print("C after Net, cuda available =", torch.cuda.is_available())

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS - warmup_epochs,
                                                        eta_min=LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

train_dataset = get_training_data(train_dir,text_path_train)
print(f"训练数据路径: {train_dir}")
print(f"路径是否存在: {os.path.exists(train_dir)}")
print(f"路径内容: {os.listdir(train_dir)}")
print(f"数据集长度: {len(train_dataset)}")
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, drop_last=True,
                         pin_memory=True)  # drop_last=False
print('===> Loading datasets')
img_dir_test = os.path.join('./dataset/test/', data)
test_dataset = get_test_data(img_dir_test,text_path_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False,
                         pin_memory=True)
result_dir = os.path.join('./results/', data, EXP_NAME)
eva_dir = os.path.join('./eva/', data, EXP_NAME)
log_file = os.path.join(eva_dir, 'train_log.txt')
best_record_file = os.path.join(eva_dir, 'best_record.txt')


os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(eva_dir, exist_ok=True)

l1_criterion = nn.L1Loss()
# 像素值标为255表示”无效区域/模糊区域“，不参与计算
CE_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
Dice_criterion = DiceLoss(smooth=0.05, ignore_index=255)

best_EN, best_SD, best_SF, best_AG, best_SCD, best_VIFF = 0, 0, 0, 0, 0, 0
best_epoch = -1
# 训练前把权重写在日志中
with open(log_file, 'a', encoding='utf-8') as f:
    f.write(f"Experiment: {EXP_NAME}\n")
    f.write(f"SEG_LOSS_WEIGHT: {SEG_LOSS_WEIGHT}\n")
    f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"LR_INITIAL: {LR_INITIAL}\n\n")
# 训练开始
for epoch in range(NUM_EPOCHS):
    for step, data in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"), 0):
        model.train()
        target_ir = data[0].to(device)
        input_ir = data[1].to(device)
        target_rgb = data[2].to(device)
        input_rgb = data[3].to(device)
        target_seg = data[4].to(device)
        infrared_prompt = data[5]
        visible_prompt = data[6]

        input_ycbcr = rgb_to_ycbcr(input_rgb)
        target = Y_Upper(target_ir[:, :1, :, :], input_ycbcr[:, :1, :, :], 1.7).to(device)
        optimizer.zero_grad()  # 清空梯度
        # 返回重建的ir、重建的vir、交叉重建的ir、交叉重建的vir、输出的语义分割图、（重建的融合图像）
        res_ir, res_y, res_rec_ir, res_rec_y, res_seg, _ = model(input_ir[:, :1, :, :], input_ycbcr[:, :1, :, :],infrared_prompt,visible_prompt)
        # torch.cat((A,B,C),dim=1)在第二个维度上将A，B，C拼接
        res_ycbcr = torch.cat((torch.clamp(res_y, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]), dim=1)
        res_rgb = kornia.color.ycbcr_to_rgb(res_ycbcr)
        res_rec_ycbcr = torch.cat((torch.clamp(res_rec_y, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
                                  dim=1)
        res_rec_rgb = kornia.color.ycbcr_to_rgb(res_rec_ycbcr)

        loss_ir = l1_criterion(res_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_ir)
        loss_rgb = l1_criterion(res_rgb, target_rgb) + th_SSIM_LOSS(target, res_y)
        loss_rec_ir = l1_criterion(res_rec_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_rec_ir)
        loss_rec_rgb = l1_criterion(res_rec_rgb, target_rgb) + th_SSIM_LOSS(target, res_rec_y)
        loss_seg = Dice_criterion(res_seg, target_seg) + CE_criterion(res_seg, target_seg)  # to(torch.int64)

        loss = loss_ir + loss_rgb + loss_rec_ir + loss_rec_rgb + SEG_LOSS_WEIGHT * loss_seg

        loss.backward()

        grad = []
        for param in model.parameters():
            if param.grad is not None:
                grad.append(param.grad.norm().item())

        gr = torch.tensor(grad).mean().item()
        # 更新模型参数
        optimizer.step()

        model.eval()
        # agent.eval()
        # Segmodel.eval()

    if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
        EN_list, SD_list, SF_list, AG_list = [], [], [], []
        SCD_list, VIFF_list = [], []
        with torch.no_grad():
            for ii, data_test in enumerate(tqdm(test_loader, desc=f"Val   [{epoch + 1}/{NUM_EPOCHS}]"), 0):
                torch.cuda.ipc_collect()
                inp_ir = data_test[0].to(device)
                inp_rgb = data_test[1].to(device)
                tar_seg = data_test[2].to(device)
                inp_ir_prompt = data_test[3]
                inp_rgb_prompt = data_test[4]
                filenames = data_test[5]

                inp_ycbcr = rgb_to_ycbcr(inp_rgb)

                _, _, _, _, res_seg, fus = model(inp_ir[:, :1, :, :], inp_ycbcr[:, :1, :, :],inp_ir_prompt,inp_rgb_prompt)

                fus = torch.clamp(fus, 0, 1)
                fus_ycbcr = torch.cat((fus, inp_ycbcr[:, 1:2, :, :], inp_ycbcr[:, 2:, :, :]), dim=1)
                fus_rgb = kornia.color.ycbcr_to_rgb(fus_ycbcr)

                ones = torch.ones_like(fus_rgb)
                zeros = torch.zeros_like(fus_rgb)
                fus_rgb = torch.where(fus_rgb > ones, ones, fus_rgb)
                fus_rgb = torch.where(fus_rgb < zeros, zeros, fus_rgb)  # 将异常值截断

                fus_rgb = fus_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()  # BCHW -> BHWC  因为numpy只能在CPU上调用，所以调用.cpu
                for i in range(fus_rgb.shape[0]):   # 按每张图单独归一化
                    fus_rgb[i] = (fus_rgb[i] - np.min(fus_rgb[i])) / (np.max(fus_rgb[i]) - np.min(fus_rgb[i]) + 1e-8)
                fus_rgb = np.uint8(255.0 * fus_rgb)

                for batch in range(len(fus)):
                    fus_img = fus_rgb[batch]
                    fusion_path = os.path.join(result_dir, filenames[batch] + '.png')
                    # 保存融合图像
                    save_img(fusion_path, fus_img)

                    # 读取IR、RGB图像
                    ir_path = os.path.join(img_dir_test, 'ir', filenames[batch] + '.png')
                    rgb_path = os.path.join(img_dir_test, 'rgb', filenames[batch] + '.png')

                    EN, SD, SF, AG, SCD, VIFF = evaluation_one(ir_path, rgb_path, fusion_path)
                    EN_list.append(EN)
                    SD_list.append(SD)
                    SF_list.append(SF)
                    AG_list.append(AG)
                    SCD_list.append(SCD)
                    VIFF_list.append(VIFF)
        # print(f"=== Average Fusion Metrics on {data} ===")
        mean_EN = np.mean(EN_list)
        mean_SD = np.mean(SD_list)
        mean_SF = np.mean(SF_list)
        mean_AG = np.mean(AG_list)
        mean_SCD = np.mean(SCD_list)
        mean_VIFF = np.mean(VIFF_list)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"EN:   {mean_EN:.4f}")
        print(f"SD:   {mean_SD:.4f}")
        print(f"SF:   {mean_SF:.4f}")
        print(f"AG:   {mean_AG:.4f}")
        print(f"SCD:  {mean_SCD:.4f}")
        print(f"VIFF: {mean_VIFF:.4f}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
                f"EN: {mean_EN:.4f}, SD: {mean_SD:.4f}, SF: {mean_SF:.4f}, "
                f"AG: {mean_AG:.4f}, SCD: {mean_SCD:.4f}, VIFF: {mean_VIFF:.4f}\n"
            )
        if mean_SCD > best_SCD:
            best_SCD = mean_SCD
            best_EN = mean_EN
            best_SD = mean_SD
            best_SF = mean_SF
            best_AG = mean_AG
            best_VIFF = mean_VIFF
            best_epoch = epoch + 1

            torch.save(model.state_dict(), best_model_path)

            with open(best_record_file, 'w', encoding='utf-8') as f:
                f.write(f"Best Epoch: {best_epoch}\n")
                f.write(f"EN:   {best_EN:.4f}\n")
                f.write(f"SD:   {best_SD:.4f}\n")
                f.write(f"SF:   {best_SF:.4f}\n")
                f.write(f"AG:   {best_AG:.4f}\n")
                f.write(f"SCD:  {best_SCD:.4f}\n")
                f.write(f"VIFF: {best_VIFF:.4f}\n")

            print(f"发现更优模型，已保存！Best Epoch: {best_epoch}, Best SCD: {best_SCD:.4f}")
    scheduler.step()
