import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import kornia
import numpy as np

from data_RGB_potsdam import get_training_data, get_test_data
from utils.MEF_SSIM_loss import th_SSIM_LOSS, Y_Upper
from utils.dice import DiceLoss
from utils.seg_metrics import SegmentationMetric
from model import MMFNet as Net

warnings.filterwarnings("ignore", category=UserWarning)


def rgb_to_ycbcr(img):
    return kornia.color.rgb_to_ycbcr(img)


def print_batch_info(name, batch):
    print(f"\n========== {name} ==========")
    print("type(batch):", type(batch))
    print("len(batch):", len(batch))

    for i, x in enumerate(batch):
        if isinstance(x, torch.Tensor):
            print(f"[{i}] Tensor shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
        else:
            if isinstance(x, (list, tuple)) and len(x) > 0:
                print(f"[{i}] {type(x)} len={len(x)} sample0={x[0]}")
            else:
                print(f"[{i}] {type(x)} value={x}")


def main():
    # =========================
    # 基本配置
    # =========================
    BATCH_SIZE = 2
    LR_INITIAL = 2e-5
    data = "potsdam"

    train_dir = os.path.join("./dataset/train/", data)
    img_dir_test = os.path.join("./dataset/test/", data)

    text_path_train_ir = "./dataset/train/potsdam/annotations/potsdam_ir_cl.csv"
    text_path_train_rgb = "./dataset/train/potsdam/annotations/potsdam_rgb_cl.csv"
    text_path_test_ir = "./dataset/test/potsdam/annotations/potsdam_ir_test_cl.csv"
    text_path_test_rgb = "./dataset/test/potsdam/annotations/potsdam_rgb_test_cl.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("device:", device)

    # =========================
    # 模型
    # =========================
    model = Net(6).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)

    l1_criterion = nn.L1Loss()
    CE_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
    Dice_criterion = DiceLoss(smooth=0.05, ignore_index=255)

    # =========================
    # 数据集
    # =========================
    train_dataset = get_training_data(train_dir, text_path_train_rgb, text_path_train_ir)
    test_dataset = get_test_data(img_dir_test, text_path_test_rgb, text_path_test_ir)

    print("\n训练集长度:", len(train_dataset))
    print("测试集长度:", len(test_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      # debug时先设0，最稳
        drop_last=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,      # debug时先设0，最稳
        drop_last=False,
        pin_memory=True
    )

    # =========================================================
    # 1) 先测一个 train batch
    # =========================================================
    print("\n\n########## DEBUG TRAIN ONE BATCH ##########")
    data_train = next(iter(train_loader))
    print_batch_info("train batch raw", data_train)

    target_ir = data_train[0].to(device)
    input_ir = data_train[1].to(device)
    target_rgb = data_train[2].to(device)
    input_rgb = data_train[3].to(device)
    target_seg = data_train[4].to(device)
    infrared_prompt = data_train[5]
    visible_prompt = data_train[6]
    filenames_train = data_train[7]

    print("\n[Train] filenames:", filenames_train)
    print("[Train] infrared_prompt sample:", infrared_prompt[0] if len(infrared_prompt) > 0 else "EMPTY")
    print("[Train] visible_prompt sample:", visible_prompt[0] if len(visible_prompt) > 0 else "EMPTY")

    print("\n[Train] target_ir:", target_ir.shape, target_ir.dtype, target_ir.device)
    print("[Train] input_ir:", input_ir.shape, input_ir.dtype, input_ir.device)
    print("[Train] target_rgb:", target_rgb.shape, target_rgb.dtype, target_rgb.device)
    print("[Train] input_rgb:", input_rgb.shape, input_rgb.dtype, input_rgb.device)
    print("[Train] target_seg:", target_seg.shape, target_seg.dtype, target_seg.device)
    print("[Train] target_seg unique:", torch.unique(target_seg).detach().cpu().tolist())

    input_ycbcr = rgb_to_ycbcr(input_rgb)
    target = Y_Upper(target_ir[:, :1, :, :], input_ycbcr[:, :1, :, :], 1.7).to(device)

    optimizer.zero_grad()

    print("\n[Train] start forward...")
    res_ir, res_y, res_rec_ir, res_rec_y, res_seg, fus = model(
        input_ir[:, :1, :, :],
        input_ycbcr[:, :1, :, :],
        infrared_prompt,
        visible_prompt
    )
    print("[Train] forward done.")

    print("[Train] res_ir:", res_ir.shape)
    print("[Train] res_y:", res_y.shape)
    print("[Train] res_rec_ir:", res_rec_ir.shape)
    print("[Train] res_rec_y:", res_rec_y.shape)
    print("[Train] res_seg:", res_seg.shape)
    print("[Train] fus:", fus.shape if isinstance(fus, torch.Tensor) else type(fus))

    res_ycbcr = torch.cat(
        (torch.clamp(res_y, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
        dim=1
    )
    res_rgb = kornia.color.ycbcr_to_rgb(res_ycbcr)

    res_rec_ycbcr = torch.cat(
        (torch.clamp(res_rec_y, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
        dim=1
    )
    res_rec_rgb = kornia.color.ycbcr_to_rgb(res_rec_ycbcr)

    print("\n[Train] start compute loss...")
    loss_ir = l1_criterion(res_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_ir)
    loss_rgb = l1_criterion(res_rgb, target_rgb) + th_SSIM_LOSS(target, res_y)
    loss_rec_ir = l1_criterion(res_rec_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_rec_ir)
    loss_rec_rgb = l1_criterion(res_rec_rgb, target_rgb) + th_SSIM_LOSS(target, res_rec_y)
    loss_seg = Dice_criterion(res_seg, target_seg) + CE_criterion(res_seg, target_seg)
    loss = loss_ir + loss_rgb + loss_rec_ir + loss_rec_rgb + loss_seg

    print("[Train] loss_ir =", float(loss_ir.detach().cpu()))
    print("[Train] loss_rgb =", float(loss_rgb.detach().cpu()))
    print("[Train] loss_rec_ir =", float(loss_rec_ir.detach().cpu()))
    print("[Train] loss_rec_rgb =", float(loss_rec_rgb.detach().cpu()))
    print("[Train] loss_seg =", float(loss_seg.detach().cpu()))
    print("[Train] total loss =", float(loss.detach().cpu()))

    print("\n[Train] start backward...")
    loss.backward()
    optimizer.step()
    print("[Train] backward + optimizer.step done.")

    # =========================================================
    # 2) 再测一个 test batch
    # =========================================================
    print("\n\n########## DEBUG TEST ONE BATCH ##########")
    model.eval()
    data_test = next(iter(test_loader))
    print_batch_info("test batch raw", data_test)

    with torch.no_grad():
        inp_ir = data_test[0].to(device)
        inp_rgb = data_test[1].to(device)
        tar_seg = data_test[2].to(device)
        inp_ir_prompt = data_test[3]
        inp_rgb_prompt = data_test[4]
        filenames_test = data_test[5]

        print("\n[Test] filenames:", filenames_test)
        print("[Test] inp_ir_prompt sample:", inp_ir_prompt[0] if len(inp_ir_prompt) > 0 else "EMPTY")
        print("[Test] inp_rgb_prompt sample:", inp_rgb_prompt[0] if len(inp_rgb_prompt) > 0 else "EMPTY")

        print("\n[Test] inp_ir:", inp_ir.shape, inp_ir.dtype, inp_ir.device)
        print("[Test] inp_rgb:", inp_rgb.shape, inp_rgb.dtype, inp_rgb.device)
        print("[Test] tar_seg:", tar_seg.shape, tar_seg.dtype, tar_seg.device)
        print("[Test] tar_seg unique:", torch.unique(tar_seg).detach().cpu().tolist())

        inp_ycbcr = rgb_to_ycbcr(inp_rgb)

        print("\n[Test] start forward...")
        _, _, _, _, res_seg, fus = model(
            inp_ir[:, :1, :, :],
            inp_ycbcr[:, :1, :, :],
            inp_ir_prompt,
            inp_rgb_prompt
        )
        print("[Test] forward done.")

        print("[Test] res_seg:", res_seg.shape)
        print("[Test] fus:", fus.shape if isinstance(fus, torch.Tensor) else type(fus))

        seg_metric = SegmentationMetric(num_classes=6, ignore_index=255)
        seg_metric.update(res_seg, tar_seg)
        seg_scores = seg_metric.get_scores_100()

        print("\n[Test] seg_scores:", seg_scores)

        fus = torch.clamp(fus, 0, 1)
        fus_ycbcr = torch.cat((fus, inp_ycbcr[:, 1:2, :, :], inp_ycbcr[:, 2:, :, :]), dim=1)
        fus_rgb = kornia.color.ycbcr_to_rgb(fus_ycbcr)

        print("[Test] fus_rgb:", fus_rgb.shape, fus_rgb.dtype, fus_rgb.device)

    print("\n========== DEBUG SUCCESS: train/test one batch both passed ==========")


if __name__ == "__main__":
    main()