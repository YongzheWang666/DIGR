import numpy as np
import torch


class SegmentationMetric:
    """
    用于多分类语义分割评估，计算：
    1. mDice
    2. mAcc

    支持：
    - pred 为类别图 [B,H,W] / [H,W]
    - pred 为 logits/prob [B,C,H,W] / [C,H,W]
    - label 为类别图 [B,H,W] / [H,W]
    - ignore_index，例如 255
    """

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """
        清空累计统计量
        """
        self.total_tp = np.zeros(self.num_classes, dtype=np.float64)
        self.total_fp = np.zeros(self.num_classes, dtype=np.float64)
        self.total_fn = np.zeros(self.num_classes, dtype=np.float64)

    def _to_numpy(self, x):
        """
        torch.Tensor -> numpy
        numpy -> numpy
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _prepare_prediction(self, pred):
        """
        将预测结果统一转换为类别索引图

        支持输入：
        1. [B, C, H, W] -> argmax -> [B, H, W]
        2. [C, H, W]    -> argmax -> [H, W]
        3. [B, H, W]    -> 直接使用
        4. [H, W]       -> 直接使用
        """
        pred = self._to_numpy(pred)

        if pred.ndim == 4:
            # [B, C, H, W]
            pred = np.argmax(pred, axis=1)
        elif pred.ndim == 3:
            # 可能是 [C, H, W] 或 [B, H, W]
            # 这里通过和 num_classes 比较来判断
            if pred.shape[0] == self.num_classes:
                # [C, H, W]
                pred = np.argmax(pred, axis=0)
            else:
                # [B, H, W]
                pass
        elif pred.ndim == 2:
            # [H, W]
            pass
        else:
            raise ValueError(f"Unsupported pred shape: {pred.shape}")

        return pred

    def _prepare_label(self, label):
        """
        将标签统一为类别索引图

        支持：
        1. [B, H, W]
        2. [H, W]
        """
        label = self._to_numpy(label)

        if label.ndim not in [2, 3]:
            raise ValueError(f"Unsupported label shape: {label.shape}")

        return label

    def update(self, pred, label):
        """
        累计一个 batch 的 TP / FP / FN

        参数:
            pred:
                可以是：
                - logits/prob: [B,C,H,W] 或 [C,H,W]
                - 类别图:      [B,H,W]   或 [H,W]

            label:
                真实标签类别图：
                - [B,H,W]
                - [H,W]
        """
        pred = self._prepare_prediction(pred)
        label = self._prepare_label(label)

        # 统一 batch 维
        if pred.ndim == 2:
            pred = pred[None, ...]   # [1,H,W]
        if label.ndim == 2:
            label = label[None, ...] # [1,H,W]

        if pred.shape != label.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, label {label.shape}")

        # 有效区域掩码
        valid_mask = (label != self.ignore_index)

        for c in range(self.num_classes):
            pred_c = (pred == c) & valid_mask    # 返回的是布尔数组
            label_c = (label == c) & valid_mask

            tp = np.logical_and(pred_c, label_c).sum()
            fp = np.logical_and(pred_c, np.logical_not(label_c)).sum()
            fn = np.logical_and(np.logical_not(pred_c), label_c).sum()

            self.total_tp[c] += tp
            self.total_fp[c] += fp
            self.total_fn[c] += fn

    def get_scores(self):
        """
        返回每类和整体指标

        返回:
            dict:
            {
                "dice_per_class": ndarray [C],
                "acc_per_class": ndarray [C],
                "mDice": float,
                "mAcc": float
            }
        """
        dice_per_class = (2.0 * self.total_tp) / (
            2.0 * self.total_tp + self.total_fp + self.total_fn + 1e-8
        )

        acc_per_class = self.total_tp / (
            self.total_tp + self.total_fn + 1e-8
        )

        mDice = np.mean(dice_per_class)
        mAcc = np.mean(acc_per_class)

        return {
            "dice_per_class": dice_per_class,
            "acc_per_class": acc_per_class,
            "mDice": float(mDice),
            "mAcc": float(mAcc),
        }

    def get_scores_100(self):
        """
        返回百分制结果，便于论文表格直接写

        返回:
            {
                "dice_per_class": ndarray [C],   # 0~100
                "acc_per_class": ndarray [C],    # 0~100
                "mDice": float,                  # 0~100
                "mAcc": float                    # 0~100
            }
        """
        scores = self.get_scores()
        return {
            "dice_per_class": scores["dice_per_class"] * 100.0,
            "acc_per_class": scores["acc_per_class"] * 100.0,
            "mDice": scores["mDice"] * 100.0,
            "mAcc": scores["mAcc"] * 100.0,
        }