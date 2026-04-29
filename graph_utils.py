import math
import torch
import torch.nn.functional as F


def pad_to_multiple(x, patch_size):
    """
    把输入 pad 到能被 patch_size 整除
    x: [B, C, H, W]
    """
    B, C, H, W = x.shape
    # 计算一下高和宽方向要补多少
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)

    # pad format: (left, right, top, bottom)，做反射填充
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, (0, pad_w, 0, pad_h)


def remove_padding(x, pad_info):
    """
    移除 pad
    x: [B, C, H_pad, W_pad]
    """
    left, right, top, bottom = pad_info
    H, W = x.shape[-2:]
    return x[:, :, top:H-bottom if bottom > 0 else H, left:W-right if right > 0 else W]


def patchify(x, patch_size):
    """
    x: [B, C, H, W]
    return:
        patches: [B, N, C*p*p]
        Hp, Wp: patch网格大小
        pad_info: padding信息
        H_pad, W_pad: pad后的空间大小
    """
    x, pad_info = pad_to_multiple(x, patch_size)
    B, C, H_pad, W_pad = x.shape

    unfold = F.unfold(x, kernel_size=patch_size, stride=patch_size)  # [B, C*p*p, N]
    patches = unfold.transpose(1, 2).contiguous()  # [B, N, C*p*p]

    Hp = H_pad // patch_size
    Wp = W_pad // patch_size
    return patches, Hp, Wp, pad_info, H_pad, W_pad


def tokens_to_map(tokens, Hp, Wp):
    """
    把图结构还原回特征图维度
    tokens: [B, N, D]
    return: [B, D, Hp, Wp]
    """
    B, N, D = tokens.shape
    assert N == Hp * Wp, "N must equal Hp * Wp"
    return tokens.transpose(1, 2).contiguous().view(B, D, Hp, Wp)# 只要节点顺序不打乱，view就可以还原回去


def build_grid_coords(Hp, Wp, device):
    """
    生成网格坐标表
    return:
        coords: [N, 2], 每个节点的(row, col)
    """
    rows = torch.arange(Hp, device=device)
    cols = torch.arange(Wp, device=device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack([grid_r.reshape(-1), grid_c.reshape(-1)], dim=1)
    return coords  # [N, 2]


def get_candidate_indices(Hp, Wp, window_size, dilation, device):
    """
    为每个节点生成局部候选邻域索引
    即对每一个节点，找出它周围一个局部窗口里的候选邻居节点，并把这些邻居的编号记下来
    dilation：膨胀系数，这个参数表示邻域采样时不是每次走1格，而是每次走dilation格
    return:
        candidate_idx: [N, M]
    """
    coords = build_grid_coords(Hp, Wp, device=device)  # [N, 2]
    N = coords.shape[0]
    radius = window_size // 2 # 计算窗口半径

    candidate_list = []       # 用来保存每个节点各自的候选邻居索引列表
    for i in range(N):
        r, c = coords[i].tolist()
        cur_candidates = []   # 当前节点的候选列表
        for dr in range(-radius, radius + 1): # 遍历窗口内行偏移
            for dc in range(-radius, radius + 1): # 遍历窗口内列偏移
                rr = r + dr * dilation
                cc = c + dc * dilation
                if 0 <= rr < Hp and 0 <= cc < Wp:
                    idx = rr * Wp + cc
                    if idx != i:  # 一般不把自己当邻居
                        cur_candidates.append(idx)
        # 候选列表为空时，加自己
        if len(cur_candidates) == 0:
            cur_candidates.append(i)

        candidate_list.append(torch.tensor(cur_candidates, device=device, dtype=torch.long))
    # 找一下最多的邻居节点数，为了将每一个节点的候选列表都补齐
    max_len = max(x.numel() for x in candidate_list)
    padded = []
    for i, x in enumerate(candidate_list):
        if x.numel() < max_len:
            pad = x.new_full((max_len - x.numel(),), i)
            x = torch.cat([x, pad], dim=0)
        padded.append(x)
    # 转换成shape为[N, M]的二维张量
    candidate_idx = torch.stack(padded, dim=0)
    return candidate_idx


def cosine_topk_graph(tokens, candidate_idx, k):
    """
    在候选邻域里做 cosine 相似度 + topk + softmax

    tokens:        [B, N, D]
    candidate_idx: [N, M]
    return:
        索引topk_idx:   [B, N, k]
        权重topk_w:     [B, N, k]
    """
    B, N, D = tokens.shape
    _, M = candidate_idx.shape

    # [1, N, M] -> [B, N, M]
    candidate_idx_b = candidate_idx.unsqueeze(0).expand(B, -1, -1)

    # gather candidate tokens收集候选邻居节点特征
    # tokens_expand[b,i]表示第b个样本，以第i个节点为中心时，所有N个候选被查询节点的特征表
    tokens_expand = tokens.unsqueeze(1).expand(B, N, N, D)
    candidate_tokens = torch.gather(
        # 相当于做了candidate_tokens[b, i, m, d] = tokens_expand[b, i, candidate_idx_b[b, i, m], d]
        tokens_expand,
        2,
        candidate_idx_b.unsqueeze(-1).expand(B, N, M, D)
    )  # [B, N, M, D]

    center_tokens = tokens.unsqueeze(2)  # [B, N, 1, D]，便于后面和candidate_tokens作余弦相似性比较
    # 比较的是center_tokens[b, i, 0, :] 和 candidate_tokens[b, i, m, :]的相似度
    sim = F.cosine_similarity(center_tokens, candidate_tokens, dim=-1)  # [B, N, M]

    k = min(k, M)
    topk_sim, topk_pos = torch.topk(sim, k=k, dim=-1)  # [B, N, k]，注意这个topk_pos不是真实位置，只是在邻居节点列表中的索引

    topk_idx = torch.gather(candidate_idx_b, 2, topk_pos)  # [B, N, k]
    topk_w = torch.softmax(topk_sim, dim=-1)  # [B, N, k]

    return topk_idx, topk_w


def gather_neighbors(tokens, nbr_idx):
    """
    根据邻居索引提取邻居节点的token特征
    tokens:  [B, N, D]
    nbr_idx: [B, N, k]
    return:
        nbr_tokens: [B, N, k, D]
    """
    B, N, D = tokens.shape
    _, _, k = nbr_idx.shape

    tokens_expand = tokens.unsqueeze(1).expand(B, N, N, D)
    nbr_tokens = torch.gather(
        tokens_expand,
        2,
        nbr_idx.unsqueeze(-1).expand(B, N, k, D)
    )
    return nbr_tokens


def cross_modal_aggregate(other_tokens, nbr_idx, nbr_w):
    """
    other_tokens: [B, N, D]
    nbr_idx:      [B, N, k]
    nbr_w:        [B, N, k]
    return:
        msg:       [B, N, D]
    """
    nbr_tokens = gather_neighbors(other_tokens, nbr_idx)      # [B, N, k, D]
    msg = (nbr_tokens * nbr_w.unsqueeze(-1)).sum(dim=2)       # [B, N, D]
    return msg