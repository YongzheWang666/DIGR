import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_utils import (
    patchify,
    remove_padding,
    tokens_to_map,
    get_candidate_indices,
    cosine_topk_graph,
    cross_modal_aggregate,
)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GraphInteractionBlock(nn.Module):
    """
    单个图交互子模块
    """

    def __init__(self, dim, patch_size, window_size, k, dilation):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.k = k
        self.dilation = dilation

        patch_dim = dim * patch_size * patch_size

        self.vis_token_proj = nn.Linear(patch_dim, dim)
        self.ir_token_proj = nn.Linear(patch_dim, dim)

        self.ln_msg_vis = nn.LayerNorm(dim)
        self.ln_msg_ir = nn.LayerNorm(dim)

        self.ln_ffn_vis = nn.LayerNorm(dim)
        self.ln_ffn_ir = nn.LayerNorm(dim)

        self.mlp_vis = MLP(dim)
        self.mlp_ir = MLP(dim)

        self.out_conv_vis = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.out_conv_ir = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        self.candidate_idx = None

    def forward(self, f_vis, f_ir):
        """
        f_vis, f_ir: [B, C, H, W]
        """
        B, C, H, W = f_vis.shape
        assert C == self.dim

        # 1) patchify ->[B,N,C*patch_size*patch_size]
        vis_patches, Hp, Wp, vis_pad_info, H_pad, W_pad = patchify(f_vis, self.patch_size)
        ir_patches, Hp2, Wp2, ir_pad_info, _, _ = patchify(f_ir, self.patch_size)

        assert Hp == Hp2 and Wp == Wp2, "VIS and IR patch grids must match"

        # 2) token projection
        x_vis = self.vis_token_proj(vis_patches)  # [B, N, C]
        x_ir = self.ir_token_proj(ir_patches)  # [B, N, C]

        # 3) build graph in each modality
        device = f_vis.device
        if self.candidate_idx is None or self.candidate_idx.device != device:
            self.candidate_idx = get_candidate_indices(
                Hp=Hp,
                Wp=Wp,
                window_size=self.window_size,
                dilation=self.dilation,
                device=device)  # [N,M]

        candidate_idx = self.candidate_idx  # [N, M]

        nbr_idx_vis, nbr_w_vis = cosine_topk_graph(x_vis, candidate_idx, self.k)
        nbr_idx_ir, nbr_w_ir = cosine_topk_graph(x_ir, candidate_idx, self.k)

        # 4) cross-modal aggregation
        m_vis = cross_modal_aggregate(x_ir, nbr_idx_ir, nbr_w_ir)  # [B, N, C]
        m_ir = cross_modal_aggregate(x_vis, nbr_idx_vis, nbr_w_vis)  # [B, N, C]

        # 5) node update
        m_vis = self.ln_msg_vis(m_vis)
        m_ir = self.ln_msg_ir(m_ir)

        z_vis = x_vis + m_vis
        z_ir = x_ir + m_ir

        # 第二个残差加原始 x，不是 z
        y_vis = x_vis + self.ln_ffn_vis(self.mlp_vis(z_vis))
        y_ir = x_ir + self.ln_ffn_ir(self.mlp_ir(z_ir))

        # 6) token map
        vis_map = tokens_to_map(y_vis, Hp, Wp)  # [B, N, C] -> [B, C, Hp, Wp]
        ir_map = tokens_to_map(y_ir, Hp, Wp)  # [B, N, C] -> [B, C, Hp, Wp]

        # 7) 上采样回原分辨率，再conv
        vis_map = F.interpolate(vis_map, size=(H_pad, W_pad), mode='bilinear', align_corners=False)
        ir_map = F.interpolate(ir_map, size=(H_pad, W_pad), mode='bilinear', align_corners=False)

        vis_out = self.out_conv_vis(vis_map)
        ir_out = self.out_conv_ir(ir_map)

        # 8) remove pad
        vis_out = remove_padding(vis_out, vis_pad_info)
        ir_out = remove_padding(ir_out, ir_pad_info)

        return vis_out, ir_out  # ->[B,C,H,W]

"""
class GraphRefinementStack(nn.Module):

    def __init__(self, dim, patch_size, window_size, k_list, d_list):
        super().__init__()
        assert len(window_size) == len(k_list) == len(d_list)

        self.blocks = nn.ModuleList([
            GraphInteractionBlock(
                dim=dim,
                patch_size=patch_size,
                window_size=window_size[i],
                k=k_list[i],
                dilation=d_list[i]
            )
            for i in range(len(k_list))
        ])

    def forward(self, f_vis, f_ir):
        x_vis, x_ir = f_vis, f_ir
        for blk in self.blocks:
            x_vis, x_ir = blk(x_vis, x_ir)
        return x_vis, x_ir
"""
class GraphRefinementStack(nn.Module):
    """
    每个 stage 内部串联 3 个独立 graph interaction sub-block
    """

    def __init__(self, dim, patch_size, window_size: list, k_list=(4, 6, 8), d_list=(1, 2, 3)):
        super().__init__()
        assert len(k_list) == 3 and len(d_list) == 3

        self.blocks = nn.ModuleList([
            GraphInteractionBlock(dim, patch_size, window_size[0], k=k_list[0], dilation=d_list[0]),
            GraphInteractionBlock(dim, patch_size, window_size[1], k=k_list[1], dilation=d_list[1]),
            GraphInteractionBlock(dim, patch_size, window_size[2], k=k_list[2], dilation=d_list[2]),
        ])

    def forward(self, f_vis, f_ir):
        x_vis, x_ir = f_vis, f_ir
        for blk in self.blocks:
            x_vis, x_ir = blk(x_vis, x_ir)
        return x_vis, x_ir
