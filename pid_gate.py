import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorPIDGate(nn.Module):
    """
    面向你的融合网络设计的向量 PID-Gate

    输入:
        err:        [B, C]   当前 stage 的误差向量 e_i = text_proj - pooled_feat
        prev_err:   [B, C]   上一 stage 的误差
        integ:      [B, C]   上一 stage 的积分状态
        deriv_ema:  [B, C]   上一 stage 的微分平滑状态

    输出:
        delta:      [B, C]   当前 stage 的通道补偿向量
        state: dict，包含新的 prev_err / integ / deriv_ema
    """
    def __init__(
        self,
        dim: int,
        integral_limit: float = 5.0, # 积分项裁剪上限，防止积分爆炸
        use_layernorm: bool = True,  # 是否对输入误差和输出做LayerNorm
        hidden_ratio: int = 2,       # gate隐藏层宽度倍数
    ):
        super().__init__()
        self.dim = dim
        self.integral_limit = integral_limit

        # 用 softplus 保证增益为正，更稳定
        self.kp_raw = nn.Parameter(torch.zeros(dim))
        self.ki_raw = nn.Parameter(torch.zeros(dim))
        self.kd_raw = nn.Parameter(torch.zeros(dim))

        self.deriv_beta_raw = nn.Parameter(torch.empty(dim))

        self.norm_err = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.norm_out = nn.LayerNorm(dim) if use_layernorm else nn.Identity()

        hidden_dim = dim * hidden_ratio

        # Gate：决定 PID 输出注入多少
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid()
        )

        # 输出投影：让 PID 输出更贴合当前通道空间
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self._init_parameters()

    def _init_parameters(self):
        # 给一个比较稳的初始值：
        # P 稍强，I 较弱，D 更弱
        nn.init.constant_(self.kp_raw, 0.5)
        nn.init.constant_(self.ki_raw, -1.5)
        nn.init.constant_(self.kd_raw, -2.0)

        nn.init.constant_(self.deriv_beta_raw, 1.3863)

        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.out_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _positive_gain(self, x):
        return F.softplus(x)

    def init_state(self, err: torch.Tensor):
        """
        根据当前 err 初始化状态
        """
        zeros = torch.zeros_like(err)
        return {
            "prev_err": zeros,
            "integ": zeros,
            "deriv_ema": zeros,
        }

    def forward(self, err, prev_err=None, integ=None, deriv_ema=None):
        """
        err: [B, C]
        """
        if prev_err is None or integ is None or deriv_ema is None:
            state = self.init_state(err)
            prev_err = state["prev_err"]
            integ = state["integ"]
            deriv_ema = state["deriv_ema"]

        # ---- PID 三项 ----
        kp = self._positive_gain(self.kp_raw).view(1, -1)   # [1, C]
        ki = self._positive_gain(self.ki_raw).view(1, -1)
        kd = self._positive_gain(self.kd_raw).view(1, -1)

        # P
        p_term = kp * err

        # I: 累积 + anti-windup（防积分饱和）
        integ_new = integ + err
        integ_new = torch.clamp(
            integ_new,
            min=-self.integral_limit,
            max=self.integral_limit
        )
        i_term = ki * integ_new

        # D: 误差差分 + EMA(指数滑动平均) 平滑
        deriv_beta = torch.sigmoid(self.deriv_beta_raw).view(1, -1)

        deriv = err - prev_err
        deriv_ema_new = deriv_beta * deriv_ema + (1.0 - deriv_beta) * deriv
        d_term = kd * deriv_ema_new

        # PID 合成
        #pid_out = p_term + i_term + d_term
        pid_out = p_term + d_term

        # Gate
        gate = self.gate(self.norm_err(err))   # [B, C]

        # 最终补偿
        delta = self.out_proj(self.norm_out(gate * pid_out))  # [B, C]

        new_state = {
            "prev_err": err.detach(),
            "integ": integ_new.detach(),
            "deriv_ema": deriv_ema_new.detach(),
        }

        return delta, new_state



class pid(nn.Module):
    def __init__(self,dim,prev_dim=None,pid_cls=VectorPIDGate):
        super().__init__()
        self.dim = dim
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.pid_vis = pid_cls(dim)
        self.pid_ir = pid_cls(dim)

        # 如果上一stage的通道数和当前不同，则做状态对齐
        if prev_dim is not None and prev_dim != dim:
            self.align_prev_err_vis = nn.Linear(prev_dim, dim)
            self.align_integ_vis = nn.Linear(prev_dim, dim)
            self.align_deriv_vis = nn.Linear(prev_dim, dim)

            self.align_prev_err_ir = nn.Linear(prev_dim, dim)
            self.align_integ_ir = nn.Linear(prev_dim, dim)
            self.align_deriv_ir = nn.Linear(prev_dim, dim)
        else:
            self.align_prev_err_vis = nn.Identity()
            self.align_integ_vis = nn.Identity()
            self.align_deriv_vis = nn.Identity()

            self.align_prev_err_ir = nn.Identity()
            self.align_integ_ir = nn.Identity()
            self.align_deriv_ir = nn.Identity()

    def _align_state(self, state, branch="vis"):
        if state is None:
            return None

        if branch == "vis":
            return {
                "prev_err": self.align_prev_err_vis(state["prev_err"]),
                "integ": self.align_integ_vis(state["integ"]),
                "deriv_ema": self.align_deriv_vis(state["deriv_ema"]),
            }
        else:
            return {
                "prev_err": self.align_prev_err_ir(state["prev_err"]),
                "integ": self.align_integ_ir(state["integ"]),
                "deriv_ema": self.align_deriv_ir(state["deriv_ema"]),
            }

    def forward(self, f_vis, f_ir, text_feat_vis,text_feat_ir, pid_state_vis=None, pid_state_ir=None):

        # 1) GAP
        vis_pool = self.gap(f_vis).flatten(1)   # [B, C]
        ir_pool = self.gap(f_ir).flatten(1)

        # 2) 误差
        e_vis = text_feat_vis - vis_pool
        e_ir =  text_feat_ir - ir_pool

        #3) 通道数校准
        pid_state_vis = self._align_state(pid_state_vis, branch="vis")
        pid_state_ir = self._align_state(pid_state_ir, branch="ir")

        # 3) PID
        if pid_state_vis is None:
            pid_state_vis = self.pid_vis.init_state(e_vis)
        if pid_state_ir is None:
            pid_state_ir = self.pid_ir.init_state(e_ir)

        delta_vis, new_state_vis = self.pid_vis(
            e_vis,
            prev_err=pid_state_vis["prev_err"],
            integ=pid_state_vis["integ"],
            deriv_ema=pid_state_vis["deriv_ema"],
        )
        delta_ir, new_state_ir = self.pid_ir(
            e_ir,
            prev_err=pid_state_ir["prev_err"],
            integ=pid_state_ir["integ"],
            deriv_ema=pid_state_ir["deriv_ema"],
        )

        # 4) broadcast 加回 feature map
        delta_vis = delta_vis.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        delta_ir = delta_ir.unsqueeze(-1).unsqueeze(-1)

        out_vis = f_vis + delta_vis
        out_ir = f_ir + delta_ir

        return out_vis, out_ir, new_state_vis, new_state_ir

