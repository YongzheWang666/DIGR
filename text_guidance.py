import torch
import torch.nn as nn
#import open_clip
from transformers import CLIPTokenizer,CLIPTextModel

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="./clip-vit-base-patch32", device="cuda",
                 freeze=True, max_length=77):
        super(CLIPTextEncoder,self).__init__()
        self.device = device
        self.max_length = max_length

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name,local_files_only=True)
        self.text_model = CLIPTextModel.from_pretrained(model_name,
                                                        local_files_only=True).to(device)

        if freeze:
            for p in self.text_model.parameters():
                p.requires_grad = False
        self.text_model.eval()

    @torch.no_grad()
    def forward(self,texts):
        inputs = self.tokenizer(texts,padding=True,truncation=True,
                                max_length=self.max_length,return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        outputs = self.text_model(**inputs)  # [B,D]

        text_feat = outputs.pooler_output #[B,512]
        return text_feat

"""
class CLIPTextEncoder(nn.Module):
    def __init__(self, ckpt_path="./RemoteCLIP-ViT-B-32.pt", device="cuda",
                 freeze=True):
        super(CLIPTextEncoder, self).__init__()
        self.device = device
        self.freeze = freeze

        # 1) 创建与权重匹配的模型结构
        # 你的文件名是 ViT-B-32，所以这里用 ViT-B-32
        self.model = open_clip.create_model(
            model_name="ViT-B-32",
            pretrained=None
        )

        # 2) tokenizer
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # 3) 加载 .pt 权重
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt,dict) and "state_dict" in ckpt else ckpt
        
        # 如果是 DataParallel / DDP 保存的，去掉 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v

        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print("RemoteCLIP load msg:", msg)

        self.model = self.model.to(device)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, texts):
        # open_clip 的 tokenizer 直接返回 token ids
        tokens = self.tokenizer(texts).to(self.device)

        if not self.freeze:
            with torch.no_grad():
                text_feat = self.model.encode_text(tokens)   # [B, 512]
        else:
            text_feat = self.model.encode_text(tokens)       # [B, 512]

        return text_feat
"""

class TextGuidanceAdapter(nn.Module):
    """
    将 CLIP 文本全局向量投影到各个 stage 对应的通道维度
    假设 text_feat 已经是 [B, text_dim]
    """
    def __init__(self, text_dim=512, stage_dims=(32, 64, 128),hidden_ratio=2):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, dim*hidden_ratio),
                nn.GELU(),
                nn.Linear(dim*hidden_ratio,dim),
                nn.LayerNorm(dim)
            )
            for dim in stage_dims
        ])

    def forward(self, text_feat, stage_idx):
        """
        text_feat: [B, text_dim]
        stage_idx: 0 / 1 / 2
        return:    [B, stage_dim]
        """
        return self.proj_layers[stage_idx](text_feat)