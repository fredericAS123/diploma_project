import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from dataclasses import dataclass

# --- 1. 定义 ModelArgs 配置类 ---
@dataclass
class ModelArgs:
    hidden_size: int = 512       # 隐藏层维度 (例如 4096)
    num_attention_heads: int = 8 # 注意力头数 (例如 32)
    max_seq_len: int = 1024      # 最大序列长度

# --- 2. RoPE（复数计算版）---
class RoPEAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.num_attention_heads
        self.head_dim = args.hidden_size // self.n_head
        
        self.wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wv = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        
        # 预计算频率表
        freqs_cls = self.precompute_freq_cls(self.head_dim, args.max_seq_len)
        self.register_buffer("freqs_cls", freqs_cls)
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        #  拆分多头 [B, L, H, D]
        xq = xq.view(batch_size, seq_len, self.n_head, self.head_dim).float()
        xk = xk.view(batch_size, seq_len, self.n_head, self.head_dim).float()
        xv = xv.view(batch_size, seq_len, self.n_head, self.head_dim).float()
        # 截取当前长度的频率
        freqs_cls = self.freqs_cls[:seq_len, :]
        # 应用 RoPE
        r_xq, r_xk = self.apply_rotatary_emb(xq, xk, freqs_cls) 
        #  转置 [B, H, L, D]
        r_xq = r_xq.transpose(1, 2)
        r_xk = r_xk.transpose(1, 2)
        xv = xv.transpose(1, 2) 
        # Attention 计算
        scores = torch.matmul(r_xq, r_xk.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        outputs = attn_weights @ xv
        # 还原形状 [B, L, Hidden]
        outputs = outputs.transpose(1, 2).flatten(2)
        return outputs

    @staticmethod
    def apply_rotatary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cls: torch.Tensor): 
        # 重塑为 [B, L, H, D/2, 2]
        xq_ = xq.view(*(xq.shape[:-1]), -1, 2)
        xk_ = xk.view(*(xk.shape[:-1]), -1, 2)
        
        # 转复数
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)
        
        # 广播频率 [1, L, 1, D/2]
        freqs = freqs_cls.view(1, freqs_cls.shape[0], 1, -1)
        
        # 旋转并展平回 [B, L, H, D]
        xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
        
        return xq_out, xk_out

    @staticmethod
    def precompute_freq_cls(dim: int, max_seq_len: int, theta: float = 10000.0):
        freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float()[:dim // 2] / dim))
        t = torch.arange(max_seq_len, device=freq.device)
        freqs = torch.outer(t, freq)
        freqs_cls = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cls

class LlamaRoPEAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.num_attention_heads
        self.head_dim = args.hidden_size // self.n_head
        self.wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wv = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        
        cos_cached, sin_cached = self.precompute_cos_sin(args.max_seq_len, self.head_dim)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_head, self.head_dim).float()
        xk = xk.view(batch_size, seq_len, self.n_head, self.head_dim).float()
        xv = xv.view(batch_size, seq_len, self.n_head, self.head_dim).float()

        cos = self.cos_cached[:seq_len].view(1, seq_len, 1, self.head_dim) # [L,D] -> [1,L,1,D]
        sin = self.sin_cached[:seq_len].view(1, seq_len, 1, self.head_dim) # [L,D] -> [1,L,1,D]

        r_xq = (xq * cos) + (self.rotate_half(xq) * sin) # [B,L,H,D]*[1,L,1,D] -> [B,L,H,D]
        r_xk = (xk * cos) + (self.rotate_half(xk) * sin)

        r_xq = r_xq.transpose(1, 2)
        r_xk = r_xk.transpose(1, 2)

        scores = r_xq @ r_xk.transpose(-1, -2) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        outputs = attn_weights @ xv.transpose(1, 2) # [B,H,L,D]

        outputs = outputs.transpose(1, 2).flatten(2) # [B,L,H,D] -> [B,L,H*D]
        return outputs
    @staticmethod
    def precompute_cos_sin(max_seq_len: int, head_dim: int, theta: float = 10000.0):
        freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float()[:head_dim // 2] / head_dim))
        t = torch.arange(max_seq_len, device=freq.device)
        freqs = torch.outer(t, freq)
        freqs_emd = torch.cat([freqs, freqs], dim=-1).to(device=freq.device)
        freqs_emd = freqs_emd.float() # 转为 float32 以提高精度

        cos_cached = torch.cos(freqs_emd)
        sin_cached = torch.sin(freqs_emd) # 转为 bfloat16 以匹配模型精度
        
        return cos_cached, sin_cached
    @staticmethod
    def rotate_half(x: torch.Tensor):
        x1 = x[... , : x.shape[-1] // 2]
        x2 = x[... , x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

# --- 3. 测试运行 ---
if __name__ == "__main__":
    # 配置参数
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ModelArgs(
        hidden_size=512,
        num_attention_heads=8,
        max_seq_len=20
    )
    
    # 实例化模型(复数运算)
    # model = RoPEAttention(args).to(device)
    model = LlamaRoPEAttention(args).to(device) # 实例化模型(Llama RoPE 版本)
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, args.hidden_size).to(device)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输入设备： {device}")
    # 验证形状是否一致
    assert output.shape == x.shape
    print("✅ 测试通过！代码完美运行。")