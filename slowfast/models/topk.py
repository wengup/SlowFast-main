import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from einops import rearrange
from math import sqrt
import time
from operator import mul
from functools import reduce
import math
from torch.nn.init import trunc_normal_

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = torch.mean(x[:,:, C//2:], dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
 

class PromptTransformer(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., num_tokens=10., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention=Attention, MLP=Mlp):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        if num_tokens == 0:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.fc_layer = nn.Linear(dim, dim)
        self.score_layer = nn.Linear(dim, 1)
        
        # prepend the prompt to the tokens
        self.prompt_dropout = nn.Dropout(drop)
        self.num_tokens = num_tokens
        if num_tokens is not None:
            self.prompt_embedding = nn.Parameter(torch.zeros(1, num_tokens, dim))
            # xavier_uniform initialization
            # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + dim))  # noqa
            # nn.init.uniform_(self.prompt_embedding.data, -val, val)
            trunc_normal_(self.prompt_embedding, std=0.02)
        else:
            self.prompt_embedding = nn.Identity()
        
    def incorporate_prompt(self, x):
        # taken from https://github.com/KMnP/vpt/blob/main/src/models/vit_prompt/vit.py
        # combine prompt embeddings with frames\patch embeddings
        B = x.shape[0]
        # all before image patches
        # x = self.embeddings(x)  # (batch_size, n_patches, hidden_dim)
        x_prompt = torch.cat((
                # x[:, :1, :], # cls_token
                self.prompt_dropout(self.prompt_embedding.expand(B, -1, -1)),
                x # x[:, 1:, :]
            ), dim=1)
        # (batch_size, n_prompt + n_patches, hidden_dim)
        return x_prompt
        
    def forward(self, x):
        # append the frame/patch prompt if necessary
        x = self.incorporate_prompt(x) if self.num_tokens is not None else x
        # pass all into a transformer layer with prompt
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.fc_layer(self.norm2(x)))
        s = self.num_tokens if self.num_tokens is not None else 0
        scores = self.score_layer(x[:,s:,:]) # get the token scores
        return scores


def HardTopK(k, x):
    topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices # b, k
    indices = torch.sort(indices, dim=-1).values
    return indices


class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.k = k

    def __call__(self, x, sigma):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def extract_patches_from_indices(x, indices):
    batch_size, _, channels = x.shape
    k = indices.shape[-1]
    patches = x
    patches = batched_index_select(patches, 1, indices)
    patches = patches.contiguous().view(batch_size, k, channels)
    return patches


def extract_patches_from_indicators(x, indicators):
    indicators = rearrange(indicators, "b d k -> b k d") # [bs,left_t,t]
    patches = torch.einsum("b k d, b d c -> b k c",
                         indicators, x)
    return patches

def min_max_norm(x):
    flatten_score_min = x.min(axis=-1, keepdim=True).values
    flatten_score_max = x.max(axis=-1, keepdim=True).values
    norm_flatten_score = (x - flatten_score_min) / (flatten_score_max - flatten_score_min + 1e-5)
    return norm_flatten_score


class SelectionNet(nn.Module):
    """
    Selection Network: select top-k frames or patches
    k: top-k
    stride: stride of the anchor 
    score: temporal or spatial
    num_prompt: append the number of the prompt tokens
    """
    def __init__(self, score, k, in_channels, num_prompt=10, stride=None, num_samples=500):
        super(SelectionNet, self).__init__()
        self.k = k
        self.anchor_size = int(sqrt(k))
        self.stride = stride
        self.score = score
        self.in_channels = in_channels
        self.num_samples = num_samples
        self.num_prompt = num_prompt
        
        if score == 'tpool':
            self.score_network = PromptTransformer(dim=2*in_channels, num_heads=3, mlp_ratio=4, 
                                                   drop_path=0.1, qkv_bias=True, num_tokens=num_prompt)
        elif score == 'spatch':
            self.score_network = PromptTransformer(dim=in_channels, num_heads=3, mlp_ratio=4, 
                                                   drop_path=0.1, qkv_bias=True, num_tokens=num_prompt)
            self.init = torch.eye(self.k).unsqueeze(0).unsqueeze(-1).cuda()
    
    def get_indicator(self, scores, k, sigma):
        indicator = PerturbedTopKFunction.apply(scores, k, self.num_samples, sigma)
        indicator = einops.rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def get_indices(self, scores, k):
        indices = HardTopK(k, scores)
        return indices
    
    def generate_random_indices(self, b, n, k):
        indices = []
        for _ in range(b):
            indice = np.sort(np.random.choice(n, k, replace=False))
            indices.append(indice)
        indices = np.vstack(indices)
        indices = torch.Tensor(indices).long().cuda()
        return indices
    
    def generate_uniform_indices(self, b, n, k):
        indices = torch.linspace(0, n-1, steps=k).long()
        indices = indices.unsqueeze(0).cuda()
        indices = indices.repeat(b, 1)
        return indices
            
        
    def forward(self, x, type, N, T, sigma):
        B = x.size(0)
        H = W = int(sqrt(N))
        indicator = None
        indices = None

        if type == 'time':
            if self.score == 'tpool':
                x = rearrange(x, 'b (t n) m -> b t n m', t=T) # [bs,t,hw,d]
                avg = torch.mean(x, dim=2, keepdim=False) # [bs,t,d]
                max_ = torch.max(x, dim=2).values
                x_ = torch.cat((avg, max_), dim=2) # [bs,t,2*d]
                scores = self.score_network(x_).squeeze(-1) # [bs,t]
                scores = min_max_norm(scores) # [bs,t]
                
                if self.training:
                    indicator = self.get_indicator(scores, self.k, sigma) # [bs,t,left_t]
                else:
                    indices = self.get_indices(scores, self.k)
                x = rearrange(x, 'b t n m -> b t (n m)') # [bs,t,h*w*d]
        else:
            s = self.stride if self.stride is not None else int(max((H - self.anchor_size) // 2, 1))
          
            if self.score == 'spatch':
                x = rearrange(x, 'b (t n) c -> (b t) n c', t=T)
                scores = self.score_network(x)
                scores = rearrange(scores, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
                scores = F.unfold(scores, kernel_size=self.anchor_size, stride=s)
                scores = scores.mean(dim=1)
                scores = min_max_norm(scores)
                
                x = rearrange(x, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
                x = F.unfold(x, kernel_size=self.anchor_size, stride=s).permute(0, 2, 1).contiguous()

                if self.training:
                    indicator = self.get_indicator(scores, 1, sigma)
                    
                else:
                    indices = self.get_indices(scores, 1)

        if self.training:
            if indicator is not None:
                patches = extract_patches_from_indicators(x, indicator) #[bs,left_t,h*w*d]

            elif indices is not None:
                patches = extract_patches_from_indices(x, indices)
                
            if type == 'time':
                patches = rearrange(patches, 'b k (n c) -> b (k n) c', n = N)

            elif self.score == 'spatch':
                patches = rearrange(patches, '(b t) k (c kh kw) -> b (t k kh kw) c',
                    b=B, c=self.in_channels, kh=self.anchor_size) 

            return patches
        else:
            patches = extract_patches_from_indices(x, indices)
            if type == 'time':
                patches = rearrange(patches, 'b k (n c) -> b (k n) c', n = N)

            elif self.score == 'spatch':
                patches = rearrange(patches, '(b t) k (c kh kw) -> b (t k kh kw) c', 
                    b=B, c=self.in_channels, kh=self.anchor_size)
            return patches

    

class PatchNet(nn.Module):
    def __init__(self, score, k, in_channels, stride=None, num_samples=500):
        super(PatchNet, self).__init__()
        self.k = k
        self.anchor_size = int(sqrt(k))
        self.stride = stride
        self.score = score
        self.in_channels = in_channels
        self.num_samples = num_samples

        if score == 'tpool':
            self.score_network = PredictorLG(embed_dim=2*in_channels)
        
        elif score == 'spatch':
            self.score_network = PredictorLG(embed_dim=in_channels)
            self.init = torch.eye(self.k).unsqueeze(0).unsqueeze(-1).cuda()
    
    def get_indicator(self, scores, k, sigma):
        indicator = PerturbedTopKFunction.apply(scores, k, self.num_samples, sigma)
        indicator = einops.rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def get_indices(self, scores, k):
        indices = HardTopK(k, scores)
        return indices
    
    def generate_random_indices(self, b, n, k):
        indices = []
        for _ in range(b):
            indice = np.sort(np.random.choice(n, k, replace=False))
            indices.append(indice)
        indices = np.vstack(indices)
        indices = torch.Tensor(indices).long().cuda()
        return indices
    
    def generate_uniform_indices(self, b, n, k):
        indices = torch.linspace(0, n-1, steps=k).long()
        indices = indices.unsqueeze(0).cuda()
        indices = indices.repeat(b, 1)
        return indices


    def forward(self, x, type, N, T, sigma):
        B = x.size(0)
        H = W = int(sqrt(N))
        indicator = None
        indices = None

        if type == 'time':
            if self.score == 'tpool':
                x = rearrange(x, 'b (t n) m -> b t n m', t=T) # [bs,t,hw,d]
                avg = torch.mean(x, dim=2, keepdim=False) # [bs,t,d]
                max_ = torch.max(x, dim=2).values
                x_ = torch.cat((avg, max_), dim=2) # [bs,t,2*d]
                scores = self.score_network(x_).squeeze(-1) # [bs,t]
                scores = min_max_norm(scores) # [bs,t]
                
                if self.training:
                    indicator = self.get_indicator(scores, self.k, sigma) # [bs,t,left_t]
                else:
                    indices = self.get_indices(scores, self.k)
                x = rearrange(x, 'b t n m -> b t (n m)') # [bs,t,h*w*d]
            
        else:
            s = self.stride if self.stride is not None else int(max((H - self.anchor_size) // 2, 1))
          
            if self.score == 'spatch':
                x = rearrange(x, 'b (t n) c -> (b t) n c', t=T)
                scores = self.score_network(x)
                scores = rearrange(scores, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
                scores = F.unfold(scores, kernel_size=self.anchor_size, stride=s)
                scores = scores.mean(dim=1)
                scores = min_max_norm(scores)
                
                x = rearrange(x, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
                x = F.unfold(x, kernel_size=self.anchor_size, stride=s).permute(0, 2, 1).contiguous()

                if self.training:
                    indicator = self.get_indicator(scores, 1, sigma)
                    
                else:
                    indices = self.get_indices(scores, 1)
         
            

        
        if self.training:
            if indicator is not None:
                patches = extract_patches_from_indicators(x, indicator) #[bs,left_t,h*w*d]

            elif indices is not None:
                patches = extract_patches_from_indices(x, indices)
                
            if type == 'time':
                patches = rearrange(patches, 'b k (n c) -> b (k n) c', n = N)

            elif self.score == 'spatch':
                patches = rearrange(patches, '(b t) k (c kh kw) -> b (t k kh kw) c',
                    b=B, c=self.in_channels, kh=self.anchor_size) 

            return patches
        
        
        else:
            patches = extract_patches_from_indices(x, indices)
            
            if type == 'time':
                patches = rearrange(patches, 'b k (n c) -> b (k n) c', n = N)

            elif self.score == 'spatch':
                patches = rearrange(patches, '(b t) k (c kh kw) -> b (t k kh kw) c', 
                    b=B, c=self.in_channels, kh=self.anchor_size)
            
            return patches
            
            

