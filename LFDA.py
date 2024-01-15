import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
import random

seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        mindisp = -4
        maxdisp = 4
        self.angRes = cfg.angRes
        self.maxdisp = maxdisp

        self.init_feature = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            ResB(16), ResB(16), ResB(16), ResB(16),
            ResB(16), ResB(16), ResB(16), ResB(16),
            nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(16, 8, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.lastconv = nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        self.build_cost = BuildCost(8, self.angRes, mindisp, maxdisp)
        self.transformer = Transformer(cfg)
        self.aggregate = Aggregate(648, 160, mindisp, maxdisp)

    def forward(self, x, dispGT=None):
        lf = rearrange(x, 'b c (a1 h) (a2 w) -> b c a1 a2 h w', a1=self.angRes, a2=self.angRes)
        x = rearrange(x, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        b, c, _, h, w = x.shape
        feat = self.init_feature(x)
        feat = self.lastconv(feat)
        if dispGT is not None: # Training
            mask = Generate_mask(lf, dispGT)
            cost, ctr = self.build_cost(feat, mask)
            attn_cost = self.transformer(cost, ctr)
            disp = self.aggregate(attn_cost)
            init_disp = None
        else: # Inference
            # print('Validation...')
            mask = torch.ones(1, self.angRes ** 2, h, w).to(x.device)
            cost, ctr = self.build_cost(feat, mask)
            attn_cost = self.transformer(cost, ctr)
            init_disp = self.aggregate(attn_cost)
            mask = Generate_mask(lf, init_disp)
            cost, ctr = self.build_cost(feat, mask)
            attn_cost = self.transformer(cost, ctr)
            disp = self.aggregate(attn_cost)

        return disp, init_disp

class BuildCost(nn.Module):
    def __init__(self, channel_in, angRes, mindisp, maxdisp):
        super(BuildCost, self).__init__()
        self.oacc = ModulateConv2d(channel_in, kernel_size=angRes, stride=1, dilation=1,  bias=False)
        self.angRes = angRes
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        self.flatten = nn.Unfold(kernel_size=1, stride=1, dilation=1, padding=0)

    def forward(self, x, mask):
        b, c, aa, h, w = x.shape
        ctr = x[:,:,40:41, :,:] # center view
        x = rearrange(x, 'b c (a1 a2) h w -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)

        bdr = (self.angRes // 2) * self.maxdisp
        pad = nn.ZeroPad2d((bdr, bdr, bdr, bdr))
        x_pad = pad(x)
        x_pad = rearrange(x_pad, '(b a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        h_pad, w_pad = h + 2 * bdr, w + 2 * bdr

        mask_avg = torch.mean(mask, dim=1, keepdim=True).repeat(1,8,1,1)
        mask_avg = self.flatten(self.flatten(mask_avg).permute(0,2,1).unsqueeze(1)).permute(0,2,1).repeat(1,1,self.angRes**2)
        
        cost = []
        for d in range(self.mindisp, self.maxdisp + 1):
            dila = [h_pad - d, w_pad - d]
            self.oacc.dilation = dila
            crop = (self.angRes // 2) * (d - self.mindisp)
            if d == self.mindisp:
                feat = x_pad
            else:
                feat = x_pad[:, :, crop: -crop, crop: -crop]
            current_cost = self.oacc(feat, mask)
            cost.append(current_cost)

        cost = torch.stack(cost, dim=1)
        return cost, ctr

class Aggregate(nn.Module):
    def __init__(self, inC, channel, mindisp, maxdisp):
        super(Aggregate, self).__init__()
        self.sq = nn.Sequential(
            nn.Conv3d(inC, channel, 1, 1, 0, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Conv2 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Resb1 = ResB3D(channel)
        self.Resb2 = ResB3D(channel)
        self.Conv3 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), nn.BatchNorm3d(channel), nn.LeakyReLU(0.1, inplace=True))
        self.Conv4 = nn.Conv3d(channel, 1, 3, 1, 1, bias=False)
        self.softmax = nn.Softmax(1)
        self.mindisp = mindisp
        self.maxdisp = maxdisp

    def forward(self, psv):
        buffer = self.sq(psv) # psv.shape :: b c a h w
        buffer = self.Conv1(buffer)
        buffer = self.Conv2(buffer)
        buffer = self.Resb1(buffer)
        buffer = self.Resb2(buffer)
        buffer = self.Conv3(buffer)
        score = self.Conv4(buffer)
        attmap = self.softmax(score.squeeze(1))
        temp = torch.zeros(attmap.shape).to(attmap.device)
        for d in range(self.maxdisp - self.mindisp + 1):
            temp[:, d, :, :] = attmap[:, d, :, :] * (self.mindisp + d)
        disp = torch.sum(temp, dim=1, keepdim=True)
        return disp


class ResB(nn.Module):
    def __init__(self, feaC):
        super(ResB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(feaC, feaC, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(feaC, feaC, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(feaC))

    def forward(self, x):
        out = self.conv(x)
        return x + out


class ResB3D(nn.Module):
    def __init__(self, channels):
        super(ResB3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels))
        self.calayer = CALayer(channels, 9)

    def forward(self, x):
        buffer = self.body(x)
        return self.calayer(buffer) + x

class CALayer(nn.Module):
    def __init__(self, channel, num_views):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((num_views, 1, 1))
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // 16, 1, 1, 0, bias=True),
                nn.BatchNorm3d(channel // 16),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(channel // 16, channel, 1, 1, 0, bias=True),
                nn.BatchNorm3d(channel),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def Generate_mask(lf, disp):
    b, c, angRes, _, h, w = lf.shape
    x_base = torch.linspace(0, 1, w).repeat(1, h, 1).to(lf.device)
    y_base = torch.linspace(0, 1, h).repeat(1, w, 1).transpose(1, 2).to(lf.device)
    center = (angRes - 1) // 2
    img_ref = lf[:, :, center, center, :, :]
    img_res = []
    for u in range(angRes):
        for v in range(angRes):
            img = lf[:, :, u, v, :, :]
            if (u == center) & (v == center):
                img_warped = img
            else:
                du, dv = u - center, v - center
                img_warped = warp(img, -disp, du, dv, x_base, y_base)
            img_res.append(abs((img_warped - img_ref)))
    mask = torch.cat(img_res, dim=1)
    out = (1 - mask) ** 2
    return out


def warp(img, disp, du, dv, x_base, y_base):

    b, _, h, w = img.size()
    x_shifts = dv * disp[:, 0, :, :] / w
    y_shifts = du * disp[:, 0, :, :] / h
    flow_field = torch.stack((x_base + x_shifts, y_base + y_shifts), dim=3)
    img_warped = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')
    return img_warped    
    
class ModulateConv2d(nn.Module):
    def __init__(self, channel_in, kernel_size, stride=1, dilation=1, bias=False):
        super(ModulateConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.angRes = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.flatten = nn.Unfold(kernel_size=1, stride=1, dilation=1, padding=0)
        self.channel_in = channel_in

    def forward(self, x, mask):
        mask_flatten = self.flatten(mask)
        Unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
        Unfold2 = nn.Unfold(kernel_size=(1,self.channel_in),stride=1,dilation=(1,self.angRes**2))

        x_unfold = Unfold(x)
        x_unfold_modulated = x_unfold * mask_flatten.repeat(1, x.shape[1], 1)
        x_unfold_modulated_unfold = Unfold2(x_unfold_modulated.permute(0,2,1).unsqueeze(2))
        
        return x_unfold_modulated_unfold

class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.angRes = cfg.angRes
        self.patch_size = cfg.patchsize


        self.depth_attention = DepthAttention()

    def forward(self, cost, ctr): 
        ctr = rearrange(ctr, 'b c a h w -> (b h w) a c')
        ctr = ctr.unsqueeze(2)
        
        out = rearrange(cost, 'b d (h w c) a -> (b h w) a d c', a=self.angRes**2, h=self.patch_size, w=self.patch_size)
        out = self.depth_attention(out, ctr)
        out = rearrange(out, '(b h w) a d c -> b (a c) d h w', h=self.patch_size, w=self.patch_size, a=self.angRes**2, d=self.angRes)

        return out

class DepthAttention(nn.Module):
    def __init__(self):
        super(DepthAttention, self).__init__()
        self.dim = 8
        self.norm = nn.LayerNorm(self.dim)
        self.cross_attention = Depth_MCA(dim = self.dim, 
                                        qkv_bias=False, 
                                        proj_drop=0.)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.LayerNorm(self.dim),
            nn.ReLU(True),
        )


    def forward(self, buffer, ctr):
        buffer = self.cross_attention(query=buffer,
                                    key=ctr,
                                    value=ctr)
        
        buffer = self.norm(buffer)
        buffer = self.feed_forward(buffer)

        return buffer

class Depth_MCA(nn.Module):
    def __init__(self, dim=8, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.manual_seed(0)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.k.bias is not None:
            nn.init.xavier_normal_(self.k.bias)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, query, key, value):
        B, d, N_q, C = query.shape
        B, d_k, N_kv, C = key.shape

        q = self.q(query).reshape(B, d, N_q, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = self.k(key).reshape(B, d_k, N_kv, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-2)
        attn = attn.reshape(B, d, N_q, 1)

        out = query * attn
        return out # b d a c


if __name__ == "__main__":
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', type=str, default='cuda:0')
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--num_workers', type=int, default=10)
        parser.add_argument("--angRes", type=int, default=9, help="angular resolution")
        parser.add_argument('--model_name', type=str, default='')
        parser.add_argument('--trainset_dir', type=str, default='./dataset/training/')
        parser.add_argument('--validset_dir', type=str, default='./dataset/validation/')
        parser.add_argument('--patchsize', type=int, default=32)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--n_epochs', type=int, default=3500, help='number of epochs to train')
        parser.add_argument('--n_steps', type=int, default=3500, help='number of epochs to update learning rate')
        parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
        parser.add_argument('--load_pretrain', type=bool, default=False)
        parser.add_argument('--model_path', type=str, default='./log/LFDT.pth.tar')

        return parser.parse_args()

    cfg = parse_args()
    angRes = 9
    patch_size = 32
    net = Net(cfg).cuda()
    from thop import profile
    input = torch.randn(1, 1, 32 * angRes, 32 * angRes).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
