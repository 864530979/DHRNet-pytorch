import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class SGA(nn.Module):  #multi-scale feature  #todo:greatly decreased the implemention difficulty   #todo:初始化权重
    def __init__(self,ch,ratio=8,eps=1e-6):  #自注意力
        super(SGA, self).__init__()
        self.mapping_conv_q = nn.Conv2d(ch, ch//ratio, 1)
        self.mapping_conv_k  = nn.Conv2d(ch, ch//ratio, 1)
        self.mapping_conv_v  = nn.Conv2d(ch, ch, 1)
        self.softplus = nn.Softplus()
        self.eps = eps
        self.ratio = ratio


    def forward(self,x):
        B,C,H,W = x.shape
        low_q = self.mapping_conv_q(x).reshape(B, C//self.ratio, -1).transpose(-2,-1) #B,L,C
        low_k = self.mapping_conv_k(x).reshape(B, C//self.ratio, -1)
        low_v = self.mapping_conv_v(x).reshape(B, C, -1)
        low_q = self.softplus(low_q)
        low_k = self.softplus(low_k)
        attn_map = torch.einsum("bmn, bcn->bmc",low_k,low_v)
        norm = 1 / torch.einsum("bnc, bc->bn",low_q,torch.sum(low_k,dim=-1)+self.eps)
        out = torch.einsum("bnm, bmc, bn->bcn",low_q,attn_map,norm)
        out = out.reshape(B,C,H,W)
        return out.contiguous()


class MFNA_PM(nn.Module):  #multi-scale feature  #todo:greatly decreased the implemention difficulty  #todo:初始化权重
    def __init__(self,low_ch,ratio=1,eps=1e-6):  #Multi-feature -non-local - attention  64--128
        super(MFNA_PM, self).__init__()
        self.high_mapping_conv_q = nn.Conv2d(low_ch, low_ch//ratio, 1)
        self.mapping_conv_k = nn.Conv2d(low_ch, low_ch//ratio, 1)
        self.low_mapping_conv_v  = nn.Conv2d(low_ch, low_ch, 1)
        self.softplus = nn.Softplus()
        self.eps = eps
        self.ratio = ratio


    def forward(self,high,low):
        B, C, H, W = low.shape
        high_q = self.high_mapping_conv_q(high).reshape(B, C//self.ratio, -1).transpose(-2,-1)
        k = self.mapping_conv_k(low).reshape(B, C//self.ratio, -1)
        low_v = self.low_mapping_conv_v(low).reshape(B, C, -1)
        high_q = self.softplus(high_q)
        k = self.softplus(k)
        attn_map = torch.einsum("bmn, bcn->bmc", k, low_v)
        norm = 1 / torch.einsum("bnc, bc->bn", high_q, torch.sum(k, dim=-1) + self.eps)
        out = torch.einsum("bnm, bmc, bn->bcn", high_q, attn_map, norm)
        out = out.reshape(B, C, H, W)
        return out.contiguous()

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0,dilation=1,bias=False,activate_first=True,inplace=True):
        super(SeparableConv2d,self).__init__()
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.GELU()
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.GELU()
        self.activate_first = activate_first
    def forward(self,x):
        x = self.depthwise(x)
        x = self.bn1(x)
        if self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x


class MCA(nn.Module):  #multi-scale feature  #todo:greatly decreased the implemention difficulty  #todo:初始化权重
    def __init__(self,low_ch,eps=1e-6,n_heads=8):  #Multi-feature -non-local - attention  64--128
        super(MCA, self).__init__()
        self.high_mapping_conv_q = nn.Conv2d(low_ch, low_ch//2, 1)
        self.low_mapping_conv_q = nn.Conv2d(low_ch,low_ch//2,1)
        self.mapping_conv_k = nn.Conv2d(low_ch, low_ch, 1)
        self.low_mapping_conv_v  = nn.Conv2d(low_ch, low_ch, 1)
        self.out_mapping_conv = nn.Conv2d(low_ch,low_ch,1)
        self.softplus = nn.Softplus()
        self.eps = eps
        self.n_heads=n_heads


    def forward(self,high,low):
        B, C, H, W = low.shape
        high_q = self.high_mapping_conv_q(high).reshape(B,self.n_heads//2,C//(self.n_heads), H*W).permute(0,1,3,2).contiguous() #B,N//2,C//N,L->B,N//2,L,C//N
        low_q = self.low_mapping_conv_q(low).reshape(B,self.n_heads//2,C//(self.n_heads), H*W).permute(0,1,3,2).contiguous()
        q = torch.cat([high_q,low_q],dim=1) #B,N,L,C//N
        k = self.mapping_conv_k(low).reshape(B, self.n_heads,C//self.n_heads, H*W)  #B,N,C//N,L
        v = self.low_mapping_conv_v(low).reshape(B, self.n_heads,C//self.n_heads, H*W) #B,N,C//N,L
        q = self.softplus(q)
        k = self.softplus(k)
        attn_map = torch.einsum("bpmn, bpcn->bpmc", k, v)
        norm = 1 / torch.einsum("bpnc, bpc->bpn", q, torch.sum(k, dim=-1) + self.eps)
        out = torch.einsum("bpnm, bpmc, bpn->bpcn", q, attn_map, norm)
        out = out.reshape(B, C, H, W)
        out = self.out_mapping_conv(out)
        return out.contiguous()

class MSA(nn.Module):  #multi-scale feature
    def __init__(self,ch,eps=1e-6,num_heads=8):  #多头自注意力
        super(MSA, self).__init__()
        self.num_heads = num_heads
        self.mapping_conv_q = nn.Conv2d(ch, ch, 1)
        self.mapping_conv_k  = nn.Conv2d(ch, ch, 1)
        self.mapping_conv_v  = nn.Conv2d(ch, ch, 1)
        self.softplus = nn.Softplus()
        self.eps = eps

    def forward(self,x):
        B,C,H,W = x.shape
        low_q = self.mapping_conv_q(x).reshape(B,self.num_heads, C//self.num_heads, H*W).transpose(-2,-1) #B,L,C
        low_k = self.mapping_conv_k(x).reshape(B,self.num_heads, C//self.num_heads, H*W)
        low_v = self.mapping_conv_v(x).reshape(B,self.num_heads, C//self.num_heads, H*W)
        low_q = self.softplus(low_q)
        low_k = self.softplus(low_k)
        attn_map = torch.einsum("bpmn, bpcn->bpmc",low_k,low_v)
        norm = 1 / torch.einsum("bpnc, bpc->bpn",low_q,torch.sum(low_k,dim=-1)+self.eps)
        out = torch.einsum("bpnm, bpmc, bpn->bpcn",low_q,attn_map,norm)
        out = out.reshape(B,C,H,W)
        return out.contiguous()

class Attention_FFT(nn.Module):
    def __init__(self, dim,mode_query=None,mode_query_shape=None, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        if mode_query:
            self.mode_query = mode_query  #B,NUM_HEADS,1,H,W//2+1
        else:
            assert mode_query_shape!=None,'one of mode_query and mode_query_shape can not be none '
            self.mode_query = nn.Parameter(torch.ones(mode_query_shape))
        self.mapping = nn.Conv2d(dim,dim,1,1,0)


    def forward(self, x):
        B,C,H,W = x.shape
        x_proj = self.mapping(x).reshape(B,self.num_heads,C//self.num_heads,H,W)
        x_fd = torch.fft.rfft2(x_proj) # B,Num_heads,C//Num_heads,H,W//2+1
        if self.training == False and x_fd.shape[-2]!=self.mode_query.shape[-2]:
            query_h,query_w = self.mode_query.shape[-2:]
            up_samp_query = self.mode_query.reshape(1,self.num_heads,query_h,query_w)
            up_samp_query = F.interpolate(up_samp_query,x_fd.shape[-2:],mode='bilinear',align_corners=True).reshape(1,self.num_heads,1,x_fd.shape[-2],x_fd.shape[-1])
            x_enhance = x_fd * up_samp_query
        else:
            x_enhance = x_fd * self.mode_query
        x_reproj = torch.fft.irfft2(x_enhance) # B,Num_heads,C//Num_heads,H,W
        x = x_reproj.reshape(B,C,H,W)
        return x



if __name__ =='__main__':
    low = torch.randn((2,256,32,32))
    high = torch.randn((2,256,32,32))
    mca = MCA(256)
    out = mca(high,low)
    print(out.shape)
    msa = MSA(256)
    out = msa(high)
    print(out.shape)
