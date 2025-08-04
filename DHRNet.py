import torch
import torch.nn.functional as F

from model.van import van_b2,van_b0,van_b1
from model._resnet import resnet18,resnet34,resnet50
from model.modules import *
from mmcv.ops.deform_conv import DeformConv2dPack



init=False

class layer_Norm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_first'):
        super(layer_Norm, self).__init__()
        self.weight =nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        # [batch_size, height, weight, channel]
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class StripBlock(nn.Module):
    def __init__(self, in_channel, kernel_size,groups):
        super(StripBlock, self).__init__()
        self.strip1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (1, kernel_size), padding=(0, (kernel_size - 1) // 2),groups=groups),
        )
        self.strip2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (kernel_size, 1), padding=((kernel_size - 1) // 2, 0),groups=groups),
            nn.Conv2d(in_channel,in_channel, kernel_size=1, stride=1, padding=0),
        )
        if init:
            self.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')

    def forward(self, x):
        u = x.clone()
        strip = self.strip1(x)
        strip = self.strip2(strip)
        out = u * strip

        return out

class RFEnhance(nn.Module):
    def __init__(self,in_channel,out_channel,type = 'align'):
        super(RFEnhance, self).__init__()
        if type == 'align':
            self.conv_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 5, padding=2, groups=in_channel, bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.Conv2d(in_channel, in_channel, 5, padding=2, groups=in_channel, bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.GELU(),
                                        )

        elif type == 'enhance':
            self.conv_1 = nn.Sequential(nn.Conv2d(in_channel,in_channel,5,padding=2,groups=in_channel,bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.Conv2d(in_channel, in_channel, 7, padding=9, dilation=3, groups=in_channel,
                                                  bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.GELU(),
                                        )
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channel,out_channel,1,1,0,bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    )
        if init:
            self.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')

    def forward(self,x):
        x= self.conv_1(x)
        x = self.conv_2(x)
        return x


class Up_Transpose(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_Transpose, self).__init__()

        self.transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1,1,0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        if init:
            self.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')

    def forward(self, x):
        x = self.transpose(x)
        x = self.out_conv(x)
        return x

class  Multi_Scale_Branch(nn.Module):
    def __init__(self, dim_in, dim_out, bn_mom=0.1,big_kernel = False): #todo:全部加上sga
        super(Multi_Scale_Branch, self).__init__()
        if big_kernel:
            kernel_size = [11,23,31]
        else:
            kernel_size = [7,13,25]

        self.mf1 = nn.Sequential(nn.Conv2d(dim_in,dim_out,kernel_size=1,stride=1,bias=False),
                                 nn.BatchNorm2d(dim_out),
                                 nn.GELU()
        )

        self.mf2 = nn.Sequential(
            nn.Conv2d(dim_in,dim_out,1,1,0),
            StripBlock(dim_out, kernel_size[0], dim_out),
            DeformConv2dPack(dim_out, dim_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.GELU()
        )
        self.mf3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, 0),
            StripBlock(dim_out, kernel_size[1], dim_out),
            DeformConv2dPack(dim_out, dim_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.GELU()

        )
        self.mf4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, 0),
            StripBlock(dim_out, kernel_size[2], dim_out),
            DeformConv2dPack(dim_out, dim_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.GELU()

        )

        self.mf5 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Conv2d(dim_in, dim_out, 1, 1, bias=False),
                                 nn.BatchNorm2d(dim_out, bn_mom),
                                 nn.GELU()
                                 )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, bn_mom),
            nn.GELU()
        )
        if init:
            self.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')

    def forward(self, x):
        mf1 = self.mf1(x)
        mf2 = self.mf2(x)
        mf3 = self.mf3(x)
        mf4 = self.mf4(x)
        mf5 = self.mf5(x)
        mf5 = F.interpolate(mf5, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=True)
        mf_cat = torch.cat([mf1, mf2, mf3, mf4, mf5], dim=1)
        mf_high = self.conv_cat(mf_cat)
        return mf_high


class CAE(nn.Module):
    def __init__(self,high_ch,low_ch,ratio=1):
        super(CAE, self).__init__()
        self.g1 = MFNA_PM(low_ch,ratio=ratio)
        self.catch_ball = RFEnhance(low_ch+high_ch,low_ch,type='align')
        self.add_conv = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, 1, 1, 0
                      ,bias=False),
            nn.BatchNorm2d(low_ch),
            nn.GELU()
        )
        if init:
            self.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')

    def forward(self,high,low):
        u = low.clone()
        ball = u + self.catch_ball(torch.cat([high,low],dim=1))
        g1 = self.g1(ball, low)
        fusion = u+g1
        global_fusion = self.add_conv(fusion)
        return global_fusion

class SAE(nn.Module):
    def __init__(self,in_channels,out_channels,ratio=1):
        super(SAE, self).__init__()
        self.down_channel = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.g1 = SGA(out_channels,ratio=ratio)
        self.out =nn.Sequential(
            nn.Conv2d(out_channels,out_channels,1,1,0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        if init:
            self.apply(self.weight_init)


    def weight_init(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')

    def forward(self,x):
        down = self.down_channel(x)
        attn = self.g1(down)
        out = self.out(attn+down)
        return out


class CFB(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(CFB, self).__init__()
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,1,1,0,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )
        self.strip = RFEnhance(in_channel,out_channel,type='enhance')
        self.add_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )
        if init:
            self.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')

    def forward(self,x):
        short_cut = self.short_cut(x)
        strip = self.strip(x)
        out = self.add_conv(short_cut+strip)
        return out

class Concat(nn.Module):
    def __init__(self, in_size, out_size):
        super(Concat, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1,bias= False)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=1, padding=0,bias= False)
        self.gelu =nn.GELU()
        if init:
            self.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1,inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.gelu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.gelu(outputs)
        return outputs

class Global_Branch(nn.Module):
    def __init__(self,channels,ratio=1):
        super(Global_Branch, self).__init__()
        self.linear_attn = SAE(channels[-1],channels[-1],ratio=ratio)
        self.cross_attn1 = CAE(channels[-1],channels[-2],ratio=ratio)
        self.concat1 = Concat(channels[-1]+channels[-2],channels[-2])
        self.up_trans1 = Up_Transpose(channels[-2],channels[-2])
        self.cross_attn2 = CAE(channels[-2],channels[-3],ratio=ratio)
        self.concat2 = Concat(channels[-2] + channels[-3], channels[-3])
        self.up_trans2 = Up_Transpose(channels[-3], channels[-3])
        self.cfb = CFB(channels[-4],channels[-4])
        self.concat3 = Concat(channels[-3]+channels[-4],channels[-3])


    def forward(self,feat):
        feat4 = self.linear_attn(feat[-1])
        enhance1 = self.cross_attn1(feat4,feat[-2])
        fusion1 = self.concat1(feat4,enhance1)
        fusion1 = self.up_trans1(fusion1)
        enhance2 = self.cross_attn2(fusion1,feat[-3])
        fusion2 = self.concat2(fusion1,enhance2)
        fusion2 = self.up_trans2(fusion2)
        low_feat = self.cfb(feat[-4])
        fusion3 = self.concat3(fusion2,low_feat)

        return fusion3




class DHRNet(nn.Module):
    def __init__(self,num_classes=6,big_kernel=True, pretrained=True,mode='tiny',ratio=1,aux=False):
        super(DHRNet, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.aux = aux
        if mode == 'tiny':
            self.backbone = van_b0(pretrained=pretrained)
            channels = [32, 64, 160, 256]
        elif mode =='base':
            self.backbone = van_b1(pretrained=pretrained)
            channels = [64, 128, 320, 512]
        elif mode == 'huge':
            self.backbone = van_b2(pretrained=pretrained)
            channels = [64, 128, 320, 512]
        else:
            channels=[]
            assert 'Unsupported model selected'

        self.multi_scale_branch = Multi_Scale_Branch(channels[-1],channels[-2],big_kernel=big_kernel)
        self.global_context_and_detail_enhancement_branch = Global_Branch(channels,ratio=ratio)
        self.up_4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels[-2]+channels[-3],channels[-2]+channels[-3],3,1,1,bias=False),
            nn.BatchNorm2d(channels[-2]+channels[-3]),
            nn.GELU(),
            nn.Conv2d(channels[-2]+channels[-3],(channels[-2]+channels[-3])//2,1,1,0,bias=False),
            nn.BatchNorm2d((channels[-2]+channels[-3])//2),
            nn.GELU())

        self.up_trans = nn.Sequential(
            Up_Transpose((channels[-2]+channels[-3])//2,channels[-4]),
            Up_Transpose(channels[-4],channels[-4])
        )

        self.cls_conv_out = nn.Conv2d(channels[-4], num_classes, 1, stride=1)

        if aux:
            self.aux = nn.Sequential(nn.Conv2d((channels[-2] + channels[-3]) // 2, num_classes, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(num_classes),
                                     nn.GELU(),
                                     nn.Conv2d(num_classes, num_classes, 1, 1, 0),
                                     )


        if init:
            self.fusion.apply(self.weight_init)
            self.cls_conv_out.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')

    def forward(self, x):
        B,_,H,W = x.shape
        feat = self.backbone(x)
        multi_scale_branch = self.multi_scale_branch(feat[-1])
        global_branch = self.global_context_and_detail_enhancement_branch(feat)
        multi_scale_branch = self.up_4x(multi_scale_branch)
        fusion = self.fusion(torch.cat([multi_scale_branch,global_branch],dim=1))
        x = self.up_trans(fusion)
        x = self.cls_conv_out(x)
        if self.aux and self.training:
            aux = self.aux(fusion)
            # aux = aux.reshape(B, x.shape[1])
            return x,aux
        return x


if __name__ == '__main__':
    backbone = 'tiny'  ##backbone
    downsample_factor = 16
    pretrained = False
    model = DHRNet(num_classes=6,
                    pretrained=pretrained,mode=backbone).cuda()
    inputs = torch.randn((2, 3, 512,512)).cuda()
    output = model(inputs)

    print(output.shape)
