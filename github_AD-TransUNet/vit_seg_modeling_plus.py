# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce
from torchvision.ops import DeformConv2d
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, BatchNorm2d, Sigmoid
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from .gate_attention import GateAttention
from .grid_attention_layer import GridAttentionBlock2D
from .EATformer import EATBlock,GLI
from .utils import UnetGridGatingSignal2
from .SELayer import SELayer,PSPPSEPreActBlock
from .CBAM import CBAM
from .SimAM import SimAM
from .CCNet import CCNet
from .CCNet_plus import CCNetPlus
from .dattentionbaseline import DAttentionBaseline


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def find_module_path(module, state_dict):
    """
    查找模块在状态字典中对应的路径
    如果找到,返回路径
    否则,返回 None
    """
    for path, _ in state_dict.items():
        if ".".join(path.split(".")[:-1]) == str(module):
            return path
    return None

def copy_weights(weights, dest_module, src_path, dest_path=None):
    if dest_path is None:
        dest_state = dest_module.state_dict()
        module_path = find_module_path(dest_module, dest_state)
        if module_path is None:
            raise ValueError(f"Could not find path for module {dest_module} in state dict")
        dest_state[module_path].copy_(np2th(weights[src_path]))
    else:
        dest_module_path = dest_module
        if isinstance(dest_module_path, str):
            dest_module_path = dest_module
        dest_state = dest_module_path.state_dict()
        dest_state[dest_path].copy_(np2th(weights[src_path]))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis): # vis 主要用于控制是否记录注意力权重
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.attention_baseline = DAttentionBaseline(
            q_size=(14, 14),  # 根据实际输入特征图大小调整
            n_heads=self.num_attention_heads,
            n_head_channels=self.attention_head_size,
            n_groups=8,  # 根据实际设置调整
            attn_drop=config.transformer["attention_dropout_rate"],
            proj_drop=config.transformer["attention_dropout_rate"],
            stride=1,
            offset_range_factor=1.0,
            use_pe=False,
            dwc_pe=False,
            no_off=False,
            fixed_pe=False,
            ksize=3,
            log_cpb=True
        )

        self.out = Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        attention_output, weights = self.attention_baseline(hidden_states)
        attention_output = self.out(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"] # grid=(14,14) 切片大小
            # grid_size 是一个包含两个元素的元组，表示希望在图像上划分的块的 行数 和 列数
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1]) # (1,1)
            # patch_size 是一个元组，包含两个元素，表示每个块的高度和宽度
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16) # (16,16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  # 14*14=196
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16 # hybrid_model.width = 64
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # 添加位置嵌入 hidden_size表示位置嵌入的维度
        # （1,196,768）
        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid: # True
            x, features = self.hybrid_model(x) # (B,3,H,W) -> (B,)
            #           x: (B,512,H/16,W/16)
            # features[0]: (B,256,H/8,W/8)
            # features[1]: (B,128,H/4,H/4)
            # features[2]: (B,64,H/2,W/2)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden， n_patches^(1/2), n_patches^(1/2)) = (B,768,H/16,W/16)
        x = x.flatten(2) # (B, hidden, n_patches)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings # 位置嵌入
        embeddings = self.dropout(embeddings)
        return embeddings, features


class DCN2_OP(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, deform_groups=4):
        super().__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv1_offset = nn.Conv2d(dim, deform_groups * offset_channels, kernel_size=3, stride=stride, padding=1)
        self.conv1 = DeformConv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.norm1 = nn.BatchNorm2d(dim)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x, img_size=(14, 14)):
        batch_size, patches, hidden_size = x.shape
        # 将补丁序列转换为特征图
        x = x.permute(0, 2, 1).reshape(batch_size, hidden_size, *img_size)
        # 生成偏移量
        offset = self.conv1_offset(x)
        # 执行可变形卷积
        x = self.conv1(x, offset)
        # 归一化和激活
        x = self.norm1(x)
        x = self.act1(x)
        # 1x1卷积
        x = self.conv2(x)
        # 将特征图转换回补丁序列
        x = x.flatten(2).permute(0, 2, 1)
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attn = DAttentionBaseline(
            q_size=(14,14),
            n_heads=self.num_attention_heads,
            n_head_channels=self.attention_head_size,
            n_groups=8,
            attn_drop=config.transformer["attention_dropout_rate"],
            proj_drop=config.transformer["attention_dropout_rate"],
            offset_range_factor=1.0,
            use_pe=False,
            dwc_pe=False,
            no_off=False,
            fixed_pe=False,
            ksize=3,
            stride=1,
            log_cpb=True
        )
        # self.gli_layer = GLI(in_dim=config.hidden_size,
        #                     dim_head=16,
        #                     window_size=7,
        #                     op_names=['mdmsa', 'dw'])
        # self.dcn_op = DCN2_OP(config.hidden_size, deform_groups=4)  # 添加可变形卷积模块

    def forward(self, x, img_size=(14, 14)):
        # 在进入LayerNorm之前执行可变形卷积
        # x = self.dcn_op(x, img_size) # 存在Gli+中了

        h = x # [3,196,768]
        x = self.attention_norm(x)
        # print("after attention_norm x shape:",x.shape) # [3, 196, 768]
        x, weights = self.attn(x)
        # x = self.gli_layer(x) # 替换self-Attention
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            query_weight = query_weight.view(768, 768, 1, 1)  # 重新调整形状

            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            key_weight = key_weight.view(768, 768, 1, 1)  # 重新调整形状
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            value_weight = value_weight.view(768, 768, 1, 1)
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = out_weight.view(768, 768, 1, 1)  # 重新调整形状

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.proj_q.weight.copy_(query_weight)
            self.attn.proj_k.weight.copy_(key_weight)
            self.attn.proj_v.weight.copy_(value_weight)
            self.attn.proj_out.weight.copy_(out_weight)
            self.attn.proj_q.bias.copy_(query_bias)
            self.attn.proj_k.bias.copy_(key_bias)
            self.attn.proj_v.bias.copy_(value_bias)
            self.attn.proj_out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis # 保存可视化标志，用于在训练过程中可视化注意力权重
        self.layer = nn.ModuleList() # # 创建一个模块列表moduleList容器，用于存储编码器中的所有编码块
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6) # 该层用于对每个Block的输出进行归一化
        for _ in range(config.transformer["num_layers"]): #num_layers = 12
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = [] # 创建一个列表，用于存储每层编码块产生的注意力权重
        for layer_block in self.layer:
            hidden_states= layer_block(hidden_states) # 对输入进行编码，并获取输出和注意力权重
            # if self.vis: # 如果需要可视化注意力权重
            #     attn_weights.append(weights) # 将注意力权重添加到列表中
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size) # 创建嵌入层，用于生成图像的嵌入表示
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids): # (B,3,H,W)
        # 将x送入self.embeddings中
        embedding_output, features = self.embeddings(input_ids)
        encoded= self.encoder(embedding_output)  # (B, n_patch, hidden) (B,196,768)
        return encoded,  features # features：表示CNN支路中的3个特征图


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        # 创建一个二维卷积层，其参数根据传入的值进行设置。如果use_batchnorm为True，则卷积层自动禁用偏置项（bias=False），因为批量归一化层将接手这部分功能。
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels) #如果use_batchnorm为True，则实例化一个二维批量归一化层，其通道数等于输出通道数
        # 将卷积层、批量归一化层（如果启用）和ReLU激活函数按照顺序添加到nn.Sequential容器中，这样当调用这个类的实例的forward方法时，数据会依次经过这三个层。
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class CoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = self.pool_h(x)                     # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (B, C, 1, W) -> (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)         # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        y_h, y_w = torch.split(y, [H, W], dim=2)
        y_h = self.conv_h(y_h).sigmoid()          # (B, C, H, 1)
        y_w = self.conv_w(y_w).sigmoid()          # (B, C, W, 1)

        return x * y_h * y_w.permute(0, 1, 3, 2)

# class DecoderBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             skip_channels=0,
#             use_batchnorm=True,
#     ):
#         super().__init__()
#         self.conv1 = Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )

#         # 坐标注意力模块
#         self.coord_att = CoordAtt(out_channels)

#         self.conv2 = Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         # # 初始化一个上采样模块，如双线性插值或者子像素卷积（Transposed Convolution）
#          self.up = nn.UpsamplingBilinear2d(scale_factor=2)

# #-----------------实例化GLI模块------------------
#         # if skip_channels != 0:
#         #     self.gli = EATBlock(emb_dim=skip_channels)

#         # 初始化CCNetplus实例
#         self.ccnetplus = CCNetPlus(in_channels=out_channels, out_channels=out_channels)

#     def forward(self, x, skip=None):
#         # k = x
#         # gating = self.gating(k) # 对最后一个下采样后的特征图设置为门控信号g
#         # gating = self.up(gating)
#         # print("skip[1]:",skip.shape[1])
#         # skip, attn = self.gate_attention_layer(skip, gating)

#         # x = self.up(x)

#         if skip is not None:

# #-----------------------引入GLI 和 Attention Gate-----------------------
#             # skip = self.gli(skip)
#             # x = self.gate_attention(x, skip)
#             # x, att = self.gate_attention_layer(x, skip)
# #----------------------------------------------------------------------

#             x = torch.cat([x, skip], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x) # 1*1
#         return x

class DecoderBlockProgressive(nn.Module):
    def __init__(
            self,
            in_channels,          # 输入通道数（低分辨率特征）
            out_channels,         # 输出通道数
            skip_channels=0,      # 跳跃连接通道数（没有则为0）
            use_batchnorm=True,
    ):
        super().__init__()
        self.skip_channels = skip_channels

        # Pixel Shuffle 上采样部分（固定 2 倍上采样）
        self.pixel_shuffle_conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # 第一个卷积：融合 skip 后，将通道数压缩回 out_channels
        self.conv1 = Conv2dReLU(
            out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        # 坐标注意力（放在两个卷积之间）
        self.coord_att = CoordAtt(out_channels)

        # 第二个卷积
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        # 1. Pixel Shuffle 上采样（自动处理尺寸）
        x = self.pixel_shuffle_conv(x)      # (B, out*4, H, W)
        x = self.pixel_shuffle(x)           # (B, out, 2H, 2W)

        # 2. 融合跳跃连接（如果存在）
        if skip is not None:
            x = torch.cat([x, skip], dim=1)   # (B, out+skip_c, 2H, 2W)

        # 3. 卷积1 -> 坐标注意力 -> 卷积2
        x = self.conv1(x)                   # (B, out, 2H, 2W)
        x = self.coord_att(x)               # 坐标注意力增强空间结构
        x = self.conv2(x)                   # (B, out, 2H, 2W)

        return x

class SegmentationHead(nn.Sequential): # 作用是将图像分类任务的输出适应为图像分割任务的输出

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # 卷积层用于将解码器的输出映射到最终的类别数量
        # kernel_size定义了卷积核的大小，padding通常是kernel_size的一半，以避免尺寸减小
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # 如果upsampling的值大于1，则使用双线性上采样层来增加输出的分辨率
        # 如果upsampling的值是1或更小，则不进行上采样，使用Identity模块，即直接传递输入
        super().__init__(conv2d, upsampling)
        # 将前面定义的卷积层conv2d和上采样层（或者恒等映射层）upsampling按照顺序添加到Sequential模块中。
        # 这样，在调用模型时，输入会先通过卷积层，然后根据设定条件决定是否进行上采样操作。


class DecoderCup(nn.Module): # 级联上采样
    # 将编码器的输出转换回图像的空间维度
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512 # 定义解码器头部的通道数，这里假设为512.用于计算第一个解码块的输入通道数。
        self.conv_more = Conv2dReLU( # 将输入特征从config.hidden_size 转换为head_channels
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels # 获取解码器通道配置 (256, 128, 64, 16)
        in_channels = [head_channels] + list(decoder_channels[:-1]) # 从列表中排除最后一个元素 [512,256,128,64]
        out_channels = decoder_channels  #(256, 128, 64, 16)

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels # [512, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            # DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
            DecoderBlockProgressive(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        '''
                (1)self.block共有4层, 对应上面对DecoderBlock()的定义, 
                   其中zip(in_channels, out_channels, skip_channels)这里面的参数有
                   4 组：(512, 256, 512); (256, 128, 256); (128, 64, 64); (64, 16, 0)。
                (2)skip为features的三个特征图: (B,512,H/8,W/8);(B,256,H/4,W/4);(B,64,H/2,W/2)。
                (3)Decoder_Block()作用：先对 x 进行上采样,然后将 x 与 skip 进行cat, 再对cat后
                   的x进行卷积使其channel变成256, 128, 64。
                   但是当i=3时，不在进行skip与x拼接(cat)，Decoder_Block()的作用：对x的channel降维到16,
                   最后输出的x:(B,16,H,W)=(B,16,224,224)
        '''
        self.se_layers_fusion = SELayer(in_channel=768, out_channel=768, reduction=8)

    def forward(self, hidden_states, features=None):

        hidden_states, *_ = hidden_states
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # print("hidden_states.size:",hidden_states.shape) [3, 196, 768]
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1) # (B,hidden,n_patch)
        x = x.contiguous().view(B, hidden, h, w) # [3,768,14,14]

#-------------------------加入CCNet模块-------------------------------
        # x = self.ccnet(x)
        # x = self.ccnetplus(x) # 可以
#-----------------------------------------------------------

        x = self.conv_more(x)
        # print("x shape:",x.shape) # [3, 512, 14, 14]

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            # print(x.shape, skip.shape)
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1: # 通过判断输入的通道数是否为1  x 是一个四维的张量 表示图像 将其复制三遍
            x = x.repeat(1,3,1,1)   # 它的维度为 (B, C, H, W) (1,3,1,1) 表示在每个维度上的复制次数即(B,3,H,W)
        # 然后将x送入self.transformer中
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x,  features = self.transformer(x)  # (B, n_patch, hidden)
        # (B, n_patch, hidden):(B,196,768) # features:(B,512,H/8,W/8);(B,256,H/4,W/4);(B,64,H/2,W/2)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits # 返回最后的分类结果

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


