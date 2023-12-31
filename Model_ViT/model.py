from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

# class PatchEmbedInverse(nn.Module):
#     def __init__(self, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
#         super().__init__()
#         patch_size = (patch_size, patch_size)
#         self.proj_inverse = nn.ConvTranspose2d(embed_dim, in_c, kernel_size=patch_size, stride=patch_size)
#         self.norm_inverse = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, num_patches, embed_dim = x.shape
#         assert num_patches == self.num_patches, \
#             f"Input embedding size ({num_patches}) doesn't match model ({self.num_patches})."
#
#         x = x.transpose(1, 2)
#         x = self.norm_inverse(x)
#
#         x = x.transpose(1, 2)
#         x = x.view(B, embed_dim, self.grid_size[0], self.grid_size[1])
#         x = self.proj_inverse(x)
#
#         return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])   # 224//16=14
        self.num_patches = self.grid_size[0] * self.grid_size[1]    # 计算patch的数目

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
       
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):     # 实现Multi_Head_Attention模块
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads     # ----->求出head_dim，下一步可能用到
        self.scale = qk_scale or head_dim ** -0.5   # 根据实际情况看是否传入了qk_scale，head_dim ** -0.5（公式）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # 直接使用一个全连接层得到QKV，可能有助于并行化
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)     # 在最后一个维度上进行 softmax 操作，以获得注意力权重。
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]--拼接了最后两维度信息
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):   # Encoder Block层中的MLP Block
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # hidden_feature变为原来的4倍之后，out_feature又变为原来的大小
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)   # 
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,   # 每个token的dimension
                 num_heads,   # multi_head_attention中使用head的个数
                 mlp_ratio=4.,  # 对应第一个全连接层的节点个数是输入的节点四倍
                 qkv_bias=False,
                 qk_scale=None,     # q@k.transpose()
                 drop_ratio=0.,     # multi_head_attention中最后的全连接层使用的
                 attn_drop_ratio=0.,    # softmax层后面用的
                 drop_path_ratio=0.,    # Dropout层使用的
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)  # 调用前面定义的类(实例化)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Trans_conv(nn.Module):
    def __init__(self):
        super(Trans_conv, self).__init__()

    def forward(self, x):
        batch_size, num_patches, embedding_dim = x.size()
        image_height = image_width = int(num_patches ** 0.5)
        reshaped_output = x.view(batch_size, image_height, image_width, embedding_dim)
        x = reshaped_output.permute(0, 3, 1, 2)  # 将通道维度移到第二维度

        return x


# class UpSample(nn.Module):
#     def __init__(self):
#         super(UpSample, self).__init__()
#         # 目标尺寸
#         target_height, target_width = 80, 80
#
#     def forward(self, x):
#         # 输入数据形状：(16, 768, 14, 14)
#         _, _, H, W = x.size()
#         # 目标尺寸
#         target_height, target_width = 80, 80
#         # 计算高度和宽度的缩放比例
#         scale_factor_h = target_height / H
#         scale_factor_w = target_width / W
#         # 使用双线性插值进行上采样
#         x = F.interpolate(x, scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear', align_corners=False)
#         # 输出数据形状：(16, 768, 80, 80)
#         return x


# class Sample(nn.Module):
    # def __init__(self, embedding_dim):
    #     super(Sample, self).__init__()
    #     self.conv_layer1 = nn.Sequential(
    #         nn.Conv2d(in_channels=embedding_dim, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1),
    #         nn.BatchNorm2d(num_features=256),
    #         nn.LeakyReLU(0.1),
    #         nn.MaxPool2d(kernel_size=2))
    #
    #     self.conv_layer2 = nn.Sequential(
    #         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1),
    #         nn.BatchNorm2d(num_features=256),
    #         nn.LeakyReLU(0.1),
    #         nn.MaxPool2d(kernel_size=2)
    #     )
    #     self.conv_layer3 = nn.Sequential(
    #         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1),
    #         nn.BatchNorm2d(num_features=256),
    #         nn.LeakyReLU(0.1),
    #         nn.MaxPool2d(kernel_size=2)
    #     )
    #
    # def forward(self, x):
    #     # 输入数据形状：(16, 768, 14, 14)
    #     B, C, H, W = x.size()
    #     # 目标尺寸
    #     target_height, target_width = 160, 160
    #     # 计算高度和宽度的缩放比例
    #     scale_factor_h = target_height / H
    #     scale_factor_w = target_width / W
    #     # 使用双线性插值进行上采样(16, 768, 160, 160)
    #     x = F.interpolate(x, scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear', align_corners=False)
    #     x = self.conv_layer1(x)
    #     x = self.conv_layer2(x)
    #     x = self.conv_layer3(x)
    #
    #     return x


class Process(nn.Module):
    def __init__(self, embed_dim):
        super(Process, self).__init__()
        '''
        # Trans-Conv
        self.trans_conv = Trans_conv(batch_size=patch_size, num_patches=num_patches, embedding_dim=embed_dim)
        self.usample = Sample(embedding_dim=embed_dim)
        
        self.W = w
        self.H = h
        self.conv1 = nn.Conv2d(in_channels=embed_dim, out_channels=256, kernel_size=3)
        '''
        # self.cv1 = Conv(c1, c_, 1, 1)

       # 下采样
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.SiLU()
        )
        self.conv1 = nn.Conv2d(256, 256, 1, 1)

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.SiLU()
        )
        self.conv2 = nn.Conv2d(512, 256, 1, 1)

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.SiLU()
        )
        self.conv3 = nn.Conv2d(1024, 256, 1, 1)

    def forward(self, x):
        # Trans-Conv（16，197，768）
        batch_size, num_patches, embedding_dim = x.size()
        image_height = image_width = int(self.num_patches ** 0.5)
        reshaped_output = x.view(batch_size, image_height, image_width, embedding_dim)
        x = reshaped_output.permute(0, 3, 1, 2)  # 将通道维度移到第二维度（16， 768， 14，14）

        # 上采样,输入数据形状：(16, 768, 80, 80)
        _, _, H, W = x.size()
        # 目标尺寸
        target_height, target_width = 80, 80
        # 计算高度和宽度的缩放比例
        scale_factor_h = target_height / H
        scale_factor_w = target_width / W
        # 使用双线性插值进行上采样
        x = F.interpolate(x, scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear', align_corners=False)
        # 输出数据形状：(16, 768, 80, 80)

        # 下采样
        # 16, 256, 80, 80
        x11 = self.conv_layer1(x)
        x12 = self.conv1(x11)

        # 16, 512, 40, 40
        x21 = self.conv_layer2(x11)
        # 16, 256, 40, 40
        x22 = self.conv2(x21)

        # 16, 1024, 20, 20
        x31 = self.conv_layer3(x22)
        # 16, 256, 20, 20
        x32 = self.conv3(x31)

        return x31


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=1,  num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)      # 用partial方法传入参数eps
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 获取num_patches的个数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))     # nn.Parameter构建可训练的参数，使用0矩阵初始化
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 通过.item()方法将张量中的数值转换为Python中的浮点数而得来。

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)   # 通过nn.sequential把列表中的所有模块打包为一个整体，赋值给self。blocks
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Trans-Conv、上采样、下采样
        self.blocks0 = Process(embed_dim=embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
            # OrderedDict有序字典
        else:       # representation_size为None的情况
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        # self.head_dist = None
        # if distilled:
        #     self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()



        # Weight init权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
  
        if self.dist_token is None:     # 在ViT部分dist_token就是None
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        x = x[:, 1:, :]  # 切片操作，去掉序列的第一个位置
        x = self.blocks0(x)


        # if self.dist_token is None:     # 在ViT部分dist_token就是None
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:      # head_dist就是None
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
    

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):       # 构建ViT模型的过程
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


if __name__ == '__main__':
    model1 = VisionTransformer()
    print(model1)