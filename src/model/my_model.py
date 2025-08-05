"""
based on
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
Using distortion classification and image quality assessment
"""
######### For re-train on other dataset, use 6 heads

from collections import OrderedDict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Mlp, VisionTransformer

from .patch_embed import PatchEmbed


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cosine



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, image_token):
        B, N, C = image_token.shape
        kv = (
            self.kv(image_token)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        B, N, C = query.shape
        q = (
            self.q(query)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Learner(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        num_classes = cfg.model.learner_param.num_classes
        embed_dim = cfg.model.learner_param.embed_dim
        feature_channels = cfg.model.learner_param.feature_channels
        cnn_feature_num = cfg.model.learner_param.cnn_feature_num
        interaction_block_num = cfg.model.learner_param.interaction_block_num
        latent_dim = cfg.model.learner_param.latent_dim
        grid_size = cfg.model.learner_param.grid_size
        #cross_attn_num_heads = cfg.model.learner_param.cross_attn_num_heads

        # hyper net MLP version # TODO: another vesion MHA
        # for CNN features extraction and combination
        # Still extract 4 level of features, and 
        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            feature_channels[i],
                            latent_dim, #down all to 64
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                        nn.GELU(),
                        nn.AdaptiveAvgPool2d(grid_size)
                    ]
                )
                for i in range(cnn_feature_num)
            ]
        )
        # up project features' dimension to ViT emmbedd_dim
        self.up_proj_cnn = Mlp(
             in_features=208,
             hidden_features=latent_dim,
             out_features=1,
        )

        

        self.head_distortion = Mlp(
             in_features=196,
             hidden_features=latent_dim,
             out_features=1,
        )
        # new head for quality score
        self.head = NormedLinear(latent_dim, num_classes)
        self.head2 = NormedLinear(latent_dim, num_classes)
        


    def forward(self, x, y):
        x = x.permute(0,2,1) #B, 64, 208
        x = self.up_proj_cnn(x) #B,64,64
        #print("x shape ", x.shape)
        x = self.head(x[:,:,0])
        y = self.head_distortion(y)
        y = self.head2(y[:,:,0])
        return x, y


class MyModel_v1(VisionTransformer):
    def __init__(
        self,
        cfg=None,
        embed_layer=PatchEmbed,
        basic_state_dict=None,
        *argv,
        **karg,
    ):
        # Recreate ViT
        super().__init__(
            embed_layer=embed_layer,
            *argv,
            **karg,
            **(cfg.model.vit_param),
        )

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.learner = Learner(cfg)
        self.dropout = nn.Dropout(cfg.model.hyper_vit.dropout_rate)
        self.head = nn.Identity()

        # feature_extraction model CNN ResMet50
        self.feature_model = timm.create_model(
            cfg.model.feature_model.name,
            pretrained=cfg.model.feature_model.load_timm_model,
            features_only=True,
            out_indices=cfg.model.feature_model.out_indices,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        # for param in self.blocks[6].parameters(): #ViT block 7
        #     param.requires_grad = True

        for param in self.learner.parameters():
            param.requires_grad = True

        # for re-train
        for param in self.learner.conv.parameters():
            param.requires_grad = False
        for param in self.learner.head_distortion.parameters():
            param.requires_grad = False
        for param in self.learner.head2.parameters():
            param.requires_grad = False

    def un_freeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_state_to_save(self):
        feature_model_state_dict = self.feature_model.state_dict()
        bn_buffer = OrderedDict()
        for key, value in feature_model_state_dict.items():
            if (
                "running_mean" in key
                or "running_var" in key
                or "num_batches_tracked" in key
            ):
                bn_buffer[key] = value

        state_dict_to_save = {
            "learner": self.learner.state_dict(),
            "bn_buffer": bn_buffer,
        }
        return state_dict_to_save

    def load_saved_state(self, saved_state_dict, strict=False):
        self.learner.load_state_dict(saved_state_dict["learner"], strict)
        self.feature_model.load_state_dict(saved_state_dict["bn_buffer"], strict)

    def forward_hyper_net(self, x):
        batch_size = x.shape[0]
        features_list = self.feature_model(x)

        cnn_token_list = []
        for i in range(len(features_list)):
            cnn_image_token = self.learner.conv[i](features_list[i])
            #print("cnn token before reshpae ", cnn_image_token.shape)
            latent_dim = cnn_image_token.shape[1]
            cnn_image_token = cnn_image_token.permute(0, 2, 3, 1).reshape(
                batch_size, -1, latent_dim
            )
            #print("cnn token 1 level ", cnn_image_token.shape)
            cnn_token_list.append(cnn_image_token)

        return torch.cat(cnn_token_list, dim=1) #N,7x7x4, latent_dim

    def forward_features(self, x, cnn_tokens):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i in range(6):
            x = self.blocks[i](x)

        #print("ViT features after block 6: ", x.shape)
        #cnn_tokens = self.learner.up_proj_cnn(cnn_tokens)
        batch_size = cnn_tokens.shape[0]
        latent_dim = cnn_tokens.shape[2]

        cls_feat = x[:,0,:] #N,1,C
        cls_feat = cls_feat.reshape(batch_size, -1, latent_dim)
        #print("cls shape ", cls_feat.shape)
        x = torch.cat((cls_feat, cnn_tokens), dim=1) #N, 49x4+ 12, 64 #check here, add distortion token
        #print("ViT features after concate cnn_token: ", x.shape)
        

        return x

    def forward(self, x):
        cnn_tokens = self.forward_hyper_net(x)
        #print("CNN token shape ", cnn_tokens.shape)
        x = self.forward_features(x, cnn_tokens) #for quality prediction

        x = self.dropout(x)
        cnn_tokens = cnn_tokens.permute(0, 2, 1)
        x, y = self.learner(x, cnn_tokens)
        x = self.head(x)
        return x