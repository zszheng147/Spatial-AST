# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import numba
from functools import partial

import torch
import torch.nn as nn
import torchaudio

from torchlibrosa.stft import STFT, LogmelFilterBank
from timm.models.layers import to_2tuple

from vision_transformer import VisionTransformer as _VisionTransformer

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=4):
        super().__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, self.in_chans, img_size[0], img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x) # 32, 1, 1024, 128 -> 32, 768, 101, 12
        x = x.flatten(2) # 32, 768, 101, 12 -> 32, 768, 1212
        x = x.transpose(1, 2) # 32, 768, 1212 -> 32, 1212, 768
        return x

class VisionTransformer(_VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, mask_2d=True, use_custom_patch=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        img_size = (1024, 128) # 1024, 128
        in_chans = 2
        emb_dim = 768

        self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=in_chans, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
        num_patches = self.patch_embed.num_patches
        #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

        self.spectrogram_extractor = STFT(
            n_fft=1024, hop_length=320, win_length=1024, window='hann', 
            center=True, pad_mode='reflect', freeze_parameters=True
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=32000, n_fft=1024, n_mels=128, fmin=50, 
            fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True
        )
        
        self.timem = torchaudio.transforms.TimeMasking(192)
        self.freqm = torchaudio.transforms.FrequencyMasking(48)

        self.mel_norm = nn.BatchNorm2d(2, affine=False)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
        del self.norm  # remove the original norm
        self.mask_2d = mask_2d
        self.use_custom_patch = use_custom_patch
        self.target_frame = 1024

        # self.proj = nn.Identity()
        # self.spatial_proj = nn.Sequential(
        #     nn.Linear(emb_dim, emb_dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(emb_dim * 2, emb_dim),
        #     nn.GELU(),
        # )

        self.distance_head = nn.Linear(emb_dim, 11)
        self.azimuth_head = nn.Linear(emb_dim, 360)
        self.elevation_head = nn.Linear(emb_dim, 180)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        x = self.pos_drop(x)
        
        spatial_x = x
        for idx, blk in enumerate(self.blocks):
            x = blk(x)

        return x, spatial_x
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch:
            # # for AS
            T=101 #64,101
            F=12 #8,12
        else:
            # ## for AS 
            T=64
            F=8
        
        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0,2,1,3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None

    def forward_features_mask(self, x, mask_t_prob, mask_f_prob):
        B = x.shape[0] #4,1,1024,128
        x = self.patch_embed(x) # 4, 512, 768

        x = x + self.pos_embed[:, 1:, :]
        if self.random_masking_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_t_prob)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)        
        x = self.pos_drop(x)
        
        spatial_x = x
        for idx, blk in enumerate(self.blocks):
            x = blk(x)

        return x, spatial_x

    # overwrite original timm
    def forward(self, waveforms, reverbs, mask_t_prob=0.0, mask_f_prob=0.0):
        waveforms = torchaudio.functional.fftconvolve(waveforms, reverbs, mode='full')[..., :waveforms.shape[-1]]
        B, C, T = waveforms.shape

        waveforms = waveforms.reshape(B * C, T)
        # bsz* channels, 1024, 513
        real, imag = self.spectrogram_extractor(waveforms) 
        x = self.logmel_extractor(torch.sqrt(real**2 + imag**2)).reshape(B, C, -1, 128)

        if x.shape[2] < self.target_frame:
            x = nn.functional.interpolate(x, (self.target_frame, x.shape[3]), mode="bicubic", align_corners=True)
        
        x = self.mel_norm(x) * 0.5
        if self.training:
            x = x.transpose(-2, -1) # bsz, 4, 1024, 128 --> bsz, 4, 128, 1024
            x = self.freqm(x)
            x = self.timem(x)
            x = x.transpose(-2, -1)

        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x, spatial_x = self.forward_features_mask(x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)
        else:
            x, spatial_x = self.forward_features(x)
        
        for blk2 in self.blocks2:
            spatial_x = blk2(spatial_x)

        x = x[:, 1:].mean(dim=1)
        x = self.fc_norm(x)
        classifier = self.head(x)

        distance = self.distance_head(spatial_x)[:, 1:].mean(dim=1)
        azimuth = self.azimuth_head(spatial_x)[:, 1:].mean(dim=1)
        elevation = self.elevation_head(spatial_x)[:, 1:].mean(dim=1)

        return classifier, distance, azimuth, elevation


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

