import torch
import torch.nn as nn
from models_mamba import create_block, RMSNorm, rms_norm_fn
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from common import (StrideEmbed, PatchEmbed, segm_init_weights, _init_weights,
                    stride_patchify, patchify, random_masking)


class NetMamba(nn.Module):
    def __init__(self,
                 arr_length=1600,
                 stride_size=4,
                 in_chans=1,
                 embed_dim=192, depth=4, 
                 decoder_embed_dim=128, decoder_depth=2,
                 num_classes=1000,
                 norm_pix_loss=False,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 bimamba_type="none",
                 is_pretrain=False,
                 if_patch_embed=False,
                 if_pos_embed=True,
                 device=None, dtype=None,
                 conf_learning=False,
                 num_embeddings=None,
                 seq_len=50,
                 **kwargs):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.is_pretrain = is_pretrain
        self.conf_learning = conf_learning
        self.stride_size = stride_size
        self.num_embeddings = num_embeddings
 
        # --------------------------------------------------------------------------
        # NetMamba encoder specifics
        self.if_patch_embed = if_patch_embed
        if num_embeddings is None:
            self.patch_embed = StrideEmbed(arr_length=arr_length, stride_size=stride_size, embed_dim=embed_dim)
            if if_patch_embed:
                self.patch_embed = PatchEmbed(byte_length=arr_length, in_chans=in_chans, embed_dim=embed_dim)
            self.num_patches = self.patch_embed.num_patches
        else:
            self.patch_embed = nn.Embedding(num_embeddings, embed_dim)
            self.num_patches = seq_len
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_cls_token = 1
        self.if_pos_embed = if_pos_embed
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, embed_dim))
        print(f"pos_embed shape: {self.pos_embed.shape}")
        self.pos_drop = nn.Dropout(p=drop_rate)
        # Mamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.blocks = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=None,
                norm_epsilon=1e-5,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                layer_idx=i,
                if_bimamba=False,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                if_devide_out=True,
                init_layer_scale=None,
            )  for i in range(depth)])
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        # --------------------------------------------------------------------------

        if is_pretrain:
            # --------------------------------------------------------------------------
            # NetMamba decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, decoder_embed_dim))
            decoder_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]  # stochastic depth decay rule
            decoder_inter_dpr = [0.0] + decoder_dpr
            self.decoder_blocks = nn.ModuleList([
                create_block(
                    decoder_embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type=bimamba_type,
                    drop_path=decoder_inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                )
                for i in range(decoder_depth)])
            self.decoder_norm_f = RMSNorm(decoder_embed_dim, eps=1e-5)
            if self.if_patch_embed:
                self.decoder_pred = nn.Linear(decoder_embed_dim, 4, bias=True)
            else:
                self.decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)  # decoder to stride
            # --------------------------------------------------------------------------
        else:
            # --------------------------------------------------------------------------
            # NetMamba classifier specifics
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            if self.conf_learning:
                self.confidence = nn.Linear(self.num_features, 1)
            # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights(depth)

    def initialize_weights(self, depth):
        self.patch_embed.apply(segm_init_weights)
        if not self.is_pretrain:
            self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.is_pretrain:
            trunc_normal_(self.decoder_pos_embed, std=.02)
            trunc_normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(partial(_init_weights, n_layer=depth,))
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float, if_mask: bool=True):
        """
        x: [B, 1, H, W]
        """
        # embed patches
        B, C, H, W = x.shape
        if self.num_embeddings is None:
            if not self.if_patch_embed:
                x = self.patch_embed(x.reshape(B, C, -1))
            else:
                x = self.patch_embed(x.reshape(B, C, 40, -1))
        else:
            x = self.patch_embed(x.reshape(B, -1).to(torch.int32))

        # add pos embed w/o cls token
        if self.if_pos_embed:
            x = x + self.pos_embed[:, :-1, :]

        # masking: length -> length * mask_ratio
        if if_mask:
            x, mask, ids_restore = random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, -1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.pos_drop(x)

        # apply Mamba blocks
        residual = None
        hidden_states = x
        for blk in self.blocks:
            hidden_states, residual = blk(hidden_states, residual)
        fused_add_norm_fn = rms_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )
        if if_mask:
            return x, mask, ids_restore
        else:
            return x

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        visible_tokens = x[:, :-1, :]
        x_ = torch.cat([visible_tokens, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x_, x[:, -1:, :]], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Mamba blocks
        residual = None
        hidden_states = x
        for blk in self.decoder_blocks:
            hidden_states, residual = blk(hidden_states, residual)
        fused_add_norm_fn = rms_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.decoder_norm_f.weight,
            self.decoder_norm_f.bias,
            eps=self.decoder_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, :-1, :]
        return x

    def forward_rec_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        if self.if_patch_embed:
            B, C, H, W = imgs.shape
            target = patchify(imgs.reshape(B, C, 40, -1))
        else:
            target = stride_patchify(imgs, self.stride_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: float=0.9, return_cls_token: bool=False,
                extra_labels=None, lengths=None, intervals=None):
        # imgs: [B, 1, H, W]
        B, C, H, W = imgs.shape
        assert C == 1, "Input images should be grayscale"
        if self.is_pretrain:
            latent, mask, ids_restore = self.forward_encoder(imgs, 
                    mask_ratio=mask_ratio,)
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_rec_loss(imgs, pred, mask)
            return loss
        else:
            x = self.forward_encoder(imgs, mask_ratio=mask_ratio, if_mask=False)
            cls_token = x[:, -1, :]  # [B, D]
            confidence = self.confidence(cls_token) if self.conf_learning else None
            logits = self.head(cls_token)
            return {
                "logits": logits,
                "confidence": confidence,
                "cls_token": cls_token,
            }


def net_mamba_pretrain(**kwargs):
    model = NetMamba(
        is_pretrain=True, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2, **kwargs)
    return model

def net_mamba_classifier(**kwargs):
    model = NetMamba(
        is_pretrain=False, embed_dim=256, depth=4,
        **kwargs)
    return model
