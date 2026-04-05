import torch
import torch.nn as nn
from models_rnn import minGRU, minLSTM
from timm.models.layers import trunc_normal_
from functools import partial
from common import StrideEmbed, _init_weights, segm_init_weights, stride_patchify, random_masking


def create_block(d_model=192, block_name="minGRU", expansion_factor=2.0):
    if block_name == "minGRU":
        return minGRU(d_model, expansion_factor=expansion_factor)
    elif block_name == "minLSTM":
        return minLSTM(d_model, expansion_factor=expansion_factor)
    else:
        raise NotImplementedError(f"Block {block_name} not implemented")


class NetRNN(nn.Module):
    def __init__(self,
                 arr_length=1600,
                 stride_size=4,
                 in_chans=1,
                 embed_dim=192, depth=4, 
                 decoder_embed_dim=128, decoder_depth=2,
                 num_classes=1000,
                 drop_rate=0.,
                 block_name="minGRU",
                 expansion_factor=2.0,
                 norm_pix_loss=False,
                 is_pretrain=False,
                 if_cls_token=True,
                 device=None, dtype=None,
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
        self.stride_size = stride_size
        self.num_embeddings = num_embeddings

        # --------------------------------------------------------------------------
        # NetMamba encoder specifics
        if num_embeddings is None:
            self.patch_embed = StrideEmbed(arr_length=arr_length, stride_size=stride_size, embed_dim=embed_dim)
            self.num_patches = self.patch_embed.num_patches
        else:
            self.patch_embed = nn.Embedding(num_embeddings, embed_dim)
            self.num_patches = seq_len
        self.if_cls_token = if_cls_token
        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_cls_token = 1
        else:
            self.num_cls_token = 0
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_cls_token, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # Mamba blocks
        self.blocks = nn.ModuleList([
            create_block(d_model=embed_dim, block_name=block_name, expansion_factor=expansion_factor)
            for _ in range(depth)])
        # --------------------------------------------------------------------------

        if is_pretrain:
            # --------------------------------------------------------------------------
            # NetMamba decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_cls_token, decoder_embed_dim))
            self.decoder_blocks = nn.ModuleList([
                create_block(d_model=decoder_embed_dim, block_name=block_name, expansion_factor=expansion_factor)
                for _ in range(decoder_depth)])
            self.decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)  # decoder to stride
            # --------------------------------------------------------------------------
        else:
            # --------------------------------------------------------------------------
            # NetMamba classifier specifics
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
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
    
    def forward_encoder(self, x, mask_ratio, if_mask=True,):
        """
        x: [B, 1, H, W]
        """
        # embed patches
        B, C, H, W = x.shape
        if self.num_embeddings is None:
            x = self.patch_embed(x.reshape(B, C, -1))
        else:
            x = self.patch_embed(x.reshape(B, -1).to(torch.int32))

        # add pos embed w/o cls token
        if self.if_cls_token:
            x = x + self.pos_embed[:, :-1, :]
        else:
            x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        if if_mask:
            x, mask, ids_restore = random_masking(x, mask_ratio)

        # append cls token
        if self.if_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, -1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)
        x = self.pos_drop(x)

        # apply Mamba blocks
        for blk in self.blocks:
            x = blk(x)
        if if_mask:
            return x, mask, ids_restore
        else:
            return x

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + self.num_cls_token - x.shape[1], 1)
        if self.if_cls_token:
            visible_tokens = x[:, :-1, :]
        else:
            visible_tokens = x
        x_ = torch.cat([visible_tokens, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        if self.if_cls_token:
            x = torch.cat([x_, x[:, -1:, :]], dim=1)  # append cls token
        else:
            x = x_

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Mamba blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        if self.if_cls_token:
            x = x[:, :-1, :]
        return x

    def forward_rec_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = stride_patchify(imgs, self.stride_size)  # [N, L, p*p*1]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.9, **kwargs):
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
            cls_token = x[:, -1, :] if self.if_cls_token else torch.mean(x, dim=1)
            logits = self.head(cls_token)
            return {
                "logits": logits,
                "features": x,
            }


def net_mingru_base_pretrain(**kwargs):
    model = NetRNN(
        is_pretrain=True, embed_dim=192, depth=4,
        decoder_embed_dim=128, decoder_depth=2, block_name="minGRU",
        expansion_factor=2.0, **kwargs)
    return model


def net_mingru_base_classifier(**kwargs):
    model = NetRNN(
        is_pretrain=False, embed_dim=192, depth=4,
        block_name="minGRU", expansion_factor=2.0, **kwargs)
    return model


def net_mingru_seq_base_classifier(**kwargs):
    model = NetRNN(
        is_pretrain=False, embed_dim=192, depth=4,
        block_name="minGRU", expansion_factor=2.0, 
        num_embeddings=3000, **kwargs)
    return model


def net_minlstm_seq_base_classifier(**kwargs):
    model = NetRNN(
        is_pretrain=False, embed_dim=192, depth=4,
        block_name="minLSTM", expansion_factor=2.0, 
        num_embeddings=3000, **kwargs)
    return model


def net_mingru_medium_pretrain(**kwargs):
    model = NetRNN(
        is_pretrain=True, embed_dim=384, depth=4,
        decoder_embed_dim=256, decoder_depth=2, block_name="minGRU",
        expansion_factor=2.0, **kwargs)
    return model


def net_mingru_medium_classifier(**kwargs):
    model = NetRNN(
        is_pretrain=False, embed_dim=384, depth=4,
        block_name="minGRU", expansion_factor=2.0, **kwargs)
    return model
