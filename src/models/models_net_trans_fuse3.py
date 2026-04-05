import torch
import torch.nn as nn
import torch.nn.functional as F
from models_trans import create_block, RMSNorm
from timm.models.layers import trunc_normal_
from common import (StrideEmbed, FixedCosineEmbed, LearnedCosineEmbed,
                    segm_init_weights, random_masking, NUM_EMBEDDINGS, MTU, MASK_ID, random_masking_seq,
                    compute_byte_rec_loss, compute_size_rec_loss, compute_iat_rec_loss)


class NetTransformer(nn.Module):
    def __init__(self, 
                 arr_length=1600,
                 stride_size=4,
                 in_chans=1,
                 embed_dim=192, 
                 encoder_depth=4,
                 decoder_embed_dim=128, 
                 decoder_depth=2,
                 num_classes=1000,
                 n_heads=8, block_name="basic",
                 drop_rate=0.,
                 is_pretrain=False,
                 size_key="sizes",
                 seq_len=50,
                 cls_fusion="add", # add / concat / concat_dense
                 head_bias=False,
                 num_shared_encoder=1,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.is_pretrain = is_pretrain
        self.stride_size = stride_size
        
        if size_key == "sizes":
            self.num_embeddings = NUM_EMBEDDINGS
            self.mask_id = MASK_ID
        elif size_key == "signed_sizes":
            self.num_embeddings = NUM_EMBEDDINGS + MTU
            self.mask_id = MASK_ID + MTU
        else:
            raise ValueError(f"Unknown size key: {size_key}")

        self.num_shared_encoder = num_shared_encoder
        
        self.byte_embed = StrideEmbed(arr_length=arr_length, stride_size=stride_size, embed_dim=embed_dim)
        self.num_byte_patches = self.byte_embed.num_patches
        self.byte_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.size_embed = FixedCosineEmbed(embed_dim)
        self.num_size_patches = seq_len
        self.size_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.iat_embed = FixedCosineEmbed(embed_dim)
        self.num_iat_patches = seq_len
        self.iat_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.byte_indicator = nn.Parameter(torch.zeros(1, 1, embed_dim))  # byte indicator for fusion
        self.size_indicator = nn.Parameter(torch.zeros(1, 1, embed_dim))  # size indicator for fusion
        self.iat_indicator = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_byte_patches + self.num_size_patches + self.num_iat_patches + 3, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.encoder_blocks = nn.ModuleList([
            create_block(d_model=embed_dim, n_heads=n_heads, d_head=embed_dim // n_heads, dropout=0.1, 
                        block_name=block_name) for _ in range(encoder_depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Size Encoder
        if is_pretrain:
            # Byte Decoder
            self.byte_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.byte_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.byte_decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_byte_patches + 1, decoder_embed_dim))
            self.byte_decoder_blocks = nn.ModuleList([
                create_block(d_model=decoder_embed_dim, n_heads=n_heads, d_head=decoder_embed_dim // n_heads, dropout=0.1, 
                            block_name=block_name)
                for _ in range(decoder_depth)])
            self.byte_decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)  # decoder to stride
            # Size Decoder
            self.size_decoder_pred = nn.Linear(embed_dim, self.num_embeddings, bias=True)  # decoder to size
            # IAT Decoder
            self.iat_decoder_pred = nn.Linear(embed_dim, 1, bias=True)
        else:
            # --------------------------------------------------------------------------
            # Classifier
            if cls_fusion == "concat":
                cls_fusion_dim = 3 * embed_dim
            else:
                cls_fusion_dim = None
            if cls_fusion_dim is not None:
                self.dense = nn.Sequential(
                    nn.Linear(cls_fusion_dim, 2 * embed_dim),
                    nn.SELU(),
                    nn.Dropout(p=drop_rate),
                    nn.Linear(2 * embed_dim, embed_dim),
                    nn.SELU(),
                    nn.Dropout(p=drop_rate),
                )
            self.head = nn.Linear(self.num_features, num_classes, bias=head_bias) if num_classes > 0 else nn.Identity()
        self.cls_fusion = cls_fusion
        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.byte_cls_token, std=.02)
        trunc_normal_(self.size_cls_token, std=.02)
        trunc_normal_(self.iat_cls_token, std=.02)
        trunc_normal_(self.byte_indicator, std=.02)
        trunc_normal_(self.size_indicator, std=.02)
        trunc_normal_(self.iat_indicator, std=.02)
        if self.is_pretrain:
            trunc_normal_(self.byte_decoder_pos_embed, std=.02)
            trunc_normal_(self.byte_mask_token, std=.02)
        self.apply(segm_init_weights)
    
    @torch.jit.ignore # type: ignore
    def no_weight_decay(self):
        return {"pos_embed", "byte_cls_token", "norm", "byte_mask_token", 
                "size_cls_token", "iat_cls_token"}

    def forward_encoder(self, x_byte: torch.Tensor, x_size: torch.Tensor, x_iat: torch.Tensor,
                        byte_mask_ratio: float, size_mask_ratio: float, 
                        iat_mask_ratio: float, 
                        if_mask: bool=True):
        """
        x_byte: [B, 1, H, W], byte features
        x_size: [B, L], size features
        """
        # embed patches
        B, C, H, W = x_byte.shape
        assert C == 1
        x_byte = self.byte_embed(x_byte.reshape(B, C, -1))

        # add pos embed w/o cls token
        x_byte = x_byte + self.pos_embed[:, self.num_size_patches+self.num_iat_patches+2:-1, :] \
            + self.byte_indicator.expand(B, self.num_byte_patches, -1)
        # masking: length -> length * mask_ratio
        if if_mask:
            x_byte, mask_byte, ids_restore = random_masking(x_byte, byte_mask_ratio)
            x_size, mask_size = random_masking_seq(x_size, size_mask_ratio, mask_value=self.mask_id)
            x_iat, mask_iat = random_masking_seq(x_iat, iat_mask_ratio, mask_value=0)
        # append cls token for bytes
        byte_cls_token = self.byte_cls_token + self.pos_embed[:, -1, :] + self.byte_indicator.expand(B, 1, -1)  # add pos embed and byte indicator
        x_byte = torch.cat((x_byte, byte_cls_token.expand(B, -1, -1)), dim=1)

        x_size = self.size_embed(x_size)  # [B, L, D]
        x_size = torch.cat((x_size, self.size_cls_token.expand(B, -1, -1)), dim=1)  # append cls token
        x_size = x_size + self.size_indicator.expand(B, self.num_size_patches + 1, -1)  # add size indicator
        x_size = x_size + self.pos_embed[:, :self.num_size_patches+1, :]  # add pos embed

        x_iat = self.iat_embed(x_iat)  # [B, L, D], use size embed for IAT
        x_iat = torch.cat((x_iat, self.iat_cls_token.expand(B, -1, -1)), dim=1)  # append cls token
        x_iat = x_iat + self.iat_indicator.expand(B, self.num_iat_patches + 1, -1)  # add iat indicator
        x_iat = x_iat + self.pos_embed[:, self.num_size_patches+1: self.num_size_patches+self.num_iat_patches+2, :]  # add pos embed
        
        # dropout
        x = self.pos_drop(torch.cat((x_size, x_iat, x_byte), dim=1))  # [B, L1+L2+2, D]
        # apply encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x)[0]
        if self.num_shared_encoder > 1:
            shared_encoder = self.encoder_blocks[-1] # share the last encoder block
            for _ in range(self.num_shared_encoder - 1):
                x = shared_encoder(x)[0]
        x = self.norm(x)
        if if_mask:
            return x, mask_byte, ids_restore, mask_size, mask_iat# type: ignore
        else:
            return x

    def forward_byte_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        """
        x: [B, L, D],
        ids_restore: [B, L], indices to restore the original order of tokens 
        """
        # embed tokens
        x = self.byte_decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.byte_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        visible_tokens = x[:, :-1, :]
        x_ = torch.cat([visible_tokens, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x_, x[:, -1:, :]], dim=1)  # append cls token
        # add pos embed
        x = x + self.byte_decoder_pos_embed
        # apply Mamba blocks
        for blk in self.byte_decoder_blocks:
            x = blk(x)[0]
        # predictor projection
        x = self.byte_decoder_pred(x)
        # remove cls token
        x = x[:, :-1, :]
        return x
    
    def forward_size_decoder(self, x: torch.Tensor):
        """
        x: [B, L, D]
        """
        # predictor projection
        x = self.size_decoder_pred(x)[:, :-1, :]  # remove cls token
        return x
    
    def forward_iat_decoder(self, x: torch.Tensor):
        """
        x: [B, L, D]
        """
        # predictor projection
        x = self.iat_decoder_pred(x)[:, :-1, :] # remove cls token
        return x

    def forward(self, x_byte: torch.Tensor, x_size: torch.Tensor, x_iat: torch.Tensor,
                byte_mask=None, size_mask=None, iat_mask=None,
                byte_mask_ratio=0.9, size_mask_ratio=0.15, iat_mask_ratio=0.15, **kwargs):
        # x_size = x_size.to(dtype=torch.long)
        if self.is_pretrain:
            # byte pre-training
            latent, byte_mask, ids_restore, size_mask, iat_mask = self.forward_encoder(x_byte, x_size, x_iat,
                                                                byte_mask_ratio, size_mask_ratio, iat_mask_ratio, 
                                                                byte_mask, size_mask, iat_mask, if_mask=True)
            size_latent = latent[:, :self.num_size_patches+1, :]  # size features
            iat_latent = latent[:, self.num_size_patches+1: self.num_size_patches+self.num_iat_patches+2, :]  # iat features
            byte_latent = latent[:, self.num_size_patches+self.num_iat_patches+2:, :]  # byte features
            # byte decoder
            byte_pred = self.forward_byte_decoder(byte_latent, ids_restore=ids_restore)
            byte_loss = compute_byte_rec_loss(x_byte, byte_pred, byte_mask, self.stride_size)
            # size decoder
            size_pred = self.forward_size_decoder(size_latent)
            size_loss = compute_size_rec_loss(x_size.to(dtype=torch.long), size_pred, size_mask, self.num_embeddings)
            # iat decoder
            iat_pred = self.forward_iat_decoder(iat_latent)
            iat_loss = compute_iat_rec_loss(x_iat, iat_pred, iat_mask)
            # return losses
            return byte_loss, size_loss, iat_loss
        else:
            latent = self.forward_encoder(x_byte, x_size, x_iat, 0.0, 0.0, 0.0, 
                                          byte_mask, size_mask, iat_mask, if_mask=False)
            byte_h = latent[:, -1, :]  # type: ignore
            size_h = latent[:, self.num_size_patches, :]  # type: ignore
            iat_h = latent[:, self.num_size_patches+self.num_iat_patches+1, :]  # type: ignore
            # classification head
            # modal cls_fusion
            if self.cls_fusion == "concat":
                x = torch.cat((size_h, iat_h, byte_h), dim=-1)
                x = self.dense(x)  # [B, 2*D] -> [B, D]
            else:
                x = byte_h + size_h + iat_h  # [B, D]
            logits = self.head(x)  # classification head
            return {
                "logits": logits,
                "cls_token": x,
                "byte_feat": byte_h,
                "size_feat": size_h,
                "iat_feat": iat_h,
            }


def fuse3_fgt_base_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=192, encoder_depth=4, decoder_embed_dim=128, decoder_depth=2, 
        block_name="flash-gated", **kwargs)
    return model

def fuse3_fgt_base_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=192, encoder_depth=4, block_name="flash-gated", **kwargs)
    return model

def fuse3_bgt_base_pretrain(**kwargs):
    model = NetTransformer(
        is_pretrain=True, embed_dim=192, encoder_depth=4, decoder_embed_dim=128, decoder_depth=2, 
        block_name="basic-gated", **kwargs)
    return model

def fuse3_bgt_base_classifier(**kwargs):
    model = NetTransformer(
        is_pretrain=False, embed_dim=192, encoder_depth=4, block_name="basic-gated", **kwargs)
    return model
