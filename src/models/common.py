import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, lecun_normal_
import torch.nn.functional as F

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02, mean=0)


class StrideEmbed(nn.Module):
    def __init__(self, arr_length=1600, stride_size=4, in_chans=1, embed_dim=192):
        super().__init__()
        assert arr_length % stride_size == 0
        self.num_patches = arr_length // stride_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=stride_size, stride=stride_size)
        
    def forward(self, x):
        """
        x: [B, C, L], where C is the number of channels, L is the length of the sequence.
        """
        return self.proj(x).transpose(1, 2) # [B, N, D], where N = L // stride_size


class PatchEmbed(nn.Module):
    def __init__(self, byte_length=1600, patch_size=2, in_chans=1, embed_dim=192):
        super().__init__()
        self.num_patches = byte_length // (patch_size ** 2)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x # [B, N, D], N is the number of patches, D is the embedding dimension


class FixedCosineEmbed(nn.Module):
    def __init__(self, d_model: int):
        super(FixedCosineEmbed, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for fixed value encoding."
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L]
        d_model: embedding dimension
        """
        B, L = x.shape
        x = x.unsqueeze(-1) # [B, L, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device) *
                            (-math.log(10000.0) / self.d_model)) # [D//2]
        x_scaled = x * div_term # [B, L, D//2]
        pe = torch.zeros(B, L, self.d_model, dtype=torch.float32, device=x.device) # [B, L, D]
        pe[:, :, 0::2] = torch.sin(x_scaled)
        pe[:, :, 1::2] = torch.cos(x_scaled)
        return pe # [B, L, D]


class LearnedCosineEmbed(nn.Module):
    def __init__(self, d_model: int):
        super(LearnedCosineEmbed, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for learnable cosine encoding."
        self.omega = nn.Parameter(torch.randn(d_model // 2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L]
        """
        assert x.dim() == 2, "Input tensor must be of shape [B, L]"
        B, L = x.shape
        x = x.unsqueeze(-1) * self.omega # [B, L, D//2]
        x_embed = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) # [B, L, D]
        return x_embed  # [B, L, D]


def stride_patchify(imgs: torch.Tensor, stride_size: int):
    """
    imgs: (N, 1, H, W)
    x: (N, L, patch_size**2 *1)
    """
    B, C, H, W = imgs.shape
    assert C == 1, "Input images should be grayscale"
    x = imgs.reshape(B, H*W // stride_size, stride_size)
    return x


def patchify(imgs: torch.Tensor, patch_size: int=2):
    """
    imgs: (N, 1, H, W)
    x: (N, L, patch_size**2 *1)
    """
    h, w = imgs.shape[2] // patch_size, imgs.shape[3] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size ** 2 * 1))
    return x


def random_masking(x: torch.Tensor, mask_ratio: float):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [B N D], sequence
    """
    B, N, D = x.shape  # batch, length, dim
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # ids_shuffle[b][i] = j means i-th element in b-th sample is the j-th smallest noise value
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # ids_restore[b][j] = i means j-th smallest noise value in b-th sample is at the i-th position

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_unmasked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_unmasked, mask, ids_restore


## for uni-directional size sequences, a MTU shift should be added for bi-directional sizes
MTU = 1500 # legal indices [0, MTU]
PAD_ID = MTU + 1
MASK_ID = MTU + 2
NUM_EMBEDDINGS = MTU + 3  # +1 for PAD_ID, +1 for MASK_ID


def random_masking_seq(x: torch.Tensor, mask_ratio: float, mask_value: float):
    """
    Randomly masks a portion of each sequence in x with MASK_ID.
    x: [B, N]
    Returns masked x and the binary mask (0 = unmasked, 1 = masked).
    """
    B, N = x.shape  # batch, length
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  
    # ids_shuffle[b][i] = j means i-th element in b-th sample is the j-th smallest noise value
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # ids_restore[b][j] = i means j-th smallest noise value in b-th sample is at the i-th position

    # generate the binary mask: 0 is unmasked, 1 is masked
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    x_masked = torch.where(mask.bool(), torch.full_like(x, mask_value), x)

    return x_masked, mask


def random_mask_keep_all(x: torch.Tensor, pad_mask: torch.Tensor, mask_ratio: float, mask_value: float):
    """
    Only mask the non-padding tokens in x according to mask_ratio.
    x: [B, N]
    pad_mask: [B, N], 0 for non-padding, 1 for padding
    """
    B, N = x.shape  # batch, length
    len_nonpad = (pad_mask == 0).sum(dim=1)  # [B]
    len_keep = (len_nonpad * (1 - mask_ratio)).long()  # [B]

    noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
    # set padding positions to large value
    noise = noise + pad_mask.float() * 1e6

    ids_shuffle = torch.argsort(noise, dim=1)  
    # ids_shuffle[b][i] = j means i-th element in b-th sample is the j-th smallest noise value
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # ids_restore[b][j] = i means j-th smallest noise value in b-th sample is at the i-th position

    # generate the binary mask: 0 is unmasked, 1 is masked
    mask = torch.ones([B, N], device=x.device)
    for i in range(B):
        mask[i, :len_keep[i]] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    x_masked = torch.where(mask.bool(), torch.full_like(x, mask_value), x)

    return x_masked, mask


def random_mask_keep_visible(x: torch.Tensor, pad_mask: torch.Tensor, mask_ratio: float):
    """
    Only mask the non-padding tokens in x according to mask_ratio, only return the visible tokens.
    x: [B, N, D]
    pad_mask: [B, N], 0 for non-padding, 1 for padding
    """
    B, N, D = x.shape  # batch, length, dim
    len_nonpad = (pad_mask == 0).sum(dim=1)  # [B]
    len_keep = (len_nonpad * (1 - mask_ratio)).long()  # [B]

    noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
    # set padding positions to large value
    noise = noise + pad_mask.float() * 1e6
    ids_shuffle = torch.argsort(noise, dim=1)  
    # ids_shuffle[b][i] = j means i-th element in b-th sample is the j-th smallest noise value
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # ids_restore[b][j] = i means j-th smallest noise value in b-th sample is at the i-th position
    # keep the first subset
    x_unmasked = torch.zeros(B, N, D, device=x.device)
    for i in range(B):
        ids_keep = ids_shuffle[i, :len_keep[i]]
        x_unmasked[i, :len_keep[i], :] = x[i].gather(dim=0, index=ids_keep.unsqueeze(-1).repeat(1, D))
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([B, N], device=x.device)
    for i in range(B):
        mask[i, :len_keep[i]] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_unmasked, mask, ids_restore


def compute_byte_rec_loss(imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, stride_size: int):
    """
    imgs: [B, 1, H, W]
    pred: [B, L, p*p*1]
    mask: [B, L], 0 is keep, 1 is remove,
    """
    target = stride_patchify(imgs, stride_size)
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

def compute_size_rec_loss(input_ids: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, num_embeddings: int):
    """
    input_ids: [B, L], input ids for size features
    pred: [B, L, num_embeddings]
    mask: [B, L], 0 is unmasked, 1 is masked,
    """
    target = input_ids.clone()
    target[mask == 0] = -100  # ignore the unmasked tokens
    pred = pred.reshape(-1, num_embeddings)  # [N*L, num_embeddings]
    target = target.reshape(-1)  # [N*L]
    loss = F.cross_entropy(pred, target, ignore_index=-100)
    return loss

def compute_iat_rec_loss(seq: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
    """
    seq: [B, L], input sequence for IAT features
    pred: [B, L, 1]
    mask: [B, L], 0 is unmasked, 1 is masked,
    """
    pred = pred.squeeze(-1)  # [B, L]
    target = seq
    loss = (pred - target) ** 2 # [B, L]
    loss = (loss * mask).sum() / mask.sum()
    return loss
